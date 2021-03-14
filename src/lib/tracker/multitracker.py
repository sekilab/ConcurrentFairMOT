import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F
import threading
import queue

from models.model import create_model, load_model
from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat
import copy

import tracker.cmc

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, frame_skip=0):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, frame_skip)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        #self._tlwh = new_track.tlwh
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    def transform_box(self, transform, tf_type='aff'):
        tlbr_boxes = [STrack.tlwh_to_tlbr(self._tlwh)]
        if self.mean is not None:
            tlbr_boxes.append(self.tlbr)
        for i, tlbr_box in enumerate(tlbr_boxes):
            lt_xy1 = np.r_[tlbr_box[:2], 1]
            rb_xy1 = np.r_[tlbr_box[2:4], 1]
            if tf_type == 'aff':
                tlbr_box[:2] = np.matmul(lt_xy1, transform.transpose())
                tlbr_box[2:4] = np.matmul(rb_xy1, transform.transpose())
            elif tf_type == 'proj':
                lt_xy1 = np.matmul(lt_xy1, transform.transpose())
                tlbr_box[:2] = lt_xy1[:2] / lt_xy1[2]
                rb_xy1 = np.matmul(rb_xy1, transform.transpose())
                tlbr_box[2:4] = rb_xy1[:2] / rb_xy1[2]
            if i == 0:
                self._tlwh = STrack.tlbr_to_tlwh(tlbr_box)
            else:
                self.mean[:4] = STrack.tlwh_to_xyah(STrack.tlbr_to_tlwh(tlbr_box))

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class UpdateThread(threading.Thread):
    def __init__(self, jdetracker):
        super().__init__(daemon=True)
        self.jdetracker = jdetracker

    def run(self):
        while(True):
            #start= time.time()
            dets, id_features, is_last = self.jdetracker.detection_queue.get()
            #dets, id_features, is_last, enqueue_time = self.jdetracker.detection_queue.get()
            #print("det q fetching time:", time.time() - enqueue_time)
            self.jdetracker.update(dets, id_features, is_last)
            if is_last:
                break
            #print("update loop:", time.time() - start)

class CmcThread(threading.Thread):
    def __init__(self, jdetracker):
        super().__init__(daemon=True)
        self.jdetracker = jdetracker

    def run(self):
        while(True):
            img0, frame_id, is_last = self.jdetracker.img_queue.get()
            if self.jdetracker.frame_skip > 0 and frame_id % (self.jdetracker.frame_skip + 1) != 1:
                self.jdetracker.cmc.getCMCTransform(img0, update_only=True)
                continue
            transform = self.jdetracker.cmc.getCMCTransform(img0)
            if self.jdetracker.cmc_type == 'aff':
                inverse_transform = cv2.invertAffineTransform(transform)
            elif self.jdetracker.cmc_type == 'proj':
                inverse_transform = np.linalg.inv(transform)
            self.jdetracker.cmc_transform_queue.put((transform, inverse_transform))
            if is_last:
                break


class JDETracker(object):
    def __init__(self, opt, frame_rate=30, track_boxes=None, frame_skip=0):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.frame_skip = frame_skip
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        #self.buffer_size = int(frame_rate / 30.0 / (frame_skip+1.0)* opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

        self.cmc_on = opt.cmc_on
        if self.cmc_on:
            if opt.cmc_type != "aff" and opt.cmc_type != "proj":
                raise(Exception("cmc type can only be aff and proj, got %s instead" % opt.cmc_type))
            if len(track_boxes) > 0:
                if opt.cmc_type == 'aff' and len(track_boxes) < 2:
                    raise(Exception("cmc type aff requires at least 2 key-points, got %d instead" % len(track_boxes)))
                if opt.cmc_type == 'proj' and len(track_boxes) < 4:
                    raise(Exception("cmc type proj requires al least 4 key-points, got %d instead" % len(track_boxes)))
        self.cmc_type = opt.cmc_type
        self.cmc = None
        self.cmc_thread = None
        self.track_boxes = track_boxes
        self.tb_ready_q = queue.Queue(1)

        self.detection_queue = queue.Queue(2)

        self.output_queue = queue.Queue(2)
        self.up_frame_id = 0

        self.cmc_transform_queue = queue.Queue(2)
        self.img_queue = queue.Queue(2)

        self.update_thread = UpdateThread(self)
        self.update_thread.start()

        self.runtime_t = None
        self.runtime_d = None

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def landmark_dialog(self, img0):
        if self.track_boxes is None:
            #cv2.imshow("Base Image", imim_blob)
            cv2.imshow("Base Image", img0)
            self.track_boxes = []
            tb_index = 0
            min_point = 2
            if self.cmc_type == 'proj':
                min_point = 4
            while(True):
                print("Selecting track box ", tb_index)
                #self.track_boxes.append(cv2.selectROI("Base Image", imim_blob, fromCenter=True, showCrosshair=True))
                tb = cv2.selectROI("Base Image", img0, fromCenter=True, showCrosshair=True)
                if tb == (0,0,0,0):
                    if len(self.track_boxes) < min_point:
                        print("At least %d trackboxes have to be selected" % min_point)
                    else:
                        del self.track_boxes[-1]
                        break
                self.track_boxes.append(tb)
                tb_index += 1
            cv2.destroyAllWindows()
        self.cmc = tracker.cmc.NPoint2DCMC(img0, self.track_boxes, self.cmc_type)
        self.track_boxes_set = True
        self.cmc_thread = CmcThread(self)
        self.cmc_thread.start()

    def detect(self, im_blob, img0, is_last):
        self.frame_id += 1
        #print("detect", self.frame_id)
        start = time.time()
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        '''Step 0: Calculate CMC transform'''
        #transform = None
        #inverse_transform = None
        if self.cmc_on:
            self.img_queue.put((img0, self.frame_id, is_last))
            #imim_blob = cv2.resize(cv2.cvtColor(np.array(im_blob.cpu().squeeze()).transpose([1,2,0]), cv2.COLOR_RGB2BGR), (width, height))
            """if self.cmc == None:
                if self.track_boxes is None:
                    #cv2.imshow("Base Image", imim_blob)
                    cv2.imshow("Base Image", img0)
                    self.track_boxes = []
                    for i in range(2):
                        #self.track_boxes.append(cv2.selectROI("Base Image", imim_blob, fromCenter=True, showCrosshair=True))
                        self.track_boxes.append(cv2.selectROI("Base Image", img0, fromCenter=True, showCrosshair=True))
                    cv2.destroyAllWindows()
                #self.cmc = tracker.cmc.NPointAffine2DCMC(imim_blob, self.track_boxes)
                self.cmc = tracker.cmc.NPointAffine2DCMC(img0, self.track_boxes)"""
            #else:
                #transform = np.array([[1,0,0],[0,1,0]], dtype=float)
                #transform = self.cmc.getCMCTransform(imim_blob)
                #transform = self.cmc.getCMCTransform(img0)
                #inverse_transform = cv2.invertAffineTransform(transform)

        if self.frame_skip > 0 and self.frame_id % (self.frame_skip + 1) != 1:
            dets = None
            id_feature = None

        else:
            ''' Step 1: Network forward, get detections & embeddings'''
            with torch.no_grad():
                output = self.model(im_blob)[-1]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                id_feature = output['id']
                id_feature = F.normalize(id_feature, dim=1)

                reg = output['reg'] if self.opt.reg_offset else None
                dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
                id_feature = _tranpose_and_gather_feat(id_feature, inds)
                id_feature = id_feature.squeeze(0)
                id_feature = id_feature.cpu().numpy()

            dets = self.post_process(dets, meta)
            dets = self.merge_outputs([dets])[1]

            remain_inds = dets[:, 4] > self.opt.conf_thres
            dets = dets[remain_inds]
            id_feature = id_feature[remain_inds]

            # vis
            '''
            for i in range(0, dets.shape[0]):
                bbox = dets[i][0:4]
                cv2.rectangle(img0, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0), 2)
            cv2.imshow('dets', img0)
            cv2.waitKey(0)
            id0 = id0-1
            '''
            #print("detection time:", time.time() - start)
            #start = time.time()
            #self.detection_queue.put((dets, id_feature, is_last, start))
        self.detection_queue.put((dets, id_feature, is_last))
            #print("queueing detection time:", time.time() - start)


    def update(self, dets, id_feature, is_last):
        self.up_frame_id += 1
        #print("update", self.up_frame_id)
        #start= time.time()
        if dets is None:
            self.output_queue.put((None, is_last))
            return

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        '''apply cmc transform to detections'''
        #if self.cmc_on and transform is not None:
        if self.cmc_on:
            #frame = cv2.warpAffine(imim_blob, transform, (imim_blob.shape[1], imim_blob.shape[0]))
            #frame = (frame * 255).astype(np.uint8)
            transform, inverse_transform = self.cmc_transform_queue.get()
            for detection in detections:
                #mybb = detection.tlbr.copy().astype(np.int32)
                #cv2.rectangle(frame, tuple(mybb[:2]), tuple(mybb[2:4]), (0, 255, 0), 1)
                detection.transform_box(transform, self.cmc_type)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool, 0)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        #print(dists)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)
        #print("first ass:", time.time() - ps_time)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks, and store them in new structure
        output_stracks = [copy.deepcopy(track) for track in self.tracked_stracks if track.is_activated]
        #output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #print("avg tracking", self.runtime_t / self.execcount, "seconds")
        #print("tracking", end_time - start_time , "seconds")

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        '''apply inverse cmc transform to output tracks'''
        if self.cmc_on:
            #output_stracks = copy.deepcopy(output_stracks)
            for track in output_stracks:
                #trackbb = track.tlbr.copy().astype(np.int32)
                #cv2.rectangle(frame, tuple(trackbb[:2]), tuple(trackbb[2:4]), (0, 0, 255), 2)
                track.transform_box(inverse_transform, self.cmc_type)
            #cv2.imshow("Frame", frame)
            #cv2.imwrite('/home/ubuntu/work/torakkingu/FairMOT/blackmagic/%d.jpg' % self.frame_id, frame)

        #print("update time:", time.time() - start)
        #start = time.time()
        #self.output_queue.put((output_stracks, is_last, start))
        self.output_queue.put((output_stracks, is_last))
        #print("queueing update time:", time.time() - start)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb