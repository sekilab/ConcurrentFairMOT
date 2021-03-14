from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import threading
import queue
import time
import json

import sys
if "/home/ubuntu/py3envtf3/models/research/slim" in sys.path:
    sys.path.remove("/home/ubuntu/py3envtf3/models/research/slim")

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def detecting(opt, dataloader, tracker, img_queue):
    for i, (path, img, img0) in enumerate(dataloader):
        #start = time.time()
        if i == 0:
            if opt.cmc_on:
                tracker.landmark_dialog(img0)
            tracker.tb_ready_q.put(True)

        img_queue.put(img0)
        if opt.gpus[0] == -1:
            blob = torch.from_numpy(img).unsqueeze(0)
        else:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        tracker.detect(blob, img0, i == (len(dataloader) - 1))
        #print('detection loop:', time.time() - start)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, tbs=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate, track_boxes=tbs, frame_skip=opt.mot_frame_skip)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    img_queue = queue.Queue()
    prev_online_tlwhs = None
    prev_online_ids = None
    detector_thread = threading.Thread(target=detecting, name='detector-thread',args=[opt, dataloader, tracker, img_queue],daemon=True)
    detector_thread.start()
    if opt.cmc_on and tbs is None:
                tb_ready = tracker.tb_ready_q.get()
                if not tb_ready:
                    raise Exception("track box dialogue returned corrupted value", tb_ready)
    img_buffer = []
    is_last = False
    #with tracker.out_buffer_change:
    while(True):
        timer.tic()
        #online_targets, is_last, enqueue_time = tracker.output_queue.get()
        online_targets, is_last = tracker.output_queue.get()
        #print("output q fetch time:", time.time() - enqueue_time)
        img_buffer.append(img_queue.get())

        if frame_id % 20 == 0:
            logger.info('Processed frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # handling skipped frames
        if online_targets is None:
            frame_id += 1
            timer.toc()
            if is_last:
                break
            else:
                continue
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_ims = vis.plot_tracking(img_buffer, online_tlwhs, online_ids, prev_online_tlwhs, prev_online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            prev_online_tlwhs = online_tlwhs
            prev_online_ids = online_ids
        #if show_image:
        #    cv2.imshow('online_im', online_im)
        if save_dir is not None:
            for i, online_im in enumerate(online_ims):
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id - len(online_ims) + i + 1)), online_im)
        img_buffer = []
        frame_id += 1
        if is_last:
            break
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


default_seq = [{"subdir" : "TH2020_MOT/eval_q",
                "seqs" : {"01B_1": {},
                          "01B_2": {},
                          "01B_3": {},
                          "01B_4": {},
                          "02D_1": {},
                          "02D_2": {}
                        }
               }]


def main(opt, data_root='../../Dataset', det_root=None, seqs=default_seq, exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    all_seqs = []

    for set in seqs:
        result_root = os.path.join(data_root, set['subdir'], '..', 'results', exp_name)
        mkdir_if_missing(result_root)
        for seqi, seq_entry in enumerate(set["seqs"]):
            seq = seq_entry["seq"]
            kps = seq_entry["kps"]
            seq_save_name = seq + "_" + str(seqi)
            all_seqs.append(os.path.join(set['subdir'], seq))
            tbs = None
            if opt.cmc_on:
                tbs = [tuple(tb) for tb in kps.values()]
            output_dir = os.path.join(data_root, set['subdir'], '..', 'outputs', exp_name, seq_save_name) if save_images or save_videos else None
            logger.info('start seq: {}'.format(seq))
            dataloader = datasets.LoadImages(osp.join(data_root, set['subdir'], seq, 'img1'), opt.img_size)
            if len(dataloader) == 0:
                logger.info(f"Sequence {osp.join(data_root, set['subdir'], seq, 'img1')} is missing or empty.")
                continue
            result_filename = os.path.join(result_root, '{}.txt'.format(seq_save_name))
            meta_info = open(os.path.join(data_root, set['subdir'], seq, 'seqinfo.ini')).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
            nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, tbs=tbs)
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            # eval
            logger.info('Evaluate seq: {}'.format(seq_save_name))
            evaluator = Evaluator(os.path.join(data_root, set['subdir']), seq, data_type)
            accs.append(evaluator.eval_file(result_filename, opt.mot_frame_skip))
            if save_videos:
                output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
                cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
                os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, all_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    result_sum_root = os.path.join(data_root, 'results', exp_name)
    mkdir_if_missing(result_sum_root)
    Evaluator.save_summary(summary, os.path.join(result_sum_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    with open(os.path.join("data", opt.val_seq_file)) as val_seq_file:
        seqs = json.load(val_seq_file)

    main(opt,
         data_root=opt.data_dir,
         seqs=seqs,
         exp_name='TH2020_MOT_test',
         show_image=False,
         save_images=True,
         save_videos=True)
