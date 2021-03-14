import numpy as np
import cv2
import threading

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=True)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class NPoint2DCMC(object):
    def __init__(self, base_img, base_kp_boxes, cmc_type):
        base_img = (base_img * 255).astype(np.uint8)
        if cmc_type != "aff" and cmc_type != "proj":
            raise(Exception("cmc type can only be aff and proj, got %s instead" % cmc_type))
        self.cmc_type = cmc_type
        self.base_kp_boxes = base_kp_boxes
        self.base_kps = np.zeros((len(self.base_kp_boxes), 2), dtype=np.int32)
        self.landmark_trackers = []
        for i in range(len(self.base_kps)):
            landmark_tracker = cv2.TrackerCSRT_create()
            landmark_tracker.init(base_img, self.base_kp_boxes[i])
            self.landmark_trackers.append(landmark_tracker)
            self.base_kps[i,:] = [int(self.base_kp_boxes[i][0] + self.base_kp_boxes[i][2] / 2),
                                  int(self.base_kp_boxes[i][1] + self.base_kp_boxes[i][3] / 2)]

    def update(self, new_img):
        result = []
        threads = []
        for i in range(self.base_kps.shape[0]):
            threads.append(ThreadWithReturnValue(target=self.landmark_trackers[i].update, args=([new_img])))
            threads[i].start()
        for thread in threads:
            result.append(thread.join())
        #for i in range(self.base_kps.shape[0]):
        #    result.append(self.landmark_trackers[i].update(new_img))
        return result

    def getCMCTransform(self, new_img, update_only=False):
        new_img = (new_img * 255).astype(np.uint8)
        update_result = self.update(new_img)
        if update_only:
            return None
        new_kps = np.zeros_like(self.base_kps)
        for i, (res, kp_box) in enumerate(update_result):
            if res:
                 new_kps[i,:] = [int(kp_box[0] + kp_box[2] / 2),
                                  int(kp_box[1] + kp_box[3] / 2)]
            else:
                #Need to reinitialize in this case
                print('Error while tracking points for calculating CMC transformation')
                return None
        #print('kps', self.base_kps, new_kps)
        if self.cmc_type == 'aff':
            return cv2.estimateAffinePartial2D(new_kps, self.base_kps)[0]
        if self.cmc_type == 'proj':
            #print(new_kps, self.base_kps)
            return cv2.findHomography(new_kps, self.base_kps)[0]