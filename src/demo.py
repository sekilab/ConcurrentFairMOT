from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

#Hack to avoid name collision with slim dataset module
#import sys
#if <path to your slim package> in sys.path:
#    sys.path.remove(<path to your slim package>)

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import json




logger.setLevel(logging.INFO)


def demo(opt):
    with open(os.path.join("data", opt.demo_seq_file)) as seq_file:
        seqs = json.load(seq_file)

    result_root = opt.output_root if opt.output_root != '' else '.'
    for set in seqs:
        result_root = os.path.join(result_root, set['subdir'])
        mkdir_if_missing(result_root)
        for seqi, seq_entry in enumerate(set["seqs"]):
            seq = seq_entry["seq"]
            kps = seq_entry["kps"]
            tbs = None
            if opt.cmc_on:
                tbs = [tuple(tb) for tb in kps.values()]
            input_video = os.path.join(opt.input_root, set['subdir'], seq)


            logger.info(f'Starting tracking for {input_video}...')
            dataloader = datasets.LoadVideo(input_video, opt.img_size)
            if len(dataloader) == 0:
                logger.info(f"Sequence {input_video} is missing or empty.")
                continue
            result_filename = os.path.join(result_root, 'results.txt')
            frame_rate = dataloader.frame_rate

            frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
            #tbs = [(445, 175, 35, 35), (680, 210, 35, 35)]
            eval_seq(opt, dataloader, 'mot', result_filename,
                     save_dir=frame_dir, show_image=False, frame_rate=frame_rate,tbs=tbs)

            if opt.output_format == 'video':
                output_video_path = osp.join(result_root, osp.splitext(osp.basename(input_video))[0] + '-results.mp4')
                #cmd_str = 'ffmpeg -f image2 -r:v 30 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
                cmd_str = 'ffmpeg -f image2 -r:v 30 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(result_root, 'frame'), output_video_path)
                os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
