import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import re

from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from mmdet.apis import init_detector, inference_detector, show_result


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)


    def __enter__(self):
        def tryint(s):
            try:
                return int(s)
            except ValueError:
                return s

        def str2int(v_str):
            return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

        def sort_humanly(v_list):
            return sorted(v_list, key=str2int)

        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            self.files = os.listdir(self.args.VIDEO_PATH)
            self.files = sort_humanly(self.files)

            _image = cv2.imread(self.args.VIDEO_PATH + self.files[0])
            self.im_width = int(_image.shape[1])
            self.im_height = int(_image.shape[0])

            for _ in range(2):  # The first frame is repeated three times
                self.files = np.insert(self.files, 0, '1.jpg',axis=0)

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        config_file = './models/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
        checkpoint_file = './models/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.pth'
        detector_model = init_detector(config_file, checkpoint_file)
        idx_frame = 0

        if self.args.VIDEO_PATH[-3] == '1':
            f = open('./results/Track1' + self.args.VIDEO_PATH[-2] + '.txt', 'w')
        else:
            f = open('./results/Track' + self.args.VIDEO_PATH[-2] + '.txt', 'w')

        for img_path in self.files:
            img_path = self.args.VIDEO_PATH + img_path
            idx_frame += 1

            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()

            # read image
            ori_im = cv2.imread(img_path)
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            result = inference_detector(detector_model, im)
            result = result[0][0]
            bbox_xywh =  np.zeros((result.shape[0],4))
            bbox_xywh[:, 0] = 0.5 * result[:, 0] + 0.5 * result[:, 2]
            bbox_xywh[:, 1] = 0.5 * result[:, 1] + 0.5 * result[:, 3]
            bbox_xywh[:, 2] = result[:, 2] - result[:, 0]
            bbox_xywh[:, 3] = result[:, 3] - result[:, 1]
            cls_conf = result[:, 4]

            if bbox_xywh is not None:
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    # write to txt
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        w = x2 - x1
                        h = y2 - y1
                        person_id = int(identities[i]) if identities is not None else 0
                        person_conf = 0.9

                        f.write(str(idx_frame-2) + ',' + str(person_id) + ',' + str(x1) + ',' + \
                                str(y1) + ',' + str(w) + ',' + str(h) + ',' + str(person_conf) + ',0\n')

            end = time.time()
            print("frame: {:d}, time: {:.03f}s, fps: {:.03f}".format(idx_frame, end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", dest="display", action="store_true", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default=False)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
