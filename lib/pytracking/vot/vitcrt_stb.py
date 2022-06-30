from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import cv2
import torch
from lib.pytracking.vot import vot
import sys
import time
import os
from lib.pytracking.evaluation import Tracker
from lib.pytracking.vot.vot_utils import *
import numpy as np

'''vitcrt_stb class'''


class ViTCRT_STB(object):
    def __init__(self, tracker_name='vit_crt', para_name='baseline'):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "VOT22", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)
        self.W, self.H = 0, 0

    def initialize(self, img_rgb, region):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        region = [region.x, region.y, region.width, region.height]
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': region}
        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track
        outputs = self.tracker.track(img_rgb)

        pred_bbox = outputs['target_bbox']

        final_mask = mask_from_rect(pred_bbox, (self.W, self.H))
        return pred_bbox, final_mask


def run_vot_exp(tracker_name, para_name, vis=False):

    torch.set_num_threads(1)
    save_root = os.path.join('./debug/vot_debug', para_name)
    if vis and (not os.path.exists(save_root)):
        os.makedirs(save_root, exist_ok=True)
    tracker = ViTCRT_STB(tracker_name=tracker_name, para_name=para_name)
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    # mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, selection)
    if vis:
        with open(os.path.join(save_dir, 'info.txt'), 'a') as f:
            f.write(f'Frame[INIT]\t\tBbox = x: {selection.x}, y: {selection.y}, w: {selection.width}, h: {selection.height}\n')

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, m = tracker.track(image)
        rect = vot.Rectangle(x=b1[0], y=b1[1], width=b1[2], height=b1[3])
        handle.report(rect)
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_ori)
            # tracker box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg', '_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
            # original image + mask
            image_m = image_ori.copy().astype(np.float32)
            image_m[:, :, 1] += 127.0 * m
            image_m[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_m = cv2.drawContours(image_m, contours, -1, (0, 255, 255), 2)
            image_m = image_m.clip(0, 255).astype(np.uint8)
            image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
            save_path = os.path.join(save_dir, image_mask_name_m)
            cv2.imwrite(save_path, image_m)

            with open(os.path.join(save_dir, 'info.txt'), 'a') as f:
                f.write(f'\nFrame[{image_name}]\t\tBbox = x: {b1[0]}, y: {b1[1]}, w: {b1[2]}, h: {b1[3]}\n')
