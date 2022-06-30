from lib.pytracking.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import os
import numpy as np
from lib.utils.merge import merge_template_search_vit
from lib.train.models.vit_crt import build_vitcrt
from lib.pytracking.tracker.vit_crt_utils import Preprocessor
from lib.utils.box_ops import clip_box


class VIT_CRT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(VIT_CRT, self).__init__(params)
        network = build_vitcrt(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.window = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

    def initialize(self, image, info: dict):
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        hanning = np.hanning(20)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _convert_score(self, score):
        # 1, 400, 2 -> Permute(2, 1, 0) -> 2, 400, 1 -> View(2, -1) -> 2, 400 -> Permute(1, 0) -> 400, 2
        score = score.cpu().permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        # 400, 2 -> Softmax(dim=1) -> .data[:, 0] -> 400, 1
        # .data[:, 0] -> Foreground class
        score = F.softmax(score, dim=1).data[:, 0].numpy()
        return score

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search_vit(feat_dict_list)
            # run the transformer
            out_dict, _ = self.network.forward_transformer(seq_dict=seq_dict)

        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        # get confidence score (whether the search region is reliable)
        if self.cfg.TEST.USE_CLS_SCORE:
            score = self._convert_score(out_dict["pred_logits"])

            score_xmin = (pred_boxes[:, 0] - pred_boxes[:, 2]/2) * 20
            score_ymin = (pred_boxes[:, 1] - pred_boxes[:, 3]/2) * 20
            w = pred_boxes[:, 2] * 20
            h = pred_boxes[:, 3] * 20

            # # window penalty
            pscore = score * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + \
                self.window * self.cfg.TRACK.WINDOW_INFLUENCE

            best_score_idx = np.argmax(pscore)
            best_score = pscore[best_score_idx]
            best_score_idx = np.unravel_index(best_score_idx, (20, 20))
            score_in_box = score_xmin < best_score_idx[0] < score_xmin + w and \
                           score_ymin < best_score_idx[1] < score_ymin + h

        else:
            score = -1
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update template
        if self.cfg.TEST.USE_CLS_SCORE:
            for idx, update_i in enumerate(self.update_intervals):
                if self.frame_id % update_i == 0 and best_score > 0.45 and score_in_box:
                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                    with torch.no_grad():
                        z_dict_t = self.network.forward_backbone(template_t)
                    self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return VIT_CRT
