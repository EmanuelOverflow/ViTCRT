from got10k.trackers import Tracker
from lib.pytracking.evaluation import Tracker as TE
import importlib
import numpy as np


class ViTCRT(Tracker):
    def __init__(self, tracker_name='vit_crt', param_name='baseline', dataset_name='otb'):
        super(ViTCRT, self).__init__(name='ViTCRT')

        # create tracker
        tracker_module = importlib.import_module('lib.pytracking.tracker.vit_crt')
        self.tracker_class = tracker_module.get_tracker_class()

        tracker_info = TE(tracker_name, param_name, dataset_name, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def init(self, img, box):
        img = np.array(img)
        init_info = {'init_bbox': box}
        _ = self.tracker.initialize(img, init_info)

    def update(self, frame):
        frame = np.array(frame)
        outputs = self.tracker.track(frame)
        return outputs['target_bbox']
