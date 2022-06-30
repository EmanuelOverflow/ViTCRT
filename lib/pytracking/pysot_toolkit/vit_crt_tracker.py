from lib.pytracking.evaluation import Tracker as TE
import importlib


class ViTCRT(object):
    def __init__(self, tracker_name='vit_crt', param_name='baseline', dataset_name='otb'):
        # create tracker
        tracker_module = importlib.import_module('lib.pytracking.tracker.vit_cr')
        self.tracker_class = tracker_module.get_tracker_class()

        tracker_info = TE(tracker_name, param_name, dataset_name, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def init(self, img, box):
        init_info = {'init_bbox': box}
        _ = self.tracker.initialize(img, init_info)

    def track(self, frame):
        outputs = self.tracker.track(frame)

        return {
            'bbox': outputs['target_bbox'],  # coordinates are x, y, w, h
            'best_score': outputs['conf_score']
        }
