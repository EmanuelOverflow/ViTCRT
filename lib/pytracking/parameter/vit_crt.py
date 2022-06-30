from lib.pytracking.utils import TrackerParams
import os
from lib.pytracking.evaluation.environment import env_settings
from lib.config.vit_crt.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, f'experiments/vit_crt/{yaml_name}.yaml')
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir,
                                     f"checkpoints/train/vit_crt/{yaml_name}/ViTCRT_ep{cfg.TEST.EPOCH:04d}.pth.tar")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
