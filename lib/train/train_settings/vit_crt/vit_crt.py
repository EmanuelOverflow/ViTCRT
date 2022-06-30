import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss, cross_entropy, smooth_l1_loss, binary_cross_entropy
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from lib.train.train_settings.vit_crt.base_functions import *
# network related
from lib.train.models.vit_crt import build_vitcrt
# forward propagation related
from lib.train.actors import ViTCRTActor
# for import modules
import importlib
import torch


def run(settings):
    settings.description = 'Training script for ViTCRT'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    net = build_vitcrt(cfg)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    # Loss functions and Actors
    objective = {'giou': giou_loss, 'l1': smooth_l1_loss}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
    settings.head_cls_type = getattr(cfg.MODEL, "HEAD_CLS_TYPE", "None")

    if cfg.MODEL.HEAD_CLS_TYPE != "None":
        objective['cls'] = cross_entropy
        loss_weight['cls'] = cfg.TRAIN.CE_WEIGHT

    if cfg.MODEL.HEAD_MASK:
        objective['mask'] = binary_cross_entropy
        loss_weight['mask'] = cfg.TRAIN.MASK_WEIGHT

    actor = ViTCRTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
