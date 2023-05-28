#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ravi Joshi
# Date: 2023/05/28
# Src: https://github.com/facebookresearch/detectron2/issues/4368
# Src: https://github.com/facebookresearch/detectron2/issues/810

import torch
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.engine import HookBase
import detectron2.utils.comm as comm


class ValLossHook(HookBase):

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg.clone()
    self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
    mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.NoOpTransform()])
    self._loader = iter(build_detection_train_loader(self.cfg, mapper=mapper))

  def after_step(self):
    """
        After each step calculates the validation loss and adds it to the train storage
    """
    data = next(self._loader)
    with torch.no_grad():
      loss_dict = self.trainer.model(data)

      losses = sum(loss_dict.values())
      assert torch.isfinite(losses).all(), loss_dict

      loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
      losses_reduced = sum(loss for loss in loss_dict_reduced.values())
      if comm.is_main_process():
        self.trainer.storage.put_scalars(val_total_loss=losses_reduced, **loss_dict_reduced)
