#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ravi Joshi
# Date: 2023/05/28
# Src: https://github.com/facebookresearch/detectron2/issues/4368
# Src: https://github.com/facebookresearch/detectron2/issues/810

import os

from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.data.build import build_detection_train_loader

from custom_tensorboardx_writer import CustomTensorboardXWriter


class Trainer(DefaultTrainer):
  @classmethod
  def build_train_loader(cls, cfg):
    """
    Train loader with custom data augmentation
    """
    # augmentations = [
    #     T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1080, sample_style="choice"),
    #     T.RandomFlip(),
    # ]
    # mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
    # return build_detection_train_loader(cfg, mapper=mapper)
    return build_detection_train_loader(cfg)

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, cfg, True, output_folder)

  def build_writers(self):
    """
    Overwrites the default writers to contain our custom tensorboard writer

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        CommonMetricPrinter(self.max_iter),
        JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        CustomTensorboardXWriter(self.cfg.OUTPUT_DIR),
    ]
