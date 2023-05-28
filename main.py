#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ravi Joshi
# Date: 2023/05/28
# Src: https://github.com/facebookresearch/detectron2/issues/4368
# Src: https://github.com/facebookresearch/detectron2/issues/810
# Src: https://github.com/facebookresearch/detectron2#getting-started

import os
import cv2
import json
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import hooks
from detectron2.structures import BoxMode

from trainer import Trainer
from val_loss_hook import ValLossHook


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_custom_cfg():
  # DatasetCatalog.clear()
  for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

  cfg = get_cfg()

  config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
  cfg.merge_from_file(model_zoo.get_config_file(config))
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)

  cfg.DATASETS.TRAIN = ("balloon_train",)
  cfg.DATASETS.TEST = ("balloon_val",)

  cfg.DATALOADER.NUM_WORKERS = 4

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
  cfg.MODEL.MASK_ON = True

  cfg.SOLVER.MAX_ITER = 1000
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.CHECKPOINT_PERIOD = 100
  cfg.SOLVER.IMS_PER_BATCH = 2

  cfg.TEST.EVAL_PERIOD = 10

  cfg.OUTPUT_DIR = "output"
  return cfg


def main():
  cfg = get_custom_cfg()

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  # setup trainer
  trainer = Trainer(cfg)

  # creates a hook that after each iter calculates the validation loss on the next batch
  # Register the hoooks
  trainer.register_hooks([ValLossHook(cfg)])

  # The PeriodicWriter needs to be the last hook, otherwise it wont have access to valloss metrics
  # Ensure PeriodicWriter is the last called hook
  periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, hooks.PeriodicWriter)]
  all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, hooks.PeriodicWriter)]
  trainer._hooks = all_other_hooks + periodic_writer_hook

  trainer.resume_or_load(resume=False)
  trainer.train()


if __name__ == "__main__":
  main()
