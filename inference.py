#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ravi Joshi
# Date: 2023/05/28
# Src: https://github.com/facebookresearch/detectron2/issues/4368
# Src: https://github.com/facebookresearch/detectron2/issues/810
# Src: https://github.com/facebookresearch/detectron2#getting-started

import os
import cv2
import random

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from main import get_balloon_dicts


def main():
  # DatasetCatalog.clear()
  for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
  balloon_metadata = MetadataCatalog.get("balloon_train")

  cfg = get_cfg()
  config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
  cfg.merge_from_file(model_zoo.get_config_file(config))
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

  predictor = DefaultPredictor(cfg)

  dataset_dicts = get_balloon_dicts("balloon/val")
  for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE_BW,   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_name = os.path.join(cfg.OUTPUT_DIR, d["file_name"])
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    cv2.imwrite(save_name, out.get_image()[:, :, ::-1])
    print(f"file saved {save_name}")


if __name__ == "__main__":
  main()
