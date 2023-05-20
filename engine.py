# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import numpy as np
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, logger = None):
    model.train()
    criterion.train()
    print_freq = 10
    len_dl = len(data_loader)
    pbar = tqdm(data_loader)
    pbar.set_description(f"Epoch {epoch}, loss = init")
    for i, (samples, targets) in enumerate(pbar):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not math.isfinite(losses.item()):
            print("Loss is {}, stopping training".format(losses.item()))
            print(loss_dict_reduced)
            sys.exit(1)

        if logger is not None: 
            logger.add_scalar("Loss/train",losses.item(),len_dl*epoch + i)

        optimizer.zero_grad()
        losses.backward()
        pbar.set_description(f"Epoch {epoch}, loss = {losses.item():.4f}")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch=0, logger=None,nb_keypoints=24):
    model.eval()
    criterion.eval()
    iou_types = ["keypoints"]
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # From openpifpaf
    CAR_SIGMAS = [0.05] * nb_keypoints
    coco_evaluator.set_scale(np.array(CAR_SIGMAS))

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        results = postprocessors['keypoints'](outputs, targets)
        if coco_evaluator is not None:
            coco_evaluator.update_keypoints(results)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if logger is not None:
            stats = coco_evaluator.coco_eval['keypoints'].stats.tolist()
            logger.add_scalar("AP", stats[0], epoch)  # for the checkpoint callback monitor.
            logger.add_scalar("val/AP", stats[0],epoch)
            logger.add_scalar("val/AP.5", stats[1],epoch)
            logger.add_scalar("val/AP.75", stats[2],epoch)
            logger.add_scalar("val/AP.med", stats[3],epoch)
            logger.add_scalar("val/AP.lar", stats[4],epoch)
            logger.add_scalar("val/AR", stats[5],epoch)
            logger.add_scalar("val/AR.5", stats[6],epoch)
            logger.add_scalar("val/AR.75", stats[7],epoch)
            logger.add_scalar("val/AR.med", stats[8],epoch)
            logger.add_scalar("val/AR.lar", stats[9],epoch)

    return coco_evaluator
