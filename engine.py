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
import matplotlib.pyplot as plt
import cv2
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from tqdm import tqdm
import itertools

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
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch=0, logger=None,nb_keypoints=24,visualize_keypoints=False,out_folder=""):
    model.eval()
    criterion.eval()
    iou_types = ["keypoints"]
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # From openpifpaf
    CAR_SIGMAS = [0.05] * nb_keypoints
    coco_evaluator.set_scale(np.array(CAR_SIGMAS))

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)

        #eval_images =[t["image"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != "image" } for t in targets ]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        results = postprocessors['keypoints'](outputs, targets)

        if coco_evaluator is not None:
            coco_evaluator.update_keypoints(results)

        if visualize_keypoints:
            for target,image in zip(targets,eval_images):
                filt =[out for out in results if out["image_id"] == target["image_id"]]
                plot_and_save_keypoints_inference(image,target["image_id"].item(), filt, out_folder)


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



def plot_and_save_keypoints_inference(img, img_id,data, output_folder):
    skeleton = []#CAR_SKELETON_24 if True else CAR_SKELETON_66
    nb_kps = 24 if True else nb_kps
    colors =  plt.cm.tab20( (10./9*np.arange(20*9/10)).astype(int) )

    for lst in data: 
      kps = lst["keypoints"]
      all_found_kps = [ ]
      all_kps_coordinate=[]
      for i in range(nb_kps):
        x,y,z = tuple(kps[i*3:(i+1)*3])
        if z > 0:
          x= int(x)
          y = int(y)
          all_found_kps.append(int(i+1))
          all_kps_coordinate.append((x,y))
        else:
          all_kps_coordinate.append((-1,-1))
        
      set_of_pairs = set(itertools.permutations(all_found_kps,2))

      for idx, (a,b) in enumerate(skeleton):
        if (a,b) in set_of_pairs:
          r,g,bc,ac = colors[idx%len(colors)]                  
          cv2.line(img,all_kps_coordinate[a-1], all_kps_coordinate[b-1],color=[int(bc*255),int(g*255),int(r*255)],thickness=18)

      for a in all_found_kps:
        cv2.circle(img, all_kps_coordinate[a-1],20, color=[0,0,255],thickness=-1)
    cv2.imwrite(os.path.join(output_folder, f"{img_id}.jpg"),img)