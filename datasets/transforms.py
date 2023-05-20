# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate



def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "keypoints" in target:
        keypoints = target["keypoints"]
        keypoints = keypoints[:]*torch.as_tensor([-1,1,1]) + torch.as_tensor([w,0,0])
        target["keypoints"] = keypoints

    if "links" in target:
        links = target["links"]
        links = links[:]*torch.as_tensor([-1,1,-1,1,1]) + torch.as_tensor([w,0,w,0,0])
        target["links"] = links

    return flipped_image, target


def resize(image, target, size):
    # size can be min_size (scalar) or (w, h) tuple
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints*torch.as_tensor([ratio_width, ratio_height, 1.0])
        target["keypoints"] = scaled_keypoints
        
    if "links" in target:
        links = target["links"]
        scaled_links = links*torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height, 1.0])
        target["links"] = scaled_links

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class Resize(object):
    def __init__(self, size ):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints/torch.tensor([w,h,1.0],dtype=torch.float32)
            target["keypoints"] = keypoints

        if "links" in target:
            links = target["links"]
            links = links/torch.tensor([w,h,w,h,1.0],dtype=torch.float32)
            target["links"] = links
            
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        raise ValueError()
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
