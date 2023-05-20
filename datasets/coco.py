# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import albumentations as al

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms,al_transforms,apply_augm,apply_occlusion_augmentation, segmentation_folder):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.al_transforms = al_transforms
        self.prepare = ConvertCocoPolysToMask()
        self.apply_augm = apply_augm
        self.apply_occlusion_augmentation = apply_occlusion_augmentation
        self.segmentation_folder = segmentation_folder

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self.apply_occlusion_augmentation:
            pass

        if self.al_transforms is not None and self.apply_augm: 
            img = self.al_transforms(image=img)["image"]
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        links = None
        if anno and "links" in anno[0]:
            links = [obj["links"] for obj in anno]
            links = torch.as_tensor(links, dtype=torch.float32)
            num_links = links.shape[0]
            if num_links:
                links = links.view(num_links, -1, 5)

        keep = (boxes[:, 3] >= boxes[:, 1]) & (boxes[:, 2] >= boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        if keypoints is not None:
            keypoints = keypoints[keep]

        if links is not None:
            links = links[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        target["image_id"] = image_id
        
        if keypoints is not None:
            labels_keypoints = torch.as_tensor(range(keypoints.shape[1]), dtype=torch.float32).expand(*keypoints.shape[:2])
            keep_keypoints =  keypoints[:,:,-1] > 0

            keypoints = keypoints[keep_keypoints][:,:-1]
            labels_keypoints = labels_keypoints[keep_keypoints]

            target["labels_keypoints"] = labels_keypoints
            target["keypoints"] = keypoints

        if links is not None:
            labels_links = torch.as_tensor(range(links.shape[1]), dtype=torch.float32).expand(*links.shape[:2])
            keep_links =  links[:,:,-1] > 0

            links = links[keep_links][:,:-1]
            labels_links = labels_links[keep_links]

            target["labels_links"] = labels_links
            target["links"] = links

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target



def make_coco_transforms(image_set, size, apply_augm):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        lst = []
        
        if apply_augm: 
            lst.append(T.RandomHorizontalFlip())
        lst.append(T.Resize(size))
        lst.append(normalize)
        return T.Compose(lst)       
        

    if image_set == 'val' or image_set =="test":
        return T.Compose([
            T.Resize(size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def albumentations_transform(image_set):
    if image_set == "train": 
        return al.Compose([
        al.ColorJitter(0.4, 0.4, 0.5, 0.2, p=0.6),
        al.RandomBrightnessContrast(p=0.5),
        al.ToGray(p=0.01),
        al.FancyPCA(p=0.3),
        al.ImageCompression(50, 80,p=0.1),
        al.RandomSunFlare(p=0.05),
        al.Solarize(p=0.05),
        al.GaussNoise(var_limit=(1.0,30.0), p=0.2)
      ])
    elif image_set == "val":
        return None
    elif image_set == "test":
        return None

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'person_keypoints'
    PATHS = {
        "train": (root / "train",  root/"annotations"/f'keypoints_train_{args.nb_keypoints}.json',root/"train_segm_npz"),
        "val": (root / "val",  root/"annotations"/f'keypoints_val_{args.nb_keypoints}.json',root/"val_segm_npz"),
        "test": (root / "test", root/"annotations"/f'keypoints_test_{args.nb_keypoints}.json',root/"test_segm_npz")
    }

    img_folder, ann_file, segmentation_folder = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, 
        transforms=make_coco_transforms(image_set,args.input_image_resize,args.apply_augmentation), 
        al_transforms=albumentations_transform(image_set),
        apply_augm=args.apply_augmentation, 
        apply_occlusion_augmentation=args.apply_occlusion_augmentation,
        segmentation_folder=segmentation_folder)
    return dataset
