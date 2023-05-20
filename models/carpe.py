# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class CARPE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_keypoints, num_links, num_queries_keypoints, num_queries_links):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries_keypoints = num_queries_keypoints
        self.num_queries_links = num_queries_links
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_keypoints = nn.Linear(hidden_dim, num_keypoints + 1)
        self.pos_keypoints = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries_keypoints, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # Keep only the output
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        outputs_class = self.class_keypoints(hs)
        outputs_coord = self.pos_keypoints(hs).sigmoid()

        out = {'pred_logits_keypoints': outputs_class[-1], 'pred_keypoints': outputs_coord[-1]}
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, nb_keypoints,nb_links, matcher,matcher_links, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes_keypoints = nb_keypoints
        self.num_classes_links = nb_links
        self.matcher = matcher
        self.matcher_links = matcher_links
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(nb_keypoints + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits_keypoints']
        idx = self._get_src_permutation_idx(indices)

        # Concatenate the classes of all keypoints / images
        target_classes_o = torch.cat([t["labels_keypoints"][J] for t, (_, J) in zip(targets, indices)]).long()
        # Create a tensor with the "non-object class"
        target_classes = torch.full(src_logits.shape[:2], self.num_classes_keypoints,
                                    dtype=torch.int64, device=src_logits.device)
        # Set the index of the matched indices to the correct target
        target_classes[idx] = target_classes_o

        # Apply a special weight (from the configuration) to the no-object category
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_keypoints'][idx]
        target_boxes = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_keypoints'] = loss_bbox.sum() / num_boxes
        return losses


    def _get_src_permutation_idx(self, indices):
        # Create a tensor to select the correct item in batch
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # Get the source indices (from the outputs) that have been matched to one target
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Create a tensor to select the correct item in batch
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        # Get the source indices (from the targets) that have been matched to one output
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'keypoints': self.loss_keypoints
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels_keypoints"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


class PostProcessCOCO(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode

    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_positions = outputs[f'pred_logits_{self.mode}'], outputs[f'pred_{self.mode}']

        nb_class = out_logits.shape[-1] - 1


        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        assert len(out_logits) == len(target_sizes)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)


        target_sizes = target_sizes.expand(out_positions.shape[1], *target_sizes.shape).permute(1,0,2)
        positions = out_positions * target_sizes

        image_ids = torch.stack([t["image_id"] for t in targets], dim=0).squeeze(1).cpu().numpy()

        results = []
        for image_id, score, label, position in zip(image_ids, scores, labels, positions):
            for s, l, p in zip(score, label, position):
                if(l < nb_class):
                    keypoints = np.zeros(nb_class * 3)
                    keypoints[l*3+0] = p[0]
                    keypoints[l*3+1] = p[1]
                    keypoints[l*3+2] = 2
                    
                    results.append({
                        'image_id': image_id, 'category_id': 1, 'score': s.item(), "nbr_keypoints": 1, 'area': 200, 'keypoints':list(keypoints)
                    })

        return results

class PostProcess(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode

    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        
        out_logits, out_pos = outputs[f'pred_logits_{self.mode}'], outputs[f'pred_{self.mode}']


        target_sizes = torch.stack([t["orig_size"] for t in target], dim=0)
        assert len(out_logits) == len(target_sizes)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)


        target_sizes = target_sizes.expand(out_pos.shape[1], out_pos.shape[2]//target_sizes.shape[1], *target_sizes.shape).permute(2,0,1,3).reshape(out_pos.shape)
        pos = out_pos * target_sizes

        results = [{'scores': s, 'labels': l, self.mode: b} for s, l, b in zip(scores, labels, pos)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = CARPE(
        backbone,
        transformer,
        num_keypoints=args.nb_keypoints,
        num_queries_keypoints=args.nb_keypoints_queries,
        num_links=args.nb_links,
        num_queries_links=args.nb_links_queries
    )

    matcher = build_matcher(args,mode="keypoints")
    matcher_links = build_matcher(args, mode="links")
    weight_dict = {'loss_ce': 1, 'loss_keypoints': args.keypoints_loss_coef, 'loss_ce_links':1, 'loss_links':args.keypoints_loss_coef}

    losses = ['labels', 'keypoints']

    criterion = SetCriterion(args.nb_keypoints,args.nb_links, matcher=matcher, matcher_links=matcher_links, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'keypoints' : PostProcessCOCO('keypoints'), 'keypoints_': PostProcess('keypoints'), 'links_': PostProcess('links')}

    return model, criterion, postprocessors
