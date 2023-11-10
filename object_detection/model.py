from collections import OrderedDict, namedtuple
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import warnings
import numpy as np
from torch import Tensor, nn
import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchviz import make_dot

import torchvision
from torchvision.utils import save_image, make_grid
from torchvision.models.detection.ssd import (
    SSD,
    DefaultBoxGenerator,
    SSDHead,
    SSDRegressionHead,
    SSDClassificationHead,
    SSDScoringHead,
)
from torchvision.ops import boxes as box_ops
import tqdm
import wandb

import constants


class NonNegativeConv2d(nn.Conv2d):
    """Convolutional variant of the NonNegLinear of PIPNet

    Required because object detection is not Location Invariant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        # super().register_forward_pre_hook(self._clamp_weights)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, F.relu(self.weight), self.bias)

    def clamp_weights(self, *args):
        """Before every forward call the weights are explicitly clamped to 0"""
        w_rg = self.weight.requires_grad
        b_rg = self.bias.requires_grad
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.weight.clamp_min_(0.0)
        self.bias.clamp_min_(0.0)
        self.weight.requires_grad = w_rg
        self.bias.requires_grad = b_rg


PIPSSDLoss = namedtuple(
    "PIPSSDLoss", ["bbox_regression", "classification", "align_loss", "tanh_loss"]
)


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
@torch.jit.script
def align_loss(inputs, targets, EPS: float = 1e-12):
    # assert inputs.shape == targets.shape
    # assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss


class PIPSSD(SSD):
    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict = super().compute_loss(targets, head_outputs, anchors, matched_idxs)
        loss_dict["align_loss"] = self._compute_align_loss(head_outputs["add_on_out"])
        loss_dict["tanh_loss"] = self._compute_tanh_loss(head_outputs["add_on_out"])
        return PIPSSDLoss(**loss_dict)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
        calc_losses: bool = True,
        calc_detections: bool = True,
        calc_prototypes: bool = False,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Copied from torchvision.models.detection.ssd and adjusted to allow the calculation of prototypes.

        Seperated it into smaller functions and simplified it due to some assumptions we adhere to such as:
        * Always same image size
        """
        images_, targets = self._prepare_input(images, targets)

        # get the features from the backbone
        features = self.backbone(images_.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images_, features)

        output = {}
        if calc_losses:
            output["losses"] = self._calc_losses(targets, head_outputs, anchors)
        if calc_detections:
            output["detections"] = self.postprocess_detections(
                head_outputs, anchors, images_.image_sizes
            )

            # We do not reshape the images within the model so this step is redundent
            # detections = self.transform.postprocess(detections, images_.image_sizes, original_image_sizes)
        if calc_prototypes:
            output["prototypes"] = self._calc_prototypes(
                head_outputs, images, output["detections"]
            )

        return output

    def _calc_prototypes(self, head_outputs, images, detections):
        # return
        # Get idx for examples where a specific prototype is very present
        batch_idx, prototype_idx, h_idx, w_idx = torch.where(
            head_outputs["add_on_out"] > 0.9
        )
        num_classes = self.head.classification_head.num_columns
        weights = self.head.classification_head.classification_layers[0].weight
        prototype_weights_per_class = [
            weights[list(range(c, weights.shape[0], num_classes))]
            for c in range(num_classes)
        ]

        # For each class get the x most important prototypes
        x = 5
        most_important_prototypes_per_class = [None] * num_classes
        for c in range(num_classes):
            _, indices = prototype_weights_per_class[c].flatten().topk(x, sorted=True)
            most_important_prototypes = np.unravel_index(
                indices.cpu(), prototype_weights_per_class[c].shape
            )[1]
            most_important_prototypes_per_class[c] = most_important_prototypes

        # For each prototype get the top y most prominent (i.e. highest softmax)
        y = 4 * 4
        img_w, img_h = images.shape[-2:]
        fea_w, fea_h = head_outputs["add_on_out"].shape[-2:]
        mul_w = int(img_w / fea_w)
        mul_h = int(img_h / fea_h)
        sample_prototypes = [None] * head_outputs["add_on_out"].shape[1]
        sample_prototypes_weight = [None] * head_outputs["add_on_out"].shape[1]
        log_images = []
        os.makedirs("prototypes", exist_ok=True)
        for proto_idx in range(head_outputs["add_on_out"].shape[1]):
            value, idx = head_outputs["add_on_out"][:, proto_idx].flatten(1).max(1)
            topk_value, batch_idx = value.topk(y, sorted=True)
            w_idx, h_idx = np.unravel_index(
                idx[batch_idx].cpu(), head_outputs["add_on_out"].shape[-2:]
            )
            # If the prototype is on the edge of the image,
            # part of the 'focus' would be outside of the original image.
            # To make it easier on ourselves we clip it to the inside.
            patch_padding = 3
            w_idx = w_idx.clip(patch_padding, fea_w - (patch_padding + 1))
            h_idx = h_idx.clip(patch_padding, fea_h - (patch_padding + 1))
            # Convert w,h to patches of the original image.
            w_idx_min, h_idx_min = (w_idx - patch_padding) * mul_w, (
                h_idx - patch_padding
            ) * mul_h
            w_idx_max, h_idx_max = (w_idx + (patch_padding + 1)) * mul_w, (
                h_idx + (patch_padding + 1)
            ) * mul_h

            sample_prototypes[proto_idx] = torch.stack(
                [
                    images[
                        batch_idx[i],
                        :,
                        w_idx_min[i] : w_idx_max[i],
                        h_idx_min[i] : h_idx_max[i],
                    ]
                    for i in range(len(batch_idx))
                ],
                0,
            )
            sample_prototypes_weight[proto_idx] = topk_value.mean()

            log_images.append(
                wandb.Image(
                    make_grid(
                        sample_prototypes[proto_idx], nrow=4, pad_value=0.5
                    ).cpu(),
                    caption=f"Example prototypes {proto_idx}",
                ),
            )

            save_image(
                sample_prototypes[proto_idx],
                f"prototypes/{proto_idx}.png",
                nrow=int(np.sqrt(y)),
            )
        wandb.log({"prototype examples": log_images})

        log_images = []
        class_weights = torch.cat(prototype_weights_per_class[1:])
        for img, softmaxes in zip(images, head_outputs["add_on_out"]):
            # Get the idx that were important for the classification of the object
            most_important_prototypes_value, most_important_prototypes_idx = (
                F.conv2d(softmaxes, class_weights).flatten(1).max(1)
            )
            # Look up which prototype was important for that.
            w, h = np.unravel_index(
                most_important_prototypes_idx.cpu(), softmaxes.shape[1:]
            )
            important_prototypes_idx = softmaxes[:, w, h].argmax(0)

            log_images.append(
                wandb.Image(
                    img.cpu(),
                    caption="Important prototypes found:"
                    + " ".join(
                        f"{split.cpu().numpy()}"
                        for split in important_prototypes_idx.split(num_classes - 1)
                    ),
                )
            )

        wandb.log({"samples_with_prototype_idx": log_images}, commit=False)

    def _calc_losses(self, targets, head_outputs, anchors):
        losses = {}
        matched_idxs = []
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (anchors_per_image.size(0),),
                            -1,
                            dtype=torch.int64,
                            device=anchors_per_image.device,
                        )
                    )
                    continue

                match_quality_matrix = box_ops.box_iou(
                    targets_per_image["boxes"], anchors_per_image
                )
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        return losses

    def _prepare_input(self, images, targets):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        # get the original image sizes
        # original_image_sizes: List[Tuple[int, int]] = []
        # for img in images:
        #     val = img.shape[-2:]
        #     torch._assert(
        #         len(val) == 2,
        #         f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        #     )
        #     original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # We do not generate degenerate boxes, so we can skip this check
        if False and targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        return images, targets

    @staticmethod
    @torch.jit.script
    def _compute_align_loss(proto_features: torch.Tensor) -> torch.Tensor:
        """Computes the align loss.

        The align loss ensures that patches that are (to a human) similar,
        also are similar within the latent space of the model. Which in the case
        of PIP are the prototypes.

        Note: It is assumed that the input to the model was: torch.stack(x, x_{prime})
        Where x, is the original image, and x_{prime} is a augmented version which
        preserves the classification and localisation of the original image.
        """
        pf1, pf2 = proto_features.chunk(2)

        embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
        embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

        return (
            align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())
        ) / 2.0

    @staticmethod
    @torch.jit.script
    def _compute_tanh_loss(proto_features: torch.Tensor) -> torch.Tensor:
        """Compute the tanh loss.

        The tanh loss incentivices the model to use all prototypes available.
        It is achieved by maximizing the maximum value of each prototype within the batch.
        """
        pooled = F.adaptive_max_pool2d(proto_features, (1, 1)).flatten()
        return -torch.log(torch.tanh(torch.sum(pooled, dim=0)) + 1e-12).mean()


class PIPHead(nn.Module):
    def __init__(
        self, in_channels: List[int], num_anchors: List[int], num_classes: int
    ):
        super().__init__()
        self.classification_head = PIPClassificationHead(
            in_channels, num_anchors, num_classes
        )
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self: "PIPHead", x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits, add_on_out = self.classification_head(x)
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": cls_logits,
            "add_on_out": add_on_out,
        }


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class PIPClassificationHead(SSDScoringHead):
    def __init__(
        self, in_channels: List[int], num_anchors: List[int], num_classes: int
    ):
        cls_logits = nn.ModuleList()
        add_on_layers = nn.ModuleList()
        classification_layers = nn.ModuleList()

        for channels, anchors in zip(in_channels, num_anchors):
            add_on_layer = nn.Softmax(dim=1)
            add_on_layers.append(add_on_layer)
            classification_layer = NonNegativeConv2d(
                channels, num_classes * anchors, kernel_size=1
            )
            classification_layers.append(classification_layer)
            cls_logits.append(
                nn.Sequential(
                    # nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    add_on_layer,
                    classification_layer,
                )
            )
            # cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)
        self.add_on_layers = add_on_layers
        self.classification_layers = classification_layers
        self.num_anchors = num_anchors

    @staticmethod
    def _get_result_from_module_list(module_list, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(module_list)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(module_list):
            if i == idx:
                out = module(x)
        return out

    @staticmethod
    def _permute_output(tensor, num_columns):
        # Permute output from (N, A * K, H, W) to (N, HWA, K).
        N, _, H, W = tensor.shape
        tensor = tensor.view(N, -1, num_columns, H, W)
        tensor = tensor.permute(0, 3, 4, 1, 2)
        tensor = tensor.reshape(N, -1, num_columns)  # Size=(N, HWA, K)
        return tensor

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []
        all_add_on_outs = []

        for i, (features, num_anchors) in enumerate(zip(x, self.num_anchors)):
            add_on_out = self._get_result_from_module_list(
                self.add_on_layers, features, i
            )
            results = self._get_result_from_module_list(
                self.classification_layers, add_on_out, i
            )

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            all_add_on_outs.append(
                add_on_out
                # self._permute_output(add_on_out, self.num_columns * num_anchors)
            )
            all_results.append(self._permute_output(results, self.num_columns))

        return torch.cat(all_results, dim=1), torch.cat(all_add_on_outs, dim=1)


def create_model(
    num_classes: int,
    img_size: Tuple[int, int],
    nms_treshold: float = 0.45,
    *args,
    **kwargs,
) -> SSD:
    pretrained_backbone = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT
    )

    # Extract the layers we want (Just the beginning)
    _backbone = nn.Sequential(
        pretrained_backbone.conv1,
        pretrained_backbone.bn1,
        pretrained_backbone.relu,
        pretrained_backbone.maxpool,
        pretrained_backbone.layer1,
        # pretrained_backbone.layer2,
        # pretrained_backbone.layer3,
        # pretrained_backbone.layer4,
    )

    _backbone.out_channels = [64]
    anchor_generator = DefaultBoxGenerator(
        # We do not add any aspect ratios, only using the default (1 and s'k)
        [[]],
    )

    num_anchors = anchor_generator.num_anchors_per_location()
    _head = PIPHead(_backbone.out_channels, num_anchors, num_classes)

    _model = PIPSSD(
        backbone=_backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=img_size,
        head=_head,
        nms_thresh=nms_treshold,
    )
    _model.to(constants.DEVICE)
    return _model


if __name__ == "__main__":
    num_classes = 2
    img_size = (128, 128)
    model = create_model(num_classes=num_classes, img_size=img_size)
    model.training = False
    batch_size = 30
    for _ in tqdm.trange(1000):
        x = torch.rand((batch_size, 3, *img_size)).to(constants.DEVICE)
        y = model(x)

    graph = make_dot(tuple(y[0].values()), params=dict(model.named_parameters()))
    graph.format = "png"
    graph.render("ssd_model.png")
