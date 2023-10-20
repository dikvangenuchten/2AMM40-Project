from typing import Any, Tuple
from torch import nn
import torch
from torchviz import make_dot

import torchvision
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator, SSDHead
import tqdm

import constants


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._out_channels = []


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, for object detection.
    Adapted from:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/model.py#L532

    The details are in the SSD: MultiBox paper section 2.2 Training.
    https://arxiv.org/pdf/1512.02325.pdf

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """


def create_model(
    num_classes: int,
    img_size: Tuple[int, int],
    nms_treshold: float = 0.45,
    *args,
    **kwargs
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
        # For now we will only have shapes with aspect ration of 1
        [[1]],
    )

    num_anchors = anchor_generator.num_anchors_per_location()
    _head = SSDHead(_backbone.out_channels, num_anchors, num_classes)

    _model = SSD(
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
