from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, nn
import torch
from torchviz import make_dot

import torchvision
from torchvision.models.detection.ssd import (
    SSD,
    DefaultBoxGenerator,
    SSDHead,
    SSDRegressionHead,
    SSDClassificationHead,
    SSDScoringHead,
)
import tqdm

import constants

class NonNegativeConv2d(nn.Conv2d):
    """Convolutional variant of the NonNegLinear of PIPNet
    
    Required because object detection is not Location Invariant.
    """
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, torch.relu(self.weight), self.bias)

class PIPSSD(SSD):
    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        return super().compute_loss(targets, head_outputs, anchors, matched_idxs)


class PIPHead(nn.Module):
    def __init__(
        self, in_channels: List[int], num_anchors: List[int], num_classes: int
    ):
        super().__init__()
        self.classification_head = PIPClassificationHead(
            in_channels, num_anchors, num_classes
        )
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
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
