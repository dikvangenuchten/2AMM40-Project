from typing import Dict, Tuple

import torch
from constants import DEVICE


def move_targets_to_device(
    targets: Tuple[Dict[str, torch.Tensor]], device: str = DEVICE
) -> Tuple[Dict[str, torch.Tensor]]:
    """Moves the targets to a (cuda) device

    Args:
        targets (Tuple[Dict[str, torch.Tensor]]): _description_

    Returns:
        Tuple[Dict[str, torch.Tensor]]: _description_
    """
    return [{k: v.to(device) for (k, v) in target.items()} for target in targets]


def cat_targets(
    targets: Tuple[Dict[str, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor]]:
    return targets + targets
