from functools import cache
from os import cpu_count
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision.io import read_image
import torchvision
from torchvision import transforms, datasets
import numpy as np
import tqdm


from constants import DEVICE
from custom_transforms import (
    TrivialAugmentWideNoShape,
    TrivialAugmentWideNoShapeWithColor,
)
from dataset import create_batch


class SimpleMnistDataset(Dataset):
    """A simple dataset

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        train: bool = True,
        transform=None,
        target_transform=None,
        pip_net: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size

        self.base_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.uint8),
            ]
        )

        # Shape is important for our problem, so we do not augment that.
        self.transform1 = transforms.Compose(
            [
                TrivialAugmentWideNoShapeWithColor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        self.transform2 = transforms.Compose(
            [
                TrivialAugmentWideNoShape(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        self.mnist_dataset = datasets.MNIST(
            root="mnist/data/", train=train, download=True, transform=self.base_transform
        )
        self._bboxes = [
            self._generate_bbox(img_size) for _ in range(len(self.mnist_dataset))
        ]
        
        self.classes = self.mnist_dataset.classes

    @staticmethod
    def _generate_bbox(img_size):
        object_size = 28
        x_pos = torch.randint(0, img_size[0] - object_size, [])
        y_pos = torch.randint(0, img_size[1] - object_size, [])
        return [x_pos, y_pos, x_pos + object_size, y_pos + object_size]

    def __len__(self):
        return len(self.mnist_dataset)

    @cache
    def __getitem__(self, index) -> Any:
        img, label = self.mnist_dataset[index]
        bboxes = self._bboxes[index]
        target = {"boxes": torch.tensor([bboxes,]), "labels": torch.tensor([label + 1,])}

        full_image = torch.zeros(1, *self.img_size, dtype=torch.uint8)
        full_image[:, bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]] = img

        image = self.transform1(full_image)
        image_prime = self.transform2(full_image)

        return image, image_prime, target


def create_mnist_dataloader(
    batch_size=128,
    img_size=(128, 128),
    train=True,
) -> DataLoader:
    dataset = SimpleMnistDataset(
        img_size=img_size,
        transform=None,
        train=train
    )
    return DataLoader(
        dataset=dataset,
        shuffle=train,
        batch_size=batch_size,
        collate_fn=create_batch,
        num_workers=cpu_count(),
        prefetch_factor=16,
        pin_memory=True,
    )


if __name__ == "__main__":
    loader = create_mnist_dataloader(img_size=(64, 64))
    for x in loader:
        pass
