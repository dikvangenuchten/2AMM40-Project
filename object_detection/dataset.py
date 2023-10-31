from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision.io import read_image
import torchvision
from torchvision import transforms
import numpy as np
import tqdm

from constants import DEVICE

NUM_OBJECTS = 2


class SimpleDataset(Dataset):
    """A simple dataset

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        num_shapes: int = 1,
        img_size: Tuple[int, int] = (128, 128),
        length: int = 10_000,
        object_size: Tuple[int, int] = (20, 40),
        transform=None,
        target_transform=None,
        pip_net: bool= True,
    ) -> None:
        super().__init__()
        assert 1 < num_shapes < 4
        self._num_shapes = num_shapes
        self.length = length
        self.img_size = img_size
        self.object_size = object_size

        # Shape is important for our problem, so we do not augment that.
        self.transform1 = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.uint8),
                TrivialAugmentWideNoShapeWithColor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        self.transform2 = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.uint8),
                TrivialAugmentWideNoShape(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        self._images, self._targets = self._generate_dataset()

    def _generate_dataset(self):
        return zip(
            *[
                generate_single_sample(
                    self.img_size,
                    object_size=self.object_size,
                    num_objects=(1, 3),
                    num_shapes=self._num_shapes,
                )
                for _ in tqdm.trange(self.length)
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        image = self._images[index]
        target = self._targets[index]

        # if self.transform1:
        image_ = self.transform1(image)
        # if self.transform2:
        image_prime = self.transform2(image)
        # if self.target_transform:
            # target = self.target_transform(target)
        return image_, image_prime, target


def generate_single_sample(
    img_size: Tuple[int, int],
    object_size: Tuple[int, int],
    num_shapes: int,
    num_objects: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random

    # Create a blank canvas on which we will draw everything.
    image = Image.new("1", img_size, 255)
    draw = ImageDraw.Draw(image)

    num_object = torch.randint(*num_objects, [])
    labels = []
    bboxes = []
    for _ in range(num_object):
        size = torch.randint(*object_size, [])
        # y_size = torch.randint(*object_size, [])
        x_pos = torch.randint(0, img_size[0] - size, [])
        y_pos = torch.randint(0, img_size[1] - size, [])

        bbox = [x_pos, y_pos, x_pos + size, y_pos + size]

        shape_type = torch.randint(1, num_shapes, [])
        if shape_type == 1:
            draw.rectangle(bbox, fill=0)
            labels.append(1)
        elif shape_type == 2:
            draw.ellipse(bbox, fill=0)
            labels.append(2)
        else:
            assert False, f"{shape_type=} is unsupported."
        bboxes.append(bbox)

    return image, ({"boxes": torch.tensor(bboxes), "labels": torch.tensor(labels)})


def create_batch(to_be_batched) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Create a batch from the images, boxes, and labels

    For the build in SSD loss function we need a list of Mappings
    """
    images, images_prime, targets = zip(*to_be_batched)
    return torch.stack(images, 0), torch.stack(images_prime, 0), targets


def create_simple_dataloader(
    size: int = 10_000,
    batch_size=128,
    img_size=(128, 128),
    object_size=(20, 30),
    num_shapes=1,
) -> DataLoader:
    dataset = SimpleDataset(
        length=size,
        img_size=img_size,
        object_size=object_size,
        num_shapes=num_shapes,
        transform=None,
    )
    return DataLoader(
        dataset=dataset, shuffle=True, batch_size=batch_size, collate_fn=create_batch
    )

# Copied from PIPNET
# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
# Copied from PIPNET

if __name__ == "__main__":
    loader = create_simple_dataloader(100, img_size=(10, 10))
    for x in loader:
        pass
