from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import tqdm

from constants import DEVICE
from custom_transforms import (
    TrivialAugmentWideNoShape,
    TrivialAugmentWideNoShapeWithColor,
)

NUM_OBJECTS = 2


class SimpleDataset(Dataset):
    """A simple dataset

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        num_shapes: int = 2,
        img_size: Tuple[int, int] = (128, 128),
        length: int = 10_000,
        object_size: Tuple[int, int] = (20, 40),
        num_objects: Tuple[int, int] = (1, 3),
        transform=None,
        target_transform=None,
        pip_net: bool = True,
    ) -> None:
        super().__init__()
        assert 1 < num_shapes < 5
        self._num_shapes = num_shapes
        self.length = length
        self.img_size = img_size
        self.object_size = object_size
        self._num_objects = num_objects

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

        images, self._targets = self._generate_dataset()
        self._raw_images = images
        self._images = [None] * len(self._raw_images)
        self._images_prime = [None] * len(self._raw_images)

    def _generate_dataset(self):
        return zip(
            *[
                generate_single_sample(
                    self.img_size,
                    object_size=self.object_size,
                    num_objects=self._num_objects,
                    num_shapes=self._num_shapes,
                )
                for _ in tqdm.trange(self.length)
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        # Create cache on first pass
        if self._images[index] is None:
            self._images[index] = self.transform1(self._raw_images[index])
        if self._images_prime[index] is None:
            self._images_prime[index] = self.transform2(self._raw_images[index])
        image = self._images[index]
        image_prime = self._images_prime[index]
        target = self._targets[index]

        return image, image_prime, target


def generate_single_sample(
    img_size: Tuple[int, int],
    object_size: Tuple[int, int],
    num_shapes: int,
    num_objects: Tuple[int, int],
):
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
        elif shape_type == 3:
            draw = draw_hexagon(draw, bbox, fill=0)
            labels.append(3)
        elif shape_type == 4:
            draw = draw_triangle(draw, bbox, fill=0)
            labels.append(4)
        else:
            assert False, f"{shape_type=} is unsupported."
        bboxes.append(bbox)

    return image, ({"boxes": torch.tensor(bboxes), "labels": torch.tensor(labels)})


def draw_triangle(draw, bbox, fill=0):
    x1, y1, x2, y2 = bbox
    draw.polygon(((x1, y1), (x1, y2), (x2, y2)), fill=fill)
    return draw


def draw_hexagon(draw, bbox, fill=0):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    points = [
        (x1, (y1 + h / 2)),
        (x1 + w / 4, y1),
        (x1 + 3 * w / 4, y1),
        (x1 + w, (y1 + h / 2)),
        (x1 + 3 * w / 4, y2),
        (x1 + w / 4, y2),
    ]
    draw.polygon(points, fill=fill)
    return draw


def label_to_caption(label: int) -> str:
    return ["square", "circle", "triangle"][label]


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
    num_shapes=2,
    num_objects=(1, 3),
) -> DataLoader:
    dataset = SimpleDataset(
        length=size,
        img_size=img_size,
        object_size=object_size,
        num_shapes=num_shapes,
        num_objects=num_objects,
        transform=None,
    )
    return DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=create_batch,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


if __name__ == "__main__":
    import os

    os.makedirs("example", exist_ok=True)

    image = Image.new("1", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    bbox = [14, 14, 114, 114]
    draw.rectangle(bbox, fill=0)
    image.save("example/square.png")

    image = Image.new("1", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    bbox = [14, 14, 114, 114]
    draw.ellipse(bbox, fill=0)
    image.save("example/circle.png")

    image = Image.new("1", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    bbox = [14, 14, 114, 114]
    draw_hexagon(draw, bbox, fill=0)
    image.save("example/hexagon.png")

    # loader = create_simple_dataloader(100, img_size=(40, 40), num_shapes=3, num_objects=(1,2))
    # for x in loader:
    #     pass
