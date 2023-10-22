from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision.io import read_image
import torchvision
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
        num_labels: int = 1,
        length: int = 10_000,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        assert num_labels == 1
        self._num_labels = num_labels
        self.length = length
        self.transform = transform
        self.target_transform = target_transform

        self._images, self._targets = self._generate_dataset()

    def _generate_dataset(self):
        return zip(
            *[
                generate_single_sample((256, 256), (10, 30), (1, 3))
                for _ in tqdm.trange(self.length)
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        image = self._images[index]
        target = self._targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


def generate_single_sample(
    img_size: Tuple[int, int],
    object_size: Tuple[int, int],
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
        x_size = torch.randint(*object_size, [])
        y_size = torch.randint(*object_size, [])
        x_pos = torch.randint(0, img_size[0] - x_size, [])
        y_pos = torch.randint(0, img_size[1] - y_size, [])

        bbox = [x_pos, y_pos, x_pos + x_size, y_pos + y_size]

        draw.rectangle(bbox, fill=0)
        labels.append(1)
        bboxes.append(bbox)

    return image, ({"boxes": torch.tensor(bboxes), "labels": torch.tensor(labels)})


def create_batch(to_be_batched) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Create a batch from the images, boxes, and labels

    For the build in SSD loss function we need a list of Mappings
    """
    images, targets = zip(*to_be_batched)
    return torch.stack(images, 0), targets


def create_simple_dataloader(size: int = 10_000, batch_size=128) -> DataLoader:
    dataset = SimpleDataset(
        length=size,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        ),
    )
    return DataLoader(
        dataset=dataset, shuffle=True, batch_size=batch_size, collate_fn=create_batch
    )


if __name__ == "__main__":
    dataset = SimpleDataset(transform=torchvision.transforms.PILToTensor())
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=256)
    for x in loader:
        pass
