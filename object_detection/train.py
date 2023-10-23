from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.models.detection.ssd import SSD
import tqdm
import wandb

from model import create_model
from dataset import SimpleDataset, create_simple_dataloader
from constants import DEVICE
from ssd_loss import MultiBoxLoss


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


def convert_to_box_data(target):
    return [
        {
            # one box expressed in the default relative/fractional domain
            "domain": "pixel",
            "position": {
                "minX": int(box[0]),
                "maxX": int(box[2]),
                "minY": int(box[1]),
                "maxY": int(box[3]),
            },
            "class_id": int(label),
            # "box_caption": int(label),
            "scores": {"score": float(score)},
        }
        for (box, label, score) in zip(
            target["boxes"],
            target["labels"],
            target.get("scores", torch.ones_like(target["labels"])),
        )
    ]


def wandb_log_images(
    images: torch.Tensor,
    targets: List[Tuple[Dict[str, torch.Tensor]]],
    predicted: List[Dict[str, torch.Tensor]],
    commit: bool = True,
) -> None:
    wandb.log(
        {
            "images": [
                wandb.Image(
                    image,
                    boxes={
                        "ground_truth": {"box_data": convert_to_box_data(target)},
                        "prediction": {"box_data": convert_to_box_data(prediction)},
                    },
                )
                for image, target, prediction in zip(images, targets, predicted)
            ]
        },
        commit=commit,
    )


def train(train_loader, test_loader, model: SSD, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    wandb.watch(model)

    for i in (pbar := tqdm.trange(epoch)):
        model.train()
        # Batches
        for j, (images, targets) in enumerate(tqdm.tqdm(train_loader, leave=False)):
            optimizer.zero_grad()
            # Move to default device
            images = images.to(DEVICE)  # (batch_size (N), 3, 300, 300)
            targets = move_targets_to_device(targets, DEVICE)

            # Forward prop.
            loss_dict = model(images, targets)
            # Loss
            loss = loss_dict["bbox_regression"] + loss_dict["classification"]
            loss.backward()
            wandb.log({"train_loss": loss}, commit=False)
            # # Clip gradients, if necessary
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)
            # Update model
            optimizer.step()

        model.train(False)
        for j, (images, targets) in enumerate(tqdm.tqdm(test_loader, leave=False)):
            images = images.to(DEVICE)
            targets = move_targets_to_device(targets, DEVICE)
            predicted = model(images)

        wandb_log_images(images, targets, predicted)

        pbar.set_description(f"loss: {float(loss)}")


def main():
    config = {
        "epochs": 100,
        "img_size": (128, 128),
        "batch_size": 128,
        "object_size": (20, 40),
        "num_shapes": 2,
    }

    wandb.init(project="pip-object-detection", entity="dikvangenuchten", config=config)

    model = create_model(
        # +1 for background class
        num_classes=config["num_shapes"] + 1,
        img_size=config["img_size"]
    )
    train_loader = create_simple_dataloader(
        100_000,
        num_shapes=2 + 1,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    test_loader = create_simple_dataloader(
        100,
        num_shapes=2 + 1,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    optimizer = optim.Adam(model.parameters())
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epoch=config["epochs"],
    )


if __name__ == "__main__":
    main()
