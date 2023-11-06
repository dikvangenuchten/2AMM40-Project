from typing import Dict, List, Tuple
import torch
from torch import optim
from torchvision.models.detection.ssd import SSD
import tqdm
import wandb

from model import PIPSSDLoss, create_model
from dataset import create_simple_dataloader, label_to_caption
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


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    x_prime: torch.Tensor,
    targets: Tuple[Dict[str, torch.Tensor]],
    bbox_regression_mul: float = 1.0,
    classification_mul: float = 1.0,
    align_loss_mul: float = 1.0,
    tanh_loss_mul: float = 1.0,
) -> torch.Tensor:
    optimizer.zero_grad()
    # Stack the images and images prime
    # This is done for the calculation of the align loss
    images = torch.cat((x, x_prime), dim=0).to(DEVICE)
    targets = move_targets_to_device(cat_targets(targets), DEVICE)

    # Forward prop.
    loss_dict: PIPSSDLoss = model(images, targets)
    # Loss
    # TODO add pip loss components
    loss = (
        bbox_regression_mul * loss_dict.bbox_regression
        + classification_mul * loss_dict.classification
        + align_loss_mul * loss_dict.align_loss
        + tanh_loss_mul * loss_dict.tanh_loss
    ) / (bbox_regression_mul + classification_mul + align_loss_mul + tanh_loss_mul)
    loss.backward()
    # # Clip gradients, if necessary
    # if grad_clip is not None:
    #     clip_gradient(optimizer, grad_clip)
    # Update model
    optimizer.step()
    return {
        "Loss": loss.detach(),
        "localization_loss": loss_dict.bbox_regression.detach(),
        "classification_loss": loss_dict.classification.detach(),
        "align_loss": loss_dict.align_loss.detach(),
        "tanh_loss": loss_dict.tanh_loss.detach(),
    }


def train(
    train_loader,
    test_loader,
    model: SSD,
    optimizer,
    epoch: int,
    pretraining_epochs: int,
    bbox_regression_mul: float = 1.0,
    classification_mul: float = 1.0,
    align_loss_mul: float = 1.0,
    tanh_loss_mul: float = 1.0,
) -> SSD:
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
        pretraining = i < pretraining_epochs
        # Batches
        for j, (images, images_prime, targets) in enumerate(
            tqdm.tqdm(train_loader, leave=False)
        ):
            loss = train_step(
                model,
                optimizer,
                images,
                images_prime,
                targets,
                bbox_regression_mul=bbox_regression_mul if not pretraining else 0,
                classification_mul=classification_mul if not pretraining else 0,
                align_loss_mul=align_loss_mul,
                tanh_loss_mul=tanh_loss_mul,
            )
            wandb.log(loss, commit=False)

        model.eval()
        for j, (images, _, targets) in enumerate(tqdm.tqdm(test_loader, leave=False)):
            # Stack the images and images prime
            # This is done for the calculation of the align loss
            images = torch.cat((images, images_prime), dim=0)
            images = images.to(DEVICE)

            targets = cat_targets(targets)
            targets = move_targets_to_device(targets, DEVICE)

            predicted = model(images)

        wandb_log_images(images, targets, predicted)

        pbar.set_description(str(loss))

    return model


def main():
    config = {
        "pretraining_epochs": 10,
        "epochs": 100,
        "img_size": (64, 64),
        "batch_size": 1024,
        "object_size": (10, 15),
        "num_shapes": 2,
        "localization": 1.0,
        "classification": 1.0,
        "align": 1.0,
        "tanh": 1.0,
    }

    wandb.init(
        project="pip-object-detection",
        entity="dikvangenuchten",
        config=config,
    )
    # Get the config from wandb for sweeps
    config = wandb.config

    model = create_model(
        # +1 for background class
        num_classes=config["num_shapes"] + 1,
        img_size=config["img_size"],
    )
    train_loader = create_simple_dataloader(
        config["batch_size"] * 50,
        num_shapes=2 + 1,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    test_loader = create_simple_dataloader(
        64,
        num_shapes=3 + 1,
        batch_size=64,
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    optimizer = optim.Adam(model.parameters())
    model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epoch=config["epochs"],
        pretraining_epochs=config["pretraining_epochs"],
    )

    wandb.save(model)


if __name__ == "__main__":
    main()
