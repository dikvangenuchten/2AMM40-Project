from typing import Dict, List, Tuple
import torch
from torch import optim
from torchvision.models.detection.ssd import SSD
import tqdm
import wandb

from model import PIPSSD, PIPSSDLoss, create_model
from dataset import create_simple_dataloader, label_to_caption
from constants import DEVICE
from mnist_dataset import create_mnist_dataloader
from utils import move_targets_to_device, cat_targets


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
                for image, target, prediction in zip(images[:64], targets, predicted)
            ]
        },
        commit=commit,
    )


def train_step(
    model: PIPSSD,
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
    images = torch.cat((x.to(DEVICE), x_prime.to(DEVICE)), dim=0)
    targets = move_targets_to_device(cat_targets(targets), DEVICE)

    # Forward prop.
    model_output = model.forward(
        images, targets, calc_losses=True, calc_detections=False, calc_prototypes=False
    )
    loss_dict: PIPSSDLoss = model_output["losses"]

    # Loss
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
        "loc_loss": loss_dict.bbox_regression.detach(),
        "class_loss": loss_dict.classification.detach(),
        "align_loss": loss_dict.align_loss.detach(),
        "tanh_loss": loss_dict.tanh_loss.detach(),
    }


def train(
    train_loader,
    test_loader,
    model: PIPSSD,
    optimizer,
    epoch: int,
    pretraining_epochs: int,
    bbox_regression_mul: float = 1.0,
    classification_mul: float = 1.0,
    align_loss_mul: float = 1.0,
    tanh_loss_mul: float = 1.0,
) -> PIPSSD:
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
        for j, (images, images_prime, targets) in enumerate(
            tqdm.tqdm(test_loader, leave=False)
        ):
            # Stack the images and images prime
            # This is done for the calculation of the align loss
            images = torch.cat((images, images_prime), dim=0)
            images = images.to(DEVICE)

            targets = cat_targets(targets)
            targets = move_targets_to_device(targets, DEVICE)

            output = model.forward(
                images,
                targets,
                calc_losses=True,
                calc_detections=True,
                calc_prototypes=True,
            )
            
            test_loss = {
                # "test_loss": output["losses"].detach(),
                "test_loc_loss": output["losses"].bbox_regression.detach(),
                "test_class_loss": output["losses"].classification.detach(),
                "test_align_loss": output["losses"].align_loss.detach(),
                "test_tanh_loss": output["losses"].tanh_loss.detach(),
            }
            wandb.log(test_loss, commit=False)
            # Only do 1 test batch
            break

        # TODO log test loss
        wandb_log_images(images, targets, output["detections"])
        del images
        del targets
        del output

        pbar.set_description("||".join(f"{k}:{float(v):.4f}" for k, v in loss.items()))

        torch.save(model, f"mnist_model_epoch:{i}.pt")
    return model


def main():
    config = {
        "pretraining_epochs": 2,
        "epochs": 10,
        "img_size": (128, 128),
        "batch_size": 256,
        "object_size": (16, 64),
        "num_shapes": 10,
        "localization": 1.0,
        "classification": 1.0,
        "align": 10.0,
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
    train_loader = create_mnist_dataloader(
        # config["batch_size"] * 500,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        train=True,
        # object_size=config["object_size"],
    )
    test_loader = create_mnist_dataloader(
        # 50,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        train=False,
        # object_size=config["object_size"],
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

    torch.save(model, "final_model_256.pt")
    wandb.save("final_model_256.pt")


if __name__ == "__main__":
    main()
