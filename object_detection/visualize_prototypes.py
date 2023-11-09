import torch
from model import PIPSSD
from dataset import create_simple_dataloader
from torchvision.utils import save_image
from constants import DEVICE
from utils import move_targets_to_device


def visualize(model: PIPSSD, data_loader):
    for image, image_prime, target in data_loader:
        image = image.to(DEVICE)
        # target = move_targets_to_device(target)
        output = model(
            image, target, calc_losses=False, calc_detections=True, calc_prototypes=True
        )
        pass
    pass


if __name__ == "__main__":
    config = {
        "pretraining_epochs": 2,
        "epochs": 10,
        "img_size": (256, 256),
        "batch_size": 64,
        "object_size": (10, 15),
        "num_shapes": 2,
        "localization": 1.0,
        "classification": 1.0,
        "align": 1.0,
        "tanh": 1.0,
    }

    model = torch.load("object_detection/mnist_model_epoch:1.pt")
    data_loader = create_simple_dataloader(
        config["batch_size"],
        num_shapes=3 + 1,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    visualize(model, data_loader)
    pass
