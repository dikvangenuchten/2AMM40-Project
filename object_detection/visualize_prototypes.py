import torch
from model import PIPSSD
from dataset import create_simple_dataloader
from torchvision.utils import save_image


def visualize(model: PIPSSD, data_loader):
    for image, image_prime, target in data_loader:
        output = model(
            image,
            target,
            calc_losses = False,
            calc_detections = True,
            calc_prototypes = True
        )
        pass
    pass

if __name__ == "__main__":
    config = {
        "pretraining_epochs": 2,
        "epochs": 10,
        "img_size": (64, 64),
        "batch_size": 1024,
        "object_size": (10, 15),
        "num_shapes": 2,
        "localization": 1.0,
        "classification": 1.0,
        "align": 1.0,
        "tanh": 1.0,
    }

    model = torch.load("object_detection/final_model.pt").cpu()
    data_loader = create_simple_dataloader(
        512,
        num_shapes=3 + 1,
        batch_size=512,
        img_size=config["img_size"],
        object_size=config["object_size"],
    )
    visualize(model, data_loader)
    pass