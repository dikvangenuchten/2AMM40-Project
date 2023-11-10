import torch
import wandb
from model import PIPSSD
from dataset import create_simple_dataloader
from torchvision.utils import save_image
from constants import DEVICE
from mnist_dataset import create_mnist_dataloader
from train import wandb_log_images
from utils import move_targets_to_device

def topk_boxes(output, topk: int=2):
    return {
        "boxes": output["boxes"][:topk],
        "scores": output["scores"][:topk],
        "labels": output["labels"][:topk]
    }
    pass

def visualize(model: PIPSSD, data_loader):
    for image, image_prime, target in data_loader:
        image = image.to(DEVICE)
        # target = move_targets_to_device(target)
        output = model(
            image, target, calc_losses=False, calc_detections=True, calc_prototypes=True
        )
        
        detections = [topk_boxes(det) for det in output["detections"]]
        wandb_log_images(image[:32], target, detections, commit=True)


if __name__ == "__main__":
    config = {
        "pretraining_epochs": 2,
        "epochs": 10,
        "img_size": (100, 100),
        "batch_size": 256,
        "object_size": (20, 40),
        "num_shapes": 4,
        "localization": 1.0,
        "classification": 1.0,
        "align": 1.0,
        "tanh": 1.0,
    }
    
    wandb.init(
        project="pip-object-detection-visualization",
        entity="dikvangenuchten",
        config=config,
    )

    model = torch.load("object_detection/final_model_256.pt")
    data_loader = create_simple_dataloader(
        config["batch_size"] * 2,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        num_shapes=config["num_shapes"] + 1,
        num_objects=(1, 2),
        object_size=config["object_size"],
    )
    visualize(model, data_loader)
    pass
