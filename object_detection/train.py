from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.models.detection.ssd import SSD
import tqdm

from model import create_model
from dataset import SimpleDataset, create_simple_dataloader
from constants import DEVICE
from ssd_loss import MultiBoxLoss


def move_targets_to_device(targets: Tuple[Dict[str, torch.Tensor]], device: str=DEVICE) -> Tuple[Dict[str, torch.Tensor]]:
    """Moves the targets to a (cuda) device

    Args:
        targets (Tuple[Dict[str, torch.Tensor]]): _description_

    Returns:
        Tuple[Dict[str, torch.Tensor]]: _description_
    """
    return [{k: v.to(device) for (k, v) in target.items()} for target in targets]
    

def train(train_loader, model: SSD, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()
    
    for i in (pbar:=tqdm.trange(epoch)):
        # Batches
        for j, (images, targets) in enumerate(tqdm.tqdm(train_loader, leave=False)):
            optimizer.zero_grad()
            # Move to default device
            images = images.to(DEVICE)  # (batch_size (N), 3, 300, 300)
            targets = move_targets_to_device(targets, DEVICE)

            # Forward prop.
            loss_dict = model(
                images, targets
            )  
            # Loss
            loss = loss_dict["bbox_regression"] + loss_dict["classification"]
            loss.backward()
            # # Clip gradients, if necessary
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)
            # Update model
            optimizer.step()
        pbar.set_description(f"loss: {float(loss)}")



if __name__ == "__main__":
    model = create_model(num_classes=2, img_size=(128, 128))
    train_loader = create_simple_dataloader()
    optimizer = optim.Adam(model.parameters())
    epochs = 100
    train(model=model, train_loader=train_loader, optimizer=optimizer, epoch=epochs)
