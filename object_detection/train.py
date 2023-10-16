from torch import nn

from object_detection.object_detection import SSD


def train(model: nn.Module, dataset: nn):
    model.train()


if __name__ == "__main__":
    model = SSD()
    train()
