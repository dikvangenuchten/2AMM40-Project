from torch import cuda

DEVICE = "cuda" if cuda.is_available() else "cpu"
