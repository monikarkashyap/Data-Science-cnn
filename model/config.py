import torch

config = {
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "model_path": "mnist_cnn.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}