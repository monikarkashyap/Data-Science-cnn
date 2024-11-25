from data_utils import prepare_data
from model import MNIST_CNN
from train import train_model, save_model
from evaluate import evaluate_model
from config import config
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # Prepare data
    trainloader, testloader = prepare_data(config["batch_size"])

    # Initialize model, criterion, and optimizer
    model = MNIST_CNN().to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train the model
    train_model(model, trainloader, criterion, optimizer, config["device"], config["num_epochs"])

    # Save the model
    save_model(model, config["model_path"])

    # Evaluate the model
    evaluate_model(model, testloader, config["device"])