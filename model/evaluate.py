import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, testloader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))