# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch
from logger import get_logger

logger = get_logger(__name__)

class MelanomaModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MelanomaModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_model(train_loader, val_loader, num_epochs=10):
    model = MelanomaModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    mlflow.start_run()
    mlflow.pytorch.log_model(model, "model")
    # Create a mapping from label strings to numeric indices

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
           # Ensure images and labels are tensors
            if isinstance(images, (list, tuple)):
                images = torch.stack(images)
            # Ensure labels are tensors

            # Define a mapping from labels to numerical values
            label_map = {'benign': 0, 'malignant': 1}

            # Convert labels to numerical values using list comprehension
            labels_numerical = [label_map[label] for label in labels]

            # Convert list to tensor
            labels = torch.tensor(labels_numerical)

            #print("images", type(images))
            #print("labels", type(labels))
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        mlflow.log_metric("loss", running_loss/len(train_loader), step=epoch)

    mlflow.end_run()

    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Accuracy: {accuracy}")
    mlflow.log_metric("accuracy", accuracy)

    return accuracy
