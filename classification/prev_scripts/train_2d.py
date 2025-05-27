import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.models.video as models
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import sys
import json
import numpy as np
import pickle
from utils import *
import utils_2d
import yaml
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

torch.cuda.empty_cache()

# read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

print("Using previous dataset ...")

train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
with open(train_test_split_file, 'r') as f:
    data = json.load(f)
    class_to_root_data = data['class_to_root_data']
    train_neurons = data['train_neurons']
    test_neurons = data['test_neurons']
print(f'Loaded train_test_split file: {train_test_split_file}')

with open(config["TRAIN_EMBED_FILE_PATH"], 'rb') as f:
    train_data_files = pickle.load(f)
with open(config["TEST_EMBED_FILE_PATH"], 'rb') as f:    
    test_data_files = pickle.load(f)    
print("Loaded.")

train_data, test_data, train_labels, test_labels = utils_2d.create_dataset_embeds(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"], config["EMBEDDING_DIR"])

# Define a custom dataset class
class EMSegDataset(Dataset):
    def __init__(self, data, labels, aug):
        self.data = data
        self.labels = labels
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vol = self.data[idx]
        vol = torch.tensor(vol/255., dtype=torch.float32)
        vol = self.aug(vol)
        label = torch.tensor(self.labels[idx])
        return vol, label

train_aug =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
])

val_aug = preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Create dataset and dataloader
train_dataset = EMSegDataset(train_data, train_labels, aug=train_aug)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_dataset = EMSegDataset(test_data, test_labels, aug=val_aug)
test_loader = DataLoader(test_dataset, batch_size=24)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
#model = models.r3d_18(pretrained=True)

model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#model.fc = nn.Linear(model.fc.in_features, 9)
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.fc.in_features, 9)
)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 1000

scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_total, running_correct = 0.0, 0.0

    for volumes, labels in train_loader:
        volumes, labels = volumes.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(volumes)
            loss = criterion(outputs, labels)

        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * running_correct / running_total:.2f}")


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for volumes, labels in test_loader:
            volumes, labels = volumes.to(device), labels.to(device)
            outputs = model(volumes)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%\n")


torch.save(model.state_dict(), 'resnet2d_sn_classifier.pth')
print('Saved model')
