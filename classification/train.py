import caveclient
import cloudvolume
import os
import json
import numpy as np
import pickle
from utils import *
import random
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import EmbeddingDataset, MLP
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import yaml

""" Use this script to train a MLP classifier for cell type prediction. 
Assumes embeddings are already created """

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fix seed for reproducability
set_seed(42)

# Read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Create util folders
for folder in ["datasets", "models"]:
    os.makedirs(folder, exist_ok=True)

# Caveclient
client = caveclient.CAVEclient("brain_and_nerve_cord")

# Segmentation cloud volume
proofread_seg_spinalcord = cloudvolume.CloudVolume("graphene://https://cave.fanc-fly.com/segmentation/table/wclee_mouse_spinalcord_cltmr", use_https=True, mip=(32,32,45), agglomerate=True, progress=False)


if __name__ == "__main__":
    ############################ Set up training testing dataset #################################
    print("Setting up training and testing data")
    with open('datasets/train_test_data_split_040625.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_data_files = dataset['train']
    test_data_files = dataset['test']

    print('\n')

    print("Dataset breakdown") # Visualize data set being used
    visualize_dataset(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"], only_overall=False)
    print("\n")

    train_embed, test_embed, train_files_per_class, test_files_per_class = create_dataset_embeds(train_data_files, test_data_files, os.path.dirname(config["EMBEDDING_DIR"]), config["LABEL_MAP"],
                                                   config["MODELS2USE"], config["EMBEDDING_DIR"], config["AGGREGATE_RADIUS_UM"], use_meshdata=False)
    print("\n")

    train_data_files_flatten = [file for key in sorted(list(config["LABEL_MAP"].keys())) for file in sorted(train_data_files[key])]
    test_data_files_flatten = [file for key in sorted(list(config["LABEL_MAP"].keys())) for file in sorted(test_data_files[key])]
    # validate_train_test_split(seg_vol=proofread_seg_spinalcord, client=client, train_data_files=train_data_files_flatten, 
    #     test_data_files=test_data_files_flatten)

    class_counts = torch.tensor([0 for _ in range(len(set(list(config["LABEL_MAP"].values()))))])
    for classname, classindex in config["LABEL_MAP"].items():
        class_counts[int(classindex)] += len(train_embed[classname])

    # Compute inverse frequency weights
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize
    #class_weights[5] = class_weights[5] * 2
    #class_weights[6] = class_weights[6] * 2

    # Gather all embeddings and create labels
    all_train_embeddings, train_labels = setup_embeds_and_labels(train_embed, config["LABEL_MAP"])
    all_test_embeddings, test_labels = setup_embeds_and_labels(test_embed, config["LABEL_MAP"])

    print("Training shape:")
    print("- embedding:", all_train_embeddings.shape)
    print("- label:", train_labels.shape)
    print("Testing shape:")
    print("- embedding:", all_test_embeddings.shape)
    print("- label:", test_labels.shape)
    print('Class Weights:', class_weights)

    # train_mesh_features = all_train_embeddings[:, -12:]
    # mesh_mean = train_mesh_features.mean(axis=0)
    # mesh_std = train_mesh_features.std(axis=0)
    # all_train_embeddings[:, -12:] = (train_mesh_features - mesh_mean) / (mesh_std + 1e-8)
    # test_mesh_features = all_test_embeddings[:, -12:]
    # all_test_embeddings[:, -12:] = (test_mesh_features - mesh_mean) / (mesh_std + 1e-8)

    #all_train_embeddings = all_train_embeddings[:,-12:]
    #all_test_embeddings = all_test_embeddings[:,-12:]

    # Create dataset
    X_train = torch.tensor(all_train_embeddings, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    X_val = torch.tensor(all_test_embeddings, dtype=torch.float32)
    y_val = torch.tensor(test_labels, dtype=torch.long)

    # Convert to PyTorch DataLoader
    train_dataset = EmbeddingDataset(X_train, y_train, neuron_ids=train_data_files_flatten, train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = EmbeddingDataset(X_val, y_val, neuron_ids=test_data_files_flatten, train=False)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Training Parameters
    learning_rate = 1e-4
    weight_decay = 1e-4
    min_val_score = .70
    log_every = 100
    num_classes = len(class_counts)
    num_epochs = 10000

    print("Number of classes:", num_classes)
    print("Number of embeddings per datapoint:", X_train.shape[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    model = MLP(input_dim=X_train.shape[-1], num_classes=num_classes)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_score = 0
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        for i, (X_batch, y_batch, neuron_id_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            _,predicted = torch.max(outputs,1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == y_batch).sum().item()
            running_total += y_batch.size(0)

        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0
        with torch.no_grad():
            for i, (X_batch, y_batch, neuron_id_batch) in enumerate(val_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                val_loss = nn.CrossEntropyLoss()(outputs, y_batch)
                val_running_loss += val_loss.item()

        val_score = correct/total
        if val_score > min_val_score:
            if val_score > best_val_score:
                best_val_score = val_score
                model_path = config["SAVE_MODEL_PATH"]
                torch.save(model.state_dict(), model_path)

        if epoch % log_every == 0 and epoch != 0:
            print(f"#### Epoch {epoch} ####")
            print(f"- Train Acc: {100 * running_correct / running_total:.2f}%")
            print(f"- Train Loss: {running_loss / len(train_loader):.4f}")
            print(f"- Val Acc: {100 * correct / total:.2f}%")
            print(f"- Val Loss: {val_running_loss / len(val_loader):.4f}")

            if best_val_score > 0:
                print(f"- Best Val Acc {best_val_score} saved to {model_path}", )
