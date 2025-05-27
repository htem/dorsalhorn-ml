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
    if config["CREATE_NEW_DATASET"]: 
        if config["new_train_test_split"]:
            class_to_root_data, train_neurons, test_neurons = create_neuron_train_test_split(config["EMBEDDING_DIR"], 
                                                config["CLASS_NAMES"], config["LABEL_MAP"], 
                                                num_train_neurons=config["num_train_neurons"], 
                                                num_test_neurons=config["num_test_neurons"], 
                                                use_all_train_neurons=True, save_to_json=True)
        else:
            train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
            with open(train_test_split_file, 'r') as f:
                data = json.load(f)
                class_to_root_data = data['class_to_root_data']
                train_neurons = data['train_neurons']
                test_neurons = data['test_neurons']
            print(f'Loaded train_test_split file: {train_test_split_file}')

        train_data_files, test_data_files = convert_neurons_to_embedding_files(train_neurons, test_neurons, 
                                                                   class_to_root_data, use_all_synapses=True, 
                                                                   use_all_synapses_for_test=True)
    else:
        print("- Using previous dataset ...")
        train_test_split_file = os.path.join('datasets', 'train_test_split_neurons.json')
        with open(train_test_split_file, 'r') as f:
            data = json.load(f)
            class_to_root_data = data['class_to_root_data']
            train_neurons = data['train_neurons']
            test_neurons = data['test_neurons']
        print(f'- Loaded train_test_split file {train_test_split_file}')

        with open(config["TRAIN_EMBED_FILE_PATH"], 'rb') as f:
            train_data_files = pickle.load(f)
        with open(config["TEST_EMBED_FILE_PATH"], 'rb') as f:
            test_data_files = pickle.load(f)
        print(f"- Loaded training files {config['TRAIN_EMBED_FILE_PATH']}")
        print(f"- Loaded testing files {config['TEST_EMBED_FILE_PATH']}")

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
    # class_weights[6] = class_weights[6] * 2

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

    val_dataset = EmbeddingDataset(X_val, y_val, neuron_ids=test_data_files_flatten)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Training Parameters
    learning_rate = 1e-5
    weight_decay = 1e-4
    min_val_score = .70
    log_every = 100
    num_classes = len(class_counts)
    num_epochs = 4000

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
        for i, (X_batch, y_batch, neuron_id_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_loss = nn.CrossEntropyLoss(reduction='none')(outputs, y_batch)

            if epoch % 200 == 0:
                for j in range(len(X_batch)):
                    if batch_loss[j].item() > 500.0:
                        print(f"{neuron_id_batch[j]}, Loss: {batch_loss[j].item()}")

        model.eval()
        correct = 0
        total = 0
        incorrect_neurons = []
        with torch.no_grad():
            for i, (X_batch, y_batch, neuron_id_batch) in enumerate(val_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                # if epoch % 200 == 0:
                #     incorrect = (predicted != y_batch)
                #     incorrect_neurons.append([neuron_id_batch[i] for i in range(len(neuron_id_batch)) if incorrect[i]])
        # if epoch % 200 == 0 and epoch != 0:
        #     print('Incorrect neurons:', incorrect_neurons)
            
        val_score = correct/total
        if val_score > min_val_score:
            if val_score > best_val_score:
                best_val_score = val_score
                model_path = config["SAVE_MODEL_PATH"]
                torch.save(model.state_dict(), model_path)

        if epoch % log_every == 0 and epoch != 0:
            print(f"Epoch {epoch}:")
            print(f"- Train Loss: {running_loss / len(train_loader):.4f}")
            print(f"- Val Acc: {100 * correct / total:.2f}%")

            if best_val_score > 0:
                print(f"- Best Val Acc {best_val_score} saved to {model_path}", )
