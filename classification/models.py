import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class MLP(nn.Module):
    def __init__(self, input_dim=204, hidden_dims=[512, 256, 128], num_classes=9, dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            #layers.append(nn.LayerNorm(h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=9, dropout_rate=0.25, use_softmax=False):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.res_block1 = self._residual_block(64, 64, dropout_rate)
        self.res_block2 = self._residual_block(64, 128, dropout_rate)
        self.res_block3 = self._residual_block(128, 256, dropout_rate)
        self.res_block4 = self._residual_block(256, 512, dropout_rate)

        # Pointwise convolutions to match dimensions for residual connections
        self.residual_conv2 = nn.Conv1d(64, 128, kernel_size=1)  
        self.residual_conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.residual_conv4 = nn.Conv1d(256, 512, kernel_size=1)  

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)  # Updated fully connected layer to match 512 output
        self.use_softmax = use_softmax

    def _residual_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Adding Dropout after ReLU activation
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 128)

        x = F.relu(self.bn1(self.conv1(x)))

        # Residual block 1 (input channels = 64)
        x_res = self.res_block1(x)
        x = x_res + x  # Residual connection

        # Residual block 2 (input channels = 128)
        x_res = self.res_block2(x)
        x = x_res + self.residual_conv2(x)  # Match dimensions using 1x1 conv

        # Residual block 3 (input channels = 256)
        x_res = self.res_block3(x)
        x = x_res + self.residual_conv3(x)  # Match dimensions using 1x1 conv

        # Residual block 4 (input channels = 512)
        x_res = self.res_block4(x)
        x = x_res + self.residual_conv4(x)  # Match dimensions using 1x1 conv

        # Global pooling and final fully connected layer
        x = self.global_pool(x).squeeze(-1)  # (batch, 512)
        x = self.fc(x)

        if self.use_softmax:
            x = F.softmax(x, dim=-1)
        return x


def add_gaussian_noise(embeddings, std=0.01):
    noise = np.random.normal(0, std, embeddings.shape)
    return embeddings + noise


class EmbeddingDataset(Dataset):
    def __init__(self, data, labels, neuron_ids=None, transform=None, train=False):
        self.data = data
        self.labels = labels
        self.neuron_ids = neuron_ids
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.neuron_ids == None:
            neuron_id = -1
        else:
            neuron_id = self.neuron_ids[idx]

        if self.train:
            sample = add_gaussian_noise(sample, std=0.05)
            sample = torch.tensor(sample, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)  # Apply transformation if provided

        return sample, label, neuron_id
