#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate
from albumentations.pytorch import ToTensorV2


# In[ ]:


MG_SIZE = 256  
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class FetalUltrasoundDataset(Dataset):
    def __init__(self, image_dir, landmarks_csv, transform=None):
        self.image_dir = Path(image_dir)
        self.landmarks = pd.read_csv(landmarks_csv)
        self.transform = transform


# In[ ]:


def __len__(self):
        return len(self.landmarks)


# In[ ]:


points = self.landmarks.iloc[idx, 1:].values.reshape(-1, 2).astype(np.float32)
        points = points / IMG_SIZE  # Normalize coordinates
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        return image.unsqueeze(0), torch.tensor(points, dtype=torch.float32)


# In[ ]:


transform = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=10, p=0.5),
    RandomBrightnessContrast(p=0.2),
    ToTensorV2()
])


# In[ ]:


train_dataset = FetalUltrasoundDataset("data/images", "data/landmarks.csv", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[ ]:


class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256)
        self.fc2 = nn.Linear(256, 8)


# In[ ]:


def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 4, 2)


# In[ ]:


model = LandmarkNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


# In[ ]:


def train():
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()

