#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FetalUltrasoundDataset
from model import LandmarkNet
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 16

train_dataset = FetalUltrasoundDataset("data/images", "data/landmarks.csv")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = LandmarkNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


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
        
        # Save Model
        os.makedirs("model_weights", exist_ok=True)
        torch.save(model.state_dict(), f"model_weights/hypothesis_{epoch+1}.pth")

if __name__ == "__main__":
    train()

