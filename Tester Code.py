#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from model import LandmarkNet
from dataset import FetalUltrasoundDataset
from torch.utils.data import DataLoader


# In[ ]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkNet().to(DEVICE)
model.load_state_dict(torch.load("model_weights/hypothesis_final.pth"))
model.eval()


# In[ ]:


test_dataset = FetalUltrasoundDataset("data/test_images", "data/test_landmarks.csv")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# In[ ]:


with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        print(f"Predicted Landmarks: {outputs.cpu().numpy()}, Ground Truth: {targets.cpu().numpy()}")

