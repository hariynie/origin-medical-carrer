#!/usr/bin/env python
# coding: utf-8

# In[ ]:


torch.save(model.state_dict(), "model_weights/hypothesis_1.pth")
torch.save(model.state_dict(), "model_weights/hypothesis_2.pth")
torch.save(model.state_dict(), "model_weights/hypothesis_final.pth")


# In[ ]:


import torch

model = LandmarkNet()
model.load_state_dict(torch.load("model_weights/hypothesis_10.pth")) 
model.eval()

