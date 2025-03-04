"""
Variational Autoencoders to Perform Data Augmentation for Bank Fraud Detection
---

🔍 **Situation**:
- A bank needs a fraud detection model but cannot share real customer transactions with your consulting team due to customer data sensitivity.

📌 **Task**:
- Generate synthetic banking transactions that mimic real-world fraud patterns.
- Train a fraud detection model on the synthetic data and evaluate its performance.

✨ **Action**: 
- Use Pytorch to build a Variational Autoencoder (VAE) to generate synthetic banking transactions.

📈 **Result**:
- blah

✍ **Author**: Justin Wall
📅 **Updated**: 03/04/2025
"""

# --- Step 1: Load Libraries and Data ---
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Generate Synthetic Transaction Data
np.random.seed(42)
num_samples = 10000

# Features: Amount, Time, Merchant Type, Transaction Type, Fraud Label
real_data = np.hstack([
    np.random.normal(100, 50, (num_samples, 1)),  # Transaction Amount
    np.random.normal(500, 200, (num_samples, 1)),  # Time of transaction
    np.random.randint(1, 10, (num_samples, 1)),  # Merchant type
    np.random.randint(0, 2, (num_samples, 1)),  # Transaction type (0 = Online, 1 = POS)
    np.random.choice([0, 1], size=(num_samples, 1), p=[0.98, 0.02])  # Fraud Label
])

# Convert to PyTorch tensors
data_tensor = torch.tensor(real_data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)
#%%