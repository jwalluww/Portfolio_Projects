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
# ===================================================
# --- Step 1: Generate Synthetic Transaction Data ---
# ===================================================
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# ===================================================
# --- Step 2: Define the Variational Autoencoder (VAE) ---
# ===================================================
#%%
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        params = self.encoder(x)
        mu, log_var = params.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)  # Sampling
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
#%%

# ===================================================
# --- Step 3: Train the Variational Autoencoder (VAE) ---
# ===================================================
#%%
input_dim = real_data.shape[1]
hidden_dim = 16
latent_dim = 8

vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(20):  # Train for 20 epochs
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        x_recon, mu, log_var = vae(x)
        
        # VAE Loss = Reconstruction Loss + KL Divergence
        recon_loss = loss_fn(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
#%%

# ===================================================
# --- Step 4: Generate Synthetic Transactions ---
# ===================================================
#%%
with torch.no_grad():
    z_samples = torch.randn((5000, latent_dim))  # Generate 5,000 synthetic transactions
    synthetic_data = vae.decoder(z_samples).numpy()

# Step 5: Convert Generated Data to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=["Amount", "Time", "Merchant Type", "Transaction Type", "Fraud Label"])
print(synthetic_df.head())
#%%