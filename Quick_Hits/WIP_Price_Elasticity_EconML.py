"""
Price Elasticity of Demand - EconML DML Model

📌 **Objective**:
- 

🔍 **Key Takeaways**:
- 

📌 **Methodology**:
1. 

📊 **Interpretation**:
- 

✍ **Author**: Justin Wall  
📅 **Date**: 02/25/2025  
"""

# ==========================================
# Create synthetic dataset
# ==========================================
#%%
import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# Simulate observational data
n = 5000  # Number of data points

# Unobserved demand factor (e.g., product popularity)
demand = np.random.normal(50, 10, size=n)  

# Confounders
marketing = np.random.uniform(0, 100, size=n)  # Marketing spend
competitor_price = np.random.uniform(5, 20, size=n)  # Competitor's price

# Price is influenced by demand, marketing, and competitor pricing
price = 20 + 0.5 * demand + 0.2 * marketing - 0.3 * competitor_price + np.random.normal(0, 2, size=n)

# Sales depend on price, demand, and marketing but NOT directly on competitor price
sales = 500 - 2.5 * price + 1.5 * demand + 0.8 * marketing + np.random.normal(0, 10, size=n)

# Create DataFrame
df = pd.DataFrame({'sales': sales, 'price': price, 'marketing': marketing, 'competitor_price': competitor_price})

df.head()
#%%

# ==========================================
# Train Test Split & Standardize Features
# ==========================================
#%%
# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[['marketing', 'competitor_price']])
X_test = scaler.transform(test[['marketing', 'competitor_price']])

# Treatment (Price) and Outcome (Sales)
T_train, T_test = train['price'], test['price']
Y_train, Y_test = train['sales'], test['sales']
#%%

# ==========================================
# EconML Model
# ==========================================
#%%
# Define ML models
model_y = RandomForestRegressor(n_estimators=100, max_depth=5)  # Predict sales given controls
model_t = LassoCV()  # Predict price given controls

# DML Model
dml = LinearDML(model_y=model_y, model_t=model_t)

# Fit the model
dml.fit(Y_train, T_train, X=X_train)

# Get price elasticity estimate
price_elasticity = dml.effect(X_test)
print("Estimated Price Elasticity (Mean):", np.mean(price_elasticity))
#%%