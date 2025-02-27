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
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from econml.iv.dml import DMLIV

# Set seed for reproducibility
np.random.seed(42)

# Simulate observational data
n = 5000  # Number of data points

# Unobserved demand factor (e.g., product popularity)
demand = np.random.normal(50, 10, size=n)  

# Confounders
marketing = np.random.uniform(0, 100, size=n)  # Marketing spend
competitor_price = np.random.uniform(5, 20, size=n)  # Competitor's price

# Instrumental variable: Supplier cost (affects price but not sales)
supplier_cost = np.random.uniform(10, 50, size=n)

# Price is influenced by demand, marketing, competitor pricing, and supplier cost
price = 20 + 0.5 * demand + 0.2 * marketing - 0.3 * competitor_price + 0.4 * supplier_cost + np.random.normal(0, 2, size=n)

# Sales depend on price, demand, and marketing but NOT directly on competitor price or supplier cost
sales = 500 - 2.5 * price + 1.5 * demand + 0.8 * marketing + np.random.normal(0, 10, size=n)

# Create DataFrame
df = pd.DataFrame({'sales': sales, 'price': price, 'marketing': marketing, 'competitor_price': competitor_price, 'supplier_cost': supplier_cost})

# Convert to datetime
df['date'] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

# Extract time-based features
df['month'] = df['date'].dt.month  # Seasonal effects (e.g., December sales surge)
df['day_of_week'] = df['date'].dt.dayofweek  # Monday (0) to Sunday (6)
df['week_of_year'] = df['date'].dt.isocalendar().week  # Weekly trends

# Add a lagged sales variable (1-day lag)
df['sales_lag_1'] = df['sales'].shift(1)

# Add a rolling 7-day moving average for sales
df['sales_rolling_avg_7'] = df['sales'].rolling(7, min_periods=1).mean()

df = df.dropna() # Drop row with NaN value - first row from the lag_1
df = df.drop(columns=['date'])  # Drop date column

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
X_train = scaler.fit_transform(train.drop(columns=['sales', 'price', 'supplier_cost']))
X_test = scaler.transform(test.drop(columns=['sales', 'price', 'supplier_cost']))

# Treatment (Price) and Outcome (Sales)
Y_test, T_test, Z_test = test['sales'], test['price'], test['supplier_cost']
Y_train, T_train, Z_train = train['sales'], train['price'], train['supplier_cost']
#%%

# ==========================================
# Fit price & sales models
# ==========================================
#%%
# Define hyperparameter grids
rf_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

lasso_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

# Tune model_y (Random Forest)
rf = RandomForestRegressor(random_state=42)
rf_cv = GridSearchCV(rf, rf_grid, cv=3, scoring="neg_mean_squared_error")
rf_cv.fit(X_train, Y_train)

# Tune model_t (Lasso)
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, lasso_grid, cv=3, scoring="neg_mean_squared_error")
lasso_cv.fit(X_train, T_train)

# Best models
best_rf = rf_cv.best_estimator_
best_lasso = lasso_cv.best_estimator_
#%%

# ==========================================
# Fit DML Model
# ==========================================
#%%
# Use in DML
dml = LinearDML(model_y=best_rf, model_t=best_lasso, random_state=42)
dml.fit(Y_train, T_train, X=X_train)

# Get price elasticity estimate
price_elasticity = dml.effect(X_test)
print("Estimated Price Elasticity (Mean):", np.mean(price_elasticity))
# A price elasticity of 0.96 means that a 1% increase in price leads to a 0.96% increase in sales
#%%

# ==========================================
# Fit DML & IV Model
# ==========================================
#%%
iv_model = DMLIV(discrete_treatment=False,
                 discrete_instrument=False,
                 random_state=42)

iv_model.fit(Y=Y_train, T=T_train, Z=Z_train, X=X_train)

# Get price elasticity estimate
price_elasticity = iv_model.effect(X_test)
print("Estimated Price Elasticity (Mean):", np.mean(price_elasticity))
# A price elasticity of 0.96 means that a 1% increase in price leads to a 0.96% increase in sales
#%%

# ==========================================
# Graph out price change by sales
# ==========================================
#%%
# Simulating price changes
price_changes = np.linspace(-20, 20, 10)  # -20% to +20% price change
sales_impact = [(1 + e/100) * np.mean(Y_test) for e in price_changes * np.mean(price_elasticity)]

plt.plot(price_changes, sales_impact, marker='o', linestyle='-', color='b')
plt.axhline(np.mean(Y_test), linestyle="--", color="red", label="Current Sales")
plt.xlabel("Price Change (%)")
plt.ylabel("Predicted Sales")
plt.title("Price Elasticity Impact on Sales")
plt.legend()
plt.show()
#%%