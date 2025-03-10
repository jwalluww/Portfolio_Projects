"""
Treatment Effects Analysis - Observational & Experimental Studies

📌 **Objective**:
- Estimate the impact of a treatment on purchases.
- Use **Propensity Score Matching (PSM)** to estimate **ATT**.
- Compare treatment and control groups while reducing bias.

🔍 **Key Takeaways**:
- **ATE**: The treatment had a lift of X% across all users.
- **ATT (Using PSM)**: The effect was higher for those who received the email, showing X% uplift.
- **PSM worked well** to create a balanced comparison group.
- **Next Steps**: 
    - Try Causal Forests to estimate **CATE** for personalized targeting.
    - Use **Inverse Probability Weighting (IPW)** for an alternative ATT estimation.

📌 **Methodology**:
1. **Compute Propensity Scores** (Logistic Regression).
2. **Perform Nearest Neighbor Matching** to find similar untreated users.
3. **Estimate ATT** using matched pairs.

✍ **Author**: Justin Wall
📅 **Date**: 02/13/2025
"""

# =====================================
# Import Libraries and Create Data
# =====================================
#%%
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
date_range = pd.date_range(start="2022-01-01", periods=730, freq="D")  # 2 years of daily data

# Simulate stock levels with seasonality and randomness
trend = np.linspace(100, 50, len(date_range))  # Downward trend over time
seasonality = 20 * np.sin(np.linspace(0, 12 * np.pi, len(date_range)))  # Yearly seasonality
random_noise = np.random.normal(scale=5, size=len(date_range))  # Random noise

# Stock level = base + trend + seasonality + noise
stock_levels = 200 + trend + seasonality + random_noise

# Generate demand fluctuations (influences stock level)
daily_demand = np.random.randint(5, 20, size=len(date_range))  # Random daily demand
replenishment = (np.random.rand(len(date_range)) < 0.1) * np.random.randint(50, 150, size=len(date_range))  # Occasional restocks

# Adjust stock based on demand and replenishment
for i in range(1, len(stock_levels)):
    stock_levels[i] = max(0, stock_levels[i - 1] - daily_demand[i] + replenishment[i])  # Ensure no negative stock

# Create DataFrame
df = pd.DataFrame({"date": date_range, "stock_level": stock_levels, "daily_demand": daily_demand, "replenishment": replenishment})

# Save to CSV (optional)
# df.to_csv("inventory_data.csv", index=False)

df.head()
#%%

# =====================================
# Visualize Time Series
# =====================================
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Plot stock levels over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df["stock_level"], label="Stock Level", color="blue")
plt.title("Inventory Stock Levels Over Time")
plt.xlabel("Date")
plt.ylabel("Stock Level")
plt.legend()
plt.grid()
plt.show()

# There is a slight upward trend
# There is definitely seasonality
# There is lots of volatility expecially towards the end
#%%

# =====================================
# Visualize Time Series
# =====================================
#%%
from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(df["stock_level"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Interpretation
if result[1] < 0.05:
    print("The data is stationary (reject H0).")
else:
    print("The data is non-stationary (fail to reject H0). Differencing needed.")

# For some reason the data is stationary
# If not stationary: df["stock_level"].diff().dropna()
#%%

# =====================================
# ARIMA
# =====================================
#%%
from statsmodels.tsa.arima.model import ARIMA

# Fit arima
# p = number of past values to use (autoregressive terms)
# d = number of differencing steps to make data stationary
# q = number of past forecast errors to useds

# Differencing if needed
# df["stock_diff"] = df["stock_level"].diff().dropna()

# Fit ARIMA model (p, d, q) - Example: (2, 1, 2)
model_arima = ARIMA(df["stock_level"], order=(2, 1, 2))
model_arima_fit = model_arima.fit()

# Forecast next 30 days
forecast_arima = model_arima_fit.forecast(steps=30)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["stock_level"], label="Actual Stock Level", color="blue")
plt.plot(pd.date_range(df.index[-1], periods=30, freq="D"), forecast_arima, label="ARIMA Forecast", color="red")
plt.title("ARIMA Forecast for Inventory Stock Levels")
plt.xlabel("Date")
plt.ylabel("Stock Level")
plt.legend()
plt.grid()
plt.show()
#%%

# =====================================
# SARIMA
# =====================================
#%%
from statsmodels.tsa.statespace.sarimax import SARIMAX

# P = Seasonal autoregressive terms
# D = Seasonal differencing
# Q = Seasonal moving average terms
# s = Season length (e.g., 12 for monthly, 7 for weekly)

# Fit SARIMA model (p, d, q) x (P, D, Q, s) - Example: (2,1,2)x(1,1,1,7) for weekly seasonality
model_sarima = SARIMAX(df["stock_level"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))
model_sarima_fit = model_sarima.fit()

# Forecast next 30 days
forecast_sarima = model_sarima_fit.forecast(steps=30)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["stock_level"], label="Actual Stock Level", color="blue")
plt.plot(pd.date_range(df.index[-1], periods=30, freq="D"), forecast_sarima, label="SARIMA Forecast", color="green")
plt.title("SARIMA Forecast for Inventory Stock Levels")
plt.xlabel("Date")
plt.ylabel("Stock Level")
plt.legend()
plt.grid()
plt.show()
#%%