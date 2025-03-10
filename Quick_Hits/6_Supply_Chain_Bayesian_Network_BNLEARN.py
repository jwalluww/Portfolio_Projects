"""
Bayesian Network for Supply Chain Risk Mitigation using BNLearn
---

🔍 **Situation**:
- Determine the causal relationships between various factors in a supply chain to estimate the risk of disruptions.
- Use Bayesian Networks to model the dependencies between supplier delays, inventory levels, production delays, demand surges, and customer delays.

📌 **Task**:
- Bayesian Networks (BNs)
- Directed Acyclic Graph (DAG) where nodes represent variables, and edges represent causal or probabilistic dependencies.
- Uses Conditional Probability Tables (CPTs) to define relationships.
- Ideal for causal modeling, decision support, and prediction.
- Use case fit: ✅ Best for supply chain risk estimation because disruptions often follow a causal chain (e.g., raw material shortages → production delays → supplier failure).
- PGMPY also supports Markov Networks but follow undirect graphs that do not have causal relationship - not as good for supply chain risk estimation.

✨ **Action**: 
- This library is best for fitting Bayesian Networks to data and performing causal inference. 

📈 **Result**:
- Compare BNLearn with PGMPY for Bayesian Network modeling on the same dataset.

✍ **Author**: Justin Wall
📅 **Updated**: 03/04/2025 
"""

# =============================================
# Create Fake Dataset for Supply Chain Risk
# =============================================
#%%
import bnlearn as bn
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Number of samples
n_samples = 1000

# Existing features
supplier_delay = np.random.choice(["Yes", "No"], size=n_samples, p=[0.2, 0.8])
inventory_level = np.where(supplier_delay == "Yes",
                           np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.5, 0.4, 0.1]),
                           np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.2, 0.5, 0.3]))

production_delay = np.where(np.isin(inventory_level, ["Low"]),
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6]),
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.1, 0.9]))

demand_surge = np.random.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85])

# New feature 1: Shipping Issues (random, but more likely with Supplier Delay)
shipping_issues = np.where(supplier_delay == "Yes",
                           np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6]),
                           np.random.choice(["Yes", "No"], size=n_samples, p=[0.1, 0.9]))

# New feature 2: Labor Shortages (random, but more likely when Inventory is Low)
labor_shortages = np.where(inventory_level == "Low",
                           np.random.choice(["Yes", "No"], size=n_samples, p=[0.5, 0.5]),
                           np.random.choice(["Yes", "No"], size=n_samples, p=[0.2, 0.8]))

# Updated Customer Delay (Now depends on Shipping Issues and Demand Surge too)
customer_delay = np.where((production_delay == "Yes") & (demand_surge == "Yes") & (shipping_issues == "Yes"),
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3]),
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7]))

# Create DataFrame
data = pd.DataFrame({
    "Supplier_Delay": supplier_delay,
    "Inventory_Level": inventory_level,
    "Production_Delay": production_delay,
    "Demand_Surge": demand_surge,
    "Shipping_Issues": shipping_issues,
    "Labor_Shortages": labor_shortages,
    "Customer_Delay": customer_delay
})

# Show first 5 rows
print(data.head())
#%%

# =============================================
# Create Bayesian Network Structure
# =============================================
#%%
# Learn the Bayesian Network structure
model = bn.structure_learning.fit(data, methodtype='hc', scoretype='bic')

# Print the discovered structure
print(model)
#%%

# =============================================
# Plot the model
# =============================================
#%%
bn.plot(model)
#%%

# =============================================
# Model Inferences
# =============================================
#%%
# Learn the probabilities (CPTs)
model = bn.parameter_learning.fit(model, data)

# Show learned parameters
bn.print_CPD(model)
#%%

# =============================================
# Causal Inferences
# =============================================
#%%
# Perform inference: What happens if we set Production_Delay = "No"?
query = bn.inference.fit(model, variables=["Inventory_Level"], evidence={"Production_Delay": "No"})
print(query)
#%%