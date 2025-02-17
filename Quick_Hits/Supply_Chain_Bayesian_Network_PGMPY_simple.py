"""
Bayesian Network for Supply Chain Risk Mitigation using PGMPY

📌 **Objective**:
- Create a synthetic dataset for a retail pricing problem where we analyze the causal effects of a price change on customer demand
- Understand relationship between product price, advertising spend, competitor pricing, and customer demand

🔍 **Key Takeaways**:
- **BLAH**: 
- **Next Steps**: 
    - 

📌 **Methodology**:
Bayesian Networks (BNs)

Directed Acyclic Graph (DAG) where nodes represent variables, and edges represent causal or probabilistic dependencies.
Uses Conditional Probability Tables (CPTs) to define relationships.
Ideal for causal modeling, decision support, and prediction.
Use case fit: ✅ Best for supply chain risk estimation because disruptions often follow a causal chain (e.g., raw material shortages → production delays → supplier failure).
PGMPY also supports Markov Networks but follow undirect graphs that do not have causal relationship - not as good for supply chain risk estimation.

✍ **Author**: Justin Wall
📅 **Date**: 02/16/2025
"""

# =============================================
# Create Fake Dataset for Supply Chain Risk
# =============================================
#%%
import bnlearn as bn
import pandas as pd
import numpy as np
#%%

# =============================================
# Generate Dataset for Supply Chain Risk
# =============================================
#%%
# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic categorical data for supply chain risk
data = pd.DataFrame({
    "Supplier_Delay": np.random.choice(["Yes", "No"], size=1000, p=[0.2, 0.8]),
    "Inventory_Level": np.random.choice(["Low", "Medium", "High"], size=1000, p=[0.3, 0.5, 0.2]),
    "Production_Delay": np.random.choice(["Yes", "No"], size=1000, p=[0.25, 0.75]),
    "Demand_Surge": np.random.choice(["Yes", "No"], size=1000, p=[0.15, 0.85]),
    "Customer_Delay": np.random.choice(["Yes", "No"], size=1000, p=[0.3, 0.7])
})

# Show first 5 rows
print(data.head())
#%%

# =============================================
# Build the Model
# =============================================
#%%
# Learn the structure using bnlearn
model = bn.structure_learning.fit(data, methodtype='hc', scoretype='bic')

# Print the learned structure
print(model)
#%%

# =============================================
# Visualize the DAG
# =============================================
#%%
bn.plot(model)
#%%

# =
# Perform Model Inference
# = 
#%%
# Learn parameters
model = bn.parameter_learning.fit(model, data)

# Perform inference: What is the probability of Customer_Delay given Supplier_Delay = Yes?
query = bn.inference.fit(model, variables=["Customer_Delay"], evidence={"Supplier_Delay": "Yes"})
print(query)
#%%