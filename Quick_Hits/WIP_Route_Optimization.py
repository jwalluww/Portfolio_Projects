"""
Route Optimization

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
# Create Fake Dataset for Route Optimization
# =============================================
#%%
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of delivery locations
num_locations = 10

# Generate random (x, y) coordinates for warehouse and delivery locations
warehouse = (0, 0)  # Assume warehouse is at origin
locations = np.random.uniform(-50, 50, size=(num_locations, 2))

# Combine warehouse and locations into a DataFrame
df = pd.DataFrame(np.vstack([warehouse, locations]), columns=["x", "y"])
df.index.name = "Location_ID"
df["Type"] = ["Warehouse"] + ["Delivery"] * num_locations

print(df)
#%%