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
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 5000

# Generate synthetic supply chain data
data = pd.DataFrame({
    "Supplier_Delay": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    "Inventory_Level": np.random.choice(["High", "Medium", "Low"], size=n_samples, p=[0.5, 0.3, 0.2]),
    "Production_Delay": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    "Demand_Surge": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
    "Logistics_Delay": np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12]),
    "Regulatory_Issues": np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
})

# Define the customer delay probability based on risk factors
def calculate_customer_delay(row):
    if row["Supplier_Delay"] == 1 or row["Production_Delay"] == 1 or row["Logistics_Delay"] == 1:
        return np.random.choice([0, 1], p=[0.4, 0.6])
    elif row["Inventory_Level"] == "Low" or row["Demand_Surge"] == 1 or row["Regulatory_Issues"] == 1:
        return np.random.choice([0, 1], p=[0.6, 0.4])
    else:
        return np.random.choice([0, 1], p=[0.9, 0.1])

# Apply function to create target variable
data["Customer_Delay"] = data.apply(calculate_customer_delay, axis=1)

# Encode categorical variable Inventory_Level
data["Inventory_Level"] = data["Inventory_Level"].map({"High": 0, "Medium": 1, "Low": 2})

# Display sample data
print(data.head())

# Supplier_Delay – Delay in receiving raw materials from suppliers.
# Inventory_Level – Stock availability (High, Medium, Low).
# Production_Delay – Manufacturing issues slowing down production.
# Demand_Surge – Sudden spike in demand (e.g., holiday season).
# Logistics_Delay – Issues in transportation and delivery.
# Regulatory_Issues – Delays due to compliance or customs.
# Customer_Delay – Whether the final product reaches the customer late.
#%%

# =============================================
# Define Bayesian Network Structure
# =============================================
#%%
# Define Bayesian Network Structure (DAG)
model = BayesianNetwork([
    ("Supplier_Delay", "Inventory_Level"),
    ("Inventory_Level", "Production_Delay"),
    ("Production_Delay", "Customer_Delay"),
    ("Demand_Surge", "Customer_Delay"),
    ("Logistics_Delay", "Customer_Delay"),
    ("Regulatory_Issues", "Customer_Delay")
])

# Display the structure
model.edges()
#%%

# =============================================
# Visualize the Bayesian Network
# =============================================
#%%
# Visualize the DAG
plt.figure(figsize=(8, 5))
G = nx.DiGraph()
G.add_edges_from(model.edges())
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, edge_color='gray')
plt.title("Bayesian Network - Supply Chain Risk Model")
plt.show()
#%%

# =============================================
# Run MLE Estimator to Fit the Model
# =============================================
#%%
# Estimate Conditional Probability Tables (CPTs) using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)
# Display learned CPDs (Conditional Probability Distributions)
for cpd in model.get_cpds():
    print(cpd)
#%%

# =============================================
# Run Inferences
# =============================================
#%%
# Initialize inference object
inference = VariableElimination(model)

# Test inference questions (examples)
print("P(Customer Delay | Supplier Delay = 1):")
print(inference.query(variables=["Customer_Delay"], evidence={"Supplier_Delay": 1}))

print("P(Customer Delay | Production Delay = 1, Demand Surge = 1):")
print(inference.query(variables=["Customer_Delay"], evidence={"Production_Delay": 1, "Demand_Surge": 1}))

print("P(Production Delay | Low Inventory Level):")
print(inference.query(variables=["Production_Delay"], evidence={"Inventory_Level": 2}))  # Low Inventory Level = 2
#-----

#%%