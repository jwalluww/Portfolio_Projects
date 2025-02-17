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

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate supplier delays (10% chance of delay)
supplier_delay = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Inventory level depends on supplier delay
inventory_level = np.array([
    np.random.choice(["High", "Medium", "Low"], p=[0.7, 0.2, 0.1]) if s == 0 else
    np.random.choice(["High", "Medium", "Low"], p=[0.2, 0.3, 0.5])
    for s in supplier_delay
])

# Production delay depends on inventory level
production_delay = np.array([
    np.random.choice([0, 1], p=[0.95, 0.05]) if inv == "High" else
    np.random.choice([0, 1], p=[0.85, 0.15]) if inv == "Medium" else
    np.random.choice([0, 1], p=[0.5, 0.5])
    for inv in inventory_level
])

# Demand surge (independent, 15% chance of surge)
demand_surge = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

# Customer delivery delay depends on production delay and demand surge
customer_delay = np.array([
    np.random.choice([0, 1], p=[0.9, 0.1]) if (p == 0 and d == 0) else
    np.random.choice([0, 1], p=[0.7, 0.3]) if (p == 1 and d == 0) else
    np.random.choice([0, 1], p=[0.6, 0.4]) if (p == 0 and d == 1) else
    np.random.choice([0, 1], p=[0.4, 0.6])
    for p, d in zip(production_delay, demand_surge)
])

# Create DataFrame
supply_chain_data = pd.DataFrame({
    "Supplier_Delay": supplier_delay,
    "Inventory_Level": inventory_level,
    "Production_Delay": production_delay,
    "Demand_Surge": demand_surge,
    "Customer_Delay": customer_delay
})

# Display first few rows
supply_chain_data.head()

# Supplier_Delay (Binary: 0 = No Delay, 1 = Delay)
# Inventory_Level (Categorical: High, Medium, Low)
# Production_Delay (Binary: 0 = No Delay, 1 = Delay)
# Demand_Surge (Binary: 0 = Normal, 1 = Surge)
# Customer_Delay (Binary: 0 = On-time, 1 = Delayed)
#%%

# =============================================
# Define Bayesian Network Structure
# =============================================
#%%
from pgmpy.models import BayesianNetwork

# Define the structure of the Bayesian Network
supply_chain_model = BayesianNetwork([
    ("Supplier_Delay", "Inventory_Level"),
    ("Inventory_Level", "Production_Delay"),
    ("Production_Delay", "Customer_Delay"),
    ("Demand_Surge", "Customer_Delay"),
    ("Inventory_Level", "Customer_Delay")
])

# Display the structure
supply_chain_model.edges()
#%%

# =============================================
# Visualize the Bayesian Network
# =============================================
#%%
import matplotlib.pyplot as plt
import networkx as nx

# Create a graph from the Bayesian Network structure
plt.figure(figsize=(6, 4))
G = nx.DiGraph()
G.add_edges_from(supply_chain_model.edges())

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # Positioning for nodes
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")

# Show plot
plt.title("Bayesian Network Structure - Supply Chain")
plt.show()
#%%

# =============================================
# Run MLE Estimator to Fit the Model
# =============================================
#%%
from pgmpy.estimators import MaximumLikelihoodEstimator

# Learn the CPTs from data using Maximum Likelihood Estimation (MLE)
supply_chain_model.fit(supply_chain_data, estimator=MaximumLikelihoodEstimator)

# Display learned CPDs (Conditional Probability Distributions)
for cpd in supply_chain_model.get_cpds():
    print(cpd)

# Output shows Conditional Probability Distributions (CPDs) which tell us how the nodes depend on their parent nodes
# Key Insights from the CPTs
# Supplier Delay:

# 90% chance of no delay (Supplier_Delay = 0).
# 10% chance of a delay (Supplier_Delay = 1).
# This suggests that supplier reliability is high but not perfect.
# Inventory Levels (given Supplier Delay):

# If no Supplier Delay (Supplier_Delay = 0):
# High inventory: ~70%
# Medium inventory: ~20%
# Low inventory: ~9.7%
# If Supplier Delay occurs (Supplier_Delay = 1):
# High inventory drops significantly to 21%.
# Low inventory increases to 62%.
# Interpretation: Supplier delays reduce high inventory and increase low inventory.
# Production Delay (given Inventory Level):

# High inventory → Very low chance of production delay (~98%)
# Low inventory → Very high chance of production delay
# Interpretation: Production is heavily dependent on inventory levels.
# Customer Delay (given Demand Surge, Inventory, and Production Delay):

# Balanced 50%-50% probability in some cases, indicating a high uncertainty in final delays due to multiple interacting factors.
# If Production Delay = 1, Customer Delay increases significantly.
# If Demand Surge = 1, Customer Delay is more likely.
# Interpretation: Customer delays are primarily driven by inventory, production, and demand surges.
# Demand Surge:

# 87% chance of no demand surge.
# 13% chance of a demand surge.
# Interpretation: Most of the time, demand remains stable, but there’s a small chance of a spike.
#%%

# =
# Run Inferences
# = 
#%%
# Predictive Analysis:
# "Given a supplier delay, what’s the probability of a customer delay?"
# "If we increase inventory, how much does that reduce customer delays?"

# Diagnostic Analysis:
# "If a customer delay occurred, what was the most likely cause?"

#-----
# Given a supplier delay, what's the probability of a customer delay?
from pgmpy.inference import VariableElimination

# Create an inference object
inference = VariableElimination(supply_chain_model)

# Query: Given Supplier_Delay = 1, what's the probability of Customer_Delay?
query_result = inference.query(variables=["Customer_Delay"], evidence={"Supplier_Delay": 1})

# Print result
print(query_result)

# The probability of a customer delay given a supplier delay is 18.49%
#-----

#-----
# If we increase inventory, how much does that reduce customer delays?
query_result = inference.query(variables=["Customer_Delay"], evidence={"Inventory_Level": "High"})
print(query_result)

# If we increase inventory, we reduce customer delays from 86.56% to 13.44%
#-----

#-----
# If a customer delay occurred, what was the most likely cause?
query_result = inference.query(variables=["Supplier_Delay"], evidence={"Customer_Delay": 1})
print(query_result)

# Given that a customer delay has already happened, there is only a 12.28% chance that a supplier delay was the cause.
#-----

#%%