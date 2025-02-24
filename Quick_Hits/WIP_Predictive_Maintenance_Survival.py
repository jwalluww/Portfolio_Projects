"""
Predictive Machine Maintenance using Survival Analysis

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
# Create Fake Dataset for Predictive Maintenance
# =============================================
#%%
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of machines
n_machines = 500

# Generate synthetic data
machine_ids = np.arange(1, n_machines + 1)
operating_hours = np.random.randint(100, 5000, size=n_machines)  # Hours run
temperature = np.random.normal(75, 10, size=n_machines)  # Avg temperature (°C)
vibration = np.random.normal(3, 1, size=n_machines)  # Vibration levels (mm/s)
load = np.random.uniform(50, 100, size=n_machines)  # Load percentage (0-100%)

# Simulate failure events (1 = failed, 0 = still operational)
failure_probability = (temperature * 0.01) + (vibration * 0.05) + (load * 0.001)
failure_events = np.random.binomial(1, failure_probability / max(failure_probability))  # Normalize probability

# Time-to-failure (or last observed time for non-failures)
time_to_failure = operating_hours * (0.8 + 0.4 * (1 - failure_events) * np.random.rand(n_machines))

df = pd.DataFrame({
    'Machine_ID': machine_ids,
    'Operating_Hours': operating_hours,
    'Temperature': temperature,
    'Vibration': vibration,
    'Load': load,
    'Failure_Event': failure_events,
    'Time_to_Failure': time_to_failure
})

# Display sample data
df.head()

#%%