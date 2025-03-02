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
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import seaborn as sns

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

# =============================================
# Visualize Survival Curves using Kaplan-Meier
# =============================================
#%%
# Kaplan-Meier Estimator
kmf = KaplanMeierFitter()
kmf.fit(df['Time_to_Failure'], event_observed=df['Failure_Event'])

# Plot survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve for Machine Failures')
plt.xlabel('Time (Operating Hours)')
plt.ylabel('Survival Probability')
plt.grid()
plt.show()
#%%

# =============================================
# Compare Survival Curves by Machine Vibration
# =============================================
#%%
# Define high vs low vibration groups based on median
median_vibration = df['Vibration'].median()
df['Vibration_Group'] = np.where(df['Vibration'] >= median_vibration, 'High Vibration', 'Low Vibration')

# Kaplan-Meier Estimator
kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

# Fit survival curves
high_vibration = df[df['Vibration_Group'] == 'High Vibration']
low_vibration = df[df['Vibration_Group'] == 'Low Vibration']

kmf_high.fit(high_vibration['Time_to_Failure'], event_observed=high_vibration['Failure_Event'], label='High Vibration')
kmf_low.fit(low_vibration['Time_to_Failure'], event_observed=low_vibration['Failure_Event'], label='Low Vibration')

# Plot survival curves
plt.figure(figsize=(10, 6))
kmf_high.plot_survival_function()
kmf_low.plot_survival_function()
plt.title('Kaplan-Meier Survival Curves by Vibration Levels')
plt.xlabel('Time (Operating Hours)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid()
plt.show()

# Log-rank test
logrank_result = logrank_test(
    high_vibration['Time_to_Failure'], low_vibration['Time_to_Failure'],
    event_observed_A=high_vibration['Failure_Event'], event_observed_B=low_vibration['Failure_Event']
)

print(f"Log-rank Test p-value: {logrank_result.p_value}")
# There is a significant difference in survival curves between high and low vibration groups
#%%

# =============================================
# Predictive Maintenance using Cox Proportional Hazards Model
# =============================================
#%%
# Cox Proportional Hazards Model
cph = CoxPHFitter()
cph_data = df[['Time_to_Failure', 'Failure_Event', 'Temperature', 'Vibration', 'Load']]
cph.fit(cph_data, duration_col='Time_to_Failure', event_col='Failure_Event')

# Display model summary
cph.print_summary()

# Plot hazard ratios
plt.figure(figsize=(8, 6))
cph.plot()
plt.title("Cox Proportional Hazards Model - Hazard Ratios")
plt.show()

# 0.55 concordance indicates poor model fit
# Temperature and Load have significant p-values, while Vibration does not
#%%

# =============================================
# Check proportional hazards assumption
# =============================================
#%%
data = df.drop(columns=['Vibration_Group'])
# Check proportional hazards assumption using Schoenfeld residuals
results = cph.check_assumptions(data, p_value_threshold=0.05, show_plots=True)
#%%

# =============================================
# Visualize Hazard Ratios
# =============================================
#%%
hazard_ratios = np.exp(cph.params_)
plt.figure(figsize=(8, 5))
sns.barplot(x=hazard_ratios.index, y=hazard_ratios.values, palette="coolwarm")
plt.ylabel("Hazard Ratio (exp(coef))")
plt.title("Hazard Ratios from Cox Model")
plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.show()
#%%

# =============================================
# Strategy
# =============================================
#%%
# from lifelines import CoxPHFitter

# cph = CoxPHFitter()
# cph.fit(df, duration_col='Time_to_Failure', event_col='Failure_Event')

# Simulate a time period for analysis (e.g., 365 days)
time_horizon = 365

# Predict survival probabilities for each machine
predicted_survival = cph.predict_survival_function(df, times=np.arange(1, time_horizon+1))

# Define cost parameters
cost_failure_repair = 10000  # Cost to repair after failure
cost_scheduled_maintenance = 3000  # Preventive maintenance cost
cost_downtime_per_day = 1000  # Unplanned downtime cost

# Initialize cost trackers
costs = {"Reactive": 0, "Preventive": 0, "Condition-Based": 0}
failures = {"Reactive": 0, "Preventive": 0, "Condition-Based": 0}

# Simulating maintenance strategies
for i in range(len(df)):  # Loop over machines
    survival_curve = predicted_survival.iloc[:, i]

    # Reactive Maintenance: Count failures
    failure_time = (survival_curve < 0.05).idxmax() if (survival_curve < 0.05).any() else time_horizon
    costs["Reactive"] += cost_failure_repair + (cost_downtime_per_day * failure_time)
    failures["Reactive"] += 1

    # Time-Based Preventive Maintenance: Every 50 days
    num_maintenances = time_horizon // 50
    costs["Preventive"] += num_maintenances * cost_scheduled_maintenance

    # Condition-Based Maintenance: When failure probability > 50% in next 10 days
    for day in range(10, time_horizon, 10):
        if survival_curve[day] < 0.5:  # Maintenance if risk is too high
            costs["Condition-Based"] += cost_scheduled_maintenance
            break  # Stop scheduling if already maintained

# Display cost comparison
cost_summary = pd.DataFrame.from_dict(costs, orient='index', columns=["Total Cost"])
cost_summary["Failures"] = pd.Series(failures)
cost_summary
#%%

# New variable
# non parametric model Weibull or something
# Random Survival Forest

