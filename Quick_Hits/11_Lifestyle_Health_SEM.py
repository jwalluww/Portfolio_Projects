"""
Structural Equation Modeling (SEM) for Lifestyle and Health Data
---

🔍 **Situation**:
We wanted to understand how lifestyle choices such as exercise, diet, and smoking influence health indicators like BMI, blood pressure, and cholesterol — and how these factors collectively impact heart disease risk.
The goal was to assess both direct and indirect effects of lifestyle on heart disease risk to uncover deeper relationships beyond simple correlations.

📌 **Task**:
We aimed to build a Structural Equation Model (SEM) to:
- Identify key latent variables that group related behaviors (e.g., "Lifestyle" and "Health Indicators").
- Model the indirect effects of lifestyle factors on heart disease through health indicators.
- Quantify the relative impact of different variables to inform practical insights for healthcare interventions.

✨ **Action**: 
Data Generation:
- Created a synthetic dataset with 500,000 observations.
- Balanced the HeartDiseaseRisk variable using the median risk score to ensure variability between at-risk and not-at-risk groups.
- Scaled the data to improve model convergence.
Model Definition:
- Defined two latent variables:
  - Lifestyle (Exercise + HealthyEating + Smoking)
  - Health Indicators (BMI + BloodPressure + Cholesterol)
- Specified the SEM structure:
  - HealthIndicators ~ Lifestyle (Indirect effect)
  - HeartDiseaseRisk ~ HealthIndicators + Lifestyle (Direct and indirect effects)
Model Fitting & Evaluation:
- Used semopy with the Maximum Likelihood Weighted (MLW) objective function and SLSQP optimizer to improve convergence.
- Diagnosed model issues with modification indices and gradient checks.

📈 **Result**:
Key Findings from Model Estimates:
- BMI had a strong relationship with Health Indicators (Coefficient = 1.000)
- Blood Pressure showed a meaningful effect on Health Indicators (Coefficient = 2.528, p < 0.001)
- Cholesterol had the largest effect on Health Indicators (Coefficient = 9.070, p < 0.001)
- The total effect of Health Indicators on HeartDiseaseRisk was modest (Coefficient = 0.103, p < 0.001)
Insights for Decision-Making:
- Cholesterol emerged as the most influential health indicator, making it a prime target for interventions.
- Lifestyle changes (exercise, healthy eating, quitting smoking) had their strongest impact indirectly by improving BMI, blood pressure, and cholesterol.
- To reduce heart disease risk, focusing on reducing cholesterol levels appears to have the highest impact.

🚀 Next Steps / Additional Analysis
- Consider introducing interaction terms (e.g., Exercise × Smoking) to capture complex lifestyle behaviors.
- Test alternative models to explore non-linear relationships or feedback loops.
- Perform a Causal DAG analysis to validate causal effects and explore intervention scenarios like smoking cessation or dietary improvements.

✍ **Author**: Justin Wall
📅 **Updated**: 03/04/2025
"""

# ==================================
# Import Libraries & Generate Data
# ==================================
#%%
import numpy as np
import pandas as pd
import semopy as sem
from semopy import Model
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Sample size
n = 500000

# Lifestyle factors
exercise_freq = np.random.randint(1, 6, n)  # 1 (never) to 5 (daily)
healthy_eating = np.random.randint(1, 6, n)  # 1 (poor diet) to 5 (excellent diet)
smoking_intensity = np.random.randint(0, 4, n)  # 0 (heavy smoker) to 3 (non-smoker)

# Health indicators (dependent on lifestyle)
BMI = 25 - (exercise_freq * 0.6) - (smoking_intensity * 1.0) + np.random.normal(0, 2, n)
blood_pressure = 120 - (exercise_freq * 1.5) - (smoking_intensity * 2.0) + np.random.normal(0, 7, n)
cholesterol = 200 - (exercise_freq * 4) + (healthy_eating * 2) - (smoking_intensity * 3) + np.random.normal(0, 12, n)

# Adjust heart disease risk calculation to balance 0s and 1s
risk_score = BMI * 0.03 + blood_pressure * 0.015 + cholesterol * 0.008 + np.random.normal(0, 1, n)
threshold = np.median(risk_score)  # Use the median to balance the distribution
heart_disease_risk = (risk_score > threshold).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Exercise": exercise_freq,
    "HealthyEating": healthy_eating,
    "Smoking": smoking_intensity,
    "BMI": BMI,
    "BloodPressure": blood_pressure,
    "Cholesterol": cholesterol,
    "HeartDiseaseRisk": heart_disease_risk
})

df["HeartDiseaseRisk"] = df["HeartDiseaseRisk"].astype(float)

# Check distribution of HeartDiseaseRisk
# print(df["HeartDiseaseRisk"].value_counts(normalize=True))

df.head()

# 500 Observations
# Lifestyle Factors:
# Exercise (1-5 scale)
# HealthyEating (1-5 scale)
# Smoking (0-3 scale)

# Health Indicators:
# BMI (Body Mass Index)
# BloodPressure (Systolic BP)
# Cholesterol (Cholesterol level in mg/dL)

# Outcome:
# HeartDiseaseRisk (Binary: 1 = At risk, 0 = Not at risk)

# print(df.nunique())  # Check unique values per column
# print(df.isnull().sum())  # Check for missing values
#%%

# ==========================
# Define SEM Model
# ==========================
#%%
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[["Exercise", "HealthyEating", "Smoking", "BMI", "BloodPressure", "Cholesterol"]] = scaler.fit_transform(
    df[["Exercise", "HealthyEating", "Smoking", "BMI", "BloodPressure", "Cholesterol"]]
)

# Define the SEM model using lavaan-like syntax - don't use subtraction
sem_model = """
# Latent variables
Lifestyle =~ Exercise + HealthyEating + Smoking
HealthIndicators =~ BMI + BloodPressure + Cholesterol

# Relationships
HealthIndicators ~ Lifestyle
HeartDiseaseRisk ~ HealthIndicators + Lifestyle
"""

sem_model_simpler = """
HealthIndicators =~ BMI + BloodPressure + Cholesterol
HeartDiseaseRisk ~ HealthIndicators
"""


# Create and fit the model
# model = Model(sem_model)
model = Model(sem_model_simpler)
# model.fit(df)
# model.fit(df_scaled)
model.fit(df, obj="MLW", solver="SLSQP")
#%%

# ==========================
# Evaluate SEM Model
# ==========================
#%%

# # Get model summary
# model.inspect(mode='list', what="names", std_est=True)

# # Get model fit statistics
# sem.calc_stats(model)

# Plot the model
g = sem.semplot(model, "model.png", show=False)
g.view()

# Get parameter estimates
# estimates = model.inspect("estimates")
# print(estimates)

# # Model Diagnostics
# print(model.inspect("modindices"))  # Modification indices
# print(model.inspect("gradient"))  # Gradient to check optimization issue
# print(model.inspect("fit"))

# A lot of these model.inspect aspects are returning None
# Using teh model.png, from healthindicators node...
# - BMI: 1.000
# - Blood Pressure: 2.528, pval 0.00
# - Cholesterol: 9.070, pval 0.00
# - HeartDiseaseRisk: 0.103, pval 0.00

# When to use each
# Approach: Scenario
# Structural Equation Model (SEM): You want to model direct and indirect effects of exercise, diet, and smoking on heart disease using latent variables like "Lifestyle" and "Health Indicators".
# Bayesian Network (BN): You have incomplete data and want to infer missing values and calculate the probability of heart disease given certain conditions (e.g., P(HeartDisease
# Causal DAG (DoWhy/PyMC): You want to perform causal inference using do-calculus, e.g., “If we force someone to quit smoking (intervention), how much will their heart disease risk decrease?”
#%%