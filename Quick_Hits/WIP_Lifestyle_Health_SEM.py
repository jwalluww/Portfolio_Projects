"""
Structural Equation Modeling (SEM) for Lifestyle and Health Data
---

📌 **Objective**:
- Model the impact of lifestyle choices (exercise diet smoking) on health indicators (BMI, blood pressure, cholesterol) which in tern affect heart disease.

🔍 **Key Takeaways**:
- blah

🔍 **Next Steps**: 
- blah

📌 **Methodology**:
1. Define latent variabels for lifestyle factors from observed variables
    - Lifestyle choices are latent - observed through exercise frequency, healthy eating score, smoking intensity
    - Healht indicators are latent - observed through BMI, blood pressure, cholesterol
    - Heart disease is observed (1=at risk, 0=not at risk)
2. Model indirect effects of lifesytle factors on heart disease through health indicators

✍ **Author**: Justin Wall
📅 **Date**: 03/03/2025
"""

# ==================================
# Import Libraries & Generate Data
# ==================================
#%%
import numpy as np
import pandas as pd
from semopy import Model

# Set random seed for reproducibility
np.random.seed(42)

# Sample size
n = 500  

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

# Check distribution of HeartDiseaseRisk
print(df["HeartDiseaseRisk"].value_counts(normalize=True))

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

#%%

# ==========================
# Define SEM Model
# ==========================
#%%
# Define the SEM model using lavaan-like syntax - don't use subtraction
sem_model = """
# Latent variables
Lifestyle =~ Exercise + HealthyEating + Smoking
HealthIndicators =~ BMI + BloodPressure + Cholesterol

# Relationships
HealthIndicators ~ Lifestyle
HeartDiseaseRisk ~ HealthIndicators + Lifestyle
"""

# Create and fit the model
model = Model(sem_model)
model.fit(df)

# Get parameter estimates
estimates = model.inspect("estimates")
print(estimates)
#%%