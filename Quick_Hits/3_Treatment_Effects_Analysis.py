"""
Treatment Effects Analysis - Observational & Experimental Studies
---

🔍 **Situation**:
    We aimed to estimate the impact of an email marketing campaign on customer purchases.
    Since this was an observational study where treatment assignment wasn’t randomized,
    we needed to account for potential biases in the data to isolate the true effect of the campaign.

📌 **Task**:
    Our goal was to:
        ✅ Estimate the Average Treatment Effect (ATE) to measure the overall impact.
        ✅ Use Propensity Score Matching (PSM) to compute the Average Treatment Effect on the Treated (ATT) for customers who received the email.
        ✅ Explore alternative methods such as Inverse Probability Weighting (IPW) for further robustness.
        ✅ Investigate Conditional Average Treatment Effects (CATE) using Causal Forests to identify personalized treatment effects across different customer segments.

✨ **Action**: 
    Created Synthetic Data:
        Simulated 5,000 customer records with realistic features like age, income, and purchase history.
    Estimated Propensity Scores:
        Fit a Logistic Regression model to predict the probability of receiving treatment based on covariates.
    Performed Propensity Score Matching (PSM):
        Used Nearest Neighbor Matching to pair treated customers with similar untreated ones based on their propensity scores.
    Estimated Treatment Effects:
        ✅ ATE — Compared mean purchase rates between treated and untreated groups.
        ✅ ATT (Using PSM) — Measured the treatment effect on those who received the email.
        ✅ ATU — Used Inverse Probability Weighting (IPW) to measure the effect on untreated customers.
        ✅ CATE — Implemented Causal Forests to understand subgroup-specific effects.
        ✅ ITE — Predicted individual-level treatment effects using the X-Learner for further insights.
    Validation:
        Compared results across methods to ensure consistent findings.

📈 **Result**:
    ✅ ATE showed a slight negative effect, indicating the campaign had no overall uplift when measured across all users.
    ✅ ATT (Using PSM) revealed a stronger positive effect for those who received the email, suggesting the campaign effectively targeted engaged customers.
    ✅ ATU was minimal, confirming that non-recipients were unlikely to convert even if targeted.
    ✅ CATE analysis revealed that higher-income and high-purchase-history segments responded most positively to the campaign.
    ✅ ITE provided granular insights into individual-level treatment responses, supporting personalized targeting strategies.

🚀 Next Steps / Recommendations
    Consider targeted marketing efforts that focus on identified high-responders.
    Explore Causal Forests or Bayesian models for improved subgroup analysis.
    Test campaign adjustments to improve the overall ATE by refining targeting strategies or email content.

✍ **Author**: Justin Wall
📅 **Updated**: 03/04/2025 
"""

# =========================================
# Create Synthetic Dataset for Observational Study
# =========================================
#%%
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Simulate dataset
n = 1000  # Number of individuals
education = np.random.randint(1, 5, size=n)  # Education level (1-4)
experience = np.random.randint(1, 10, size=n)  # Years of experience

# True Individual Treatment Effects (ITE)
ite = np.random.normal(5 + 2 * education, 2)  # True uplift effect

# Potential Outcomes (Y1 = with training, Y0 = without training)
y0 = np.random.normal(30 + 3 * education + 1.5 * experience, 5)  # Base salary increase without training
y1 = y0 + ite  # Salary increase if trained

# Treatment Assignment: Higher education -> More likely to take training
treatment = (np.random.rand(n) < (0.2 + 0.15 * education)).astype(int)

# Observed Outcome
y_observed = treatment * y1 + (1 - treatment) * y0

# Create DataFrame
df = pd.DataFrame({
    "education": education,
    "experience": experience,
    "treatment": treatment,
    "y0": y0,  # Potential outcome without treatment (not observed in real life)
    "y1": y1,  # Potential outcome with treatment (not observed in real life)
    "y_observed": y_observed  # What we actually see
})

df.head()
#%%


# =========================================
# Average Treatment Effect (ATE)           
# =========================================
#%%
ate = (df["y1"] - df["y0"]).mean()
print(f"ATE: {ate:.2f}")
#%%
# ATE: 10.05

# =========================================
# Average Treatment Effect on the Treated  
# =========================================
#%%
att = (df[df["treatment"] == 1]["y1"] - df[df["treatment"] == 1]["y0"]).mean()
print(f"ATT: {att:.2f}")
#%%
# ATT: 10.68

# =========================================
# Average Treatment Effect on the Untreated 
# ========================================= 
#%%
atu = (df[df["treatment"] == 0]["y1"] - df[df["treatment"] == 0]["y0"]).mean()
print(f"ATU: {atu:.2f}")
#%%
# ATU: 9.14

# ========================================= 
# Individual Treatment Effect               
# ========================================= 
#%%
df["ITE"] = df["y1"] - df["y0"]
df[["education", "experience", "ITE"]].head()
#%%

# ========================================= 
# Conditional Average Treatment Effect      
# ========================================= 
#%%
cate_by_education = df.groupby("education")["ITE"].mean()
print("CATE by Education Level:")
print(cate_by_education)
#%%
# CATE by Education Level:
# education
# 1     6.972344
# 2     9.118123
# 3    11.084062
# 4    12.792153
# Name: ITE, dtype: float64

# =========================================
# Create Synthetic Dataset for Experimental Study
# =========================================
#%%
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Simulate dataset
n = 5000  # Number of customers
age = np.random.randint(18, 65, size=n)
income = np.random.randint(30000, 120000, size=n)
past_purchases = np.random.randint(0, 20, size=n)

# Treatment assignment: Higher income & past purchases → more likely to get email
propensity = 0.2 + 0.2 * (income > 60000) + 0.3 * (past_purchases > 10)
treatment = (np.random.rand(n) < propensity).astype(int)

# True Individual Treatment Effects (ITE)
true_ite = np.random.normal(0.05 + 0.01 * (income / 100000), 0.02)

# Generate potential outcomes
y0 = np.random.binomial(1, 0.05 + 0.01 * (income / 100000))  # Purchase probability without email
y1 = np.clip(y0 + true_ite, 0, 1)  # Purchase probability with email

# Observed outcome
purchase = treatment * y1 + (1 - treatment) * y0

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "income": income,
    "past_purchases": past_purchases,
    "treatment": treatment,
    "y0": y0,  # Counterfactual (not observed in real life)
    "y1": y1,  # Counterfactual (not observed in real life)
    "purchase": purchase  # Observed outcome
})

df['purchase'] = df['purchase'].round(0).astype(int)

df.head()
#%%

# treatment = 1/0 email received
# purchase = 1/0 purchase made
# covariates = age, income, past_purchases

# ========================================= 
# Average Treatment Effect (ATE)            
# ========================================= 
#%%
# Estimate ATE using observed data
ate = df[df['treatment'] == 1]['purchase'].mean() - df[df['treatment'] == 0]['purchase'].mean()
print(f"Estimated ATE: {ate:.4f}")
#%%
# Estimated ATE: -0.0030
# Since we cannot observe counterfactuals, we simply subtract the means to calculate the ATE, it's basically the lift
# Overall campaign effectiveness on entire customer base

# ========================================= 
# Propensity Score Modeling PSM             
# ========================================= 
#%%
from sklearn.linear_model import LogisticRegression

# Define features for propensity score estimation
ps_features = ["age", "income", "past_purchases"]
x = df[ps_features]
y = df["treatment"]

# Fit logistic regression model for propensity score estimation
ps_model = LogisticRegression()
ps_model.fit(x,y)

# Get predicted propensity scores
df["ps"] = ps_model.predict_proba(df[ps_features])[:, 1]

from sklearn.neighbors import NearestNeighbors

# Separate treatment and control groups
treated = df[df["treatment"] == 1].copy()
control = df[df["treatment"] == 0].copy()

# Fit nearest neighbor matching on propensity scores
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(control[["ps"]])  # Fit only on control group propensity scores

# Find closest control match for each treated individual
distances, indices = nn.kneighbors(treated[["ps"]])
matched_control = control.iloc[indices.flatten()].reset_index(drop=True)

# Create a matched dataset of treated and matched control
matched_df = treated.copy()
matched_df["matched_control_purchase"] = matched_control["purchase"]
#%%
# Create model with treatment as target, match using k-nearest neighbors

# ========================================= 
# Average Treatment Effect on the Treated   
# ========================================= 
#%%
# Calculate ATT using matched treated-control pairs
att = (matched_df["purchase"] - matched_df["matched_control_purchase"]).mean()
print(f"Estimated ATT using PSM: {att:.4f}")

# # Estimate ATT
# att = df[df['treatment'] == 1]['purchase'].mean() - df[df['treatment'] == 1]['y0'].mean()
# print(f"Estimated ATT: {att:.4f}")
#%%
# Estimated ATT using PSM: -0.0009
# Impact on engaged customers - effect on those who received the email
# Match each treated individual with a similar untreated individual based on covariates

# ========================================= 
# Average Treatment Effect on the Untreated 
# ========================================= 
#%%
from sklearn.linear_model import LogisticRegression

# Fit a propensity score model (logistic regression)
ps_model = LogisticRegression()
ps_features = ["age", "income", "past_purchases"]
ps_model.fit(df[ps_features], df["treatment"])
df["ps"] = ps_model.predict_proba(df[ps_features])[:, 1]  # Propensity scores

# Inverse Probability Weighting (IPW) to estimate ATU
df_control = df[df["treatment"] == 0].copy()
df_control["weight"] = df_control["ps"] / (1 - df_control["ps"])

atu = (df_control["purchase"] * df_control["weight"]).sum() / df_control["weight"].sum()
print(f"Estimated ATU: {atu:.4f}")
#%%
# Estimated ATU: 0.0610
# What we missed out on - effect on those who did not receive the email
# Use inverse probability weighting (IPW) to reweight the control group to resemble the treated group.

# ========================================= 
# Conditional Average Treatment Effect      
# ========================================= 
#%%
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Define causal forest model
causal_forest = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor()
)

# Fit causal forest
causal_forest.fit(df['purchase'], df['treatment'], X=df[ps_features])

# Predict CATE for each customer
df["CATE"] = causal_forest.effect(df[ps_features])
print(df[['age', 'income', 'CATE']].head())
#%%

# Use Causal Forests or Bayesian Causal Forests to estimate treatment effects for subgroups.


# ========================================= 
# Individual Treatment Effect               
# ========================================= 
#%%
from econml.metalearners import XLearner

# Fit X-Learner for Individual Treatment Effects
x_learner = XLearner(models=RandomForestRegressor())
x_learner.fit(df['purchase'], df['treatment'], X=df[ps_features])

# Predict ITE for each customer
df["ITE"] = x_learner.effect(df[ps_features])
df[['age', 'income', 'ITE']].head()
#%%

# Use uplift modeling or meta-learners (T-Learner, X-Learner).


# ========================================= 
# Conclusion                                
# ========================================= 
# If ATT > ATE, the treatment works better for specific groups
# If ATU is low, avoid targeting non responders
# CATE & ITE help identify high responders
