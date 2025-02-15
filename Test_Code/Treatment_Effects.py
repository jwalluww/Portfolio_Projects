"""
Email Campaign Uplift Model - Causal Inference Study

📌 **Objective**:
- Estimate the impact of an email marketing campaign on purchases.
- Use **Propensity Score Matching (PSM)** to estimate **ATT**.
- Compare treatment and control groups while reducing bias.

🔍 **Key Takeaways**:
- **ATE**: The email had a lift of X% across all users.
- **ATT (Using PSM)**: The effect was higher for those who received the email, showing X% uplift.
- **PSM worked well** to create a balanced comparison group.
- **Next Steps**: 
    - Try Causal Forests to estimate **CATE** for personalized targeting.
    - Use **Inverse Probability Weighting (IPW)** for an alternative ATT estimation.

📌 **Methodology**:
1. **Compute Propensity Scores** (Logistic Regression).
2. **Perform Nearest Neighbor Matching** to find similar untreated users.
3. **Estimate ATT** using matched pairs.

✍ **Author**: Justin Wall
📅 **Date**: 02/13/2025
"""

# ========================================= #
# Treatment Effects Analysis - Observation  #
# ========================================= #
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


# ========================================= #
# Average Treatment Effect (ATE)            #
# ========================================= #
#%%
ate = (df["y1"] - df["y0"]).mean()
print(f"ATE: {ate:.2f}")
#%%

# ========================================= #
# Average Treatment Effect on the Treated   #
# ========================================= #
#%%
att = (df[df["treatment"] == 1]["y1"] - df[df["treatment"] == 1]["y0"]).mean()
print(f"ATT: {att:.2f}")
#%%

# ========================================= #
# Average Treatment Effect on the Untreated #
# ========================================= #
#%%
atu = (df[df["treatment"] == 0]["y1"] - df[df["treatment"] == 0]["y0"]).mean()
print(f"ATU: {atu:.2f}")
#%%

# ========================================= #
# Individual Treatment Effect               #
# ========================================= #
#%%
df["ITE"] = df["y1"] - df["y0"]
df[["education", "experience", "ITE"]].head()
#%%

# ========================================= #
# Conditional Average Treatment Effect      #
# ========================================= #
#%%
cate_by_education = df.groupby("education")["ITE"].mean()
print("CATE by Education Level:")
print(cate_by_education)
#%%





# ========================================= #
# Treatment Effects Analysis - Experimental #
# ========================================= #
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

# ========================================= #
# Average Treatment Effect (ATE)            #
# ========================================= #
#%%
# Estimate ATE using observed data
ate = df[df['treatment'] == 1]['purchase'].mean() - df[df['treatment'] == 0]['purchase'].mean()
print(f"Estimated ATE: {ate:.4f}")
#%%

# Since we cannot observe counterfactuals, we simply subtract the means to calculate the ATE, it's basically the lift
# Overall campaign effectiveness on entire customer base

# ========================================= #
# Propensity Score Modeling PSM             #
# ========================================= #
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

# ========================================= #
# Average Treatment Effect on the Treated   #
# ========================================= #
#%%
# Calculate ATT using matched treated-control pairs
att = (matched_df["purchase"] - matched_df["matched_control_purchase"]).mean()
print(f"Estimated ATT using PSM: {att:.4f}")

# # Estimate ATT
# att = df[df['treatment'] == 1]['purchase'].mean() - df[df['treatment'] == 1]['y0'].mean()
# print(f"Estimated ATT: {att:.4f}")
#%%

# Impact on engaged customers - effect on those who received the email
# Match each treated individual with a similar untreated individual based on covariates

# ========================================= #
# Average Treatment Effect on the Untreated #
# ========================================= #
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

# What we missed out on - effect on those who did not receive the email
# Use inverse probability weighting (IPW) to reweight the control group to resemble the treated group.

# ========================================= #
# Conditional Average Treatment Effect      #
# ========================================= #
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


# ========================================= #
# Individual Treatment Effect               #
# ========================================= #
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


# ========================================= #
# Conclusion                                #
# ========================================= #
# If ATT > ATE, the treatment works better for specific groups
# If ATU is low, avoid targeting non responders
# CATE & ITE help identify high responders
