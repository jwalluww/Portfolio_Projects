"""
Double Delta (Difference-in-Differences) Method for Loyalty Program Impact Analysis

📌 **Objective**:
- Measure the causal impact of a treatment on customer spending.
- Use the Double Delta (Difference-in-Differences) method to control for natural spending trends.
- Validate results using OLS regression.

🔍 **Key Takeaways**:
- **Pre-Treatment Spending**: Average spending before the program for both groups.
- **Post-Treatment Spending**: Spending changes after the program.
- **Double Delta Effect**: The true treatment effect of the loyalty program, isolating organic spending trends.
- **Robustness Check**: Regression approach confirms statistical significance.

📌 **Methodology**:
1. **Simulate customer spending data** before and after the program.
2. **Compute Double Delta Effect**: Difference-in-Differences (DiD) estimator.
3. **Run Regression for Robustness**:
   - Fit an **Ordinary Least Squares (OLS) model**: `spending ~ treatment * time`.
   - Validate statistical significance of treatment effects.

📊 **Interpretation**:
- **If the Double Delta Effect is positive**, the loyalty program increased spending.
- **If not significant**, the observed effect may be due to external factors.

✍ **Author**: Justin Wall  
📅 **Date**: 02/15/2025  
"""

# ==========================================
# Create synthetic dataset
# ==========================================
#%%
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Create dataset
n = 1000  # Total customers
df = pd.DataFrame({
    "customer_id": np.arange(n),
    "treatment": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),  # 50% join the loyalty program
})

# Baseline spending before the loyalty program
df["pre_spending"] = np.random.normal(100, 20, size=n)

# Natural spending growth (without treatment)
df["spending_growth"] = np.random.normal(10, 5, size=n)  

# Treatment effect (applies only to treated customers)
treatment_effect = np.random.normal(15, 5, size=n)  # Additional boost from the loyalty program

# Post-treatment spending
df["post_spending"] = df["pre_spending"] + df["spending_growth"] + (df["treatment"] * treatment_effect)

# Show sample data
df.head()
#%%

# ==========================================
# Compute Double Delta Effect               
# ==========================================
#%%
# Compute average spending before & after for both groups
pre_treated = df[df["treatment"] == 1]["pre_spending"].mean()
post_treated = df[df["treatment"] == 1]["post_spending"].mean()
pre_control = df[df["treatment"] == 0]["pre_spending"].mean()
post_control = df[df["treatment"] == 0]["post_spending"].mean()

# Calculate Double Delta (Difference-in-Differences)
double_delta = (post_treated - pre_treated) - (post_control - pre_control)

# Print results
print(f"Pre-Treatment Spending (Treated): ${pre_treated:.2f}")
print(f"Post-Treatment Spending (Treated): ${post_treated:.2f}")
print(f"Pre-Treatment Spending (Control): ${pre_control:.2f}")
print(f"Post-Treatment Spending (Control): ${post_control:.2f}")
print(f"Double Delta Effect: ${double_delta:.2f}")
#%%


# ==========================================
# Robustness Check using Regression         
# ==========================================
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Create a time indicator (0 = pre, 1 = post)
df_long = pd.melt(df, id_vars=["customer_id", "treatment"], value_vars=["pre_spending", "post_spending"],
                  var_name="time", value_name="spending")
df_long["time"] = df_long["time"].map({"pre_spending": 0, "post_spending": 1})

# Run Difference-in-Differences regression
model = smf.ols("spending ~ treatment * time", data=df_long).fit()
print(model.summary())
#%%