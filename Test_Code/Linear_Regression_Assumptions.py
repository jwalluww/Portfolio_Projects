"""
Linear Regression Assumptions

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, levene, boxcox

# Generate synthetic dataset with some assumption violations
np.random.seed(42)
n = 200
X1 = np.random.normal(0, 1, n)
X2 = 0.5 * X1 + np.random.normal(0, 0.1, n)  # Introduce multicollinearity
X3 = np.random.normal(0, 1, n)
noise = np.random.normal(0, 2, n)
Y = 3 + 2 * X1 + 3 * X2 + 1.5 * X3 + noise

data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

# Splitting dataset
X = data[['X1', 'X2', 'X3']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant term for statsmodels
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit linear regression model
model = sm.OLS(y_train, X_train_const).fit()
print(model.summary())

# 1. Linearity Assumption (Check Residuals vs. Fitted Values)
y_pred = model.predict(X_train_const)
residuals = y_train - y_pred
plt.figure(figsize=(8, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Fix: Apply polynomial or log transformation if necessary

# 2. Independence of Errors (Durbin-Watson Test)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_stat}")

# 3. Normality of Residuals (Shapiro-Wilk Test & Histogram)
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.show()
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test: W={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Fix: Apply log(Y) or Box-Cox transformation if p-value < 0.05

# 4. Homoscedasticity (Levene’s Test & Residual Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Homoscedasticity Check")
plt.show()

levene_test = levene(y_train, y_pred)
print(f"Levene’s Test: W={levene_test.statistic}, p-value={levene_test.pvalue}")

# Fix: Use weighted least squares regression if necessary

# 5. Multicollinearity (VIF Test)
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(vif_data)

# Fix: Drop or combine highly correlated variables if VIF > 5
