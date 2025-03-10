"""
Price Elasticity of Demand - EconML DML Model
---

🔍 **Situation**:

📌 **Task**:

✨ **Action**: 

📈 **Result**:

✍ **Author**: Justin Wall
📅 **Updated**: 03/04/2025
"""

# ==========================================
# Create synthetic dataset
# ==========================================
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import LinearDML
from econml.iv.dml import DMLIV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# Simulate observational data with 3 product categories
n = 5000
categories = np.random.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])

# Unobserved demand factor
base_demand = np.random.normal(50, 10, size=n)

# Category-specific demand effects
demand = base_demand + np.where(categories == "A", 5, 0) + np.where(categories == "B", -5, 0)

# Marketing and competitor price
marketing = np.random.uniform(0, 100, size=n)
competitor_price = np.random.uniform(5, 20, size=n)
supplier_cost = np.random.uniform(10, 50, size=n)

# Price function with nonlinear effects for category C
price = (
    20 + 0.5 * demand + 0.2 * marketing - 0.3 * competitor_price + 0.4 * supplier_cost
    + np.where(categories == "C", 0.05 * demand**2, 0)
    + np.random.normal(0, 2, size=n)
)

# Sales function with category-based elasticity
sales = (
    500 - 2.5 * price + 1.5 * demand + 0.8 * marketing
    + np.where(categories == "B", -1.5 * price, 0)  # More elastic for B
    + np.where(categories == "C", -0.5 * price**2, 0)  # Nonlinear for C
    + np.random.normal(0, 10, size=n)
)

# Create DataFrame
df = pd.DataFrame({
    'sales': sales, 'price': price, 'marketing': marketing,
    'competitor_price': competitor_price, 'supplier_cost': supplier_cost,
    'category': categories
})

df.head()
#%%

# ==========================================
# Train Test Split & Standardize Features
# ==========================================
#%%
# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
scaler = StandardScaler()

# Encode category as dummies
X_train = pd.get_dummies(train.drop(columns=['sales', 'price', 'supplier_cost']), drop_first=True)
X_test = pd.get_dummies(test.drop(columns=['sales', 'price', 'supplier_cost']), drop_first=True)

# Standardize features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y_train, T_train, Z_train = train['sales'], train['price'], train['supplier_cost']
Y_test, T_test, Z_test = test['sales'], test['price'], test['supplier_cost']
#%%

# ==========================================
# Fit price & sales models
# ==========================================
#%%
# Define hyperparameter grids
rf_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

lasso_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

# Tune model_y (Random Forest)
rf = RandomForestRegressor(random_state=42)
rf_cv = GridSearchCV(rf, rf_grid, cv=3, scoring="neg_mean_squared_error")
rf_cv.fit(X_train, Y_train)

# Tune model_t (Lasso)
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, lasso_grid, cv=3, scoring="neg_mean_squared_error")
lasso_cv.fit(X_train, T_train)

# Best models
best_rf = rf_cv.best_estimator_
best_lasso = lasso_cv.best_estimator_
#%%

# ==========================================
# Fit DML Model
# ==========================================
#%%
# Use in DML
dml = LinearDML(model_y=best_rf, model_t=best_lasso, random_state=42)
dml.fit(Y_train, T_train, X=X_train)

# Get price elasticity estimate
price_elasticity = dml.effect(X_test)
print("Estimated Price Elasticity (Mean):", np.mean(price_elasticity))
# A price elasticity of 0.96 means that a 1% increase in price leads to a 0.96% increase in sales
#%%

# ==========================================
# Fit DML & IV Model
# ==========================================
#%%
iv_model = DMLIV(discrete_treatment=False,
                 discrete_instrument=False,
                 random_state=42)

iv_model.fit(Y=Y_train, T=T_train, Z=Z_train, X=X_train)

# Get price elasticity estimate
price_elasticity = iv_model.effect(X_test)
print("Estimated Price Elasticity (Mean):", np.mean(price_elasticity))
# A price elasticity of 0.96 means that a 1% increase in price leads to a 0.96% increase in sales
#%%

# ==========================================
# Graph out price change by sales
# ==========================================
#%%
# Simulating price changes
price_changes = np.linspace(-20, 20, 10)  # -20% to +20% price change

# Compute sales impact for DML Model
sales_impact_dml = [(1 + e / 100) * np.mean(Y_test) for e in price_changes * np.mean(dml.effect(X_test))]

# Compute sales impact for DML-IV Model
sales_impact_iv = [(1 + e / 100) * np.mean(Y_test) for e in price_changes * np.mean(iv_model.effect(X_test))]

# Plot both models
plt.figure(figsize=(8, 6))
plt.plot(price_changes, sales_impact_dml, marker='o', linestyle='-', color='b', label="DML Model")
plt.plot(price_changes, sales_impact_iv, marker='s', linestyle='--', color='g', label="DML-IV Model")
plt.axhline(np.mean(Y_test), linestyle="--", color="red", label="Current Sales")

# Labels and legend
plt.xlabel("Price Change (%)")
plt.ylabel("Predicted Sales")
plt.title("Comparison of Price Elasticity Models on Sales Impact")
plt.legend()
plt.show()
#%%

# ==========================================
# Heat Map for Elascitiy by Category
# ==========================================
#%%
# Compute category-level elasticity
elasticity = {}
for category in ["A", "B", "C"]:
    cat_mask = test["category"] == category
    elasticity[category] = np.mean(iv_model.effect(X_test[cat_mask]))

# Heatmap for price elasticity
sns.heatmap(pd.DataFrame(elasticity, index=["Elasticity"]), annot=True, cmap="coolwarm", center=0)
plt.title("Price Elasticity by Product Category")
plt.show()
#%%

# ==========================================
# Revenue Function
# ==========================================
#%%
# Revenue function
def calculate_revenue(price_change_pct, units_sold):
    elasticity_adjustment = {category: (1 + price_change_pct / 100) * elasticity[category] for category in elasticity}
    new_sales = {category: units_sold * (1 + elasticity_adjustment[category]) for category in elasticity}
    revenue = {category: new_sales[category] * (1 + price_change_pct / 100) * np.mean(price) for category in new_sales}
    return revenue

# Example usage
print(calculate_revenue(-10, 1000))
#%%