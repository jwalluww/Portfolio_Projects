

# =======================================
# Create dataset for MAB simulation
# =======================================
#%%
# Ride ID: Unique identifier for each ride.
# Time of Day: Morning, Afternoon, Evening, Night.
# Day of Week: Weekday vs. Weekend.
# Base Price: Different price points (e.g., $5, $7, $10, $12, etc.).
# Demand Level: Low, Medium, High.
# Completion Rate: % of users who accept the ride at a given price.
# Revenue per Ride: Price × Completion Rate (to help measure optimal pricing).
# We'll also assume that higher prices lower completion rates, but that demand fluctuates based on the time of day and whether it's a weekday or weekend.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
num_rides = 5000

# Time of day categories
time_of_day = np.random.choice(["Morning", "Afternoon", "Evening", "Night"], size=num_rides, p=[0.3, 0.3, 0.3, 0.1])

# Day of week categories (Weekdays more frequent than weekends)
day_of_week = np.random.choice(["Weekday", "Weekend"], size=num_rides, p=[0.7, 0.3])

# Price options
price_options = [5, 7, 10, 12, 15]
prices = np.random.choice(price_options, size=num_rides)

# Demand levels (higher in evening and weekends)
demand_levels = []
for tod, dow in zip(time_of_day, day_of_week):
    if tod == "Evening" or dow == "Weekend":
        demand_levels.append(np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.3, 0.5]))
    else:
        demand_levels.append(np.random.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2]))
demand_levels = np.array(demand_levels)

# Completion rate (probability of accepting the ride) - varies with price and demand
# This would be based on historical data in real life
def get_completion_rate(price, demand):
    base_rate = {5: 0.9, 7: 0.8, 10: 0.6, 12: 0.5, 15: 0.3}[price]
    demand_factor = {"Low": 0.7, "Medium": 1.0, "High": 1.2}[demand]
    return np.clip(base_rate * demand_factor + np.random.normal(0, 0.05), 0, 1)

completion_rates = np.array([get_completion_rate(p, d) for p, d in zip(prices, demand_levels)])

# Revenue per ride = price * completion rate
revenues = prices * completion_rates

# Create DataFrame
df = pd.DataFrame({
    "Ride_ID": np.arange(num_rides),
    "Time_of_Day": time_of_day,
    "Day_of_Week": day_of_week,
    "Price": prices,
    "Demand_Level": demand_levels,
    "Completion_Rate": completion_rates,
    "Revenue": revenues
})

df.head()
#%%

# =======================================
# Visualize the dataset
# =======================================
#%%
# Summary statistics
summary = df.groupby(["Price", "Demand_Level"]).agg({"Completion_Rate": "mean", "Revenue": "mean"}).reset_index()

# Plot completion rate vs. price
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Price", y="Completion_Rate", hue="Demand_Level")
plt.title("Completion Rate by Price and Demand Level")
plt.ylabel("Completion Rate")
plt.xlabel("Price ($)")
plt.legend(title="Demand Level")
plt.show()

# Plot revenue distribution per price
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Price", y="Revenue", hue="Demand_Level")
plt.title("Revenue by Price and Demand Level")
plt.ylabel("Revenue ($)")
plt.xlabel("Price ($)")
plt.legend(title="Demand Level")
plt.show()
#%%

# =======================================
# Greedy Algorithm
# =======================================
#%%
import numpy as np
import pandas as pd
import random

# Define possible price points
prices = [5, 7, 10, 12, 15]

# Simulated demand function (higher price -> lower acceptance rate)
def get_completion_rate(price, demand_level):
    base_rate = {5: 0.75, 7: 0.6, 10: 0.45, 12: 0.35, 15: 0.25}  # Base acceptance rates
    demand_factor = 1.2 if demand_level == "High" else 0.8  # Adjust for demand
    return min(1, base_rate[price] * demand_factor)  # Ensure rate is <= 1

# Generate synthetic dataset
np.random.seed(42)
n_samples = 5000  # Total ride offers
demand_levels = ["Low", "High"]
time_of_day = ["Morning", "Afternoon", "Evening", "Night"]
days_of_week = ["Weekday", "Weekend"]

data = []
for _ in range(n_samples):
    price = random.choice(prices)
    demand = random.choice(demand_levels)
    tod = random.choice(time_of_day)
    dow = random.choice(days_of_week)
    
    completion_rate = get_completion_rate(price, demand)
    accepted = np.random.rand() < completion_rate  # Simulate ride acceptance
    revenue = price if accepted else 0  # Earn revenue if ride is accepted
    
    data.append([price, demand, tod, dow, accepted, revenue])

df = pd.DataFrame(data, columns=["Price", "Demand_Level", "Time_of_Day", "Day_of_Week", "Accepted", "Revenue"])

# Implement Greedy Algorithm
price_revenue = {p: [] for p in prices}  # Track revenue for each price

def greedy_bandit(n_rounds=1000):
    chosen_prices = []
    total_revenue = 0
    
    for _ in range(n_rounds):
        # Pick price with highest average revenue (or random if no history)
        avg_revenues = {p: np.mean(price_revenue[p]) if price_revenue[p] else 0 for p in prices}
        best_price = max(avg_revenues, key=avg_revenues.get)
        
        # Simulate a new ride at the chosen price
        sample = df[df['Price'] == best_price].sample(1).iloc[0]
        revenue = sample["Revenue"]
        
        # Update records
        price_revenue[best_price].append(revenue)
        total_revenue += revenue
        chosen_prices.append(best_price)
    
    return chosen_prices, total_revenue

# Run the greedy algorithm
chosen_prices, total_revenue = greedy_bandit()

# Display results
chosen_price_counts = pd.Series(chosen_prices).value_counts().sort_index()
print("Final chosen price distribution:")
print(chosen_price_counts)
print(f"Total revenue over {len(chosen_prices)} rounds: ${total_revenue}")
#%%