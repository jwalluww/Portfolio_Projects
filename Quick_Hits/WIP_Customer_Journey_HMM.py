"""
Hidden Markov Model (HMM) for Customer Journey Analysis

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
# Create Fake Dataset for Customer Journey
# =============================================
#%%
# Re-import necessary libraries since execution state was reset
import numpy as np
import pandas as pd

# Define dataset parameters
num_users = 500  # Number of users
max_days = 30  # Max tracking window for each user
dormancy_threshold = 7  # Days of inactivity before considering a user lost

# Define the new observation types
observations = ["browse", "email_engagement", "app_engagement", "engaged_browse", "purchase"]
obs_probs = [0.5, 0.15, 0.1, 0.2, 0.05]  # Probabilities for each action type

# Generate user journeys with the new observations
user_data = []
for user_id in range(1, num_users + 1):
    start_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 10), unit='D')
    current_date = start_date
    last_action_date = start_date
    actions = []
    outcome = None

    while True:
        # Decide next action
        action = np.random.choice(observations, p=obs_probs)
        actions.append((current_date, action))
        last_action_date = current_date

        # If purchase, stop tracking
        if action == "purchase":
            outcome = "purchased"
            break

        # Move to the next day
        current_date += pd.Timedelta(days=1)

        # Check dormancy
        if (current_date - last_action_date).days >= dormancy_threshold:
            outcome = "lost"
            break

        # Stop if we exceed max tracking window
        if (current_date - start_date).days > max_days:
            break

    # Store user journey
    for event_date, action in actions:
        user_data.append([user_id, start_date, event_date, action, (event_date - start_date).days, outcome])

# Create DataFrame
df = pd.DataFrame(user_data, columns=["User ID", "Start Date", "Event Date", "Action", "Days Since Start", "Outcome"])

# Sort data
df = df.sort_values(by=["User ID", "Event Date"]).reset_index(drop=True)

# Display sample data
df.head(10)

# The dataset is now updated with the five observation types:
# Browse (quick website visit)
# Email Engagement (clicked an email)
# App Engagement (downloaded & used app)
# Engaged Browse (spent time or added to cart)
# Purchase (completed a transaction)

# Each user has a sequence of events tracked from their start date until they either:
# Make a purchase (outcome: "purchased")
# Become inactive for 7 days (outcome: "lost")
# Continue browsing beyond 30 days (rare but possible)

# Browse → 0
# Email Engagement → 1
# App Engagement → 2
# Engaged Browse → 3
# Purchase → 4
#%%

# =============================================
# Build the HMM Model
# =============================================
#%%
# Re-import necessary libraries since execution state was reset
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# Encode categorical actions into numeric values
# order does not matter here!
label_encoder = LabelEncoder()
df["Action Encoded"] = label_encoder.fit_transform(df["Action"])

# Group sequences by User ID
user_sequences = df.groupby("User ID")["Action Encoded"].apply(list).values

# Convert sequences into a format usable by HMM
X = np.concatenate(user_sequences)  # Flatten the sequences
lengths = [len(seq) for seq in user_sequences]  # Track sequence lengths

# Define and train the Hidden Markov Model
num_hidden_states = 4  # Assume 4 hidden states for different customer behaviors
hmm_model = hmm.MultinomialHMM(n_components=num_hidden_states, n_iter=100, random_state=42)
hmm_model.fit(X.reshape(-1, 1), lengths)

# Predict hidden states for each sequence
hidden_states = [hmm_model.predict(np.array(seq).reshape(-1, 1)) for seq in user_sequences]

# Store hidden state predictions back into DataFrame
df["Hidden State"] = np.concatenate(hidden_states)

# Display a sample of the updated dataset with hidden states
df.head(10)
#%%

# =============================================
# Evaluate the HMM Model
# =============================================
#%%
# Assuming X and lengths are your data and sequence lengths
# Define a function to calculate AIC and BIC
def compute_aic_bic(model, X, lengths):
    logL = model.score(X, lengths)
    n_params = (model.n_components ** 2) + (model.n_components * len(np.unique(X))) - 1
    n_obs = len(X)
    aic = 2 * n_params - 2 * logL
    bic = n_params * np.log(n_obs) - 2 * logL
    return logL, aic, bic

# Train HMM with 1 hidden state
model_1 = hmm.MultinomialHMM(n_components=1, n_iter=100)
model_1.fit(X.reshape(-1, 1))
logL_1, aic_1, bic_1 = compute_aic_bic(model_1, X.reshape(-1, 1), lengths)

# Train HMM with 2 hidden states
model_2 = hmm.MultinomialHMM(n_components=2, n_iter=100)
model_2.fit(X.reshape(-1, 1))
logL_2, aic_2, bic_2 = compute_aic_bic(model_2, X.reshape(-1, 1), lengths)

# Train HMM with 3 hidden states
model_3 = hmm.MultinomialHMM(n_components=3, n_iter=100)
model_3.fit(X.reshape(-1, 1))
logL_3, aic_3, bic_3 = compute_aic_bic(model_3, X.reshape(-1, 1), lengths)

# Train HMM with 4 hidden states
model_4 = hmm.MultinomialHMM(n_components=4, n_iter=100)
model_4.fit(X.reshape(-1, 1))
logL_4, aic_4, bic_4 = compute_aic_bic(model_4, X.reshape(-1, 1), lengths)

# Train HMM with 5 hidden states
model_5 = hmm.MultinomialHMM(n_components=5, n_iter=100)
model_5.fit(X.reshape(-1, 1))
logL_5, aic_5, bic_5 = compute_aic_bic(model_5, X.reshape(-1, 1), lengths)

# Train HMM with 6 hidden states
model_6 = hmm.MultinomialHMM(n_components=6, n_iter=100)
model_6.fit(X.reshape(-1, 1))
logL_6, aic_6, bic_6 = compute_aic_bic(model_6, X.reshape(-1, 1), lengths)

# Display the results
print(f"Model with 1 Hidden State: LogL = {logL_1}, AIC = {aic_1}, BIC = {bic_1}")
print(f"Model with 2 Hidden States: LogL = {logL_2}, AIC = {aic_2}, BIC = {bic_2}")
print(f"Model with 3 Hidden States: LogL = {logL_3}, AIC = {aic_3}, BIC = {bic_3}")
print(f"Model with 4 Hidden States: LogL = {logL_4}, AIC = {aic_4}, BIC = {bic_4}")
print(f"Model with 5 Hidden States: LogL = {logL_5}, AIC = {aic_5}, BIC = {bic_5}")
print(f"Model with 6 Hidden States: LogL = {logL_6}, AIC = {aic_6}, BIC = {bic_6}")
#%%

# =============================================
# Visualize the Hidden States
# =============================================
#%%
# Re-import necessary libraries since the execution state was reset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# Define the labels for observed actions
observed_labels = ["Browse", "Email Engagement", "App Engagement", "Engaged Browse", "Purchase"]

# Define and visualize the transition matrix and emission probabilities for the chosen HMM models

def plot_matrix(matrix, labels, title, cmap="Blues"):
    """Helper function to plot transition and emission matrices."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()

# Plot transition matrices
plot_matrix(model_4.transmat_, ["State 0", "State 1", "State 2", "State 3"], "Transition Matrix (4 States)")
plot_matrix(model_3.transmat_, ["State 0", "State 1", "State 2"], "Transition Matrix (3 States)")

# Plot emission probabilities
plot_matrix(model_4.emissionprob_, observed_labels, "Emission Probabilities (4 States)")
plot_matrix(model_3.emissionprob_, observed_labels, "Emission Probabilities (3 States)")

# Extract the emission probabilities for interpretation
emission_probs_4 = model_4.emissionprob_
emission_probs_3 = model_3.emissionprob_

emission_probs_4, emission_probs_3
#%%