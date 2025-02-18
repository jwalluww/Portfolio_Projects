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
import numpy as np
import pandas as pd
from hmmlearn import hmm

# Define observed events (website actions)
events = ["home", "category_page", "product_page", "add_to_cart", "checkout", "purchase"]

# Encode events as integers
event_to_index = {event: i for i, event in enumerate(events)}
index_to_event = {i: event for event, i in event_to_index.items()}

# Simulate user sessions with observed actions
np.random.seed(42)
user_sessions = []
num_users = 500

for _ in range(num_users):
    session = []
    num_events = np.random.randint(3, 10)  # Each user has 3-10 actions
    
    for _ in range(num_events):
        event = np.random.choice(events, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])  # Skewed probabilities
        session.append(event_to_index[event])
    
    user_sessions.append(session)

# Flatten data for HMM
X = np.concatenate(user_sessions).reshape(-1, 1)

# Define and fit HMM model
num_hidden_states = 4  # Define 4 hidden states (browsing, exploring, considering, purchasing)
model = hmm.MultinomialHMM(n_components=num_hidden_states, n_iter=100, random_state=42)
model.fit(X)

# Predict hidden states for the dataset
hidden_states = model.predict(X)

# Convert results to a DataFrame for analysis
df = pd.DataFrame({
    "Observed Event": [index_to_event[i] for i in X.flatten()],
    "Hidden State": hidden_states
})

# Display a sample of the results
df.head(20)
#%%