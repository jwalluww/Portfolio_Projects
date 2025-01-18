# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wal

# ============================================================== #
# Import Libraries                                               #
# ============================================================== #
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
#%%

# ============================================================== #
# Hit API for the consumer complaints dataset                    #
# ============================================================== #
#%%

# Define the API endpoint
url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

# Query parameters
params = {
    "field": ["company"],
    "size": 10,  # Number of records to retrieve
    "from": 0,   # Starting record
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Extract the relevant data from the response
    complaints = [complaint["_source"] for complaint in data["hits"]["hits"]]
    
    # Convert the list of complaints into a DataFrame
    df = pd.DataFrame(complaints)
    
    # Print the DataFrame
    print(df)
else:
    print(f"Error: {response.status_code}, {response.text}")
#%%