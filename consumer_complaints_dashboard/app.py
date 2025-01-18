# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Import Libraries                                               #
# ============================================================== #
#%%
import streamlit as st
from utils.data_loader import fetch_complaints
import pandas as pd
#%%

# ============================================================== #
# Set up Streamlit App Basics                                    #
# ============================================================== #
#%%
# Title and header
st.title("Consumer Complaints Dashboard")
st.header("Explore and analyze consumer complaints data")

# Sidebar inputs
st.sidebar.header("Filters")
num_complaints = st.sidebar.slider("Number of complaints", 10, 10000, 100)
fields = st.sidebar.multiselect(
    "Fields to display",
    options=['product',
             'complaint_what_happened',
             'date_sent_to_company',
             'issue',
             'sub_product',
             'zip_code',
             'tags',
             'has_narrative',
             'complaint_id',
             'timely',
             'consumer_consent_provided',
             'company_response',
             'submitted_via',
             'company',
             'date_received',
             'state',
             'consumer_disputed',
             'company_public_response',
             'sub_issue'],
    default=["company", "product"]
)

# Fetch data
try:
    st.write("Fetching data from API...")
    complaints_data = fetch_complaints(size=num_complaints,fields=fields)
    st.write("### Complaints Data", complaints_data)
except Exception as e:
    st.error(f"Error fetching data: {e}")

# Visualization example
if not complaints_data.empty:
    st.bar_chart(complaints_data["product"].value_counts())

if not complaints_data.empty:
    st.bar_chart(complaints_data["company"].value_counts())
#%%