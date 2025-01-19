# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Import Libraries                                               #
# ============================================================== #
#%%
import streamlit as st
import pandas as pd
import os
os.chdir('C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\consumer_complaints_dashboard\\utils')
from data_loader import fetch_complaints
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
    options=['Date received',
            'Product',
            'Sub-product',
            'Issue',
            'Sub-issue',
            'Consumer complaint narrative',
            'Company public response',
            'Company',
            'State',
            'ZIP code',
            'Tags',
            'Consumer consent provided?',
            'Submitted via',
            'Date sent to company',
            'Company response to consumer',
            'Timely response?',
            'Consumer disputed?',
            'Complaint ID'],
    default=["Date received", "Product"]
)

# Fetch data
try:
    st.write("Fetching data from API...")
    complaints_data = fetch_complaints(size=num_complaints,fields=fields)
    # complaints_data = pd.read_csv('C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\consumer_complaints_dashboard\\data\\complaints.csv'
    #                               ,usecols=fields
    #                               ,nrows=num_complaints)
    st.write("### Complaints Data", complaints_data)
except Exception as e:
    st.error(f"Error fetching data: {e}")

# Visualization example
if not complaints_data.empty:
    st.bar_chart(complaints_data["Product"].value_counts())

if not complaints_data.empty:
    st.bar_chart(complaints_data["Company"].value_counts())
#%%