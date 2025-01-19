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
#%%

# ============================================================== #
# Set up Streamlit App Basics                                    #
# ============================================================== #
#%%
# Set page config
st.set_page_config(
    page_title="Consumer Complaints Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and header
st.title("Consumer Complaints Dashboard")
st.header("Explore and analyze consumer complaints data")
#%%

# ============================================================== #
# Sidebar inputs                                                 #
# ============================================================== #
#%%
st.sidebar.header("Filters for the Overview Table")

# Number of complaints slider (extended to allow "All" records)
num_complaints = st.sidebar.slider(
    "Number of complaints", 
    min_value=10, 
    max_value=1000,  # Example max value; adjust to fit your dataset
    value=100, 
    step=10,
    help="Adjust to display the number of complaints."
)

# Multiselect for fields to display
options=[
        'Date received',
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
        'Complaint ID'
    ]
fields = st.sidebar.multiselect(
    "Fields to display",
    options=options,
    default=options,
    help="Select fields to display in the overview table."
)

# Selectbox for sorting
sort_field = st.sidebar.selectbox(
    "Sort by field",
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
    help="Select a field to sort by."
)

# Sorting order (ascending or descending)
sort_order = st.sidebar.radio(
    "Sort order",
    options=["Ascending", "Descending"],
    help="Choose the sort order."
)

# Search specific value within a field
search_field = st.sidebar.selectbox(
    "Search field",
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
    help="Select a field to search within."
)

search_value = st.sidebar.text_input(
    f"Search for value in {search_field}",
    value="",
    help=f"Enter a value to search within the selected field ({search_field})."
)

#%%

# ============================================================== #
# Fetch data                                                     #
# ============================================================== #
#%%
from utils.data_loader import fetch_complaints

# Set up a placeholder for the loading message
loading_message = st.empty()  # Placeholder

try:
    # Display the loading message
    loading_message.write("Fetching data from API...")

    # Fetch the complaints data
    complaints_data = fetch_complaints(size=num_complaints, fields=fields)

    # Remove the loading message
    loading_message.empty()

    # Display the data
    st.write("### Complaints Data", complaints_data)
except Exception as e:
    # Remove the loading message and display an error
    loading_message.empty()
    st.error(f"Error fetching data: {e}")
#%%