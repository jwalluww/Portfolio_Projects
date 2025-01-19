# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Page 1: Overview and Trends of Consumer Complaints             #
# ============================================================== #
#%%
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import fetch_complaints

# Set up page-specific configurations (optional, if not global)
st.set_page_config(page_title="Overview Trends", layout="wide")

# Title and Description
st.title("Overview Trends")
st.markdown("""
This page provides high-level insights into the trends of consumer complaints. 
Use the filters below to customize the trend view.
""")

# Load Data
try:
    # Load data via data_loader utility
    complaints_data = fetch_complaints()  # Example size
    complaints_data['Date received'] = pd.to_datetime(complaints_data['Date received'])
    earliest_date = complaints_data['Date received'].min().date()
    latest_date = complaints_data['Date received'].max().date()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters for Trend Graph")

# Date Range Filter
date_range = st.sidebar.date_input(
    f"Filter by Date Received from {earliest_date} to {latest_date}",
    value=(complaints_data['Date received'].min(), complaints_data['Date received'].max())
)

# Multiselect Filters
filter_product = st.sidebar.multiselect(
    "Filter by Product",
    options=complaints_data['Product'].unique(),
    default=complaints_data['Product'].unique()
)

filter_state = st.sidebar.multiselect(
    "Filter by State",
    options=complaints_data['State'].unique(),
    default=complaints_data['State'].unique()
)

# Apply Filters
filtered_data = complaints_data[
    (complaints_data['Date received'] >= pd.Timestamp(date_range[0]))
    &(complaints_data['Date received'] <= pd.Timestamp(date_range[1]))
    &(complaints_data['Product'].isin(filter_product))
    &(complaints_data['State'].isin(filter_state))
]

# Trend Graph
if not filtered_data.empty:
    trend_data = filtered_data.groupby('Date received').size().reset_index(name='Complaint Count')
    fig = px.line(
        trend_data, 
        x='Date received', 
        y='Complaint Count', 
        title="Trend of Complaints Over Time",
        labels={"Complaint Count": "Number of Complaints"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")
#%%