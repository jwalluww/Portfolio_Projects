# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Page 2: Details and Deep Dive into Complaint Details           #
# ============================================================== #
#%%
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import fetch_complaints
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Complaints & Outcomes", layout="wide")

# Page Title
st.title("Consumer Complaints & Outcomes")
st.markdown("""
This page provides detailed insights into consumer complaints and their outcomes.
Use the filters below to drill into specific details and explore resolution trends.
""")

# Fetch Data
try:
    complaints_data = fetch_complaints()  # Example size
    complaints_data['Date received'] = pd.to_datetime(complaints_data['Date received'])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")

# Date Range Filter
date_range = st.sidebar.date_input(
    "Filter by Date Received",
    value=(complaints_data['Date received'].min(), complaints_data['Date received'].max())
)

# Multi-select Filters
filter_product = st.sidebar.multiselect(
    "Filter by Product",
    options=complaints_data['Product'].unique(),
    default=complaints_data['Product'].unique()
)

filter_company = st.sidebar.multiselect(
    "Filter by Company",
    options=complaints_data['Company'].unique(),
    default=complaints_data['Company'].unique()
)

filter_state = st.sidebar.multiselect(
    "Filter by State",
    options=complaints_data['State'].unique(),
    default=complaints_data['State'].unique()
)

filter_submission = st.sidebar.multiselect(
    "Filter by Submission Method",
    options=complaints_data['Submitted via'].unique(),
    default=complaints_data['Submitted via'].unique()
)

# Apply Filters
filtered_data = complaints_data[
    (complaints_data['Date received'] >= pd.Timestamp(date_range[0])) &
    (complaints_data['Date received'] <= pd.Timestamp(date_range[1])) &
    (complaints_data['Product'].isin(filter_product)) &
    (complaints_data['Company'].isin(filter_company)) &
    (complaints_data['State'].isin(filter_state)) &
    (complaints_data['Submitted via'].isin(filter_submission))
]

# Detailed Complaint Table
st.subheader("Detailed Complaints Table")
st.dataframe(filtered_data, height=400)

# Export Filtered Data
st.download_button(
    label="Export Filtered Data",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name="filtered_complaints.csv",
    mime="text/csv"
)

# Drill-Down Charts
st.subheader("Drill-Down Charts")

# Horizontal Bar Chart: Most Common Issues
issue_counts = filtered_data['Issue'].value_counts().head(10).reset_index()
issue_chart = px.bar(
    issue_counts,
    x='Issue',
    y='index',
    orientation='h',
    title="Top 10 Issues",
    labels={'index': 'Issue', 'Issue': 'Count'}
)
st.plotly_chart(issue_chart, use_container_width=True)

# Pie Chart: Submission Methods
submission_counts = filtered_data['Submitted via'].value_counts().reset_index()
submission_chart = px.pie(
    submission_counts,
    values='Submitted via',
    names='index',
    title="Submission Methods",
    labels={'index': 'Submission Method', 'Submitted via': 'Count'}
)
st.plotly_chart(submission_chart, use_container_width=True)

# Resolution Metrics
st.subheader("Resolution Metrics")

# Stacked Bar Chart: Company Responses
response_data = filtered_data.groupby(['Company', 'Company response to consumer']).size().reset_index(name='Count')
response_chart = px.bar(
    response_data,
    x='Company',
    y='Count',
    color='Company response to consumer',
    title="Company Responses to Consumers",
    labels={'Count': 'Number of Responses'},
    barmode='stack'
)
st.plotly_chart(response_chart, use_container_width=True)

# Dispute Rates
dispute_data = filtered_data.groupby(['Product', 'Consumer disputed?']).size().reset_index(name='Count')
dispute_chart = px.bar(
    dispute_data,
    x='Product',
    y='Count',
    color='Consumer disputed?',
    title="Dispute Rates by Product",
    labels={'Count': 'Number of Disputes'},
    barmode='stack'
)
st.plotly_chart(dispute_chart, use_container_width=True)

# Consumer Sentiment (Optional)
st.subheader("Consumer Sentiment (Word Cloud)")

if st.checkbox("Generate Word Cloud from Complaint Narratives"):
    all_text = " ".join(filtered_data['Consumer complaint narrative'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)