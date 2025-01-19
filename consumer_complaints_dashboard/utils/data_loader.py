# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Import Libraries                                               #
# ============================================================== #
#%%
import pandas as pd
import numpy as np
import requests
import streamlit as st
# from uszipcode import SearchEngine
#%%

# ============================================================== #
# Read in relevant data from parquet file                        #
# ============================================================== #
#%%
all_cols = options=['Date received',
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
             'Complaint ID']
# full_dataset = 7352801
partial_dataset = 100000

@st.cache_data
def fetch_complaints(fields=all_cols, size=partial_dataset):
    """
    Fetch consumer complaints from the CFPB API.

    Args:
        size (int): Number of complaints to fetch.
        fields (list): Fields to retrieve.
        sort_field (str): Field for sorting.

    Returns:
        pd.DataFrame: DataFrame of complaints.
    """
    df = pd.read_csv('C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\consumer_complaints_dashboard\\data\\complaints.csv'
                                  ,usecols=fields
                                  ,nrows=size
                                  ,low_memory=False)
    
    # Categorical
    df['Consumer disputed?'] = df['Consumer disputed?'].fillna('No')
    df['Company response to consumer'] = df['Company response to consumer'].fillna('No response')
    df['Consumer consent provided?'] = df['Consumer consent provided?'].fillna('Consent not provided')
    df['Company public response'] = df['Company public response'].fillna('No response')
    df['Tags'] = df['Tags'].fillna('No tags')
    df['Sub-product'] = df['Sub-product'].fillna(df['Product'])
    df['Issue'] = df['Issue'].fillna("Issue not recorded")
    df['Sub-issue'] = df['Sub-issue'].fillna(df['Issue'])

    # Text
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].fillna('None provided')

    # State & Zip
    def clean_zip_code(zip_code):
        if pd.isna(zip_code):
            return "None provided"
        zip_code = str(zip_code)
        if len(zip_code) != 5 or not zip_code.isdigit() or len(set(zip_code)) == 1:
            return "None provided"
        return zip_code

    # Apply the function to the ZIP code column
    df['ZIP code'] = df['ZIP code'].apply(clean_zip_code)

    # # Initialize the uszipcode search engine
    # search = SearchEngine(simple_zipcode=True)

    # # Function to get state from ZIP code
    # def get_state_from_zip(zip_code):
    #     if zip_code == "None provided":
    #         return np.nan
    #     result = search.by_zipcode(zip_code)
    #     if result:
    #         return result.state
    #     return np.nan

    # # Fill missing state values based on ZIP code
    # df['State'] = df.apply(lambda row: get_state_from_zip(row['ZIP code']) if pd.isna(row['State']) else row['State'], axis=1)

    df['State'] = df['State'].fillna('None provided')

    return df
# df = fetch_complaints()
#%%

# ============================================================== #
# Test Dataset                                                   #
# ============================================================== #
#%%
# df = pd.read_csv('C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\consumer_complaints_dashboard\\data\\complaints.csv')
#%%

# # ============================================================== #
# # Hit API for the consumer complaints dataset                    #
# # ============================================================== #
# #%%
# API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

# def fetch_complaints(fields=["company", "product"], size=10000):
#     """
#     Fetch consumer complaints from the CFPB API.

#     Args:
#         size (int): Number of complaints to fetch.
#         fields (list): Fields to retrieve.
#         sort_field (str): Field for sorting.

#     Returns:
#         pd.DataFrame: DataFrame of complaints.
#     """
#     params = {
#         "size": size,
#     }
    
#     response = requests.get(API_URL, params=params)
#     if response.status_code == 200:
#         hits = response.json()["hits"]["hits"]
#         data = [hit["_source"] for hit in hits]
#         df = pd.DataFrame(data)
#         df = df[fields]
#     else:
#         raise Exception(f"API Error: {response.status_code}, {response.text}")
# #%%