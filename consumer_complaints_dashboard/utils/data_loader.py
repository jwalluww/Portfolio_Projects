# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Import Libraries                                               #
# ============================================================== #
#%%
import pandas as pd
import requests
#%%

# ============================================================== #
# Read in relevant data from parquet file                        #
# ============================================================== #
#%%
def fetch_complaints(fields=["Date received", "Product","Company"], size=10000):
    """
    Fetch consumer complaints from the CFPB API.

    Args:
        size (int): Number of complaints to fetch.
        fields (list): Fields to retrieve.
        sort_field (str): Field for sorting.

    Returns:
        pd.DataFrame: DataFrame of complaints.
    """
    complaints_data = pd.read_csv('C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\consumer_complaints_dashboard\\data\\complaints.csv'
                                #   ,usecols=fields
                                  ,nrows=size)
    return complaints_data
df = fetch_complaints()
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