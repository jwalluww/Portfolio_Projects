# ============================================================== #
# Streamlit App for Consumer Complaints Analysis                 #
# ============================================================== #
# Author: Justin Wall

# ============================================================== #
# Page 3: Geospatial Data from Consumer Complaints               #
# ============================================================== #
#%%
import streamlit as st
import pandas as pd
import pydeck as pdk
from utils.data_loader import fetch_complaints

# Page Configuration
st.set_page_config(page_title="Consumers by Location", layout="wide")

# Page Title
st.title("Consumers by Location")
st.markdown("""
Visualize consumer complaints geographically using the state and ZIP code fields.
""")

# Fetch Data
try:
    complaints_data = fetch_complaints(size=10000)  # Example size
    # Ensure valid ZIP codes and states
    complaints_data = complaints_data.dropna(subset=["ZIP code", "State"])
    complaints_data["ZIP code"] = complaints_data["ZIP code"].astype(str).str[:5]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")

# Multi-select Filter for States
filter_state = st.sidebar.multiselect(
    "Filter by State",
    options=complaints_data['State'].unique(),
    default=complaints_data['State'].unique()
)

# Filter Data by State
filtered_data = complaints_data[complaints_data['State'].isin(filter_state)]

# Generate Latitude and Longitude (using placeholder values for demo)
# You might use a geocoding service or pre-mapped dataset for actual lat/lon
@st.cache_data
def map_zip_to_lat_lon(zip_codes):
    """Map ZIP codes to latitude and longitude. Placeholder logic."""
    import random
    zip_df = pd.DataFrame({
        "ZIP code": zip_codes,
        "latitude": [random.uniform(25, 49) for _ in zip_codes],  # Approx US latitude
        "longitude": [random.uniform(-125, -67) for _ in zip_codes]  # Approx US longitude
    })
    return zip_df

zip_to_lat_lon = map_zip_to_lat_lon(filtered_data["ZIP code"].unique())
mapped_data = pd.merge(filtered_data, zip_to_lat_lon, on="ZIP code")

# Pydeck Map Configuration
st.subheader("Geographical Distribution of Complaints")

# Scatterplot Layer for Pydeck
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=mapped_data,
    get_position='[longitude, latitude]',
    get_radius=10000,  # Radius of points
    get_color='[200, 30, 0, 160]',  # RGBA color
    pickable=True
)

# Pydeck View Configuration
view_state = pdk.ViewState(
    latitude=37.5,  # Center of the map
    longitude=-96.0,
    zoom=3,
    pitch=0
)

# Pydeck Deck
deck = pdk.Deck(
    layers=[scatter_layer],
    initial_view_state=view_state,
    tooltip={"text": "ZIP Code: {ZIP code}\nState: {State}"}
)

# Render the Map
st.pydeck_chart(deck)

# Add Option to Download Filtered Data
st.download_button(
    label="Export Filtered Data",
    data=mapped_data.to_csv(index=False).encode('utf-8'),
    file_name="filtered_consumer_complaints.csv",
    mime="text/csv"
)
#%%