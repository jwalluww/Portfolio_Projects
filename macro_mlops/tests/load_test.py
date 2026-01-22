import requests

payload = {
    "cpi_lag1": 305.4,
    "unemployment_rate_lag1": 3.8,
    "interest_rate_lag1": 5.3,
    "oil_price_lag1": 82.4,
    "gdp_lag1": 22800,
    "m2_money_lag1": 22000
}

r = requests.post("http://localhost:8000/predict", json=payload)
print(r.json())