import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0)),
    "age": Column(int, Check.between(18, 100)),
    "income": Column(float, Check.greater_than(0)),
})

def validate():
    df = pd.read_csv("data/sample.csv")
    schema.validate(df)

if __name__ == "__main__":
    validate()