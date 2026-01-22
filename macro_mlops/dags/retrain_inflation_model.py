from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime,timedelta

default_args = {
    "owner": "justin",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_inflation_model",
    default_args=default_args,
    description="DAG to Fetch FRED data, engineer features, retrain inflation prediction model",
    schedule_interval="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command="python src/ingestion/fetch_data.py",
    )

    make_features = BashOperator(
        task_id="make_features",
        bash_command="python src/features/make_features.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python src/models/train_model.py",
    )

    fetch_data >> make_features >> train_model