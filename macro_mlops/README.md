<!-- C:\\Users\\wallj\\DS_Projects\\Portfolio_Projects\\econ_env\\scrits\\activate      -->
<!-- Started here https://medium.com/@maximilianoliver25/end-to-end-data-science-in-production-how-i-built-and-deployed-a-full-ml-workflow-6b3d0b0dd262 -->
<!-- Working on econ_env, python 3.13.9 -->
<!-- app -> includes the dashboard -->
<!-- dags -> retrain pipeline -->
<!-- data -> raw & processed csv from FRED api -->
<!-- notebooks -> notebook for training and debugging the model -->
<!-- tests -> running a test on the api -->
<!-- src api -> serve model to manage api -->
<!-- src features -> process features after pulled from FRED api -->
<!-- src ingestion -> fetch data from FRED api -->
<!-- src models -> train the inflation model & store it here -->
<!-- src monitoring -> monitor the feature drift for potential retraining purposes -->
<!-- src utils -> db.py for working with postgresql & helpers for any other useful code -->


<!-- Use macro_mlops/notebooks/model_train.ipynb to train inflation model & test code -->
<!-- 1. macro_mlops/src/ingestion/fetch_data.py -> gets data from api -->
<!-- 2. macro_mlops/src/features/make_features.py -> takes raw data from api and makes features -->
<!-- 3. macro_mlops/src/models/train_model.py -> takes transformed dataset from features script and trains model -->
<!-- 4. macro_mlops/src/api/serve_model.py -> makes api predictions, retrains model, and accesses fastapi app for model -->
<!-- Dockerfile -> manages API -->


<!-- to run docker for api (cli): docker run -p 8000:8000 inflation-api -->
<!-- or run it in the background: -->
<!-- docker run -d -p 8000:8000 -e RETRAIN_SECRET=retraininflationapi123 --name inflation-api-container inflation-api -->
<!-- then open this in browser: http://localhost:8000/docs/ -->
<!-- make sure to open docker first if you get errors -->
<!-- using postgres db to log predictions & metrics, macro_mlops/src/utils/db.py -> to log data -->
<!-- to create db: docker run --name inflation-db -e POSTGRES_PASSWORD=inflation -e POSTGRES_USER=justin -e POSTGRES_DB=inflation -p 5432:5432 -d postgres -->
<!-- python -c "from src.utils.db import init_db; init_db()" -->
<!-- to run docker for db: docker start inflation-db -->
<!-- open the hood for pgsql: docker exec -it inflation-db psql -U justin -d inflation -->
<!-- to find tables in sql: \dt -->
<!-- using dbeaver application as ui to visually explore postgresql db for app -->
<!-- tables: prediction_logs & retrain_runs -->
<!-- retrain function using evidently to establish rules for retraining model -->
<!-- will use data pipeline to manage flow for retraining model -->
<!-- Using prefect instead of apache airflow because it works better w/windows -->

<!-- Architecture -->
<!-- /retrain -> drift check -> log retrain run -> background training -> model swap -> finish log -->