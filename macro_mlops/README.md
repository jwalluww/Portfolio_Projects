<!-- Started here https://medium.com/@maximilianoliver25/end-to-end-data-science-in-production-how-i-built-and-deployed-a-full-ml-workflow-6b3d0b0dd262 -->
<!-- Working on econ_env, python 3.13.9 -->
<!-- to run docker: docker run -p 8000:8000 inflation-api --!>
<!-- then open this: http://localhost:8000/docs/ -->
<!-- make sure to open docker first if you get errors --!>
<!-- to create db: docker run --name inflation-db -e POSTGRES_PASSWORD=inflation -e POSTGRES_USER=justin -e POSTGRES_DB=inflation -p 5432:5432 -d postgres -->
<!-- python -c "from src.utils.db import init_db; init_db()" -->
<!-- docker start inflation-db -->
<!-- open the hood for pgsql: docker exec -it inflation-db psql -U justin -d inflation -->
