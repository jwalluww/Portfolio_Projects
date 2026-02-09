from sqlalchemy import create_engine, Column, Float, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import datetime
import os

db_url_container = "postgresql://justin:inflation@inflation-db:5432/inflation"
db_url_local = "postgresql://justin:inflation@localhost:5432/inflation"
DB_URL = os.getenv("DATABASE_URL", db_url_local)

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Table Schema
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) # this still works despite the cross off
    cpi_lag1 = Column(Float)
    unemployment_rate_lag1 = Column(Float)
    interest_rate_lag1 = Column(Float)
    oil_price_lag1 = Column(Float)
    gdp_lag1 = Column(Float)
    m2_money_lag1 = Column(Float)
    prediction = Column(Float)

# One-Time: Create Table
def init_db():
    Base.metadata.create_all(bind=engine)

class RetrainRun(Base):
    __tablename__ = "retrain_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(Text, default="started", nullable=False)
    metrics = Column(JSONB, nullable=True)