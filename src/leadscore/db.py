import os

import pandas as pd
from sqlalchemy import create_engine, text

from .config import PROJECT_ROOT

SCHEMA_FILE = PROJECT_ROOT / "sql" / "schema.sql"


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (docker compose sets it for the app service)")
    return create_engine(url, pool_pre_ping=True)


def init_schema(engine):
    sql = SCHEMA_FILE.read_text()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def read_sql(engine, query, **params):
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def execute(engine, query, **params):
    with engine.begin() as conn:
        conn.execute(text(query), params)
