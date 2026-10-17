"""Load the raw CSV into Postgres. Everything downstream reads from the database."""
import logging
import re

import pandas as pd

from .db import init_schema

log = logging.getLogger(__name__)


def snake_case(name):
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [snake_case(c) for c in df.columns]
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_csv_to_postgres(engine, cfg):
    init_schema(engine)
    df = read_csv(cfg["data"]["csv_path"])
    schema, table = cfg["data"]["raw_table"].split(".")

    # full reload keeps the command idempotent
    with engine.begin() as conn:
        conn.exec_driver_sql(f"TRUNCATE {schema}.{table}")
        df.to_sql(table, conn, schema=schema, if_exists="append", index=False,
                  method="multi", chunksize=5000)

    log.info("loaded %d rows into %s.%s", len(df), schema, table)
    return len(df)
