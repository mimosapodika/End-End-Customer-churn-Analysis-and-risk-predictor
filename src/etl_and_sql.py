import sqlite3
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/telcochurn.csv")
DB_PATH = Path("data/churn.db")
SQL_PATH = Path("sql/query.sql")


def run_etl():
    df = pd.read_csv(DATA_PATH)

    # Fix blank spaces in TotalCharges (occurs when tenure == 0 or for new emp)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)  #if yes in the churn colom its 1 and viceversa'''

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)#table name customer if already exist just override no duplicate
        print(f"[ETL] Loaded {len(df)} rows into {DB_PATH} (table: customers).\n")

        sql_script = SQL_PATH.read_text()
        queries = [q.strip() for q in sql_script.split(";") if q.strip()]

        for idx, query in enumerate(queries, start=1):
            print(f"--- Query {idx} ---")
            result = pd.read_sql_query(query, conn)
            print(result.to_string(index=False))
            print()


if __name__ == "__main__":
    run_etl()
