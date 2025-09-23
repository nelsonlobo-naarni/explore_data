import trino
import pandas as pd
import io
import boto3
from datetime import datetime, date, timedelta
import logging
import json

# Configure logging for this file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def connect_to_trino():
    """Establishes a connection to the Trino database."""
    logging.info("🔌 STEP 1: Connecting to Trino...")
    try:
        conn = trino.dbapi.connect(
            host="trino",
            port=8080,
            user="admin",
            catalog="adhoc",
            schema="default"
        )
        logging.info("✅ STEP 1: Connected to Trino")
        return conn
    except Exception as e:
        logging.critical(f"❌ ERROR: Failed to connect to Trino: {e}")
        return None

def execute_query(conn, query: str) -> pd.DataFrame:
    """Executes a given SQL query and returns the results in a DataFrame."""
    logging.info("⚙️ Executing query...")
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        logging.info("✅ Query executed successfully!")
        return df
    except Exception as e:
        logging.error(f"❌ ERROR: Query execution failed: {e}")
        return pd.DataFrame()


def fetch_data_for_day(conn, date_str: str, columns_to_select: list, table: str, ids: list = None) -> pd.DataFrame:
    """
    Constructs a query and fetches data for a given date and optional list of IDs.
    
    Args:
        conn: The Trino connection object.
        date_str (str): The date in 'YYYY-MM-DD' format.
        columns_to_select (list): A list of strings representing the columns and their expressions.
        table (str): The name of the database table to query.
        ids (list, optional): A list of vehicle IDs. If None, all IDs are fetched.
        
    Returns:
        pd.DataFrame: A DataFrame with the fetched raw data.
    """
    logging.info(f"📥 STEP 2a: Validating and fetching data for {date_str}...")

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        previous_date = target_date - timedelta(days=1)
    except ValueError as e:
        logging.error(f"Invalid date format: {e}")
        return pd.DataFrame()  

    where_clause = ""
    # if ids is not None and len(ids) > 0:
    #     id_list_str = ", ".join(f"'{id}'" for id in ids)
    #     where_clause = f"AND id IN ({id_list_str})"
    if ids is not None and len(ids) > 0:
        id_list_str = ", ".join(f"'{id}'" for id in ids)
        where_clause = f"AND vehicle_id IN ({id_list_str})"
    columns_str = ", ".join(columns_to_select)
    
    query = f"""
    WITH two_days_data AS (
        SELECT
            {columns_str}
        FROM adhoc.facts_prod.{table}
        WHERE
            (dt = DATE '{target_date.isoformat()}' OR dt = DATE '{previous_date.isoformat()}')
            {where_clause}
    )
    SELECT
        *
    FROM
        two_days_data
    WHERE
        CAST(IST AS DATE) = DATE '{target_date.isoformat()}'    
    """
    # logging.info(query)

    df = execute_query(conn, query)
    
    if not df.empty:
        logging.info(f"✅ STEP 2d: Data fetching for {date_str} completed, Rows fetched: {len(df)}")
    else:
        logging.warning(f"⚠️ STEP 2d: No data found for {date_str}. Returning empty DataFrame.")
    
    return df


def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'

def _qualify_and_quote(table_fq: str) -> str:
    parts = table_fq.split(".")
    if len(parts) != 3:
        raise ValueError(f"Table must be 'catalog.schema.table', got: {table_fq}")
    return ".".join(_quote_ident(p) for p in parts)

def _trino_type_from_series(s: pd.Series) -> str:
    dtype = str(s.dtype)
    if "datetime64" in dtype:
        return "timestamp(6)"
    if dtype == "bool":
        return "boolean"
    if dtype.startswith("int"):
        return "bigint"
    if dtype.startswith("float"):
        return "double"
    return "varchar"


def write_df_to_iceberg(conn, df: pd.DataFrame, table: str, batch_size: int = 5000):
    print("💾 [4/5] STEP 4a: Preparing to write results to Iceberg...")
    # This variable is already correctly formatted as 'adhoc.facts_prod.power_consumption_report'
    REPORT_TABLE = f"adhoc.facts_prod.{table}"

    # This variable is not used anywhere else, and should be removed.
    # REPORT_S3_LOCATION = f"s3a://naarni-data-lake/aqua/warehouse/facts_prod.db/{table}/"

    cur = conn.cursor()
    # Corrected: Pass the fully qualified REPORT_TABLE to the function
    fq_table = _qualify_and_quote(REPORT_TABLE)

    if df is None or df.empty:
        cols = list(df.columns) if df is not None else []
        if cols:
            column_defs = [f'{_quote_ident(col)} {_trino_type_from_series(df[col])}' for col in cols]
        else:
            column_defs = ['"id" varchar']
    else:
        column_defs = [f'{_quote_ident(col)} {_trino_type_from_series(df[col])}' for col in df.columns]

    cols_sql = ",\n        ".join(column_defs)

    # Note: If your Trino catalog is not configured to accept the `WITH` properties for Iceberg,
    # you may need to add the WITH clause back as per our previous discussion if that catalog
    # is a Hive catalog. However, this fix addresses the immediate 'Table must be' error.
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {fq_table} (
        {cols_sql}
    )
    """
    print("🔍 [4/5] STEP 4b: Creating table if not exists...")
    cur.execute(create_sql)
    print(f"🛠️ [4/5] STEP 4c: Ensured Iceberg table {table} exists")

    if df is None or df.empty:
        print("⚠️ [4/5] No rows to insert, skipping insert step")
        cur.close()
        return

    col_idents = ", ".join(_quote_ident(c) for c in df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    insert_sql = f"INSERT INTO {fq_table} ({col_idents}) VALUES ({placeholders})"
    print("🔍 [4/5] STEP 4d: Prepared INSERT statement")

    converters = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        if "datetime64" in dtype:
            converters.append(lambda v: (pd.to_datetime(v).tz_localize(None).to_pydatetime()
                                         if pd.notna(v) else None))
        elif dtype == "bool":
            converters.append(lambda v: (bool(v) if pd.notna(v) else None))
        elif dtype.startswith(("int", "float")):
            converters.append(lambda v: (v if pd.notna(v) else None))
        else:
            converters.append(lambda v: (str(v) if pd.notna(v) else None))

    def to_row(t):
        return tuple(conv(val) for conv, val in zip(converters, t))

    tuples_iter = (to_row(t) for t in df.itertuples(index=False, name=None))
    from itertools import islice
    peek = list(islice(tuples_iter, 1))
    if peek:
        print(f"🔍 [4/5] STEP 4e: First row preview: {peek[0]}")

    def chain_peek_and_rest():
        if peek:
            yield peek[0]
        for t in tuples_iter:
            yield t

    def chunks(iterable, size):
        it = iter(iterable)
        for first in it:
            yield [first] + list(islice(it, size - 1))

    total = 0
    for batch in chunks(chain_peek_and_rest(), batch_size):
        cur.executemany(insert_sql, batch)
        total += len(batch)
        print(f"✅ [4/5] STEP 4f: Inserted {len(batch)} rows (total {total}) into {table}")

    print(f"🎉 [4/5] STEP 4g: Finished inserting {total} rows into {table}")
    cur.close()

def drop_table(conn, table_name: str):
    """Drops a specified table from the database."""
    REPORT_TABLE = f"adhoc.facts_prod.{table_name}"
    try:
        query = f"DROP TABLE IF EXISTS {REPORT_TABLE}"
        cursor = conn.cursor()
        cursor.execute(query)
        logging.info(f"✅ Table {REPORT_TABLE} dropped successfully.")
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to drop table {REPORT_TABLE}: {e}")