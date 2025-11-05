import trino
import pandas as pd
import io
import boto3
from datetime import datetime
from itertools import islice
import numpy as np


# ---- reporting config (edit ONLY this) ----
TABLE_NAME = "braking_analysis_results"   # <— change only this

# derived (don’t edit)
REPORT_TABLE = f"adhoc.facts_prod.{TABLE_NAME}"
REPORT_S3_LOCATION = f"s3a://naarni-data-lake/aqua/warehouse/facts_prod.db/{TABLE_NAME}/"

# --------------------
# Step 1: Connect to Trino
# --------------------
print("🔌 [1/5] STEP 1: Connecting to Trino...")

conn = trino.dbapi.connect(
    host="trino",       # just hostname or IP
    port=8080,
    user="admin",
    catalog="adhoc",
    schema="default"
)

print("✅ [1/5] STEP 1: Connected to Trino")

# --------------------
# Step 2: Function to fetch data for a given day
# --------------------
def fetch_data_for_day(conn, date_str: str) -> pd.DataFrame:
    print(f"📥 [2/5] STEP 2a: Validating and fetching data for {date_str}...")

    datetime.strptime(date_str, "%Y-%m-%d")

    cursor = conn.cursor()
    print(f"⚙️ [2/5] STEP 2b: Executing query for {date_str}...")
    cursor.execute(f"""
        SELECT 
            "id","timestamp","BrakePedalPos", "Vehicle_speed_VCU"
        FROM
            facts_prod.can_parsed_output_100
        WHERE
            dt = DATE '{date_str}'
    """)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()

    df = pd.DataFrame(rows, columns=columns)
    print(f"📊 [2/5] STEP 2c: Rows fetched for {date_str}: {len(df)}")
    return df

# -------------------- 
# Step 3: Function to process braking data
# --------------------
import pandas as pd
def analyze_braking_events(df):
    """
    Analyzes braking events for multiple device IDs from a single DataFrame,
    filtering based on a fixed top speed of 30 km/h.
    
    The logic is fine-tuned to extract the exact start point of braking
    (BrakePedalPos > 0.0) within a 10-second window before a hard stop.

    Args:
        df (pd.DataFrame): The input DataFrame containing all bus data.
    
    Returns:
        pd.DataFrame: A DataFrame with a summary of the braking events.
    """
    
    # Define fixed parameters as requested
    top_speed = 30.0  # Fixed top speed filter (km/h)
    search_window_seconds = 10.0  # Fixed time window to look for brake press
    
    # Identify unique device IDs from the DataFrame
    device_ids = df['id'].unique().tolist()

    all_event_data = []
    
    # Iterate through each unique device ID
    for device_id in device_ids:
        # Filter the DataFrame for the current device ID
        device_df = df[df['id'] == device_id].copy()
        
        if device_df.empty:
            continue

        # Convert timestamp to human-readable IST and clean the data
        device_df['IST'] = pd.to_datetime(device_df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        device_df = device_df[["IST", "BrakePedalPos", "Vehicle_speed_VCU"]].copy()
        device_df.dropna(subset=["Vehicle_speed_VCU", "BrakePedalPos"], inplace=True)
        device_df.sort_values(by='IST', inplace=True)

        # Identify hard stop events (speed becomes 0 from a non-zero value)
        hard_stop_mask = (device_df['Vehicle_speed_VCU'] == 0.0) & (device_df['Vehicle_speed_VCU'].shift(1) > 0.0)
        hard_stop_events = device_df[hard_stop_mask].copy()

        if hard_stop_events.empty:
            continue
        
        # Extract and aggregate the data for each filtered event
        for _, event_row in hard_stop_events.iterrows():
            end_time = event_row['IST']
            search_start_time = end_time - pd.Timedelta(seconds=search_window_seconds)

            search_segment = device_df[(device_df['IST'] >= search_start_time) & (device_df['IST'] <= end_time)].copy()

            first_brake_press = search_segment[search_segment['BrakePedalPos'] > 0.0].head(1)

            if not first_brake_press.empty:
                start_time = first_brake_press.iloc[0]['IST']
                event_segment = device_df[(device_df['IST'] >= start_time) & (device_df['IST'] <= end_time)].copy()

                if not event_segment.empty and event_segment['Vehicle_speed_VCU'].max() >= top_speed:
                    all_event_data.append(event_segment)

    if not all_event_data:
        return pd.DataFrame() # Return an empty DataFrame if no events are found

    # Define constants for the kgf calculation
    BUS_MASS_KG = 13500  # 13.5 tonnes * 1000 kg/tonne
    G_ACCELERATION = 9.80665 # Standard acceleration due to gravity

    table_data = []
    
    for i, event_group in enumerate(all_event_data):
        start_time = event_group['IST'].iloc[0]
        end_time = event_group['IST'].iloc[-1]
        start_velocity = event_group['Vehicle_speed_VCU'].iloc[0]
        peak_velocity = event_group['Vehicle_speed_VCU'].max()
        max_brake_pedal_pos = event_group['BrakePedalPos'].max()
        avg_brake_pedal_pos = event_group['BrakePedalPos'].mean()
        
        event_group.loc[:, 'speed_mps'] = event_group['Vehicle_speed_VCU'] * (1000 / 3600)
        time_diffs_sec = event_group['IST'].diff().dt.total_seconds().fillna(0)
        distance_covered_m = (event_group['speed_mps'] * time_diffs_sec).sum()
        total_time_s = (end_time - start_time).total_seconds()
        
        if total_time_s > 0:
            avg_deceleration = (peak_velocity * 1000/3600) / total_time_s
        else:
            avg_deceleration = 0
            
        braking_force_kgf = (BUS_MASS_KG * avg_deceleration) / G_ACCELERATION
        
        table_data.append({
            'vehicle_id': device_id,
            'start': start_time.strftime('%d/%m/%y %H:%M:%S'),
            'end': end_time.strftime('%d/%m/%y %H:%M:%S'),
            'max_bpp': f"{max_brake_pedal_pos:.2f}",
            'avg_bpp': f"{avg_brake_pedal_pos:.2f}",
            'ttl_dist_m': f"{distance_covered_m:.2f}",
            'start_vel': f"{start_velocity:.2f}",
            'peak_vel': f"{peak_velocity:.2f}",
            'avg_decel_mps2': f"{avg_deceleration:.2f}",
            'braking_force_kgf': f"{braking_force_kgf:.2f}"
        })

    results_df = pd.DataFrame(table_data)
    results_df['start'] = pd.to_datetime(results_df['start'], format='%d/%m/%y %H:%M:%S')
    results_df['end'] = pd.to_datetime(results_df['end'], format='%d/%m/%y %H:%M:%S')

    return results_df

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


def write_df_to_iceberg(
    df: pd.DataFrame,
    table: str = REPORT_TABLE,
    s3_location: str = REPORT_S3_LOCATION,
    batch_size: int = 5000,
):
    print("💾 [4/5] STEP 4a: Preparing to write results to Iceberg...")
    cur = conn.cursor()
    fq_table = _qualify_and_quote(table)

    if df is None or df.empty:
        cols = list(df.columns) if df is not None else []
        if cols:
            column_defs = [f'{_quote_ident(col)} {_trino_type_from_series(df[col])}' for col in cols]
        else:
            column_defs = ['"id" varchar']
    else:
        column_defs = [f'{_quote_ident(col)} {_trino_type_from_series(df[col])}' for col in df.columns]

    cols_sql = ",\n        ".join(column_defs)

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

def drop_table(table: str = REPORT_TABLE):
    print(f"🗑️ [X] Dropping table if exists: {table} ...")
    cursor = conn.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        print(f"✅ [X] Table {table} dropped successfully.")
    except Exception as e:
        print(f"❌ [X] Failed to drop table {table}: {e}")
    finally:
        cursor.close()

# --------------------
# Step 4: Run for a single day
# --------------------
# date_str = "2025-08-31"
# print(f"▶️ [0/5] Starting job for {date_str}")

# df = fetch_data_for_day(conn, date_str)
# df = process_soc_charging_data(df)
# write_df_to_iceberg(df)
# drop_table()

from datetime import date, timedelta

# START_DATE = date(2025, 8, 1)
# YESTERDAY = date.today() - timedelta(days=1)

# d = START_DATE
# idx = 1
# while d <= YESTERDAY:
#     date_str = d.isoformat()
#     print(f"▶️ [{idx}] Starting job for {date_str}")

#     df = fetch_data_for_day(conn, date_str)
#     df = process_soc_charging_data(df)
#     write_df_to_iceberg(df)

#     d += timedelta(days=1)
#     idx += 1


yesterday = date.today() - timedelta(days=1)
date_str = yesterday.isoformat()
df = fetch_data_for_day(conn, date_str)
df = perform_braking_analysis(df)
write_df_to_iceberg(df)


# --------------------
# Step 5: Close connection
# --------------------
print("🔒 [5/5] STEP 5: Closing Trino connection...")
conn.close()
print("✅ [5/5] STEP 5: Connection closed. Job complete.")