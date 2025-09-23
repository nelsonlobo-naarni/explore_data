import trino
import pandas as pd
import io
import boto3
from datetime import datetime
from itertools import islice
import numpy as np


# ---- reporting config (edit ONLY this) ----
TABLE_NAME = "soc_units_report"   # <— change only this

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
            "id","timestamp","BAT_SOC", "Bat_Voltage","Total_Battery_Current","GUN_Connection_Status",
            "Chargingcontactor1positive","Chargingcontactor1negative",
            "Chargingcontactor2positive","Chargingcontactor2negative"
        FROM
            facts_prod.can_parsed_output_100
        WHERE
            dt = DATE '{date_str}'
    """)

    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    print(f"📊 [2/5] STEP 2c: Rows fetched for {date_str}: {len(df)}")
    
    return df

# --------------------
# Step 3: Function to process SOC and charging data
# --------------------
def process_soc_charging_data(df: pd.DataFrame):
    """
    Generates a summary, and performs outlier analysis for charging events
    across multiple device IDs.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing raw data for one or more devices.
        
    Returns:
        pd.DataFrame: A DataFrame summarizing charging events for each device.
    """

    # Create a copy of the DataFrame to avoid modifying a slice.
    df = df.copy()    
    
    if df.empty or 'id' not in df.columns:
        print("Input DataFrame is empty or does not contain an 'id' column.")
        return pd.DataFrame()
        
    device_ids = df['id'].unique().tolist()
    all_summary_data = []

    for device_id in device_ids:
        device_df = df[df['id'] == device_id].copy()

        # Check if the grouped DataFrame is empty
        if device_df.empty:
            print(f"No charging events were detected for device {device_id}.")
            continue        
        
        if 'timestamp' in device_df.columns:
            device_df.loc[:, 'IST'] = pd.to_datetime(device_df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        elif 'IST' in device_df.columns:
            device_df.loc[:, 'IST'] = pd.to_datetime(device_df['IST'])
        else:
            print(f"Neither 'timestamp' nor 'IST' column found for device {device_id}. Cannot proceed.")
            continue
            
        device_df.sort_values(by='IST', inplace=True)
        device_df.reset_index(drop=True, inplace=True)

        current_threshold = 3200
        device_df = device_df[(device_df['Total_Battery_Current'].abs() != current_threshold)].copy()
        for col in ['BAT_SOC', 'Bat_Voltage']:
            device_df.loc[:, col] = device_df[col].replace(0.0, np.nan).ffill().bfill()
            
        # NEW LOGIC: Fill missing current data with 0 to be included in idle time calculation
        device_df.loc[:, 'Total_Battery_Current'] = device_df['Total_Battery_Current'].fillna(0)

        device_df.dropna(subset=['BAT_SOC', 'Bat_Voltage', 'Chargingcontactor1positive'], inplace=True)
        
        charging_start_indices = device_df[device_df['Chargingcontactor1positive'].diff() == 1].index.tolist()
        charging_end_indices = device_df[device_df['Chargingcontactor1positive'].diff() == -1].index.tolist()

        if device_df.iloc[0]['Chargingcontactor1positive'] == 1 and (not charging_start_indices or charging_start_indices[0] != device_df.index[0]):
            charging_start_indices.insert(0, device_df.index[0])

        if device_df.iloc[-1]['Chargingcontactor1positive'] == 1 and len(charging_start_indices) > len(charging_end_indices):
            charging_end_indices.append(device_df.index[-1])
            
        merged_events = []
        if len(charging_start_indices) > 0 and len(charging_end_indices) > 0:
            current_start_index = charging_start_indices[0]
            current_end_index = charging_end_indices[0]
            
            time_threshold_seconds = 5 * 60
            soc_threshold = 1.0
            
            for i in range(1, len(charging_start_indices)):
                prev_end_index = charging_end_indices[i-1]
                current_start_index_next = charging_start_indices[i]
                
                prev_end_time = device_df.loc[prev_end_index, 'IST']
                current_start_time = device_df.loc[current_start_index_next, 'IST']
                prev_end_soc = device_df.loc[prev_end_index, 'BAT_SOC']
                current_start_soc = device_df.loc[current_start_index_next, 'BAT_SOC']

                time_diff = (current_start_time - prev_end_time).total_seconds()
                soc_diff = abs(current_start_soc - prev_end_soc)
                
                if (time_diff <= time_threshold_seconds and soc_diff <= soc_threshold) or (time_diff <= 60):
                    current_end_index = charging_end_indices[i]
                else:
                    merged_events.append((current_start_index, current_end_index))
                    current_start_index = charging_start_indices[i]
                    current_end_index = charging_end_indices[i]
            
            merged_events.append((current_start_index, current_end_index))
        else:
            print(f"No charging events were detected for device {device_id}.")
            continue

        summary_data_device = []
        BATTERY_CAPACITY_KWH = 423
        
        for start_index, end_index in merged_events:
            event_df = device_df.loc[start_index:end_index].copy()

            # Check if the event_df is empty before processing.
            if event_df.empty:
                print(f"Warning: Empty event data found for device {device_id}. Skipping.")
                continue            
                
            charging_periods = event_df[event_df['GUN_Connection_Status'] == 1].copy()
            
            total_duration = 0
            if not charging_periods.empty:
                charging_periods.loc[:, 'time_diff'] = charging_periods['IST'].diff().dt.total_seconds().fillna(0)
                total_duration = charging_periods['time_diff'].sum()

            start_row = event_df.iloc[0].copy()
            end_row = event_df.iloc[-1].copy()
            
            energy_Wh = 0
            if not charging_periods.empty:
                charging_periods.loc[:, 'power_W'] = charging_periods['Bat_Voltage'] * charging_periods['Total_Battery_Current'].abs()
                energy_Wh = np.trapz(charging_periods['power_W'], x=charging_periods['IST'].astype(np.int64) / 10**9) / 3600
            
            total_kwh_consumed_tpc = energy_Wh / 1000

            total_kwh_consumed_soc = (end_row['BAT_SOC'] - start_row['BAT_SOC']) * BATTERY_CAPACITY_KWH / 100
            total_kwh_consumed_soc = abs(total_kwh_consumed_soc)

            percent_diff = 0
            if total_kwh_consumed_tpc + total_kwh_consumed_soc != 0:
                percent_diff = (abs(total_kwh_consumed_tpc - total_kwh_consumed_soc) / 
                                ((total_kwh_consumed_tpc + total_kwh_consumed_soc) / 2)) * 100
            
            summary_data_device.append({
                'vehicle_id': device_id,
                'start_time': start_row['IST'],
                'end_time': end_row['IST'],
                'charge_dur': total_duration,
                'soc_start': start_row['BAT_SOC'],
                'soc_end': end_row['BAT_SOC'],
                'tpc_kwh': total_kwh_consumed_tpc,
                'soc_kwh': total_kwh_consumed_soc,
                'diff_kw_percent': percent_diff
            })
        
        all_summary_data.extend(summary_data_device)
            
    return pd.DataFrame(all_summary_data)


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

from datetime import date, timedelta

#################################################################################
### Execute the following lines for the first time to gather historical data ###
#################################################################################
drop_table()

START_DATE = date(2025, 8, 1)
YESTERDAY = date.today() - timedelta(days=1)

d = START_DATE
idx = 1
while d <= YESTERDAY:
    date_str = d.isoformat()
    print(f"▶️ [{idx}] Starting job for {date_str}")

    df = fetch_data_for_day(conn, date_str)
    df = process_soc_charging_data(df)

    d += timedelta(days=1)
    idx += 1    
    
    if df.empty:
        print(f"No charging events were detected for device {device_id}.")
        continue        
    else:
        write_df_to_iceberg(df)


#################################################################################
### Execute the following lines to execute the daily fetch and process tasks ###
#################################################################################
# yesterday = date.today() - timedelta(days=1)
# date_str = yesterday.isoformat()
# df = fetch_data_for_day(conn, date_str)
# df = process_soc_charging_data(df)
# write_df_to_iceberg(df)


# --------------------
# Step 5: Close connection
# --------------------
print("🔒 [5/5] STEP 5: Closing Trino connection...")
conn.close()
print("✅ [5/5] STEP 5: Connection closed. Job complete.")