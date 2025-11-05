import trino
import pandas as pd
import io
import boto3
from datetime import datetime, date, timedelta
from itertools import islice
import numpy as np


# ---- reporting config (edit ONLY this) ----
TABLE_NAME = "energy_stats_report"   # <— change only this

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
            "id","timestamp","BAT_SOC", "Bat_Voltage","Total_Battery_Current",
            "GUN_Connection_Status","OdoMeterReading", "Gear_Position", "Vehiclereadycondition"
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
# Step 3: Function to process vehicle energy stats
# --------------------
def analyze_vehicle_energy_stats(df:pd.DataFrame):
    """
    Performs a combined analysis on an electric bus dataset that may contain
    data from multiple vehicles. The function calculates daily mileage, driving
    energy consumption, regenerative braking energy, and idling energy for
    each vehicle.

    Args:
        file_path (str): The path to the input CSV file.
    """
    try:
        # Load the dataset
        # df = pd.read_csv(file_path)

        # --- General Data Preparation and Filtering ---
        # Ignore data points with extreme current values.
        df = df[df['Total_Battery_Current'].abs() <= 2500].copy()
                
        # Drop rows with missing data in key columns, including IST.
        df.loc[:, 'IST'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df.dropna(subset=['OdoMeterReading', 'Gear_Position', 'Vehiclereadycondition', 'Total_Battery_Current', 'Bat_Voltage','IST'], inplace=True)
        
        # Sort data by vehicle ID and timestamp to ensure correct sequential calculations.
        df.sort_values(by=['id', 'IST'], inplace=True)
        
        # Extract the date for daily grouping
        df['Date'] = df['IST'].dt.date

        # Get a list of unique vehicle IDs to iterate through
        unique_ids = df['id'].unique()
        all_daily_stats = []

        print("\n--- Daily Vehicle Statistics (Driving, Regenerative Braking, and Idling) ---")
        
        # Iterate over each unique vehicle ID
        for vehicle_id in unique_ids:
            vehicle_df = df[df['id'] == vehicle_id].copy()

            # Check if the grouped DataFrame is empty
            if vehicle_df.empty:
                print(f"No charging events were detected for device {vehicle_id}.")
                continue  
                            
            # Calculate the time difference between consecutive data points for the current vehicle.
            vehicle_df['time_diff_seconds'] = vehicle_df['IST'].diff().dt.total_seconds().fillna(0)
            
            # Filter out records where time difference is zero or negative.
            vehicle_df = vehicle_df[vehicle_df['time_diff_seconds'] > 0]
            
            # Calculate the power in kW
            vehicle_df['power_kW'] = (vehicle_df['Bat_Voltage'] * vehicle_df['Total_Battery_Current']) / 1000

            # --- Analysis 1: Daily Driving and Regenerative Braking Statistics ---
            driving_df = vehicle_df[vehicle_df['Gear_Position'] == 2.0].copy()

            # Calculate energy consumption (positive power) and regenerative braking energy (negative power)
            driving_df['energy_consumption_kwh'] = driving_df.apply(
                lambda row: row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] > 0 else 0, axis=1)
            driving_df['regen_energy_kwh'] = driving_df.apply(
                lambda row: -row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] < 0 else 0, axis=1)
            
            # Calculate incremental mileage for each row.
            driving_df['mileage_increment'] = driving_df['OdoMeterReading'].diff().fillna(0)
            
            # Filter out unrealistic mileage jumps (e.g., > 10 km in a single time step)
            driving_df = driving_df[driving_df['mileage_increment'] <= 10]

            if not driving_df.empty:
                grouped_by_day_driving = driving_df.groupby(['id', 'Date'])
                daily_driving_mileage = grouped_by_day_driving['mileage_increment'].sum()
                daily_driving_energy = grouped_by_day_driving['energy_consumption_kwh'].sum()
                daily_regen_energy = grouped_by_day_driving['regen_energy_kwh'].sum()

                daily_stats_df = pd.DataFrame({
                    'Daily Mileage (km)': round(daily_driving_mileage, 2),
                    'Daily Driving Energy Consumed (kWh)': round(daily_driving_energy, 2),
                    'Daily Regenerative Braking Energy (kWh)': round(daily_regen_energy, 2)
                }).reset_index()
            else:
                daily_stats_df = pd.DataFrame()

            # --- Analysis 2: Daily Idling Energy Consumption ---
            # Filter for idling periods using 'Gear_Position' == 0 (Neutral) and 'Vehiclereadycondition' == 1
            stationary_df = vehicle_df[(vehicle_df['Gear_Position'] == 0.0) & 
                                       (vehicle_df['Vehiclereadycondition'] == 1.0)].copy()
            
            if not stationary_df.empty:
                stationary_df['energy_kwh'] = stationary_df.apply(
                    lambda row: row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] > 0 else 0, axis=1)
                grouped_by_day_stationary = stationary_df.groupby(['id', 'Date'])
                daily_idling_energy = grouped_by_day_stationary['energy_kwh'].sum()
                
                daily_idling_df = daily_idling_energy.reset_index(name='Daily Idling Energy Consumed (kWh)')
                daily_idling_df = daily_idling_df.round({'Daily Idling Energy Consumed (kWh)': 2})  

                if not daily_stats_df.empty:
                    daily_stats_df = pd.merge(daily_stats_df, daily_idling_df, on=['id', 'Date'], how='left')
                    # Corrected line to avoid FutureWarning
                    daily_stats_df['Daily Idling Energy Consumed (kWh)'] = daily_stats_df['Daily Idling Energy Consumed (kWh)'].fillna(0)
                else:
                    daily_stats_df = daily_idling_df
            
            if not daily_stats_df.empty:
                daily_stats_df['Net Energy Consumed (kWh)'] = daily_stats_df['Daily Driving Energy Consumed (kWh)'] - daily_stats_df['Daily Regenerative Braking Energy (kWh)']
                daily_stats_df['Net Consumption Rate (kWh/km)'] = (daily_stats_df['Net Energy Consumed (kWh)'] / daily_stats_df['Daily Mileage (km)']).round(2)
                all_daily_stats.append(daily_stats_df)
        
        if all_daily_stats:
            final_df = pd.concat(all_daily_stats, ignore_index=True)
            #'Daily Mileage (km)', 'Daily Idling Energy Consumed (kWh)', 'Net Energy Consumed (kWh)','Net Consumption Rate (kWh/km)','Daily Regenerative Braking Energy (kWh)'
            final_df.rename(columns={'id': 'vehicle_id', 'Daily Mileage (km)': 'distance_travelled','Daily Idling Energy Consumed (kWh)': 'idling_energy_consumed_kwh','Net Energy Consumed (kWh)': 'net_energy_consumed_kwh','Net Consumption Rate (kWh/km)': 'net_consumption_rate_kwh_per_km','Daily Regenerative Braking Energy (kWh)': 'regen_energy_consumed_kwh'}, inplace=True)
            return final_df
        else:
            print("No data found for driving or idling periods.")
            return pd.DataFrame()

    except FileNotFoundError:
        # print(f"Error: The dataset was not found.")
        print("No entries were found in the dataframe.")
        # Return empty DataFrame on error to prevent AttributeError
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Return empty DataFrame on error to prevent AttributeError
        return pd.DataFrame()


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


#################################################################################
### Execute single_day_data() for single day data ###
### Arguments: date_val -> yyyy,mm,dd eg. date(2025, 7, 24)
################################################################################## 
def single_day_data(date_val):
    date_str = date_val.isoformat()
    print(f"▶️ [0/5] Starting job for {date_str}")
    
    df = fetch_data_for_day(conn, date_str)
    df = analyze_vehicle_energy_stats(df)
    write_df_to_iceberg(df)


#################################################################################
### Execute recall_historical_data() for the first time to gather historical data ###
### Arguments: date_val -> yyyy,mm,dd eg. date(2025, 7, 24)
#################################################################################
def recall_historical_data(date_val):
    drop_table()
    
    START_DATE = date_val
    YESTERDAY = date.today() - timedelta(days=1)
    
    d = START_DATE
    idx = 1
    while d <= YESTERDAY:
        date_str = d.isoformat()
        print(f"▶️ [{idx}] Starting job for {date_str}")
    
        df = fetch_data_for_day(conn, date_str)
        df = analyze_vehicle_energy_stats(df)
    
        d += timedelta(days=1)
        idx += 1    
        
        if df.empty:
            print(f"No charging events were detected for device {id}.")
            continue        
        else:
            write_df_to_iceberg(df)


#################################################################################
### Execute the following lines to execute the daily fetch and process tasks ###
#################################################################################
def daily_routine_acquisition():
    yesterday = date.today() - timedelta(days=1)
    date_str = yesterday.isoformat()
    df = fetch_data_for_day(conn, date_str)
    df = analyze_vehicle_energy_stats(df)
    write_df_to_iceberg(df)


recall_historical_data(date(2025, 7, 24))

# --------------------
# Step 5: Close connection
# --------------------
print("🔒 [5/5] STEP 5: Closing Trino connection...")
conn.close()
print("✅ [5/5] STEP 5: Connection closed. Job complete.")