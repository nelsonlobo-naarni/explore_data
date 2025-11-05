# %%
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import platform
import logging
import argparse
import trino
import io
import boto3
from itertools import islice
from datetime import datetime, date, timedelta
import pendulum
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# Add parent directory to path
repo_path = '/Users/apple/Documents/naarni/repo/dview-naarni-data-platform'
sys.path.append(os.path.join(repo_path, 'tasks'))

# Import necessary files and its respective functions
from common.db_operations import connect_to_trino, fetch_data_for_day, write_df_to_iceberg,drop_table,execute_query
from common.optimizer_logic import optimize_dataframe_memory

# Import business logic functions
from biz_logic.energy_mileage.energy_mileage_daily_v0 import energy_mileage_stats ,impute_odometer_readings

from biz_logic.energy_consumption.energy_consumption_report import energy_consumption_stats

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Print the Python version being used
print(f"Using Python version: {platform.python_version()}")

# %%
# ---- reporting config (edit ONLY this) ----
TABLE_NAME = "can_parsed_output_100"   # <— change only this

# derived (don’t edit)
REPORT_TABLE = f"adhoc.facts_prod.{TABLE_NAME}"
REPORT_S3_LOCATION = f"s3a://naarni-data-lake/aqua/warehouse/facts_prod.db/{TABLE_NAME}/"

# %%
date_str = "2025-10-27"  # Date for which data is to be processed

# Parse the date string as a date object
target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

# Create datetime objects for the start and end of the day in IST
ist_start = datetime.combine(target_date, datetime.min.time())
ist_end = ist_start + timedelta(days=1)

# Convert IST to UTC for the database query
# IST is UTC+5:30, so we subtract 5 hours and 30 minutes
utc_start = ist_start - timedelta(hours=5, minutes=30)
utc_end = ist_end - timedelta(hours=5, minutes=30)

logging.info(f"🔍 Query window (UTC): {utc_start} → {utc_end}")
logging.info(f"🔍 Query window (IST): {ist_start} → {ist_end}")

# %% [markdown]
# ### AC Fault Code:
# 1. No_Fault
# 2. Low_Voltage
# 3. Outside_Temp_Sensor_Fault
# 4. High_Voltage
# 5. Exhaust_Temp_Protection
# 6. Eva_Temp_Sesnor_Fault
# 7. AC Communication Fail
# 
# ### AC Status:
# 1. Start
# 2. Stop
# 
# ### TMS Fault Code:
# 1. No Fault
# 2. Water_Sensor_Failure
# 3. Water_Pump_Failure
# 4. Water_IN_Sensor_Failure
# 5. Exhaust_Temp_Protection
# 6. Low_Water_Level_Alarm
# 7. LV Undervoltage
# 
# ### TMS Working Mode:
# 1. Charging_Cooling
# 2. Fast_Discharge_Cooling
# 3. Self_Circulation
# 4. Low_Coolant
# 5. Off
# 
# ### B2T TMS Control Cmd:
# 1. Charging_Cooling
# 2. Fast_Discharge_Cooling
# 3. Self_Circulation
# 4. Off

# %%
# query = f"""
# select 
#     id,date(timestamp + interval '5:30' hour to minute) as dateval,count(*) 
# FROM 
#     facts_prod.can_parsed_output_100
# where 
#     id in ('6') and 
#     timestamp >= TIMESTAMP '{utc_start.strftime('%Y-%m-%d %H:%M:%S')}' and
#     timestamp < TIMESTAMP '{utc_end.strftime('%Y-%m-%d %H:%M:%S')}'
# group by 1,2"""

# conn = connect_to_trino(host="analytics.internal.naarni.com",port=443,user="admin",catalog="adhoc",schema="default")

# df = execute_query(conn, f"SELECT * FROM {REPORT_TABLE} LIMIT 5", return_results=True)
# display(df)

# %%
def fetch_battery_data(start_date, end_date, vehicle_ids):
    """
    Fetch raw battery data from the database for the specified date range and vehicle IDs.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        vehicle_ids: List of vehicle IDs
        
    Returns:
        Tuple of (df_cpo100, df_can_ac) containing raw data from both tables
    """
    logging.info(f"Fetching raw battery data from {start_date} to {end_date} for vehicles {vehicle_ids}")
    
    # Format vehicle IDs for the query
    vehicle_ids_str = ', '.join([f"'{vid}'" for vid in vehicle_ids])
    
    # Connect to Trino
    conn = connect_to_trino(host="analytics.internal.naarni.com", port=443, user="admin", 
                           catalog="adhoc", schema="default")

    # Query for cpo100 data
    cpo100_query = f"""
    SELECT 
        id, CAST(timestamp AS TIMESTAMP) AT TIME ZONE 'Asia/Kolkata' as timestamp, dt, 
        batterycoolingstate, batterycoolanttemperature,
        temperaturedifferencealarm, chargingcurrentalarm, dischargecurrentalarm,
        vehiclereadycondition, gun_connection_status, ignitionstatus,
        vehicle_speed_vcu,gear_position,bat_soc,
        pack1_cellmax_temperature, pack1_maxtemperature_cell_number, pack1_cell_min_temperature, pack1_celltemperature_cellnumber,
        bat_voltage,cellmax_voltagecellnumber,cell_max_voltage,cellminvoltagecellnumber,cell_min_voltage,
        lowpressureoilpumpfaultcode,bms_fault_code,vcu_fault_code,fiveinone_faultcode
    FROM 
        facts_prod.can_parsed_output_100
    WHERE 
        id IN ({vehicle_ids_str})
        AND DATE(timestamp AT TIME ZONE 'Asia/Kolkata') >= DATE('{start_date}')
        AND DATE(timestamp AT TIME ZONE 'Asia/Kolkata') <= DATE('{end_date}')
    """

    # Query for can_ac data
    can_ac_query = f"""
    SELECT 
        id, CAST(timestamp AS TIMESTAMP) AT TIME ZONE 'Asia/Kolkata' as timestamp, date,
        b2t_tms_control_cmd, b2t_set_water_out_temp, b2t_battery_min_temp, b2t_battery_max_temp,
        tms_working_mode, tms_fault_code, ac_fault_code, coolant_out_temp, coolant_in_temp,
        comp_status,comp_target_hz as comp_target_freq,
        comp_running_frequency as comp_running_freq,
        v2t_vehicle_coolant_low, comp_current, hv_voltage
    FROM 
        facts_prod.can_output_ac
    WHERE
        id IN ({vehicle_ids_str})
        AND DATE(timestamp AT TIME ZONE 'Asia/Kolkata')  >= DATE('{start_date}')
        AND DATE(timestamp AT TIME ZONE 'Asia/Kolkata') <= DATE('{end_date}')
        and b2t_battery_min_temp > 0
        and b2t_battery_max_temp > 0
        and coolant_in_temp > 0
    """

    # Execute queries and fetch data
    cur = conn.cursor()

    # Fetch cpo100 data
    cur.execute(cpo100_query)
    cpo100_columns = [desc[0] for desc in cur.description]
    cpo100_rows = cur.fetchall()
    df_cpo100 = pd.DataFrame(cpo100_rows, columns=cpo100_columns)

    # Fetch can_ac data
    cur.execute(can_ac_query)
    can_ac_columns = [desc[0] for desc in cur.description]
    can_ac_rows = cur.fetchall()
    df_can_ac = pd.DataFrame(can_ac_rows, columns=can_ac_columns)

    logging.info(f"Done Fetching data.")
    logging.info(f"Retrieved {len(df_cpo100)} cpo100 records and {len(df_can_ac)} can_ac records")
    
    # Close connections
    cur.close()
    conn.close()
    
    return df_cpo100, df_can_ac

# %%
def process_battery_data(df_cpo100, df_can_ac):
    """
    Process raw battery data to create the final merged dataset.
    
    Args:
        df_cpo100: Raw DataFrame from can_parsed_output_100 table
        df_can_ac: Raw DataFrame from can_output_ac table
        
    Returns:
        Tuple of (result_df, df_cpo100_processed, df_can_ac_processed)
        where result_df is the final merged dataset and the others are processed versions
    """
    logging.info("Processing battery data")
    
    # Convert timestamp columns to datetime if they aren't already
    df_cpo100['timestamp'] = pd.to_datetime(df_cpo100['timestamp'])
    df_can_ac['timestamp'] = pd.to_datetime(df_can_ac['timestamp'])
    
    # Perform minute truncation in pandas
    df_cpo100['ts_mins'] = df_cpo100['timestamp'].dt.floor('min')
    df_can_ac['ts_mins_cac'] = df_can_ac['timestamp'].dt.floor('min')

    # Add row numbers (equivalent to SQL row_number() window function)
    # For cpo100
    df_cpo100 = df_cpo100.sort_values(['id', 'ts_mins', 'timestamp'])
    df_cpo100['cpo_rn'] = df_cpo100.groupby(['id', 'ts_mins']).cumcount() + 1

    # For can_ac
    df_can_ac = df_can_ac.sort_values(['id', 'ts_mins_cac', 'timestamp'])
    df_can_ac['cac_rn'] = df_can_ac.groupby(['id', 'ts_mins_cac']).cumcount() + 1

    # Perform the join in pandas (equivalent to SQL right join)
    merged_df = pd.merge(
        df_can_ac,
        df_cpo100,
        left_on=['id', 'ts_mins_cac', 'cac_rn'],
        right_on=['id', 'ts_mins', 'cpo_rn'],
        how='left',
        suffixes=('_can_ac','_cpo100')
    )

    # Select only the columns we need
    result_columns = []

    # Add id column
    if 'id' in merged_df.columns:
        result_columns.append('id')
    elif 'id_cpo100' in merged_df.columns:
        result_columns.append('id_cpo100')

    # Add other columns from cpo100
    cpo100_cols = ['dt', 'ignitionstatus', 'vehiclereadycondition', 'gun_connection_status',
                   'vehicle_speed_vcu','gear_position','bat_soc',
                   'pack1_cellmax_temperature', 'pack1_cell_min_temperature',
                   'pack1_maxtemperature_cell_number','pack1_celltemperature_cellnumber',
                   'bat_voltage','cellmax_voltagecellnumber','cell_max_voltage','cellminvoltagecellnumber','cell_min_voltage']
    
    for col in cpo100_cols:
        if col in merged_df.columns:
            result_columns.append(col)
        elif f'{col}_cpo100' in merged_df.columns:
            result_columns.append(f'{col}_cpo100')

    # Add timestamp (from cpo100)
    if 'timestamp_cpo100' in merged_df.columns:
        result_columns.append('timestamp_cpo100')
    elif 'timestamp' in merged_df.columns:
        result_columns.append('timestamp')

    # Add ts_mins (from cpo100)
    if 'ts_mins_cpo100' in merged_df.columns:
        result_columns.append('ts_mins_cpo100')
    elif 'ts_mins' in merged_df.columns:
        result_columns.append('ts_mins')

    # Add columns from can_ac
    can_ac_cols = ['b2t_tms_control_cmd', 'b2t_set_water_out_temp', 
                  'b2t_battery_min_temp', 'b2t_battery_max_temp', 'tms_working_mode',
                  'coolant_out_temp', 'coolant_in_temp', 'hv_voltage',
                  'comp_status','tms_fault_code','ac_fault_code']

    for col in can_ac_cols:
        if col in merged_df.columns:
            result_columns.append(col)
        elif f'{col}_can_ac' in merged_df.columns:
            result_columns.append(f'{col}_can_ac')

    # Create result DataFrame with selected columns
    result_df = merged_df[result_columns].copy()

    # Rename columns to match original query output
    rename_dict = {}
    if 'id_cpo100' in result_df.columns:
        rename_dict['id_cpo100'] = 'id'
    if 'timestamp_cpo100' in result_df.columns:
        rename_dict['timestamp_cpo100'] = 'timestamp'
    if 'ts_mins_cpo100' in result_df.columns:
        rename_dict['ts_mins_cpo100'] = 'ts_mins'

    # Rename any columns with _can_ac suffix
    for col in result_df.columns:
        if col.endswith('_can_ac'):
            rename_dict[col] = col[:-7]  # Remove the '_can_ac' suffix

    result_df = result_df.rename(columns=rename_dict)

    # Sort the result as in the original query
    result_df = result_df.sort_values(['id', 'timestamp'])

    logging.info(f"Processed {len(result_df)} battery data records")
    return result_df, df_cpo100, df_can_ac

# %%
def get_all_battery_data(start_date, end_date, vehicle_ids):
    """
    Fetch and process all battery data for the specified date range and vehicle IDs.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        vehicle_ids: List of vehicle IDs
        
    Returns:
        DataFrame with battery data
    """
    # Fetch raw data
    df_cpo100, df_can_ac = fetch_battery_data(start_date, end_date, vehicle_ids)
    
    # Process the data
    result_df, df_cpo100_processed, df_can_ac_processed = process_battery_data(df_cpo100, df_can_ac)
    
    return result_df, df_cpo100_processed, df_can_ac_processed

# %%
df, df_cpo100, df_can_ac = get_all_battery_data('2025-10-01', '2025-10-02', ['6'])

# %%
display(df.head(10))

# %%
# display(df_cpo100.head())
# display(df_cpo100.lowpressureoilpumpfaultcode.value_counts())
# display(df_cpo100.bms_fault_code.value_counts())
# display(df_cpo100.fiveinone_faultcode.value_counts())

# %%
def safe_get(df, col):
    return df[col] if col in df.columns else pd.Series(dtype=float)

def safe_median(series):
    """Return median safely without warnings on empty or all-NaN slices."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    return round(s.median(), 3) if len(s) > 0 else np.nan

def safe_mean(series):
    """Return mean safely without warnings on empty or all-NaN slices."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    return round(s.mean(), 3) if len(s) > 0 else np.nan


def scalar_mode(s: pd.Series):
    """Return a single scalar mode value from a pandas Series."""
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan


def process_tms_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify transitions in TMS state parameters and summarize continuous states
    with temperature and cell voltage details.
    """
    # --- Preprocess timestamp ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- Clean compressor status ---
    df["comp_status"] = (df["comp_status"].ffill().replace({"nan": np.nan}))
    # ensure first value isn't NaN to avoid false trigger at start
    if pd.isna(df.loc[0, "comp_status"]):
        df.loc[0, "comp_status"] = "Off"  # or whichever baseline makes sense

    # --- Clean AC fault code ---
    df["ac_fault_code"] = (df["ac_fault_code"].ffill().replace({"nan": np.nan}))
    # ensure first value isn't NaN to avoid false trigger at start
    if pd.isna(df.loc[0, "ac_fault_code"]):
        df.loc[0, "ac_fault_code"] = "No Fault"  # or whichever baseline makes sense

    # --- Define key columns ---
    transition_cols = ["b2t_tms_control_cmd", "tms_working_mode", "comp_status"]
    temp_cols = ["b2t_set_water_out_temp", "b2t_battery_max_temp", "coolant_out_temp", "coolant_in_temp"]
    volt_cols = ["bat_voltage","hv_voltage","cellmax_voltagecellnumber", "cell_max_voltage", "cellminvoltagecellnumber", "cell_min_voltage"]

    # --- Identify transitions ---
    change_mask = (df[transition_cols] != df[transition_cols].shift()).any(axis=1)
    df["state_group"] = change_mask.cumsum()

    # --- Summarize each state group ---
    summary_list = []
    for group_id, group in df.groupby("state_group"):
        if group.empty:
            continue

        start_row = group.iloc[0]
        end_row = group.iloc[-1]

        # Determine vehicle mode
        ignition = start_row.get("ignitionstatus", np.nan)
        vehicle_ready = start_row.get("vehiclereadycondition", np.nan)
        gun_status = start_row.get("gun_connection_status", np.nan)
        if vehicle_ready == 1.0:
            mode = "Drive"
        elif gun_status == 1.0:
            mode = "Charging/Park"
        else:
            mode = "Idle/Other"

        # Build summary record
        record = {
            "start_time": start_row["timestamp"],
            "end_time": end_row["timestamp"],
            "duration_mins": round((end_row["timestamp"] - start_row["timestamp"]).total_seconds() / 60.0, 2),
            "mode": mode,
            "vehicle_speed_start": round(start_row.get("vehicle_speed_vcu", np.nan), 1),
            "ignitionstatus_start": start_row.get("ignitionstatus", np.nan),
            "vehilceready_start": start_row.get("vehiclereadycondition", np.nan),
            "gun_connection_status_start": start_row.get("gun_connection_status", np.nan),
            **{f"{col}_start": start_row[col] for col in temp_cols},
            **{f"{col}_end": end_row[col] for col in temp_cols},
            **{col: start_row[col] for col in transition_cols},
            # Temperature statistics
            "b2t_battery_max_temp_med": safe_median(group["b2t_battery_max_temp"]),
            "coolant_out_temp_med": safe_median(group["coolant_out_temp"]),
            "coolant_in_temp_med": safe_median(group["coolant_in_temp"]),

            # Temperature cell data
            "mintemp_cellnum_start": start_row.get("pack1_celltemperature_cellnumber", np.nan),
            "mintemp_cellnum_end": end_row.get("pack1_celltemperature_cellnumber", np.nan),
            "mintemp_cellnum_mode": scalar_mode(group["pack1_celltemperature_cellnumber"]),
            
            "maxtemp_cellnum_start": start_row.get("pack1_maxtemperature_cell_number", np.nan),
            "maxtemp_cellnum_end": end_row.get("pack1_maxtemperature_cell_number", np.nan),
            "maxtemp_cellnum_mode": scalar_mode(group["pack1_maxtemperature_cell_number"]),

            # Voltage cell data
            "minvolt_cellnum_start": start_row.get("cellminvoltagecellnumber", np.nan),
            "minvolt_cellnum_end": end_row.get("cellminvoltagecellnumber", np.nan),
            "minvolt_cellnum_mode": scalar_mode(safe_get(group, "cellminvoltagecellnumber")),

            "cell_min_voltage_start": start_row.get("cell_min_voltage", np.nan),
            "cell_min_voltage_end": end_row.get("cell_min_voltage", np.nan),
            "cell_min_voltage_med": safe_median(group["cell_min_voltage"]),                
            
            "maxvolt_cellnum_start": start_row.get("cellmax_voltagecellnumber", np.nan),
            "maxvolt_cellnum_end": end_row.get("cellmax_voltagecellnumber", np.nan),            
            "maxvolt_cellnum_mode": scalar_mode(safe_get(group, "cellmax_voltagecellnumber")),

            "cell_max_voltage_start": start_row.get("cell_max_voltage", np.nan),
            "cell_max_voltage_end": end_row.get("cell_max_voltage", np.nan),
            "cell_max_voltage_med": safe_median(group["cell_max_voltage"]),

            "bat_voltage_start": start_row.get("bat_voltage", np.nan),
            "bat_voltage_end": end_row.get("bat_voltage", np.nan),
            "bat_voltage_med": safe_median(group["bat_voltage"]),

            "bat_soc_start": start_row.get("bat_soc", np.nan),
            "bat_soc_end": end_row.get("bat_soc", np.nan),
            
            "hv_voltage_start": start_row.get("hv_voltage", np.nan),
        }

        summary_list.append(record)

    # --- Build final DataFrame ---
    state_summary = pd.DataFrame(summary_list)
    state_summary = state_summary[state_summary["duration_mins"] > 0]

    # --- Reorder columns for readability ---
    ordered_cols = [
        "start_time", "end_time", "duration_mins", "mode", "vehicle_speed_start",
        "vehilceready_start", "gun_connection_status_start","bat_soc_start","bat_soc_end",
        "b2t_battery_max_temp_start", "b2t_battery_max_temp_end", "b2t_battery_max_temp_med",
        "mintemp_cellnum_start", "mintemp_cellnum_end", "mintemp_cellnum_mode",
        "maxtemp_cellnum_start", "maxtemp_cellnum_end", "maxtemp_cellnum_mode",
        "coolant_out_temp_start", "coolant_out_temp_end", "coolant_out_temp_med",
        "coolant_in_temp_start", "coolant_in_temp_end", "coolant_in_temp_med",
        "cell_max_voltage_start", "cell_max_voltage_end","cell_max_voltage_med", 
        "cell_min_voltage_start", "cell_min_voltage_end","cell_min_voltage_med",
        "minvolt_cellnum_start", "minvolt_cellnum_end", "minvolt_cellnum_mode",
        "maxvolt_cellnum_start", "maxvolt_cellnum_end", "maxvolt_cellnum_mode",
        "bat_voltage_start", "bat_voltage_end", "bat_voltage_med","hv_voltage_start",
        "b2t_tms_control_cmd", "tms_working_mode", "comp_status"
    ]

    # --- Clean up integer-like columns ---
    int_like_cols = [
        "vehilceready_start", "gun_connection_status_start",
        "mintemp_cellnum_start", "mintemp_cellnum_end", "mintemp_cellnum_mode",
        "maxtemp_cellnum_start", "maxtemp_cellnum_end", "maxtemp_cellnum_mode",
        "minvolt_cellnum_start", "minvolt_cellnum_end", "minvolt_cellnum_mode",
        "maxvolt_cellnum_start", "maxvolt_cellnum_end", "maxvolt_cellnum_mode"
    ]

    for col in int_like_cols:
        if col in state_summary.columns:
            # Convert float values that are actually integers into proper integers
            state_summary[col] = (
                pd.to_numeric(state_summary[col], errors='coerce')
                .astype("Int64")  # Pandas nullable integer type (keeps NaN clean)
            )


    return state_summary.reindex(columns=ordered_cols)

# %%
# --- Example usage ---
state_summary = process_tms_transitions(df)
state_summary.head(10)

# %%
state_summary.to_csv('battery_temp_v2.csv', index=False)

# %%
# --- Clean compressor status ---
df["comp_status_clean"] = (df["comp_status"].ffill().replace({"nan": np.nan}))
# ensure first value isn't NaN to avoid false trigger at start
if pd.isna(df.loc[0, "comp_status_clean"]):
    df.loc[0, "comp_status_clean"] = "Off"  # or whichever baseline makes sense

# --- Clean AC fault code ---
df["ac_fault_code_clean"] = (df["ac_fault_code"].ffill().replace({"nan": np.nan}))
# ensure first value isn't NaN to avoid false trigger at start
if pd.isna(df.loc[0, "ac_fault_code_clean"]):
    df.loc[0, "ac_fault_code_clean"] = "No Fault"  # or whichever baseline makes sense



# Identify where transitions are detected
transition_cols = ["b2t_tms_control_cmd", "tms_working_mode", "tms_fault_code", "comp_status_clean", "ac_fault_code_clean"]
change_mask = (df[transition_cols] != df[transition_cols].shift())

# Add debugging info: which column changed
df["_change_cols"] = change_mask.apply(lambda row: ",".join(row.index[row]), axis=1)
transitions_debug = df.loc[change_mask.any(axis=1), ["timestamp", "_change_cols"] + transition_cols]

transitions_debug.head(10)


# %% [markdown]
# 1. b2t_battery_max_temp, mintemp_cellnum, maxtemp_cellnum, 
# 2. coolant_out_temp, coolant_in_temp,
# 3. cell_max_voltage, cell_min_voltage,
# 4. minvolt_cellnum, maxvolt_cellnum, 
# 5. b2t_tms_control_cmd, tms_working_mode, comp_status

# %%
# --- Copy base dataframe ---
df_summary = state_summary.copy()

# --- Basic prep ---
comp_norm = df_summary["comp_status"].astype(str).str.strip().str.lower()
df_summary["comp_status_on_time"] = np.where(comp_norm.eq("on"), df_summary["duration_mins"], 0.0)
df_summary["comp_status_off_time"] = np.where(comp_norm.eq("off"), df_summary["duration_mins"], 0.0)

# --- Rename key start/end attributes for clarity ---
rename_map = {
    "b2t_battery_max_temp_start": "batt_temp_start",
    "b2t_battery_max_temp_end": "batt_temp_end",
    "coolant_in_temp_start": "coolant_in_start",
    "coolant_in_temp_end": "coolant_in_end",
    "coolant_out_temp_start": "coolant_out_start",
    "coolant_out_temp_end": "coolant_out_end",
    "bat_soc_start": "soc_start",
    "bat_soc_end": "soc_end"
}
df_summary.rename(columns={k: v for k, v in rename_map.items() if k in df_summary.columns}, inplace=True)

# --- Helper for safe deltas ---
def safe_delta(start, end):
    if start in df_summary.columns and end in df_summary.columns:
        return df_summary[end] - df_summary[start]
    return np.nan

# --- Derived metrics ---
df_summary["batt_temp_delta"] = safe_delta("batt_temp_start", "batt_temp_end")
df_summary["batt_temp_med"] = df_summary.get("b2t_battery_max_temp_med", np.nan)
df_summary["temp_rate_C_per_min"] = df_summary["batt_temp_delta"] / df_summary["duration_mins"]
df_summary["soc_delta"] = safe_delta("soc_start", "soc_end")

# --- Voltage metrics for modal cells ---
if {"cell_max_voltage_med", "cell_min_voltage_med"}.issubset(df_summary.columns):
    df_summary["max_cellvolt"] = df_summary["cell_max_voltage_med"]
    df_summary["min_cellvolt"] = df_summary["cell_min_voltage_med"]
else:
    df_summary["max_cellvolt"] = np.nan
    df_summary["min_cellvolt"] = np.nan

# --- Compressor duty cycle ---
df_summary["comp_duty_cycle"] = (
    df_summary["comp_status_on_time"] * 100.0 /
    (df_summary["comp_status_on_time"] + df_summary["comp_status_off_time"])
)

# --- Build a contiguous run ID to separate independent segments ---
state_key = (
    df_summary["mode"].astype(str) + "|" +
    df_summary["b2t_tms_control_cmd"].astype(str) + "|" +
    df_summary["tms_working_mode"].astype(str)
)
df_summary["state_run_id"] = (state_key != state_key.shift()).cumsum()

# --- Grouping and aggregation ---
group_cols = ["state_run_id", "mode", "b2t_tms_control_cmd", "tms_working_mode"]
agg_dict = {
    "start_time": "first",
    "end_time": "last",
    "duration_mins": "sum",
    # Compressor metrics
    "comp_status_on_time": "sum",
    "comp_status_off_time": "sum",
    "comp_duty_cycle": "mean",
    # Battery temperature
    "batt_temp_start": "first",
    "batt_temp_end": "last",
    "batt_temp_med": "median",
    # Coolant start/end
    "coolant_in_start": "first",
    "coolant_in_end": "last",
    "coolant_out_start": "first",
    "coolant_out_end": "last",
    # SOC
    "soc_start": "first",
    "soc_end": "last",
    # Voltage (modal cell numbers + actual voltages)
    "maxvolt_cellnum_mode": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    "max_cellvolt": "mean",
    "minvolt_cellnum_mode": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    "min_cellvolt": "mean",
}

summary = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

# --- Compute post-aggregation deltas (true start → end transitions) ---
summary["soc_delta"] = summary["soc_end"] - summary["soc_start"]
summary["batt_temp_delta"] = summary["batt_temp_end"] - summary["batt_temp_start"]
summary["temp_rate_C_per_min"] = summary["batt_temp_delta"] / summary["duration_mins"]
summary["coolant_in_delta"] = summary["coolant_in_end"] - summary["coolant_in_start"]
summary["coolant_out_delta"] = summary["coolant_out_end"] - summary["coolant_out_start"]
summary["voltage_delta"] = summary["max_cellvolt"] - summary["min_cellvolt"]

# --- Rounding and cleanup ---
summary = summary.round({
    "comp_duty_cycle": 2,
    "temp_rate_C_per_min": 3,
    "batt_temp_delta": 2,
    "soc_delta": 2,
    "coolant_in_delta": 3,
    "coolant_out_delta": 3,
    "max_cellvolt": 4,
    "min_cellvolt": 4,
    "voltage_delta": 4
})

# --- Simplify timestamp precision ---
for col in ["start_time", "end_time"]:
    if col in summary.columns:
        summary[col] = pd.to_datetime(summary[col]).dt.strftime("%Y-%m-%d %H:%M:%S")


display(summary.head(10))

# %%
summary[['state_run_id', 'mode', 'b2t_tms_control_cmd', 'tms_working_mode','start_time', 'end_time', 'duration_mins',
        'batt_temp_start', 'batt_temp_end', 'batt_temp_med', 'batt_temp_delta', 'temp_rate_C_per_min',
        'coolant_in_start', 'coolant_in_end', 'coolant_in_delta',
        'coolant_out_start', 'coolant_out_end', 'coolant_out_delta',
        'comp_status_on_time', 'comp_status_off_time', 'comp_duty_cycle',
        'soc_start', 'soc_end', 'soc_delta',
        'maxvolt_cellnum_mode', 'max_cellvolt',
        'minvolt_cellnum_mode', 'min_cellvolt', 'voltage_delta']]

# %%
# summary.to_csv('battery_temp_summary_v2.csv', index=False)
display(summary[summary['mode']=='Charging/Park'])


