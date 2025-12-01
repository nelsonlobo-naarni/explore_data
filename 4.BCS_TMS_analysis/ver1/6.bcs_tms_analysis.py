#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc
import ctypes
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype
import platform
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import zipfile
import duckdb 
import warnings
import fastparquet
from tqdm import tqdm 
from typing import List, Optional, Union
import psutil
import time # For timing the execution

warnings.filterwarnings('ignore')


# Optional: adjust pandas display for debugging; you can comment these out
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# jupyter nbconvert --to python 6.bcs_tms_analysis.ipynb \
#     --TemplateExporter.exclude_markdown=True \
#     --TemplateExporter.exclude_output_prompt=True \
#     --TemplateExporter.exclude_input_prompt=True


mapping_file = "../../../data_points/Naarni VehicleID_RegNo_links - Vehicle_mapping.csv"
try:
    df_mapping = pd.read_csv(mapping_file)
except FileNotFoundError:
    print(f"Error: Mapping file '{mapping_file}' not found. Cannot enrich data.")
    # Create an empty mapping table to allow the rest of the script to run without crashing
    df_mapping = pd.DataFrame(columns=["id", "reg_num", "customer", "model"])
else:
    df_mapping = df_mapping.rename(columns={
        "Device No.": "id",
        "Registration No": "reg_num",
        "Customer": "customer",
        "Model": "model"
    })
    # Ensure the merge key ('id') is a string to match the chunks
    if "id" in df_mapping.columns:
        df_mapping["id"] = df_mapping["id"].astype(str)
        df_mapping = df_mapping[["id", "reg_num", "customer", "model"]]
    else:
        print("Warning: 'Device No.' column not found in mapping file.")
        df_mapping = pd.DataFrame(columns=["id", "reg_num", "customer", "model"])

print(f"Loaded mapping table with {len(df_mapping)} entries.")
# df_mapping is now ready to be passed into the processing function

df_mapping.head()


CORE_COLS = [
    "id", "timestamp", "dt",
    "vehiclereadycondition", "gun_connection_status", "ignitionstatus","odometerreading",
    "vehicle_speed_vcu", "gear_position",
    "bat_soc", "soh", "total_battery_current",
    "pack1_cellmax_temperature", "pack1_cell_min_temperature",
    "pack1_maxtemperature_cell_number", "pack1_celltemperature_cellnumber",
    "bat_voltage", "cellmax_voltagecellnumber", "cellminvoltagecellnumber", 
    "cell_min_voltage","cell_max_voltage",
]


def free_mem():
    """Try to return freed memory back to the OS (no-op on some platforms)."""
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def rename_battery_temp_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Uses df.rename(inplace=False), creating one copy, which is fine for chunks
    rename_map = {
        "pack1_cellmax_temperature": "batt_maxtemp",
        "pack1_cell_min_temperature": "batt_mintemp",
        "pack1_maxtemperature_cell_number":"batt_maxtemp_tc", 
        "pack1_celltemperature_cellnumber":"batt_mintemp_tc",
        "cell_max_voltage":"batt_maxvolt",
        "cellmax_voltagecellnumber":"batt_maxvolt_cell",
        "cell_min_voltage":"batt_minvolt",
        "cellminvoltagecellnumber":"batt_minvolt_cell", 
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if not existing:
        return df
    return df.rename(columns=existing)


# =====================================================================
# CONFIG
# =====================================================================

MAX_SPEED_KMPH = 120.0      # physical upper bound (bus never >120 km/h)
MAX_ODO_DIST_KM = 0.2       # max plausible odo jump per sample (~200 m)
MAX_DT_SEC = 3.0            # dt_sec cap (you already use this)
BIG_ODO_CAP = 1.0           # sanity cap for odo (km)
DT_DISCONTINUITY_SEC = 180  # >3 min gap can be treated as discontinuity in Stage-2


def finalize_odometer(df):
    df = df.sort_values(["id", "timestamp"]).copy()

    for vid, grp in df.groupby("id"):
        idx = grp.index
        odo = grp["odometer_final"].to_numpy()

        for i in range(1, len(odo)):
            if odo[i] < odo[i-1]:
                odo[i] = odo[i-1]

        df.loc[idx, "odometer_final"] = odo

    return df


def impute_odometer(df, odo_col="odometerreading"):
    """
    TRUE, CORRECT, NULL-ONLY, SESSION-AWARE, 3-PASS ODOMETER IMPUTER.

    Rules implemented exactly:

      • PASS 1: Fix top/bottom NULL islands.
      • PASS 2: SINGLE NULL: bracket logic + speed/dt estimate + clamping.
      • PASS 3: MULTI NULL: iterative bounded fill, updating L → new L.
      • Session protection: If R < L → treat as session break → propagate L.
      • STRICT: Never modify original *non-null* odometer readings.
    """

    df = df.sort_values(["id", "timestamp"]).copy()
    df["odometer_final"] = df[odo_col].astype(float)

    for vid, grp in df.groupby("id"):
        idx = grp.index
        odo_raw = grp[odo_col].astype(float).to_numpy()
        speed = grp["vehicle_speed_vcu"].astype(float).to_numpy()
        dt = grp["dt_sec"].astype(float).to_numpy()

        # Final output buffer:
        fill = odo_raw.copy()

        n = len(odo_raw)
        i = 0

        while i < n:
            if not np.isnan(odo_raw[i]):
                i += 1
                continue

            # Start of a null island
            start = i
            while i < n and np.isnan(odo_raw[i]):
                i += 1
            end = i - 1  # inclusive

            prev_idx = start - 1
            next_idx = end + 1

            L = odo_raw[prev_idx] if prev_idx >= 0 else None
            R = odo_raw[next_idx] if next_idx < n else None

            # -------------------------------
            # PASS 1: TOP & BOTTOM NULL ISLANDS
            # -------------------------------
            if L is None and R is not None:
                # Top NULL block → propagate next known value backward
                for pos in range(start, end + 1):
                    fill[pos] = R
                continue

            if R is None and L is not None:
                # Bottom NULL block → propagate previous known value forward
                for pos in range(start, end + 1):
                    fill[pos] = L
                continue

            # If both missing → extremely rare but fallback to zero change
            if L is None and R is None:
                continue

            # -------------------------------
            # SESSION BREAK PROTECTION
            # -------------------------------
            if R < L:
                # Monotonic break → treat as end-of-session
                for pos in range(start, end + 1):
                    fill[pos] = L
                continue

            # -------------------------------
            # PASS 2 & PASS 3 (Unified Engine)
            # -------------------------------
            gap = end - start + 1

            # The active bounds shrink as we impute
            curr_L = L
            curr_R = R

            for k in range(gap):
                pos = start + k

                if speed[pos] == 0:
                    est = curr_L  # idle → no movement
                else:
                    # compute movement in km
                    est = curr_L + (speed[pos] * dt[pos] / 3600.0)

                # Clamp within [curr_L, curr_R]
                est_clamped = max(curr_L, min(est, curr_R))

                # STRICT: only fill if original was NULL
                if np.isnan(odo_raw[pos]):
                    fill[pos] = est_clamped

                # Shrink left boundary → progressive update
                curr_L = fill[pos]

        df.loc[idx, "odometer_final"] = np.round(fill, 3)

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean & impute all sensors EXCEPT odometer.

    Includes:
      - dt_sec sanitisation
      - SOC fixing (SOC=0 → NaN → interpolated)
      - temperature, voltage, TC/cell sanity
      - refined battery current clamping
      - ignition/ready/gun consistency
      - charging-mode overrides
      - parked-mode overrides
      - speed imputation for all stable states
      - gear correction
    """

    df = df.sort_values(["id", "timestamp"]).copy()

    # Round column 'odometerreading' to 3 decimal places, preserving NaNs
    mask_odo = df['odometerreading'].notna() # Create a boolean mask for non-null values
    df.loc[mask_odo, 'odometerreading'] = df.loc[mask_odo, 'odometerreading'].round(3)

    # ----------------------------------------------------
    # 0. dt_sec calculation
    # ----------------------------------------------------
    df["dt_sec"] = (
        df.groupby("id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )
    df.loc[df["dt_sec"] > 3, "dt_sec"] = 0

    # ----------------------------------------------------
    # 1. SANITISATION
    # ----------------------------------------------------
    # 1a temperature sanitisation
    for col in ["batt_maxtemp", "batt_mintemp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < -10, col] = pd.NA

    # 1b voltage
    for col in ["batt_maxvolt", "batt_minvolt", "bat_voltage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = pd.NA

    # 1c thermocouple + cell
    for col in ["batt_maxtemp_tc", "batt_mintemp_tc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 1) | (df[col] > 108), col] = pd.NA

    for col in ["batt_maxvolt_cell", "batt_minvolt_cell"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 1) | (df[col] > 576), col] = pd.NA

    # ----------------------------------------------------
    # 1d SOC FIX — VERY IMPORTANT
    # ----------------------------------------------------
    df["bat_soc"] = pd.to_numeric(df["bat_soc"], errors="coerce")

    # SOC=0 is almost always a sensor glitch → treat as missing
    df.loc[df["bat_soc"] == 0, "bat_soc"] = np.nan

    # Interpolate SOC per vehicle
    df["bat_soc"] = (
        df.groupby("id")["bat_soc"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )


    df["bat_soc"] = df["bat_soc"].clip(lower=0, upper=100)

    # ----------------------------------------------------
    # 1e refined battery current clamp
    # ----------------------------------------------------
    curr = pd.to_numeric(df["total_battery_current"], errors="coerce")
    valid_mask = curr.abs().between(0, 2500)
    valid_values = curr.where(valid_mask)

    curr_ff = valid_values.ffill().fillna(0.0)
    curr = curr.where(valid_mask, curr_ff)
    df["total_battery_current"] = curr

    # ----------------------------------------------------
    # 2. GROUPWISE IMPUTATION (per vehicle)
    # ----------------------------------------------------
    impute_cols = [
        ("batt_maxtemp", 80),
        ("batt_mintemp", 80),
        ("batt_maxtemp_tc", 80),
        ("batt_mintemp_tc", 80),
        ("batt_maxvolt", 30),
        ("batt_minvolt", 30),
        ("batt_maxvolt_cell", 30),
        ("batt_minvolt_cell", 30),
        ("bat_voltage", 20),
        ("bat_soc", 300),   # now cleaned
        ("soh", 300),
    ]

    for vid, grp in df.groupby("id"):
        idx = grp.index

        # -----------------------------------------
        # 2a forward/backfill for regular sensors
        # -----------------------------------------
        for col, limit in impute_cols:
            df.loc[idx, col] = grp[col].ffill().bfill()

        # -----------------------------------------
        # 2b current interpolation for small gaps
        # -----------------------------------------
        df.loc[idx, "total_battery_current"] = grp["total_battery_current"].interpolate(
            limit=10, limit_direction="both"
        )

        # -----------------------------------------
        # 2c BASIC READY + GUN fill
        # -----------------------------------------
        for col in ["vehiclereadycondition", "gun_connection_status"]:df.loc[idx, col] = grp[col].ffill().bfill()

        # -----------------------------------------
        # 2d IGNITIONSTATUS CLEANUP
        # -----------------------------------------
        ign = pd.to_numeric(grp["ignitionstatus"], errors="coerce")
        ign = ign.ffill().bfill()

        # Ready=1 & ignition null → ignition=1
        ready_mask = df.loc[idx, "vehiclereadycondition"].fillna(0).astype(int).eq(1)
        ign.loc[ready_mask & ign.isna()] = 1
        df.loc[idx, "ignitionstatus"] = ign

        # -----------------------------------------
        # 2e SPEED IMPUTATION
        # -----------------------------------------
        if "vehicle_speed_vcu" in grp.columns:
            v = pd.to_numeric(grp["vehicle_speed_vcu"], errors="coerce")
            v = v.where(v.between(0, 120), np.nan)

            v = v.ffill().bfill()
            df.loc[idx, "vehicle_speed_vcu"] = v.round(2)

        # -----------------------------------------
        # 2f GEAR POSITION
        # -----------------------------------------
        if "gear_position" in grp.columns:
            g = pd.to_numeric(grp["gear_position"], errors="coerce")
            g = g.where(g.isin([0, 1, 2]), np.nan)

            ready0 = df.loc[idx, "vehiclereadycondition"].fillna(0).astype(int).eq(0)
            ign0 = df.loc[idx, "ignitionstatus"].fillna(0).astype(int).eq(0)

            force_neutral = ready0 | ign0
            g[force_neutral] = 0

            df.loc[idx, "gear_position"] = g.ffill().bfill().astype("Int64")

        # ----------------------------------------------------
        # 2g CHARGING STATE CONSISTENCY
        # ----------------------------------------------------
        charging = df.loc[idx, "gun_connection_status"].fillna(0).astype(int).eq(1)

        # ignition ON during charging
        df.loc[idx[charging], "ignitionstatus"] = 1
        df.loc[idx[charging], "vehiclereadycondition"] = 0
        df.loc[idx[charging], "gear_position"] = 0

        # speed=0 when charging & missing
        df.loc[idx[charging & df.loc[idx, "vehicle_speed_vcu"].isna()],"vehicle_speed_vcu"] = 0.0

        # ----------------------------------------------------
        # 2h OFF/PARKED SPEED FIX
        # gun=0 & ready=0 & ignition=0 & speed NA → 0
        # ----------------------------------------------------
        off_mask = (
            (df.loc[idx, "gun_connection_status"].fillna(0).astype(int) == 0) &
            (df.loc[idx, "vehiclereadycondition"].fillna(0).astype(int) == 0) &
            (df.loc[idx, "ignitionstatus"].fillna(0).astype(int) == 0)
        )

        df.loc[idx[off_mask & df.loc[idx, "vehicle_speed_vcu"].isna()],"vehicle_speed_vcu"] = 0.0

        # ----------------------------------------------------
        # 2i READY BUT IGNITION=0 (contradictory → treat as stationary)
        # ----------------------------------------------------
        ready_ign_off = (
            (df.loc[idx, "gun_connection_status"].fillna(0).astype(int) == 0) &
            (df.loc[idx, "vehiclereadycondition"].fillna(0).astype(int) == 1) &
            (df.loc[idx, "ignitionstatus"].fillna(0).astype(int) == 0)
        )

        df.loc[idx[ready_ign_off & df.loc[idx, "vehicle_speed_vcu"].isna()],"vehicle_speed_vcu"] = 0.0

        df.vehicle_speed_vcu = df.vehicle_speed_vcu.round(2)

    return df


def prepare_df_with_state(df: pd.DataFrame, df_mapping: pd.DataFrame) -> pd.DataFrame:
    # Make a copy (chunk-safe)
    out = df.copy()

    # ------------------------------------------------------------------
    # 1. Merge mapping
    # ------------------------------------------------------------------
    out["id"] = out["id"].astype(str)
    out = out.merge(df_mapping, on="id", how="left", validate="m:1")

    # fill mapping fallbacks
    out["reg_num"] = out["reg_num"].fillna("REGNUM_" + out["id"])
    out["customer"] = out["customer"].fillna("CUST_" + out["id"])
    out["model"] = out["model"].fillna("MDL_" + out["id"])

    out["reg_num"] = out["reg_num"].astype(str)
    out["customer"] = out["customer"].astype(str)
    out["model"] = out["model"].astype(str)

    # -----------------------------------------------------------
    # 2. Timestamp handling — THE SAFE VERSION
    # -----------------------------------------------------------

    # Parse but do *not* localize or convert
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Drop invalid timestamps
    out = out.dropna(subset=["timestamp"])

    # Sort strictly by original order: id + timestamp
    out = out.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Mode + alt_mode logic (unchanged)
    # ------------------------------------------------------------------

    # Gun connection normalization
    gcs_raw = out["gun_connection_status"]
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.lower().str.strip()

    gun_connected = (gcs_num == 1) | gcs_str.isin({"1","true","yes","y","connected","on"})
    gun_connected = gun_connected.fillna(False)

    # Vehicle readiness normalization
    if "vehiclereadycondition" in out.columns:
        vrc_raw = out["vehiclereadycondition"]
        vrc_num = pd.to_numeric(vrc_raw, errors="coerce")
        vrc_str = vrc_raw.astype(str).str.strip().str.lower()
        vehicle_ready = (vrc_num == 1) | vrc_str.isin({"1","true","yes","y","ready","on"})
        vehicle_ready = vehicle_ready.fillna(False)
    else:
        vehicle_ready = pd.Series(False, index=out.index)

    # Legacy mode column
    out["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")

    # Rolling current for alt_mode
    current_rm = (
        out["total_battery_current"]
        .rolling(15, min_periods=1)
        .mean()
        .fillna(0)
    )

    # thresholds
    ACTIVE_CHG_THRESH = -15
    MAINTAIN_LOW = -15
    MAINTAIN_HIGH = +2

    # CHARGING states
    chg_active = (gun_connected &(out["total_battery_current"] < -5)).to_numpy(dtype=bool)
    chg_maint = (gun_connected &(out["total_battery_current"].abs().between(0, 5))).to_numpy(dtype=bool)
    chg_idle = (gun_connected &(out["total_battery_current"] > 5)).to_numpy(dtype=bool)

    # # DISCHARGING states
    dis_active = ((~gun_connected) &(out["vehicle_speed_vcu"].gt(0.5).fillna(False)) &(out["gear_position"].isin([1, 2]).fillna(False))).to_numpy(dtype=bool)
    # dis_idle   = ((~gun_connected) & (~dis_active)).to_numpy(dtype=bool)

    out["alt_mode"] = np.select(
        [chg_active, chg_maint, chg_idle, dis_active],
        ["CHARGING_ACTIVE", "CHARGING_MAINTAIN", "CHARGING_IDLE", "DISCHARGING_ACTIVE"],
        default="DISCHARGING_IDLE"
    )

    # ------------------------------------------------------------------
    # 4. Delta + buckets (unchanged)
    # ------------------------------------------------------------------
    for col in ["batt_maxtemp", "batt_mintemp", "batt_maxvolt", "batt_minvolt"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["batt_temp_delta"] = out["batt_maxtemp"] - out["batt_mintemp"]
    out["volt_delta_mv"] = abs((out["batt_maxvolt"] - out["batt_minvolt"]) * 1000)  # absolute value since max < min can happen

    out["date_val"] = out["timestamp"].dt.floor("D")

    # Bucketing
    out["maxtemp_bucket"] = pd.cut(
        out["batt_maxtemp"],
        [-np.inf, 28, 32, 35, 40, np.inf],
        labels=["<28", "28–32", "32–35", "35–40", ">40"]
    )

    out["temp_delta_bucket"] = pd.cut(
        out["batt_temp_delta"],
        [-np.inf, 2, 5, 8, np.inf],
        labels=["<2", "2–5", "5–8", ">8"]
    )

    out["volt_delta_bucket"] = pd.cut(
        out["volt_delta_mv"],
        [0, 10, 20, 30, np.inf],
        labels=["0–10", "10–20", "20–30", ">30"],
        include_lowest=True
    )

    soc_bins = [0,10,20,30,40,50,60,70,80,90,np.inf]
    soc_labels = ["0–10","10–20","20–30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]

    out["soc_band_bucket"] = pd.cut(out["bat_soc"], bins=soc_bins, labels=soc_labels)

    # ------------------------------------------------------------------
    # 5. Select final columns
    # ------------------------------------------------------------------
    cols_keep = [
        "id","reg_num","customer","model",
        "timestamp","date_val","dt_sec",
        "mode","alt_mode",
        "ignitionstatus","vehiclereadycondition","gun_connection_status",
        "vehicle_speed_vcu","gear_position",
        "odometerreading","odometer_final",
        "batt_maxtemp","batt_mintemp","batt_temp_delta",
        "maxtemp_bucket","temp_delta_bucket",
        "batt_maxvolt","batt_minvolt","volt_delta_mv","volt_delta_bucket",
        "batt_maxtemp_tc","batt_mintemp_tc",
        "pack_id_max","pack_id_min",
        "batt_maxvolt_cell","batt_minvolt_cell",
        "bat_voltage","total_battery_current",
        "bat_soc","soc_band_bucket","soh"
    ]

    cols_keep = [c for c in cols_keep if c in out.columns]
    out = out[cols_keep]

    return out



# --- DUCKDB CHUNK GENERATOR (Fixed) ---

def duckdb_chunk_generator(conn, sql_query, chunk_size):
    """Generates Pandas DataFrames in chunks directly from DuckDB cursor."""
    cursor = conn.cursor() 
    cursor.execute(sql_query)

    while True:
        # Uses the corrected method name: fetch_df_chunk
        chunk = cursor.fetch_df_chunk(chunk_size) 
        if chunk is None or chunk.empty:
            break
        yield chunk

# --- ROBUST FILE EXTRACTION (Fixed from OSErrors) ---

def extract_files_to_disk(zip_path, output_dir):
    """Cleans directory and extracts all Parquet files from ZIP."""
    if output_dir.exists():
        logging.info(f"🧹 Clearing existing directory: {output_dir.resolve()}")
        # Robust cleanup to avoid OS/lock issues
        try:
            shutil.rmtree(output_dir)
        except OSError:
             for item in output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    os.remove(item) 
             os.rmdir(output_dir)

    output_dir.mkdir(parents=True)

    logging.info("🔄 Extracting ALL Parquet files from ZIP to disk...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            all_files_to_extract = [f for f in z.namelist() if f.endswith(".parquet")]
            logging.info(f"🔎 Found {len(all_files_to_extract)} total Parquet files in archive.")
            for filename in all_files_to_extract:
                z.extract(filename, path=output_dir)
            return len(all_files_to_extract)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ ZIP file not found at: {zip_path}") from None

def setup_duckdb_query(output_dir, utc_start, utc_end, core_cols):
    """Sets up DuckDB connection and SQL query."""
    parquet_glob_path = str(output_dir.joinpath("**/*.parquet"))
    # Only select the columns you need for Stage 1 processing
    column_list = ", ".join([f'"{c}"' for c in core_cols])

    # CRITICAL: Predicate Pushdown filter on the internal 'timestamp' column
    sql_query = f"""
        SELECT {column_list}
        FROM read_parquet('{parquet_glob_path}')
        WHERE 
            "timestamp" >= '{utc_start.isoformat()}' AND 
            "timestamp" < '{utc_end.isoformat()}'
    """
    return duckdb.connect(), sql_query


def run_stage1_data_setup(analysis_start_date_str: str, 
                          analysis_end_date_str: str, 
                          zip_path: Path, 
                          extraction_dir: Path,
                          force_extraction: bool = False) -> tuple[datetime, datetime, int]:
    """
    Handles date range setup, IST-to-UTC conversion, file extraction, 
    and checks if data is available for processing.

    Args:
        analysis_start_date_str: Start date in YYYY-MM-DD format.
        analysis_end_date_str: End date in YYYY-MM-DD format.
        zip_path: Path to the source ZIP file.
        extraction_dir: Target directory for extracted Parquet files.
        force_extraction: If True, always clean and re-extract files. 
                          If False, skips extraction if the directory exists.

    Returns: (utc_start, utc_end, file_count)
    """

    # 1. Date Parsing and UTC Conversion (Assuming +5:30 IST offset)
    target_date = datetime.strptime(analysis_start_date_str, "%Y-%m-%d").date()
    ist_start = datetime.combine(target_date, datetime.min.time())

    end_date_obj = datetime.strptime(analysis_end_date_str, "%Y-%m-%d").date()
    ist_end = datetime.combine(end_date_obj, datetime.min.time()) + timedelta(days=1)

    utc_start = ist_start - timedelta(hours=5, minutes=30)
    utc_end = ist_end - timedelta(hours=5, minutes=30)

    logging.info(f"🔍 Analysis window (UTC): {utc_start} → {utc_end}")

    # 2. FILE EXTRACTION CONTROL
    file_count = 0

    if extraction_dir.exists() and not force_extraction:
        logging.info("♻️ Skipping file extraction: Directory exists and force_extraction=False.")
        # Recursively count all .parquet files in the existing directory
        file_count = len(list(extraction_dir.rglob('*.parquet')))
        if file_count > 0:
             logging.info(f"✅ Found {file_count} existing files. Proceeding to DuckDB loading.")

    else:
        # If directory doesn't exist, or force_extraction is True, run the full extraction.
        logging.info("🔄 Running full extraction (Cleanup + Extract)...")
        # This relies on the robust `extract_files_to_disk` function
        file_count = extract_files_to_disk(zip_path, extraction_dir)

    # 3. Validation Check
    if file_count == 0:
        logging.warning("🛑 Skipping analysis: No files were found.")
        sys.exit() # Exit the script cleanly if no files were found

    return utc_start, utc_end, file_count


# --- CONFIGURATION (Ensure these are defined at the top of your script) ---
ZIP_FILE_PATH = "../../../data_points/naarni75_cpoall.zip" 
EXTRACTION_DIR = Path("../../../data_points/extracted_parts/cpo_all") 
# --------------------------------------------------------------------------

# --- Set 75-Day Date Range (Using your target dates) ---
analysis_start_date_str = "2025-09-06" 
# NOTE: Using 2025-11-14 since 75 days starts on 2025-09-01 and ends on 2025-11-14.
# Using 2025-11-15 will include the start of the 76th day (if data exists).
analysis_end_date_str = "2025-09-15"   
# --------------------------------------------------------

# NEW STAGE 1 EXECUTION:
utc_start, utc_end, file_count = run_stage1_data_setup(
    analysis_start_date_str=analysis_start_date_str,
    analysis_end_date_str=analysis_end_date_str,
    zip_path=ZIP_FILE_PATH,
    extraction_dir=EXTRACTION_DIR,
    force_extraction = False)


# 2. SETUP DUCKDB QUERY
conn, sql_query = setup_duckdb_query(EXTRACTION_DIR, utc_start, utc_end, CORE_COLS)

# A quick DuckDB query to get the distinct IDs from the 75-day filtered dataset
get_ids_query = f"""
    SELECT DISTINCT id 
    FROM ({sql_query})
"""
# Fetch the list of IDs (this is a very small amount of data)
vehicle_ids = conn.execute(get_ids_query).fetchdf()["id"].astype(str).tolist()
# vehicle_ids = ['3','16','18','19','32','42','6','7','9','11','12','13','14','15','20','25','27','28','29','30','31','33','35','41','46']

logging.info(f"✅ Found {len(vehicle_ids)} unique vehicle IDs in the dataset.")
print(sorted(vehicle_ids))


# NOTE: The functions 'duckdb_chunk_generator', 'rename_battery_temp_columns', 
# 'impute_missing_values', 'prepare_df_with_state', 'finalize_odometer', and 'free_mem' 
# MUST be defined in your Jupyter Notebook environment before calling this.

def process_and_save_data(
    conn: duckdb.DuckDBPyConnection, 
    sql_query: str, 
    chunk_size: int, 
    parquet_feather_path: str,
    vehicle_ids: List[str],
    df_mapping: pd.DataFrame,
    extract_data: bool = False,
    chunk_log_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Executes the memory-safe chunked processing loop for Stage 1.

    MODIFIED (corrected odometer handling):
      - Cross-chunk odometer continuity is tracked via last_known_odo.
      - last_known_odo is now updated *after* finalize_odometer(), using
        the finalized odometer_final values in df_chunk_state.
      - This guarantees no backward odometer jumps across chunks.
    """

    output_path = Path(parquet_feather_path)
    # Cross-chunk odometer cache: {vehicle_id: last_valid_odometer_final}
    last_known_odo: Dict[str, float] = {}

    # -------------------------------------------------------
    # SKIP IF FILE EXISTS
    # -------------------------------------------------------
    if not extract_data and output_path.exists():
        conn_count = duckdb.connect()
        try:
            total_rows = conn_count.execute(
                f"SELECT count(*) FROM '{parquet_feather_path}'"
            ).fetchone()[0]
            logging.info(f"✅ Skipping: File already exists with {total_rows:,} rows.")
            return total_rows
        finally:
            conn_count.close()

    # -------------------------------------------------------
    # INIT & PRE-CALCULATION
    # -------------------------------------------------------
    logging.info(f"🧠 Preparing data stream...")

    try:
        total_input_rows = conn.execute(
            f"SELECT COUNT(*) FROM ({sql_query})"
        ).fetchone()[0]
        logging.info(f"📊 Total rows to process: {total_input_rows:,}")
    except Exception as e:
        logging.warning(
            f"Could not determine total row count: {e}. Progress bar will be indefinite."
        )
        total_input_rows = None

    logging.info(f"💾 Output path: {parquet_feather_path}")

    first_chunk = True
    total_processed_rows = 0
    chunk_index = 0

    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # prime CPU sampling

    # Chunk timing log setup
    log_file = None
    if chunk_log_path is not None:
        log_file = Path(chunk_log_path)
        if not log_file.exists():
            with log_file.open("w") as f:
                f.write(
                    "timestamp,chunk_idx,chunk_rows,total_rows,"
                    "duration_sec,rows_per_sec,cpu_pct,ram_mb\n"
                )

    # -------------------------------------------------------
    # LOGGING SUPPRESSION
    # -------------------------------------------------------
    current_log_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)

    try:
        # -------------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------------
        progress = tqdm(
            total=total_input_rows,
            desc="Processing Data",
            unit="row",
            mininterval=0.2,
            dynamic_ncols=True,
            colour="cyan",
        )

        # -------------------------------------------------------
        # MAIN LOOP: STREAM CHUNKS
        # -------------------------------------------------------
        for chunk in duckdb_chunk_generator(conn, sql_query, chunk_size):
            t0 = time.perf_counter()
            chunk_index += 1

            raw_chunk_len = len(chunk)
            df_chunk = chunk

            # --- PREP 1: Rename ---
            df_chunk = rename_battery_temp_columns(df_chunk)

            df_chunk['vehicle_speed_vcu'] = df_chunk['vehicle_speed_vcu'].round(2)

            # --- PREP 2: Filter & Cast ---
            if "id" in df_chunk.columns:
                df_chunk["id"] = df_chunk["id"].astype(str)
                if vehicle_ids:
                    df_chunk = df_chunk[df_chunk["id"].isin(vehicle_ids)]
                df_chunk = df_chunk.convert_dtypes()

            # --- PREP 2b: Cross-chunk odometer continuity (odometerreading) ---
            if "odometerreading" in df_chunk.columns:
                for vid in df_chunk["id"].unique():
                    if vid in last_known_odo:
                        first_idx = df_chunk[df_chunk["id"] == vid].index.min()
                        if pd.isna(df_chunk.loc[first_idx, "odometerreading"]):
                            df_chunk.loc[first_idx, "odometerreading"] = last_known_odo[vid]

            # --- PREP 3: Time & Impute ---
            df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], errors="coerce")
            df_chunk = df_chunk.sort_values(["id", "timestamp"]).copy()
            df_chunk = impute_missing_values(df_chunk)
            df_chunk = impute_odometer(df_chunk)

            # # --- PREP 4: State Prep ---
            df_chunk_state = prepare_df_with_state(df_chunk, df_mapping)

            # --- PREP 4b: FINALIZE ODOMETER on state DataFrame ---
            # This should enforce rounding + monotonicity.
            df_chunk_state = finalize_odometer(df_chunk_state)

            # --- PREP 4c: Update cross-chunk continuity AFTER finalization ---
            if "odometer_final" in df_chunk_state.columns:
                for vid, sub in df_chunk_state.groupby("id"):
                    sub_final = sub["odometer_final"].dropna()
                    if len(sub_final):
                        # Use *finalized* odometer for continuity
                        last_known_odo[vid] = float(sub_final.iloc[-1])

            if df_chunk_state.empty:
                progress.update(raw_chunk_len)
                progress.set_postfix_str("Skipped empty chunk")
                del df_chunk, df_chunk_state
                gc.collect()
                free_mem()
                continue

            rows_saved_this_chunk = len(df_chunk_state)
            total_processed_rows += rows_saved_this_chunk

            # --- SAVE CHUNK ---
            if first_chunk:
                df_chunk_state.to_parquet(
                    parquet_feather_path, compression="zstd", index=False
                )
                first_chunk = False
            else:
                fastparquet.write(
                    parquet_feather_path,
                    df_chunk_state,
                    compression="zstd",
                    write_index=False,
                    append=True,
                )

            # --- METRICS ---
            t1 = time.perf_counter()
            duration = t1 - t0
            rows_per_sec = rows_saved_this_chunk / duration if duration > 0 else 0
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_mb = process.memory_info().rss / (1024 * 1024)

            # --- UPDATE PROGRESS ---
            progress.update(raw_chunk_len)
            progress.set_postfix(
                saved=f"{total_processed_rows:,}",
                cpu=f"{cpu_pct:4.1f}%",
                ram=f"{ram_mb:6.1f}MB",
                speed=f"{rows_per_sec:8.1f} r/s",
            )

            # --- FILE LOGGING ---
            if log_file is not None:
                with log_file.open("a") as f:
                    f.write(
                        f"{datetime.now().isoformat()},{chunk_index},"
                        f"{rows_saved_this_chunk},{total_processed_rows},"
                        f"{duration:.3f},{rows_per_sec:.1f},"
                        f"{cpu_pct:.1f},{ram_mb:.1f}\n"
                    )

            # --- CLEANUP ---
            del df_chunk, df_chunk_state
            gc.collect()
            free_mem()

    finally:
        if "progress" in locals():
            progress.close()

        logging.getLogger().setLevel(current_log_level)
        conn.close()

    logging.info(f"✅ Finished processing. Total rows saved: {total_processed_rows:,}")
    return total_processed_rows


# --- ASSUMING run_stage1_data_setup WAS CALLED AND RETURNED utc_start, utc_end ---
# Example configuration that needs to be available:
# EXTRACTION_DIR = Path("../extracted_parts") 
# CORE_COLS = [...]
# CRITICAL FIX: Drastically reduced chunk size to prevent memory spike
CHUNK_SIZE = 500 # Process 50,000 rows max at any time


# 1. Setup DuckDB Query (as shown previously)
conn, sql_query = setup_duckdb_query(EXTRACTION_DIR, utc_start, utc_end, CORE_COLS)

# 2. Define Inputs
# output_feather_file = "df_with_state_30days.feather"
output_parquet_file = "../df_with_state.parquet"

# 3. Run the memory-safe processing loop
total_rows = process_and_save_data(
    conn=conn,
    sql_query=sql_query,
    chunk_size=CHUNK_SIZE,
    parquet_feather_path=output_parquet_file,
    vehicle_ids=vehicle_ids,
    df_mapping=df_mapping,
    extract_data=True
)

logging.info(f"🎉 Final DataFrame saved. Total rows processed: {total_rows:,}")
logging.info("✅ Data Processing (Stage 1) complete. Feather file ready for analysis.")


def read_parquet_subset(parquet_path: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Reads a subset of the processed Feather file using a date filter
    applied directly by DuckDB (predicate pushdown).

    Args:
        feather_path: Path to the processed Feather file.
        start_dt: Start datetime for the filter (inclusive).
        end_dt: End datetime for the filter (exclusive).

    Returns:
        A new DataFrame containing only the filtered data.
    """
    logging.info(f"Loading data subset from {start_dt} to {end_dt}...")

    # Use DuckDB to query the Feather file directly on disk
    con = duckdb.connect()

    # The SQL query filters rows on the disk file based on the 'timestamp' column.
    sql_query = f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        WHERE 
            "timestamp" >= '{start_dt.isoformat()}' AND 
            "timestamp" < '{end_dt.isoformat()}'
    """

    # Fetch the filtered, smaller DataFrame
    df_subset = con.execute(sql_query).fetchdf()
    con.close()

    logging.info(f"✅ Loaded {len(df_subset):,} rows for the requested subset.")
    return df_subset


# Define the 30-day filter window
filter_start_date = datetime(2025, 9, 1)
filter_end_date = datetime(2025, 9, 2) # Exclusive end date

# 1. Load the filtered subset safely
df_subset = read_parquet_subset(
    parquet_path="../df_with_state.parquet",
    start_dt=filter_start_date,
    end_dt=filter_end_date
)

# # 2. Sort the data by vehicle ID and timestamp
# # This is CRUCIAL for the cumulative maximum to work correctly for each vehicle.
df_subset = df_subset.sort_values(by=['id', 'timestamp']).reset_index(drop=True)

# # 3. Apply the cumulative maximum, GROUPED BY 'id'
# # This enforces monotonicity (non-decreasing readings) for the odometer.
df_subset['odometer_final'] = df_subset.groupby('id')['odometer_final'].cummax()

display(df_subset.head())

# 3. Aggressive memory cleanup after use (CRITICAL)
# del df_subset
# gc.collect()
# free_mem()


df_subset.timestamp.min(),df_subset.timestamp.max()


# df_subset.bat_soc.describe(percentiles=[0.005,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])


df_subset.isna().sum()


# df_subset['timestamp'] = pd.to_datetime(df_subset['timestamp'], utc=True)
# df_subset = df_subset.sort_values(by=['id', 'timestamp']).reset_index(drop=True)
# df_subset['odometer_final'] = df_subset.groupby('id')['odometer_final'].cummax()


df_subset.id.unique()


df_subset[df_subset.id == '3'].odometer_final.diff().describe(percentiles=[0.001,0.01,0.25,0.5,0.75,0.9,0.95,0.99,0.9995])


df16 = df_subset[df_subset.id == '16'].copy()
df16 = df16.reset_index(drop=True)  # CRITICAL FIX

# find raw negative diffs
neg_idx = df16.index[df16["odometer_final"].diff() >110]

# Build ±2 context window
context_idx = (
    set(neg_idx - 10)
    | set(neg_idx - 1)
    | set(neg_idx)
    | set(neg_idx + 1)
    | set(neg_idx + 2)
)

# Keep only valid iloc rows
context_idx = [i for i in context_idx if 0 <= i < len(df16)]

# Show the window
df16.iloc[context_idx][[
    "timestamp",
    "odometerreading",
    "odometer_final",
    "vehicle_speed_vcu",
    "dt_sec"
]].sort_index()


df16.iloc[context_idx][[
    "timestamp",
    "odometerreading",
    "odometer_final",
    "vehicle_speed_vcu",
    "dt_sec"
]].sort_index().to_csv("df16_negative_odo_context.csv")


df_subset.alt_mode.value_counts()


# df_subset.volt_delta_mv.describe(percentiles=[0.9,0.95,0.9992])
# df_subset[df_subset.batt_mintemp>-40].batt_mintemp.describe()
# len(df_subset[(df_subset.odometer_final.isnull())&(df_subset.vehicle_speed_vcu>1)])
# len(df_subset[df_subset.dt_sec==0])
# df_subset[(df_subset.odometer_final.isnull())]#.vehicle_speed_vcu.describe()
# df_subset[['timestamp','vehicle_speed_vcu','gear_position','odometerreading','odometer_final']].isnull().sum()*100.0/len(df_subset)
# missing = df_subset[df_subset.odometer_final.isna()]
# print("Total missing:", len(missing))
# print(missing[['timestamp','vehicle_speed_vcu','dt_sec','odometerreading']].head(20))
# print("Speed null %:", missing.vehicle_speed_vcu.isna().mean()*100)
# print("dt null %:", missing.dt_sec.isna().mean()*100)
# df_subset[(df_subset.vehiclereadycondition == 0)&(df_subset.gun_connection_status == 1)]
# df_subset[(df_subset.alt_mode=='CHARGING')&(df_subset.total_battery_current<-5)].total_battery_current.describe(percentiles=[0.7,0.8,0.9,0.95,0.98,0.99,0.999])
# df_subset.groupby(['id','date_val','alt_mode'])['alt_mode'].count()
# display(df_subset[(df_subset.mode=='CHARGING')&(df_subset.vehiclereadycondition=='1')&(df_subset.gun_connection_status=='1')].head(100))
# df_subset.volt_delta_mv.max()
# df_subset.total_battery_current[df_subset.total_battery_current>-3200].min()
# pd.set_option('display.float_format', '{:.2f}'.format)
# pd.reset_option('display.float_format')
# df_subset.head(1000).to_csv('data_ref.csv')
# df_subset[df_subset.dt_sec>1].dt_sec.describe(percentiles=[0.9,0.95,0.99,0.995,0.997]).round(2)
# df_subset.groupby(['id','date_val'])['dt_sec'].sum()/60.0

# # chk = ((df_subset.groupby(['date_val','mode','batt_maxtemp_tc','pack_id_max'])['dt_sec'].sum()*100.0/60.0)/(df_subset.groupby(['date_val'])['dt_sec'].sum()/60.0)).sort_values()
# chk = (
#         (df_subset[df_subset['mode'] == 'CHARGING'].groupby(['date_val','mode','batt_maxtemp_tc','pack_id_max'])['dt_sec'].sum()*100.0/60.0) / 
#         (df_subset[df_subset['mode'] == 'CHARGING'].groupby(['date_val'])['dt_sec'].sum()/60.0)
#       ).sort_index(level='batt_maxtemp_tc')
# display(chk)


# # chk = ((df_subset.groupby(['date_val','mode','batt_maxtemp_tc','pack_id_max'])['dt_sec'].sum()*100.0/60.0)/(df_subset.groupby(['date_val'])['dt_sec'].sum()/60.0)).sort_values()
# chk2 = (
#         (df_subset[df_subset['mode'] == 'CHARGING'].groupby(['date_val','mode','batt_maxtemp_tc','pack_id_max'])['dt_sec'].sum()/60.0)).sort_index(level='batt_maxtemp_tc')
# display(chk2)


# How close two charging blocks can be and still count as one session
CHARGING_GAP_MERGE_MIN = 15.0   # minutes
SPEED_MOTION_THRESHOLD = 0.5   # km/h, for motion_pct


import pandas as pd
import numpy as np

def _get_charging_mask(df_vid: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series marking rows that belong to a *charging envelope*.

    Priority:
      1) If 'alt_mode' exists:
           any alt_mode starting with 'CHARGING'
           (CHARGING_ACTIVE / CHARGING_MAINTAIN / CHARGING_IDLE)
           is treated as charging.
      2) Else if 'mode' exists:
           mode == 'CHARGING'.
      3) Else: fall back to gun_connection_status / current-based heuristic.
    """
    df_vid = df_vid.copy()

    # --- Preferred: multi-state alt_mode ---
    if "alt_mode" in df_vid.columns:
        return df_vid["alt_mode"].astype(str).str.startswith("CHARGING")

    # --- Fallback: simple mode ---
    if "mode" in df_vid.columns:
        return df_vid["mode"].astype(str).str.upper() == "CHARGING"

    # --- Last resort: gun + current heuristic ---
    gun = pd.Series(False, index=df_vid.index)
    if "gun_connection_status" in df_vid.columns:
        g = df_vid["gun_connection_status"]
        g_num = pd.to_numeric(g, errors="coerce")
        g_str = g.astype(str).str.strip().str.lower()
        gun = (g_num == 1) | g_str.isin({"1", "true", "yes", "y", "connected", "on"})
        gun = gun.fillna(False)

    if "total_battery_current" in df_vid.columns:
        cur = pd.to_numeric(df_vid["total_battery_current"], errors="coerce")
        cur_cond = (cur < -5).fillna(False)
    else:
        cur_cond = pd.Series(False, index=df_vid.index)

    return gun | cur_cond


def _build_charging_sessions_for_vehicle(df_vid: pd.DataFrame) -> list[dict]:
    """
    Builds stitched CHARGING envelopes for a single vehicle.

    A charging envelope is any contiguous region where _get_charging_mask() is True,
    with gaps ≤ CHARGING_GAP_MERGE_MIN minutes merged into a single session.

    Returns a list of dicts:
        { 'start_idx', 'end_idx', 'start_time', 'end_time' }
    """
    df_vid = df_vid.sort_values("timestamp").reset_index(drop=True)
    is_chg = _get_charging_mask(df_vid)

    if not is_chg.any():
        return []

    tmp = df_vid.copy()
    tmp["is_chg"] = is_chg
    tmp["chg_block"] = tmp["is_chg"].ne(tmp["is_chg"].shift()).cumsum()

    raw_blocks: list[dict] = []
    for block_id, g in tmp.groupby("chg_block", sort=True):
        # skip non-charging blocks
        if not g["is_chg"].iloc[0]:
            continue

        start_idx = int(g.index[0])
        end_idx   = int(g.index[-1])

        raw_blocks.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": df_vid.loc[start_idx, "timestamp"],
                "end_time":   df_vid.loc[end_idx,   "timestamp"],
            }
        )

    if not raw_blocks:
        return []

    raw_blocks.sort(key=lambda b: b["start_idx"])

    stitched: list[dict] = []
    current = raw_blocks[0].copy()

    for nxt in raw_blocks[1:]:
        gap_min = (nxt["start_time"] - current["end_time"]).total_seconds() / 60.0
        if gap_min <= CHARGING_GAP_MERGE_MIN:
            # merge into current envelope
            current["end_idx"] = nxt["end_idx"]
            current["end_time"] = nxt["end_time"]
        else:
            stitched.append(current)
            current = nxt.copy()

    stitched.append(current)
    return stitched


def _compute_session_metrics(seg: pd.DataFrame) -> dict:
    """
    Given a contiguous segment (one session) for a single vehicle,
    compute all requested metrics.
    """
    seg = seg.copy()
    seg = seg.sort_values("timestamp")

    if len(seg) < 2:
        return None

    # --- time deltas ---
    seg["dt"] = seg["timestamp"].diff().dt.total_seconds().fillna(0)
    # For safety: zero-out first dt so we don't pull in time before the session
    seg.iloc[0, seg.columns.get_loc("dt")] = 0.0

    total_time = seg["dt"].sum()
    if total_time <= 0:
        total_time = (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds()

    # --- base metadata ---
    id_val = seg["id"].iloc[0]
    reg_num = seg["reg_num"].dropna().iloc[0] if "reg_num" in seg.columns and seg["reg_num"].notna().any() else None
    customer = seg["customer"].dropna().iloc[0] if "customer" in seg.columns and seg["customer"].notna().any() else None
    model = seg["model"].dropna().iloc[0] if "model" in seg.columns and seg["model"].notna().any() else None

    start_time = seg["timestamp"].iloc[0]
    end_time = seg["timestamp"].iloc[-1]
    duration_mins = round((end_time - start_time).total_seconds() / 60.0, 2)

    # --- energy integration ---
    # Ensure numeric
    seg["bat_voltage"] = pd.to_numeric(seg.get("bat_voltage"), errors="coerce")
    seg["total_battery_current"] = pd.to_numeric(seg.get("total_battery_current"), errors="coerce")

    # kW
    seg["power_kw"] = (seg["bat_voltage"] * seg["total_battery_current"]) / 1000.0

    # kWh components (sign-aware)
    # charging (I < 0) → energy INTO pack
    mask_chg = seg["total_battery_current"] < 0
    mask_dis = seg["total_battery_current"] > 0

    seg["energy_kwh_chg"] = 0.0
    seg.loc[mask_chg, "energy_kwh_chg"] = -seg.loc[mask_chg, "power_kw"] * seg.loc[mask_chg, "dt"] / 3600.0

    seg["energy_kwh_dis"] = 0.0
    seg.loc[mask_dis, "energy_kwh_dis"] = seg.loc[mask_dis, "power_kw"] * seg.loc[mask_dis, "dt"] / 3600.0

    kwh_charging = round(seg["energy_kwh_chg"].sum(), 2)
    kwh_discharging = round(seg["energy_kwh_dis"].sum(), 2)

    # --- SOC metrics ---
    soc_col = "bat_soc" if "bat_soc" in seg.columns else None
    soc_start = soc_end = soc_gain = soc_drop = None
    if soc_col:
        soc_valid = seg[soc_col].dropna()
        if not soc_valid.empty:
            soc_start = soc_valid.iloc[0]
            soc_end = soc_valid.iloc[-1]
            soc_gain = max(soc_end - soc_start, 0)
            soc_drop = max(soc_start - soc_end, 0)

    # --- percentage metrics (time-weighted) ---
    def pct(mask: pd.Series) -> float:
        if total_time <= 0:
            return 0.0
        return round(100.0 * seg.loc[mask, "dt"].sum() / total_time, 2)

    # charging / discharging % by current sign
    charging_pct = pct(mask_chg)
    discharging_pct = pct(mask_dis)

    # motion_pct: vehicle_speed_vcu > SPEED_MOTION_THRESHOLD
    if "vehicle_speed_vcu" in seg.columns:
        speed = pd.to_numeric(seg["vehicle_speed_vcu"], errors="coerce")
        motion_pct = pct(speed > SPEED_MOTION_THRESHOLD)
    else:
        motion_pct = np.nan

    # lv_pct and off_pct
    if all(col in seg.columns for col in ["ignitionstatus", "gun_connection_status", "vehiclereadycondition"]):
        ign = pd.to_numeric(seg["ignitionstatus"], errors="coerce")
        gun = pd.to_numeric(seg["gun_connection_status"], errors="coerce")
        ready = pd.to_numeric(seg["vehiclereadycondition"], errors="coerce")

        lv_mask = (ign == 1) & (gun == 0) & (ready == 0)
        off_mask = (ign == 0) & (gun == 0) & (ready == 0)

        lv_pct = pct(lv_mask)
        off_pct = pct(off_mask)
    else:
        lv_pct = off_pct = np.nan

    return {
        "id": id_val,
        "reg_num": reg_num,
        "customer": customer,
        "model": model,
        "start_time": start_time,
        "end_time": end_time,
        "duration_mins": duration_mins,
        "kwh_charging": kwh_charging,
        "kwh_discharging": kwh_discharging,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_gain": soc_gain,
        "soc_drop": soc_drop,
        "charging_pct": charging_pct,
        "discharging_pct": discharging_pct,
        "motion_pct": motion_pct,
        "lv_pct": lv_pct,
        "off_pct": off_pct,
    }


def _build_sessions_for_vehicle(df_vid: pd.DataFrame) -> list[dict]:
    """
    For a single vehicle, build sessions around stitched charging envelopes
    and compute metrics for each slice.

    Session boundaries:
      - Charging envelopes from _build_charging_sessions_for_vehicle()
      - Gaps between them
      - Pre-first and post-last intervals

    Session label:
      - 'activity' is derived from the dominant `alt_mode` inside the slice:
            CHARGING_ACTIVE / CHARGING_MAINTAIN / CHARGING_IDLE
            DISCHARGING_ACTIVE / DISCHARGING_IDLE
        falling back to simple CHARGING/DISCHARGING/UNKNOWN if needed.
    """
    rows: list[dict] = []
    if df_vid.empty:
        return rows

    df_vid = df_vid.sort_values("timestamp").reset_index(drop=True)
    n = len(df_vid)

    charging_sessions = _build_charging_sessions_for_vehicle(df_vid)

    BATT_KWH = 423.96  # for C-rate

    def add_session(start_idx: int, end_idx: int):
        """Slice [start_idx, end_idx] → metrics + activity."""
        if start_idx > end_idx or start_idx < 0 or end_idx >= n:
            return

        seg = df_vid.iloc[start_idx:end_idx + 1].copy()
        if seg.empty:
            return

        metrics = _compute_session_metrics(seg)
        if metrics is None:
            return

        # --- derive activity from alt_mode ---
        activity = None
        if "alt_mode" in seg.columns:
            am = seg["alt_mode"].dropna().astype(str)
            if not am.empty:
                activity = am.value_counts().idxmax()

        if activity is None:
            # fallback: sign of current
            if "total_battery_current" in seg.columns:
                cur = pd.to_numeric(seg["total_battery_current"], errors="coerce")
                if cur.mean(skipna=True) < 0:
                    activity = "CHARGING"
                else:
                    activity = "DISCHARGING"
            else:
                activity = "UNKNOWN"

        # --- C-rate style metrics (same as earlier) ---
        charge_rate = None
        discharge_rate = None

        if "bat_voltage" in seg.columns and "total_battery_current" in seg.columns:
            seg["power_kw"] = (seg["bat_voltage"] * seg["total_battery_current"]) / 1000.0

            chg = seg.loc[seg["power_kw"] < 0, "power_kw"]
            if not chg.empty:
                avg_chg_kw = abs(chg.mean())
                charge_rate = avg_chg_kw / BATT_KWH

            dch = seg.loc[seg["power_kw"] > 0, "power_kw"]
            if not dch.empty:
                avg_dch_kw = dch.mean()
                discharge_rate = avg_dch_kw / BATT_KWH

        metrics["charge_rate"] = round(charge_rate, 3) if charge_rate is not None else None
        metrics["discharge_rate"] = round(discharge_rate, 3) if discharge_rate is not None else None

        metrics["activity"] = activity
        rows.append(metrics)

    # --------------------------
    # build segments around envelopes
    # --------------------------
    if not charging_sessions:
        # whole day is a single discharging-family session
        add_session(0, n - 1)
        return rows

    # pre-first-charging
    first = charging_sessions[0]
    if first["start_idx"] > 0:
        add_session(0, first["start_idx"] - 1)

    # each charging envelope + gap to next
    for i, chg in enumerate(charging_sessions):
        # charging region itself
        add_session(chg["start_idx"], chg["end_idx"])

        # gap (discharging region) to next envelope
        if i < len(charging_sessions) - 1:
            nxt = charging_sessions[i + 1]
            gap_start = chg["end_idx"] + 1
            gap_end = nxt["start_idx"] - 1
            if gap_start <= gap_end:
                add_session(gap_start, gap_end)

    # post-last-charging
    last = charging_sessions[-1]
    if last["end_idx"] < n - 1:
        add_session(last["end_idx"] + 1, n - 1)

    return rows


def enrich_discharging_metrics(session_df, raw_df):
    """
    Adds distance, speed stats, voltage delta stats, temp delta stats,
    kWh/km, and now:
        - odo_start
        - odo_end
    """

    raw_df = raw_df.sort_values(["id", "timestamp"]).copy()

    # ---- Holder columns ----
    session_df["dist_km"] = 0.0
    session_df["avg_speed"] = 0.0
    session_df["med_speed"] = 0.0
    session_df["max_speed"] = 0.0

    session_df["avg_volt_delta_mv"] = 0.0
    session_df["med_volt_delta_mv"] = 0.0
    session_df["max_volt_delta_mv"] = 0.0
    session_df["p95_volt_delta_mv"] = 0.0

    session_df["avg_batt_temp_delta"] = 0.0
    session_df["med_batt_temp_delta"] = 0.0
    session_df["max_batt_temp_delta"] = 0.0
    session_df["p95_batt_temp_delta"] = 0.0

    session_df["energy_active_kwh"] = 0.0
    session_df["kwh_per_km"] = np.nan

    # NEW COLUMNS
    session_df["odo_start"] = np.nan
    session_df["odo_end"]   = np.nan

    # ---- Loop sessions ----
    for idx, row in session_df.iterrows():

        if row["activity"] != "DISCHARGING_ACTIVE":
            continue

        vid = row["id"]
        t1  = row["start_time"]
        t2  = row["end_time"]

        mask = (
            (raw_df["id"] == vid) &
            (raw_df["timestamp"] >= t1) &
            (raw_df["timestamp"] <= t2)
        )
        chunk = raw_df[mask].copy()

        if chunk.empty:
            continue

        # -----------------------
        # NEW: Store raw odo range
        # -----------------------
        odo_vals = chunk["odometer_final"].dropna()
        if not odo_vals.empty:
            session_df.at[idx, "odo_start"] = odo_vals.iloc[0]
            session_df.at[idx, "odo_end"]   = odo_vals.iloc[-1]

        # -----------------------
        # 1. Distance (forward-only)
        # -----------------------
        odo_diff = chunk["odometer_final"].diff()
        dist_km = odo_diff[odo_diff > 0].sum()
        session_df.at[idx, "dist_km"] = dist_km

        # -----------------------
        # 2. Speed stats
        # -----------------------
        v = chunk["vehicle_speed_vcu"].dropna()
        if len(v):
            session_df.at[idx, "avg_speed"] = v.mean()
            session_df.at[idx, "med_speed"] = v.median()
            session_df.at[idx, "max_speed"] = v.max()

        # -----------------------
        # 3. Voltage deltas
        # -----------------------
        vd = chunk["volt_delta_mv"].dropna()
        if len(vd):
            session_df.at[idx, "avg_volt_delta_mv"] = vd.mean()
            session_df.at[idx, "med_volt_delta_mv"] = vd.median()
            session_df.at[idx, "max_volt_delta_mv"] = vd.max()
            session_df.at[idx, "p95_volt_delta_mv"] = vd.quantile(0.95)

        # -----------------------
        # 4. Temp deltas
        # -----------------------
        td = chunk["batt_temp_delta"].dropna()
        if len(td):
            session_df.at[idx, "avg_batt_temp_delta"] = td.mean()
            session_df.at[idx, "med_batt_temp_delta"] = td.median()
            session_df.at[idx, "max_batt_temp_delta"] = td.max()
            session_df.at[idx, "p95_batt_temp_delta"] = td.quantile(0.95)

        # -----------------------
        # 5. Energy integration
        # -----------------------
        chunk["power_kw"] = (chunk["bat_voltage"] * chunk["total_battery_current"]) / 1000.0
        chunk["energy_kwh"] = chunk["power_kw"] * (chunk["dt_sec"] / 3600)

        energy_active_kwh = chunk.loc[chunk["energy_kwh"] > 0, "energy_kwh"].sum()
        session_df.at[idx, "energy_active_kwh"] = energy_active_kwh

        # -----------------------
        # 6. kWh per km
        # -----------------------
        if dist_km > 0:
            session_df.at[idx, "kwh_per_km"] = energy_active_kwh / dist_km

    return session_df


def build_charging_and_discharging_sessions(df_day: pd.DataFrame) -> pd.DataFrame:
    """
    Build unified CHARGING / DISCHARGING sessions for a day's data.
    Also adds bucket distributions and SOC stats.

    Output columns:
        id, reg_num, customer, model, activity, session, date,
        start_time, end_time, duration_mins,
        charging_pct, discharging_pct, motion_pct,
        lv_pct, off_pct,
        kwh_charging, kwh_discharging,
        soc_start, soc_end, soc_gain, soc_drop,
        ... + bucket percentage columns
    """

    if df_day.empty:
        return pd.DataFrame()

    df_day = df_day.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # -----------------------------------------------------------
    # STEP 1 — Build sessions for each vehicle
    # -----------------------------------------------------------
    all_rows = []
    for vid, df_vid in df_day.groupby("id"):
        vid_rows = _build_sessions_for_vehicle(df_vid)   # <-- your existing function
        all_rows.extend(vid_rows)

    if not all_rows:
        return pd.DataFrame()

    # Convert list of dicts into DataFrame
    sessions = pd.DataFrame(all_rows)

    # -----------------------------------------------------------
    # STEP 2 — Sort & assign per-vehicle session number
    # -----------------------------------------------------------
    sessions = sessions.sort_values(["id", "start_time"]).reset_index(drop=True)
    sessions["session"] = sessions.groupby("id").cumcount() + 1
    sessions["date"] = sessions["start_time"].dt.date

    # -----------------------------------------------------------
    # STEP 3 — Push 'session' assignment back into raw df_day rows
    # -----------------------------------------------------------
    df_day["session"] = None

    for ses in sessions.itertuples(index=False):
        mask = (
            (df_day["id"] == ses.id) &
            (df_day["timestamp"] >= ses.start_time) &
            (df_day["timestamp"] <= ses.end_time)
        )
        df_day.loc[mask, "session"] = ses.session

    # -----------------------------------------------------------
    # STEP 4 — Bucket distributions (TEMP, VOLT, SOC)
    # -----------------------------------------------------------
    BUCKET_MAP = {
        'maxtemp_bucket': ["<28", "28–32", "32–35", "35–40", ">40"],
        'temp_delta_bucket': ["<2", "2–5", "5–8", ">8"],
        'volt_delta_bucket': ["0–10", "10–20", "20–30", ">30"],
        'soc_band_bucket': [
            "0–10","10–20","20–30","30–40","40–50",
            "50–60","60–70","70–80","80–90","90–100"
        ],
    }

    for col, categories in BUCKET_MAP.items():
        if col not in df_day.columns:
            continue

        # normalize category labels
        df_day[col] = (
            df_day[col].astype(str).str.replace("-", "–")
        )
        df_day[col] = pd.Categorical(df_day[col], categories=categories, ordered=True)

        # compute percentage distributions
        pct = (
            df_day.groupby("session")[col]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
        )

        pct_pivot = pct.unstack(fill_value=0)

        pct_pivot.columns = [
            f"{col}_{str(c).replace('–','_').replace('<','lt').replace('>','gt')}_pct"
            for c in pct_pivot.columns
        ]

        sessions = sessions.join(pct_pivot, on="session", how="left")

    # -----------------------------------------------------------
    # STEP 5 — Round SOC columns
    # -----------------------------------------------------------
    for col in ["soc_start", "soc_end", "soc_gain", "soc_drop"]:
        if col in sessions.columns:
            sessions[col] = sessions[col].apply(
                lambda x: round(x, 2) if pd.notna(x) else x
            )

    # -----------------------------------------------------------
    # STEP 6 — Final ordering
    # -----------------------------------------------------------
    ordered_cols = [
        "id", "reg_num", "customer", "model",
        "activity", "session", "date",
        "start_time", "end_time", "duration_mins",
        "charging_pct", "discharging_pct", "motion_pct",
        "lv_pct", "off_pct",
        "kwh_charging", "kwh_discharging","charge_rate","discharge_rate",
        "soc_start", "soc_end", "soc_gain", "soc_drop",
    ]

    # keep any extra bucket columns also
    ordered_cols += [c for c in sessions.columns if c not in ordered_cols]

    return sessions[ordered_cols]


def run_multi_day_session_analysis(
    parquet_path: str, 
    start_date: datetime, 
    num_days: int
):
    """
    Iteratively loads, processes, and aggregates session reports
    (CHARGING_* / DISCHARGING_* via `activity`) for a defined
    number of days across all vehicles.
    """
    all_sessions: list[pd.DataFrame] = []
    total_rows_ingested = 0

    current_log_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)

    try:
        progress_bar = tqdm(
            range(num_days),
            desc="Processing Daily Sessions",
            unit="day",
            colour="cyan",
        )

        for i in progress_bar:
            current_start_dt = start_date + timedelta(days=i)
            current_end_dt = current_start_dt + timedelta(days=1)

            progress_bar.set_postfix_str(
                f"Date: {current_start_dt.strftime('%Y-%m-%d')}"
            )

            df_subset = read_parquet_subset(
                parquet_path=parquet_path,
                start_dt=current_start_dt,
                end_dt=current_end_dt,
            )

            if df_subset.empty:
                continue

            total_rows_ingested += len(df_subset)

            # 1) Build unified sessions (activity comes from alt_mode)
            day_sessions = build_charging_and_discharging_sessions(df_subset)

            # 2) Enrich discharging-active sessions with distance / kWh/km
            if not day_sessions.empty:
                day_sessions = enrich_discharging_metrics(day_sessions, df_subset)
                all_sessions.append(day_sessions)

            del df_subset
            gc.collect()

    finally:
        logging.getLogger().setLevel(current_log_level)

    if all_sessions:
        final_df = pd.concat(all_sessions, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df, total_rows_ingested


# =========================================================================
# --- Execution Example ---
# =========================================================================

# Define the 75-day process window
filter_start_date = datetime(2025, 9, 1) # Start date
total_days_to_process = 75

print(f"--- Starting 75-Day Session Analysis ---")
print(f"Processing range: {filter_start_date.strftime('%Y-%m-%d')} to {(filter_start_date + timedelta(days=total_days_to_process-1)).strftime('%Y-%m-%d')}")
print("-" * 40)

start_time = time.time()

charge_discharge_df, total_rows = run_multi_day_session_analysis(
    parquet_path="../df_with_state.parquet",
    start_date=filter_start_date,
    num_days=total_days_to_process
)

charge_discharge_df["start_time"] = charge_discharge_df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
charge_discharge_df["end_time"] = charge_discharge_df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

end_time = time.time()
elapsed_time_sec = end_time - start_time
elapsed_time_min = elapsed_time_sec / 60

# --- Final Summary ---
print("\n" + "=" * 40)
print("✅ ANALYSIS COMPLETE")
print(f"Total time taken: {elapsed_time_sec:.2f} seconds ({elapsed_time_min:.2f} minutes)")
print(f"Days processed:   {total_days_to_process}")
print(f"Total Rows Ingested: {total_rows:,}")
print(f"Final Report Shape: {charge_discharge_df.shape} (Rows: {charge_discharge_df.shape[0]}, Columns: {charge_discharge_df.shape[1]})")
print("=" * 40)


charge_discharge_df[charge_discharge_df.id == '16'].head(30)


charge_discharge_df.to_excel('charge_discharge_analysis.xlsx')

