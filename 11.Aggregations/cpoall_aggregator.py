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
# import duckdb 
import warnings
# import fastparquet
from tqdm import tqdm 
import psutil
import time # For timing the execution
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Tuple, Dict, List, Optional
import argparse
import pytz

sys.path.append('..')
from common import db_operations
from common.db_operations import connect_to_trino, fetch_data_for_day_trino, fetch_distinct_device_ids, fetch_distinct_ids_for_day_trino, write_df_to_iceberg,execute_query

warnings.filterwarnings('ignore')


# Optional: adjust pandas display for debugging; you can comment these out
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')

SESSION_COL_ORDER = [
    # 1) Identity
    "id", #"reg_num", "customer", "model",

    # 2) Session timeline
    "date", "activity", "session",
    "start_time", "end_time", "duration_mins",

    # 3) Odometer & distance
    "odo_start", "odo_end", "dist_km",

    # 5) High-level utilisation
    "motion_pct", "lv_pct", "off_pct","vrc_pct", "gcs_pct",
    "charging_pct", "discharging_pct",

    # 4) Energy & efficiency
    "kwh_charging", "kwh_discharging",
    "energy_active_kwh", "kwh_per_km","net_kwh_per_km",
    "charge_rate", "discharge_rate",

    # 8) SOC range
    "soc_start", "soc_end", "soc_gain", "soc_drop",
    
    # 6) Speed stats
    "avg_speed", "med_speed", "max_speed",

    # 7) Delta stats (voltage & temp)
    "avg_volt_delta_mv", "med_volt_delta_mv", "p95_volt_delta_mv", "max_volt_delta_mv",
    "avg_batt_temp_delta", "med_batt_temp_delta", "p95_batt_temp_delta", "max_batt_temp_delta",

    # 10) Temp buckets
    "maxtemp_bucket_lt28_pct",
    "maxtemp_bucket_28_32_pct",
    "maxtemp_bucket_32_35_pct",
    "maxtemp_bucket_35_40_pct",
    "maxtemp_bucket_gt40_pct",

    # 11) Temp delta buckets
    "temp_delta_bucket_lt2_pct",
    "temp_delta_bucket_2_5_pct",
    "temp_delta_bucket_5_8_pct",
    "temp_delta_bucket_gt8_pct",

    # 12) Voltage delta buckets
    "volt_delta_bucket_0_10_pct",
    "volt_delta_bucket_10_20_pct",
    "volt_delta_bucket_20_30_pct",
    "volt_delta_bucket_gt30_pct",

    # 13) SOC band buckets
    "soc_band_bucket_0_10_pct",
    "soc_band_bucket_10_20_pct",
    "soc_band_bucket_20_30_pct",
    "soc_band_bucket_30_40_pct",
    "soc_band_bucket_40_50_pct",
    "soc_band_bucket_50_60_pct",
    "soc_band_bucket_60_70_pct",
    "soc_band_bucket_70_80_pct",
    "soc_band_bucket_80_90_pct",
    "soc_band_bucket_90_100_pct",

    # 9) Glitch flags
    "net_odo_km", "dist_km_raw",  "max_physical_km",
    "glitch_flag", "glitch_reason",
    
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD (IST)")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD (IST)")
    parser.add_argument("--yesterday", action="store_true", help="Run for yesterday in IST")
    return parser.parse_args()


def resolve_dates(args):
    if args.yesterday:
        now_ist = datetime.now(IST)
        y = (now_ist - timedelta(days=1)).date()
        return str(y), str(y)

    if args.start_date and args.end_date:
        return args.start_date, args.end_date

    raise ValueError("Provide --yesterday OR both --start-date and --end-date")


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
    valid_mask = curr.abs().between(0, 1250)
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

    # -----------------------------------------------------------
    # OPTIONAL: Merge mapping_df (reg_num, customer, model, etc.)
    # -----------------------------------------------------------
    # if df_mapping is not None:
    #     if not isinstance(df_mapping, pd.DataFrame):
    #         raise TypeError("df_mapping must be a DataFrame or None")
    
    #     # convert id to string on both sides
    #     out["id"] = out["id"].astype(str)
    #     df_mapping = df_mapping.copy()
    #     df_mapping["id"] = df_mapping["id"].astype(str)
    
    #     # merge
    #     out = out.merge(df_mapping, on="id", how="left", validate="m:1")
    
    # else:
    #     # Mapping not supplied → do NOT merge anything
    #     # You can decide whether you want to create empty columns or skip them.
    #     # Option A: create empty metadata columns (HARmless)
    #     out["reg_num"] = None
    #     out["customer"] = None
    #     out["model"] = None
    
        # If you prefer NOT to create them, tell me and I’ll remove these lines.

    
    # -----------------------------------------------------------
    # 2. Timestamp handling — THE SAFE VERSION
    # -----------------------------------------------------------

    # --- TIMESTAMP FIX (UTC → IST) ---

    # 1. Parse raw timestamp exactly as received
    out["ts_utc"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    # 2. Drop invalid rows
    out = out.dropna(subset=["ts_utc"])

    # 3. Convert to Asia/Kolkata (IST)
    out["timestamp"] = out["ts_utc"].dt.tz_convert("Asia/Kolkata")

    # 4. Remove timezone info from final timestamp if needed
    #    (matplotlib, parquet, feather become safer with tz-naive)
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)

    # 5. Sort by vehicle + IST timestamp
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
    # out["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")

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
    # chg_maint = (gun_connected &(out["total_battery_current"].abs().between(0, 5))).to_numpy(dtype=bool)
    chg_idle = (gun_connected &(out["total_battery_current"] >= -5)).to_numpy(dtype=bool)

    # # DISCHARGING states
    dis_active = ((~gun_connected) &(out["vehicle_speed_vcu"].gt(0.5).fillna(False)) &(out["gear_position"].isin([1, 2]).fillna(False))).to_numpy(dtype=bool)
    # dis_idle   = ((~gun_connected) & (~dis_active)).to_numpy(dtype=bool)

    out["alt_mode"] = np.select(
        [chg_active, chg_idle, dis_active],
        ["CHARGING_ACTIVE", "CHARGING_IDLE", "DISCHARGING_ACTIVE"],
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
        "id",#"reg_num","customer","model",
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




# ---------------------------------------------------------------------
# 1. Canonical activity state (4 states only)
# ---------------------------------------------------------------------

def add_activity_state(
    df: pd.DataFrame,
    current_col: str = "total_battery_current",
    gun_col: str = "gun_connection_status",
    active_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Add a canonical 4-state 'activity' column based on gun connection
    + total_battery_current sign/magnitude.

    Rules (as per your spec):
      - < 0 A means CHARGING domain
      - > 0 A means DISCHARGING domain
      - When gun_connection_status == 1:
          * Anything that isn't CHARGING_ACTIVE is CHARGING_IDLE
      - DISCHARGING logic mirrors CHARGING, but respects gun=0.

    Final allowed states:
        - CHARGING_ACTIVE
        - CHARGING_IDLE
        - DISCHARGING_ACTIVE
        - DISCHARGING_IDLE
    """
    out = df.copy()

    if gun_col not in out.columns:
        raise KeyError(f"{gun_col!r} not found in DataFrame")

    if current_col not in out.columns:
        raise KeyError(f"{current_col!r} not found in DataFrame")

    gun_raw = out[gun_col]
    gun_num = pd.to_numeric(gun_raw, errors="coerce")
    gun_str = gun_raw.astype(str).str.strip().str.lower()

    gun_connected = (gun_num == 1) | gun_str.isin({"1", "true", "yes", "y", "connected", "on"})
    gun_connected = gun_connected.fillna(False)

    curr = pd.to_numeric(out[current_col], errors="coerce").fillna(0.0)

    # Charging / discharging domains based purely on gun
    is_chg_domain = gun_connected
    is_dis_domain = ~gun_connected

    # Within domains, decide ACTIVE vs IDLE based on sign + magnitude
    # NOTE: <0A => charging, >0A => discharging (per spec)
    chg_active = is_chg_domain & (curr <= -active_threshold)
    chg_idle   = is_chg_domain & ~chg_active

    dis_active = is_dis_domain & (curr >= active_threshold)
    dis_idle   = is_dis_domain & ~dis_active

    activity = np.select(
        [chg_active, chg_idle, dis_active],
        ["CHARGING_ACTIVE", "CHARGING_IDLE", "DISCHARGING_ACTIVE"],
        default="DISCHARGING_IDLE",
    )

    out["activity"] = activity.astype("object")

    return out


def _bucket_label_to_suffix(label: str) -> str:
    """
    Convert bucket labels like '<28', '28–32', '0–10', '>40'
    into suffixes like 'lt28', '28_32', '0_10', 'gt40'.
    """
    if pd.isna(label):
        return "unknown"

    s = str(label)
    # Normalise Unicode dashes
    s = s.replace("–", "-").replace("—", "-")

    if s.startswith("<"):
        return "lt" + s[1:]
    if s.startswith(">"):
        return "gt" + s[1:]

    # Ranges like '28-32' or '0-10'
    s = s.replace(" ", "")
    if "-" in s:
        lo, hi = s.split("-", 1)
        return f"{lo}_{hi}"

    # Fallback
    return s.replace("%", "").replace("+", "p")


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
    # if "mode" in df_vid.columns:
    #     return df_vid["mode"].astype(str).str.upper() == "CHARGING"

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
    # reg_num = seg["reg_num"].dropna().iloc[0] if "reg_num" in seg.columns and seg["reg_num"].notna().any() else None
    # customer = seg["customer"].dropna().iloc[0] if "customer" in seg.columns and seg["customer"].notna().any() else None
    # model = seg["model"].dropna().iloc[0] if "model" in seg.columns and seg["model"].notna().any() else None

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
        vrc_mask = (ready == 1)
        gcs_mask = (gun == 1)
        off_mask = (ign == 0) & (gun == 0) & (ready == 0)

        lv_pct = pct(lv_mask)
        off_pct = pct(off_mask)
        vrc_pct = pct(vrc_mask)
        gcs_pct = pct(gcs_mask)
    else:
        lv_pct = off_pct = np.nan

    return {
        "id": id_val,
        # "reg_num": reg_num,
        # "customer": customer,
        # "model": model,
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
        "vrc_pct": vrc_pct,
        "gcs_pct": gcs_pct,
    }


def _build_sessions_for_vehicle(df_vid: pd.DataFrame) -> list[dict]:
    """
    Build stitched charging + discharging sessions for one vehicle,
    using your original envelope logic but with 4-state activity and
    micro-idle suppression.
    """
    rows: list[dict] = []
    if df_vid.empty:
        return rows

    df_vid = df_vid.sort_values("timestamp").reset_index(drop=True)

    df_vid = df_vid.sort_values("timestamp").reset_index(drop=True)
    
    # STEP 1: activity must already exist
    df_vid = add_activity_state(df_vid)
    
    # STEP 2: NOW apply smoothing to entire day
    df_vid = fix_micro_idle_blips(df_vid, threshold_sec=60)

    n = len(df_vid)
    
    # STEP 3: NOW detect charging envelopes
    charging_sessions = _build_charging_sessions_for_vehicle(df_vid)
    
    # charging_sessions = _build_charging_sessions_for_vehicle(df_vid)
    BATT_KWH = 423.96

    def add_session(start_idx: int, end_idx: int):
        """Slice [start_idx, end_idx] → metrics + activity."""
        if start_idx > end_idx or start_idx < 0 or end_idx >= n:
            return

        seg = df_vid.iloc[start_idx:end_idx + 1].copy()
        if seg.empty:
            return

        # ------------------------------------------
        # Step 1: derive activity (your 4-state logic)
        # ------------------------------------------
        activity = None

        # from alt_mode if available
        if "alt_mode" in seg.columns:
            am = seg["alt_mode"].dropna().astype(str)
            if not am.empty:
                activity = am.value_counts().idxmax()

        # derive gun_connected
        if "gun_connection_status" in seg.columns:
            g = pd.to_numeric(seg["gun_connection_status"], errors="coerce")
            gun_connected = (g == 1).mean() > 0.5
        else:
            gun_connected = False

        # derive mean_current
        if "total_battery_current" in seg.columns:
            cur = pd.to_numeric(seg["total_battery_current"], errors="coerce")
            mean_cur = cur.mean(skipna=True)
        else:
            mean_cur = 0.0

        # your updated rule:
        #   <0A = CHARGING_ACTIVE
        #   >0A = DISCHARGING_ACTIVE
        #   otherwise IDLE depending on gun connection
        if gun_connected:
            if mean_cur < 0:
                activity = "CHARGING_ACTIVE"
            else:
                activity = "CHARGING_IDLE"
        else:
            if mean_cur > 0:
                activity = "DISCHARGING_ACTIVE"
            else:
                activity = "DISCHARGING_IDLE"

        seg["activity"] = activity

        # recompute dominant activity after smoothing
        activity = seg["activity"].value_counts().idxmax()

        # ------------------------------------------
        # Step 3: compute metrics
        # ------------------------------------------
        # Recompute dt_sec fresh for THIS segment ONLY
        seg = seg.copy().sort_values("timestamp").reset_index(drop=True)
        seg["dt_sec"] = seg["timestamp"].diff().dt.total_seconds().fillna(0)
        seg["dt_sec"] = seg["dt_sec"].clip(lower=0)  # just in case
        
        metrics = _compute_session_metrics(seg)
        if metrics is None:
            return

        # C-rate
        charge_rate = 0.0
        discharge_rate = 0.0

        if "bat_voltage" in seg.columns and "total_battery_current" in seg.columns:
            seg["power_kw"] = (seg["bat_voltage"] * seg["total_battery_current"]) / 1000.0

            chg = seg.loc[seg["power_kw"] < 0, "power_kw"]
            if not chg.empty:
                charge_rate = abs(chg.mean()) / BATT_KWH

            dch = seg.loc[seg["power_kw"] > 0, "power_kw"]
            if not dch.empty:
                discharge_rate = dch.mean() / BATT_KWH

        metrics["charge_rate"] = round(charge_rate, 3)
        metrics["discharge_rate"] = round(discharge_rate, 3)

        metrics["activity"] = activity
        rows.append(metrics)

    # ------------------------------------------
    # Envelope stitching logic below remains same
    # ------------------------------------------
    if not charging_sessions:
        add_session(0, n - 1)
        return rows

    first = charging_sessions[0]
    if first["start_idx"] > 0:
        add_session(0, first["start_idx"] - 1)

    for i, chg in enumerate(charging_sessions):
        add_session(chg["start_idx"], chg["end_idx"])

        if i < len(charging_sessions) - 1:
            nxt = charging_sessions[i + 1]
            gap_start = chg["end_idx"] + 1
            gap_end = nxt["start_idx"] - 1
            if gap_start <= gap_end:
                add_session(gap_start, gap_end)

    last = charging_sessions[-1]
    if last["end_idx"] < n - 1:
        add_session(last["end_idx"] + 1, n - 1)

    return rows


def fix_micro_idle_blips(df: pd.DataFrame, threshold_sec: float = 60.0) -> pd.DataFrame:
    """
    Smooth out micro-blips in activity classification.
    
    If there is a short segment of CHARGING_IDLE or DISCHARGING_IDLE
    sandwiched between two segments that belong to the same DOMAIN 
    (charging or discharging), and its duration is below threshold_sec,
    then rewrite its activity to match the surrounding domain.

    This restores continuous charging/discharging sessions and prevents
    fragmentation such as:
        CH_ACTIVE → DIS_IDLE(20s) → CH_ACTIVE.
    """
    if "activity" not in df.columns:
        return df

    out = df.copy()
    
    # ------------------------------------------------------------------
    # Utility: determine domain
    # ------------------------------------------------------------------
    def domain(act: str):
        if act.startswith("CHARGING"):
            return "CH"
        elif act.startswith("DISCHARGING"):
            return "DIS"
        return None

    acts = out["activity"].astype(str).values
    ts = out["timestamp"].values
    n = len(out)

    # compute row-wise durations
    dt = np.zeros(n)
    dt[1:] = (out["timestamp"].iloc[1:].values - out["timestamp"].iloc[:-1].values).astype("timedelta64[s]").astype(float)

    # ------------------------------------------------------------------
    # Scan for 3-point patterns: X → idle-blip → X
    # but broaden to domain-aware detection
    # ------------------------------------------------------------------
    for i in range(1, n - 1):
        a_prev = acts[i-1]
        a_mid  = acts[i]
        a_next = acts[i+1]

        d_prev = domain(a_prev)
        d_mid  = domain(a_mid)
        d_next = domain(a_next)

        # must be same domain before & after
        if d_prev is None or d_next is None:
            continue
        if d_prev != d_next:
            continue

        # middle must be micro-blip AND opposite domain
        if d_mid == d_prev:
            continue  # not a blip

        # micro duration?
        if dt[i] <= threshold_sec:
            # rewrite the blip to match surrounding domain
            if d_prev == "CH":
                acts[i] = "CHARGING_ACTIVE"
            else:
                acts[i] = "DISCHARGING_ACTIVE"

    out["activity"] = acts
    return out


def _append_missing_segment(
    ref_row: pd.Series,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    df_day: pd.DataFrame,
    min_gap_mins: float,
    reason: str,
    new_rows: list,
):
    """Helper: build a MISSING_INFO session for one gap."""
    gap_sec = (gap_end - gap_start).total_seconds()
    gap_mins = gap_sec / 60.0
    if gap_mins < min_gap_mins:
        return

    # choose the right timestamp column (IST preferred)
    ts_col = "timestamp_ist" if "timestamp_ist" in df_day.columns else "timestamp"

    seg = df_day[
        (df_day["id"] == ref_row["id"])
        & (df_day[ts_col] >= gap_start)
        & (df_day[ts_col] < gap_end)
    ].copy()

    # CASE 1: true blackout → no rows inside the gap
    # we only know duration; everything else stays NaN
    if seg.empty:
        new_rows.append({
            "id": ref_row["id"],
            # "reg_num": ref_row.get("reg_num", None),
            # "customer": ref_row.get("customer", None),
            # "model": ref_row.get("model", None),
            "activity": "MISSING_INFO",
            "start_time": gap_start,
            "end_time": gap_end,
            "duration_mins": round(gap_mins, 2),
            # top-level metrics left NaN; bucket % will also be NaN because
            # no rows will carry this session id
            "kwh_charging": np.nan,
            "kwh_discharging": np.nan,
            "charge_rate": np.nan,
            "discharge_rate": np.nan,
            "soc_start": np.nan,
            "soc_end": np.nan,
            "soc_gain": np.nan,
            "soc_drop": np.nan,
            "charging_pct": np.nan,
            "discharging_pct": np.nan,
            "motion_pct": np.nan,
            "lv_pct": np.nan,
            "off_pct": np.nan,
            "vrc_pct": np.nan,
            "gcs_pct": np.nan,
            "glitch_flag": False,
            "glitch_reason": reason,
        })
        return

    # CASE 2: there *are* rows → compute full metrics from seg
    metrics = _compute_session_metrics(seg)
    if metrics is None:
        return

    # override / fill fields
    metrics["id"] = ref_row["id"]
    # metrics["reg_num"] = ref_row.get("reg_num", None)
    # metrics["customer"] = ref_row.get("customer", None)
    # metrics["model"] = ref_row.get("model", None)

    # keep the time window aligned to the gap we defined
    metrics["start_time"] = gap_start
    metrics["end_time"] = gap_end
    metrics["duration_mins"] = round(gap_mins, 2)

    metrics["activity"] = "MISSING_INFO"
    metrics["glitch_flag"] = False
    metrics["glitch_reason"] = reason

    new_rows.append(metrics)


def insert_missing_info_sessions_for_day(
    sessions: pd.DataFrame,
    df_day: pd.DataFrame,
    min_gap_mins: float = 2.0,
) -> pd.DataFrame:
    """
    For each vehicle + day, detect gaps and insert MISSING_INFO sessions.

    - Before first session: [midnight → first.start_time)
    - Between sessions:      [end_time_i → start_time_{i+1})
    - After last session:    [last.end_time → next midnight)

    If there are rows in df_day inside a gap, we compute full metrics
    from those rows; otherwise only duration is known.
    """
    if sessions.empty:
        return sessions

    # infer local day boundaries from df_day
    if "timestamp_ist" in df_day.columns:
        ts = df_day["timestamp_ist"]
    else:
        ts = df_day["timestamp"]
    day_date = ts.dt.date.min()
    day_start = pd.to_datetime(str(day_date))
    day_end = day_start + pd.Timedelta(days=1)

    new_rows = []

    for vid, grp in sessions.groupby("id"):
        grp = grp.sort_values("start_time")

        # pre-first gap
        first = grp.iloc[0]
        _append_missing_segment(
            ref_row=first,
            gap_start=day_start,
            gap_end=first["start_time"],
            df_day=df_day,
            min_gap_mins=min_gap_mins,
            reason="MISSING_INFO_EDGE_START",
            new_rows=new_rows,
        )

        # internal gaps
        for i in range(len(grp) - 1):
            cur = grp.iloc[i]
            nxt = grp.iloc[i + 1]
            _append_missing_segment(
                ref_row=cur,
                gap_start=cur["end_time"],
                gap_end=nxt["start_time"],
                df_day=df_day,
                min_gap_mins=min_gap_mins,
                reason="MISSING_INFO_GAP",
                new_rows=new_rows,
            )

        # post-last gap
        last = grp.iloc[-1]
        _append_missing_segment(
            ref_row=last,
            gap_start=last["end_time"],
            gap_end=day_end,
            df_day=df_day,
            min_gap_mins=min_gap_mins,
            reason="MISSING_INFO_EDGE_END",
            new_rows=new_rows,
        )

    if not new_rows:
        return sessions

    sessions = pd.concat([sessions, pd.DataFrame(new_rows)], ignore_index=True)
    return sessions



def build_charging_and_discharging_sessions(df_day: pd.DataFrame) -> pd.DataFrame:
    """
    Build unified CHARGING / DISCHARGING sessions for a day's data.
    Also adds bucket distributions and SOC stats.

    Output columns:
        id, reg_num, customer, model, activity, session, date,
        start_time, end_time, duration_mins,
        charging_pct, discharging_pct, motion_pct,
        lv_pct, off_pct, vrc_pct, gcs_pct,
        kwh_charging, kwh_discharging,
        soc_start, soc_end, soc_gain, soc_drop,
        ... + bucket percentage columns
    """
    if df_day.empty:
        return pd.DataFrame()

    # -----------------------------------------------------------
    # STEP 1 — Build sessions for each vehicle
    # -----------------------------------------------------------
    # all_rows: list[dict] = []
    # for vid, df_vid in df_day.groupby("id"):
    #     vid_rows = _build_sessions_for_vehicle(df_vid)
    #     all_rows.extend(vid_rows)

    # if not all_rows:
    #     return pd.DataFrame()

    # sessions = pd.DataFrame(all_rows)
    all_rows: list[dict] = []
    for vid, df_vid in df_day.groupby("id"):
        vid_rows = _build_sessions_for_vehicle(df_vid)
        all_rows.extend(vid_rows)

    if not all_rows:
        return pd.DataFrame()

    sessions = pd.DataFrame(all_rows)

    # -----------------------------------------------------------
    # 1B — Drop micro-sessions (< 2 min) for real activity
    # -----------------------------------------------------------
    if "duration_mins" in sessions.columns:
        sessions = sessions[sessions["duration_mins"] >= 2.0].copy()
    if sessions.empty:
        return pd.DataFrame()

    # -----------------------------------------------------------
    # 1C — Insert MISSING_INFO for day edges + internal gaps
    # -----------------------------------------------------------
    sessions = insert_missing_info_sessions_for_day(sessions=sessions,df_day=df_day,min_gap_mins=2.0,)

    # -----------------------------------------------------------
    # 2 — Sort & assign per-vehicle session number
    # -----------------------------------------------------------
    sessions = sessions.sort_values(["id", "start_time"]).reset_index(drop=True)
    sessions["session"] = sessions.groupby("id").cumcount() + 1
    sessions["date"] = sessions["start_time"].dt.date

    # -----------------------------------------------------------
    # STEP 2B — SOC DROP GLITCH DETECTION (unchanged)
    # -----------------------------------------------------------
    THRESH_ACTIVE = 0.0    # no drop allowed during active charging
    THRESH_MAINT  = 0.3    # small balancing jitter allowed
    THRESH_IDLE   = 0.5    # small taper jitter allowed

    if "activity" in sessions.columns:
        sessions["glitch_flag"] = False
        sessions["glitch_reason"] = ""

        for idx, row in sessions.iterrows():
            mode = row["activity"]
            if not isinstance(mode, str):
                continue

            # only apply to charging modes
            if not mode.startswith("CHARGING"):
                continue

            soc_drop = row.get("soc_drop", 0) or 0

            # CHARGING_ACTIVE – absolutely no SOC drop expected
            if mode == "CHARGING_ACTIVE" and soc_drop > THRESH_ACTIVE:
                sessions.at[idx, "glitch_flag"] = True
                sessions.at[idx, "glitch_reason"] = (
                    f"SOC dropped {soc_drop:.2f}% during CHARGING_ACTIVE"
                )
                sessions.at[idx, "activity"] = "GLITCH"
                continue

            # CHARGING_MAINTAIN – small drop allowed
            if mode == "CHARGING_MAINTAIN" and soc_drop > THRESH_MAINT:
                sessions.at[idx, "glitch_flag"] = True
                sessions.at[idx, "glitch_reason"] = (
                    f"SOC dropped {soc_drop:.2f}% during CHARGING_MAINTAIN"
                )
                sessions.at[idx, "activity"] = "GLITCH"
                continue

            # CHARGING_IDLE – small jitter allowed
            if mode == "CHARGING_IDLE" and soc_drop > THRESH_IDLE:
                sessions.at[idx, "glitch_flag"] = True
                sessions.at[idx, "glitch_reason"] = (
                    f"SOC dropped {soc_drop:.2f}% during CHARGING_IDLE"
                )
                sessions.at[idx, "activity"] = "GLITCH"
                continue

    # -----------------------------------------------------------
    # STEP 3 — Push 'session' back into raw df_day rows
    # -----------------------------------------------------------
    df_day = df_day.copy()
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
        df_day[col] = df_day[col].astype(str).str.replace("-", "–")
        df_day[col] = pd.Categorical(df_day[col], categories=categories, ordered=True)

        # per-vehicle, per-session distribution
        pct = (
            df_day
            .dropna(subset=["session"])
            .groupby(["id", "session"])[col]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        pct_pivot = pct.unstack(fill_value=0)

        pct_pivot.columns = [
            f"{col}_{str(c).replace('–','_').replace('<','lt').replace('>','gt')}_pct"
            for c in pct_pivot.columns
        ]

        pct_pivot = pct_pivot.reset_index()
        sessions = sessions.merge(pct_pivot, on=["id", "session"], how="left")
    
    # -----------------------------------------------------------
    # STEP 5 — Final column ordering
    # -----------------------------------------------------------
    sessions["charge_rate"] = sessions["charge_rate"].fillna(0);
    sessions["discharge_rate"] = sessions["discharge_rate"].fillna(0);
    ordered_cols = [
        "id", #"reg_num", "customer", "model",
        "activity", "session", "date",
        "start_time", "end_time", "duration_mins",
        "charging_pct", "discharging_pct", "motion_pct",
        "lv_pct", "off_pct", "vrc_pct", "gcs_pct",
        "kwh_charging", "kwh_discharging", "charge_rate", "discharge_rate",
        "soc_start", "soc_end", "soc_gain", "soc_drop",
    ]

    ordered_cols += [c for c in sessions.columns if c not in ordered_cols]

    return sessions[ordered_cols]



def enrich_discharging_metrics(
    session_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    max_kmph_for_physics: float = 120.0,
    physics_tolerance: float = 1.3,
) -> pd.DataFrame:
    """
    Option 3: physics-aware, GLITCH-capable enrichment.

    Adds per-session:
      - dist_km (final, physics-sanitised)
      - avg_speed, med_speed, max_speed
      - avg/med/max/p95 volt_delta_mv
      - avg/med/max/p95 batt_temp_delta
      - energy_active_kwh, kwh_per_km
      - odo_start, odo_end, net_odo_km
      - dist_km_raw (pre-physics cumulative diffs)
      - max_physical_km
      - glitch_flag (bool) + glitch_reason (text)

    If session_df has an 'activity' column, GLITCH sessions get
    activity="GLITCH".
    """

    # Ensure time alignment
    raw_df = raw_df.sort_values(["id", "timestamp"]).copy()

    # --- Holder columns (base metrics) ---
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
    session_df["kwh_per_km"] = 0.0
    session_df["net_kwh_per_km"] = 0.0    

    # --- NEW diagnostic / physics fields ---
    session_df["odo_start"] = np.nan
    session_df["odo_end"]   = np.nan
    session_df["net_odo_km"] = 0.0      # odo_end - odo_start (clamped ≥ 0)
    session_df["dist_km_raw"] = 0.0     # sum of positive diffs before physics clamp
    session_df["max_physical_km"] = 0.0

    session_df["glitch_flag"] = False
    session_df["glitch_reason"] = ""

    has_activity_col = "activity" in session_df.columns

    # --- Loop through each session ---
    for idx, row in session_df.iterrows():
        vid = row["id"]
        t1  = row["start_time"]
        t2  = row["end_time"]

        # Keep your original intent: only DISCHARGING_ACTIVE are "drive" sessions
        # if has_activity_col and row["activity"] != "DISCHARGING_ACTIVE":
        #     continue

        mask = (
            (raw_df["id"] == vid) &
            (raw_df["timestamp"] >= t1) &
            (raw_df["timestamp"] <= t2)
        )
        chunk = raw_df[mask].copy()

        if chunk.empty:
            continue

        # -------------------------------
        # 0. Choose odometer source
        # -------------------------------
        if "odometer_final" in chunk.columns:
            odo_series = chunk["odometer_final"].astype("float64")
        # else:
        #     odo_series = chunk["odometerreading"].astype("float64")

        odo_series = odo_series.dropna()
        if odo_series.empty:
            # No odo → skip distance & energy, but still do volt/temp/speed
            odo_start = np.nan
            odo_end = np.nan
            net_odo = 0.0
        else:
            odo_start = float(odo_series.iloc[0])
            odo_end   = float(odo_series.iloc[-1])
            net_odo   = max(odo_end - odo_start, 0.0)

        session_df.at[idx, "odo_start"] = odo_start
        session_df.at[idx, "odo_end"]   = odo_end
        session_df.at[idx, "net_odo_km"] = net_odo

        # -------------------------------
        # 1. Distance via forward-only diffs (raw)
        # -------------------------------
        if "odometer_final" in chunk.columns:
            odo_full = chunk["odometer_final"].astype("float64")
        # else:
        #     odo_full = chunk["odometerreading"].astype("float64")

        odo_diff = odo_full.diff()
        # Keep only strictly positive increments
        dist_km_raw = odo_diff[odo_diff > 0].sum(skipna=True)
        if pd.isna(dist_km_raw):
            dist_km_raw = 0.0
        session_df.at[idx, "dist_km_raw"] = float(dist_km_raw)
        
        # --------------------------------------------------------------
        # 2) Speed stats – for ALL sessions
        # --------------------------------------------------------------
        v_all = chunk["vehicle_speed_vcu"].astype("float64")
        v_all = v_all.replace([np.inf, -np.inf], np.nan).dropna()
    
        if not v_all.empty:
            # 1) Average over all samples (including stops)
            avg_speed = float(v_all.mean())
    
            # 2) Median of *moving* speeds (>0.1 km/h); fall back to global median
            v_mov = v_all[v_all > 0.1]
            if not v_mov.empty:
                med_speed = float(v_mov.median())
            else:
                med_speed = float(v_all.median())
    
            # 3) Max speed
            max_speed = float(v_all.max())
        else:
            avg_speed = med_speed = max_speed = 0.0
    
        session_df.at[idx, "avg_speed"] = round(avg_speed, 2)
        session_df.at[idx, "med_speed"] = round(med_speed, 2)
        session_df.at[idx, "max_speed"] = round(max_speed, 2)

        # -------------------------------
        # 3. Voltage delta stats
        # -------------------------------
        vd = chunk["volt_delta_mv"].dropna()
        if not vd.empty:
            session_df.at[idx, "avg_volt_delta_mv"] = round(float(vd.mean()), 2)
            session_df.at[idx, "med_volt_delta_mv"] = round(float(vd.median()), 2)
            session_df.at[idx, "max_volt_delta_mv"] = round(float(vd.max()),2)
            session_df.at[idx, "p95_volt_delta_mv"] = round(float(vd.quantile(0.95)), 2)

        # -------------------------------
        # 4. Temperature delta stats
        # -------------------------------
        td = chunk["batt_temp_delta"].dropna()
        if not td.empty:
            session_df.at[idx, "avg_batt_temp_delta"] = round(float(td.mean()),2)
            session_df.at[idx, "med_batt_temp_delta"] = round(float(td.median()),2)
            session_df.at[idx, "max_batt_temp_delta"] = round(float(td.max()),2)
            session_df.at[idx, "p95_batt_temp_delta"] = round(float(td.quantile(0.95)), 2)

        # -------------------------------
        # 5. Energy integration (kWh)
        # -------------------------------
        # Power (kW) = V * I / 1000
        # Energy (kWh) = Σ power * (dt_sec / 3600)
        chunk["power_kw"] = round((
            chunk["bat_voltage"].astype("float64") *
            chunk["total_battery_current"].astype("float64")
        ) / 1000.0, 2)

        chunk["energy_kwh"] = round(chunk["power_kw"] * (
            chunk["dt_sec"].astype("float64") / 3600.0
        ), 2)

        energy_active_kwh = chunk.loc[chunk["energy_kwh"] > 0, "energy_kwh"].sum()
        if pd.isna(energy_active_kwh):
            energy_active_kwh = 0.0

        session_df.at[idx, "energy_active_kwh"] = round(float(energy_active_kwh), 2)
        
        # -------------------------------
        # 6. Physics model: max possible km
        # -------------------------------
        ts_min = chunk["timestamp"].min()
        ts_max = chunk["timestamp"].max()
        if pd.isna(ts_min) or pd.isna(ts_max):
            duration_hr = 0.0
        else:
            duration_sec = (ts_max - ts_min).total_seconds()
            duration_hr = max(duration_sec / 3600.0, 0.0)

        # cap median speed by max_kmph_for_physics
        eff_avg_speed = min((max_speed+med_speed)/2, max_kmph_for_physics)
        max_physical_km = eff_avg_speed * duration_hr * physics_tolerance

        session_df.at[idx, "max_physical_km"] = round(float(max_physical_km), 3)

        # -------------------------------
        # 7. GLITCH detection (Option 3)
        # -------------------------------
        glitch_flag = False
        reasons = []

        eps = 1e-6

        # A: net odometer itself exceeds physics limit
        if net_odo > max_physical_km + eps:
            glitch_flag = True
            reasons.append(
                f"net_odo {net_odo:.3f}km > max_phys {max_physical_km:.3f}km"
            )

        # B: raw cumulative distance exceeds physics limit dramatically
        if dist_km_raw > max_physical_km + eps:
            glitch_flag = True
            reasons.append(
                f"dist_km_raw {dist_km_raw:.3f}km > max_phys {max_physical_km:.3f}km"
            )

        # C: backward odometer (shouldn't happen after your finaliser, but guard anyway)
        if odo_end is not np.nan and odo_start is not np.nan and odo_end + eps < odo_start:
            glitch_flag = True
            reasons.append(
                f"odo_end {odo_end:.3f} < odo_start {odo_start:.3f}"
            )

        # -------------------------------
        # 8. Final distance selection
        # -------------------------------
        # Start with raw cumulative
        dist_final = float(dist_km_raw)

        # If raw cumulative is significantly higher than net change, it's jitter
        if net_odo > 0 and dist_final > net_odo * physics_tolerance:
            reasons.append(
                f"dist_km_raw {dist_final:.3f}km >> net_odo {net_odo:.3f}km, using net_odo"
            )
            dist_final = float(net_odo)

        # If GLITCH due to physics but net_odo is still sane, keep net_odo as best guess
        if glitch_flag:
            if net_odo <= max_physical_km + eps:
                dist_final = float(net_odo)
            else:
                # Completely impossible → distance is untrustworthy
                dist_final = 0.0

        session_df.at[idx, "dist_km"] = dist_final


        # 5b. Net energy over the session: (discharging - charging)
        kwh_dis = session_df.at[idx, "kwh_discharging"] if "kwh_discharging" in session_df.columns else 0.0
        kwh_chg = session_df.at[idx, "kwh_charging"] if "kwh_charging" in session_df.columns else 0.0
    
        # robust against NaN
        if pd.isna(kwh_dis):
            kwh_dis = 0.0
        if pd.isna(kwh_chg):
            kwh_chg = 0.0
    
        net_energy_kwh = kwh_dis - kwh_chg
    
        # net kWh/km using the *final* distance choice
        if dist_final > 20:
            session_df.at[idx, "net_kwh_per_km"] = round(float(net_energy_kwh / dist_final), 2)
        else:
            session_df.at[idx, "net_kwh_per_km"] = 0.0

        
        # -------------------------------
        # 9. kWh/km using final distance
        # -------------------------------
        if dist_final > 0:
            session_df.at[idx, "kwh_per_km"] = round(float(energy_active_kwh / dist_final),2)
        else:
            session_df.at[idx, "kwh_per_km"] = 0

        # -------------------------------
        # 10. Persist GLITCH info
        # -------------------------------
        if glitch_flag:
            session_df.at[idx, "glitch_flag"] = True
            session_df.at[idx, "glitch_reason"] = "; ".join(reasons)
            if has_activity_col:
                session_df.at[idx, "activity"] = "GLITCH"

    return session_df



# ---------------------------------------------------------------------
# 3. Fleet-level temp summary (used for heatmaps, etc.)
# ---------------------------------------------------------------------

def build_fleet_temp_table(
    sessions_df: pd.DataFrame,
    threshold_pct: float = 40.0,
) -> pd.DataFrame:
    """
    Build a per-vehicle summary table from the session-level DataFrame
    (output of build_charging_and_discharging_sessions).

    For CHARGING_ACTIVE sessions:
      - Computes % of sessions where
            (maxtemp_bucket_35_40_pct + maxtemp_bucket_gt40_pct) > threshold_pct
      - Computes mean % time >35°C across all charging-active sessions.

    Returns a DataFrame with:
        id, reg_num, customer, model,
        num_sessions, num_hot_sessions,
        hot_session_pct,
        mean_pct_above35
    """
    if sessions_df.empty:
        return pd.DataFrame()

    df = sessions_df.copy()
    df["id"] = df["id"].astype("int32", errors="ignore")

    d = df[df["activity"] == "CHARGING_ACTIVE"].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "id", #"reg_num", "customer", "model",
            "num_sessions", "num_hot_sessions",
            "hot_session_pct", "mean_pct_above35",
        ])

    d["maxtemp_bucket_35_40_pct"] = d.get("maxtemp_bucket_35_40_pct", 0.0)
    d["maxtemp_bucket_gt40_pct"] = d.get("maxtemp_bucket_gt40_pct", 0.0)

    d["pct_above_35"] = (
        d["maxtemp_bucket_35_40_pct"].fillna(0.0)
        + d["maxtemp_bucket_gt40_pct"].fillna(0.0)
    )

    d["is_hot"] = d["pct_above_35"] > threshold_pct

    agg = (
        d.groupby(["id"])#, "reg_num", "customer", "model"])
        .agg(
            num_sessions=("session", "nunique"),
            num_hot_sessions=("is_hot", "sum"),
            mean_pct_above35=("pct_above_35", "mean"),
        )
        .reset_index()
    )

    agg["hot_session_pct"] = (
        agg["num_hot_sessions"] * 100.0 / agg["num_sessions"].clip(lower=1)
    )

    return agg


def run_daily_tms_analysis(
    df_raw: pd.DataFrame,
    df_mapping: Optional[pd.DataFrame] = None,
    max_gap_sec: float = 600.0,
    active_threshold: float = 5.0,
):
    """
    Full daily pipeline
    1. rename columns
    2. impute missing values
    3. odometer repair
    4. finalize odometer
    5. state preparation
    6. tms session extraction
    7. fleet summaries
    """

    # --------------------------------------------------------------
    # 1) RENAME RAW COLUMNS → canonical battery column names
    # --------------------------------------------------------------
    df = rename_battery_temp_columns(df_raw)

    # --------------------------------------------------------------
    # 2) IMPUTE missing values (temperature, voltage, etc.)
    # --------------------------------------------------------------
    df = impute_missing_values(df)

    # --------------------------------------------------------------
    # 3) IMPUTE ODOMETER intelligently using speed + interpolation
    # --------------------------------------------------------------
    df = impute_odometer(df)

    # --------------------------------------------------------------
    # 4) FINALIZE ODOMETER (clip negatives, fix small reversals, etc.)
    # --------------------------------------------------------------
    df = finalize_odometer(df)

    # --------------------------------------------------------------
    # 5) PREPARE STATE (temp buckets, volt buckets, dt_sec, activity, etc.)
    # --------------------------------------------------------------
    df_state = prepare_df_with_state(df, df_mapping)

    # --------------------------------------------------------------
    # 6) EXTRACT TMS SESSIONS
    # --------------------------------------------------------------
    sessions = build_charging_and_discharging_sessions(df_state)
    
    # 🔹 Enrich discharging sessions with distance / kWh/km
    if not sessions.empty:
        sessions = enrich_discharging_metrics(sessions, df_state)
        # 🔹 Enforce canonical column ordering here
        cols_present = [c for c in SESSION_COL_ORDER if c in sessions.columns]
        other_cols = [c for c in sessions.columns if c not in cols_present]
        sessions = sessions[cols_present + other_cols]        


   # 👇👇 NEW: post-process blackout MISSING_INFO rows only
    if not sessions.empty:
        blackout_mask = (
            (sessions["activity"] == "MISSING_INFO")
            & (sessions["kwh_charging"].isna())
        )

        # Choose which columns you want to normalise
        numeric_cols = [
            c for c in sessions.columns
            if c not in [
                "id",
                # "reg_num",
                # "customer",
                # "model",
                "activity",
                "session",
                "start_time",
                "end_time",
                "glitch_reason",
                "date",
            ]
        ]

        # If you want them as 0.0 instead of NULL:
        sessions.loc[blackout_mask, numeric_cols] = (
            sessions.loc[blackout_mask, numeric_cols].fillna(0.0)
        )    
    # --------------------------------------------------------------
    # 7) FLEET-LEVEL SUMMARY TABLE
    # --------------------------------------------------------------
    fleet_temp = build_fleet_temp_table(sessions)
    
    return {
        "df_state": df_state,
        "sessions": sessions,
        "fleet_temp": fleet_temp,
    }

def run_multi_day_tms_analysis_memory_only(
    conn,
    start_date: str,
    num_days: int,
    df_mapping: Optional[pd.DataFrame],
    core_cols: list,
    schema: str = "facts_prod",
    table: str = "can_parsed_output_all",
):
    """
    Multi-day wrapper around the daily TMS pipeline.

    - start_date is an IST calendar date string, e.g. '2025-09-01'
    - num_days is how many IST days to process
    - Data is fetched in UTC using the correct IST→UTC window.
    """
    ids = fetch_distinct_device_ids(conn, schema=schema, table=table)
    print(f"Found {len(ids)} device IDs")

    start_dt = pd.to_datetime(start_date).date()
    results_all_days = []

    for offset in range(num_days):
        day = start_dt + timedelta(days=offset)
        day_str = day.strftime("%Y-%m-%d")

        print(f"\n=== Processing IST day {day_str} ===")

        # 🔹 Use the IST-aware fetch
        df_raw = fetch_data_for_day_trino(
            conn=conn,
            day_str=day_str,           # <-- matches fetch_data_for_day signature
            ids=ids,
            core_cols=core_cols,
            table=table,
            schema=schema,
        )

        if df_raw.empty:
            print(f"No data for IST day {day_str}")
            results_all_days.append({
                "date": day_str,
                "state_df": pd.DataFrame(),
                "sessions_df": pd.DataFrame(),
                "fleet_df": pd.DataFrame(),
            })
            continue

        # --- NEW: per-day NO DATA log ----------------------------------
        present_ids = set(df_raw["id"].astype(str).unique())
        missing_ids = [vid for vid in ids if vid not in present_ids]

        if missing_ids:
            print(
                f"  → {len(missing_ids)} IDs had NO DATA on {day_str} "
                f"(showing first 10): {missing_ids[:10]}"
            )
        else:
            print(f"  → All {len(ids)} IDs present on {day_str}")
        # ---------------------------------------------------------------
        
        # Build IST timestamp & day column for downstream logic
        # If timestamp is naive UTC, localize first
        if not pd.api.types.is_datetime64tz_dtype(df_raw["timestamp"].dtype):
            df_raw["timestamp"] = df_raw["timestamp"].dt.tz_localize("UTC")

        df_raw["timestamp_ist"] = df_raw["timestamp"].dt.tz_convert("Asia/Kolkata")
        df_raw["date_val"] = df_raw["timestamp_ist"].dt.date

        # Your existing daily pipeline, which ultimately calls prepare_df_with_state,
        # session builder, fleet table, etc.
        # may be None, prepare_df_with_state handles it
        daily = run_daily_tms_analysis(df_raw=df_raw,df_mapping=df_mapping,  )

        results_all_days.append({"date": day_str,"state_df": daily["df_state"],"sessions_df": daily["sessions"],"fleet_df": daily["fleet_temp"],})

        del df_raw, daily
        gc.collect()

    return {"daily": results_all_days, "device_ids": ids}


