from pathlib import Path

import numpy as np
import pandas as pd
import logging
# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from bcs_tms_core import (
    CORE_COLS,
    DEFAULT_VEHICLE_IDS,
    compute_utc_window_from_ist,
    free_mem,
)


def rename_battery_temp_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "pack1_cellmax_temperature": "batt_maxtemp",
        "pack1_cell_min_temperature": "batt_mintemp",
        "pack1_maxtemperature_cell_number":"batt_maxtemp_pack", 
        "pack1_celltemperature_cellnumber":"batt_mintemp_pack",
        "cell_max_voltage":"batt_maxvolt",
        "cellmax_voltagecellnumber":"batt_maxvolt_cell",
        "cell_min_voltage":"batt_minvolt",
        "cellminvoltagecellnumber":"batt_minvolt_cell", 
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if not existing:
        return df
    return df.rename(columns=existing)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "timestamp"])
    for vid, grp in df.groupby("id", sort=False):
        idx = grp.index
        if "batt_maxtemp" in df.columns:
            df.loc[idx, "batt_maxtemp"] = grp["batt_maxtemp"].ffill(limit=60)
        if "batt_mintemp" in df.columns:
            df.loc[idx, "batt_mintemp"] = grp["batt_mintemp"].ffill(limit=60)
        if "batt_maxvolt" in df.columns:
            df.loc[idx, "batt_maxvolt"] = grp["batt_maxvolt"].ffill(limit=30)
        if "batt_minvolt" in df.columns:
            df.loc[idx, "batt_minvolt"] = grp["batt_minvolt"].ffill(limit=30)
        if "bat_voltage" in df.columns:
            df.loc[idx, "bat_voltage"] = grp["bat_voltage"].ffill(limit=20)
        if "bat_soc" in df.columns:
            df.loc[idx, "bat_soc"] = grp["bat_soc"].ffill(limit=300)
        if "soh" in df.columns:
            df.loc[idx, "soh"] = grp["soh"].ffill(limit=300)
        if "total_battery_current" in df.columns:
            df.loc[idx, "total_battery_current"] = grp["total_battery_current"].interpolate(
                limit=10, limit_direction="both"
            )
        if "vehiclereadycondition" in df.columns:
            df.loc[idx, "vehiclereadycondition"] = grp["vehiclereadycondition"].ffill()
        if "gun_connection_status" in df.columns:
            df.loc[idx, "gun_connection_status"] = grp["gun_connection_status"].ffill()
    return df


def prepare_df_with_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw CAN data and compute:
    - timestamp sorting
    - MODE (CHARGING / DISCHARGING)
    - temp / voltage deltas
    - dt_sec between samples
    Returns a compact df_with_state used by later stages.
    """
    out = df.copy()

    # --- timestamp & sorting ---
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = (
        out.dropna(subset=["timestamp"])
           .sort_values(["id", "timestamp"])
           .reset_index(drop=True)
    )

    # --- MODE computation (ALWAYS create 'mode') ---
    if "gun_connection_status" in out.columns:
        gcs_raw = out["gun_connection_status"]
        gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
        gcs_str = gcs_raw.astype(str).str.strip().str.lower()

        gun_connected = (gcs_num == 1) | gcs_str.isin(
            {"1", "true", "yes", "y", "connected", "on"}
        )
        out["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")
    else:
        logging.warning(
            "gun_connection_status missing in input → setting mode='DISCHARGING' for all rows."
        )
        out["mode"] = "DISCHARGING"

    # --- numeric conversions for safety ---
    for col in ["batt_maxtemp", "batt_mintemp", "batt_maxvolt", "batt_minvolt"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- derived fields ---
    if {"batt_maxtemp", "batt_mintemp"} <= set(out.columns):
        out["batt_temp_delta"] = out["batt_maxtemp"] - out["batt_mintemp"]
    else:
        out["batt_temp_delta"] = np.nan

    if {"batt_maxvolt", "batt_minvolt"} <= set(out.columns):
        out["volt_delta_mv"] = (out["batt_maxvolt"] - out["batt_minvolt"]) * 1000.0
    else:
        out["volt_delta_mv"] = np.nan

    out["dt_sec"] = (
        out.groupby("id")["timestamp"]
           .diff()
           .dt.total_seconds()
           .fillna(0)
    )

    # --- final column selection ---
    cols_keep = [
        "id", "timestamp", "mode",
        "vehiclereadycondition", "gun_connection_status",
        "batt_maxtemp", "batt_mintemp", "batt_temp_delta",
        "batt_maxvolt", "batt_minvolt", "volt_delta_mv",
        "batt_maxtemp_pack", "batt_mintemp_pack",
        "batt_maxvolt_cell", "batt_minvolt_cell",
        "bat_voltage", "total_battery_current",
        "bat_soc", "soh", "dt_sec",
    ]
    cols_keep = [c for c in cols_keep if c in out.columns]

    out = out[cols_keep]

    # quick sanity log (optional)
    logging.info("prepare_df_with_state: output columns = %s",", ".join(out.columns))

    return out



def run_stage_1(
    date_str: str = "2025-10-01",
    days: int = 31,
    csv_path: str | None = None,
    feather_path: str | None = None,
    vehicle_ids: list[str] | None = None,
) -> str:
    """
    Stage 1:
    - Load raw monthly CSV
    - Filter by date + vehicle
    - Impute
    - Compute deltas
    - Save df_with_state_30days.feather
    """
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent

    if csv_path is None:
        csv_path = str(parent_dir / "oct25_can_parsed_data.csv")
    if feather_path is None:
        feather_path = str(base_dir / "df_with_state_30days.feather")
    if vehicle_ids is None:
        vehicle_ids = DEFAULT_VEHICLE_IDS

    utc_start, utc_end = compute_utc_window_from_ist(date_str, days)

    # Load header to find usable columns
    head_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in CORE_COLS if c in head_cols]

    df_cpo100 = pd.read_csv(csv_path, usecols=usecols)
    df_cpo100 = rename_battery_temp_columns(df_cpo100)

    if "id" in df_cpo100.columns:
        df_cpo100["id"] = df_cpo100["id"].astype(str)
        df_cpo100 = df_cpo100[df_cpo100["id"].isin(vehicle_ids)]

    df_cpo100["timestamp"] = pd.to_datetime(df_cpo100["timestamp"], errors="coerce")
    df_cpo100 = df_cpo100.dropna(subset=["timestamp"])
    df_cpo100 = df_cpo100[
        (df_cpo100["timestamp"] >= utc_start) &
        (df_cpo100["timestamp"] <= utc_end)
    ]

    df_cpo100 = impute_missing_values(df_cpo100)
    df_with_state = prepare_df_with_state(df_cpo100)

    print("\n=== DEBUG: df_with_state COLUMNS ===")
    print(df_with_state.columns.tolist())

    print("\n=== DEBUG: MODE VALUE COUNTS ===")
    if "mode" in df_with_state.columns:
        print(df_with_state["mode"].value_counts(dropna=False))
    else:
        print("❌ MODE COLUMN IS MISSING HERE")

    df_with_state.to_feather(feather_path)

    del df_cpo100
    del df_with_state
    free_mem()

    print(f"Stage 1 complete → {feather_path}")
    return feather_path
