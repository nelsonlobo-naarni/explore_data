# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: naarni_env (3.13.7)
#     language: python
#     name: python3
# ---


# # BCS / TMS Analysis – Staged, Memory-Safe Pipeline
#
# This notebook restructures the original `3.bcs_tms_analysis.py` into
# **stages**, so that:
#
# - Large DataFrames are created once per stage.
# - Intermediate results are saved to disk (Feather/CSV) and reloaded later.
# - We explicitly delete big objects and trigger garbage collection.
# - PDF/table generation releases figure memory after use.
#
# Stages:
# 1. Load raw monthly CSV → preprocess → add state → save `df_with_state.feather`
# 2. Battery condition analysis (vehicle-wise + fleet) using `df_with_state.feather`
# 3. SoC session analysis + SoC accuracy PDF using `df_with_state.feather`
# 4. Time-weighted energy metrics using saved SoC sessions
#
# Run one stage at a time to keep memory low on an 8 GB M1.
import os
import sys
import gc
import ctypes
import numpy as np
import pandas as pd
import platform
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime, timedelta


pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 50)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
print(f"Using Python version: {platform.python_version()}")


repo_path = '/Users/apple/Documents/naarni/repo/dview-naarni-data-platform'
if repo_path not in sys.path:
    sys.path.append(os.path.join(repo_path, 'tasks'))


from common.db_operations import connect_to_trino, fetch_data_for_day



def free_mem():
    """Try to return freed memory back to the OS (no-op on some platforms)."""
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass



CORE_COLS = [
    "id", "timestamp", "dt",
    "vehiclereadycondition", "gun_connection_status", "ignitionstatus",
    "vehicle_speed_vcu", "gear_position",
    "bat_soc", "soh", "total_battery_current",
    "pack1_cellmax_temperature", "pack1_cell_min_temperature",
    "pack1_maxtemperature_cell_number", "pack1_celltemperature_cellnumber",
    "bat_voltage", "cellmax_voltagecellnumber", "cellmax_voltagecellnumber",
    "cellminvoltagecellnumber", "cell_min_voltage",
    "dcdcbus",
]



def rename_battery_temp_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "pack1_cellmax_temperature": "batt_maxtemp",
        "pack1_cell_min_temperature": "batt_mintemp",
        "pack1_maxtemperature_cell_number":"batt_maxtemp_pack", 
        "pack1_celltemperature_cellnumber":"batt_mintemp_pack",
        "batt_maxvolt":"batt_maxvolt",
        "cellmax_voltagecellnumber":"batt_maxvolt_cell",
        "cell_min_voltage":"batt_minvolt",
        "cellminvoltagecellnumber":"batt_minvolt_cell", 
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if not existing:
        logging.warning("No matching temperature columns found to rename.")
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
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values(["id", "timestamp"]).reset_index(drop=True)
    gcs_raw = out["gun_connection_status"]
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.strip().str.lower()
    gun_connected = (gcs_num == 1) | gcs_str.isin({"1", "true", "yes", "y", "connected", "on"})
    out["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")
    out["batt_maxtemp"] = pd.to_numeric(out.get("batt_maxtemp"), errors="coerce")
    out["batt_mintemp"] = pd.to_numeric(out.get("batt_mintemp"), errors="coerce")
    out["batt_maxvolt"] = pd.to_numeric(out.get("batt_maxvolt"), errors="coerce")
    out["batt_minvolt"] = pd.to_numeric(out.get("batt_minvolt"), errors="coerce")
    out["batt_temp_delta"] = out["batt_maxtemp"] - out["batt_mintemp"]
    out["volt_delta_mv"] = (out["batt_maxvolt"] - out["batt_minvolt"]) * 1000.0
    out["dt_sec"] = out.groupby("id")["timestamp"].diff().dt.total_seconds().fillna(0)
    cols_keep = [
        "id", "timestamp", "mode",
        "vehiclereadycondition", "gun_connection_status",
        "batt_maxtemp", "batt_mintemp", "batt_temp_delta",
        "batt_maxvolt", "batt_minvolt", "volt_delta_mv",
        "batt_maxtemp_pack","batt_mintemp_pack",
        "batt_maxvolt_cell","batt_minvolt_cell",
        "bat_voltage", "total_battery_current",
        "bat_soc", "soh", "dt_sec",
    ]
    cols_keep = [c for c in cols_keep if c in out.columns]
    out = out[cols_keep]
    return out



def analyze_battery_conditions_vehiclewise(df: pd.DataFrame,
                                           output_pdf: str = "battery_conditions_by_vehicle.pdf"):
    df = df.copy()
    df["temp_delta"] = df["batt_maxtemp"] - df["batt_mintemp"]
    df["volt_delta_mv"] = (df["batt_maxvolt"] - df["batt_minvolt"]) * 1000
    temp_bins = [-np.inf, 28, 32, 35, 40, np.inf]
    temp_labels = ["<28", "28–32", "32–35", "35–40", ">40"]
    delta_bins = [-np.inf, 2, 5, 8, np.inf]
    delta_labels = ["<2", "2–5", "5–8", ">8"]
    volt_bins = [0, 10, 20, 30, np.inf]
    volt_labels = ["0–10", "10–20", "20–30", ">30"]
    df["temp_bucket"] = pd.cut(df["batt_maxtemp"], bins=temp_bins, labels=temp_labels)
    df["temp_delta_bucket"] = pd.cut(df["temp_delta"], bins=delta_bins, labels=delta_labels)
    df["volt_delta_bucket"] = pd.cut(df["volt_delta_mv"], bins=volt_bins, labels=volt_labels)
    vehicle_results = {}
    with PdfPages(output_pdf) as pdf:
        for vid, group in df.groupby("id"):
            mode_results = {}
            for mode, subset in group.groupby("mode"):
                mode_results[mode] = {
                    "Battery Max Temp (%)": (subset["temp_bucket"].value_counts(normalize=True) * 100).round(2),
                    "ΔT (°C) Range (%)": (subset["temp_delta_bucket"].value_counts(normalize=True) * 100).round(2),
                    "Voltage Δ (mV) (%)": (subset["volt_delta_bucket"].value_counts(normalize=True) * 100).round(2),
                }
            temp_df = pd.concat({m: r["Battery Max Temp (%)"] for m, r in mode_results.items()}, axis=1).fillna(0)
            delta_df = pd.concat({m: r["ΔT (°C) Range (%)"] for m, r in mode_results.items()}, axis=1).fillna(0)
            volt_df = pd.concat({m: r["Voltage Δ (mV) (%)"] for m, r in mode_results.items()}, axis=1).fillna(0)
            vehicle_results[vid] = {"temp_df": temp_df, "delta_df": delta_df, "volt_df": volt_df}
            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Vehicle ID: {vid}", fontsize=14, fontweight="bold")
            def draw(ax, df_table, title):
                ax.axis("off")
                ax.set_title(title, fontsize=11, pad=10)
                tbl = ax.table(
                    cellText=df_table.values,
                    rowLabels=df_table.index,
                    colLabels=df_table.columns,
                    cellLoc="center",
                    loc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(7.5)
                tbl.scale(1.1, 1.2)
            draw(axes[0], temp_df, "Battery Max Temperature Distribution (%)")
            draw(axes[1], delta_df, "Temperature Delta (°C) Distribution (%)")
            draw(axes[2], volt_df, "Voltage Delta (mV) Distribution (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()
    logging.info(f"✅ Battery Conditions vehiclewise PDF saved → {output_pdf}")
    return vehicle_results



def compute_fleet_summary(vehicle_results: dict, mode_agnostic: bool = False):
    temp_list, delta_list, volt_list = [], [], []
    for vid, res in vehicle_results.items():
        temp = res["temp_df"]
        delt = res["delta_df"]
        volt = res["volt_df"]
        if mode_agnostic:
            temp_list.append(temp.sum(axis=1))
            delta_list.append(delt.sum(axis=1))
            volt_list.append(volt.sum(axis=1))
        else:
            temp_list.append(temp)
            delta_list.append(delt)
            volt_list.append(volt)
    def combine_mode_wise(frames):
        combined = pd.concat(frames, axis=0)
        summed = combined.groupby(combined.index).sum()
        normalized = (summed.div(summed.sum()) * 100).round(2)
        return normalized
    def combine_mode_agnostic(frames):
        s = pd.concat(frames, axis=1).sum(axis=1)
        out = (s / s.sum() * 100).round(2).to_frame("Fleet %")
        return out
    if mode_agnostic:
        return {
            "temp": combine_mode_agnostic(temp_list),
            "delta": combine_mode_agnostic(delta_list),
            "volt": combine_mode_agnostic(volt_list),
        }
    else:
        return {
            "temp": combine_mode_wise(temp_list),
            "delta": combine_mode_wise(delta_list),
            "volt": combine_mode_wise(volt_list),
        }



def _draw_table(ax, df, title, font_size=9, row_height=1.2):
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, row_height)



def export_battery_condition_fleet_report(vehicle_results,
                                          fleet_mode_summary,
                                          fleet_overall_summary,
                                          output_pdf="battery_condition_fleet_report.pdf"):
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
        fig.suptitle("Fleet-Level Battery Condition Summary (By Mode)",
                     fontsize=16, fontweight="bold")
        _draw_table(axes[0], fleet_mode_summary["temp"],
                    "Battery Max Temperature Distribution (%)")
        _draw_table(axes[1], fleet_mode_summary["delta"],
                    "Temperature Delta (°C) Distribution (%)")
        _draw_table(axes[2], fleet_mode_summary["volt"],
                    "Voltage Delta (mV) Distribution (%)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        gc.collect()
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
        fig.suptitle("Fleet-Level Battery Condition Summary (Mode-Agnostic)",
                     fontsize=16, fontweight="bold")
        _draw_table(axes[0], fleet_overall_summary["temp"],
                    "Battery Max Temperature Distribution (%) — Fleet")
        _draw_table(axes[1], fleet_overall_summary["delta"],
                    "Temperature Delta (°C) Distribution (%) — Fleet")
        _draw_table(axes[2], fleet_overall_summary["volt"],
                    "Voltage Delta (mV) Distribution (%) — Fleet")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        gc.collect()
        for vid, tables in vehicle_results.items():
            temp_df = tables["temp_df"]
            delta_df = tables["delta_df"]
            volt_df = tables["volt_df"]
            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Vehicle ID: {vid} — Battery Condition Summary",
                         fontsize=14, fontweight="bold")
            _draw_table(axes[0], temp_df, "Battery Max Temperature Distribution (%)")
            _draw_table(axes[1], delta_df, "Temperature Delta (°C) Distribution (%)")
            _draw_table(axes[2], volt_df, "Voltage Delta (mV) Distribution (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()
    logging.info(f"✅ Battery Condition Fleet Report Saved → {output_pdf}")



def calc_soc_accuracy_sessions(df: pd.DataFrame,
                               capacity_kwh: float = 423.0,
                               max_gap_sec: int = 300) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["id", "timestamp"]).reset_index(drop=True)
    df["dt_sec"] = df.groupby("id")["timestamp"].diff().dt.total_seconds().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0
    gcs_raw = df["gun_connection_status"]
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.strip().str.lower()
    gun_connected = (gcs_num == 1) | gcs_str.isin({"1", "true", "yes", "y", "connected", "on"})
    df["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")
    CURRENT_LIMIT = 1000
    def clean_current(series):
        s = pd.to_numeric(series, errors="coerce").copy()
        s[s.abs() > CURRENT_LIMIT] = np.nan
        return s.interpolate(limit=30, limit_direction="both").ffill().bfill()
    df["total_battery_current"] = (
        df.groupby("id", group_keys=False)["total_battery_current"].apply(clean_current)
    )
    mode_change = df["mode"] != df["mode"].shift(fill_value=df["mode"].iloc[0])
    new_vehicle = df["id"] != df["id"].shift(fill_value=df["id"].iloc[0])
    gap_break = df["dt_sec"] > max_gap_sec
    df["session_break"] = (mode_change | new_vehicle | gap_break).astype(int)
    df["session_id"] = df["session_break"].cumsum()
    ACTIVE_I = 10
    MAX_DT = 60
    results = []
    for (vid, sid), g in df.groupby(["id", "session_id"], sort=False):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        mode = g["mode"].iloc[0]
        if mode not in ["CHARGING", "DISCHARGING"]:
            continue
        g["dt_sess"] = g["dt_sec"].clip(upper=MAX_DT)
        g_active = g[g["total_battery_current"].abs() > ACTIVE_I]
        if g_active.empty:
            continue
        g["bat_soc"] = pd.to_numeric(g["bat_soc"], errors="coerce")
        g.loc[(g["bat_soc"] <= 0) | (g["bat_soc"] > 100), "bat_soc"] = np.nan
        g["bat_soc"] = g["bat_soc"].ffill().bfill()
        soc_start = g["bat_soc"].iloc[0]
        soc_end = g["bat_soc"].iloc[-1]
        if mode == "DISCHARGING" and soc_end > soc_start:
            soc_end = soc_start
        if mode == "CHARGING" and soc_end < soc_start:
            soc_end = soc_start
        soh_avg = pd.to_numeric(g["soh"], errors="coerce").mean()
        if mode == "CHARGING":
            delta_soc = soc_end - soc_start
        else:
            delta_soc = soc_start - soc_end
        energy_soc_kwh = abs(delta_soc * soh_avg * capacity_kwh / 10000.0)
        e_meas_kwh = (
            g_active["bat_voltage"] *
            g_active["total_battery_current"] *
            g_active["dt_sess"]
        ).sum() / 3.6e6
        e_meas_kwh = abs(e_meas_kwh)
        accuracy = np.nan
        if energy_soc_kwh > 1e-6:
            accuracy = (1 - abs(e_meas_kwh - energy_soc_kwh) / energy_soc_kwh) * 100
        dur_min = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 60
        results.append({
            "vehicle_id": vid,
            "session_id": sid,
            "mode": mode,
            "start_time": g["timestamp"].iloc[0],
            "end_time": g["timestamp"].iloc[-1],
            "duration_min": round(dur_min, 2),
            "soc_start": round(soc_start, 2),
            "soc_end": round(soc_end, 2),
            "soh_avg": round(soh_avg, 2),
            "energy_soc_kwh": round(energy_soc_kwh, 3),
            "energy_measured_kwh": round(e_meas_kwh, 3),
            "accuracy_percent": round(accuracy, 2),
        })
    return pd.DataFrame(results).sort_values(
        ["vehicle_id", "start_time"]
    ).reset_index(drop=True)



def export_soc_accuracy_to_pdf(soc_accuracy_df: pd.DataFrame,
                               output_path: str = "vehiclewise_soc_accuracy.pdf",
                               max_rows_per_page: int = 28):
    df = soc_accuracy_df.copy()
    df = df.sort_values(["vehicle_id", "start_time"]).reset_index(drop=True)
    colnames = [
        "vehicle_id", "session_id", "mode",
        "start_time", "end_time",
        "duration_min", "soc_start", "soc_end",
        "soh_avg", "energy_soc_kwh",
        "energy_measured_kwh", "accuracy_percent",
    ]
    df = df[colnames].round(2)
    width_map = {
        "vehicle_id": 0.7, "session_id": 0.7, "mode": 1.0,
        "start_time": 2.5, "end_time": 2.5,
        "duration_min": 1.0, "soc_start": 1.0, "soc_end": 1.0,
        "soh_avg": 1.0, "energy_soc_kwh": 1.2,
        "energy_measured_kwh": 1.4, "accuracy_percent": 1.2,
    }
    widths = [width_map[c] for c in colnames]
    vehicle_ids = df["vehicle_id"].unique()
    with PdfPages(output_path) as pdf:
        for vid in vehicle_ids:
            vdf = df[df["vehicle_id"] == vid].copy()
            total_rows = len(vdf)
            num_pages = int(np.ceil(total_rows / max_rows_per_page))
            for page_i in range(num_pages):
                start = page_i * max_rows_per_page
                end = start + max_rows_per_page
                chunk = vdf.iloc[start:end]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                title = f"Vehicle ID: {vid} — SoC Accuracy (Page {page_i+1}/{num_pages})"
                fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
                table = ax.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc="center",
                    cellLoc="center",
                )
                for i, width in enumerate(widths):
                    for key, cell in table.get_celld().items():
                        if key[1] == i:
                            cell.set_width(width / 20)
                table.scale(1.0, 1.4)
                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                ax.axis("off")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
                gc.collect()
    logging.info(f"✅ SoC accuracy PDF created: {output_path}")



def compute_time_weighted_energy(soc_accuracy_df: pd.DataFrame) -> pd.DataFrame:
    df = soc_accuracy_df.copy()
    df["duration_hr"] = df["duration_min"] / 60.0
    df = df[df["duration_hr"] > 0].copy()
    grouped = df.groupby(["vehicle_id", "mode"])
    results = []
    for (vid, mode), g in grouped:
        total_time = g["duration_hr"].sum()
        if total_time <= 0:
            continue
        w_avg_soc = (g["energy_soc_kwh"] * g["duration_hr"]).sum() / total_time
        w_avg_meas = (g["energy_measured_kwh"] * g["duration_hr"]).sum() / total_time
        results.append({
            "vehicle_id": vid,
            "mode": mode,
            "total_time_hr": round(total_time, 3),
            "weighted_avg_energy_soc_kwh": round(w_avg_soc, 3),
            "weighted_avg_energy_measured_kwh": round(w_avg_meas, 3),
            "difference_kwh": round(w_avg_meas - w_avg_soc, 3),
            "difference_percent": round(
                (1 - abs(w_avg_meas - w_avg_soc) / w_avg_soc) * 100, 2
            ) if w_avg_soc > 0 else np.nan,
        })
    return pd.DataFrame(results)

# =====================================================================
# STAGE 1 — Load raw monthly CSV → preprocess → df_with_state.feather
# =====================================================================


date_str = "2025-10-01"
target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
ist_start = datetime.combine(target_date, datetime.min.time())
ist_end = ist_start + timedelta(days=31)
utc_start = ist_start - timedelta(hours=5, minutes=30)
utc_end = ist_end - timedelta(hours=5, minutes=30)
logging.info(f"🔍 Query window (UTC): {utc_start} → {utc_end}")


vehicle_ids = ['3','16','18','19','32','42','6','7','9','11','12','13','14','15','20',
               '25','27','28','29','30','31','33','35','41','46']


csv_path = "oct25_can_parsed_data.csv"
logging.info(f"📁 Loading monthly CAN data from CSV: {csv_path}")


# figure out available columns then load only needed ones
head_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
usecols = [c for c in CORE_COLS if c in head_cols]
df_cpo100 = pd.read_csv(csv_path, usecols=usecols)
df_cpo100 = rename_battery_temp_columns(df_cpo100)
logging.info(f"Raw df_cpo100 loaded with {len(df_cpo100):,} rows and {df_cpo100.shape[1]} columns")


if "id" in df_cpo100.columns:
    df_cpo100["id"] = df_cpo100["id"].astype(str)
    df_cpo100 = df_cpo100[df_cpo100["id"].isin(vehicle_ids)]
    logging.info(f"Filtered by vehicle_ids → {len(df_cpo100):,} rows")


df_cpo100["timestamp"] = pd.to_datetime(df_cpo100["timestamp"], errors="coerce")
df_cpo100 = df_cpo100.dropna(subset=["timestamp"])
df_cpo100 = df_cpo100[(df_cpo100["timestamp"] >= utc_start) & (df_cpo100["timestamp"] <= utc_end)]
logging.info(f"After date filter → {len(df_cpo100):,} rows")


logging.info("🧹 Imputing missing values ...")
df_cpo100 = impute_missing_values(df_cpo100)


logging.info("🧠 Preparing df_with_state (mode + temp/volt deltas)...")
df_with_state = prepare_df_with_state(df_cpo100)
logging.info(f"df_with_state has {len(df_with_state):,} rows and {df_with_state.shape[1]} columns")


feather_path = "df_with_state_30days.feather"
df_with_state.to_feather(feather_path)
logging.info(f"💾 Saved df_with_state → {feather_path}")


del df_cpo100
del df_with_state
gc.collect()
free_mem()


logging.info("✅ Stage 1 complete. Run Stage 2 next.")


# =====================================================================
# STAGE 2 — Battery condition analysis (vehicle-wise + fleet)
# =====================================================================


feather_path = "df_with_state_30days.feather"
logging.info(f"📁 Loading df_with_state from {feather_path} for battery condition analysis...")


import pyarrow.feather as ft


meta = ft.read_table(feather_path).schema
cols_available = [f.name for f in meta]


cols_needed = ["id", "mode", "batt_maxtemp", "batt_mintemp", "batt_maxvolt", "batt_minvolt"]


cols_to_load = [c for c in cols_needed if c in cols_available]
df_cond = pd.read_feather(feather_path, columns=cols_to_load)


logging.info(f"df_cond loaded with {len(df_cond):,} rows and {df_cond.shape[1]} columns")


logging.info("📊 Running vehicle-wise battery condition analysis...")
vehicle_results = analyze_battery_conditions_vehiclewise(
    df_cond,
    output_pdf="battery_conditions_by_vehicle_30days.pdf"
)


logging.info("📊 Computing fleet summaries (mode-wise & mode-agnostic)...")
fleet_mode = compute_fleet_summary(vehicle_results, mode_agnostic=False)
fleet_overall = compute_fleet_summary(vehicle_results, mode_agnostic=True)


logging.info("📄 Exporting consolidated fleet battery condition report...")
export_battery_condition_fleet_report(
    vehicle_results=vehicle_results,
    fleet_mode_summary=fleet_mode,
    fleet_overall_summary=fleet_overall,
    output_pdf="battery_condition_fleet_report_30days.pdf",
)


del df_cond
del vehicle_results
del fleet_mode
del fleet_overall
gc.collect()
free_mem()


logging.info("✅ Stage 2 complete. Run Stage 3 next.")

# =====================================================================
# STAGE 3 — SoC sessions + SoC accuracy PDF
# =====================================================================


feather_path = "df_with_state_30days.feather"
logging.info(f"📁 Loading df_with_state from {feather_path} for SoC analysis...")


import pyarrow.feather as ft


meta = ft.read_table(feather_path).schema
cols_available = [f.name for f in meta]


cols_needed = ["id", "timestamp", "gun_connection_status", "bat_soc", "soh","bat_voltage", "total_battery_current"]
cols_to_load = [c for c in cols_needed if c in cols_available]


df_soc_base = pd.read_feather(feather_path, columns=cols_available)
logging.info(f"df_soc_base loaded with {len(df_soc_base):,} rows and {df_soc_base.shape[1]} columns")


logging.info("⚡ Computing SoC accuracy sessions...")
soc_accuracy_df = calc_soc_accuracy_sessions(df_soc_base)
logging.info(f"SoC sessions computed: {len(soc_accuracy_df):,} rows")


soc_path = "soc_accuracy_sessions_30days.parquet"
soc_accuracy_df.to_parquet(soc_path, index=False)
logging.info(f"💾 Saved SoC sessions → {soc_path}")


export_soc_accuracy_to_pdf(soc_accuracy_df, "vehicle_wise_soc_accuracy_30days_clean.pdf")


del df_soc_base
del soc_accuracy_df
gc.collect()
free_mem()


logging.info("✅ Stage 3 complete. Run Stage 4 next.")

# =====================================================================
# STAGE 4 — Time-weighted energy metrics from SoC sessions
# =====================================================================


soc_path = "soc_accuracy_sessions_30days.parquet"
logging.info(f"📁 Loading SoC sessions from {soc_path} for energy summary...")


soc_accuracy_df = pd.read_parquet(soc_path)
logging.info(f"SoC sessions loaded: {len(soc_accuracy_df):,} rows")


logging.info("🔢 Computing time-weighted energy metrics...")
weighted_energy_summary = compute_time_weighted_energy(soc_accuracy_df)
display(weighted_energy_summary)


out_csv = "weighted_energy_summary_30days.csv"
weighted_energy_summary.to_csv(out_csv, index=False)
logging.info(f"✅ Time-weighted energy summary saved → {out_csv}")


del soc_accuracy_df
del weighted_energy_summary
gc.collect()
free_mem()


logging.info("🎉 All stages complete.")

# =====================================================================
# STAGE 5 — SoC Dynamics + Jump/Drop Detection + C-Rate (Tabular Only)
# Minimalistic Excel-style tables (Option C) — FINAL UPDATED VERSION
# =====================================================================


import pyarrow.feather as ft
from matplotlib.backends.backend_pdf import PdfPages


logging.info("📁 Loading df_with_state for tabulated SoC dynamics...")


feather_path = "df_with_state_30days.feather"
schema = ft.read_table(feather_path).schema
cols_available = [f.name for f in schema]


cols_needed = [
    "id", "timestamp", "bat_soc", "total_battery_current",
    "gun_connection_status"
]
cols_to_load = [c for c in cols_needed if c in cols_available]


df_soc_dyn = pd.read_feather(feather_path, columns=cols_to_load)
df_soc_dyn["timestamp"] = pd.to_datetime(df_soc_dyn["timestamp"], errors="coerce")
df_soc_dyn = df_soc_dyn.dropna(subset=["timestamp"]).sort_values(["id","timestamp"])
logging.info(f"df_soc_dyn loaded: {len(df_soc_dyn):,} rows")



# ---------------------------------------------------------------------
# MODE (Charge vs Discharge)
# ---------------------------------------------------------------------
gcs_raw = df_soc_dyn["gun_connection_status"]
gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
gcs_str = gcs_raw.astype(str).str.strip().str.lower()
gun_connected = (gcs_num == 1) | gcs_str.isin({"1","true","yes","y","connected","on"})
df_soc_dyn["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")



# ---------------------------------------------------------------------
# TIME DIFFERENCE
# ---------------------------------------------------------------------
df_soc_dyn["dt_sec"] = (
    df_soc_dyn.groupby("id")["timestamp"]
    .diff()
    .dt.total_seconds()
    .clip(lower=0, upper=300)
    .fillna(0)
)
df_soc_dyn["dt_min"] = df_soc_dyn["dt_sec"] / 60.0



# ---------------------------------------------------------------------
# SOC BUCKETS — REQUIRED FORMAT
# ---------------------------------------------------------------------
soc_bins = [0,10,20,30,40,50,60,70,80,85,90,95,100]
soc_labels = ["0-10","10-20","20-30","30-40","40-50",
              "50-60","60-70","70-80","80-85",
              "85-90","90-95","95-100"]


df_soc_dyn["soc_bucket"] = pd.cut(
    df_soc_dyn["bat_soc"], bins=soc_bins,
    labels=soc_labels, include_lowest=True
)



# ---------------------------------------------------------------------
#  FLEET-LEVEL SoC OCCUPANCY (% TIME)
# ---------------------------------------------------------------------
tmp = (
    df_soc_dyn.groupby(["mode","soc_bucket"], observed=False)["dt_sec"]
    .sum()
    .reset_index()
)


tmp["percent"] = (
    tmp["dt_sec"] /
    tmp.groupby("mode")["dt_sec"].transform("sum")
    * 100
)


fleet_soc_table = tmp.pivot_table(observed=False,
    index="soc_bucket",
    columns="mode",
    values="percent",
    fill_value=0
).round(2)



# ---------------------------------------------------------------------
#  VEHICLE-WISE SoC OCCUPANCY (% TIME)
# ---------------------------------------------------------------------
tmp2 = (
    df_soc_dyn.groupby(["id","mode","soc_bucket"], observed=False)["dt_sec"]
    .sum()
    .reset_index()
)


tmp2["percent"] = (
    tmp2["dt_sec"] /
    tmp2.groupby(["id","mode"])["dt_sec"].transform("sum")
    * 100
)


veh_soc_tables = {}
for vid, g in tmp2.groupby("id"):
    tab = g.pivot_table(observed=False,
        index="soc_bucket",
        columns="mode",
        values="percent",
        fill_value=0
    ).round(2)
    veh_soc_tables[vid] = tab



# ---------------------------------------------------------------------
# SOC JUMPS / DROPS (Counts + Percent Split per Mode)
# ---------------------------------------------------------------------
df_soc_dyn["soc_diff"] = df_soc_dyn.groupby("id")["bat_soc"].diff()
df_soc_dyn["soc_diff_per_min"] = (
    df_soc_dyn["soc_diff"] /
    df_soc_dyn["dt_min"].replace(0,np.nan)
)


jump_bins = [-999, -10, -5, -2, -1, 1, 2, 5, 10, 999]
jump_labels = [
    "<-10", "-10 to -5", "-5 to -2", "-2 to -1",
    "-1 to 1", "1 to 2", "2 to 5", "5 to 10", ">10"
]


df_soc_dyn["jump_bucket"] = pd.cut(
    df_soc_dyn["soc_diff_per_min"],
    bins=jump_bins, labels=jump_labels
)



# --- Fleet Jump Raw Counts ---
fleet_jumps_raw = (
    df_soc_dyn.groupby(["mode","jump_bucket"], observed=False)["soc_diff_per_min"]
    .count()
    .reset_index()
    .rename(columns={"soc_diff_per_min":"count"})
)


# Pivot to wide format
fleet_jump_counts = fleet_jumps_raw.pivot_table(observed=False,
    index="jump_bucket",
    columns="mode",
    values="count",
    fill_value=0
)


# Percent distribution per mode
fleet_jump_pct = (
    fleet_jump_counts.div(fleet_jump_counts.sum(axis=0), axis=1) * 100
).round(2)


# Combined fleet-level table
fleet_jump_table = pd.DataFrame({
    "Charge Count": fleet_jump_counts.get("CHARGING", pd.Series(dtype=float)),
    "Charge %": fleet_jump_pct.get("CHARGING", pd.Series(dtype=float)),
    "Discharge Count": fleet_jump_counts.get("DISCHARGING", pd.Series(dtype=float)),
    "Discharge %": fleet_jump_pct.get("DISCHARGING", pd.Series(dtype=float)),
}).fillna(0)



# --- Vehicle-wise jump tables ---
veh_jump_tables = {}
tmp3 = (
    df_soc_dyn.groupby(["id","mode","jump_bucket"], observed=False)["soc_diff_per_min"]
    .count()
    .reset_index()
    .rename(columns={"soc_diff_per_min":"count"})
)


for vid, g in tmp3.groupby("id"):
    wide = g.pivot_table(observed=False,
        index="jump_bucket",
        columns="mode",
        values="count",
        fill_value=0
    )
    # Compute percentages
    pct = (wide.div(wide.sum(axis=0), axis=1) * 100).round(2)
    veh_jump_tables[vid] = pd.DataFrame({
        "Charge Count": wide.get("CHARGING", pd.Series(dtype=float)),
        "Charge %": pct.get("CHARGING", pd.Series(dtype=float)),
        "Discharge Count": wide.get("DISCHARGING", pd.Series(dtype=float)),
        "Discharge %": pct.get("DISCHARGING", pd.Series(dtype=float)),
    }).fillna(0)



# ---------------------------------------------------------------------
# C-RATE CALCULATION (with anomaly clipping)
# ---------------------------------------------------------------------
BAT_CAPACITY_AH = 423.8
df_soc_dyn["c_rate"] = df_soc_dyn["total_battery_current"] / BAT_CAPACITY_AH


# Clip impossible values: realistic range −3C to +3C
df_soc_dyn["c_rate"] = df_soc_dyn["c_rate"].clip(-3, 3)


# --- Fleet summary ---
fleet_c_table = (
    df_soc_dyn.groupby("mode")["c_rate"]
    .agg(["mean","median","max"])
    .round(3)
)


# --- Vehicle-wise contiguous summary ---
veh_c_contiguous = (
    df_soc_dyn.groupby(["id","mode"])["c_rate"]
    .agg(["mean","median","max"])
    .round(3)
    .reset_index()
)


# Pivot so each vehicle has one row
veh_c_table = veh_c_contiguous.pivot_table(observed=False,
    index="id",
    columns="mode",
    values=["mean","median","max"],
    fill_value=0
)


veh_c_table.columns = [
    f"{metric}_{mode}"
    for metric, mode in veh_c_table.columns
]


veh_c_table = veh_c_table.reset_index()



# ---------------------------------------------------------------------
# TABLE RENDERER (Minimalistic Excel-style)
# ---------------------------------------------------------------------
def draw_table_minimal(ax, df, title):
    ax.axis("off")
    ax.text(
        0.0, 1.05, title,
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="bottom"
    )

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index if df.index.name != "id" else df["id"] if "id" in df else df.index,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)

    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor("white")  # No borders



# ---------------------------------------------------------------------
# GENERATE PDF
# ---------------------------------------------------------------------
pdf_path = "soc_dynamics_and_c_rate_report_30days.pdf"


with PdfPages(pdf_path) as pdf:

    # PAGE 1 — Fleet SoC occupancy
    fig, ax = plt.subplots(figsize=(10,6))
    draw_table_minimal(ax, fleet_soc_table, "Fleet-Level SoC Occupancy (%)")
    pdf.savefig(fig)
    plt.close(fig)

    # PAGE 2 — Vehicle-wise SoC occupancy
    for vid, tab in veh_soc_tables.items():
        fig, ax = plt.subplots(figsize=(10,6))
        draw_table_minimal(ax, tab, f"Vehicle {vid} — SoC Occupancy (%)")
        pdf.savefig(fig)
        plt.close(fig)

    # PAGE 3 — Fleet SoC jump distribution (count + %)
    fig, ax = plt.subplots(figsize=(10,6))
    draw_table_minimal(ax, fleet_jump_table, "Fleet-Level SoC Jumps/Drops (Count + %)")
    pdf.savefig(fig)
    plt.close(fig)

    # PAGE 4 — Vehicle-wise jump tables
    for vid, tab in veh_jump_tables.items():
        fig, ax = plt.subplots(figsize=(10,6))
        draw_table_minimal(ax, tab, f"Vehicle {vid} — SoC Jumps/Drops (Count + %)")
        pdf.savefig(fig)
        plt.close(fig)

    # PAGE 5 — Fleet C-rate summary
    fig, ax = plt.subplots(figsize=(10,6))
    draw_table_minimal(ax, fleet_c_table, "Fleet-Level C-Rate Summary")
    pdf.savefig(fig)
    plt.close(fig)

    # PAGE 6 — Contiguous vehicle-wise C-rate table (ALL VEHICLES)
    fig, ax = plt.subplots(figsize=(11,8))
    draw_table_minimal(ax, veh_c_table, "Vehicle-Wise C-Rate Summary (All Vehicles)")
    pdf.savefig(fig)
    plt.close(fig)


logging.info(f"📄 SOC Dynamics & C-Rate Report Saved → {pdf_path}")


del df_soc_dyn
gc.collect()
free_mem()


logging.info("✅ Stage 5 complete.")

# =====================================================================
# STAGE 6 — Hotspot Analysis (Temperature @ Pack-Level + Voltage @ Cell-Level)
# Fleet-Level + Per-Vehicle (Option B3)
# =====================================================================


import pyarrow.feather as ft
import pandas as pd
import logging


logging.info("📁 Loading df_with_state for Hotspot Analysis...")


feather_path = "df_with_state_30days.feather"


# Step 1 — Check schema to avoid full load
schema = ft.read_table(feather_path).schema
cols_available = [f.name for f in schema]


cols_needed = [
    "id", "mode",
    "batt_maxtemp_pack", "batt_mintemp_pack",
    "batt_maxvolt", "batt_maxvolt_cell",
    "batt_minvolt", "batt_minvolt_cell",
]


cols_to_load = [c for c in cols_needed if c in cols_available]


df_hot = pd.read_feather(feather_path, columns=cols_to_load)
logging.info(f"df_hot loaded: {len(df_hot):,} rows")


# Ensure MODE is available
if "mode" not in df_hot.columns:
    raise ValueError("Column 'mode' missing. Run Stage 5 before Stage 6.")


# =====================================================================
# SECTION 1 — TEMPERATURE HOTSPOTS (PACK-LEVEL, 108 SENSORS)
# =====================================================================


# ---------- Fleet-Level Max Temp ----------
fleet_temp_max_raw = (
    df_hot.groupby(["mode", "batt_maxtemp_pack"], observed=False)
    .size()
    .reset_index(name="count")
)


fleet_temp_max = {}


for mode in ["CHARGING", "DISCHARGING"]:

    sub = fleet_temp_max_raw[fleet_temp_max_raw["mode"] == mode].copy()
    total = sub["count"].sum() or 1  # avoid zero division

    sub["percent"] = (sub["count"] / total * 100).round(2)

    # Clean + rename
    pct = sub.rename(columns={
        "batt_maxtemp_pack": "pack_tc",
        "count": f"{mode.lower()}_max_count",
        "percent": f"{mode.lower()}_max_pct"
    })

    pct = pct.drop(columns=["mode"], errors="ignore")
    pct = pct.set_index("pack_tc")

    fleet_temp_max[mode] = pct



# ---------- Fleet-Level Min Temp ----------
fleet_temp_min_raw = (
    df_hot.groupby(["mode", "batt_mintemp_pack"], observed=False)
    .size()
    .reset_index(name="count")
)


fleet_temp_min = {}


for mode in ["CHARGING", "DISCHARGING"]:

    sub = fleet_temp_min_raw[fleet_temp_min_raw["mode"] == mode].copy()
    total = sub["count"].sum() or 1

    sub["percent"] = (sub["count"] / total * 100).round(2)

    pct = sub.rename(columns={
        "batt_mintemp_pack": "pack_tc",
        "count": f"{mode.lower()}_min_count",
        "percent": f"{mode.lower()}_min_pct"
    })

    pct = pct.drop(columns=["mode"], errors="ignore")
    pct = pct.set_index("pack_tc")

    fleet_temp_min[mode] = pct



# ---------- COMBINE Fleet Temp Max + Min ----------
fleet_temp_hotspots = (
    fleet_temp_max["CHARGING"]
    .join(fleet_temp_max["DISCHARGING"], how="outer")
    .join(fleet_temp_min["CHARGING"], how="outer")
    .join(fleet_temp_min["DISCHARGING"], how="outer")
    .fillna(0)
)


logging.info("🔥 Fleet Temperature Hotspot Table (Pack-Level) ready.")



# ---------- Per-Vehicle Temperature Hotspots ----------
veh_temp_hotspots = {}


for vid, g in df_hot.groupby("id"):

    # ---- MAX TEMP PER VEHICLE ----
    vmax_raw = (
        g.groupby(["mode", "batt_maxtemp_pack"], observed=False)
        .size()
        .reset_index(name="count")
    )

    vmax_tables = {}
    for mode in ["CHARGING", "DISCHARGING"]:
        sub = vmax_raw[vmax_raw["mode"] == mode].copy()
        total = sub["count"].sum() or 1
        sub["percent"] = (sub["count"] / total * 100).round(2)

        pct = sub.rename(columns={
            "batt_maxtemp_pack": "pack_tc",
            "count": f"{mode.lower()}_max_count",
            "percent": f"{mode.lower()}_max_pct",
        })

        pct = pct.drop(columns=["mode"], errors="ignore")
        pct = pct.set_index("pack_tc")

        vmax_tables[mode] = pct

    # ---- MIN TEMP PER VEHICLE ----
    vmin_raw = (
        g.groupby(["mode", "batt_mintemp_pack"], observed=False)
        .size()
        .reset_index(name="count")
    )

    vmin_tables = {}
    for mode in ["CHARGING", "DISCHARGING"]:
        sub = vmin_raw[vmin_raw["mode"] == mode].copy()
        total = sub["count"].sum() or 1
        sub["percent"] = (sub["count"] / total * 100).round(2)

        pct = sub.rename(columns={
            "batt_mintemp_pack": "pack_tc",
            "count": f"{mode.lower()}_min_count",
            "percent": f"{mode.lower()}_min_pct",
        })

        pct = pct.drop(columns=["mode"], errors="ignore")
        pct = pct.set_index("pack_tc")

        vmin_tables[mode] = pct

    veh_temp_hotspots[vid] = (
        vmax_tables["CHARGING"]
        .join(vmax_tables["DISCHARGING"], how="outer")
        .join(vmin_tables["CHARGING"], how="outer")
        .join(vmin_tables["DISCHARGING"], how="outer")
        .fillna(0)
    )


logging.info("🔥 Per-Vehicle Temperature Hotspot Tables ready.")



# =====================================================================
# SECTION 2 — VOLTAGE HOTSPOTS (CELL-LEVEL, 576 CELLS)
# =====================================================================


# ---------- Fleet-Level Max Voltage ----------
fleet_vmax_raw = (
    df_hot.groupby(["mode", "batt_maxvolt_cell"], observed=False)
    .size()
    .reset_index(name="count")
)


fleet_vmax = {}


for mode in ["CHARGING", "DISCHARGING"]:

    sub = fleet_vmax_raw[fleet_vmax_raw["mode"] == mode].copy()
    total = sub["count"].sum() or 1
    sub["percent"] = (sub["count"] / total * 100).round(2)

    pct = sub.rename(columns={
        "batt_maxvolt_cell": "cell_id",
        "count": f"{mode.lower()}_max_count",
        "percent": f"{mode.lower()}_max_pct"
    })

    pct = pct.drop(columns=["mode"], errors="ignore")
    pct = pct.set_index("cell_id")

    fleet_vmax[mode] = pct



# ---------- Fleet-Level Min Voltage ----------
fleet_vmin_raw = (
    df_hot.groupby(["mode", "batt_minvolt_cell"], observed=False)
    .size()
    .reset_index(name="count")
)


fleet_vmin = {}


for mode in ["CHARGING", "DISCHARGING"]:

    sub = fleet_vmin_raw[fleet_vmin_raw["mode"] == mode].copy()
    total = sub["count"].sum() or 1
    sub["percent"] = (sub["count"] / total * 100).round(2)

    pct = sub.rename(columns={
        "batt_minvolt_cell": "cell_id",
        "count": f"{mode.lower()}_min_count",
        "percent": f"{mode.lower()}_min_pct"
    })

    pct = pct.drop(columns=["mode"], errors="ignore")
    pct = pct.set_index("cell_id")

    fleet_vmin[mode] = pct



# ---------- Combine Fleet Voltage ----------
fleet_voltage_hotspots = (
    fleet_vmax["CHARGING"]
    .join(fleet_vmax["DISCHARGING"], how="outer")
    .join(fleet_vmin["CHARGING"], how="outer")
    .join(fleet_vmin["DISCHARGING"], how="outer")
    .fillna(0)
)


logging.info("⚡ Fleet Voltage Hotspot Table (Cell-Level) ready.")



# ---------- Per-Vehicle Voltage Hotspots ----------
veh_voltage_hotspots = {}


for vid, g in df_hot.groupby("id"):

    vmax_raw = (
        g.groupby(["mode", "batt_maxvolt_cell"], observed=False)
        .size()
        .reset_index(name="count")
    )

    vmin_raw = (
        g.groupby(["mode", "batt_minvolt_cell"], observed=False)
        .size()
        .reset_index(name="count")
    )

    vmax_tables = {}
    vmin_tables = {}

    for mode in ["CHARGING", "DISCHARGING"]:

        # MAX VOLT
        sub = vmax_raw[vmax_raw["mode"] == mode].copy()
        total = sub["count"].sum() or 1
        sub["percent"] = (sub["count"] / total * 100).round(2)

        pct = sub.rename(columns={
            "batt_maxvolt_cell": "cell_id",
            "count": f"{mode.lower()}_max_count",
            "percent": f"{mode.lower()}_max_pct"
        })

        pct = pct.drop(columns=["mode"], errors="ignore")
        pct = pct.set_index("cell_id")
        vmax_tables[mode] = pct

        # MIN VOLT
        sub2 = vmin_raw[vmin_raw["mode"] == mode].copy()
        total2 = sub2["count"].sum() or 1
        sub2["percent"] = (sub2["count"] / total2 * 100).round(2)

        pct2 = sub2.rename(columns={
            "batt_minvolt_cell": "cell_id",
            "count": f"{mode.lower()}_min_count",
            "percent": f"{mode.lower()}_min_pct"
        })

        pct2 = pct2.drop(columns=["mode"], errors="ignore")
        pct2 = pct2.set_index("cell_id")
        vmin_tables[mode] = pct2

    veh_voltage_hotspots[vid] = (
        vmax_tables["CHARGING"]
        .join(vmax_tables["DISCHARGING"], how="outer")
        .join(vmin_tables["CHARGING"], how="outer")
        .join(vmin_tables["DISCHARGING"], how="outer")
        .fillna(0)
    )


logging.info("⚡ Per-Vehicle Voltage Hotspot Tables ready.")



# =====================================================================
# OUTPUT
# =====================================================================
fleet_temp_hotspots       # Pack-level fleet summary
veh_temp_hotspots         # Pack-level per-vehicle
fleet_voltage_hotspots    # Cell-level fleet summary
veh_voltage_hotspots      # Cell-level per-vehicle


logging.info("✅ Stage 6 complete.")



