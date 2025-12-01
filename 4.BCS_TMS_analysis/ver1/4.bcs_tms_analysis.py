#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %% [markdown]
# # BCS / TMS Analysis – Memory-Efficient Pipeline Notebook
# 
# This notebook is a modular, pipeline-style rewrite of the earlier
# `3.bcs_tms_analysis.py`. It is organized so that:
# 
# 1. Heavy dataframes are created once and reused.
# 2. All business-logic functions live in this first cell.
# 3. Subsequent cells run the pipeline step-by-step.
# 
# You can safely re-run later cells without recomputing earlier expensive steps,
# as long as you do not clear the kernel.

# %%
import os
import sys
import gc
import numpy as np
import pandas as pd
import platform
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime, date, timedelta

# Optional: adjust pandas display for debugging; you can comment these out
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 200)

# ---- Logging ----
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
print(f"Using Python version: {platform.python_version()}")

# ---- Repo-specific imports ----
# Adapt the repo_path for your environment if needed.
repo_path = '/Users/apple/Documents/naarni/repo/dview-naarni-data-platform'
if repo_path not in sys.path:
    sys.path.append(os.path.join(repo_path, 'tasks'))

from common.db_operations import connect_to_trino, fetch_data_for_day


# In[2]:


# =====================================================================
# Helper: Fetch CAN-parsed data from Trino (optional – you can also
#         load from CSV in a later cell)
# =====================================================================
def fetch_can_parsed_data(
    start_date: str,
    end_date: str,
    vehicle_ids: list = None,
    table_name: str = "facts_prod.can_parsed_output_all",
    chunk_days: int = 1,
) -> pd.DataFrame:
    """
    Iteratively fetch CAN-parsed data from Trino for the specified date range.
    This is optional; in practice, we often load df_cpo100 from a CSV to avoid
    repeatedly hitting Trino for the same month.
    """

    logging.info(f"🚀 Fetching CAN-parsed data from {table_name} ({start_date} → {end_date})")
    conn = connect_to_trino(host="trino.naarni.internal", port=80,
                            user="admin", catalog="adhoc", schema="default")

    id_filter = ""
    if vehicle_ids:
        vehicle_ids_str = ", ".join([f"'{vid}'" for vid in vehicle_ids])
        id_filter = f"AND id IN ({vehicle_ids_str})"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    df_list = []
    cur_date = start_dt

    while cur_date <= end_dt:
        logging.info(f"📆 Fetching data for: {cur_date.date()}")

        query = f"""
            SELECT 
                id,timestamp,dt, 
                vehiclereadycondition, gun_connection_status, ignitionstatus,
                vehicle_speed_vcu,gear_position,bat_soc,soh,total_battery_current,
                pack1_cellmax_temperature, pack1_cell_min_temperature, pack1_maxtemperature_cell_number,  pack1_celltemperature_cellnumber,
                bat_voltage,cellmax_voltagecellnumber,cell_max_voltage,cellminvoltagecellnumber,cell_min_voltage      
            FROM {table_name}
            WHERE dt = DATE('{cur_date:%Y-%m-%d}')
            {id_filter}
        """
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            chunk_df = pd.DataFrame(rows, columns=cols)
            df_list.append(chunk_df)
            logging.info(f"✅ {len(chunk_df)} rows fetched for {cur_date.date()}")
        except Exception as e:
            logging.error(f"❌ Error fetching chunk {cur_date.date()}: {e}")
        finally:
            try:
                cur.close()
            except Exception:
                pass

        cur_date += timedelta(days=chunk_days)

    try:
        conn.close()
    except Exception:
        pass

    df_final = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    logging.info(f"🏁 Done — {len(df_final)} total rows fetched from {table_name}")
    return df_final


# In[3]:


# =====================================================================
# Renaming + Imputation + State Preparation
# =====================================================================
def rename_battery_temp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename temperature columns from BCS/TMS naming to standardized names.
    - pack1_cellmax_temperature → batt_maxtemp
    - pack1_cell_min_temperature → batt_mintemp
    """
    rename_map = {
        "pack1_cellmax_temperature": "batt_maxtemp",
        "pack1_cell_min_temperature": "batt_mintemp",
    }
    existing_cols = [c for c in rename_map if c in df.columns]
    if not existing_cols:
        logging.warning("No matching temperature columns found to rename.")
        return df
    return df.rename(columns={c: rename_map[c] for c in existing_cols})


# In[4]:


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle imputation consistent with the original script."""
    df = df.sort_values(["id", "timestamp"]).copy()

    for vid, grp in df.groupby("id"):
        mask = df["id"] == vid

        # 1. Temperatures – slow signals, fill small gaps (≈60 s if 1 Hz)
        df.loc[mask, "batt_maxtemp"] = grp["batt_maxtemp"].ffill(limit=60)
        df.loc[mask, "batt_mintemp"] = grp["batt_mintemp"].ffill(limit=60)

        # 2. Cell voltages – very stable, fill short gaps (≈30 s)
        if "cell_max_voltage" in grp.columns:
            df.loc[mask, "cell_max_voltage"] = grp["cell_max_voltage"].ffill(limit=30)
        if "cell_min_voltage" in grp.columns:
            df.loc[mask, "cell_min_voltage"] = grp["cell_min_voltage"].ffill(limit=30)

        # 3. Bus voltage – slightly more dynamic, fill ≤20 s
        df.loc[mask, "bat_voltage"] = grp["bat_voltage"].ffill(limit=20)

        # 4. SoC / SoH – slow metrics, fill up to 5 min
        df.loc[mask, "bat_soc"] = grp["bat_soc"].ffill(limit=300)
        df.loc[mask, "soh"] = grp["soh"].ffill(limit=300)

        # 5. Current – interpolate small gaps (≤10 s)
        df.loc[mask, "total_battery_current"] = grp["total_battery_current"].interpolate(
            limit=10, limit_direction="both"
        )

        # 6. Binary state flags – persist until changed
        df.loc[mask, "vehiclereadycondition"] = grp["vehiclereadycondition"].ffill()
        df.loc[mask, "gun_connection_status"] = grp["gun_connection_status"].ffill()

    return df


# In[5]:


def prepare_df_with_state(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal pre-processor for battery condition + SOC analysis."""
    required = [
        "id", "timestamp",
        "vehiclereadycondition", "gun_connection_status",
        "batt_maxtemp", "batt_mintemp",
        "cell_max_voltage", "cell_min_voltage",
        "bat_voltage", "total_battery_current",
        "bat_soc", "soh",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values(["id", "timestamp"]).reset_index(drop=True)

    # Normalize gun connection status
    gcs_raw = out["gun_connection_status"]
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.strip().str.lower()

    gun_connected = (
        (gcs_num == 1) |
        gcs_str.isin({"1", "true", "yes", "y", "connected", "on"})
    )

    out["mode"] = np.where(gun_connected, "CHARGING", "DISCHARGING")
    out["mode_change"] = out["mode"] != out["mode"].shift()
    out["mode_group"] = out["mode_change"].cumsum()

    out["batt_temp_delta"] = pd.to_numeric(out["batt_maxtemp"], errors="coerce") -                                   pd.to_numeric(out["batt_mintemp"], errors="coerce")
    out["volt_delta_mv"] = (pd.to_numeric(out["cell_max_voltage"], errors="coerce") -
                              pd.to_numeric(out["cell_min_voltage"], errors="coerce")) * 1000.0

    out["dt_sec"] = out.groupby("id")["timestamp"].diff().dt.total_seconds().fillna(0)

    cols_keep = [
        "id", "timestamp", "mode", "vehiclereadycondition", "gun_connection_status",
        "batt_maxtemp", "batt_mintemp", "batt_temp_delta",
        "cell_max_voltage", "cell_min_voltage", "volt_delta_mv",
        "bat_voltage", "total_battery_current",
        "bat_soc", "soh", "dt_sec",
    ]
    out = out[cols_keep]
    out = out.dropna(subset=["batt_maxtemp", "batt_mintemp", "cell_max_voltage", "cell_min_voltage"])
    return out


# In[6]:


# =====================================================================
# Battery condition analysis (vehicle-wise + fleet summaries)
# =====================================================================
def analyze_battery_conditions_vehiclewise(df: pd.DataFrame,
                                           output_pdf: str = "battery_conditions_vehiclewise.pdf"):
    """Vehicle-wise battery condition analysis + per-vehicle PDF."""
    required = ["id", "mode", "batt_maxtemp", "batt_mintemp",
                "cell_max_voltage", "cell_min_voltage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["temp_delta"] = df["batt_maxtemp"] - df["batt_mintemp"]
    df["volt_delta_mv"] = (df["cell_max_voltage"] - df["cell_min_voltage"]) * 1000

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

            temp_df = pd.concat(
                {m: r["Battery Max Temp (%)"] for m, r in mode_results.items()}, axis=1
            ).fillna(0)
            delta_df = pd.concat(
                {m: r["ΔT (°C) Range (%)"] for m, r in mode_results.items()}, axis=1
            ).fillna(0)
            volt_df = pd.concat(
                {m: r["Voltage Δ (mV) (%)"] for m, r in mode_results.items()}, axis=1
            ).fillna(0)

            vehicle_results[vid] = {
                "temp_df": temp_df,
                "delta_df": delta_df,
                "volt_df": volt_df,
            }

            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Vehicle ID: {vid}", fontsize=14, fontweight="bold")

            table_map = {
                "Battery Max Temperature Distribution (%)": temp_df,
                "Temperature Delta (°C)": delta_df,
                "Voltage Delta (mV)": volt_df,
            }

            for ax, (title, df_table) in zip(axes, table_map.items()):
                ax.axis("off")
                ax.set_title(title, fontsize=11, pad=10)
                tbl = ax.table(
                    cellText=df_table.values,
                    rowLabels=df_table.index,
                    colLabels=df_table.columns,
                    cellLoc="center",
                    loc="center",
                )
                tbl.scale(1.1, 1.2)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()

    logging.info(f"✅ Battery Conditions vehiclewise PDF saved → {output_pdf}")
    return vehicle_results


# In[7]:


def compute_fleet_summary(vehicle_results: dict, mode_agnostic: bool = False):
    """Fleet-level summary (mode-wise or mode-agnostic) from vehicle_results."""
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


# In[8]:


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
    """Unified PDF: fleet mode-wise, fleet overall, then vehicle-wise pages."""
    with PdfPages(output_pdf) as pdf:
        # Page 1 — Fleet mode-wise
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

        # Page 2 — Fleet overall (mode-agnostic)
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

        # Vehicle-wise pages
        for vid, tables in vehicle_results.items():
            temp_df = tables["temp_df"]
            delta_df = tables["delta_df"]
            volt_df = tables["volt_df"]

            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.suptitle(f"Vehicle ID: {vid} — Battery Condition Summary",
                         fontsize=14, fontweight="bold")
            _draw_table(axes[0], temp_df,
                        "Battery Max Temperature Distribution (%)")
            _draw_table(axes[1], delta_df,
                        "Temperature Delta (°C) Distribution (%)")
            _draw_table(axes[2], volt_df,
                        "Voltage Delta (mV) Distribution (%)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
            gc.collect()

    logging.info(f"✅ Battery Condition Fleet Report Saved → {output_pdf}")


# In[9]:


# =====================================================================
# SOC session analysis + SOC accuracy PDF
# =====================================================================
def calc_soc_accuracy_sessions(df: pd.DataFrame,
                               capacity_kwh: float = 423.0,
                               max_gap_sec: int = 300) -> pd.DataFrame:
    """Compute SoC vs measured energy per CHARGING / DISCHARGING session."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    df["dt_sec"] = df.groupby("id")["timestamp"].diff().dt.total_seconds().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

    gcs_raw = df["gun_connection_status"]
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.strip().str.lower()
    gun_connected = (
        (gcs_num == 1) |
        gcs_str.isin({"1", "true", "yes", "y", "connected", "on"})
    )
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
        g = g.copy().sort_values("timestamp")
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


# In[10]:


def export_soc_accuracy_to_pdf(soc_accuracy_df: pd.DataFrame,
                               output_path: str = "vehiclewise_soc_accuracy.pdf",
                               max_rows_per_page: int = 28):
    """Export per-vehicle SoC accuracy tables into a landscape PDF."""
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
        "vehicle_id": 0.7,
        "session_id": 0.7,
        "mode": 1.0,
        "start_time": 2.5,
        "end_time": 2.5,
        "duration_min": 1.0,
        "soc_start": 1.0,
        "soc_end": 1.0,
        "soh_avg": 1.0,
        "energy_soc_kwh": 1.2,
        "energy_measured_kwh": 1.4,
        "accuracy_percent": 1.2,
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
                title = (
                    f"Vehicle ID: {vid}  —  SoC Accuracy Summary"
                    f"  (Page {page_i+1}/{num_pages})"
                )
                fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

                table = ax.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc="center",
                    cellLoc="center",
                )

                for i, width in enumerate(widths):
                    table.auto_set_column_width(i)
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

    logging.info(f"✅ Clean landscape PDF created: {output_path}")


# In[11]:


# =====================================================================
# Time-weighted energy metrics from SOC sessions
# =====================================================================
def compute_time_weighted_energy(soc_accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """Time-weighted average energy metrics per vehicle & mode."""
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


# In[12]:


# %%
# ------------------------------------------------------------------
# Pipeline Step 0–2: Configure date range & load df_cpo100
# ------------------------------------------------------------------

# ---- Date window configuration (IST) ----
date_str = "2025-10-01"   # starting date in IST
target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
ist_start = datetime.combine(target_date, datetime.min.time())
ist_end = ist_start + timedelta(days=31)

# Convert IST to UTC for DB queries (if you decide to use fetch_can_parsed_data)
utc_start = ist_start - timedelta(hours=5, minutes=30)
utc_end = ist_end - timedelta(hours=5, minutes=30)

logging.info(f"🔍 Query window (UTC): {utc_start} → {utc_end}")
logging.info(f"🔍 Query window (IST): {ist_start} → {ist_end}")

# ---- Vehicle list (optional filter) ----
vehicle_ids = ['3','16','18','19','32','42','6','7','9','11','12','13','14','15','20','25','27','28','29','30','31','33','35','41','46']

# ---- Load df_cpo100 ----
# Option A: from local CSV (recommended when CSV already exists)
csv_path = "oct25_can_parsed_data.csv"   # adapt as needed
logging.info(f"📁 Loading df_cpo100 from CSV: {csv_path}")
df_cpo100 = pd.read_csv(csv_path)

# Option B: fetch from Trino (commented out by default)
# df_cpo100 = fetch_can_parsed_data(start_date=utc_start.strftime("%Y-%m-%d"),
#                                   end_date=utc_end.strftime("%Y-%m-%d"),
#                                   vehicle_ids=vehicle_ids,
#                                   table_name="facts_prod.can_parsed_output_100")

# ---- Basic preprocessing on raw df_cpo100 ----
df_cpo100 = rename_battery_temp_columns(df_cpo100)
df_cpo100["timestamp"] = pd.to_datetime(df_cpo100["timestamp"], errors="coerce")

# Filter by UTC date window if timestamp is UTC; adjust if CSV is already in IST
df_cpo100 = df_cpo100[
    (df_cpo100["timestamp"] >= utc_start) & (df_cpo100["timestamp"] <= utc_end)
].copy()

logging.info(f"df_cpo100 loaded with {len(df_cpo100):,} rows")
display(df_cpo100.head())


# In[13]:


# %%
# ------------------------------------------------------------------
# Pipeline Step 3–4: Impute, add state, and build df_with_state
# ------------------------------------------------------------------

logging.info("🧹 Imputing missing values...")
df_cpo100 = impute_missing_values(df_cpo100)

logging.info("🧠 Preparing df_with_state (TMS / BCS state & deltas)...")
df_with_state = prepare_df_with_state(df_cpo100)

logging.info(f"df_with_state has {len(df_with_state):,} rows")
display(df_with_state.head())


# In[14]:


# %%
# ------------------------------------------------------------------
# Pipeline Step 5: Battery condition analysis (vehicle-wise + fleet)
# ------------------------------------------------------------------

logging.info("📊 Running vehicle-wise battery condition analysis...")
vehicle_results = analyze_battery_conditions_vehiclewise(
    df_with_state,
    output_pdf="battery_conditions_by_vehicle_30Days.pdf"
)

logging.info("📊 Computing fleet summaries (mode-wise & mode-agnostic)...")
fleet_mode = compute_fleet_summary(vehicle_results, mode_agnostic=False)
fleet_overall = compute_fleet_summary(vehicle_results, mode_agnostic=True)

logging.info("📄 Exporting consolidated fleet battery condition report...")
export_battery_condition_fleet_report(
    vehicle_results=vehicle_results,
    fleet_mode_summary=fleet_mode,
    fleet_overall_summary=fleet_overall,
    output_pdf="battery_condition_fleet_report.pdf",
)

# Optional: free some memory (df_cpo100 is still kept as the raw source if needed)
gc.collect()


# In[15]:


# %%
# ------------------------------------------------------------------
# Pipeline Step 6: SOC session analysis + SOC accuracy PDF
# ------------------------------------------------------------------

logging.info("⚡ Computing SOC accuracy sessions from df_with_state...")
soc_accuracy_df = calc_soc_accuracy_sessions(df_with_state)
logging.info(f"SOC sessions computed: {len(soc_accuracy_df):,} rows")

display(soc_accuracy_df.head())

logging.info("📄 Exporting vehicle-wise SOC accuracy PDF...")
export_soc_accuracy_to_pdf(soc_accuracy_df, "vehicle_wise_soc_accuracy_30days_clean.pdf")

gc.collect()


# In[ ]:


# %%
# ------------------------------------------------------------------
# Pipeline Step 7: Time-weighted energy metrics
# ------------------------------------------------------------------

logging.info("🔢 Computing time-weighted energy metrics from SOC sessions...")
weighted_energy_summary = compute_time_weighted_energy(soc_accuracy_df)
display(weighted_energy_summary)

# At this point you can optionally create plots or export to CSV.
weighted_energy_summary.to_csv("weighted_energy_summary_30days.csv", index=False)
logging.info("✅ Time-weighted energy summary saved to weighted_energy_summary_30days.csv")
gc.collect()

