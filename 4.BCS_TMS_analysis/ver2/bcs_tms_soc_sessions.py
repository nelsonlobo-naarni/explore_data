import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyarrow.feather as ft

import logging
# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from bcs_tms_core import free_mem, infer_mode_from_gun_status


def calc_soc_accuracy_sessions(
    df: pd.DataFrame,
    capacity_kwh: float = 423.0,
    max_gap_sec: int = 300,
) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values(["id", "timestamp"])
        .reset_index(drop=True)
    )
    df["dt_sec"] = df.groupby("id")["timestamp"].diff().dt.total_seconds().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

    df["mode"] = infer_mode_from_gun_status(df["gun_connection_status"])

    CURRENT_LIMIT = 1000

    def clean_current(series):
        s = pd.to_numeric(series, errors="coerce").copy()
        s[s.abs() > CURRENT_LIMIT] = np.nan
        return s.interpolate(limit=30, limit_direction="both").ffill().bfill()

    df["total_battery_current"] = df.groupby("id", group_keys=False)[
        "total_battery_current"
    ].apply(clean_current)

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
            g_active["bat_voltage"]
            * g_active["total_battery_current"]
            * g_active["dt_sess"]
        ).sum() / 3.6e6
        e_meas_kwh = abs(e_meas_kwh)

        accuracy = np.nan
        if energy_soc_kwh > 1e-6:
            accuracy = (1 - abs(e_meas_kwh - energy_soc_kwh) / energy_soc_kwh) * 100

        dur_min = (
            g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]
        ).total_seconds() / 60

        results.append(
            {
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
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(["vehicle_id", "start_time"])
        .reset_index(drop=True)
    )


def export_soc_accuracy_to_pdf(
    soc_accuracy_df: pd.DataFrame,
    output_path: str = "vehiclewise_soc_accuracy.pdf",
    max_rows_per_page: int = 30,
    font_family: str = "Courier New",
):
    """
    Final SoC Accuracy PDF Generator:
    --------------------------------
    PAGE 1: Introduction + methodology
    PAGE 2: Fleet-level weighted summary
    PAGE 3: Vehicle-level weighted summary
    PAGE 4+: Per-session tables (grouped by vehicle)

    All pages are A4 Landscape, 2mm margins, full-width tables.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Landscape A4 setup
    MM = 0.0393701
    A4_WIDTH = 11.69
    A4_HEIGHT = 8.27
    SIDE_MARGIN = 2 * MM
    usable_width = A4_WIDTH - 2 * SIDE_MARGIN

    df = soc_accuracy_df.copy()

    # ---------------- Timestamp formatting ----------------
    for col in ["start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].dt.strftime("%d-%m-%y %H:%M:%S")

    colnames = [
        "vehicle_id", "session_id", "mode",
        "start_time", "end_time",
        "duration_min", "soc_start", "soc_end",
        "soh_avg", "energy_soc_kwh",
        "energy_measured_kwh", "accuracy_percent",
    ]
    df = df[colnames].round(2)

    # Ensure vehicle_id sorted correctly
    df["vehicle_id"] = pd.to_numeric(df["vehicle_id"], errors="coerce")
    df = df.sort_values(["vehicle_id", "start_time"])

    # ---------------- Fleet Summary ----------------
    fleet_summary = (
        df.assign(duration_hr=lambda x: x["duration_min"] / 60.0)
        .groupby("mode")
        .apply(lambda g: pd.Series({
            "weighted_avg_soc_kwh":
                (g["energy_soc_kwh"] * g["duration_hr"]).sum()
                / g["duration_hr"].sum(),
            "weighted_avg_measured_kwh":
                (g["energy_measured_kwh"] * g["duration_hr"]).sum()
                / g["duration_hr"].sum(),
        }))
        .round(3)
    )
    fleet_summary["difference_kwh"] = (
        fleet_summary["weighted_avg_measured_kwh"]
        - fleet_summary["weighted_avg_soc_kwh"]
    ).round(3)
    fleet_summary["accuracy_percent"] = (
        (1 - fleet_summary["difference_kwh"].abs()
         / fleet_summary["weighted_avg_soc_kwh"]) * 100
    ).round(2)

    # ---------------- Vehicle-Level Summary ----------------
    vehicle_summary = (
        df.assign(duration_hr=lambda x: x["duration_min"] / 60.0)
        .groupby(["vehicle_id", "mode"])
        .apply(lambda g: pd.Series({
            "weighted_avg_soc_kwh":
                (g["energy_soc_kwh"] * g["duration_hr"]).sum()
                / g["duration_hr"].sum(),
            "weighted_avg_measured_kwh":
                (g["energy_measured_kwh"] * g["duration_hr"]).sum()
                / g["duration_hr"].sum(),
        }))
        .reset_index()
        .round(3)
    )
    vehicle_summary["difference_kwh"] = (
        vehicle_summary["weighted_avg_measured_kwh"]
        - vehicle_summary["weighted_avg_soc_kwh"]
    ).round(3)
    vehicle_summary["accuracy_percent"] = (
        (1 - vehicle_summary["difference_kwh"].abs()
         / vehicle_summary["weighted_avg_soc_kwh"]) * 100
    ).round(2)

    # ---------------- Table Renderer ----------------
    def draw_table(ax, df, title):
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)

        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.auto_set_column_width(list(range(df.shape[1])))
        tbl.scale(1.0, 1.25)

        for _, cell in tbl.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.3)
            cell.get_text().set_fontfamily(font_family)

    # ---------------- Intro Page ----------------
    intro_text = """
    <b>SoC Accuracy Analysis Overview</b><br/><br/>
    This report analyses battery State–of–Charge (SoC) accuracy across the fleet by
    calculating energy derived from SoC movement versus the energy estimated from
    voltage-current integration.<br/><br/>

    <b>What do we compute?</b><br/>
    1. Continuous CHARGING / DISCHARGING sessions are identified based on:<br/>
       &nbsp;&nbsp;• Changes in gun connection<br/>
       &nbsp;&nbsp;• Data gaps exceeding a threshold<br/>
       &nbsp;&nbsp;• Vehicle ID transitions<br/><br/>

    2. For each session, we compute:<br/>
       &nbsp;&nbsp;• ΔSOC (start SOC – end SOC)<br/>
       &nbsp;&nbsp;• Energy_SOC = ΔSOC × SOH × battery_capacity / 10000<br/>
       &nbsp;&nbsp;• Energy_measured = ∑ (Voltage × Current × Δt) / 3.6e6<br/><br/>

    3. Accuracy = (1 – |Energy_measured – Energy_SOC| / Energy_SOC) × 100<br/><br/>

    The following pages summarise:
       • Fleet level weighted SoC accuracy<br/>
       • Vehicle-level weighted summaries<br/>
       • Session-wise tables for each vehicle<br/><br/>
    """
    def draw_intro(ax):
        ax.axis("off")
        y = 0.95
        for line in intro_text.split("\n"):
            ax.text(0.01, y, line, fontsize=10, fontfamily=font_family, va="top")
            y -= 0.05


    with PdfPages(output_path) as pdf:

        # ---------------------------------------------------
        # PAGE 1: INTRODUCTION (auto-wrap, no clipping)
        # ---------------------------------------------------
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.axis("off")

        intro_clean = [
            "SoC Accuracy Analysis Overview\n",
            "This report analyses battery State–of–Charge (SoC) accuracy across the fleet by "
            "calculating energy derived from SoC movement versus the energy estimated from "
            "voltage-current integration.\n",
            "SESSION IDENTIFICATION:",
            "• Sessions split on gun connection changes",
            "• Data gaps greater than threshold",
            "• Vehicle ID boundaries\n",
            "For each session we compute:",
            "• ΔSOC (start SOC – end SOC)",
            "• Energy_SOC = ΔSOC × SOH × battery_capacity / 10000",
            "• Energy_measured = Σ(Voltage × Current × Δt) / 3.6e6",
            "• Accuracy = (1 – |E_meas – E_soc| / E_soc) × 100\n",
            "STRUCTURE OF REPORT:",
            "• Page 2 → Fleet-level weighted SoC accuracy",
            "• Page 3 → Vehicle-level weighted SoC accuracy",
            "• Page 4+ → Session-level details for each vehicle",
        ]

        full_text = "\n".join(intro_clean)

        ax.text(
            0.01, 0.98,
            full_text,
            fontsize=11,
            family=font_family,
            va="top",
            wrap=True
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------
        # TABLE PAGINATION HELPER
        # ---------------------------------------------------
        def paginate_table(df, title, max_rows):
            pages = int(np.ceil(len(df) / max_rows))
            for p in range(pages):
                chunk = df.iloc[p * max_rows:(p + 1) * max_rows]

                fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
                ax = fig.add_axes([
                    SIDE_MARGIN / A4_WIDTH,
                    0.12,
                    usable_width / A4_WIDTH,
                    0.78
                ])
                draw_table(
                    ax,
                    chunk,
                    f"{title}  (Page {p+1}/{pages})" if pages > 1 else title
                )
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------------------------------------------
        # PAGE 2+: FLEET SUMMARY (paginated)
        # ---------------------------------------------------
        paginate_table(
            fleet_summary.reset_index(),
            "Fleet-Level SoC Accuracy Summary (Weighted)",
            max_rows_per_page
        )

        # ---------------------------------------------------
        # PAGE 3+: VEHICLE SUMMARY (paginated)
        # ---------------------------------------------------
        paginate_table(
            vehicle_summary,
            "Vehicle-Level SoC Accuracy Summary (Weighted)",
            max_rows_per_page
        )

        # ---------------------------------------------------
        # SESSION TABLES (already paginated)
        # ---------------------------------------------------
        vehicle_ids_sorted = sorted(df["vehicle_id"].unique())

        for vid in vehicle_ids_sorted:
            vdf = df[df["vehicle_id"] == vid].copy()

            paginate_table(
                vdf,
                f"Vehicle {vid} — Session SoC Accuracy",
                max_rows_per_page
            )


    print(f"📄 Final SoC Accuracy PDF created → {output_path}")


def run_stage_3(
    feather_path: str,
    parquet_out: str = "soc_accuracy_sessions_30days.parquet",
    pdf_out: str = "vehicle_wise_soc_accuracy_30days_clean.pdf",
    font_family: str = "Courier New",
):
    """
    Stage 3:
    - Load df_with_state
    - Compute SoC accuracy sessions
    - Save Parquet
    - Generate NEW Multi-Page PDF:
        Intro → Fleet Summary → Vehicle Summary → Session Pages
    """

    meta = ft.read_table(feather_path).schema
    cols_available = [f.name for f in meta]

    cols_needed = [
        "id", "timestamp", "gun_connection_status",
        "bat_soc", "soh", "bat_voltage",
        "total_battery_current"
    ]
    cols_to_load = [c for c in cols_needed if c in cols_available]

    df_soc_base = pd.read_feather(feather_path, columns=cols_to_load)

    soc_accuracy_df = calc_soc_accuracy_sessions(df_soc_base)
    soc_accuracy_df.to_parquet(parquet_out, index=False)

    export_soc_accuracy_to_pdf(
        soc_accuracy_df,
        output_path=pdf_out,
        font_family=font_family
    )

    del df_soc_base
    del soc_accuracy_df
    free_mem()

    print(f"Stage 3 complete → {parquet_out}, {pdf_out}")