import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import pandas as pd
import numpy as np

import pyarrow.feather as ft
import logging
# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from bcs_tms_core import free_mem


def analyze_battery_conditions_vehiclewise(
    df: pd.DataFrame,
    output_pdf: str | None = "battery_conditions_by_vehicle_30days.pdf",
    font_family: str = "Courier New",
):
    """
    Compute per-vehicle battery-condition distributions and (optionally) export
    a per-vehicle PDF.
    Returns:
        vehicle_results: {vid: {"temp_df":..., "delta_df":..., "volt_df":...}}
    """
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

    # --- Compute tables for all vehicles (independent of PDF) ---
    for vid, group in df.groupby("id"):
        mode_results = {}
        for mode, subset in group.groupby("mode"):
            mode_results[mode] = {
                "Battery Max Temp (%)": (
                    subset["temp_bucket"].value_counts(normalize=True) * 100
                ).round(2),
                "ΔT (°C) Range (%)": (
                    subset["temp_delta_bucket"].value_counts(normalize=True) * 100
                ).round(2),
                "Voltage Δ (mV) (%)": (
                    subset["volt_delta_bucket"].value_counts(normalize=True) * 100
                ).round(2),
            }

        temp_df = pd.concat(
            {m: r["Battery Max Temp (%)"] for m, r in mode_results.items()},
            axis=1,
        ).fillna(0)

        delta_df = pd.concat(
            {m: r["ΔT (°C) Range (%)"] for m, r in mode_results.items()},
            axis=1,
        ).fillna(0)

        volt_df = pd.concat(
            {m: r["Voltage Δ (mV) (%)"] for m, r in mode_results.items()},
            axis=1,
        ).fillna(0)

        vehicle_results[vid] = {
            "temp_df": temp_df,
            "delta_df": delta_df,
            "volt_df": volt_df,
        }

    # --- Optional per-vehicle PDF (you can disable by passing output_pdf=None) ---
    if output_pdf is not None:
        with PdfPages(output_pdf) as pdf:
            for vid, tables in vehicle_results.items():
                temp_df = tables["temp_df"]
                delta_df = tables["delta_df"]
                volt_df = tables["volt_df"]

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
                    # Font family
                    for _, cell in tbl.get_celld().items():
                        cell.get_text().set_fontfamily(font_family)

                draw(axes[0], temp_df, "Battery Max Temperature Distribution (%)")
                draw(axes[1], delta_df, "Temperature Delta (°C) Distribution (%)")
                draw(axes[2], volt_df, "Voltage Delta (mV) Distribution (%)")

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)
                gc.collect()

    return vehicle_results


def compute_fleet_summary(vehicle_results: dict, mode_agnostic: bool = False):
    temp_list, delta_list, volt_list = [], [], []
    for _, res in vehicle_results.items():
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

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# ------------------------------------------------------------
#  Small helpers for ReportLab tables
# ------------------------------------------------------------

def _excel_table(data, col_widths, title):
    """
    Build a single Excel-style table:

    Row 0 : merged title row
    Row 1 : header row (already included in `data[0]`)
    Rest  : data rows

    `data` should include the header row as its first element.
    """
    # Title row: [title, "", "", ...]
    title_row = [title] + [""] * (len(col_widths) - 1)
    table_data = [title_row] + data

    tbl = Table(table_data, colWidths=col_widths)

    style = TableStyle([
        # Outer border
        ("BOX", (0, 0), (-1, -1), 1, colors.black),

        # Grid for all non-title rows
        ("GRID", (0, 1), (-1, -1), 0.6, colors.black),

        # Title merge and style
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),

        # Header row (row 1)
        ("BACKGROUND", (0, 1), (-1, 1), colors.whitesmoke),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 1), (-1, 1), "CENTER"),
        ("VALIGN", (0, 1), (-1, 1), "MIDDLE"),

        # Body cells
        ("ALIGN", (0, 2), (-1, -1), "CENTER"),
        ("VALIGN", (0, 2), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
    ])
    tbl.setStyle(style)
    return tbl

def export_charging_conditions_csv(metric_buckets: dict, output_csv: str):
    """
    Converts metric_buckets into a CSV table:
    Metric | <20% | 20-30% | 30-40% | >40%
    """
    rows = []

    for metric, buckets in metric_buckets.items():
        rows.append({
            "Metric": metric,
            "<20%": ", ".join(map(str, buckets["<20%"])) if buckets["<20%"] else "NONE",
            "20-30%": ", ".join(map(str, buckets["20-30%"])) if buckets["20-30%"] else "NONE",
            "30-40%": ", ".join(map(str, buckets["30-40%"])) if buckets["30-40%"] else "NONE",
            ">40%": ", ".join(map(str, buckets[">40%"])) if buckets[">40%"] else "NONE",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logging.info(f"📁 Charging conditions CSV saved → {output_csv}")



def _intro_story():
    """Introductory page content."""
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        spaceAfter=12,
    )

    intro_text = """
    <b>Battery Condition Analysis Report</b><br/><br/>
    This report summarises battery temperature patterns, thermal deltas,
    and cell-voltage deltas over the selected analysis window. Values are
    calculated by grouping observations into defined bucket ranges and
    computing the percentage distribution of occurrences during
    <b>CHARGING</b> and <b>DISCHARGING</b> modes.<br/><br/>

    • <b>Battery Max Temperature Distribution</b> buckets reflect absolute maximum temperature observed per timestamp.<br/>
    • <b>Temperature Delta Distribution</b> uses (max − min) pack temperature per timestamp.<br/>
    • <b>Voltage Delta Distribution</b> uses (max − min) cell voltage per timestamp.<br/><br/>

    Fleet-level summaries aggregate all vehicles and show the proportional
    time spent in each bucket. Vehicle-level pages present the same
    distributions for each vehicle separately.
    """

    return [
        Paragraph(intro_text, body),
        Spacer(1, 10 * mm),
    ]



def build_condition_bucket_table(metric_buckets: dict):
    """
    Builds a single table summarizing:
        Metric | <20% | 20–30% | 30–40% | >40%
    """
    DISPLAY = {
        "<20%": "<20%",
        "20-30%": "20–30%",
        "30-40%": "30–40%",
        ">40%": ">40%",
    }

    data = [["Metric","<20%", "20-30%", "30-40%", ">40%"]]

    for metric, cats in metric_buckets.items():
        row = [
            metric,
            ", ".join(map(str, cats["<20%"])) if cats["<20%"] else "none",
            ", ".join(map(str, cats["20-30%"])) if cats["20-30%"] else "none",
            ", ".join(map(str, cats["30-40%"])) if cats["30-40%"] else "none",
            ", ".join(map(str, cats[">40%"])) if cats[">40%"] else "none",
        ]
        data.append(row)

    # col_widths = [50 * mm, 40 * mm, 40 * mm, 40 * mm, 40 * mm]
    col_widths = [
        50 * mm,   # Metric
        70 * mm,   # <20% (widest)
        40 * mm,   # 20-30%
        40 * mm,   # 30-40%
        40 * mm,   # >40%
    ]
    

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))

    return tbl



def export_battery_condition_pdf(
    vehicle_results: dict,
    fleet_mode_summary: dict,
    fleet_overall_summary: dict,
    charging_metric_buckets: dict,
    output_pdf: str = "battery_condition_fleet_report_30days.pdf",
):
    """
    FINAL PDF LAYOUT

    Page 1:
        Intro text only.

    Page 2:
        Fleet summary in a 3×2 grid (landscape):
            Top row:   Max Temp (CH/DIS), ΔT (CH/DIS), Volt Δ (CH/DIS)
            Bottom row: Max Temp (Fleet %), ΔT (Fleet %), Volt Δ (Fleet %)

    Pages 3+:
        One page per vehicle id:
            3 tables side-by-side (Max Temp, ΔT, Volt Δ), each CH/DIS.
    """

    # --- Bucket definitions (internal labels + display labels) ---
    TEMP_BUCKETS_INTERNAL = ["<28", "28–32", "32–35", "35–40", ">40"]
    TEMP_BUCKETS_DISPLAY = ["<28", "28-32", "32-35", "35-40", ">40"]

    DELTA_BUCKETS_INTERNAL = ["<2", "2–5", "5–8", ">8"]
    DELTA_BUCKETS_DISPLAY = ["<2", "2-5", "5-8", ">8"]

    VOLT_BUCKETS_INTERNAL = ["0–10", "10–20", "20–30", ">30"]
    VOLT_BUCKETS_DISPLAY = ["0-10", "10-20", "20-30", ">30"]

    # Helper to build mode-wise table rows: Buckets | Charging | Discharging
    def build_modewise_rows(buckets_internal, buckets_display, df_modewise):
        rows = []
        ch = df_modewise.get("CHARGING", pd.Series(dtype=float))
        dis = df_modewise.get("DISCHARGING", pd.Series(dtype=float))
        for b_int, b_disp in zip(buckets_internal, buckets_display):
            rows.append(
                [
                    b_disp,
                    f"{float(ch.get(b_int, 0)):0.2f}",
                    f"{float(dis.get(b_int, 0)):0.2f}",
                ]
            )
        return rows

    # Helper to build fleet-overall rows: Buckets | Fleet %
    def build_overall_rows(buckets_internal, buckets_display, df_overall):
        # df_overall is a DataFrame with single column "Fleet %"
        col = df_overall.iloc[:, 0]
        rows = []
        for b_int, b_disp in zip(buckets_internal, buckets_display):
            rows.append(
                [b_disp, f"{float(col.get(b_int, 0)):0.2f}"]
            )
        return rows

    # ------------------------------------------------------------
    #  Document setup
    # ------------------------------------------------------------
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    story = []

    # ============================================================
    # PAGE 1 — INTRO
    # ============================================================
    story += _intro_story()
    story.append(PageBreak())

    styles = getSampleStyleSheet()
    # ==== NEW PAGE: CHARGING CONDITION BUCKETS ====
    story.append(Paragraph("<b>Fleet Charging Stress Conditions</b>", styles["Heading2"]))
    story.append(Spacer(1, 6 * mm))

    tbl = build_condition_bucket_table(charging_metric_buckets)
    story.append(tbl)

    story.append(PageBreak())

    story.append(Paragraph("<b>Fleet-Level Battery Condition Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 6 * mm))

    # Column widths inside one "slot" (we will have 3 slots per row)
    slot_width = (297 * mm - 30 * mm) / 3.0  # (page width − margins) / 3
    modewise_colwidths = [36 * mm, 26.5 * mm, 26.5 * mm]  # sum ≈ 89 mm
    overall_colwidths = [45 * mm, 44 * mm]               # sum ≈ 89 mm

    # ------------------------------------------------------------
    # Fleet – top row (mode-wise, 3 tables)
    # ------------------------------------------------------------
    fleet_temp_mode = fleet_mode_summary["temp"]
    fleet_delta_mode = fleet_mode_summary["delta"]
    fleet_volt_mode = fleet_mode_summary["volt"]

    t1_data = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
        TEMP_BUCKETS_INTERNAL, TEMP_BUCKETS_DISPLAY, fleet_temp_mode
    )
    t2_data = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
        DELTA_BUCKETS_INTERNAL, DELTA_BUCKETS_DISPLAY, fleet_delta_mode
    )
    t3_data = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
        VOLT_BUCKETS_INTERNAL, VOLT_BUCKETS_DISPLAY, fleet_volt_mode
    )

    tbl1 = _excel_table(t1_data, modewise_colwidths, "Battery Max Temperature Distribution %")
    tbl2 = _excel_table(t2_data, modewise_colwidths, "Temperature Delta Distribution %")
    tbl3 = _excel_table(t3_data, modewise_colwidths, "Voltage Delta Distribution %")

    top_row = Table([[tbl1, tbl2, tbl3]],
                    colWidths=[slot_width, slot_width, slot_width])
    top_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(top_row)
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------
    # Fleet – second row (mode-agnostic, 3 tables)
    # ------------------------------------------------------------
    fleet_temp_overall = fleet_overall_summary["temp"]
    fleet_delta_overall = fleet_overall_summary["delta"]
    fleet_volt_overall = fleet_overall_summary["volt"]

    b1_data = [["Buckets", "Fleet Overall%"]] + build_overall_rows(
        TEMP_BUCKETS_INTERNAL, TEMP_BUCKETS_DISPLAY, fleet_temp_overall
    )
    b2_data = [["Buckets", "Fleet Overall%"]] + build_overall_rows(
        DELTA_BUCKETS_INTERNAL, DELTA_BUCKETS_DISPLAY, fleet_delta_overall
    )
    b3_data = [["Buckets", "Fleet Overall%"]] + build_overall_rows(
        VOLT_BUCKETS_INTERNAL, VOLT_BUCKETS_DISPLAY, fleet_volt_overall
    )

    btbl1 = _excel_table(b1_data, overall_colwidths, "Battery Max Temperature Distribution %")
    btbl2 = _excel_table(b2_data, overall_colwidths, "Temperature Delta Distribution %")
    btbl3 = _excel_table(b3_data, overall_colwidths, "Voltage Delta Distribution %")

    bottom_row = Table([[btbl1, btbl2, btbl3]],
                       colWidths=[slot_width, slot_width, slot_width])
    bottom_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(bottom_row)
    story.append(PageBreak())

    # ============================================================
    # VEHICLE PAGES — one page per id
    # ============================================================
    for vid, tables in vehicle_results.items():
        temp_df = tables["temp_df"]
        delta_df = tables["delta_df"]
        volt_df = tables["volt_df"]

        story.append(Paragraph(
            f"<b>Vehicle {vid} — Battery Condition Summary</b>",
            styles["Heading2"],
        ))
        story.append(Spacer(1, 6 * mm))

        t1_data_v = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
            TEMP_BUCKETS_INTERNAL, TEMP_BUCKETS_DISPLAY, temp_df
        )
        t2_data_v = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
            DELTA_BUCKETS_INTERNAL, DELTA_BUCKETS_DISPLAY, delta_df
        )
        t3_data_v = [["Buckets", "Charging", "Discharging"]] + build_modewise_rows(
            VOLT_BUCKETS_INTERNAL, VOLT_BUCKETS_DISPLAY, volt_df
        )

        vt1 = _excel_table(t1_data_v, modewise_colwidths, "Battery Max Temperature Distribution %")
        vt2 = _excel_table(t2_data_v, modewise_colwidths, "Temperature Delta Distribution %")
        vt3 = _excel_table(t3_data_v, modewise_colwidths, "Voltage Delta Distribution %")

        veh_row = Table([[vt1, vt2, vt3]],
                        colWidths=[slot_width, slot_width, slot_width])
        veh_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

        story.append(veh_row)
        story.append(PageBreak())

    # ------------------------------------------------------------
    # Build final PDF
    # ------------------------------------------------------------
    doc.build(story)
    logging.info(f"📄 Battery condition PDF saved → {output_pdf}")


def compute_charging_condition_buckets(df: pd.DataFrame):
    """
    Computes vehicle-wise % of CHARGING time spent in:
        1. batt_maxtemp > 35°C
        2. (batt_maxtemp - batt_mintemp) > 5°C
        3. (batt_maxvolt - batt_minvolt) > 0.020 V  (20 mV)

    Returns a dict:
        {
            "max temp>35 degrees": {"20-30%": [...], "30-40%": [...], ">40%": [...]},
            "DeltaT > 5 deg": {...},
            "DeltaV > 20 mV": {...}
        }
    """

    if df is None or df.empty:
        return {}

    df = df.copy()
    df = df[df["mode"] == "CHARGING"].copy()
    if df.empty:
        return {}

    # No duration column in Stage-2 → assume each row = equal sampling interval
    # So % = (#rows matching condition) / (#rows total) * 100
    total_counts = df.groupby("id").size().to_dict()

    # Conditions
    df["cond_maxtemp"] = df["batt_maxtemp"] > 35
    df["cond_deltaT"] = (df["batt_maxtemp"] - df["batt_mintemp"]) > 5
    df["cond_deltaV"] = (df["batt_maxvolt"] - df["batt_minvolt"]) > 0.020

    conditions = {
        "max temp>35 degrees": "cond_maxtemp",
        "DeltaT > 5 deg": "cond_deltaT",
        "DeltaV > 20 mV": "cond_deltaV",
    }

    results = {}

    for label, cond in conditions.items():
        pct_per_vehicle = {}

        # Count where condition is True
        cond_counts = df[df[cond]].groupby("id").size().to_dict()

        for vid, tot in total_counts.items():
            match = cond_counts.get(vid, 0)
            pct = (match / tot) * 100 if tot > 0 else 0
            pct_per_vehicle[vid] = pct

        # Bucketize
        buckets = {
            "<20%": [vid for vid, p in pct_per_vehicle.items() if p < 20],
            "20-30%": [vid for vid, p in pct_per_vehicle.items() if 20 <= p < 30],
            "30-40%": [vid for vid, p in pct_per_vehicle.items() if 30 <= p < 40],
            ">40%":   [vid for vid, p in pct_per_vehicle.items() if p >= 40],
        }

        results[label] = buckets

    return results


def run_stage_2(
    feather_path: str,
    per_vehicle_pdf: str | None = None,
    fleet_pdf: str = "battery_condition_fleet_report_30days.pdf",
    font_family: str = "Courier New",
):
    """
    Stage 2:
      1. Load df_with_state from feather.
      2. Compute vehicle-wise bucket distributions.
      3. Compute fleet mode-wise & fleet overall summaries.
      4. Export ONE consolidated PDF in A4 landscape with:
         - Intro page
         - Fleet 3×2 grid page
         - One page per vehicle (3 tables side-by-side).
    """

    meta = ft.read_table(feather_path).schema
    cols_available = [f.name for f in meta]

    cols_needed = [
        "id",
        "mode",
        "batt_maxtemp",
        "batt_mintemp",
        "batt_maxvolt",
        "batt_minvolt",
    ]

    for c in cols_needed:
        if c not in cols_available:
            raise ValueError(f"Column '{c}' missing from {feather_path}")

    df_cond = pd.read_feather(feather_path, columns=cols_needed)
    charging_condition_buckets = compute_charging_condition_buckets(df_cond)
    export_charging_conditions_csv(charging_condition_buckets,output_csv="charging_condition_buckets.csv")
    

    logging.info(f"Stage 2: loaded df_cond with {len(df_cond):,} rows.")

    # Step 1 — per-vehicle tables (no separate per-vehicle PDF here unless user wants)
    vehicle_results = analyze_battery_conditions_vehiclewise(
        df_cond,
        output_pdf=per_vehicle_pdf,   # usually None; keeps option if you want old-style PDF too
        font_family=font_family,
    )

    # Step 2 — fleet summaries
    fleet_mode = compute_fleet_summary(vehicle_results, mode_agnostic=False)
    fleet_overall = compute_fleet_summary(vehicle_results, mode_agnostic=True)

    # Step 3 — final consolidated landscape PDF
    export_battery_condition_pdf(
            vehicle_results=vehicle_results,
            fleet_mode_summary=fleet_mode,
            fleet_overall_summary=fleet_overall,
            charging_metric_buckets=charging_condition_buckets,
            output_pdf=fleet_pdf,
    )


    del df_cond
    del vehicle_results
    del fleet_mode
    del fleet_overall
    free_mem()

    logging.info(f"✅ Stage 2 complete → {fleet_pdf}")
