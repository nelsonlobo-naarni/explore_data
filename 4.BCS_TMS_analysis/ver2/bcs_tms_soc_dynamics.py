import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyarrow.feather as ft
from bcs_tms_core import free_mem, infer_mode_from_gun_status
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import A4, landscape



def _draw_table_minimal(ax, df, title, font_family="Courier New"):
    ax.axis("off")
    ax.text(
        0.0,
        1.05,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)

    # Now with visible gridlines
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.25)
        cell.get_text().set_fontfamily(font_family)



def excel_table(title, headers, rows, col_widths):
    """
    Builds a boxed table with:
    - merged bold title row
    - bold header row
    - gridlines
    """
    data = [[title] + [""]*(len(headers)-1)]  # merged title row
    data.append(headers)
    data.extend(rows)

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("SPAN", (0,0), (-1,0)),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,0), "CENTER"),

        ("BACKGROUND", (0,1), (-1,1), colors.whitesmoke),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("ALIGN", (0,1), (-1,1), "CENTER"),

        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return tbl


def create_doc(output_path):
    return SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=15*mm,
        rightMargin=15*mm,
        topMargin=12*mm,
        bottomMargin=12*mm
    )


def generate_soc_occupancy_report(
    fleet_soc_table,
    veh_soc_tables,
    output_pdf="soc_occupancy_report.pdf"
):
    """
    Creates a standalone SOC OCCUPANCY report:
    - Page 1: Intro + formulas
    - Page 2: Fleet-level table
    - Page 3+: Per-vehicle tables
    """

    doc = create_doc(output_pdf)
    story = []
    styles = getSampleStyleSheet()

    # ---------------- PAGE 1 — INTRO ----------------
    intro = """
    <b>SoC Occupancy Analysis</b><br/><br/>
    This report measures how long the battery spends in different SoC ranges
    during charging and discharging. Each timestamp is assigned a bucket 
    (0–10%, 10–20%, …, 95–100%) and time is aggregated within those ranges.<br/><br/>

    <b>What is being computed?</b><br/>
    • dt = time difference between consecutive records<br/>
    • dt is summed inside each SoC bucket<br/>
    • Percentage = dt_bucket / dt_total × 100<br/><br/>

    Fleet-level tables summarise SoC exposure of all vehicles.
    Vehicle-level tables show SoC distribution for each bus individually.
    """

    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 10*mm))
    story.append(PageBreak())

    # ---------------- PAGE 2 — FLEET ----------------
    rows = [[idx] + list(fleet_soc_table.loc[idx]) for idx in fleet_soc_table.index]
    headers = ["SoC Bucket"] + list(fleet_soc_table.columns)
    col_widths = [35*mm] + [30*mm]*len(fleet_soc_table.columns)

    fleet_tbl = excel_table("Fleet-Level SoC Occupancy (%)", headers, rows, col_widths)
    story.append(fleet_tbl)
    story.append(PageBreak())

    # ---------------- PER VEHICLE ----------------
    for vid in sorted(veh_soc_tables.keys()):
        tab = veh_soc_tables[vid]
        rows = [[idx] + list(tab.loc[idx]) for idx in tab.index]

        veh_tbl = excel_table(
            f"Vehicle {vid} — SoC Occupancy (%)",
            headers,
            rows,
            col_widths
        )
        story.append(veh_tbl)
        story.append(PageBreak())

    doc.build(story)
    print(f"📄 SoC Occupancy PDF created → {output_pdf}")


def generate_soc_jumps_report(
    fleet_jump_table,
    veh_jump_tables,
    output_pdf="soc_jumps_report.pdf"
):
    doc = create_doc(output_pdf)
    story = []
    styles = getSampleStyleSheet()

    intro = """
    <b>SoC Jump / Drop Analysis</b><br/><br/>
    This report measures the rate of SoC change normalised by time (ΔSOC / Δt).
    Positive values represent upward jumps (charging), negative values represent
    SoC drops (discharging).<br/><br/>

    Buckets include:<br/>
    • <-10, -10 to -5, -5 to -2, -2 to -1, -1 to 1, 1 to 2, 2 to 5, 5 to 10, >10<br/><br/>

    For each bucket, we compute:<br/>
    • Count of occurrences<br/>
    • Percentage share within each mode<br/><br/>
    """

    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 10*mm))
    story.append(PageBreak())

    # ---------------- FLEET ----------------
    rows = [[idx] + list(fleet_jump_table.loc[idx]) for idx in fleet_jump_table.index]
    headers = ["Jump Bucket"] + list(fleet_jump_table.columns)
    col_widths = [35*mm] + [27*mm]*len(fleet_jump_table.columns)

    fleet_tbl = excel_table("Fleet-Level SoC Jumps/Drops (Count + %)", headers, rows, col_widths)
    story.append(fleet_tbl)
    story.append(PageBreak())

    # ---------------- VEHICLES ----------------
    for vid in sorted(veh_jump_tables.keys()):
        tab = veh_jump_tables[vid]
        rows = [[idx] + list(tab.loc[idx]) for idx in tab.index]

        veh_tbl = excel_table(
            f"Vehicle {vid} — SoC Jumps/Drops (Count + %)",
            headers,
            rows,
            col_widths
        )
        story.append(veh_tbl)
        story.append(PageBreak())

    doc.build(story)
    print(f"📄 SoC Jump/Drop PDF created → {output_pdf}")


def generate_c_rate_report(
    fleet_c_table,
    veh_c_table,
    output_pdf="c_rate_report.pdf"
):
    doc = create_doc(output_pdf)
    story = []
    styles = getSampleStyleSheet()

    intro = """
    <b>C-Rate Analysis</b><br/><br/>
    C-rate represents charging or discharging intensity normalised by 
    battery capacity (A / Ah).<br/><br/>

    This report summarises:<br/>
    • Mean C-rate<br/>
    • Median C-rate<br/>
    • Maximum observed C-rate<br/><br/>

    Values are aggregated at fleet level and per vehicle.<br/><br/>
    """

    # ---------------- PAGE 1 — INTRO ----------------
    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 10*mm))
    story.append(PageBreak())

    # ---------------- PAGE 2 — FLEET SUMMARY ----------------
    rows = [[idx] + list(row) for idx, row in fleet_c_table.iterrows()]
    headers = ["Mode"] + list(fleet_c_table.columns)

    col_widths = []
    for h in headers:
        if h == "Mode":
            col_widths.append(35 * mm)
        else:
            col_widths.append(40 * mm)

    fleet_tbl = excel_table("Fleet-Level C-Rate Summary", headers, rows, col_widths)
    story.append(fleet_tbl)
    story.append(PageBreak())

    # =========================================================
    # PAGE 3 — CONSOLIDATED VEHICLE SUMMARY (Clustered Header)
    # =========================================================

    # ---- Prepare rows ----
    clustered_rows = []
    for vid in veh_c_table["id"].unique():
        sub = veh_c_table[veh_c_table["id"] == vid]

        ch_mean  = sub["mean_CHARGING"].iloc[0] if "mean_CHARGING" in sub else 0
        ch_med   = sub["median_CHARGING"].iloc[0] if "median_CHARGING" in sub else 0
        ch_max   = sub["max_CHARGING"].iloc[0] if "max_CHARGING" in sub else 0

        dis_mean = sub["mean_DISCHARGING"].iloc[0] if "mean_DISCHARGING" in sub else 0
        dis_med  = sub["median_DISCHARGING"].iloc[0] if "median_DISCHARGING" in sub else 0
        dis_max  = sub["max_DISCHARGING"].iloc[0] if "max_DISCHARGING" in sub else 0

        clustered_rows.append([
            vid,
            f"{ch_mean:.3f}", f"{ch_med:.3f}", f"{ch_max:.3f}",
            f"{dis_mean:.3f}", f"{dis_med:.3f}", f"{dis_max:.3f}"
        ])

    # ---- Table with multi-row headers ----
    data = [
        ["ID", "Charging", "", "", "Discharging", "", ""],
        ["", "Mean", "Median", "Max", "Mean", "Median", "Max"]
    ]

    data.extend(clustered_rows)

    col_widths = [
        18*mm,    # ID
        22*mm, 22*mm, 22*mm,   # Charging columns
        22*mm, 22*mm, 22*mm    # Discharging columns
    ]

    tbl = Table(data, colWidths=col_widths)

    tbl.setStyle(TableStyle([
        # --------- Title Row Merges ---------
        ("SPAN", (1,0), (3,0)),  # Charging merges across 3 cols
        ("SPAN", (4,0), (6,0)),  # Discharging merges across 3 cols

        # --------- ID header cell ----------
        ("SPAN", (0,0), (0,1)),

        # Center alignment everywhere
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

        # Header formatting
        ("BACKGROUND", (0,0), (-1,1), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,1), "Helvetica-Bold"),

        # Gridlines
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),

        # Row heights
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("BOTTOMPADDING", (0,1), (-1,1), 4)
    ]))

    story.append(Paragraph("<b>Vehicle-Level C-Rate Summary (Clustered View)</b>", styles["Heading4"]))
    story.append(Spacer(1, 5*mm))
    story.append(tbl)
    story.append(PageBreak())

    # ---------------- BUILD DOCUMENT ----------------
    doc.build(story)
    print(f"📄 C-Rate PDF created → {output_pdf}")




def run_stage_5(
    feather_path: str,
    pdf_path: str = "soc_dynamics_and_c_rate_report_30days.pdf",
    font_family: str = "Courier New",
):
    """
    Stage 5:
    - SoC occupancy
    - SoC jump/drop buckets
    - C-rate summaries
    - Generates THREE polished ReportLab PDFs:
        * soc_occupancy_report.pdf
        * soc_jumps_report.pdf
        * c_rate_report.pdf
    """

    # ---------------- LOAD ----------------
    schema = ft.read_table(feather_path).schema
    cols_available = [f.name for f in schema]

    cols_needed = [
        "id", "timestamp", "bat_soc",
        "total_battery_current", "gun_connection_status",
    ]
    cols_to_load = [c for c in cols_needed if c in cols_available]

    df_soc_dyn = pd.read_feather(feather_path, columns=cols_to_load)
    df_soc_dyn["timestamp"] = pd.to_datetime(df_soc_dyn["timestamp"], errors="coerce")
    df_soc_dyn = (
        df_soc_dyn.dropna(subset=["timestamp"])
        .sort_values(["id", "timestamp"])
        .reset_index(drop=True)
    )

    # ---------------- MODE ----------------
    df_soc_dyn["mode"] = infer_mode_from_gun_status(df_soc_dyn["gun_connection_status"])

    # ---------------- TIME DELTA ----------------
    df_soc_dyn["dt_sec"] = (
        df_soc_dyn.groupby("id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .clip(lower=0, upper=300)
        .fillna(0)
    )
    df_soc_dyn["dt_min"] = df_soc_dyn["dt_sec"] / 60.0

    # ---------------- SoC BUCKETS ----------------
    soc_bins = [0,10,20,30,40,50,60,70,80,85,90,95,100]
    soc_labels = ["0-10","10-20","20-30","30-40","40-50",
                  "50-60","60-70","70-80","80-85",
                  "85-90","90-95","95-100"]

    df_soc_dyn["soc_bucket"] = pd.cut(
        df_soc_dyn["bat_soc"], bins=soc_bins, labels=soc_labels,
        include_lowest=True
    )

    # ---------------- FLEET SoC OCCUPANCY ----------------
    tmp = (
        df_soc_dyn.groupby(["mode", "soc_bucket"], observed=False)["dt_sec"]
        .sum()
        .reset_index()
    )
    tmp["percent"] = (
        tmp["dt_sec"] / tmp.groupby("mode")["dt_sec"].transform("sum") * 100
    )

    fleet_soc_table = (
        tmp.pivot_table(
            index="soc_bucket",
            columns="mode",
            values="percent",
            fill_value=0,
            observed=False
        )
        .round(2)
        .sort_index()
    )

    # ---------------- VEHICLE SoC OCCUPANCY ----------------
    tmp2 = (
        df_soc_dyn.groupby(["id", "mode", "soc_bucket"], observed=False)["dt_sec"]
        .sum()
        .reset_index()
    )
    tmp2["percent"] = (
        tmp2["dt_sec"] /
        tmp2.groupby(["id","mode"])["dt_sec"].transform("sum") * 100
    )
    veh_soc_tables = {
        vid: (
            g.pivot_table(
                index="soc_bucket",
                columns="mode",
                values="percent",
                fill_value=0,
                observed=False
            )
            .round(2)
            .sort_index()
        )
        for vid, g in tmp2.groupby("id")
    }

    # ---------------- SoC JUMPS ----------------
    df_soc_dyn["soc_diff"] = df_soc_dyn.groupby("id")["bat_soc"].diff()
    df_soc_dyn["soc_diff_per_min"] = df_soc_dyn["soc_diff"] / df_soc_dyn["dt_min"].replace(0, np.nan)

    jump_bins = [-999,-10,-5,-2,-1,1,2,5,10,999]
    jump_labels = ["<-10","-10 to -5","-5 to -2","-2 to -1",
                   "-1 to 1","1 to 2","2 to 5","5 to 10",">10"]

    df_soc_dyn["jump_bucket"] = pd.cut(
        df_soc_dyn["soc_diff_per_min"], bins=jump_bins, labels=jump_labels
    )

    # Fleet jump distribution (count + %)
    fleet_jumps_raw = (
        df_soc_dyn.groupby(["mode","jump_bucket"], observed=False)["soc_diff_per_min"]
        .count()
        .reset_index()
        .rename(columns={"soc_diff_per_min":"count"})
    )

    fleet_jump_counts = fleet_jumps_raw.pivot_table(
        index="jump_bucket",
        columns="mode",
        values="count",
        fill_value=0,
        observed=False
    )
    fleet_jump_pct = (fleet_jump_counts.div(fleet_jump_counts.sum()) * 100).round(2)

    fleet_jump_table = pd.DataFrame({
        "Charge Count": fleet_jump_counts.get("CHARGING",0),
        "Charge %": fleet_jump_pct.get("CHARGING",0),
        "Discharge Count": fleet_jump_counts.get("DISCHARGING",0),
        "Discharge %": fleet_jump_pct.get("DISCHARGING",0),
    }).fillna(0)

    # Vehicle jump tables
    veh_jump_tables = {}
    tmp3 = (
        df_soc_dyn.groupby(["id","mode","jump_bucket"], observed=False)["soc_diff_per_min"]
        .count()
        .reset_index()
        .rename(columns={"soc_diff_per_min":"count"})
    )
    for vid, g in tmp3.groupby("id"):
        wide = g.pivot_table(
            index="jump_bucket",
            columns="mode",
            values="count",
            fill_value=0,
            observed=False
        )
        pct = (wide.div(wide.sum()) * 100).round(2)
        veh_jump_tables[vid] = pd.DataFrame({
            "Charge Count": wide.get("CHARGING",0),
            "Charge %": pct.get("CHARGING",0),
            "Discharge Count": wide.get("DISCHARGING",0),
            "Discharge %": pct.get("DISCHARGING",0)
        }).fillna(0)

    # ---------------- C-RATE ----------------
    BAT_CAPACITY_AH = 423.8
    df_soc_dyn["c_rate"] = (df_soc_dyn["total_battery_current"] / BAT_CAPACITY_AH).clip(-3,3)

    fleet_c_table = (
        df_soc_dyn.groupby("mode")["c_rate"]
        .agg(["mean","median","max"])
        .round(3)
    )

    veh_c_contiguous = (
        df_soc_dyn.groupby(["id","mode"])["c_rate"]
        .agg(["mean","median","max"])
        .round(3)
        .reset_index()
    )

    veh_c_table = veh_c_contiguous.pivot_table(
        index="id",
        columns="mode",
        values=["mean","median","max"],
        fill_value=0,
        observed=False
    )
    veh_c_table.columns = [f"{m}_{mode}" for (m,mode) in veh_c_table.columns]
    veh_c_table = veh_c_table.reset_index()

    # ---------------- FINAL PDF REPORTS ----------------
    generate_soc_occupancy_report(fleet_soc_table, veh_soc_tables)
    generate_soc_jumps_report(fleet_jump_table, veh_jump_tables)
    generate_c_rate_report(fleet_c_table, veh_c_table)

    del df_soc_dyn
    free_mem()

    print("Stage 5 complete — all three modular reports generated.")