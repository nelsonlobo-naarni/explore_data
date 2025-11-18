import pandas as pd
import pyarrow.feather as ft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from bcs_tms_core import free_mem


def build_hotspot_tables(df: pd.DataFrame, max_col: str, min_col: str, key: str):
    """
    Generic hotspot builder for:
        - max_col / min_col (e.g. batt_maxtemp_pack / batt_mintemp_pack)
        - key: "pack_tc" or "cell_id"
    Returns:
        (fleet_table, veh_tables)
    """
    fleet_tables = {}
    veh_tables = {}

    # Fleet MAX
    fleet_max_raw = (
        df.groupby(["mode", max_col], observed=False)
        .size()
        .reset_index(name="count")
    )
    fleet_max = {}
    for mode in ["CHARGING", "DISCHARGING"]:
        sub = fleet_max_raw[fleet_max_raw["mode"] == mode].copy()
        total = sub["count"].sum() or 1
        sub["percent"] = (sub["count"] / total * 100).round(2)
        sub = sub.rename(
            columns={
                max_col: key,
                "count": f"{mode.lower()}_max_count",
                "percent": f"{mode.lower()}_max_pct",
            }
        )
        fleet_max[mode] = sub.drop(columns=["mode"], errors="ignore").set_index(key)

    # Fleet MIN
    fleet_min_raw = (
        df.groupby(["mode", min_col], observed=False)
        .size()
        .reset_index(name="count")
    )
    fleet_min = {}
    for mode in ["CHARGING", "DISCHARGING"]:
        sub = fleet_min_raw[fleet_min_raw["mode"] == mode].copy()
        total = sub["count"].sum() or 1
        sub["percent"] = (sub["count"] / total * 100).round(2)
        sub = sub.rename(
            columns={
                min_col: key,
                "count": f"{mode.lower()}_min_count",
                "percent": f"{mode.lower()}_min_pct",
            }
        )
        fleet_min[mode] = sub.drop(columns=["mode"], errors="ignore").set_index(key)

    fleet_table = (
        fleet_max["CHARGING"]
        .join(fleet_max["DISCHARGING"], how="outer")
        .join(fleet_min["CHARGING"], how="outer")
        .join(fleet_min["DISCHARGING"], how="outer")
        .fillna(0)
    )

    # Vehicle-wise
    for vid, g in df.groupby("id"):
        vmax_raw = (
            g.groupby(["mode", max_col], observed=False)
            .size()
            .reset_index(name="count")
        )
        vmin_raw = (
            g.groupby(["mode", min_col], observed=False)
            .size()
            .reset_index(name="count")
        )

        vmax_tables = {}
        vmin_tables = {}

        for mode in ["CHARGING", "DISCHARGING"]:
            sub = vmax_raw[vmax_raw["mode"] == mode].copy()
            total = sub["count"].sum() or 1
            sub["percent"] = (sub["count"] / total * 100).round(2)
            sub = sub.rename(
                columns={
                    max_col: key,
                    "count": f"{mode.lower()}_max_count",
                    "percent": f"{mode.lower()}_max_pct",
                }
            )
            vmax_tables[mode] = sub.drop(columns=["mode"], errors="ignore").set_index(
                key
            )

            sub2 = vmin_raw[vmin_raw["mode"] == mode].copy()
            total2 = sub2["count"].sum() or 1
            sub2["percent"] = (sub2["count"] / total2 * 100).round(2)
            sub2 = sub2.rename(
                columns={
                    min_col: key,
                    "count": f"{mode.lower()}_min_count",
                    "percent": f"{mode.lower()}_min_pct",
                }
            )
            vmin_tables[mode] = sub2.drop(
                columns=["mode"], errors="ignore"
            ).set_index(key)

        veh_tables[vid] = (
            vmax_tables["CHARGING"]
            .join(vmax_tables["DISCHARGING"], how="outer")
            .join(vmin_tables["CHARGING"], how="outer")
            .join(vmin_tables["DISCHARGING"], how="outer")
            .fillna(0)
        )

    # ----------------------------------------------------------
    # Append pack_id column when dealing with thermocouples (tc_id)
    # ----------------------------------------------------------
    if key == "tc_id":

        # ----- Fleet table -----
        fleet_table = fleet_table.copy()
        fleet_table["pack_id"] = ((fleet_table.index.astype(int) - 1) // 9 + 1).astype(int)

        # reorder: tc_id, pack_id, all other columns
        fleet_table = (
            fleet_table
            .reset_index()
            .loc[:, ["tc_id", "pack_id"] + [c for c in fleet_table.columns if c not in ["tc_id", "pack_id"]]]
        )

        # ----- Vehicle tables -----
        new_veh_tables = {}
        for vid, tab in veh_tables.items():
            t = tab.copy()
            t["pack_id"] = ((t.index.astype(int) - 1) // 9 + 1).astype(int)
            t = (
                t
                .reset_index()
                .loc[:, ["tc_id", "pack_id"] + [c for c in t.columns if c not in ["tc_id", "pack_id"]]]
            )
            new_veh_tables[vid] = t

        veh_tables = new_veh_tables

    else:
        # For cell-id case, just reset index normally
        fleet_table = fleet_table.reset_index()
        veh_tables = {vid: t.reset_index() for vid, t in veh_tables.items()}


    # ----------------------------------------------------------------------
    # 5) Standardize column names + cast counts to integers
    # ----------------------------------------------------------------------

    rename_map = {
        "charging_max_count": "c_max_cnt",
        "charging_max_pct": "c_max_pct",
        "discharging_max_count": "dc_max_cnt",
        "discharging_max_pct": "dc_max_pct",
        "charging_min_count": "c_min_cnt",
        "charging_min_pct": "c_min_pct",
        "discharging_min_count": "dc_min_cnt",
        "discharging_min_pct": "dc_min_pct",
    }

    # -------- Fleet table renaming --------
    fleet_table = fleet_table.rename(columns=rename_map)

    # Ensure integer counts (avoid float .0)
    for col in ["c_max_cnt", "dc_max_cnt", "c_min_cnt", "dc_min_cnt"]:
        if col in fleet_table.columns:
            fleet_table[col] = fleet_table[col].fillna(0).astype(int)

    # -------- Vehicle table renaming --------
    new_veh_tables = {}
    for vid, tab in veh_tables.items():
        t = tab.rename(columns=rename_map)

        # Integerify the count columns
        for col in ["c_max_cnt", "dc_max_cnt", "c_min_cnt", "dc_min_cnt"]:
            if col in t.columns:
                t[col] = t[col].fillna(0).astype(int)

        new_veh_tables[vid] = t

    veh_tables = new_veh_tables

    # ----------------------------------------------------------------------
    # 6) Reset index so tc_id / cell_id appear as normal columns (not floats)
    # ----------------------------------------------------------------------
    fleet_table = fleet_table.reset_index()

    for vid in veh_tables:
        veh_tables[vid] = veh_tables[vid].reset_index()


    return fleet_table, veh_tables


def clean_hotspot_df(df):
    df = df.copy()

    # Drop leftover index columns
    for col in ["level_0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure IDs are integers
    for col in ["tc_id", "pack_id", "cell_id", "vehicle_id"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # Ensure counts are whole numbers
    for col in ["c_max_cnt", "dc_max_cnt", "c_min_cnt", "dc_min_cnt"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).round().astype("Int64")

    return df


def build_pack_level_hotspots(df):
    """
    Fleet-level pack hotspot summariser.
    Uses tc_pack_max / tc_pack_min instead of tc_id.
    Returns a single fleet-level table (no vehicle-wise breakdown needed here).
    """

    # --- MAX temperature per pack ---
    max_raw = (
        df.groupby(["mode", "tc_pack_max"], observed=False)
        .size()
        .reset_index(name="count")
    )

    # --- MIN temperature per pack ---
    min_raw = (
        df.groupby(["mode", "tc_pack_min"], observed=False)
        .size()
        .reset_index(name="count")
    )

    def compute_pct_table(raw, value_col, count_name, pct_name):
        out = {}
        for mode in ["CHARGING", "DISCHARGING"]:
            sub = raw[raw["mode"] == mode].copy()
            total = sub["count"].sum() or 1
            sub["percent"] = (sub["count"] / total * 100).round(2)

            sub = sub.rename(
                columns={
                    value_col: "pack_id",
                    "count": count_name,
                    "percent": pct_name,
                }
            )

            out[mode] = sub.drop(columns=["mode"]).set_index("pack_id")
        return out

    max_tbl = compute_pct_table(
        max_raw, "tc_pack_max", "c_max_cnt", "c_max_pct"
    )
    min_tbl = compute_pct_table(
        min_raw, "tc_pack_min", "c_min_cnt", "c_min_pct"
    )

    # Combine
    fleet_pack_table = (
        max_tbl["CHARGING"]
        .join(max_tbl["DISCHARGING"], lsuffix="", rsuffix="_dc", how="outer")
        .join(min_tbl["CHARGING"], lsuffix="", rsuffix="_min_c", how="outer")
        .join(min_tbl["DISCHARGING"], lsuffix="", rsuffix="_min_dc", how="outer")
        .fillna(0)
    )

    # Clean formatting
    fleet_pack_table = fleet_pack_table.astype({
        "c_max_cnt": "int",
        "c_min_cnt": "int",
    })

    # --- NEW: expose pack_id as a normal column ---
    fleet_pack_table = fleet_pack_table.reset_index()
    fleet_pack_table["pack_id"] = fleet_pack_table["pack_id"].astype(int)

    return fleet_pack_table



def export_hotspot_analysis_to_pdf(
    fleet_pack_hotspots,
    fleet_temp_hotspots,
    veh_temp_hotspots,
    fleet_voltage_hotspots,
    veh_voltage_hotspots,
    output_pdf="hotspot_analysis_30days.pdf",
    font_family="Courier New",
    rows_per_page=30
):
    """
    Clean A4-landscape hotspot report with PACK-level → TC-level → CELL-level hierarchy.
    """

    MM = 0.0393701
    A4_WIDTH = 11.69
    A4_HEIGHT = 8.27
    SIDE_MARGIN = 2 * MM
    usable_width = A4_WIDTH - 2 * SIDE_MARGIN

    # ----------------------------------------------------------
    # Helper: Draw table with gridlines
    # ----------------------------------------------------------
    def draw_table(ax, df, title):
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)

        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=None,
            loc="center",
            cellLoc="center",
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.20)
        tbl.auto_set_column_width(col=list(range(df.shape[1])))

        for _, cell in tbl.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.3)
            cell.get_text().set_fontfamily(font_family)

    # ----------------------------------------------------------
    # Helper: Paginate tables
    # ----------------------------------------------------------
    def add_paginated_tables(pdf, df, base_title):
        df = df.reset_index(drop=True)   # clean reset
        total_rows = len(df)
        num_pages = max(1, int(np.ceil(total_rows / rows_per_page)))

        for p in range(num_pages):
            chunk = df.iloc[p*rows_per_page : (p+1)*rows_per_page]

            fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
            ax = fig.add_axes([
                SIDE_MARGIN / A4_WIDTH,
                0.12,
                usable_width / A4_WIDTH,
                0.78
            ])

            page_title = f"{base_title}  (Page {p+1}/{num_pages})"
            draw_table(ax, chunk, page_title)

            pdf.savefig(fig)
            plt.close(fig)

    # ----------------------------------------------------------
    # Generate PDF
    # ----------------------------------------------------------
    with PdfPages(output_pdf) as pdf:

        # ==========================================================
        # PAGE BLOCK 1: FLEET-LEVEL PACK HOTSPOTS (NEW)
        # ==========================================================
        add_paginated_tables(
            pdf,
            fleet_pack_hotspots,
            "Fleet-Level PACK Temperature Hotspots (Max/Min)"
        )

        # ==========================================================
        # PAGE BLOCK 2: FLEET-LEVEL SENSOR (TC) HOTSPOTS
        # ==========================================================
        add_paginated_tables(
            pdf,
            fleet_temp_hotspots,
            "Fleet-Level Individual Sensor Temperature Hotspots (TC-Level)"
        )

        # ==========================================================
        # PAGE BLOCK 3: VEHICLE-WISE SENSOR (TC) HOTSPOTS
        # ==========================================================
        for vid, dfv in veh_temp_hotspots.items():
            add_paginated_tables(
                pdf,
                dfv,
                f"Vehicle {vid} – Individual Sensor Temperature Hotspots (TC-Level)"
            )

        # ==========================================================
        # PAGE BLOCK 4: FLEET-LEVEL CELL VOLTAGE HOTSPOTS
        # ==========================================================
        add_paginated_tables(
            pdf,
            fleet_voltage_hotspots,
            "Fleet-Level Cell Voltage Hotspots (Max/Min)"
        )

        # ==========================================================
        # PAGE BLOCK 5: VEHICLE-WISE CELL VOLTAGE HOTSPOTS
        # ==========================================================
        for vid, dfv in veh_voltage_hotspots.items():
            add_paginated_tables(
                pdf,
                dfv,
                f"Vehicle {vid} – Cell Voltage Hotspots"
            )

    print(f"📄 Hotspot Analysis PDF saved → {output_pdf}")


def run_stage_6(feather_path: str):
    """
    Stage 6:
    - Load df_with_state
    - Clean & validate thermocouple and cell IDs
    - Map thermocouples (tc_id) to packs (tc_pack)
    - Compute pack-level temp hotspots (tc_id-based)
    - Compute cell-level voltage hotspots (cell_id-based)

    Returns:
        fleet_temp_hotspots, veh_temp_hotspots,
        fleet_voltage_hotspots, veh_voltage_hotspots
    """
    # --- Read schema + data ---
    schema = ft.read_table(feather_path).schema
    cols_available = [f.name for f in schema]

    cols_needed = [
        "id",
        "mode",
        "batt_maxtemp_pack",
        "batt_mintemp_pack",
        "batt_maxvolt_cell",
        "batt_minvolt_cell",
    ]
    cols_to_load = [c for c in cols_needed if c in cols_available]

    df_hot = pd.read_feather(feather_path, columns=cols_to_load)

    if "mode" not in df_hot.columns:
        raise ValueError("Column 'mode' missing in df_hot. Run Stage 5 before Stage 6.")

    # ---------- 1) Convert to numeric & clean invalid ----------
    tc_cols = ["batt_maxtemp_pack", "batt_mintemp_pack"]
    cell_cols = ["batt_maxvolt_cell", "batt_minvolt_cell"]

    # Coerce to numeric
    for col in tc_cols + cell_cols:
        df_hot[col] = pd.to_numeric(df_hot[col], errors="coerce")

    # Replace +/- inf with NaN
    df_hot.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Valid thermocouple IDs: 1..108
    df_hot["batt_maxtemp_pack"] = df_hot["batt_maxtemp_pack"].mask(
        (df_hot["batt_maxtemp_pack"] <= 0) | (df_hot["batt_maxtemp_pack"] > 108)
    )
    df_hot["batt_mintemp_pack"] = df_hot["batt_mintemp_pack"].mask(
        (df_hot["batt_mintemp_pack"] <= 0) | (df_hot["batt_mintemp_pack"] > 108)
    )

    # Valid cell IDs: 1..576
    df_hot["batt_maxvolt_cell"] = df_hot["batt_maxvolt_cell"].mask(
        (df_hot["batt_maxvolt_cell"] <= 0) | (df_hot["batt_maxvolt_cell"] > 576)
    )
    df_hot["batt_minvolt_cell"] = df_hot["batt_minvolt_cell"].mask(
        (df_hot["batt_minvolt_cell"] <= 0) | (df_hot["batt_minvolt_cell"] > 576)
    )

    # Drop any row where *either* TC or cell IDs are invalid
    df_hot = df_hot.dropna(
        subset=[
            "batt_maxtemp_pack",
            "batt_mintemp_pack",
            "batt_maxvolt_cell",
            "batt_minvolt_cell",
        ]
    ).copy()

    # ---------- 2) Rename to tc_id / cell_id ----------
    df_hot = df_hot.rename(
        columns={
            "batt_maxtemp_pack": "tc_id_max",
            "batt_mintemp_pack": "tc_id_min",
            "batt_maxvolt_cell": "cell_id_max",
            "batt_minvolt_cell": "cell_id_min",
        }
    )

    # ---------- 3) Map thermocouple IDs to packs ----------
    # tc_id 1–9 → pack 1, 10–18 → pack 2, ..., 100–108 → pack 12
    df_hot["tc_pack_max"] = ((df_hot["tc_id_max"] - 1) // 9 + 1).astype("Int64")
    df_hot["tc_pack_min"] = ((df_hot["tc_id_min"] - 1) // 9 + 1).astype("Int64")

    # (tc_pack_* not yet used in hotspots, but now available for future group-bys)

    # ---------- 4) Build hotspot tables ----------
    # ---------- 4A) Build PACK-LEVEL hotspot table (NEW) ----------
    fleet_pack_hotspots = build_pack_level_hotspots(df_hot)


    # Temperature hotspots by thermocouple ID
    fleet_temp_hotspots, veh_temp_hotspots = build_hotspot_tables(
        df_hot,
        max_col="tc_id_max",
        min_col="tc_id_min",
        key="tc_id",   # renamed from pack_tc → tc_id
    )

    # Voltage hotspots by cell ID
    fleet_voltage_hotspots, veh_voltage_hotspots = build_hotspot_tables(
        df_hot,
        max_col="cell_id_max",
        min_col="cell_id_min",
        key="cell_id",
    )

    # --- Clean fleet-level tables ---
    fleet_pack_hotspots = build_pack_level_hotspots(df_hot)
    fleet_temp_hotspots = clean_hotspot_df(fleet_temp_hotspots)
    fleet_voltage_hotspots = clean_hotspot_df(fleet_voltage_hotspots)

    # --- Clean each vehicle-level table ---
    veh_temp_hotspots = {vid: clean_hotspot_df(df) for vid, df in veh_temp_hotspots.items()}
    veh_voltage_hotspots = {vid: clean_hotspot_df(df) for vid, df in veh_voltage_hotspots.items()}

    free_mem()

    print("Stage 6 complete (hotspot tables ready).")
    return (
        fleet_pack_hotspots,        # NEW
        fleet_temp_hotspots,
        veh_temp_hotspots,
        fleet_voltage_hotspots,
        veh_voltage_hotspots,
    )
