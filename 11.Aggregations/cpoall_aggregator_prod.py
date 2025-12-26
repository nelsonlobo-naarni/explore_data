import gc
import os
import sys
import platform
import logging
import argparse
from datetime import datetime, date, timedelta
import pendulum
import pandas as pd
import numpy as np

from aggregator_codes.aggregator_bms_data import run_daily_tms_analysis
sys.path.append('..')
from common import db_operations
from common.db_operations import connect_to_trino, fetch_data_for_day_trino, fetch_distinct_device_ids, fetch_distinct_ids_for_day_trino, write_df_to_iceberg,execute_query


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

def run_tms_sessions_range_to_iceberg(
    conn,
    start_date: str,
    end_date: str,
    core_cols,
    df_mapping=None,
    source_schema: str = "facts_prod",
    source_table: str = "can_parsed_output_all",
    target_schema: str = "facts_dev",
    target_table: str = "bcs_tms_sessions_v1",
    partition_by=("date",),
    push_to_db: bool = True,
    collect_sessions: bool = False,
    ids=None,   # optional explicit list of device IDs (global filter)
):
    """
    Memory-safe range runner for TMS sessions.

    - Loops from start_date to end_date (inclusive), interpreted in IST.
    - For each day:
        * If `ids` is None:
            - discover the IDs that actually have data on this IST day
          else:
            - use the provided `ids` list as a filter
        * For each id in that day's list:
            - fetch_data_for_day_trino(conn, day, [id], ...)
            - run_daily_tms_analysis(...)
        * After all ids for that day:
            - concat all sessions for that day
            - optionally write that day's sessions to Iceberg
    """

    # --- normalize start_date / end_date to date objects ---
    if isinstance(start_date, str):
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start = start_date

    if isinstance(end_date, str):
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end = end_date

    if end < start:
        raise ValueError("end_date must be >= start_date")

    n_days = (end - start).days + 1

    # Optional global filter: if user passes ids, we’ll intersect with per-day ids
    global_ids_filter = None
    if ids is not None:
        global_ids_filter = set(str(x) for x in ids)

    logging.info(
        "🚀 TMS sessions range job: %s → %s (%d days)",
        start,
        end,
        n_days,
    )

    all_days_sessions = [] if collect_sessions else None

    day = start
    while day <= end:
        day_str = day.strftime("%Y-%m-%d")
        logging.info("📅 Processing IST day %s", day_str)

        # ------------------------------------------------------------------
        # Per-day ID discovery
        # ------------------------------------------------------------------
        if global_ids_filter is None:
            ids_for_day = fetch_distinct_ids_for_day_trino(
                conn,
                day,
                schema=source_schema,
                table=source_table,
            )
        else:
            # discover ids for day, then intersect with global filter
            day_ids_raw = fetch_distinct_ids_for_day_trino(
                conn,
                day,
                schema=source_schema,
                table=source_table,
            )
            ids_for_day = [vid for vid in day_ids_raw if vid in global_ids_filter]

        if not ids_for_day:
            logging.info("  ℹ️ No IDs with data on %s. Skipping entire day.", day_str)
            day += timedelta(days=1)
            continue

        logging.info(
            "  🔎 Found %d IDs with data on %s: %s",
            len(ids_for_day),
            day_str,
            ", ".join(ids_for_day),
        )

        # Collect all sessions for this day (across these IDs)
        day_sessions_list = []

        for vid in ids_for_day:
            logging.info("  🚍 Processing id=%s on %s", vid, day_str)

            # Fetch raw data for THIS (id, day)
            df_raw = fetch_data_for_day_trino(
                conn,          # connection
                day,           # date (IST calendar day)
                [vid],         # list of ids (single id)
                core_cols=core_cols,
                schema=source_schema,
                table=source_table,
            )

            if df_raw.empty:
                # This should be rare now, but keep it safe
                del df_raw
                gc.collect()
                continue

            # TMS analysis for this id/day
            daily = run_daily_tms_analysis(
                df_raw=df_raw,
                df_mapping=df_mapping,
            )
            sessions_id = daily["sessions"]

            if sessions_id.empty:
                del df_raw, daily, sessions_id
                gc.collect()
                continue

            # ensure 'date' is this IST day
            if "date" not in sessions_id.columns:
                sessions_id["date"] = day
            else:
                sessions_id["date"] = pd.to_datetime(sessions_id["date"]).dt.date
                sessions_id["date"] = day

            day_sessions_list.append(sessions_id)

            del df_raw, daily, sessions_id
            gc.collect()

        # --- after looping over all ids for this day ---
        if not day_sessions_list:
            logging.info("  ℹ️ No sessions built for %s across all IDs. Skipping DB write.", day_str)
            day += timedelta(days=1)
            continue

        day_sessions_df = pd.concat(day_sessions_list, ignore_index=True)
        logging.info(
            "  ✅ Built %d sessions total for %s (across %d IDs).",
            len(day_sessions_df),
            day_str,
            day_sessions_df["id"].nunique() if "id" in day_sessions_df.columns else -1,
        )

        # Single write per day
        if push_to_db:
            write_df_to_iceberg(
                conn=conn,
                df=day_sessions_df,
                schema=target_schema,
                table=target_table,
                partition_by=list(partition_by) if partition_by else None,
            )
            logging.info(
                "  💾 Inserted %d rows into %s.%s for %s",
                len(day_sessions_df),
                target_schema,
                target_table,
                day_str,
            )

        if collect_sessions:
            all_days_sessions.append(day_sessions_df.copy())

        del day_sessions_list, day_sessions_df
        gc.collect()

        day += timedelta(days=1)

    logging.info("✅ Completed TMS sessions range job: %s → %s", start, end)

    if collect_sessions:
        if all_days_sessions:
            return pd.concat(all_days_sessions, ignore_index=True)
        else:
            return pd.DataFrame()

    return None

# ---- Date Handling (Fixed Timezone Logic) ----
def get_target_date(args):
    """Determine which date's data to process (IST-based)."""
    tz = pendulum.timezone("Asia/Kolkata")

    if args.date:
        target_date = pendulum.parse(args.date, tz=tz).date()
    elif args.yesterday:
        target_date = pendulum.now(tz).subtract(days=1).date()
    else:
        target_date = pendulum.now(tz).date()

    logging.info(f"Script running at {pendulum.now(tz)} (IST)")
    logging.info(f"Extracting data for {target_date}")
    return target_date


def main():
    args = parse_args()

    logger.info("🔌 STEP 1: Connecting to Trino...")
    conn = connect_to_trino()
    logger.info("✅ STEP 1: Connected to Trino")

    start_date, end_date = resolve_dates(args)

    logger.info("📅 STEP 2: Determining date range...")
    logger.info(f"   → Running TMS sessions from {start_date} to {end_date} (IST basis)")

    logger.info("⚙️ STEP 3: Running TMS sessions pipeline...")
    sessions_all = run_tms_sessions_range_to_iceberg(
        conn=conn,
        start_date="2025-08-01",
        end_date="2025-08-3",
        core_cols=CORE_COLS,
        df_mapping=None,
        source_schema="facts_prod",
        source_table="can_parsed_output_all",
        target_schema="facts_dev",
        target_table="bcs_tms_sessions_v1",
        push_to_db=True,
        partition_by=("date",),
        collect_sessions=False,)   # pure pipeline mode
    
    logger.info(f"📊 STEP 3: Pipeline produced {len(sessions_all)} session rows")

    logger.info("🔒 STEP 4: Closing Trino connection...")
    conn.close()
    logger.info("✅ STEP 4: Connection closed.")
    logger.info("🎉 STEP 5: TMS sessions job completed successfully.")


if __name__ == "__main__":
    main()


# quick sanity check
# sessions_all.query("date == '2025-09-02' and id == '16'").head()    

# conn = connect_to_trino()
# sql = f"DROP TABLE IF EXISTS adhoc.facts_dev.bcs_tms_sessions_v1"
# execute_query(conn, sql, return_results=False)
# conn.close()