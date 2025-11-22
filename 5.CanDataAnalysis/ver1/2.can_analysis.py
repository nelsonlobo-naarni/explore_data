# %% [markdown]
# - ASC data for vehicle 4268 bearing device ID: 6
# - Oct 27th 2025. 7:11pm - 7:28pm

# %%
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import platform
import logging
import argparse
import trino
import io
import boto3
from itertools import islice
from datetime import datetime, date, timedelta
import pendulum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %%
# Add parent directory to path
repo_path = '/Users/apple/Documents/naarni/repo/dview-naarni-data-platform'
sys.path.append(os.path.join(repo_path, 'tasks'))

# Import necessary files and its respective functions
from common.db_operations import connect_to_trino, fetch_data_for_day, write_df_to_iceberg,drop_table,execute_query
from common.optimizer_logic import optimize_dataframe_memory

# Import business logic functions
from biz_logic.energy_mileage.energy_mileage_daily_v0 import energy_mileage_stats ,impute_odometer_readings

from biz_logic.energy_consumption.energy_consumption_report import energy_consumption_stats

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Print the Python version being used
print(f"Using Python version: {platform.python_version()}")

# %%
# ---- reporting config (edit ONLY this) ----
TABLE_NAME = "can_parsed_output_100"   # <— change only this

# derived (don’t edit)
REPORT_TABLE = f"adhoc.facts_prod.{TABLE_NAME}"
REPORT_S3_LOCATION = f"s3a://naarni-data-lake/aqua/warehouse/facts_prod.db/{TABLE_NAME}/"

# %%
def fetch_data(start_date, end_date, vehicle_ids):
    """
    Fetch raw battery data from the database for the specified date range and vehicle IDs.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        vehicle_ids: List of vehicle IDs
        
    Returns:
        Tuple of (df_cpo100, df_can_ac) containing raw data from both tables
    """
    logging.info(f"Fetching raw battery data from {start_date} to {end_date} for vehicles {vehicle_ids}")
    
    # Format vehicle IDs for the query
    vehicle_ids_str = ', '.join([f"'{vid}'" for vid in vehicle_ids])
    
    # Connect to Trino
    conn = connect_to_trino(host="analytics.internal.naarni.com", port=443, user="admin", catalog="adhoc", schema="default")


    # Query for cpo100 data
    cpo100_query = f"""
    SELECT 
        *
    FROM
        facts_prod.can_parsed_output_100
    WHERE 
        id in ({vehicle_ids_str})
        and date(timestamp AT TIME ZONE 'Asia/Kolkata') between DATE('{start_date}') AND DATE('{end_date}')
    """

    # Execute queries and fetch data
    cur = conn.cursor()

    # Fetch cpo100 data
    cur.execute(cpo100_query)
    cpo100_columns = [desc[0] for desc in cur.description]
    cpo100_rows = cur.fetchall()
    df_cpo100 = pd.DataFrame(cpo100_rows, columns=cpo100_columns)

    logging.info(f"Done Fetching data.")
    logging.info(f"Retrieved {len(df_cpo100)} cpo100 records from the database.")
    
    # Close connections
    cur.close()
    conn.close()
    
    return df_cpo100

# %%
# conn = connect_to_trino(host="analytics.internal.naarni.com",port=443,user="admin",catalog="adhoc",schema="default")

# vehicle_ids=["6"]
# start_date = "2025-10-01"
# end_date = "2025-10-02"
# df_sample = fetch_data(start_date, end_date, vehicle_ids)
# display(df_sample.head(20))
# # df.to_csv("can_parsed_output_100_sample.csv", index=False)

# %%
# display(df_sample.sort_values(by=["sequence"], ascending=False).head())

# %%
df_sample = pd.read_csv("can_parsed_output_100_sample.csv")
df_clickhouse = pd.read_csv("can_parsed_output_100_clickhouse.csv")
df_raw = pd.read_csv("c2c_can_01Oct2025.csv")
df_sample.columns = df_sample.columns.str.strip().str.lower()
df_clickhouse.columns = df_clickhouse.columns.str.strip().str.lower()
df_raw.columns = df_raw.columns.str.strip().str.lower()

def clean_cols(*dfs):
    """
    Cleans column names by:
      - stripping leading/trailing whitespace
      - converting to lowercase
    Returns cleaned DataFrames in the same order.
    """
    cleaned = []
    for df in dfs:
        df.columns = df.columns.str.strip().str.lower()
        cleaned.append(df)
    return cleaned

# Apply the cleaning function to all three
df_sample, df_clickhouse, df_raw = clean_cols(df_sample, df_clickhouse, df_raw)

# %%
def to_epoch_ms_from_str(ts_str):
    """Convert '2025-10-01 00:00:00.776 115875' → epoch ms."""
    if pd.isna(ts_str):
        return np.nan
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", str(ts_str))
    if not match:
        return np.nan
    dt_part = match.group(1)
    dt = pd.to_datetime(dt_part, errors='coerce')
    return int(dt.timestamp() * 1000) if pd.notna(dt) else np.nan

df_sample["timestamp_epoch"] = df_sample["timestamp"].apply(to_epoch_ms_from_str)

# %%
def move_column(df, col_to_move, col_after):
    """
    Moves column `col_to_move` to appear right after `col_after`.
    """
    cols = list(df.columns)
    if col_to_move not in cols or col_after not in cols:
        return df  # nothing to do if columns missing

    cols.insert(cols.index(col_after) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]

# Apply it
df_sample = move_column(df_sample, "timestamp_epoch", "id")

df_sample = df_sample.drop(['timestamp', 'date', 'dt', 'rank'], axis=1, errors='ignore')
df_sample.rename(columns={"timestamp_epoch": "timestamp"}, inplace=True)
df_sample.drop('insert_timestamp', axis=1, inplace=True)
df_clickhouse.drop('timestamp.1', axis=1, inplace=True)

# %%
print("Min timestamp and number of records in SAMPLE:", df_sample.timestamp.min(), len(df_sample))
print("Max timestamp and number of records in SAMPLE:", df_sample.timestamp.max(), len(df_sample))
print("Min timestamp and number of records in CLICKHOUSE:", df_clickhouse.timestamp.min(), len(df_clickhouse))
print("Max timestamp and number of records in CLICKHOUSE:", df_clickhouse.timestamp.max(), len(df_clickhouse))
print("Min timestamp and number of records in RAW:", df_raw.timestamp.min(), len(df_raw))      #Correct IST timestamp

# %%
df_sample = df_sample.sort_values(by="timestamp").copy()
display(df_sample.head())

df_clickhouse = df_clickhouse[(df_clickhouse.timestamp>=df_sample.timestamp.min()) & (df_clickhouse.timestamp<=df_sample.timestamp.max())].copy()
df_clickhouse = df_clickhouse.reindex(columns=df_sample.columns)
df_clickhouse = df_clickhouse.sort_values(by="timestamp").copy()
display(df_clickhouse.head())

df_raw = df_raw[(df_raw.timestamp>=df_sample.timestamp.min()) & (df_raw.timestamp<=df_sample.timestamp.max())].copy()
df_raw = df_raw.sort_values(by="timestamp").copy()
display(df_raw.head())

# %%
print("Min timestamp and number of records in SAMPLE:", df_sample.timestamp.min(), len(df_sample))
print("Max timestamp and number of records in SAMPLE:", df_sample.timestamp.max(), len(df_sample))
print("Min timestamp and number of records in CLICKHOUSE:", df_clickhouse.timestamp.min(), len(df_clickhouse))
print("Max timestamp and number of records in CLICKHOUSE:", df_clickhouse.timestamp.max(), len(df_clickhouse))
print("Min timestamp and number of records in RAW:", df_raw.timestamp.min(), len(df_raw))      #Correct IST timestamp

# %%
# Reorder ClickHouse columns to match Lakehouse
df_clickhouse = df_clickhouse.reindex(columns=df_sample.columns)

# Rename ClickHouse timestamp for join clarity
df_sample = df_sample.rename(columns={"timestamp": "timestamp_lakehouse"})
df_clickhouse = df_clickhouse.rename(columns={"timestamp": "timestamp_clickhouse"})

# Join on epoch timestamp
df_merged = df_sample.merge(df_clickhouse,left_on="timestamp_lakehouse",right_on="timestamp_clickhouse",how="outer",suffixes=("_lakehouse", "_clickhouse"))

df_merged["timestamp_lakehouse"] = (
    df_merged["timestamp_lakehouse"]
    .round(0)            # ensure clean integers if floats snuck in
    .astype("Int64")     # converts safely while allowing NaN
)

# %%
df_merged.head()

# %%
# Rows missing in either source
missing_in_lakehouse = df_merged[df_merged["timestamp_lakehouse"].isna()]
missing_in_clickhouse = df_merged[df_merged["timestamp_clickhouse"].isna()]

print("Missing in Lakehouse:", len(missing_in_lakehouse))
print("Missing in ClickHouse:", len(missing_in_clickhouse))


# Columns to compare (excluding timestamps)
# Columns to compare (excluding timestamps)
# exclude_cols = ["timestamp", "timestamp_epoch"]
compare_cols = [c for c in df_sample.columns]

# Compare each Lakehouse vs ClickHouse column pair
def row_diff(row):
    for col in compare_cols:
        col_l = f"{col}_lakehouse"
        col_c = f"{col}_clickhouse"
        val_l = row.get(col_l, np.nan)
        val_c = row.get(col_c, np.nan)

        if pd.isna(val_l) and pd.isna(val_c):
            continue
        if val_l != val_c:
            return True
    return False

df_merged["is_diff"] = df_merged.apply(row_diff, axis=1)
df_diff = df_merged[df_merged["is_diff"]]

print("Number of differing rows:", len(df_diff))


# %%
# Filter for differing rows
# df_diff = df_merged[df_merged["is_diff"]].copy()

print(f"Number of differing rows: {len(df_diff)}")

# Display first few differences for a quick inspection
pd.set_option("display.max_columns", None)   # so all columns are visible
pd.set_option("display.max_colwidth", None)
display(df_diff.head(10))


# %%
df_clickhouse.head()


# %%
df_sample.head()

# %%
# # assuming df_sample and df_clickhouse are already aligned and cleaned
# cols = df_sample.columns.tolist()
# chunk_size = 57
# num_chunks = math.ceil(len(cols) / chunk_size)

# %%
def plot_chunked_heatmaps(df_merged, suffix_left="_lakehouse", suffix_right="_clickhouse",
                          chunk_size=57, label_left="Lakehouse", label_right="ClickHouse"):
    """
    Plot grouped heatmaps comparing missing values between two suffixed sets of columns
    within a merged DataFrame.

    Parameters:
    - df_merged: merged DataFrame with suffixed columns
    - suffix_left: suffix for the first dataset (default '_lakehouse')
    - suffix_right: suffix for the second dataset (default '_clickhouse')
    - chunk_size: number of columns to display per comparison pair
    - label_left / label_right: titles for plots
    """

    # Identify matching column roots (without suffix)
    base_cols = sorted(
        list(
            set(
                c.replace(suffix_left, "")
                for c in df_merged.columns
                if c.endswith(suffix_left)
            )
            & set(
                c.replace(suffix_right, "")
                for c in df_merged.columns
                if c.endswith(suffix_right)
            )
        )
    )

    n_chunks = math.ceil(len(base_cols) / chunk_size)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(base_cols))
        group = base_cols[start:end]

        # Build lists of suffixed column names for each side
        left_cols = [f"{c}{suffix_left}" for c in group]
        right_cols = [f"{c}{suffix_right}" for c in group]

        fig, axes = plt.subplots(1, 2, figsize=(25, 10), sharey=True)

        sns.heatmap(df_merged[left_cols].isnull(), cmap=['#007f5f', '#f94144'],
                    cbar=False, yticklabels=False, ax=axes[0])
        axes[0].set_title(f"{label_left} — Columns {start+1} to {end}", fontsize=13)

        sns.heatmap(df_merged[right_cols].isnull(), cmap=['#007f5f', '#f94144'],
                    cbar=False, yticklabels=False, ax=axes[1])
        axes[1].set_title(f"{label_right} — Columns {start+1} to {end}", fontsize=13)

        plt.suptitle(f"Null Comparison Heatmap (Columns {group[0]} → {group[-1]})", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# %%
plot_chunked_heatmaps(df_merged,suffix_left="_lakehouse",suffix_right="_clickhouse",chunk_size=57)

# %%
# Filter only matching timestamp rows
df_matched = df_merged[
    (df_merged["timestamp_lakehouse"].notna()) &
    (df_merged["timestamp_clickhouse"].notna()) &
    (df_merged["timestamp_lakehouse"] == df_merged["timestamp_clickhouse"])
].copy()

# %%
suffix_l, suffix_r = "_lakehouse", "_clickhouse"

# Identify shared base columns
base_cols = sorted(
    list(
        set(c.replace(suffix_l, "")
            for c in df_matched.columns if c.endswith(suffix_l))
        & set(c.replace(suffix_r, "")
            for c in df_matched.columns if c.endswith(suffix_r))
    )
)

# %%
mismatch_counts = {}
for col in base_cols:
    left = f"{col}{suffix_l}"
    right = f"{col}{suffix_r}"
    mismatches = (df_matched[left] != df_matched[right]) & (
        ~(df_matched[left].isna() & df_matched[right].isna())
    )
    mismatch_counts[col] = mismatches.sum()

# Convert to sorted Series for easy viewing
mismatch_summary = pd.Series(mismatch_counts).sort_values(ascending=False)

# %%
mismatch_summary[mismatch_summary>3]

# %%
# Find all rows with at least one differing value
def has_diff(row):
    for col in base_cols:
        l, r = f"{col}{suffix_l}", f"{col}{suffix_r}"
        val_l, val_r = row[l], row[r]
        if pd.isna(val_l) and pd.isna(val_r):
            continue
        if val_l != val_r:
            return True
    return False

df_matched["is_diff"] = df_matched.apply(has_diff, axis=1)
df_diff_rows = df_matched[df_matched["is_diff"]].copy()


# %%
def extract_diffs(row):
    diffs = {}
    for col in base_cols:
        l, r = f"{col}{suffix_l}", f"{col}{suffix_r}"
        if l in row and r in row:
            val_l, val_r = row[l], row[r]
            if pd.isna(val_l) and pd.isna(val_r):
                continue
            if val_l != val_r:
                diffs[col] = (val_l, val_r)
    return diffs

df_diff_rows["diff_columns"] = df_diff_rows.apply(extract_diffs, axis=1)
df_diff_view = df_diff_rows[["timestamp_lakehouse", "diff_columns"]]
display(df_diff_view.head(20))



