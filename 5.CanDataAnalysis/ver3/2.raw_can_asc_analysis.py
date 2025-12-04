#!/usr/bin/env python
# coding: utf-8

# jupyter nbconvert --to python 6.bcs_tms_analysis.ipynb \
#     --TemplateExporter.exclude_markdown=True \
#     --TemplateExporter.exclude_output_prompt=True \
#     --TemplateExporter.exclude_input_prompt=True


import os
import sys
import gc
import ctypes
import re
import numpy as np
import pandas as pd
import platform
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import zipfile
import duckdb 
import warnings
import fastparquet
from tqdm import tqdm 
from typing import List, Optional, Union
import psutil
import time # For timing the execution

warnings.filterwarnings('ignore')


# Optional: adjust pandas display for debugging; you can comment these out
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def load_asc_lines(path):
    """
    Reads a Vector .asc CAN log file and extracts only CAN data lines.
    Returns a list of clean lines ready for parsing.
    """
    clean_lines = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue

            # CAN-frame lines always start with a float timestamp
            # Example: "0.000000 1 18f0090bx Rx d 8 ff ff ff 70 7d 74 7e 7d"
            parts = ln.split()
            if len(parts) >= 12:
                # Check if first token is a float timestamp
                try:
                    float(parts[0])
                    clean_lines.append(ln)
                except:
                    continue

    return clean_lines


# volt_cols = [c for c in decoded.columns if c.startswith("Pack_cellVoltage_")]
# cell_indices = [int(c.split("_")[-1]) for c in volt_cols]

def extract_cycle_cell_values(decoded, volt_cols):
    """
    Detects full 576-cell cycles and extracts:
      - cycle_start
      - cycle_end
      - duration_sec
      - interval_sec
      - all 576 Pack_cellVoltage columns

    Returns:
        DataFrame with one row per complete cycle
    """

    cell_indices = [int(c.split("_")[-1]) for c in volt_cols]
    num_cells = len(cell_indices)

    cycles = []
    current_cells = set()
    cycle_start_time = None

    # STEP 1 — DETECT CYCLES
    for idx, row in decoded.iterrows():
        cells_present = [
            cell_indices[i] for i, present in enumerate(row[volt_cols].notna()) if present
        ]
        ts = row["timestamp_ist"]

        # Start of new sweep
        if len(current_cells) == 0:
            cycle_start_time = ts

        # Add cells detected in this row
        current_cells.update(cells_present)

        # Full sweep detected
        if len(current_cells) == num_cells:
            cycle_end_time = ts

            cycles.append({
                "cycle_start": cycle_start_time,
                "cycle_end": cycle_end_time
            })

            # Reset for next cycle
            current_cells = set()

    # STEP 2 — CALCULATE DURATION + INTERVAL
    df_cycles = pd.DataFrame(cycles)
    df_cycles["duration_sec"] = (df_cycles["cycle_end"] - df_cycles["cycle_start"]).dt.total_seconds()
    df_cycles["interval_sec"] = df_cycles["cycle_end"].diff().dt.total_seconds()
    df_cycles.loc[0, "interval_sec"] = None  # first cycle has no previous interval

    # STEP 3 — EXTRACT 576 CELL VALUES FOR EACH CYCLE
    result_rows = []

    for i, row in df_cycles.iterrows():
        start, end = row["cycle_start"], row["cycle_end"]

        # Filter data inside this cycle interval
        mask = (decoded["timestamp_ist"] >= start) & (decoded["timestamp_ist"] <= end)
        segment = decoded.loc[mask, volt_cols]

        # Forward/backfill and take the final available value
        final_values = segment.ffill().bfill().iloc[-1]

        # Build output row
        out = {
            "cycle_start": start,
            "cycle_end": end,
            "duration_sec": row["duration_sec"],
            "interval_sec": row["interval_sec"],
        }

        # Append all 576 cell values
        for col in volt_cols:
            out[col] = final_values[col]

        result_rows.append(out)

    return pd.DataFrame(result_rows)


def convert_raw_can_lines(lines, bus_id=28, base_epoch_ms=None):
    rows = []
    seq = 0

    if base_epoch_ms is None:
        base_epoch_ms = int(time.time() * 1000)

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        parts = ln.split()

        rel_ts = float(parts[0])
        can_hex = parts[2][:-1]
        byte_vals = parts[-8:]

        # Create byte1...byte8 as integers
        bytes_dec = [int(b, 16) for b in byte_vals]

        ts_ms = base_epoch_ms + int(rel_ts * 1000)

        rows.append([
            bus_id,
            ts_ms,
            seq,
            int(can_hex,16),   # raw CAN ID
        ] + bytes_dec)

        seq += 1

    cols = ["id","timestamp","sequence","can_id"] + [f"byte{i}" for i in range(1,9)]
    return pd.DataFrame(rows, columns=cols)


dbc_lines = open("canId_conversions.txt").read().splitlines()

temp_map = {}   # dbc_id → list of 1–108
volt_map = {}   # dbc_id → list of 1–576

current_dbc_id = None
current_type = None  # "temp" or "volt"

temp_pattern = re.compile(r"SG_\s+Pack_Temperature(\d+)")
volt_pattern = re.compile(r"SG_\s+Pack_cellVoltage_(\d+)")

for line in dbc_lines:
    line = line.strip()

    # Detect BO_ lines
    if line.startswith("BO_ "):
        parts = line.split()
        current_dbc_id = int(parts[1])
        name = parts[2]

        if name.startswith("Pack_Temperature"):
            current_type = "temp"
            temp_map[current_dbc_id] = []

        elif name.startswith("Pack_cellVoltage"):
            current_type = "volt"
            volt_map[current_dbc_id] = []

        else:
            current_type = None  # ignore unrelated frames

        continue

    # Inside a BO_ block → extract signal names
    if current_type == "temp":
        m = temp_pattern.search(line)
        if m:
            idx = int(m.group(1))
            temp_map[current_dbc_id].append(idx)

    elif current_type == "volt":
        m = volt_pattern.search(line)
        if m:
            idx = int(m.group(1))
            volt_map[current_dbc_id].append(idx)


print("Temperature blocks:", len(temp_map))
print("Total temperature indices:", sum(len(v) for v in temp_map.values()))

print("Voltage blocks:", len(volt_map))
print("Total voltage indices:", sum(len(v) for v in volt_map.values()))


asc_path = "can_20251027193535.asc"

raw_lines = load_asc_lines(asc_path)

df_asc = convert_raw_can_lines(
    raw_lines,
    bus_id=28,
    base_epoch_ms=1764229199971   # or any base timestamp you want
)

display(df_asc.head())


df_asc.columns


df_map = pd.read_csv("dbc_to_raw_can_id_mapping.csv")
df_map = df_map.rename(columns={
    "dbc_can_id": "dbc_id",
    "raw_can_id": "raw_id"
})
df_map = df_map.rename(columns={
    "dbc_can_id": "dbc_id",
    "raw_can_id": "raw_id"
})
df_map.head()

# RAW → DBC (for decoding incoming CAN messages)
raw_to_dbc = dict(zip(df_map["raw_id"], df_map["dbc_id"]))

# DBC → RAW (if you ever need it)
dbc_to_raw = dict(zip(df_map["dbc_id"], df_map["raw_id"]))


df_asc['timestamp_ist'] = pd.to_datetime(df_asc['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
df_asc["dbc_id"] = df_asc["can_id"].map(raw_to_dbc)
df_asc.dbc_id = df_asc.dbc_id.astype("Int64")
df_asc[~df_asc["dbc_id"].isna()].head()


decoded_asc = df_asc[["id","timestamp_ist","sequence"]].copy()

# Empty columns
for i in range(1, 577):
    decoded_asc[f"pack_cellvoltage_{i}"] = None

for i in range(1, 109):
    decoded_asc[f"pack_temperature_{i}"] = None

# decoded_asc = df_asc[["id", "timestamp_ist", "sequence"]].copy()
# decoded_asc["payload"] = df_asc[byte_cols].values.tolist()
byte_cols = [f"byte{i}" for i in range(1,9)]
df_asc["payload"] = df_asc[byte_cols].values.tolist()
display(decoded_asc.head())


df_asc[~df_asc.dbc_id.isna()].head()


for idx, row in df_asc.iterrows():
    dbc = row["dbc_id"]
    bytes_ = row["payload"]

    # Temperature frames
    if dbc in temp_map:
        sensor_indices = temp_map[dbc]
        for k, sensor_id in enumerate(sensor_indices):
            if sensor_id <= 108:
                decoded_asc.loc[idx, f"pack_temperature_{sensor_id}"] = bytes_[k] - 40

    # Voltage frames
    if dbc in volt_map:
        cells = volt_map[dbc]
        for k, cell_index in enumerate(cells):
            L = bytes_[k*2]       # lower byte
            H = bytes_[k*2 + 1]   # higher byte

            raw_val = (H << 8) | L
            decoded_value = raw_val * 0.001  # as per DBC scaling

            decoded_asc.loc[idx, f"pack_celloltage_{cell_index}"] = decoded_value


volt_cols = [c for c in decoded_asc.columns if c.startswith("pack_cellvoltage_")]

df_asc_cycle_voltages = extract_cycle_cell_values(decoded_asc, volt_cols)

display(df_asc_cycle_voltages)

