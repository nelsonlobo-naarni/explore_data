import os
import sys
import gc
import ctypes
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------
# Minimal logging: only errors and above will show
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------
def free_mem():
    """Try to return freed memory back to the OS (no-op on some platforms)."""
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------
CORE_COLS = [
    "id", "timestamp", "dt",
    "vehiclereadycondition", "gun_connection_status", "ignitionstatus",
    "vehicle_speed_vcu", "gear_position",
    "bat_soc", "soh", "total_battery_current",
    "pack1_cellmax_temperature", "pack1_cell_min_temperature",
    "pack1_maxtemperature_cell_number", "pack1_celltemperature_cellnumber",
    "bat_voltage", "cellmax_voltagecellnumber", 
    "cellminvoltagecellnumber", "cell_min_voltage","cell_max_voltage",
    "dcdcbus",
]


DEFAULT_VEHICLE_IDS = [
    '3','16','18','19','32','42','6','7','9','11','12','13','14','15','20',
    '25','27','28','29','30','31','33','35','41','46'
]


# ---------------------------------------------------------------------
# Mode inference helper (CHARGING vs DISCHARGING)
# ---------------------------------------------------------------------
def infer_mode_from_gun_status(series: pd.Series) -> pd.Series:
    """
    Infer CHARGING / DISCHARGING from gun_connection_status-like column.
    """
    gcs_raw = series
    gcs_num = pd.to_numeric(gcs_raw, errors="coerce")
    gcs_str = gcs_raw.astype(str).str.strip().str.lower()
    gun_connected = (gcs_num == 1) | gcs_str.isin(
        {"1", "true", "yes", "y", "connected", "on"}
    )
    return np.where(gun_connected, "CHARGING", "DISCHARGING")


# ---------------------------------------------------------------------
# Time window helper (IST → UTC)
# ---------------------------------------------------------------------
def compute_utc_window_from_ist(date_str: str, days: int = 31):
    """
    Given a date string (YYYY-MM-DD) interpreted in IST, compute
    corresponding UTC start and end datetimes for [date, date+days).
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    ist_start = datetime.combine(target_date, datetime.min.time())
    ist_end = ist_start + timedelta(days=days)

    # IST = UTC + 5:30
    utc_start = ist_start - timedelta(hours=5, minutes=30)
    utc_end = ist_end - timedelta(hours=5, minutes=30)
    return utc_start, utc_end
