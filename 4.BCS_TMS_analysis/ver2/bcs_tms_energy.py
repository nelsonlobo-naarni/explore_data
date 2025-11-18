import numpy as np
import pandas as pd

from bcs_tms_core import free_mem


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

        diff_kwh = w_avg_meas - w_avg_soc

        if w_avg_soc > 1e-6:
            diff_pct = (1 - abs(diff_kwh) / w_avg_soc) * 100
            diff_pct = round(diff_pct, 2)
        else:
            diff_pct = np.nan

        results.append(
            {
                "vehicle_id": vid,
                "mode": mode,
                "total_time_hr": round(total_time, 3),
                "weighted_avg_energy_soc_kwh": round(w_avg_soc, 3),
                "weighted_avg_energy_measured_kwh": round(w_avg_meas, 3),
                "difference_kwh": round(diff_kwh, 3),
                "difference_percent": diff_pct,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(["vehicle_id", "mode"])
        .reset_index(drop=True)
    )


def run_stage_4(
    parquet_path: str = "soc_accuracy_sessions_30days.parquet",
    out_csv: str = "weighted_energy_summary_30days.csv",
):
    """
    Stage 4:
    - Load SoC sessions
    - Compute time-weighted energy metrics
    - Save CSV
    """
    soc_accuracy_df = pd.read_parquet(parquet_path)
    weighted_energy_summary = compute_time_weighted_energy(soc_accuracy_df)
    weighted_energy_summary.to_csv(out_csv, index=False)

    del soc_accuracy_df
    del weighted_energy_summary
    free_mem()

    print(f"Stage 4 complete → {out_csv}")
