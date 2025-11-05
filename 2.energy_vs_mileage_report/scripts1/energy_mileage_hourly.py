#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import platform
import logging
import os

# Add parent directory to path (tasks/)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import db_operations
from common.optimizer_logic import get_optimal_chunk_size, process_chunk_with_function, parallel_process_dataframe, optimize_dataframe_memory, round_float_columns

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta


# In[2]:


# Configure basic logging for the business logic file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# In[3]:


# -------------------- 
# Step 3: Function to process energy_mileage_stats for every hourly
# --------------------
def energy_mileage_hourly_stats(df: pd.DataFrame, use_parallel: bool = True) -> pd.DataFrame:
    """Optimized hourly stats with chunk processing, parallel execution, and memory optimization"""
    try:
        # Memory optimization
        df = optimize_dataframe_memory(df)
        
        # Decide whether to use parallel processing
        if use_parallel and len(df) > 50000:  # Only use parallel for large datasets
            return parallel_process_dataframe(df, _process_hourly_chunk)
        else:
            return _process_hourly_chunk(df)
            
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        return pd.DataFrame()

def _process_hourly_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk of data for hourly stats"""
    df = df.copy()
    df['date'] = df['IST'].dt.date
    df['hour'] = df['IST'].dt.hour

    columns_to_impute = ['OdoMeterReading', 'Gear_Position', 'Vehiclereadycondition', 
                        'Total_Battery_Current', 'Bat_Voltage']
    critical_columns = columns_to_impute

    # Calculate raw counts BEFORE any processing
    raw_counts = (
        df.groupby(['id', df['IST'].dt.date, 'hour'], observed=False)
        .size()
        .reset_index(name='raw_datapoints')
        .rename(columns={'IST': 'date'})
    )

    # Optimized imputation using vectorized operations
    df = df.sort_values(['id', 'IST'])
    df['time_diff_seconds'] = df.groupby('id', observed=False)['IST'].diff().dt.total_seconds().fillna(0)
    
    # Vectorized power calculation
    df['power_kW'] = (df['Bat_Voltage'] * df['Total_Battery_Current']) / 1000
    
    # Set IST as index for time-based interpolation
    df_indexed = df.set_index('IST')
    
    # Impute critical columns with linear interpolation
    for col in columns_to_impute:
        df_indexed[col] = df_indexed.groupby('id', observed=False)[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both', limit=5)
        )
    
    # Reset index after interpolation
    df = df_indexed.reset_index()
    
    # Filter data once
    df = df.dropna(subset=critical_columns)
    df = df[df['Total_Battery_Current'].abs() <= 3000]
    df = df[df['time_diff_seconds'] > 0]

    # Pre-calculate masks for driving and stationary
    driving_mask = (df['Gear_Position'] == 2.0) | (df['Vehiclereadycondition'] == 1.0)
    stationary_mask = (df['Gear_Position'] == 0.0)

    # Vectorized energy calculations
    df['energy_consumption_kwh'] = np.where(
        driving_mask & (df['power_kW'] > 0),
        df['power_kW'] * (df['time_diff_seconds'] / 3600),
        0
    )
    df['regen_energy_kwh'] = np.where(
        driving_mask & (df['power_kW'] < 0),
        -df['power_kW'] * (df['time_diff_seconds'] / 3600),
        0
    )
    df['idling_energy_kwh'] = np.where(
        stationary_mask & (df['power_kW'] > 0),
        df['power_kW'] * (df['time_diff_seconds'] / 3600),
        0
    )
    
    # Vectorized distance calculation with validation
    df['distance_increment'] = df['OdoMeterReading'].diff().fillna(0)
    
    # FIX: Ensure distance is non-negative and reasonable
    # Negative distances can occur due to odometer resets or errors
    valid_distance_mask = (df['distance_increment'] >= 0) & (df['distance_increment'] <= 10) & driving_mask
    df['distance_increment'] = np.where(valid_distance_mask, df['distance_increment'], 0)

    # Group by and aggregate using optimized operations
    agg_funcs = {
        'distance_increment': 'sum',
        'energy_consumption_kwh': 'sum',
        'regen_energy_kwh': 'sum',
        'idling_energy_kwh': 'sum',
        'power_kW': 'count'  # This will be our non_null_datapoints
    }
    
    hourly_stats = df.groupby(['id', 'date', 'hour'], observed=False).agg(agg_funcs).reset_index()
    hourly_stats.rename(columns={
        'distance_increment': 'dist_travelled_km',
        'energy_consumption_kwh': 'energy_consumed_kwh',
        'regen_energy_kwh': 'regen_energy_kwh',
        'idling_energy_kwh': 'idling_energy_kwh',
        'power_kW': 'non_null_datapoints'
    }, inplace=True)
    
    # Calculate derived metrics
    hourly_stats['net_energy_kwh'] = (
        hourly_stats['energy_consumed_kwh'] - hourly_stats['regen_energy_kwh']
    )
    hourly_stats['mileage_kwh_per_km'] = np.where(
        hourly_stats['dist_travelled_km'] >= 0.1,
        hourly_stats['net_energy_kwh'] / hourly_stats['dist_travelled_km'],
        np.nan
    )
    
    # Round all float columns to 2 decimal places
    hourly_stats = round_float_columns(hourly_stats)
    
    # Merge with raw_counts
    hourly_stats = hourly_stats.merge(
        raw_counts, 
        on=['id', 'date', 'hour'], 
        how='left'
    ).fillna({'raw_datapoints': 0})
    
    # Calculate percentages
    hourly_stats['raw_data_pcnt'] = np.where(
        hourly_stats['raw_datapoints'] > 0,
        (hourly_stats['raw_datapoints'] / 3600.0 * 100).round(2),
        0
    )
    hourly_stats['non_null_data_pcnt'] = np.where(
        hourly_stats['raw_datapoints'] > 0,
        (hourly_stats['non_null_datapoints'] / hourly_stats['raw_datapoints'] * 100).round(2),
        0
    )

    hourly_stats = hourly_stats[hourly_stats['raw_datapoints'] > 0]

    return hourly_stats