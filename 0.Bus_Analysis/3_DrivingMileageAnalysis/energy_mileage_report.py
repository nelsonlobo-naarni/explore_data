#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import sys
import platform
import logging

sys.path.append('..')
from common import db_operations

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from common.db_operations import connect_to_trino, fetch_data_for_day, write_df_to_iceberg


# In[2]:


# Configure basic logging for the business logic file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Print the Python version being used
print(f"Using Python version: {platform.python_version()}")


# In[3]:


# ---- report configuration ----
TABLE_NAME = "energy_mileage_report"
SOURCE_TABLE = "can_parsed_output_100"
COLUMNS_TO_FETCH = [
    '"id"','"timestamp"',
    'at_timezone("timestamp", \'Asia/Kolkata\') AS IST',
    '"BAT_SOC"',
    '"Bat_Voltage"',
    '"Total_Battery_Current"',
    '"GUN_Connection_Status"',
    '"OdoMeterReading"',
    '"Gear_Position"',
    '"Vehiclereadycondition"'
]


# In[4]:


# --------------------
# Step 3: Function to process vehicle energy stats
# --------------------
def analyze_vehicle_energy_stats(df:pd.DataFrame):
    """
    Performs a combined analysis on an electric bus dataset that may contain
    data from multiple vehicles. The function calculates daily mileage, driving
    energy consumption, regenerative braking energy, and idling energy for
    each vehicle.
    """
    try:
        # Load the dataset
        # df = pd.read_csv(file_path)

        # --- General Data Preparation and Filtering ---
        # Ignore data points with extreme current values.
        df = df[df['Total_Battery_Current'].abs() <= 3000].copy()

        # Drop rows with missing data in key columns, including IST.
        # df.loc[:, 'IST'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df.dropna(subset=['OdoMeterReading', 'Gear_Position', 'Vehiclereadycondition', 'Total_Battery_Current', 'Bat_Voltage','IST'], inplace=True)

        # Sort data by vehicle ID and timestamp to ensure correct sequential calculations.
        df.sort_values(by=['id', 'IST'], inplace=True)

        # Extract the date for daily grouping
        df['date'] = df['IST'].dt.date

        # Get a list of unique vehicle IDs to iterate through
        unique_ids = df['id'].unique()
        all_daily_stats = []

        print("\n--- Daily Vehicle Statistics (Driving, Regenerative Braking, and Idling) ---")

        # Iterate over each unique vehicle ID
        for vehicle_id in unique_ids:
            vehicle_df = df[df['id'] == vehicle_id].copy()

            # Check if the grouped DataFrame is empty
            if vehicle_df.empty:
                print(f"No charging events were detected for device {vehicle_id}.")
                continue  

            # Calculate the time difference between consecutive data points for the current vehicle.
            vehicle_df['time_diff_seconds'] = vehicle_df['IST'].diff().dt.total_seconds().fillna(0)

            # Filter out records where time difference is zero or negative.
            vehicle_df = vehicle_df[vehicle_df['time_diff_seconds'] > 0]

            # Calculate the power in kW
            vehicle_df['power_kW'] = (vehicle_df['Bat_Voltage'] * vehicle_df['Total_Battery_Current']) / 1000

            # --- Analysis 1: Daily Driving and Regenerative Braking Statistics ---
            driving_df = vehicle_df[vehicle_df['Gear_Position'] == 2.0].copy()

            # Calculate energy consumption (positive power) and regenerative braking energy (negative power)
            driving_df['energy_consumption_kwh'] = driving_df.apply(
                lambda row: row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] > 0 else 0, axis=1)
            driving_df['regen_energy_kwh'] = driving_df.apply(
                lambda row: -row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] < 0 else 0, axis=1)

            # Calculate incremental mileage for each row.
            driving_df['distance_increment'] = driving_df['OdoMeterReading'].diff().fillna(0)

            # Filter out unrealistic mileage jumps (e.g., > 10 km in a single time step)
            driving_df = driving_df[driving_df['distance_increment'] <= 10]

            if not driving_df.empty:
                grouped_by_day_driving = driving_df.groupby(['id', 'date'])
                daily_driving_mileage = grouped_by_day_driving['distance_increment'].sum()
                daily_driving_energy = grouped_by_day_driving['energy_consumption_kwh'].sum()
                daily_regen_energy = grouped_by_day_driving['regen_energy_kwh'].sum()

                daily_stats_df = pd.DataFrame({
                    'dist_travelled_km': round(daily_driving_mileage, 2), #Mileage in km
                    'energy_consumed_kwh': round(daily_driving_energy, 2), #Daily Driving Energy Consumed (kWh)
                    'regen_energy_kwh': round(daily_regen_energy, 2) #Daily Regenerative Braking Energy (kWh)
                }).reset_index()
            else:
                # Create an empty DataFrame with the correct columns if there is no driving data
                daily_stats_df = pd.DataFrame(columns=[
                    'id', 'date', 'dist_travelled_km',
                    'energy_consumed_kwh',
                    'regen_energy_kwh'
                ])

            # --- Analysis 2: Daily Idling Energy Consumption ---
            # Filter for idling periods using 'Gear_Position' == 0 (Neutral) and 'Vehiclereadycondition' == 1
            stationary_df = vehicle_df[(vehicle_df['Gear_Position'] == 0.0) & 
                                       (vehicle_df['Vehiclereadycondition'] == 1.0)].copy()

            if not stationary_df.empty:
                stationary_df['energy_kwh'] = stationary_df.apply(
                    lambda row: row['power_kW'] * (row['time_diff_seconds'] / 3600) if row['power_kW'] > 0 else 0, axis=1)
                grouped_by_day_stationary = stationary_df.groupby(['id', 'date'])
                daily_idling_energy = grouped_by_day_stationary['energy_kwh'].sum()

                daily_idling_df = daily_idling_energy.reset_index(name='idling_energy_kwh') #Daily Idling Energy Consumed (kWh)
                daily_idling_df = daily_idling_df.round({'idling_energy_kwh': 2})  

                # Merge with existing stats
                daily_stats_df = pd.merge(daily_stats_df, daily_idling_df, on=['id', 'date'], how='outer')
                daily_stats_df = daily_stats_df.fillna(0)
            else:
                # If there's only driving data and no idling data
                if not daily_stats_df.empty:
                    daily_stats_df['idling_energy_kwh'] = 0

            if not daily_stats_df.empty:
                #Net Energy Consumed (kWh)
                daily_stats_df['net_energy_kwh'] = daily_stats_df['energy_consumed_kwh'] - daily_stats_df['regen_energy_kwh']
                #Net Consumption Rate (kWh/km)
                daily_stats_df['mileage_kwh_per_km'] = (daily_stats_df['net_energy_kwh'] / daily_stats_df['dist_travelled_km']).round(2)
                all_daily_stats.append(daily_stats_df)

        if all_daily_stats:            
            final_df = pd.concat(all_daily_stats, ignore_index=True)
            return final_df
        else:
            print("No data found for driving or idling periods.")
            return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


# In[5]:


# --------------------
# Main execution logic
# --------------------
def main(start_date_str: str = None, end_date_str: str = None):
    conn = connect_to_trino()
    df_duplicate_processed = pd.DataFrame()
    df_duplicate_raw = pd.DataFrame()
    vehicle_ids_for_report = []    
    if conn:
        try:
            # Determine the date range to process
            if start_date_str and end_date_str:
                start_date = date.fromisoformat(start_date_str)
                end_date = date.fromisoformat(end_date_str)
                date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
            else:
                # Default to processing yesterday's data
                date_range = [date.today() - timedelta(days=1)]


            for single_date in date_range:
                date_str = single_date.isoformat()
                logging.info(f"▶️ Starting daily report job for {date_str}")

                # Example 2: Call the function with specific vehicle IDs
                logging.info("\n--- Processing specific vehicle IDs ---")
                # vehicle_ids_for_report = ['3', '16', '18', '19']
                df_raw_specific = fetch_data_for_day(conn, date_str, COLUMNS_TO_FETCH, SOURCE_TABLE, vehicle_ids_for_report)
                df_duplicate_raw = df_raw_specific.copy()

                if not df_raw_specific.empty:
                    df_processed_specific = analyze_vehicle_energy_stats(df_raw_specific)
                    df_duplicate_processed = df_processed_specific.copy()
                    if not df_processed_specific.empty:
                        # Updated function call with the missing 'conn' and 'schema' arguments
                        # write_df_to_iceberg(conn, df_processed_specific, TABLE_NAME, db_operations.COLUMN_SCHEMA_MILEAGE)
                        logging.info("✅ Processing and write for specific IDs complete.")
                    else:
                        logging.info("Processed DataFrame is empty. No data to write.")
                else:
                    logging.info("Raw DataFrame is empty. No processing needed.")

        except Exception as e:
            logging.critical(f"❌ A critical error occurred in the main script: {e}")

        finally:
            logging.info("🔒 STEP 5: Closing Trino connection...")
            conn.close()
            logging.info("✅ STEP 5: Connection closed.")
    else:
        logging.critical("❌ Failed to establish a database connection. Exiting.")

    return df_duplicate_raw, df_duplicate_processed


# In[6]:


if __name__ == "__main__":
    global_df_raw, global_df_processed = main()
    # --- For a one-time manual backfill, uncomment the line below and set your dates ---
    # main(start_date_str='2025-07-24', end_date_str='2025-09-15')

    # --- For daily automated runs, use the existing call ---
    # main()


# In[7]:


global_df_raw.head()


# In[8]:


global_df_raw.loc[:, 'date'] = global_df_raw['IST'].dt.date


# In[9]:


global_df_raw.groupby(['id', 'date']).size().reset_index(name='count_of_instances')


# In[10]:


global_df_raw.describe()


# In[11]:


global_df_processed

