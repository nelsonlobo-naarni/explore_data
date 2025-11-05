# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
import os

# %% [markdown]
# ## Notes
# 
# - Device_ID
# - 
# - 

# %%
def analyze_filtered_braking_events(file_paths):
    """
    Analyzes and plots braking events from multiple files, filtering
    based on a user-defined top speed reached during the event.
    The logic is fine-tuned to extract the exact start point of braking
    (BrakePedalPos > 0.0) within a 20-second window before a hard stop.
    """
    # 1. Get user input for filtering and plotting
    while True:
        try:
            top_speed = float(input("Enter the top speed (km/h) to filter events by: "))
            # We still need a search window, but the analysis will be dynamic.
            search_window_seconds = 20.0
            break
        except ValueError:
            print("Error: Invalid input. Please enter a numerical value for top speed.")

    all_event_data = []
    # This list will store the dynamically calculated duration for each event.
    event_durations = []

    # 2. Iterate through each file provided in the list
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            print(f"Processing file: {file_path}")
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Skipping.")
            continue

        # Convert timestamp to human-readable IST and clean the data
        df['IST'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df = df[["IST", "BrakePedalPos", "Vehicle_speed_VCU"]].copy()
        df.dropna(subset=["Vehicle_speed_VCU", "BrakePedalPos"], inplace=True)
        df.sort_values(by='IST', inplace=True)
        df.loc[:, 'IST_formatted_string'] = df['IST'].dt.strftime('%H:%M:%S')

        # Identify hard stop events (speed becomes 0 from a non-zero value)
        hard_stop_mask = (df['Vehicle_speed_VCU'] == 0.0) & (df['Vehicle_speed_VCU'].shift(1) > 0.0)
        hard_stop_events = df[hard_stop_mask].copy()

        if hard_stop_events.empty:
            print(f"No hard stop events found in '{file_path}'.")
            continue
        
        # 3. Extract and aggregate the data for each filtered event
        for _, event_row in hard_stop_events.iterrows():
            end_time = event_row['IST']
            # Define a search window of 20 seconds before the stop
            search_start_time = end_time - pd.Timedelta(seconds=search_window_seconds)

            # Look for the first instance of brake pedal application within this window
            search_segment = df[(df['IST'] >= search_start_time) & (df['IST'] <= end_time)].copy()

            # Find the first row where BrakePedalPos is greater than 0
            first_brake_press = search_segment[search_segment['BrakePedalPos'] > 0.0].head(1)

            # Check if a brake press was found in the search window
            if not first_brake_press.empty:
                # Get the exact start time of the braking event
                start_time = first_brake_press.iloc[0]['IST']

                # Filter the event segment from the exact start of braking to the stop
                event_segment = df[(df['IST'] >= start_time) & (df['IST'] <= end_time)].copy()

                # Check if the top speed in this dynamic segment meets the filter criteria
                if not event_segment.empty and event_segment['Vehicle_speed_VCU'].max() >= top_speed:
                    # Calculate the dynamic time taken to come to a full stop
                    time_to_stop_seconds = (end_time - start_time).total_seconds()
                    
                    # Append the filtered event data and its duration
                    all_event_data.append(event_segment)
                    event_durations.append(time_to_stop_seconds)

    if not all_event_data:
        print(f"No events found across all files that reached a top speed of {top_speed} km/h or greater and had a brake press within {search_window_seconds} seconds of the stop.")
        return

    # 4. Generate combined reports
    # Determine output filenames based on the first file processed
    if file_paths:
        try:
            l1 = file_paths[0].split('/')[0]
            l2 = file_paths[0].split('/')[1].split('_')[0]
            base_name = os.path.join(l1, 'brakingAnalysis', l2)
        except IndexError:
            # Fallback for unexpected file path format
            base_name = "combined_report"
    else:
        base_name = "combined_report"
    
    output_csv_filename = f"{base_name}_combined_report.csv"
    output_pdf_filename = f"{base_name}_combined_report.pdf"
    
    # Create the directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(output_csv_filename), exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
        # If directory creation fails, fall back to a local filename
        output_csv_filename = os.path.basename(output_csv_filename)
        output_pdf_filename = os.path.basename(output_pdf_filename)
    
    generate_report_csv(all_event_data, output_csv_filename)
    # Pass the list of dynamic event durations to the PDF report function
    generate_report_pdf(all_event_data, event_durations, output_pdf_filename)    
    generate_final_report(output_csv_filename)
    plt.show()



# %%
def generate_report_csv(events, output_filename):
    """
    Generates a CSV report with a summary of braking events.
    """
    # Define constants for the kgf calculation
    BUS_MASS_KG = 13500  # 13.5 tonnes * 1000 kg/tonne
    G_ACCELERATION = 9.80665 # Standard acceleration due to gravity

    table_data = []
    
    for i, event_group in enumerate(events):
        start_time = event_group['IST'].iloc[0]
        end_time = event_group['IST'].iloc[-1]
        start_velocity = event_group['Vehicle_speed_VCU'].iloc[0]
        peak_velocity = event_group['Vehicle_speed_VCU'].max()
        max_brake_pedal_pos = event_group['BrakePedalPos'].max()
        min_brake_pedal_pos = event_group['BrakePedalPos'].min()
        avg_brake_pedal_pos = event_group['BrakePedalPos'].mean()
        
        event_group.loc[:, 'speed_mps'] = event_group['Vehicle_speed_VCU'] * (1000 / 3600)
        time_diffs_sec = event_group['IST'].diff().dt.total_seconds().fillna(0)
        distance_covered_m = (event_group['speed_mps'] * time_diffs_sec).sum()
        total_time_s = (end_time - start_time).total_seconds()
        
        if total_time_s > 0:
            # Calculate average deceleration in m/s^2
            avg_deceleration = (peak_velocity * 1000/3600) / total_time_s
        else:
            avg_deceleration = 0
            
        # Calculate braking force in kgf
        braking_force_kgf = (BUS_MASS_KG * avg_deceleration) / G_ACCELERATION
        
        table_data.append({
            'idx': i + 1,
            'start': start_time.strftime('%d/%m/%y %H:%M:%S'),
            'end': end_time.strftime('%d/%m/%y %H:%M:%S'),
            'max_bpp': f"{max_brake_pedal_pos:.2f}",
            # 'min_bpp': f"{min_brake_pedal_pos:.2f}",
            'avg_bpp': f"{avg_brake_pedal_pos:.2f}",
            'ttl_dist_m': f"{distance_covered_m:.2f}",
            'start_vel': f"{start_velocity:.2f}",
            'peak_vel': f"{peak_velocity:.2f}",
            'avg_decel_mps2': f"{avg_deceleration:.2f}",
            'braking_force_kgf': f"{braking_force_kgf:.2f}" # New column
        })

    results_df = pd.DataFrame(table_data)
    try:
        print(output_filename)
        results_df.to_csv(output_filename, index=False)        
    except (FileNotFoundError,OSError):
        # This block is executed if the directory does not exist.
        print(f"Directory for '{output_filename}' not found. Creating it now...")
    
        # Extract the directory path from the full filename
        directory = os.path.dirname(output_filename)

        os.makedirs(directory, exist_ok=True)

        # Now that the directory exists, try saving the file again.
        results_df.to_csv(output_filename, index=False)
        print(f"Directory created and file saved successfully to '{output_filename}'.")
        
    except Exception as e:
        # A generic exception handler for any other potential errors
        print(f"An unexpected error occurred: {e}")        
        
    print(f"\nCombined CSV report saved as '{output_filename}'.")
    return results_df

# %%
def generate_report_pdf(events, durations, output_filename):
    """
    Generates a multi-page PDF report with a summary page and individual plots.
    This version uses the dynamic braking duration for each plot's title.
    """
    # Define constants for the kgf calculation
    BUS_MASS_KG = 13500  # 13.5 tonnes * 1000 kg/tonne
    G_ACCELERATION = 9.80665 # Standard acceleration due to gravity

    with PdfPages(output_filename) as pdf:
        all_peak_speeds = []
        all_distances = []
        all_max_bpps = [] # List to store maximum bpp for each event
        all_avg_bpps = [] # List to store average bpp for each event
        
        for i, event_group in enumerate(events):
            start_time = event_group['IST'].iloc[0]
            end_time = event_group['IST'].iloc[-1]
            peak_velocity = event_group['Vehicle_speed_VCU'].max()
            
            event_group.loc[:, 'speed_mps'] = event_group['Vehicle_speed_VCU'] * (1000 / 3600)
            time_diffs_sec = event_group['IST'].diff().dt.total_seconds().fillna(0)
            distance_covered_m = (event_group['speed_mps'] * time_diffs_sec).sum()
            
            # Calculate and store BPP values for the summary
            max_bpp = event_group['BrakePedalPos'].max()
            avg_bpp = event_group['BrakePedalPos'].mean()
            
            all_peak_speeds.append(peak_velocity)
            all_distances.append(distance_covered_m)
            all_max_bpps.append(max_bpp)
            all_avg_bpps.append(avg_bpp)

        # Create and save the summary page
        fig_summary = plt.figure(figsize=(11, 8.5))
        ax_summary = fig_summary.add_subplot(111)
        ax_summary.axis('off')
        
        # Create a table for the summary data
        summary_data = [
            ['Total events found:', f"{len(events)}"],
            ['Max speed across all events:', f"{max(all_peak_speeds):.2f} km/h"],
            ['Average speed across all events:', f"{sum(all_peak_speeds)/len(all_peak_speeds):.2f} km/h"],
            
            ['Maximum distance covered:', f"{max(all_distances):.1f} m"],
            ['Minimum distance covered:', f"{max(all_distances):.1f} m"],
            ['Average distance covered:', f"{sum(all_distances)/len(all_distances):.1f} m"],
            
            ['Maximum duration:', f"{max(durations):.1f} s"],
            ['Minimum duration:', f"{min(durations):.1f} s"],
            ['Average duration:', f"{sum(durations)/len(durations):.1f} s"],
            
            ['Maximum BPP:', f"{max(all_max_bpps):.1f}"],
            ['Average BPP:', f"{sum(all_avg_bpps)/len(all_avg_bpps):.1f}"]
        ]
        
        # Define a title for the table
        plt.suptitle("Braking Analysis Report", fontsize=18, y=0.95)
        
        # Create the table
        summary_table = ax_summary.table(
            cellText=summary_data,
            loc='center',
            cellLoc='left',
            colWidths=[0.5, 0.5]
        )
        
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1.2, 1.5)
        
        pdf.savefig(fig_summary)
        plt.close(fig_summary)
        
        # --- Paginate the detailed event table ---
        
        table_data = []
        for i, event_group in enumerate(events):
            start_time = event_group['IST'].iloc[0]
            end_time = event_group['IST'].iloc[-1]
            max_bpp = event_group['BrakePedalPos'].max()
            avg_bpp = event_group['BrakePedalPos'].mean()
            dist_m = (event_group['Vehicle_speed_VCU'] * (1000/3600) * event_group['IST'].diff().dt.total_seconds().fillna(0)).sum()
            start_vel = event_group['Vehicle_speed_VCU'].iloc[0]
            peak_vel = event_group['Vehicle_speed_VCU'].max()
            total_time_s = (end_time - start_time).total_seconds()
            avg_decel = (peak_vel * 1000/3600) / total_time_s if total_time_s > 0 else 0
            
            braking_force_kgf = (BUS_MASS_KG * avg_decel) / G_ACCELERATION

            table_data.append([
                i + 1,
                start_time.strftime('%d/%m/%y %H:%M:%S'),
                end_time.strftime('%d/%m/%y %H:%M:%S'),
                f"{durations[i]:.2f}",
                f"{max_bpp:.2f}",
                f"{avg_bpp:.2f}",
                f"{dist_m:.2f}",
                f"{start_vel:.2f}",
                f"{avg_decel:.2f}",
                f"{braking_force_kgf:.2f}"
            ])

        # Define the number of rows per page
        ROWS_PER_PAGE = 23
        
        # Split the data into chunks for pagination
        chunks = [table_data[i:i + ROWS_PER_PAGE] for i in range(0, len(table_data), ROWS_PER_PAGE)]
        
        # Define columns for the table
        columns = [
            'idx', 'start', 'end', 'duration_s', 'max_bpp', 'avg_bpp', 
            'ttl_dist_m', 'start_vel', 'avg_decel_mps2', 'braking_force_kgf'
        ]

        # Loop through each chunk of data and create a new page
        for page_num, chunk in enumerate(chunks):
            fig_table = plt.figure(figsize=(11, 8.5))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')
            
            # Set relative column widths
            col_widths = [0.05, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.15]
            
            table = ax_table.table(cellText=chunk, colLabels=columns, loc='center', cellLoc='center', colWidths=col_widths)
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            plt.title(f"DETAILED BRAKING EVENT TABLE (Page {page_num + 1})", y=0.95)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_table)
            plt.close(fig_table)
            
        # Now, plot and save each individual graph on its own page
        for i, event_group in enumerate(events):
            fig, ax = plt.subplots(figsize=(15, 6))
            
            event_group.loc[:, 'IST_formatted_string'] = event_group['IST'].dt.strftime('%H:%M:%S')

            distance_covered_m = (event_group['Vehicle_speed_VCU'] * (1000 / 3600) * event_group['IST'].diff().dt.total_seconds().fillna(0)).sum()
            total_distance_ft = distance_covered_m * 3.28084
            distance_label = (
                f'Distance Covered:\n'
                f'{distance_covered_m:.2f} m\n'
                f'{total_distance_ft:.2f} ft'
            )

            ax.text(
                0.95, 0.95,
                distance_label,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
            )

            ax.plot(
                event_group['IST_formatted_string'],
                event_group['Vehicle_speed_VCU'],
                label='Vehicle Speed (km/h)',
                color='blue'
            )
            ax.plot(
                event_group['IST_formatted_string'],
                event_group['BrakePedalPos'],
                label='Brake Pedal Position',
                color='red'
            )
            
            start_time_str = event_group['IST'].iloc[0].strftime('%d/%m/%y %H:%M:%S')
            end_time_str = event_group['IST'].iloc[-1].strftime('%d/%m/%y %H:%M:%S')
            ax.set_title(f"Event: {start_time_str} to {end_time_str}")
            ax.set_xlabel('Time (hh:mm:ss)')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            
            ax.tick_params(axis='x', rotation=45)
            
            # Use the duration from the list to create the dynamic title
            plt.suptitle(f"Analysis of Braking Events ({durations[i]:.2f}s to stop)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            pdf.savefig(fig)
            plt.close(fig)
            
        print(f"\nCombined PDF report saved as '{output_filename}'.")


# %%
def generate_final_report(file_name):
    """
    Analyzes and compares braking performance data from a CSV file
    for two distinct time periods. It generates a summary report and a
    visual comparison chart, then combines them into a single PDF document.
    """
    # Check if the file exists in the current directory
    if not os.path.exists(file_name):
        print(f"Error: The file '{file_name}' was not found in the current directory.")
        print("Please ensure the CSV file is saved in the same folder as this script.")
        return

    try:
        # Read the CSV file directly from the local file path
        print(f"Reading data from '{file_name}'...")
        df = pd.read_csv(file_name)
        print("File read successfully.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- Data Cleaning and Preprocessing ---
    # Clean column names to handle any leading/trailing spaces or newlines
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Now, check the cleaned column names to ensure they exist before proceeding
    required_cols = ['avg_decel_mps2', 'avg_bpp', 'ttl_dist_m', 'start_vel', 'peak_vel', 'start', 'max_bpp']
    
    # Check if all required columns are present after cleaning
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print("Error: The CSV file is missing one or more required columns after cleaning.")
        print(f"Missing columns: {missing_cols}")
        print("\nAvailable columns are:")
        print(df.columns)
        return

    # Convert the 'start' column to datetime objects
    try:
        df['start_datetime'] = pd.to_datetime(df['start'], format='%d/%m/%y %H:%M:%S')
    except Exception as e:
        print(f"Error converting dates: {e}. Please check the date format in your CSV file.")
        return

    # Convert the key metrics columns to float
    numeric_cols = ['avg_decel_mps2', 'avg_bpp', 'ttl_dist_m', 'start_vel', 'peak_vel', 'max_bpp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values in critical columns
    df.dropna(subset=numeric_cols, inplace=True)

    # Define the cutoff date to split the data
    cutoff_date = pd.to_datetime('17/08/25', format='%d/%m/%y')

    # Partition the data into 'before' and 'after' the cutoff date
    df_then = df[df['start_datetime'] < cutoff_date]
    df_now = df[df['start_datetime'] >= cutoff_date]

    # Define the key metrics for analysis
    metrics_avg = {
        'avg_decel_mps2': 'Avg Deceleration ($m/s^2$)',
        'avg_bpp': 'Avg Brake Pedal Position (%)',
        'ttl_dist_m': 'Avg Distance (m)',
        'start_vel': 'Avg Start Velocity (km/h)', # Renamed to reflect the start of the event
        'peak_vel': 'Avg Peak Velocity (km/h)'
    }
    
    # Calculate summary statistics for both periods
    summary_then_avg = df_then[metrics_avg.keys()].mean()
    summary_now_avg = df_now[metrics_avg.keys()].mean()

    # Create a DataFrame for the average metrics comparison
    comparison_df_avg = pd.DataFrame({
        'Then (Aug 1 - Aug 16)': summary_then_avg,
        'Now (Aug 17 - Aug 25)': summary_now_avg
    })
    
    # Rename the index to the more descriptive names
    comparison_df_avg = comparison_df_avg.rename(index=metrics_avg)

    # Calculate min distance and max brake pedal position
    min_dist_then = df_then['ttl_dist_m'].min()
    min_dist_now = df_now['ttl_dist_m'].min()
    
    max_bpp_then = df_then['max_bpp'].max()
    max_bpp_now = df_now['max_bpp'].max()
    
    # Create a new DataFrame for these specific metrics and concatenate
    specific_metrics_df = pd.DataFrame({
        'Then (Aug 1 - Aug 16)': [min_dist_then, max_bpp_then],
        'Now (Aug 17 - Aug 25)': [min_dist_now, max_bpp_now]
    }, index=['Min Distance (m)', 'Max Brake Pedal Position (%)'])

    # Combine the average and specific metrics DataFrames and round the results
    comparison_df = pd.concat([comparison_df_avg, specific_metrics_df])

    # Re-order the rows as requested
    new_order = [
        'Avg Peak Velocity (km/h)',
        'Avg Distance (m)',
        # 'Min Distance (m)',
        'Avg Brake Pedal Position (%)',
        'Max Brake Pedal Position (%)',
        'Avg Deceleration ($m/s^2$)'
    ]
    comparison_df = comparison_df.reindex(new_order).round(2)
    
    # Add a row for total events
    comparison_df.loc['Total Events'] = [len(df_then), len(df_now)]
    
    # --- Generate the comparison chart (PNG) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Exclude 'Total Events' from the chart
    plot_df = comparison_df.drop('Total Events')

    plot_df.T.plot(kind='bar', ax=ax, width=0.8, rot=0)

    # Add labels and title
    ax.set_title('Braking Performance: Before vs. After August 17, 2025', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Customize x-axis labels
    plt.xticks(ha='center')

    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    chart_filename = file_name.split('_')[0] + "_braking_comparison_chart.png"
    plt.savefig(chart_filename)
    plt.close(fig)
    print(f"\nChart successfully saved as '{chart_filename}'.")

    # --- Generate the PDF Report ---
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Add a title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Braking Performance Comparison Report", 0, 1, 'C')

    # Add the summary table
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "Summary of Braking Metrics:", 0, 1)

    # Convert the DataFrame to a string with a fixed width for the table
    table_str = comparison_df.to_string()
    pdf.set_font("Courier", '', 10) # Using a monospace font for table formatting
    pdf.multi_cell(0, 5, table_str, 0, 1)
    
    # Add a title for the chart
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Visual Comparison", 0, 1, 'C')
    
    # Add the generated chart image
    # Note: Adjust the x, y, width, and height as needed to fit the page.
    pdf.image(chart_filename, x=15, y=pdf.get_y() + 5, w=180)

    pdf_filename = file_name.split('_')[0] + "_Braking_Analysis_Report.pdf"    
    pdf.output(pdf_filename)
    
    print(f"\nPDF report successfully generated as '{pdf_filename}'.")

# %%
file_list = [
# "AP39WF8584/8584_01250825.csv",    
# "AP39WF8589/8589_01250825.csv",
# "AP39WF8593/8593_01250825.csv",   
# "AP39WG0252/0252_01250825.csv",
# "AP39WG0271/0271_01250825.csv",
# "AP39WG4628/4268_01250825.csv",    
"AP39WG4630/4630_01250825.csv"
]

if __name__ == "__main__":
    # Example usage with multiple files
    # Replace these with the actual file paths on your system
    file_list
    analyze_filtered_braking_events(file_list)


