import pandas as pd
import datetime
import os

def append_candidates_to_csv(
    candidates_list,
    csv_file,
    max_objective,
    n_existing_points=None,
    loi=True,
    log_file=None
):
    """
    Append new candidates to the experiments CSV file,
    and optionally log meta info to a log file.
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # Create experiment data rows (only X1, X2, ..., Yield)
    data_rows = []
    for candidate in candidates_list:
        row = {}
        for idx, val in enumerate(candidate):
            row[f"X{idx+1}"] = round(val, 4)
        data_rows.append(row)
    
    # Save experiment data to CSV
    df_data = pd.DataFrame(data_rows)
    df_data.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False, sep=';')
    print(f"Appended {len(data_rows)} candidates to {csv_file}")

    # Optionally log meta data
    if log_file is not None:
        meta_rows = []
        for _ in candidates_list:
            meta_row = {
                'Date': today_str,
                'Max Objective': round(max_objective, 6),
                'N Existing Points': n_existing_points if n_existing_points is not None else '-',
                'Method': 'LOI' if loi else 'Mass'
            }
            meta_rows.append(meta_row)
        
        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False, sep=';')
        print(f"Meta info logged to: {log_file}")

def save_candidates_to_new_csv(
    candidates_list,
    max_objective,
    loi=True,
    base_name="candidates",
    log_file=None
):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"{base_name}_{today_str}.csv"

    # Create experiment data rows
    data_rows = []
    for candidate in candidates_list:
        row = {}
        for idx, val in enumerate(candidate):
            row[f"X{idx+1}"] = round(val, 4)
        data_rows.append(row)

    df_data = pd.DataFrame(data_rows)
    df_data.to_csv(filename, index=False, sep=';')
    print(f"Saved candidates to {filename}")

    # Optionally log meta data
    if log_file is not None:
        meta_rows = []
        for _ in candidates_list:
            meta_row = {
                'Date': today_str,
                'Max Objective': round(max_objective, 6),
                'Method': 'LOI' if loi else 'Mass',
                'Comment': 'Initial save of new candidates'
            }
            meta_rows.append(meta_row)
        
        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False, sep=';')
        print(f"Meta info logged to: {log_file}")