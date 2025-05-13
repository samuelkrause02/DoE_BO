import pandas as pd
from config import COLUMNS_MAP

import pandas as pd
from config import LOI  # direkt aus Config einlesen, damit es automatisch funktioniert

def export_excel_to_csv(
    excel_file,
    sheet_name,
    csv_file,
    columns_config,
    log_file=None
):
    """
    Export experiment data from Excel to CSV.

    Parameters:
    - excel_file: path to Excel file
    - sheet_name: name of the sheet in Excel
    - csv_file: output CSV file
    - columns_config: dict mapping keys (e.g., 'temperature') to Excel column names
    - log_file: optional log file to save meta information
    """
    print(f"Exporting data from {excel_file} [{sheet_name}] to {csv_file}...")

    # Decide which target key to use based on LOI from config
    target_key = 'yield_loi' if LOI else 'yield_mass'
    print(f"[DEBUG] Using target: {target_key}")

    # Automatically determine all input keys (all keys except yield_loi/yield_mass)
    input_keys = [key for key in columns_config.keys() if key not in ['yield_loi', 'yield_mass']]

    # Combine input + target
    cols_keys = input_keys + [target_key]

    try:
        cols = [columns_config[key] for key in cols_keys]
    except KeyError as e:
        raise ValueError(f"Missing expected column mapping for key: {e}")

    # Read the Excel sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Select and clean data
    df_export = df[cols].dropna(subset=[columns_config[target_key]])

    # Use original names from columns_config
    new_column_names = [columns_config[key] for key in input_keys] + [columns_config[target_key]]
    df_export.columns = new_column_names
    

    # Export experiment data
    df_export.to_csv(csv_file, index=False, sep=';')
    print(f"Export complete! Rows: {len(df_export)}")

    # Optionally write meta info to a log file
    if log_file is not None:
        meta_info = {
            'Date': pd.Timestamp.today().strftime("%Y-%m-%d"),
            'Experiments with results': len(df_export),
            'Comment': f"Initial export from Excel using target: {target_key}"
        }
        meta_df = pd.DataFrame([meta_info])
        meta_df.to_csv(log_file, index=False, sep=';', mode='a', header=not pd.io.common.file_exists(log_file))
        print(f"Meta info logged to: {log_file}")


import pandas as pd

def export_bounds_from_config(bounds, output_csv='bounds.csv'):
    """
    Save the bounds from a config (list of [min, max]) to a CSV file.

    Parameters:
    - bounds: list of [min, max] per parameter.
    - output_csv: the filename to save the CSV as.
    """
    bounds_df = pd.DataFrame(bounds, columns=['Min', 'Max'])
    bounds_df.to_csv(output_csv, index=False, sep=';')
    print(f"Bounds successfully exported to: {output_csv}")