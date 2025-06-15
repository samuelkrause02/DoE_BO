import pandas as pd
import numpy as np
import torch

def load_csv_experiment_data(
    data_or_file,
    columns_config,
    loi=True,
    input_columns=None,
    output_column=None
):
    """
    Load experiment data from DataFrame or CSV file.
    
    Args:
        data_or_file: pandas DataFrame OR path to CSV file
        columns_config: Dictionary mapping target keys to column names
        loi: Whether to use LOI-based yield (affects auto-detection of output column)
        input_columns: List of column names to use as inputs (None for auto-detection)
        output_column: Name of output column (None for auto-detection)
    
    Returns:
        tuple: (inputs_array, output_array, dataframe)
    """
    # Check if input is DataFrame or file path
    if isinstance(data_or_file, pd.DataFrame):
        print(f"[DEBUG] Processing existing DataFrame with shape: {data_or_file.shape}")
        df = data_or_file.copy()
    else:
        print(f"Reading experiment data from: {data_or_file}")
        df = pd.read_csv(data_or_file, sep=';')
    
    print(f"[DEBUG] Columns: {list(df.columns)}")
    
    # Auto-detect output column if needed
    if output_column is None:
        target_key = 'yield_loi' if loi else 'yield_mass'
        output_column = columns_config[target_key]
        print(f"[DEBUG] Auto-detected output column: {output_column}")

    if output_column not in df.columns:
        raise ValueError(f"Output column '{output_column}' not found in data.")

    # Drop NaN rows for output column
    print(f"[DEBUG] Dropping NaN rows for output column '{output_column}'")
    df = df.dropna(subset=[output_column])

    # If no input columns are specified: use numeric columns except output + meta-data
    if input_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if output_column in numeric_cols:
            numeric_cols.remove(output_column)
        input_columns = numeric_cols

    print(f"[DEBUG] Using input columns: {list(input_columns)}")
    print(f"[DEBUG] Using output column: {output_column}")

    # Extract data
    inputs = df[input_columns].astype(float).values
    output = df[output_column].astype(float).values

    print(f"Loaded {inputs.shape[0]} experiments with {inputs.shape[1]} parameters.")
    return inputs, output, df

def load_bounds_csv(csv_file):
    """
    Load parameter bounds from a CSV file.
    The CSV is expected to have two columns: min and max per row.
    """
    bounds_df = pd.read_csv(csv_file, sep=';')  # match separator
    bounds = bounds_df.values.tolist()

    print("Loaded parameter bounds:")
    for i, b in enumerate(bounds):
        print(f"Param {i+1}: [{b[0]}, {b[1]}]")
    return bounds

def scale_inputs(raw_inputs, bounds):
    """
    Scale raw input data to the [0, 1] range using the provided bounds.
    """
    bounds_arr = np.array(bounds, dtype=float)
    mins = bounds_arr[:, 0].reshape(1, -1)
    maxs = bounds_arr[:, 1].reshape(1, -1)
        
    if raw_inputs.shape[1] != bounds_arr.shape[0]:
        raise ValueError(f"Mismatch: {raw_inputs.shape[1]} input features vs {bounds_arr.shape[0]} bounds.")
        
    scaled = (raw_inputs - mins) / (maxs - mins)
    return scaled

def prepare_training_data(raw_inputs, raw_outputs, bounds):
    """
    Scale input data and prepare output for training.
    """
    scaled_x = scale_inputs(raw_inputs, bounds)
    scaled_y = raw_outputs
    return scaled_x, scaled_y

def prepare_training_tensors(scaled_x, scaled_y):
    """
    Convert scaled data to PyTorch tensors.
    """
    train_x = torch.from_numpy(scaled_x).double()
    train_y = torch.from_numpy(scaled_y).double().unsqueeze(-1)
    print(f"Training data shapes: X={train_x.shape}, Y={train_y.shape}")
    return train_x, train_y