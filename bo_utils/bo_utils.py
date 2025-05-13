import numpy as np
import torch

def rescale_single_point(x_scaled, bounds_vals):
    """
    Rescale a single scaled point back to original bounds.
    x_scaled: numpy array of shape (input_dim,)
    bounds_vals: list of [min, max] pairs
    """
    mins = np.array([b[0] for b in bounds_vals])
    maxs = np.array([b[1] for b in bounds_vals])
    return x_scaled * (maxs - mins) + mins


def rescale_batch(points_scaled, bounds_vals):
    """
    Rescale a batch of scaled points (torch Tensor) back to original bounds.
    points_scaled: torch Tensor of shape (batch_size, input_dim)
    bounds_vals: list of [min, max] pairs
    """
    arr = points_scaled.detach().cpu().numpy()
    real = []
    for row in arr:
        rescaled = [
            row[j] * (bounds_vals[j][1] - bounds_vals[j][0]) + bounds_vals[j][0]
            for j in range(len(row))
        ]
        real.append(rescaled)
    return np.array(real)



# experiment_utils.py

# experiment_utils.py

# experiment_utils.py

import pandas as pd
import streamlit as st
import io
import torch
import datetime

def append_and_display_experiment(new_x_tensor, columns_config, existing_df, original_filename="updated_experiment.csv", show_full_table=False):
    """
    Appends new experiment data to the given DataFrame and displays it in Streamlit.

    Args:
        new_x_tensor (torch.Tensor or np.ndarray): The new experiment points.
        columns_config (dict): Should include the key 'input_columns' for column names.
        existing_df (pd.DataFrame): The existing experimental data.
        original_filename (str): Suggested filename for the download button.
        show_full_table (bool): If True, displays the full updated experiment table.

    Returns:
        pd.DataFrame: The updated DataFrame after appending the new experiments.
    """
    # 
    #column_names = columns_config.get("input_columns", [f"x{i+1}" for i in range(new_x_tensor.shape[1])])
    column_names = existing_df.columns.tolist()
    
    # Convert to NumPy if it's a tensor
    if isinstance(new_x_tensor, torch.Tensor):
        data = new_x_tensor.detach().numpy()
    else:
        data = new_x_tensor  # assume it's already a NumPy array
    # last column is added and should be NaN
    data = np.hstack((data, np.full((data.shape[0], 1), np.nan)))  # add a column of NaN
    new_x_df = pd.DataFrame(data, columns=column_names)

    st.subheader("Next Suggested Experiment(s):")
    st.dataframe(new_x_df)

    # Append new experiment data to existing DataFrame
    # no new columns are added, so we can just concatenate
    
    updated_df = pd.concat([existing_df, new_x_df], ignore_index=True)

    st.success("Appended new experiment(s) to your uploaded data.")

    if show_full_table:
        st.subheader("Full Experiment Table")
        st.dataframe(updated_df)

    csv_buffer = io.StringIO()
    updated_df.to_csv(csv_buffer, index=False, sep=";")
    csv_string = csv_buffer.getvalue()

    st.download_button(
        label="Download Updated Experiment CSV",
        data=csv_string,  # ✅ Das ist ein string – encodebar
        #timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #file_name=original_filename.name,
        file_name=f"updated_experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        #file_name=original_filename.name,
        mime="text/csv"
    )

    return updated_df