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


import torch
import numpy as np


def loopH_loss(y_pred, y_true, posterior_var, delta_scale=1.0):
    """
    LOOPH loss: pseudo-Huber loss scaled by predictive variance.

    Args:
        y_pred: Tensor of predicted values (N,)
        y_true: Tensor of true values (N,)
        posterior_var: Tensor of predictive variances (N,)
        delta_scale: scalar multiplier for δ = k·σ (default = 1.0)

    Returns:
        Tensor of LOOPH losses per point (N,)
    """
    # Numerische Stabilität sicherstellen
    posterior_var = torch.clamp(posterior_var, min=1e-8)

    delta = delta_scale * posterior_var.sqrt()
    residual = y_true - y_pred

    denom = delta**2 * posterior_var
    core = torch.sqrt(1 + (residual**2) / denom) - 1
    term1 = 2 * delta**2 * core
    term2 = torch.log(posterior_var)

    return term1 + term2

import numpy as np
import pandas as pd
from itertools import combinations

def compute_pairwise_gradients(train_x_std, train_y, top_k=10, epsilon=1e-6):
    """
    Identifiziere Punktpaare mit kleinem Abstand, aber großem Funktionsgradienten.

    Args:
        train_x_std (Tensor or ndarray): standardisierte Inputs (N, D)
        train_y (Tensor or ndarray): Zielwerte (N,)
        top_k (int): Anzahl der Top-Paare mit höchstem Gradient
        epsilon (float): zur Vermeidung von Division durch 0

    Returns:
        DataFrame mit Punktindex i, j, Distanz, |Δy| und Gradient
    """
    if isinstance(train_x_std, torch.Tensor):
        x = train_x_std.cpu().numpy()
    else:
        x = train_x_std
    if isinstance(train_y, torch.Tensor):
        y = train_y.cpu().numpy().flatten()
    else:
        y = train_y.flatten()

    rows = []
    for i, j in combinations(range(len(x)), 2):
        dist = np.linalg.norm(x[i] - x[j])
        dy = abs(y[i] - y[j])
        grad = dy / (dist + epsilon)
        rows.append((i, j, dist, dy, grad))

    df_pairs = pd.DataFrame(rows, columns=["Index 1", "Index 2", "Distance", "|Δy|", "Gradient"])
    df_sorted = df_pairs.sort_values(by="Gradient", ascending=False).head(top_k)
    return df_sorted



def sample_random_model_config(seed=None, input_dim=2):
        if seed is not None:
            np.random.seed(seed)
    
        # Bevorzuge komplexere Kernel
        """
        Chemistry-optimized GP configuration for smooth, realistic functions
        Based on your existing sample_random_model_config but tuned for chemistry
        """
        if seed is not None:
            np.random.seed(seed)

            # Seed-abhängige Strategien für mehr Variation
        seed_mod = seed % 4 if seed is not None else 0
        
        if seed_mod == 0:  # Very smooth and broad - typical reaction kinetics
            config = {
                "covar_module": "RBF",
                "outputscale": float(np.random.uniform(0.8, 1.5)),  # Moderate amplitude
                "noise": float(np.random.uniform(0.005, 0.015)),    # Low noise
                "lengthscale": float(np.random.uniform(2.5, 4.0)), # Very smooth
            }
        elif seed_mod == 1:  # Smooth with gentle variations - thermodynamics
            config = {
                "covar_module": "RBF",
                "outputscale": float(np.random.uniform(1.0, 2.0)),  # Higher amplitude
                "noise": float(np.random.uniform(0.008, 0.02)),     # Low noise
                "lengthscale": float(np.random.uniform(1.8, 3.2)), # Smooth
            }
        elif seed_mod == 2:  # Moderate complexity - mass transfer
            config = {
                "covar_module": "Matern_2.5",  # Slightly less smooth than RBF
                "outputscale": float(np.random.uniform(0.6, 1.2)),  # Moderate
                "noise": float(np.random.uniform(0.01, 0.025)),     # Low noise
                "lengthscale": float(np.random.uniform(1.5, 2.8)), # Still quite smooth
            }
        else:  # Gentle gradients - typical for most chemical processes
            config = {
                "covar_module": "RBF",
                "outputscale": float(np.random.uniform(1.2, 2.2)),  # Good amplitude
                "noise": float(np.random.uniform(0.005, 0.018)),    # Very low noise
                "lengthscale": float(np.random.uniform(2.0, 3.5)), # Very smooth
            }
        
        return config
    # if seed is not None:
    #     np.random.seed(seed)

    # kernel_choice = np.random.choice(["RBF", "Matern_0.5", "Matern_1.5", "Matern_2.5"])

    # config = {
    #     "covar_module": kernel_choice,
    #     "outputscale": float(np.random.uniform(0.5, 3.0)),
    #     "noise": float(np.random.lognormal(mean=np.log(0.1), sigma=0.5)),
    # }

    # # Nur Linear-Kernel hat KEIN lengthscale
    # if not kernel_choice.startswith("Linear"):
    #     config["lengthscale"] = float(np.random.uniform(0.12, 0.35))
    # else:
    #     config["lengthscale"] = None

    # return config

    