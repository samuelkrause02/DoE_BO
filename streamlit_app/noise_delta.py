    
import streamlit as st
import numpy as np
import pandas as pd

from bo_utils.bo_model import build_gp_model, fit_model
def show_noise_delta(session_state, model_config):
    """
    Show the noise delta when removing a point from the training set.
    """
    y_std = session_state["model"].outcome_transform.stdvs.item()
    ref_noise = session_state["model"].likelihood.noise.item() * (y_std ** 2)
    print(f"Reference Noise (σ²): {ref_noise:.6f} / σ: {ref_noise ** 0.5:.4%}")

    noise_deltas = []
    for i in range(session_state["train_x"].shape[0]):
        mask = np.ones(session_state["train_x"].shape[0], dtype=bool)
        mask[i] = False

        x_loo = session_state["train_x"][mask]
        y_loo = session_state["train_y"][mask]
        input_dim = x_loo.shape[1]
        model_loo, _ = build_gp_model(x_loo, y_loo, model_config)
        fit_model(model_loo)
        y_std = model_loo.outcome_transform.stdvs.item()
        noise_i = model_loo.likelihood.noise.item() * (y_std ** 2)
        delta = (noise_i ** 0.5  - ref_noise ** 0.5)*100
        noise_deltas.append(delta)

    # Optional: als Tabelle anzeigen
    df_noise_delta = pd.DataFrame({
        "Index": list(range(len(noise_deltas))),
        "Noise Δσ² (removed)": (noise_deltas)
    })

    # --- Line Chart zur Visualisierung ---
    st.line_chart(pd.DataFrame({
        "Noise Δσ²": noise_deltas
    }))