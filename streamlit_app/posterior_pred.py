   

import streamlit as st
import torch
import pandas as pd
from bo_utils.bo_optimization import show_best_posterior_mean 
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_optimization import optimize_posterior_mean
import numpy as np
def display_posterior_predictions(session_state):
 # Annahme: dein Modell und deine Trainingsdaten sind bereits geladen
    model = st.session_state["model"].eval()
    train_x = st.session_state["train_x"]
    train_y = st.session_state["train_y"]

    # Posterior für alle trainierten Punkte berechnen
    posterior = model.posterior(train_x,observation_noise=False)
    mu = posterior.mean.detach().cpu().numpy().flatten()
    var = posterior.variance.detach().cpu().numpy().flatten()
    y_true = train_y.detach().cpu().numpy().flatten()
    residuals = mu - y_true
    std_residuals = residuals / (var ** 0.5)
    # für die letzen 12 Versuche var erhöhen
    # var: Posterior-Varianz (z. B. von GP), sollte gleiche Länge haben wie mu
    adjusted_var = var.copy()

    # Unsicherheit der letzten 12 Punkte erhöhen (Approximationsunsicherheit berücksichtigen)
    if len(adjusted_var) >= 12:
        adjusted_var[-12:] *= 9  # z. B. verdoppeln, wenn keine bessere Schätzung vorhanden

    # rel_noise_approx = 0.05  # 10 % relative Unsicherheit für approximierte CVs
    # adjusted_var[-12:] += (rel_noise_approx * y_true[-12:])**2

    # Standardisierte Residuen berechnen
    std_residuals_new = residuals / np.sqrt(adjusted_var)

    # In DataFrame umwandeln
    df_post = pd.DataFrame({
        "Index": range(len(y_true)),
        "Posterior Mean": mu,
        "Posterior Std": var** 0.5,
        "True Value": y_true,
        "Std Residuals": std_residuals
    })
    st.subheader("Standardized residuals over the index")
    st.line_chart(df_post.set_index("Index")[["Std Residuals"]], use_container_width=True)
    #mit std residuals new
    df_post["Std Residuals New"] = std_residuals_new
    st.subheader("Standardized residuals over the index (new)")
    st.line_chart(df_post.set_index("Index")[["Std Residuals New"]], use_container_width=True)
    st.line_chart
    st.subheader("Posterior Predictions on Training Data")
    st.dataframe(df_post)



def retrospective_best_posterior_evolution(train_x, train_y, model_config, bounds, limit=-2):
    """
    Train GP iteratively and track best posterior mean over time.
    """
    bounds_tensor = torch.tensor(bounds, dtype=torch.double).T
    best_values = []
    best_points = []
    best_point_std = []

    for i in range(2, len(train_x)+1):  # Start from 2 to allow model fitting
        x_subset = train_x[:i]
        y_subset = train_y[:i]

        model, _ = build_gp_model(x_subset, y_subset, config=model_config)
        fit_model(model)
        best_x, best_y = show_best_posterior_mean(
            model,
            input_dim=train_x.shape[1],
            bounds=bounds,
            limit=limit,
        )
        best_values.append(best_y)
        best_points.append(best_x.detach().cpu().numpy().flatten())
        best_point_std.append(model.posterior(best_x).variance.sqrt().detach().cpu().numpy().flatten())

    # In DataFrame packen
    from botorch.sampling import SobolQMCNormalSampler

    sampler = SobolQMCNormalSampler(torch.Size([1024]))
    samples = sampler(model.posterior(train_x)).squeeze(-1)  # [256, N]
    argmax_indices = samples.argmax(dim=1)  # [256]

    # Häufigkeiten
    counts = torch.bincount(argmax_indices, minlength=train_x.shape[0]).float()
    probs = counts / counts.sum()

    # Visualisiere Unsicherheit
    df_max_loc = pd.DataFrame({
        "Point Index": range(len(train_x)),
        "P(argmax)": probs.numpy()
    })
    st.write("Estimated distribution over argmax locations:")
    st.dataframe(df_max_loc.sort_values(by="P(argmax)", ascending=False))

    df = pd.DataFrame(best_points, columns=[f"x{i+1}" for i in range(train_x.shape[1])])
    df["Step"] = list(range(2, len(train_x)+1))
    df["Best Posterior Mean"] = best_values
    df["Best Posterior Std"] = best_point_std

    return df
