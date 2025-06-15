import torch
import numpy as np
import pandas as pd
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_utils import compute_pairwise_gradients

def remove_loocv_outliers(train_x, train_y, input_dim, config, percent_remove):
    """Remove top x% of points with highest LOOCV-MSE."""
    n_total = train_x.shape[0]
    n_remove = int(round(n_total * percent_remove / 100))
    if n_remove >= n_total:
        n_remove = n_total - 1

    current_indices = list(range(n_total))
    removed_info = []

    for _ in range(n_remove):
        point_errors = []

        for i in current_indices:
            test_indices = [idx for idx in current_indices if idx != i]
            x_loo, y_loo = train_x[test_indices], train_y[test_indices]

            model, _ = build_gp_model(x_loo, y_loo, config)
            fit_model(model)

            pred_i = model.posterior(train_x[i:i+1]).mean.item()
            true_i = train_y[i].item()
            mse_i = (pred_i - true_i) ** 2
            mae_i = abs(pred_i - true_i)

            point_errors.append({"Index": i, "True": true_i, "Pred": pred_i, "MSE": mse_i, "MAE": mae_i})

        point_errors.sort(key=lambda x: x["MSE"], reverse=True)
        worst = point_errors[0]
        current_indices.remove(worst["Index"])
        removed_info.append(worst)

    return train_x[current_indices], train_y[current_indices], pd.DataFrame(removed_info)

def remove_points_by_noise_influence(train_x, train_y, input_dim, config, percent_remove):
    """Remove x% of points with strongest negative influence on learned noise."""
    n_total = train_x.shape[0]
    n_remove = int(round(n_total * percent_remove / 100))
    if n_remove >= n_total:
        n_remove = n_total - 1

    current_indices = list(range(n_total))
    removed = []

    for _ in range(n_remove):
        ref_x, ref_y = train_x[current_indices], train_y[current_indices]

        deltas = []
        for i in current_indices:
            test_indices = [idx for idx in current_indices if idx != i]
            x_loo, y_loo = train_x[test_indices], train_y[test_indices]

            model_loo, _ = build_gp_model(x_loo, y_loo, config)
            fit_model(model_loo)
            noise_loo = model_loo.likelihood.noise.item()

            model_ref, _ = build_gp_model(ref_x, ref_y, config)
            fit_model(model_ref)
            noise_ref = model_ref.likelihood.noise.item()

            delta = (noise_loo ** 0.5 - noise_ref ** 0.5) * 100
            deltas.append((i, delta))

        deltas.sort(key=lambda x: x[1])
        worst_idx, worst_delta = deltas[0]
        current_indices.remove(worst_idx)
        removed.append((worst_idx, worst_delta))

    return train_x[current_indices], train_y[current_indices], pd.DataFrame(removed, columns=["Removed Index", "Noise Δσ"])

def remove_by_looph_loss(train_x, train_y, input_dim, config, percent_remove):
    """Remove x% of points that most negatively affect the LOOPH-Loss."""
    n_total = train_x.shape[0]
    n_remove = int(round(n_total * percent_remove / 100))
    if n_remove >= n_total:
        n_remove = n_total - 1

    current_indices = list(range(n_total))
    removed = []

    for _ in range(n_remove):
        ref_x, ref_y = train_x[current_indices], train_y[current_indices]
        model_ref, _ = build_gp_model(ref_x, ref_y, config)
        fit_model(model_ref)

        ref_sum = _compute_looph_sum(model_ref, ref_x, ref_y)
        deltas = []

        for i in current_indices:
            test_idx = [idx for idx in current_indices if idx != i]
            x_test, y_test = train_x[test_idx], train_y[test_idx]

            model_test, _ = build_gp_model(x_test, y_test, config)
            fit_model(model_test)
            test_sum = _compute_looph_sum(model_test, x_test, y_test)

            delta = (ref_sum - test_sum) / (-1 * ref_sum)
            deltas.append((i, delta))

        deltas.sort(key=lambda x: x[1], reverse=True)
        worst_idx, delta_val = deltas[0]
        current_indices.remove(worst_idx)
        removed.append((worst_idx, delta_val * 100))

    return train_x[current_indices], train_y[current_indices], pd.DataFrame(removed, columns=["Removed Index", "ΔLOOPH Reduction %"])

def _compute_looph_sum(model, x, y, delta=3.0):
    y = y.view(-1)  # Flache Form sicherstellen
    looph_sum = 0.0
    for i in range(x.shape[0]):
        posterior = model.posterior(x[i:i+1])
        mu = posterior.mean.view(-1)[0].item()
        sigma = posterior.variance.sqrt().view(-1)[0].item()
        y_true = y[i].item()
        sigma2 = max(sigma**2, 1e-8)
        core = np.sqrt(1 + (y_true - mu)**2 / (delta**2 * sigma2)) - 1
        loss = 2 * delta**2 * core + np.log(sigma2)
        looph_sum += loss
    return looph_sum

def remove_by_gradient_pairs(train_x, train_y, top_k):
    df_grad = compute_pairwise_gradients(train_x, train_y, top_k=top_k)
    indices = set(df_grad["Index 1"]).union(set(df_grad["Index 2"]))
    keep_mask = np.ones(train_x.shape[0], dtype=bool)
    keep_mask[list(indices)] = False
    df_removed = pd.DataFrame({"Removed Index": sorted(indices)})
    return train_x[keep_mask], train_y[keep_mask], df_removed
