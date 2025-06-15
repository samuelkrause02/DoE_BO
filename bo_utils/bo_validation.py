import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from sklearn.metrics import mean_squared_error
from .bo_model import build_gp_model
from config import MODEL_CONFIG

def loocv_gp_custom(train_x, train_y, input_dim, verbose=True, model_config=MODEL_CONFIG):
    """
    Perform Leave-One-Out Cross-Validation for your custom GP model.
    """
    import numpy as np
    from botorch.exceptions import ModelFittingError

    n_points = train_x.shape[0]
    preds = []
    truths = []
    noise_values = []

    for i in range(n_points):
        if verbose:
            print(f"LOOCV iteration {i+1}/{n_points}")

        mask = torch.arange(n_points) != i
        x_train_cv = train_x[mask]
        y_train_cv = train_y[mask]
        x_val = train_x[i].unsqueeze(0)
        y_val = train_y[i].unsqueeze(0)

        try:
            model_cv, likelihood_cv = build_gp_model(x_train_cv, y_train_cv, model_config)
            mll = ExactMarginalLogLikelihood(likelihood_cv, model_cv)
            fit_gpytorch_mll(mll)

            model_cv.eval()
            likelihood_cv.eval()

            with torch.no_grad():
                posterior = model_cv.posterior(x_val)
                mean_pred = posterior.mean.item()
                noise_value = model_cv.likelihood.noise.mean().detach().cpu().item()

            preds.append(mean_pred)
            truths.append(y_val.item())
            noise_values.append(noise_value)

        except Exception as e:
            print(f"[WARNING] Skipping iteration {i+1}: {e}")
            continue  # Skip failed fit

    if not preds:
        raise RuntimeError("LOOCV failed: all fits unsuccessful.")

    preds = torch.tensor(preds)
    truths = torch.tensor(truths)

    mse = mean_squared_error(truths.numpy(), preds.numpy())
    avg_noise = np.mean(noise_values)

    print(f"\nLOOCV Mean Squared Error: {mse:.4f}")
    print(f"Average Learned Noise: {avg_noise:.6f}")

    return preds, truths, mse, avg_noise

import numpy as np
def compute_loocv_k_diagnostics(train_x, train_y, input_dim, model_config=MODEL_CONFIG):
    n_p = train_x.shape[0]
    loo_log_probs = []

    # Fit full model
    full_model, full_likelihood = build_gp_model(train_x, train_y, config=model_config)
    mll = ExactMarginalLogLikelihood(full_likelihood, full_model)
    fit_gpytorch_mll(mll)
    full_model.eval()
    full_likelihood.eval()

    # Full model log-marginal likelihood per point
    full_log_probs = []
    with torch.no_grad():
        for i in range(len(train_x)):
            pred = full_model(train_x[i].unsqueeze(0))
            logprob = full_likelihood.log_marginal(train_y[i].unsqueeze(0), pred).item()
            full_log_probs.append(logprob)
    full_log_probs = np.array(full_log_probs)
    for i in range(n_p):
        mask = torch.arange(n_p) != i
        x_cv = train_x[mask]
        y_cv = train_y[mask]
        x_val = train_x[i].unsqueeze(0)
        y_val = train_y[i].unsqueeze(0)

        try:
            model_cv, likelihood_cv = build_gp_model(x_cv, y_cv, config=model_config)
            mll_cv = ExactMarginalLogLikelihood(likelihood_cv, model_cv)
            fit_gpytorch_mll(mll_cv)
            model_cv.eval()
            likelihood_cv.eval()
            with torch.no_grad():
                log_prob_i = likelihood_cv.log_marginal(y_val, model_cv(x_val)).item()
                
            loo_log_probs.append(log_prob_i)
        except Exception as e:
            loo_log_probs.append(float('-inf'))
        elpd = np.sum(loo_log_probs)  # oder: np.mean(...) * N, falls mitteln sinnvoller

    # Difference in log probs as proxy for influence (larger = more influential)
    loo_log_probs_np = np.array(loo_log_probs)
    influence_scores = full_log_probs - loo_log_probs_np
    elpd_dif = np.sum(influence_scores)

    return influence_scores, elpd, elpd_dif


import pandas as pd
import datetime
import os

def log_loocv_result(log_file, model_name, input_dim, kernel_type, noise_prior_range, ard, mse, num_points, prior, prior_lengthscale, noise_value, notes=""):
    """
    Log LOOCV results to a CSV file.
    """
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    log_entry = {
        'Timestamp': time_stamp,
        'Model_Name': model_name,
        'Input_Dim': input_dim,
        'Kernel_Type': kernel_type,
        'Noise_Prior_Range': str(noise_prior_range),
        'Mean_MSE_LOOCV': round(mse, 6),
        'ARD': ard,
        'Number_trainingpoints': num_points,
        'Prior_type': prior,
        'Lengthscale_Prior': str(prior_lengthscale),
        'Noise_value': noise_value,
        'Notes': notes
    }

    df_entry = pd.DataFrame([log_entry])

    # Check if the log file exists
    file_exists = os.path.isfile(log_file)

    # Save to CSV
    df_entry.to_csv(log_file, mode='a', header=not file_exists, index=False, sep=';')
    print(f"Logged LOOCV result to {log_file}")


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error
import streamlit as st


import numpy as np
import torch

def show_model_evaluation():
    """Display comprehensive model evaluation metrics"""
    
    if "loocv_preds" not in st.session_state:
        st.info("Run LOOCV training first to see evaluation metrics")
        return
    
    # Safe tensor conversion
    def to_numpy(data):
        """Convert tensor/list to numpy array safely"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().flatten()
        elif isinstance(data, list):
            # Handle list of tensors or mixed types
            return np.array([x.item() if hasattr(x, 'item') else x for x in data])
        else:
            return np.array(data).flatten()
    
    # Convert predictions and truths
    y_pred = to_numpy(st.session_state["loocv_preds"])
    y_true = to_numpy(st.session_state["loocv_truths"])
    
    # Get posterior variance safely
    model = st.session_state["model"]
    train_x = st.session_state["train_x"]
    
    try:
        posterior = model.posterior(train_x, observation_noise=True)
        y_std = torch.sqrt(posterior.variance).detach().cpu().numpy().flatten()
    except Exception as e:
        st.warning(f"Could not compute posterior variance: {e}")
        # Fallback: use noise estimate
        noise_var = model.likelihood.noise.mean().item()
        y_std = np.full(len(y_pred), np.sqrt(noise_var))
    
    # Evaluate predictions
    df_eval = evaluate_gp_predictions(y_true, y_pred, y_std)
    
    # Add CRPS if available
    try:
        from properscoring import crps_gaussian
        crps = crps_gaussian(y_true, y_pred, y_std)
        df_eval['CRPS'] = [np.mean(crps)]
    except ImportError:
        st.warning("properscoring not available - CRPS not computed")
    except Exception as e:
        st.warning(f"CRPS computation failed: {e}")
    
    # Add ELPD if available
    if "elpd" in st.session_state:
        df_eval['ELPD'] = [st.session_state["elpd"]]
    
    st.subheader("Model Evaluation Metrics")
    st.dataframe(df_eval, use_container_width=True)
    
    return df_eval


def evaluate_gp_predictions(y_true, y_pred, y_std):
    """
    Evaluate GP predictions with comprehensive metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values  
        y_std: Predicted standard deviations
    
    Returns:
        DataFrame with evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, median_absolute_error
    
    # Convert tensors to numpy
    def to_numpy_safe(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().flatten()
        return np.array(data).flatten()
    
    y_true = to_numpy_safe(y_true)
    y_pred = to_numpy_safe(y_pred)
    y_std = to_numpy_safe(y_std)
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred), len(y_std))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    y_std = y_std[:min_len]
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mad = median_absolute_error(y_true, y_pred)
    mdv = np.mean(y_pred - y_true)  # Mean deviation
    mae = np.mean(np.abs(y_pred - y_true))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Uncertainty metrics
    ci_size = 2 * 1.96 * y_std  # 95% CI width
    ci_size_mean = np.mean(ci_size)
    
    # Coverage (fraction of true values within 95% CI)
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    # Calibration metrics
    # Mean prediction interval width
    interval_score = np.mean(ci_size + (2/0.05) * np.maximum(0, lower - y_true) + 
                           (2/0.05) * np.maximum(0, y_true - upper))
    
    # Normalized metrics
    rmse_normalized = rmse / np.std(y_true) if np.std(y_true) > 0 else rmse
    
    metrics_df = pd.DataFrame({
        "RMSE": [rmse],
        "MAE": [mae], 
        "MAD": [mad],
        "MDV": [mdv],
        "R²": [r2],
        "RMSE (norm)": [rmse_normalized],
        "CI Size (avg)": [ci_size_mean],
        "Coverage (95%)": [coverage],
        "Interval Score": [interval_score]
    })
    
    return metrics_df


import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def evaluate_high_performance_metrics(model, ground_truth_model, X_test, train_x, train_y, 
                                    true_optimum_value, threshold_percentiles=[90, 95, 99]):
    """
    Erweiterte Metriken speziell für High-Performance Bereiche
    
    Args:
        model: BO surrogate model
        ground_truth_model: Ground truth GP
        X_test: Test grid
        train_x, train_y: Training data
        true_optimum_value: True global optimum
        threshold_percentiles: Percentiles to define "high-performance" regions
    
    Returns:
        dict: Comprehensive high-performance metrics
    """
    model.eval()
    ground_truth_model.eval()
    
    with torch.no_grad():
        # Predictions on test grid
        posterior_surrogate = model.posterior(X_test)
        mu_surrogate = posterior_surrogate.mean.squeeze(-1).cpu().numpy()
        sigma_surrogate = posterior_surrogate.variance.sqrt().squeeze(-1).cpu().numpy()
        
        # Ground truth values
        mu_true = ground_truth_model.posterior(X_test).mean.squeeze(-1).cpu().numpy()
        
        # Training data predictions vs ground truth
        train_posterior = model.posterior(train_x)
        train_mu_pred = train_posterior.mean.squeeze(-1).cpu().numpy()
        train_sigma_pred = train_posterior.variance.sqrt().squeeze(-1).cpu().numpy()
        train_y_true = ground_truth_model.posterior(train_x).mean.squeeze(-1).cpu().numpy()
    

    if isinstance(mu_surrogate, torch.Tensor):
        y_pred = mu_surrogate.detach().cpu().numpy()
    else:
        y_pred = mu_surrogate
    if isinstance(mu_true, torch.Tensor):
        y_true = mu_surrogate.detach().cpu().numpy()
    else:
        y_true = mu_true
    y_std = sigma_surrogate
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAD
    mad = median_absolute_error(y_true, y_pred)

    # MDV
    mdv = np.mean(y_pred - y_true)

    # CI Size (95%)
    ci_size = 2 * 1.96 * y_std
    ci_size_mean = np.mean(ci_size)

    # Coverage (Anteil der true-Werte im 95%-Konfidenzintervall)
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))

    # CRPS (optional, falls du es implementieren oder mit properscoring nutzen willst)
    # Zusammenfassung als DataFrame

    metrics = pd.DataFrame({
        "RMSE": [rmse],
        "MAD": [mad],
        "MDV": [mdv],
        "CI Size (avg)": [ci_size_mean],
        "Coverage (95%)": [coverage]
    })

    
    # 1. HIGH-PERFORMANCE REGION ANALYSIS
    for percentile in threshold_percentiles:
        threshold = np.percentile(mu_true, percentile)
        high_perf_mask = mu_true >= threshold
        
        if np.any(high_perf_mask):
            # RMSE in high-performance region
            rmse_hp = np.sqrt(mean_squared_error(
                mu_true[high_perf_mask], 
                mu_surrogate[high_perf_mask]
            ))
            
            # Mean absolute error in high-performance region
            mae_hp = np.mean(np.abs(mu_true[high_perf_mask] - mu_surrogate[high_perf_mask]))
            
            # Relative error in high-performance region
            rel_error_hp = np.mean(np.abs(
                (mu_true[high_perf_mask] - mu_surrogate[high_perf_mask]) / 
                (mu_true[high_perf_mask] + 1e-8)
            ))
            
            # Uncertainty quality in high-performance region
            # How well does the model know it doesn't know?
            uncertainty_correlation = np.corrcoef(
                np.abs(mu_true[high_perf_mask] - mu_surrogate[high_perf_mask]),
                sigma_surrogate[high_perf_mask]
            )[0, 1] if len(mu_true[high_perf_mask]) > 1 else 0.0
            
            metrics[f'rmse_top_{percentile}p'] = rmse_hp
            metrics[f'mae_top_{percentile}p'] = mae_hp
            metrics[f'rel_error_top_{percentile}p'] = rel_error_hp
            metrics[f'uncertainty_corr_top_{percentile}p'] = uncertainty_correlation
        else:
            metrics[f'rmse_top_{percentile}p'] = np.nan
            metrics[f'mae_top_{percentile}p'] = np.nan
            metrics[f'rel_error_top_{percentile}p'] = np.nan
            metrics[f'uncertainty_corr_top_{percentile}p'] = np.nan
    
    # 2. OPTIMUM DETECTION QUALITY
    # How well does the model identify the true optimum region?
    true_opt_distance = np.abs(mu_true - true_optimum_value)
    pred_opt_distance = np.abs(mu_surrogate - true_optimum_value)
    
    # Rank correlation: Does the model rank high-value regions correctly?
    rank_correlation = np.corrcoef(mu_true, mu_surrogate)[0, 1]
    
    # Top-k precision: Of the top-k predicted points, how many are actually top-k?
    for k in [5, 10, 20]:
        if len(mu_true) >= k:
            true_top_k = np.argsort(mu_true)[-k:]
            pred_top_k = np.argsort(mu_surrogate)[-k:]
            top_k_precision = len(np.intersect1d(true_top_k, pred_top_k)) / k
            metrics[f'top_{k}_precision'] = top_k_precision
    
    metrics['rank_correlation'] = rank_correlation
    metrics['optimum_detection_error'] = np.min(pred_opt_distance) - np.min(true_opt_distance)
    
    # 3. TRAINING DATA QUALITY IN HIGH-PERFORMANCE REGIONS
    if len(train_y) > 0:
        # What fraction of training data is in high-performance regions?
        for percentile in [90, 95]:
            threshold = np.percentile(mu_true, percentile)
            high_perf_training_fraction = np.mean(train_y_true >= threshold)
            metrics[f'training_coverage_top_{percentile}p'] = high_perf_training_fraction
        
        # Regret reduction rate
        train_y_np = train_y.detach().cpu().numpy() if torch.is_tensor(train_y) else train_y
        best_so_far = np.maximum.accumulate(train_y_np.flatten())
        regret_sequence = (true_optimum_value - best_so_far) ** 2
        
        # Simple regret (final)
        simple_regret = regret_sequence[-1]
        
        # Cumulative regret
        cumulative_regret = np.sum(regret_sequence)
        
        # Regret reduction rate (slope of log regret vs log iteration)
        if len(regret_sequence) > 2:
            iterations = np.arange(1, len(regret_sequence) + 1)
            # Avoid log(0) by adding small epsilon
            log_regret = np.log(regret_sequence + 1e-10)
            log_iter = np.log(iterations)
            regret_slope = np.polyfit(log_iter, log_regret, 1)[0]
        else:
            regret_slope = 0.0
        
        metrics['simple_regret'] = simple_regret
        metrics['cumulative_regret'] = cumulative_regret
        metrics['regret_reduction_rate'] = -regret_slope  # Negative because we want decreasing regret
    
    # 4. EXPLOITATION vs EXPLORATION BALANCE
    # How concentrated are the training points?
    if len(train_x) > 1:
        # Compute pairwise distances between training points
        train_x_np = train_x.detach().cpu().numpy() if torch.is_tensor(train_x) else train_x
        from scipy.spatial.distance import pdist
        distances = pdist(train_x_np)
        
        # Diversity metrics
        min_distance = np.min(distances) if len(distances) > 0 else 0.0
        mean_distance = np.mean(distances) if len(distances) > 0 else 0.0
        
        metrics['min_point_distance'] = min_distance
        metrics['mean_point_distance'] = mean_distance
        metrics['exploration_diversity'] = mean_distance / (min_distance + 1e-8)
    
    return metrics

