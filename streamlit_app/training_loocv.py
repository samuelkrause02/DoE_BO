import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_data import prepare_training_data, prepare_training_tensors, load_csv_experiment_data
from bo_utils.bo_utils import rescale_single_point, compute_pairwise_gradients
from bo_utils.bo_validation import loocv_gp_custom
import torch

def loocv_and_training_section(uploaded_file, bounds, output_col_name, MODEL_CONFIG, COLUMNS_CONFIG, data=None):
    
    st.subheader("Data Filtering Options")
    
    # Manual removal
    with st.expander("Manual Data Point Removal", expanded=False):
        remove_indices = st.multiselect(
            "Select indices to remove:",
            options=list(data.index),
            help="Remove outlier points manually"
        )
    
    # Automatic removal sliders
    col1, col2 = st.columns(2)
    with col1:
        percent_remove = st.slider("Remove worst LOOCV points (%)", 0, 50, 0, step=5)
        percent_looph_remove = st.slider("Remove worst Standardized Residuals points (%)", 0, 50, 0, step=5)
    
    with col2:
        percent_noise_remove = st.slider("Remove high noise influence (%)", 0, 50, 0, step=5)
        top_k_pairs = st.slider("Remove top gradient pairs", 0, 10, 0, step=1)
    
    # Summary of filtering
    if any([remove_indices, percent_remove, percent_noise_remove, percent_looph_remove, top_k_pairs]):
        total_manual = len(remove_indices)
        estimated_auto = int(len(data) * (percent_remove + percent_noise_remove + percent_looph_remove) / 100) + top_k_pairs * 2
        st.info(f"Estimated removal: {total_manual} manual + ~{estimated_auto} automatic points")
    
    if st.button("Train Model and Run LOOCV", type="primary"):
        if uploaded_file is None:
            st.error("❌ Please upload experiment data first")
            return None, None
        
        with st.spinner("Processing data and training model..."):
            # Load and prepare data
            uploaded_file.seek(0)
            raw_inputs, raw_outputs, df = load_csv_experiment_data(
                uploaded_file, columns_config=COLUMNS_CONFIG, output_column=output_col_name
            )
            
            # Manual removal
            if remove_indices:
                st.write(f"Removing {len(remove_indices)} manual points: {remove_indices}")
                raw_inputs = np.delete(raw_inputs, remove_indices, axis=0)
                raw_outputs = np.delete(raw_outputs, remove_indices, axis=0)
            
            # Prepare training data
            scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds)
            input_dim = scaled_x.shape[1]
            train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)
            
            # Store original size
            original_size = train_x.shape[0]
            
            # Filter 1: LOOCV MSE removal
            if percent_remove > 0:
                train_x, train_y, removed_mse = remove_worst_loocv_points(
                    train_x, train_y, input_dim, MODEL_CONFIG, percent_remove
                )
                st.subheader("Removed LOOCV Outliers")
                st.dataframe(pd.DataFrame(removed_mse))
            
            # Filter 2: Noise influence removal
            if percent_noise_remove > 0:
                train_x, train_y, removed_noise = remove_noise_influence_points(
                    train_x, train_y, input_dim, MODEL_CONFIG, percent_noise_remove
                )
                st.subheader("Removed Noise Influence Points")
                st.dataframe(pd.DataFrame(removed_noise, columns=["Index", "Noise Δσ"]))
            
            # Filter 3: LOOPH removal
            if percent_looph_remove > 0:
                train_x, train_y, removed_residuals = remove_worst_standardized_residuals_points(
                    train_x, train_y, MODEL_CONFIG, percent_looph_remove
                )
                st.subheader("Removed High Standardized Residuals Points")
                st.dataframe(pd.DataFrame(removed_residuals))
            
            # Filter 4: Gradient pairs removal
            if top_k_pairs > 0:
                train_x, train_y, removed_gradient = remove_gradient_pairs(
                    train_x, train_y, top_k_pairs
                )
                st.subheader("Removed Gradient Pair Points")
                st.dataframe(removed_gradient)
            
            # Summary
            final_size = train_x.shape[0]
            reduction_pct = (original_size - final_size) / original_size * 100
            
            st.success(f"✅ Training complete: {final_size}/{original_size} points remaining ({reduction_pct:.1f}% reduction)")
            
            return train_x, train_y
    
    return None, None


def remove_worst_loocv_points(train_x, train_y, input_dim, MODEL_CONFIG, percent_remove):
    """Remove points with worst LOOCV performance iteratively"""
    n_total = train_x.shape[0]
    n_remove = min(int(n_total * percent_remove / 100), n_total - 1)
    
    current_indices = list(range(n_total))
    removed_info = []
    
    for _ in range(n_remove):
        point_errors = []
        
        for i in current_indices:
            # Leave-one-out indices
            loo_indices = [idx for idx in current_indices if idx != i]
            x_loo = train_x[loo_indices]
            y_loo = train_y[loo_indices]
            
            # Train LOO model
            model_loo, _ = build_gp_model(x_loo, y_loo, MODEL_CONFIG)
            fit_model(model_loo)
            
            # Predict left-out point
            pred_i = model_loo.posterior(train_x[i:i+1]).mean.item()
            true_i = train_y[i].item()
            mse_i = (pred_i - true_i) ** 2
            
            point_errors.append({
                "Index": i, "True": true_i, "Pred": pred_i, "MSE": mse_i
            })
        
        # Remove worst point
        worst_point = max(point_errors, key=lambda x: x["MSE"])
        current_indices.remove(worst_point["Index"])
        removed_info.append(worst_point)
    
    # Return filtered data
    keep_indices = np.array(current_indices)
    return train_x[keep_indices], train_y[keep_indices], removed_info


def remove_noise_influence_points(train_x, train_y, input_dim, MODEL_CONFIG, percent_remove):
    """Remove points with strongest negative noise influence"""
    n_total = train_x.shape[0]
    n_remove = min(int(n_total * percent_remove / 100), n_total - 1)
    
    current_indices = list(range(n_total))
    removed_indices = []
    
    for _ in range(n_remove):
        # Reference model with current points
        ref_x = train_x[current_indices]
        ref_y = train_y[current_indices]
        model_ref, _ = build_gp_model(ref_x, ref_y, MODEL_CONFIG)
        fit_model(model_ref)
        ref_noise = model_ref.likelihood.noise.item()
        
        noise_deltas = []
        for i in current_indices:
            # Model without point i
            loo_indices = [idx for idx in current_indices if idx != i]
            x_loo = train_x[loo_indices]
            y_loo = train_y[loo_indices]
            
            model_loo, _ = build_gp_model(x_loo, y_loo, MODEL_CONFIG)
            fit_model(model_loo)
            noise_i = model_loo.likelihood.noise.item()
            
            delta = (noise_i**0.5 - ref_noise**0.5) * 100
            noise_deltas.append((i, delta))
        
        # Remove point with strongest negative influence
        worst_idx, worst_delta = min(noise_deltas, key=lambda x: x[1])
        current_indices.remove(worst_idx)
        removed_indices.append((worst_idx, worst_delta))
    
    keep_indices = np.array(current_indices)
    return train_x[keep_indices], train_y[keep_indices], removed_indices

def remove_worst_standardized_residuals_points(train_x, train_y, MODEL_CONFIG, percent_remove):
    """Remove points with highest absolute standardized residuals"""
    n_total = train_x.shape[0]
    n_remove = min(int(n_total * percent_remove / 100), n_total - 1)
    
    # Train model on all data
    model, _ = build_gp_model(train_x, train_y, MODEL_CONFIG)
    fit_model(model)
    model.eval()
    
    # Compute standardized residuals for all points
    with torch.no_grad():
        posterior = model.posterior(train_x, observation_noise=False)
        predictions = posterior.mean.squeeze()
        uncertainties = posterior.variance.sqrt().squeeze()
    
    # Calculate standardized residuals
    residuals = train_y.squeeze() - predictions
    standardized_residuals = residuals / uncertainties
    abs_std_residuals = torch.abs(standardized_residuals)
    
    # Get indices of worst points
    _, worst_indices = torch.topk(abs_std_residuals, k=n_remove, largest=True)
    worst_indices = worst_indices.cpu().numpy()
    
    # Create info about removed points
    removed_info = []
    for idx in worst_indices:
        removed_info.append({
            "Index": int(idx),
            "True": float(train_y[idx].item()),
            "Pred": float(predictions[idx].item()),
            "Std_Residual": float(standardized_residuals[idx].item()),
            "Abs_Std_Residual": float(abs_std_residuals[idx].item())
        })
    
    # Create mask to keep points (remove worst)
    keep_mask = torch.ones(n_total, dtype=torch.bool)
    keep_mask[worst_indices] = False
    
    return train_x[keep_mask], train_y[keep_mask], removed_info

def remove_worst_looph_points(train_x, train_y, input_dim, MODEL_CONFIG, percent_remove):
    """Remove points with worst LOOPH impact"""
    n_total = train_x.shape[0]
    n_remove = min(int(n_total * percent_remove / 100), n_total - 1)
    
    current_indices = list(range(n_total))
    removed_info = []
    
    for _ in range(n_remove):
        # Reference LOOPH
        ref_x = train_x[current_indices]
        ref_y = train_y[current_indices]
        ref_looph = compute_total_looph(ref_x, ref_y, input_dim, MODEL_CONFIG)
        
        deltas = []
        for i in current_indices:
            # LOOPH without point i
            loo_indices = [idx for idx in current_indices if idx != i]
            x_test = train_x[loo_indices]
            y_test = train_y[loo_indices]
            test_looph = compute_total_looph(x_test, y_test, input_dim, MODEL_CONFIG)
            
            delta_looph = (ref_looph - test_looph) / (-1 * ref_looph)
            deltas.append((i, delta_looph))
        
        # Remove point with best improvement
        worst_idx, best_improvement = max(deltas, key=lambda x: x[1])
        current_indices.remove(worst_idx)
        removed_info.append((worst_idx, best_improvement * 100))
    
    keep_indices = np.array(current_indices)
    return train_x[keep_indices], train_y[keep_indices], removed_info


def compute_total_looph(train_x, train_y, input_dim, MODEL_CONFIG):
    """Compute total LOOPH loss for dataset"""
    model, _ = build_gp_model(train_x, train_y, MODEL_CONFIG)
    fit_model(model)
    
    total_looph = 0.0
    for i in range(train_x.shape[0]):
        posterior = model.posterior(train_x[i:i+1])
        mu = posterior.mean.item()
        sigma = posterior.variance.sqrt().item()
        y_true = train_y[i].item()
        
        sigma2 = max(sigma**2, 1e-8)
        delta = 3
        core = np.sqrt(1 + (y_true - mu)**2 / (delta**2 * sigma2)) - 1
        loss = 2 * delta**2 * core + np.log(sigma2)
        total_looph += loss
    
    return total_looph


def remove_gradient_pairs(train_x, train_y, top_k_pairs):
    """Remove points involved in top gradient pairs"""
    df_gradient_pairs = compute_pairwise_gradients(train_x, train_y, top_k=top_k_pairs)
    
    # Extract unique indices to remove
    indices_to_remove = set(df_gradient_pairs["Index 1"]).union(set(df_gradient_pairs["Index 2"]))
    indices_to_remove = sorted(list(indices_to_remove))
    
    # Create mask to keep points
    keep_mask = np.ones(train_x.shape[0], dtype=bool)
    keep_mask[indices_to_remove] = False
    
    removed_df = pd.DataFrame({"Removed Index": indices_to_remove})
    
    return train_x[keep_mask], train_y[keep_mask], removed_df