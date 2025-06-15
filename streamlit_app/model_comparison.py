import torch
import numpy as np
import pandas as pd
import streamlit as st
import time
import warnings
warnings.filterwarnings('ignore')

# Import your GP model builder functions
from bo_utils.bo_model import build_gp_model, generate_model_configs, fit_model
from bo_utils.bo_validation import compute_loocv_k_diagnostics


def safe_tensor_to_numpy(data):
    """Convert tensor/list to numpy array safely"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().flatten()
    elif isinstance(data, list):
        return np.array([x.item() if hasattr(x, 'item') else x for x in data])
    else:
        return np.array(data).flatten()


def compute_coverage_metrics(y_true, y_pred, y_std):
    """
    DEPRECATED: Use your existing evaluate_gp_predictions function instead
    """
    pass


def evaluate_gp_predictions(y_true, y_pred, y_std):
    """
    Use the existing evaluate_gp_predictions function but extract coverage for filtering
    This is your existing function - keeping it as is!
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


def extract_coverage_90_from_existing_function(y_true, y_pred, y_std):
    """
    Extract 90% coverage using the same logic as your existing function but for 90% level
    """
    # Convert tensors to numpy (same as your function)
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
    
    # 90% coverage (same style as your 95% coverage)
    lower_90 = y_pred - 1.645 * y_std  # 90% CI
    upper_90 = y_pred + 1.645 * y_std
    coverage_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
    
    # 95% coverage (same as your existing function)
    lower_95 = y_pred - 1.96 * y_std
    upper_95 = y_pred + 1.96 * y_std
    coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))
    
    return coverage_90 * 100, coverage_95 * 100  # Return as percentages


def get_loocv_predictions_with_uncertainty(train_x, train_y, input_dim, model_config):
    """
    Compute LOOCV predictions WITH uncertainty estimates for coverage evaluation
    """
    n_samples = len(train_y)
    predictions = []
    uncertainties = []
    truths = []
    
    for i in range(n_samples):
        # Create LOO split
        mask = torch.arange(n_samples) != i
        x_train = train_x[mask]
        y_train = train_y[mask]
        x_test = train_x[i].unsqueeze(0)
        y_test = train_y[i]
        
        try:
            # Fit model on training fold
            model, likelihood = build_gp_model(x_train, y_train, config=model_config)
            fit_model(model, max_iter=100)
            
            # Make prediction WITH uncertainty
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(x_test, observation_noise=True)
                pred_mean = posterior.mean.item()
                pred_std = torch.sqrt(posterior.variance).item()
                
            predictions.append(pred_mean)
            uncertainties.append(pred_std)
            truths.append(y_test.item())
            
        except Exception as e:
            print(f"LOOCV prediction failed for sample {i}: {e}")
            predictions.append(np.nan)
            uncertainties.append(np.nan)
            truths.append(y_test.item())
    
    return np.array(predictions), np.array(uncertainties), np.array(truths)


def config_to_string(config):
    parts = []
    parts.append("K:" + str(config['kernel']['type']))
    parts.append("ARD:" + str(config.get('ard', False)))
    
    if config.get('lengthscale_prior'):
        parts.append("LS:" + str(config['lengthscale_prior']['type']))
    else:
        parts.append("LS:None")
    
    os_config = config.get('outputscale', {})
    if os_config.get('fixed', False):
        parts.append("OS:Fixed")
    else:
        parts.append("OS:Learn")
    
    if config.get('noise_prior'):
        parts.append("N:" + str(config['noise_prior']['type']))
    else:
        parts.append("N:None")
    
    return " | ".join(parts)


def extract_config_parameters(config):
    """Extract readable configuration parameters - PyArrow compatible"""
    params = {}
    
    # Kernel details
    kernel = config.get('kernel', {})
    params['Kernel'] = kernel.get('type', 'Unknown')
    
    # Nu parameter - handle N/A properly for PyArrow
    if kernel.get('type') == 'Matern':
        nu_val = kernel.get('nu')
        if nu_val is not None:
            params['Nu'] = float(nu_val)  # Convert to float
        else:
            params['Nu'] = None  # Use None instead of 'N/A'
    else:
        params['Nu'] = None  # Use None instead of 'N/A'
    
    # Power parameter - handle N/A properly
    if kernel.get('type') == 'Polynomial':
        power_val = kernel.get('power')
        if power_val is not None:
            params['Power'] = float(power_val)
        else:
            params['Power'] = None
    else:
        params['Power'] = None
    
    # ARD
    params['ARD'] = config.get('ard', False)
    
    # Lengthscale Prior
    ls_prior = config.get('lengthscale_prior')
    if ls_prior:
        if ls_prior['type'] == 'gamma':
            params['LS_Prior'] = f"Gamma(α={ls_prior.get('concentration', 'N/A')}, β={ls_prior.get('rate', 'N/A')})"
        elif ls_prior['type'] == 'lognormal':
            params['LS_Prior'] = f"LogNormal(μ={ls_prior.get('loc', 'N/A'):.2f}, σ={ls_prior.get('scale', 'N/A'):.2f})"
        else:
            params['LS_Prior'] = ls_prior['type']
    else:
        params['LS_Prior'] = 'None'
    
    # Outputscale
    os_config = config.get('outputscale', {})
    if not os_config.get('use', True):
        params['Outputscale'] = 'Disabled'
    elif os_config.get('fixed', False):
        params['Outputscale'] = f"Fixed({os_config.get('init_value', 1.0):.2f})"
    else:
        params['Outputscale'] = 'Learnable'
    
    # Noise Prior
    noise_prior = config.get('noise_prior')
    if noise_prior:
        if noise_prior['type'] == 'gamma':
            params['Noise_Prior'] = f"Gamma(α={noise_prior.get('concentration', 'N/A')}, β={noise_prior.get('rate', 'N/A')})"
        elif noise_prior['type'] == 'lognormal':
            params['Noise_Prior'] = f"LogNormal(μ={noise_prior.get('loc', 'N/A'):.2f}, σ={noise_prior.get('scale', 'N/A'):.2f})"
        else:
            params['Noise_Prior'] = noise_prior['type']
    else:
        params['Noise_Prior'] = 'None'
    
    return params


def compare_gp_models_with_coverage_filter(train_x, train_y, input_dim, configs=None, 
                                         max_configs=20, min_coverage_90=90.0, 
                                         compute_loocv_preds=True, progress_callback=None):
    """
    Compare GP models with Coverage >= min_coverage_90% filter, then sort by ELPD
    """
    
    if configs is None:
        configs = generate_model_configs(subset=True, max_configs=max_configs)
    
    results = []
    valid_models = []  # Models that pass coverage filter
    
    print(f"Comparing {len(configs)} GP model configurations...")
    print(f"Coverage filter: ≥{min_coverage_90}% at 90% level")
    
    for i, config in enumerate(configs):
        if progress_callback:
            progress_callback(i, len(configs), config)
            
        try:
            start_time = time.time()
            
            # Compute ELPD using existing function
            influence_scores, elpd, elpd_dif = compute_loocv_k_diagnostics(
                train_x, train_y, input_dim, model_config=config
            )
            
            elpd_time = time.time() - start_time
            
            # Fit full model for hyperparameter extraction
            model, likelihood = build_gp_model(train_x, train_y, config=config)
            mll_score = fit_model(model, max_iter=100)
            
            fit_time = time.time() - start_time - elpd_time
            
            # Get LOOCV predictions WITH uncertainty for coverage evaluation
            loocv_preds = None
            loocv_stds = None  
            loocv_truths = None
            loocv_mse = np.nan
            coverage_90 = np.nan
            coverage_95 = np.nan
            passes_coverage_filter = False
            
            if compute_loocv_preds:
                try:
                    loocv_preds, loocv_stds, loocv_truths = get_loocv_predictions_with_uncertainty(
                        train_x, train_y, input_dim, config
                    )
                    
                    if (loocv_preds is not None and loocv_stds is not None and 
                        loocv_truths is not None):
                        # Remove NaN values
                        mask = ~(np.isnan(loocv_preds) | np.isnan(loocv_stds) | 
                                np.isnan(loocv_truths))
                        
                        if mask.sum() > 0:
                            valid_preds = loocv_preds[mask]
                            valid_stds = loocv_stds[mask]
                            valid_truths = loocv_truths[mask]
                            
                            # Compute MSE
                            loocv_mse = np.mean((valid_truths - valid_preds)**2)
                            
                            # Compute coverage directly inline using your function's logic
                            
                            # First get 95% coverage from your existing function
                            try:
                                metrics_df = evaluate_gp_predictions(valid_truths, valid_preds, valid_stds)
                                coverage_95 = metrics_df["Coverage (95%)"].iloc[0] * 100
                            except Exception as e:
                                print(f"Error calling evaluate_gp_predictions: {e}")
                                coverage_95 = np.nan
                            
                            # Compute 90% coverage using same logic as your function
                            try:
                                lower_90 = valid_preds - 1.645 * valid_stds
                                upper_90 = valid_preds + 1.645 * valid_stds
                                coverage_90 = np.mean((valid_truths >= lower_90) & (valid_truths <= upper_90)) * 100
                            except Exception as e:
                                print(f"Error computing 90% coverage: {e}")
                                coverage_90 = np.nan
                            
                            # Check coverage filter
                            passes_coverage_filter = coverage_90 >= min_coverage_90
                            
                except Exception as e:
                    print(f"LOOCV predictions failed for config {i+1}: {e}")
            
            # Extract learned hyperparameters
            learned_lengthscale = "Failed"
            learned_outputscale = np.nan
            learned_noise = np.nan

            try:
                with torch.no_grad():
                    if hasattr(model.covar_module, 'base_kernel'):
                        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                        learned_outputscale = model.covar_module.outputscale.detach().cpu().item()
                    else:
                        lengthscale = model.covar_module.lengthscale.detach().cpu().numpy()
                        learned_outputscale = 1.0
                    
                    # Use corrected noise calculation (with outcome transform)
                    try:
                        learned_noise = model.likelihood.noise.mean().item() * (model.outcome_transform.stdvs.item() ** 2)
                    except:
                        learned_noise = model.likelihood.noise.mean().item()
                    
                    # Format learned lengthscales
                    if len(lengthscale.flatten()) == 1:
                        learned_lengthscale = f"{float(lengthscale.flatten()[0]):.3f}"
                    else:
                        learned_lengthscale = f"[{', '.join([f'{float(x):.3f}' for x in lengthscale.flatten()])}]"
                    
            except Exception as e:
                print(f"Warning: Could not extract hyperparameters for config {i+1}: {e}")
            
            # Extract configuration parameters
            config_params = extract_config_parameters(config)
            
            # Store results
            result = {
                'Config_ID': i + 1,
                'ELPD': float(elpd) if elpd is not None else np.nan,
                'ELPD_Diff': float(elpd_dif) if elpd_dif is not None else np.nan,
                'MLL_Score': float(mll_score) if mll_score > -np.inf else np.nan,
                'LOOCV_MSE': float(loocv_mse) if not np.isnan(loocv_mse) else np.nan,
                'Coverage_90%': float(coverage_90) if not np.isnan(coverage_90) else np.nan,
                'Coverage_95%': float(coverage_95) if not np.isnan(coverage_95) else np.nan,
                'Passes_Coverage_Filter': passes_coverage_filter,
                
                # Configuration Parameters (what user can set) - PyArrow compatible
                'Kernel': config_params['Kernel'],
                'Nu': config_params['Nu'],  # Now None instead of 'N/A'
                'Power': config_params['Power'],  # Now None instead of 'N/A'
                'ARD': config_params['ARD'],
                'LS_Prior': config_params['LS_Prior'],
                'Outputscale_Config': config_params['Outputscale'],
                'Noise_Prior': config_params['Noise_Prior'],
                
                # Key learned parameters
                'Learned_Noise': f"{learned_noise:.6f}" if not np.isnan(learned_noise) else "Failed",
                
                # Other learned parameters (for reference)
                'Learned_LS': learned_lengthscale,
                'Learned_OS': f"{learned_outputscale:.3f}" if not np.isnan(learned_outputscale) else "Failed",
                
                # Timing and metadata
                'Fit_Time_sec': float(fit_time),
                'ELPD_Time_sec': float(elpd_time),
                'Status': 'Success',
                
                # Objects for further use
                'Model': model,
                'Config': config,
                'LOOCV_Preds': loocv_preds,
                'LOOCV_Stds': loocv_stds,
                'LOOCV_Truths': loocv_truths,
                'Influence_Scores': influence_scores
            }
            
            results.append(result)
            
            if passes_coverage_filter:
                valid_models.append(result)
            
            coverage_status = f"✅ Pass" if passes_coverage_filter else f"❌ Fail"
            print(f"Config {i+1}/{len(configs)}: ELPD={elpd:.4f}, "
                  f"Coverage90={coverage_90:.1f}% ({coverage_status})")
            
        except Exception as e:
            print(f"Config {i+1} failed completely: {e}")
            
            # Extract what we can from failed config
            try:
                config_params = extract_config_parameters(config)
            except:
                config_params = {
                    'Kernel': 'Failed', 'Nu': None, 'Power': None, 'ARD': None,  # Use None instead of 'N/A'
                    'LS_Prior': 'Failed', 'Outputscale': 'Failed', 'Noise_Prior': 'Failed'
                }
            
            result = {
                'Config_ID': i + 1,
                'ELPD': np.nan,
                'ELPD_Diff': np.nan,
                'MLL_Score': np.nan,
                'LOOCV_MSE': np.nan,
                'Coverage_90%': np.nan,
                'Coverage_95%': np.nan,
                'Passes_Coverage_Filter': False,
                'Kernel': config_params['Kernel'],
                'Nu': config_params['Nu'],
                'Power': config_params['Power'],
                'ARD': config_params['ARD'],
                'LS_Prior': config_params['LS_Prior'],
                'Outputscale_Config': config_params['Outputscale'],
                'Noise_Prior': config_params['Noise_Prior'],
                'Learned_Noise': 'Failed',
                'Learned_LS': 'Failed',
                'Learned_OS': 'Failed',
                'Fit_Time_sec': 0,
                'ELPD_Time_sec': 0,
                'Status': f'Failed: {str(e)[:50]}',
                'Model': None,
                'Config': config,
                'LOOCV_Preds': None,
                'LOOCV_Stds': None,
                'LOOCV_Truths': None,
                'Influence_Scores': None
            }
            results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # CRITICAL: Filter by coverage first, then sort by ELPD
    valid_results_df = results_df[results_df['Passes_Coverage_Filter'] == True].copy()
    
    if len(valid_results_df) > 0:
        # Sort valid models by ELPD (higher is better)
        valid_results_df = valid_results_df.sort_values('ELPD', ascending=False)
        valid_results_df = valid_results_df.reset_index(drop=True)
        
        best_model_info = valid_results_df.iloc[0]
        best_model = best_model_info['Model']
        best_config = best_model_info['Config']
        
        print(f"\n=== BEST VALID MODEL (Coverage ≥{min_coverage_90}% + Best ELPD) ===")
        print(f"ELPD: {best_model_info['ELPD']:.4f}")
        print(f"Coverage 90%: {best_model_info['Coverage_90%']:.1f}%")
        print(f"Coverage 95%: {best_model_info['Coverage_95%']:.1f}%")
        print(f"Kernel: {best_model_info['Kernel']}")
        print(f"ARD: {best_model_info['ARD']}")
        
    else:
        print(f"\n❌ NO MODELS PASSED COVERAGE FILTER (≥{min_coverage_90}%)")
        # Fallback: show all results sorted by ELPD
        valid_results_df = results_df.sort_values('ELPD', ascending=False)
        valid_results_df = valid_results_df.reset_index(drop=True)
        best_model = None
        best_config = None
        
        print("Showing all models sorted by ELPD:")
        for i, row in results_df.head(5).iterrows():
            print(f"  Config {row['Config_ID']}: ELPD={row['ELPD']:.4f}, "
                  f"Coverage90={row['Coverage_90%']:.1f}%")
    
    # Combine results: valid models first, then others
    if len(valid_results_df) > 0:
        invalid_results_df = results_df[results_df['Passes_Coverage_Filter'] == False].copy()
        invalid_results_df = invalid_results_df.sort_values('ELPD', ascending=False)
        final_results_df = pd.concat([valid_results_df, invalid_results_df], ignore_index=True)
    else:
        final_results_df = valid_results_df
    
    return final_results_df, best_model, best_config


def streamlit_model_comparison():
    """Enhanced Streamlit interface with coverage filtering"""
    
    st.subheader("🔍 GP Model Comparison with Coverage Filter")
    
    # Check if we have training data
    if "train_x" not in st.session_state or "train_y" not in st.session_state:
        st.error("No training data found. Please load data first.")
        return
    
    train_x = st.session_state["train_x"]
    train_y = st.session_state["train_y"]
    input_dim = train_x.shape[1]
    
    st.info(f"Training data: {len(train_y)} samples, {input_dim} dimensions")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_configs = st.number_input("Max Configurations", min_value=5, max_value=50, value=15)
        subset_configs = st.checkbox("Use config subset", value=True)
    
    with col2:
        min_coverage_90 = st.number_input("Min Coverage 90% [%]", min_value=80.0, max_value=98.0, 
                                        value=90.0, step=1.0, format="%.1f")
        compute_loocv_preds = st.checkbox("Compute LOOCV predictions", value=True)
    
    with col3:
        show_failed = st.checkbox("Show models below coverage threshold", value=True)
        if st.button("🚀 Start Comparison", type="primary"):
            st.session_state["run_comparison"] = True
    
    # Highlight the filtering strategy
    st.info(f"📋 **Strategy**: First filter models with Coverage 90% ≥ {min_coverage_90}%, "
            f"then rank by ELPD (higher = better)")
    
    # Run comparison
    if st.session_state.get("run_comparison", False):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, config):
            progress = (current + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Testing model {current + 1}/{total}...")
        
        try:
            # Generate configs
            if subset_configs:
                configs = generate_model_configs(subset=True, max_configs=max_configs)
            else:
                configs = generate_model_configs(subset=False, max_configs=max_configs)
            
            st.info(f"Testing {len(configs)} different model configurations...")
            
            # Run comparison with coverage filter
            with st.spinner("Comparing GP models with coverage filter..."):
                results_df, best_model, best_config = compare_gp_models_with_coverage_filter(
                    train_x, train_y, input_dim, configs, max_configs, min_coverage_90,
                    compute_loocv_preds, progress_callback
                )
            
            # Store results
            st.session_state["comparison_results"] = results_df
            st.session_state["best_model"] = best_model
            st.session_state["best_config"] = best_config
            
            # Update model in session state if we found a good one
            if best_model is not None:
                st.session_state["model"] = best_model
                # Also store LOOCV results for evaluation
                best_result = results_df.iloc[0]
                if best_result['LOOCV_Preds'] is not None:
                    st.session_state["loocv_preds"] = best_result['LOOCV_Preds']
                    st.session_state["loocv_truths"] = best_result['LOOCV_Truths']
                    st.session_state["elpd"] = best_result['ELPD']
            
            progress_bar.empty()
            status_text.empty()
            
            # Show results summary
            valid_models = results_df[results_df['Passes_Coverage_Filter'] == True]
            if len(valid_models) > 0:
                st.success(f"✅ Found {len(valid_models)} models passing coverage filter! "
                          f"Best ELPD: {valid_models.iloc[0]['ELPD']:.4f}")
            else:
                st.warning(f"⚠️ No models passed the {min_coverage_90}% coverage threshold. "
                          f"Consider lowering the threshold or checking your data.")
            
        except Exception as e:
            st.error(f"❌ Comparison failed: {e}")
            return
        
        finally:
            st.session_state["run_comparison"] = False
    
    # Display results
    if "comparison_results" in st.session_state:
        results_df = st.session_state["comparison_results"]
        
        # Split results into valid and invalid
        valid_results = results_df[results_df['Passes_Coverage_Filter'] == True]
        invalid_results = results_df[results_df['Passes_Coverage_Filter'] == False]
        
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Tested", len(results_df))
        with col2:
            st.metric("Passed Coverage Filter", len(valid_results))
        with col3:
            if len(valid_results) > 0:
                st.metric("Best ELPD (Valid)", f"{valid_results.iloc[0]['ELPD']:.4f}")
            else:
                st.metric("Best ELPD (Valid)", "None")
        with col4:
            if len(valid_results) > 0:
                st.metric("Best Coverage 90%", f"{valid_results['Coverage_90%'].max():.1f}%")
            else:
                st.metric("Best Coverage 90%", "N/A")
        
        # Display valid models first
        if len(valid_results) > 0:
            st.subheader("🏆 Models Passing Coverage Filter (Sorted by ELPD)")
            
            # Configuration columns for valid models
            config_cols = ['Config_ID', 'ELPD', 'Coverage_90%', 'Coverage_95%', 'LOOCV_MSE', 
                          'MLL_Score', 'Kernel', 'Nu', 'ARD', 'LS_Prior', 
                          'Outputscale_Config', 'Noise_Prior', 'Learned_Noise']
            
            valid_display = valid_results[config_cols]
            st.dataframe(valid_display, use_container_width=True)
            
            # Show best model details
            st.subheader("🏆 Best Valid Model Configuration")
            best_row = valid_results.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance:**")
                st.write(f"• ELPD: {best_row['ELPD']:.4f}")
                st.write(f"• Coverage 90%: {best_row['Coverage_90%']:.1f}%")
                st.write(f"• Coverage 95%: {best_row['Coverage_95%']:.1f}%")
                st.write(f"• LOOCV MSE: {best_row['LOOCV_MSE']:.6f}")
                st.write(f"• MLL Score: {best_row['MLL_Score']:.4f}")
                st.write(f"• Learned Noise: {best_row['Learned_Noise']}")
                
            with col2:
                st.markdown("**Configuration to Reproduce:**")
                st.write(f"• Kernel: {best_row['Kernel']}")
                if best_row['Nu'] != 'N/A':
                    st.write(f"• Nu: {best_row['Nu']}")
                st.write(f"• ARD: {best_row['ARD']}")
                st.write(f"• Lengthscale Prior: {best_row['LS_Prior']}")
                st.write(f"• Outputscale: {best_row['Outputscale_Config']}")
                st.write(f"• Noise Prior: {best_row['Noise_Prior']}")
        
        # Show invalid models if requested
        if show_failed and len(invalid_results) > 0:
            st.subheader(f"⚠️ Models Below {min_coverage_90}% Coverage Threshold")
            
            config_cols = ['Config_ID', 'ELPD', 'Coverage_90%', 'Coverage_95%', 'LOOCV_MSE', 
                          'Kernel', 'ARD', 'Status']
            invalid_display = invalid_results[config_cols]
            st.dataframe(invalid_display, use_container_width=True)


def show_model_evaluation():
    """Display comprehensive model evaluation metrics using your existing function"""
    
    if "comparison_results" not in st.session_state:
        st.info("Run model comparison first to see detailed evaluation metrics")
        return
    
    results_df = st.session_state["comparison_results"]
    successful_results = results_df[results_df['Status'] == 'Success']
    
    if len(successful_results) == 0:
        st.error("No successful models found for evaluation")
        return
    
    # Model selection
    st.subheader("📈 Model Evaluation")
    
    model_options = []
    for _, row in successful_results.iterrows():
        kernel_info = f"{row['Kernel']}"
        if row['Nu'] is not None:  # Changed from != 'N/A'
            kernel_info += f"(ν={row['Nu']})"
        coverage_status = "✅" if row['Passes_Coverage_Filter'] else "❌"
        model_options.append(f"{coverage_status} #{row['Config_ID']}: ELPD={row['ELPD']:.3f}, Cov90={row['Coverage_90%']:.1f}% - {kernel_info}")
    
    selected_idx = st.selectbox("Select model to evaluate:", range(len(model_options)), 
                               format_func=lambda x: model_options[x])
    
    selected_result = successful_results.iloc[selected_idx]
    
    # Check if we have LOOCV predictions
    if selected_result['LOOCV_Preds'] is None:
        st.warning("No LOOCV predictions available for this model")
        return
    
    # Extract predictions (use your existing data structure)
    y_pred = selected_result['LOOCV_Preds']
    y_true = selected_result['LOOCV_Truths']
    y_std = selected_result['LOOCV_Stds']  # We now have this from the enhanced function
    
    # Use YOUR existing evaluate_gp_predictions function
    df_eval = evaluate_gp_predictions(y_true, y_pred, y_std)
    
    # Add model-specific metrics from our comparison
    df_eval['ELPD'] = [selected_result['ELPD']]
    df_eval['ELPD_Diff'] = [selected_result['ELPD_Diff']]
    df_eval['MLL_Score'] = [selected_result['MLL_Score']]
    df_eval['Coverage_90%_Filter'] = [selected_result['Coverage_90%']]
    df_eval['Passes_Coverage_Filter'] = [selected_result['Passes_Coverage_Filter']]
    df_eval['Model_Config'] = [f"{selected_result['Kernel']} | ARD:{selected_result['ARD']}"]
    
    # Display metrics
    st.dataframe(df_eval, use_container_width=True)
    
    # Show configuration details
    st.subheader("📋 Selected Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Kernel:** {selected_result['Kernel']}")
        if selected_result['Nu'] is not None:  # Changed from != 'N/A'
            st.write(f"**Nu:** {selected_result['Nu']}")
        st.write(f"**ARD:** {selected_result['ARD']}")
        st.write(f"**Lengthscale Prior:** {selected_result['LS_Prior']}")
        
        # Coverage status
        if selected_result['Passes_Coverage_Filter']:
            st.success(f"✅ Passes coverage filter ({selected_result['Coverage_90%']:.1f}% ≥ threshold)")
        else:
            st.error(f"❌ Below coverage threshold ({selected_result['Coverage_90%']:.1f}% < threshold)")
    
    with col2:
        st.write(f"**Outputscale:** {selected_result['Outputscale_Config']}")
        st.write(f"**Noise Prior:** {selected_result['Noise_Prior']}")
        st.write(f"**Learned Lengthscale:** {selected_result['Learned_LS']}")
        st.write(f"**Learned Noise:** {selected_result['Learned_Noise']}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictions vs Truth")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predictions vs Truth (R² = {df_eval["R²"].iloc[0]:.3f})')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Uncertainty vs Residuals")
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_std, np.abs(residuals), alpha=0.6, s=50)
        ax.set_xlabel('Predicted Uncertainty (σ)')
        ax.set_ylabel('|Residuals|')
        ax.set_title(f'Uncertainty Calibration (Coverage 95% = {df_eval["Coverage (95%)"].iloc[0]:.1%})')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    return df_eval