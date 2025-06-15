import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from bo_utils.bo_model import fit_transfer_model, build_transfer_gp_model, create_transfer_model_from_source,extract_hyperparameters_from_model
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples
from properscoring import crps_gaussian
from bo_utils.bo_orthogonal_sampling import generate_orthogonal_samples, build_param_defs, build_parameter_space
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from bo_utils.bo_model import build_groundtruth_gp_model, fit_model, build_gp_model
from bo_utils.bo_orthogonal_sampling import generate_orthogonal_samples, build_param_defs, build_parameter_space
from streamlit_app.sampling_section import normalize_data_to_bounds
from bo_utils.bo_optimization import optimize_qEI, optimize_posterior_mean
from botorch.sampling.normal import SobolQMCNormalSampler
from bo_utils.bo_data import prepare_training_data
from bo_utils.bo_utils import rescale_single_point, rescale_batch
import random
import pandas as pd
from bo_utils.bo_validation import evaluate_gp_predictions, evaluate_high_performance_metrics
from botorch.utils.sampling import draw_sobol_samples
from properscoring import crps_gaussian
from bo_utils.bo_utils import sample_random_model_config
# import LinearRegression, RandomForestRegressor, GradientBoostingRegressor, MLPRegressor
from sklearn.linear_model import LinearRegression
# UPDATED SIMULATE_BO_LOOP FUNCTION
# UPDATED SIMULATE_BO_LOOP FUNCTION MIT TRANSFER LEARNING OPTION
import streamlit as st
def simulate_bo_loop(
    train_x_full,
    train_y_full,
    data,
    seed,
    bounds,
    acquisition_type="qEI",
    num_init=10,
    bo_steps=20,
    batch_size=1,
    num_restarts=20,
    raw_samples=2000,
    
    # GROUND TRUTH PARAMETER
    ground_truth_method="gbm",  # 'rf', 'gbm', 'nn', 'gp', 'linear', 'linear_with_outliers'
    ground_truth_config=None,
    noise_strategy="observation_noise",  # 'deterministic', 'observation_noise'
    noise_level=0.03,
    
    # TRANSFER LEARNING PARAMETER
    use_transfer_learning=False,  # Ob Transfer Learning verwendet werden soll
    source_model=None,           # Source-Modell für Transfer Learning
    prior_strength=2.0,          # Stärke der Transfer-Priors
    
    # BESTEHENDE PARAMETER
    bo_model_config=None,
    output_col_name="y",
    noise=None,  # Deprecated - use noise_level instead
    num_total=1000,
    limit=-100000,
):
    """
    Simulate a Bayesian Optimization loop where:
    1. A ground truth function (RF/GBM/NN/GP/Linear+Outliers) serves as the unknown objective function
    2. BO uses either:
       - A standard GP (use_transfer_learning=False)
       - A transfer GP based on source_model (use_transfer_learning=True)
    3. We evaluate how well BO discovers the optimum of the ground truth function
    """
    print(f"\n=== SIMULATE BO LOOP DEBUG ===")
    print(f"Ground Truth Method: {ground_truth_method}")
    print(f"Noise Strategy: {noise_strategy}")
    print(f"Noise Level: {noise_level}")
    print(f"Use Transfer Learning: {use_transfer_learning}")
    if use_transfer_learning:
        print(f"Prior Strength: {prior_strength}")
        print(f"Source Model: {'Provided' if source_model is not None else 'None'}")
    
    # Check for outlier configuration in ground truth config
    if ground_truth_config and ground_truth_config.get('outliers_enabled', False):
        print(f"Outliers Enabled: {ground_truth_config['outlier_fraction']*100:.1f}% of type '{ground_truth_config['outlier_type']}'")
    
    print(f"Data type: {type(data)}")
    print(f"Data is None: {data is None}")
    print(f"Output column name: {output_col_name}")
    
    # Validate Transfer Learning inputs
    if use_transfer_learning and source_model is None:
        raise ValueError("source_model must be provided when use_transfer_learning=True")
    
    # Validate inputs
    if train_x_full is None:
        raise ValueError("train_x_full cannot be None")
    if train_y_full is None:
        raise ValueError("train_y_full cannot be None")
    
    input_dim = train_x_full.shape[1]

    # Handle legacy noise parameter
    if noise is not None:
        noise_level = noise

    # Separate configurations for ground truth and BO models
    if ground_truth_config is None:
        ground_truth_config = sample_random_ground_truth_config_with_outliers(ground_truth_method, seed=seed)
    
    if bo_model_config is None:
        from bo_utils.bo_utils import sample_random_model_config
        bo_model_config = sample_random_model_config(seed=seed + 1000)  # Different seed

    print("Initializing Ground Truth Function...")
    print(f"Ground Truth method: {ground_truth_method}")
    print(f"Ground Truth configuration: {ground_truth_config}")
    print(f"BO Model configuration: {bo_model_config}")
    
    # Create ground truth function WITH OUTLIER SUPPORT
    gt_func = create_ground_truth_function_with_outliers(
        ground_truth_method, train_x_full, train_y_full, ground_truth_config
    )
    
    # Create ground truth evaluator
    gt_evaluator = GroundTruthEvaluator(gt_func, noise_strategy, noise_level)
    
    print(f"Ground Truth {ground_truth_method} ready (this is our 'unknown' objective function).")
    
    # Log outlier status
    if ground_truth_config.get('outliers_enabled', False):
        print(f"Outliers configured: {ground_truth_config['outlier_fraction']*100:.1f}% {ground_truth_config['outlier_type']} outliers with magnitude {ground_truth_config['outlier_magnitude']}")

    def evaluate_true_function(x, add_noise=True):
        """
        Evaluate ground truth function with configurable noise
        """
        return gt_evaluator.evaluate(x, add_noise=add_noise)

    # Initialize metrics collector
    metrics_collector = FlexibleMetricsCollector()

    X_test = generate_normalized_testgrid(input_dim, n_points=512)
    
    # Find the true optimum of the ground truth function
    true_optimum_x = find_true_optimum_general(gt_evaluator, input_dim, limit=limit)
    true_optimum_value = gt_evaluator.get_true_value(true_optimum_x).item()
    print(f"True optimum value: {true_optimum_value:.4f}")

    # Extract source hyperparameters if using transfer learning
    source_hyperparams = None
    if use_transfer_learning:
        source_hyperparams = extract_hyperparameters_from_model(source_model)
        print("Transfer Learning: Source hyperparameters extracted.")

    def create_bo_model(train_x, train_y):
        """
        Creates either a standard GP or a transfer GP based on settings
        """
        if use_transfer_learning:
            model, likelihood = build_transfer_gp_model(
                train_x, train_y, source_hyperparams, prior_strength
            )
        else:
            from bo_utils.bo_model import build_gp_model
            model, likelihood = build_gp_model(
                train_x, train_y, config=bo_model_config
            )
        return model, likelihood

    if acquisition_type != "Random":
        print("=== BAYESIAN OPTIMIZATION MODE ===")
        print("Generating initial design via Orthogonal Sampling...")
        
        # FIXED: Validate data before using it (unchanged from your original)
        if data is None:
            raise ValueError("Data cannot be None for Bayesian Optimization mode")
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")
        if data.empty:
            raise ValueError("Data cannot be empty")
        if output_col_name not in data.columns:
            raise ValueError(f"Output column '{output_col_name}' not found in data. Available columns: {list(data.columns)}")
        
        print(f"Data validation passed. Shape: {data.shape}, Columns: {list(data.columns)}")

        # FIXED: Safely build parameter definitions (unchanged from your original)
        try:
            param_defs = build_param_defs(data, bounds, output_col_name=output_col_name)
            print(f"Parameter definitions built successfully")
        except Exception as e:
            raise ValueError(f"Failed to build parameter definitions: {str(e)}")
        
        try:
            space, col_names = build_parameter_space(param_defs)
            print(f"Parameter space built successfully")
        except Exception as e:
            raise ValueError(f"Failed to build parameter space: {str(e)}")
        
        try:
            df_raw = generate_orthogonal_samples(space, num_init, col_names, seed=seed)
            for name, low, high, _ in param_defs:
                df_raw[name] = df_raw[name].clip(low, high)
            print(f"Orthogonal samples generated: {len(df_raw)} points.")
        except Exception as e:
            raise ValueError(f"Failed to generate orthogonal samples: {str(e)}")

        # FIXED: Validate df_raw before normalization (unchanged from your original)
        if df_raw is None:
            raise ValueError("Generated orthogonal samples are None")
        if not isinstance(df_raw, pd.DataFrame):
            raise ValueError(f"Orthogonal samples must be DataFrame, got {type(df_raw)}")
        if df_raw.empty:
            raise ValueError("Generated orthogonal samples are empty")

        try:
            df_norm = normalize_data_to_bounds(df_raw, param_defs)
            print(f"Normalized samples to bounds: {len(df_norm)} points.")
        except Exception as e:
            raise ValueError(f"Failed to normalize samples: {str(e)}")

        # FIXED: Validate df_norm before tensor conversion (unchanged from your original)
        if df_norm is None:
            raise ValueError("Normalized samples are None")
        if not isinstance(df_norm, pd.DataFrame):
            raise ValueError(f"Normalized samples must be DataFrame, got {type(df_norm)}")
        if df_norm.empty:
            raise ValueError("Normalized samples are empty")

        # Initial evaluation of the true function
        init_x = torch.tensor(df_norm.values, dtype=torch.double)
        init_y = evaluate_true_function(init_x)
        print(f"Initial {num_init} points evaluated on true function.")

        all_x = [init_x]
        all_y = [init_y]
        best_observed = [init_y.max().item()]
        best_posterior = []
        regret_values = []
        exploitative_regret_values = []

        # Initiale Berechnung mit dem gewählten Modelltyp
        initial_model, _ = create_bo_model(init_x, init_y)
        from bo_utils.bo_model import fit_model
        fit_model(initial_model)

        initial_regret_results = calculate_both_regrets(
            [init_x], [init_y], initial_model, true_optimum_value, evaluate_true_function, input_dim, limit
        )
        regret_values.append(initial_regret_results['regret'])
        exploitative_regret_values.append(initial_regret_results['exploitative_regret'])

        print("Starting BO loop...")
        for t in range(bo_steps):
            print(f"\nBO Iteration {t + 1} / {bo_steps}")

            # Current dataset
            train_x = torch.cat(all_x, dim=0)
            train_y = torch.cat(all_y, dim=0)

            # Fit BO model (either standard or transfer)
            bo_model, _ = create_bo_model(train_x, train_y)
            fit_model(bo_model)
            
            model_type = "Transfer GP" if use_transfer_learning else "Standard GP"
            print(f"{model_type} surrogate model trained on current observations.")

            # Select next point(s) using acquisition function
            from bo_utils.bo_optimization import optimize_qEI, optimize_qUCB

            if acquisition_type in ["qEI", "qNEI", "qGIBBON"]:
                new_x = optimize_qEI(
                    model=bo_model,
                    input_dim=input_dim,
                    best_f=train_y.max().item(),
                    batch_size=batch_size,
                    x_baseline=train_x,
                    acquisition_type=acquisition_type,
                    limit=limit,
                )
            elif acquisition_type == "qUCB":
                bounds_tensor = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], dtype=torch.double)
                new_x = optimize_qUCB(
                    model=bo_model,
                    input_dim=input_dim,
                    batch_size=batch_size,
                    bounds=bounds_tensor,
                    limit=limit,
                    beta=st.session_state['beta'] if 'beta' in st.session_state else 0.1,
                )
            else:
                raise ValueError(f"Unknown acquisition type: {acquisition_type}")
            print("New point(s) selected via acquisition function.")

            # Evaluate the true function at selected point(s)
            new_y = evaluate_true_function(new_x)
            all_x.append(new_x)
            all_y.append(new_y)

            # Update best observed and regret
            current_best_observed = torch.cat(all_y, dim=0).max().item()
            best_observed.append(current_best_observed)
            regret_results = calculate_both_regrets(
                all_x, all_y, bo_model, true_optimum_value, evaluate_true_function, input_dim, limit
            )

            # Extrahiere die Werte
            current_regret = regret_results['regret']
            current_exploitative_regret = regret_results['exploitative_regret']
            regret_values.append(current_regret)
            exploitative_regret_values.append(current_exploitative_regret)

            # Best posterior prediction
            best_idx = torch.cat(all_y, dim=0).argmax()
            best_x = torch.cat(all_x, dim=0)[best_idx]
            with torch.no_grad():
                best_posterior_value = bo_model.posterior(best_x.unsqueeze(0)).mean.detach().cpu().item()
            best_posterior.append(best_posterior_value)

            print(f"Best observed (noisy): {current_best_observed:.4f}")
            print(f"Model thinks optimum at: {regret_results['true_value_at_model_optimum']:.4f}")
            print(f"Best observed true value: {regret_results['true_value_at_best_observed']:.4f}")
            print(f"True optimum: {true_optimum_value:.4f}")
            print(f"Model regret: {current_regret:.4f}")
            print(f"Exploitative regret: {current_exploitative_regret:.4f}")

            # Evaluate surrogate model quality - UPDATED für general ground truth
            try:
                mse, r2, rmse_h = evaluate_model_vs_groundtruth_general(bo_model, gt_evaluator, X_test, threshold=0.9)
                crps_mean, coverage_h = evaluate_model_crps_general(bo_model, gt_evaluator, X_test, threshold=0.9)

                # Get true function values (without noise) for comparison
                y_true_clean = evaluate_true_function(train_x, add_noise=False).detach().cpu().numpy().flatten()
                with torch.no_grad():
                    pred_y = bo_model.posterior(train_x).mean.detach().cpu().numpy().flatten()
                    y_std = np.sqrt(bo_model.posterior(train_x).variance.detach().cpu().numpy().flatten())
                
                # FIXED: Safely evaluate metrics (unchanged from your original)
                metrics_df = None
                metrics_df2 = None
                
                try:
                    from bo_utils.bo_validation import evaluate_gp_predictions
                    metrics_df = evaluate_gp_predictions(y_true=y_true_clean, y_pred=pred_y, y_std=y_std)
                except Exception as e:
                    print(f"Warning: evaluate_gp_predictions failed: {e}")
                
                # Note: evaluate_high_performance_metrics might need updating for non-GP ground truth
                try:
                    from bo_utils.bo_validation import evaluate_high_performance_metrics
                    if ground_truth_method == 'gp':
                        metrics_df2 = evaluate_high_performance_metrics(bo_model, gt_evaluator.gt_func.gp_model, X_test, train_x, train_y, true_optimum_value)
                    else:
                        # Skip this for non-GP ground truth or adapt the function
                        metrics_df2 = None
                except Exception as e:
                    print(f"Warning: evaluate_high_performance_metrics failed: {e}")

                current_metrics = {
                            "iteration": t + 1,
                            "acquisition_type": acquisition_type,
                            "ground_truth_method": ground_truth_method,
                            "use_transfer_learning": use_transfer_learning,
                            "prior_strength": prior_strength if use_transfer_learning else None,
                            # NEW: Add outlier metadata
                            "outliers_enabled": ground_truth_config.get('outliers_enabled', False),
                            "outlier_fraction": ground_truth_config.get('outlier_fraction', 0.0),
                            "outlier_type": ground_truth_config.get('outlier_type', 'none'),
                            "outlier_magnitude": ground_truth_config.get('outlier_magnitude', 0.0),
                            # ... rest of existing metrics
                            "mse": mse,
                            "rmse_h": rmse_h, 
                            "r2": r2,
                            "crps": crps_mean,
                            "coverage_high_region": coverage_h,
                            "posterior_mean": best_posterior_value,
                            "true_best": true_optimum_value,
                            "observed_best": current_best_observed,
                            "true_value_at_best_observed": regret_results['true_value_at_best_observed'],
                            "true_value_at_model_optimum": regret_results['true_value_at_model_optimum'],
                            "regret": current_regret,
                            "exploitative_regret": current_exploitative_regret,
                        }
                
                # Rest of metrics extraction unchanged...
                if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
                    try:
                        if "CI Size (avg)" in metrics_df.columns:
                            current_metrics["CI_size_avg"] = metrics_df["CI Size (avg)"].iloc[0]
                        if "Coverage (95%)" in metrics_df.columns:
                            current_metrics["coverage_95"] = metrics_df["Coverage (95%)"].iloc[0]
                    except Exception as e:
                        print(f"Warning: Failed to extract metrics_df values: {e}")
                
                if metrics_df2 is not None and isinstance(metrics_df2, pd.DataFrame) and not metrics_df2.empty:
                    try:
                        for col in metrics_df2.columns:
                            if col in metrics_df2.columns:
                                current_metrics[f"hp_{col}"] = metrics_df2[col].iloc[0]
                    except Exception as e:
                        print(f"Warning: Failed to extract metrics_df2 values: {e}")
                
                # Add to collector
                metrics_collector.add_metrics(**current_metrics)

                print(f"Surrogate quality: MSE={mse:.4f}, R²={r2:.4f}, CRPS={crps_mean:.4f}")
                
            except Exception as e:
                print(f"Warning: Model evaluation failed for iteration {t+1}: {e}")
                # Add basic metrics even if evaluation fails
                current_metrics = {
                            "iteration": t + 1,
                            "acquisition_type": acquisition_type,
                            "ground_truth_method": ground_truth_method,
                            "use_transfer_learning": use_transfer_learning,
                            "prior_strength": prior_strength if use_transfer_learning else None,
                            # NEW: Add outlier metadata
                            "outliers_enabled": ground_truth_config.get('outliers_enabled', False),
                            "outlier_fraction": ground_truth_config.get('outlier_fraction', 0.0),
                            "outlier_type": ground_truth_config.get('outlier_type', 'none'),
                            "outlier_magnitude": ground_truth_config.get('outlier_magnitude', 0.0),
                            # ... rest of existing metrics
                            "mse": mse,
                            "rmse_h": rmse_h, 
                            "r2": r2,
                            "crps": crps_mean,
                            "coverage_high_region": coverage_h,
                            "posterior_mean": best_posterior_value,
                            "true_best": true_optimum_value,
                            "observed_best": current_best_observed,
                            "true_value_at_best_observed": regret_results['true_value_at_best_observed'],
                            "true_value_at_model_optimum": regret_results['true_value_at_model_optimum'],
                            "regret": current_regret,
                            "exploitative_regret": current_exploitative_regret,
                        }
                metrics_collector.add_metrics(**current_metrics)

        print("\nBO loop complete. Final surrogate model optimization...")
        final_bo_model, _ = create_bo_model(
            torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
        )
        fit_model(final_bo_model)

        # Find the best point according to the final surrogate
        # --- FINAL SURROGATE MODEL EVALUATION ---
        from bo_utils.bo_optimization import optimize_posterior_mean

        print("Optimizing surrogate posterior mean for final recommendation...")
        # Dichte Abtastung des Raums für einen robusten "Surrogate Optimum"
        candidate_points = generate_random_samples_unit_cube(input_dim, num_total, seed=seed + 99)
        with torch.no_grad():
            surrogate_preds = final_bo_model.posterior(candidate_points).mean.squeeze(-1)
        best_surrogate_idx = surrogate_preds.argmax()
        surrogate_optimum_x = candidate_points[best_surrogate_idx:best_surrogate_idx + 1]

        # Optional: Gradient-basierte Optimierung (falls implementiert)
        try:
            refined_x = optimize_posterior_mean(final_bo_model, input_dim, limit=limit)
            refined_val = final_bo_model.posterior(refined_x.unsqueeze(0)).mean.item()
            candidate_val = final_bo_model.posterior(surrogate_optimum_x).mean.item()
            if refined_val > candidate_val:
                surrogate_optimum_x = refined_x.unsqueeze(0)
        except Exception as e:
            print(f"Warning: Gradient-based surrogate optimization failed: {e}, using sampled optimum.")

        # Evaluation des finalen surrogate-predicted optimums an der "wahren" Zielfunktion
        true_value_at_surrogate_optimum = gt_evaluator.get_true_value(surrogate_optimum_x).item()
        regret_final = abs(true_optimum_value - true_value_at_surrogate_optimum)
        print(f"Final surrogate model optimum: {surrogate_optimum_x.cpu().numpy().flatten()}")
        print(f"True value at surrogate optimum: {true_value_at_surrogate_optimum:.4f}")
        print(f"Final regret: {regret_final:.4f}")

        # Speichere finale Ergebnisse in Collector
        metrics_collector.add_metrics(
            iteration=bo_steps + 1,
            acquisition_type=acquisition_type,
            ground_truth_method=ground_truth_method,
            use_transfer_learning=use_transfer_learning,
            prior_strength=prior_strength if use_transfer_learning else None,
            mse=np.nan,
            rmse_h=np.nan,
            r2=np.nan,
            crps=np.nan,
            coverage_high_region=np.nan,
            posterior_mean=np.nan,
            true_best=true_optimum_value,
            observed_best=np.max([y.max().item() for y in all_y]),
            true_value_at_best_observed=np.nan,
            true_value_at_model_optimum=true_value_at_surrogate_optimum,
            regret=regret_final,
            exploitative_regret=np.nan,
            CI_size_avg=np.nan,
            coverage_95=np.nan
        )

        # Alle Ergebnisse als DataFrame
        metrics_df = metrics_collector.to_dataframe()
        print("\nSimulation complete.")
        return {
            "all_x": all_x,
            "all_y": all_y,
            "best_observed": best_observed,
            "best_posterior": best_posterior,
            "regret": regret_values,
            "exploitative_regret": exploitative_regret_values,
            "metrics_df": metrics_df,
            "true_optimum_x": true_optimum_x,
            "true_optimum_value": true_optimum_value,
            "surrogate_optimum_x": surrogate_optimum_x,
            "true_value_at_surrogate_optimum": true_value_at_surrogate_optimum,
        }

    else:
        # RANDOM SAMPLING MODE - bleibt unverändert
        print("=== RANDOM SAMPLING MODE ===")
        print("Transfer Learning wird im Random Sampling Modus ignoriert.")
        from bo_utils.bo_optimization import optimize_posterior_mean
        all_x = []
        all_y = []
        best_observed = []
        best_posterior = []   # Wird für Random nicht genutzt, aber für Konsistenz trotzdem erstellt
        regret_values = []
        exploitative_regret_values = []

        # Starte mit initialen Random-Samples
        init_x = generate_random_samples_unit_cube(input_dim, num_init, seed=seed)
        init_y = evaluate_true_function(init_x)
        all_x.append(init_x)
        all_y.append(init_y)
        best_observed.append(init_y.max().item())

        # Berechne initialen Regret (wie bei BO) - verwende Standard GP
        from bo_utils.bo_model import build_gp_model, fit_model
        initial_model, _ = build_gp_model(init_x, init_y, config=bo_model_config)
        fit_model(initial_model)

        initial_regret_results = calculate_both_regrets(
            [init_x], [init_y], initial_model, true_optimum_value, evaluate_true_function, input_dim, limit
        )
        regret_values.append(initial_regret_results['regret'])
        exploitative_regret_values.append(initial_regret_results['exploitative_regret'])

        # Rest des Random Sampling Codes bleibt unverändert...
        # [Der gesamte Random Sampling Block bleibt wie im Original]
        
        # Beginne mit Random-Sampling-Schleife
        for t in range(bo_steps):
            print(f"\nRandom Iteration {t + 1} / {bo_steps}")
            
            # Ziehe nächsten Punkt(e) rein zufällig
            new_x = generate_random_samples_unit_cube(input_dim, batch_size, seed=seed + 2000 + t)
            new_y = evaluate_true_function(new_x)
            all_x.append(new_x)
            all_y.append(new_y)

            current_best_observed = torch.cat(all_y, dim=0).max().item()
            best_observed.append(current_best_observed)

            # Fit Surrogate für reine Dokumentation und Metrikberechnung (Standard GP)
            train_x = torch.cat(all_x, dim=0)
            train_y = torch.cat(all_y, dim=0)
            surrogate_model, _ = build_gp_model(train_x, train_y,config=bo_model_config)
            fit_model(surrogate_model)

            # Berechne Regret analog zu BO
            regret_results = calculate_both_regrets(
                all_x, all_y, surrogate_model, true_optimum_value, evaluate_true_function, input_dim, limit
            )
            regret_values.append(regret_results['regret'])
            exploitative_regret_values.append(regret_results['exploitative_regret'])

            # Best-posterior nur als Doku
            best_idx = torch.cat(all_y, dim=0).argmax()
            best_x_curr = torch.cat(all_x, dim=0)[best_idx]
            with torch.no_grad():
                best_posterior_value = surrogate_model.posterior(best_x_curr.unsqueeze(0)).mean.detach().cpu().item()
            best_posterior.append(best_posterior_value)

            # Metrics sammlung für Random (Standard GP wird verwendet)
            try:
                mse, r2, rmse_h = evaluate_model_vs_groundtruth_general(surrogate_model, gt_evaluator, X_test, threshold=0.9)
                crps_mean, coverage_h = evaluate_model_crps_general(surrogate_model, gt_evaluator, X_test, threshold=0.9)

                current_metrics = {
                    "iteration": t + 1,
                    "acquisition_type": acquisition_type,
                    "ground_truth_method": ground_truth_method,
                    "use_transfer_learning": False,  # Random verwendet kein Transfer
                    "prior_strength": None,
                    "outliers_enabled": ground_truth_config.get('outliers_enabled', False),
                    "outlier_fraction": ground_truth_config.get('outlier_fraction', 0.0),
                    "outlier_type": ground_truth_config.get('outlier_type', 'none'),
                    "outlier_magnitude": ground_truth_config.get('outlier_magnitude', 0.0),
                    "mse": mse,
                    "rmse_h": rmse_h,
                    "r2": r2,
                    "crps": crps_mean,
                    "coverage_high_region": coverage_h,
                    "posterior_mean": best_posterior_value,
                    "true_best": true_optimum_value,
                    "observed_best": current_best_observed,
                    "true_value_at_best_observed": regret_results['true_value_at_best_observed'],
                    "true_value_at_model_optimum": regret_results['true_value_at_model_optimum'],
                    "regret": regret_results['regret'],
                    "exploitative_regret": regret_results['exploitative_regret'],
                }

                metrics_collector.add_metrics(**current_metrics)
                print(f"Surrogate quality: MSE={mse:.4f}, R²={r2:.4f}, CRPS={crps_mean:.4f}")

            except Exception as e:
                print(f"Warning: Model evaluation failed for iteration {t+1}: {e}")
                current_metrics = {
                    "iteration": t + 1,
                    "acquisition_type": acquisition_type,
                    "ground_truth_method": ground_truth_method,
                    "use_transfer_learning": False,
                    "prior_strength": None,
                    "outliers_enabled": ground_truth_config.get('outliers_enabled', False),
                    "outlier_fraction": ground_truth_config.get('outlier_fraction', 0.0),
                    "outlier_type": ground_truth_config.get('outlier_type', 'none'),
                    "outlier_magnitude": ground_truth_config.get('outlier_magnitude', 0.0),
                    "mse": np.nan,
                    "rmse_h": np.nan,
                    "r2": np.nan,
                    "crps": np.nan,
                    "coverage_high_region": np.nan,
                    "posterior_mean": best_posterior_value,
                    "true_best": true_optimum_value,
                    "observed_best": current_best_observed,
                    "regret": regret_results['regret'],
                    "exploitative_regret": regret_results['exploitative_regret'],
                }
                metrics_collector.add_metrics(**current_metrics)

        # Nach allen Iterationen: bestes Surrogate-Optimum bestimmen (Standard GP)
        final_surrogate_model, _ = build_gp_model(
            torch.cat(all_x, dim=0), torch.cat(all_y, dim=0), config=bo_model_config
        )
        fit_model(final_surrogate_model)

        candidate_points = generate_random_samples_unit_cube(input_dim, num_total, seed=seed + 99)
        with torch.no_grad():
            surrogate_preds = final_surrogate_model.posterior(candidate_points).mean.squeeze(-1)
        best_surrogate_idx = surrogate_preds.argmax()
        surrogate_optimum_x = candidate_points[best_surrogate_idx:best_surrogate_idx + 1]
        
        try:
            refined_x = optimize_posterior_mean(final_surrogate_model, input_dim, limit=limit)
            refined_val = final_surrogate_model.posterior(refined_x.unsqueeze(0)).mean.item()
            candidate_val = final_surrogate_model.posterior(surrogate_optimum_x).mean.item()
            if refined_val > candidate_val:
                surrogate_optimum_x = refined_x.unsqueeze(0)
        except Exception as e:
            print(f"Warning: Gradient-based surrogate optimization failed: {e}, using sampled optimum.")

        true_value_at_surrogate_optimum = gt_evaluator.get_true_value(surrogate_optimum_x).item()
        regret_final = abs(true_optimum_value - true_value_at_surrogate_optimum)

        print(f"Final surrogate model optimum: {surrogate_optimum_x.cpu().numpy().flatten()}")
        print(f"True value at surrogate optimum: {true_value_at_surrogate_optimum:.4f}")
        print(f"Final regret: {regret_final:.4f}")

        # Speichere finale Ergebnisse in Collector
        metrics_collector.add_metrics(
            iteration=bo_steps + 1,
            acquisition_type=acquisition_type,
            ground_truth_method=ground_truth_method,
            use_transfer_learning=use_transfer_learning,
            prior_strength=prior_strength if use_transfer_learning else None,
            # NEW: Add outlier metadata to final metrics
            outliers_enabled=ground_truth_config.get('outliers_enabled', False),
            outlier_fraction=ground_truth_config.get('outlier_fraction', 0.0),
            outlier_type=ground_truth_config.get('outlier_type', 'none'),
            outlier_magnitude=ground_truth_config.get('outlier_magnitude', 0.0),
            # ... rest of final metrics
            mse=np.nan,
            rmse_h=np.nan,
            r2=np.nan,
            crps=np.nan,
            coverage_high_region=np.nan,
            posterior_mean=np.nan,
            true_best=true_optimum_value,
            observed_best=np.max([y.max().item() for y in all_y]),
            true_value_at_best_observed=np.nan,
            true_value_at_model_optimum=true_value_at_surrogate_optimum,
            regret=regret_final,
            exploitative_regret=np.nan,
            CI_size_avg=np.nan,
            coverage_95=np.nan
        )

        # Alle Ergebnisse als DataFrame
        metrics_df = metrics_collector.to_dataframe()
        print("\nSimulation complete.")
        return {
            "all_x": all_x,
            "all_y": all_y,
            "best_observed": best_observed,
            "best_posterior": best_posterior,
            "regret": regret_values,
            "exploitative_regret": exploitative_regret_values,
            "metrics_df": metrics_df,
            "true_optimum_x": true_optimum_x,
            "true_optimum_value": true_optimum_value,
            "surrogate_optimum_x": surrogate_optimum_x,
            "true_value_at_surrogate_optimum": true_value_at_surrogate_optimum,
            # NEW: Add ground truth configuration info to results
            "ground_truth_config": ground_truth_config,
            "outliers_enabled": ground_truth_config.get('outliers_enabled', False),
        }
# NEUE GROUND TRUTH KLASSEN
class GroundTruthFunction:
    """Base class for different ground truth functions"""
    def __init__(self, train_x, train_y, config=None):
        self.train_x = train_x
        self.train_y = train_y
        self.config = config or {}
        self.is_fitted = False
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self, x):
        """Returns deterministic prediction (no uncertainty)"""
        raise NotImplementedError
class LinearModelGroundTruth(GroundTruthFunction):
    """Einfache lineare Modell Ground Truth basierend auf deiner Analyse"""
    def __init__(self, train_x, train_y, config=None):
        super().__init__(train_x, train_y, config)
        # Bestimme automatisch ob 7 oder 8 Features basierend auf deiner Analyse
        self.T_mean = train_x[:, 0].mean().item()
        self.use_8_features = config.get('use_8_features', False) if config else False
    
    def fit(self):
        # Erstelle Features wie in deiner Response Surface Analyse
        T, RT, CO, CC = self.train_x[:, 0], self.train_x[:, 1], self.train_x[:, 2], self.train_x[:, 3]
        T_centered = T - self.T_mean
        TT_centered = T_centered ** 2  
        RTTT = T * RT
        RT_squared = RT ** 2
        
        if self.use_8_features:
            RT_T_squared = RT * T_centered**2
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared, RT_T_squared], dim=1)
        else:
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared], dim=1)
        
        # Sklearn LinearRegression
        X = features.cpu().numpy()
        y = self.train_y.cpu().numpy().flatten()
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"Linear GT fitted with R² = {self.model.score(X, y):.3f}")
    
    def predict(self, x):
        # Gleiche Feature-Erstellung für Prediction
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        T, RT, CO, CC = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        T_centered = T - self.T_mean
        TT_centered = T_centered ** 2  
        RTTT = T * RT
        RT_squared = RT ** 2
        
        if self.use_8_features:
            RT_T_squared = RT * T_centered**2
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared, RT_T_squared], dim=1)
        else:
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared], dim=1)
        
        X = features.cpu().numpy()
        pred = self.model.predict(X)
        return torch.tensor(pred, dtype=torch.float64).view(-1, 1)

class RandomForestGroundTruth(GroundTruthFunction):
    def __init__(self, train_x, train_y, config=None):
        super().__init__(train_x, train_y, config)
        self.default_config = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.config = {**self.default_config, **(config or {})}
    
    def fit(self):
        X = self.train_x.cpu().numpy()
        y = self.train_y.cpu().numpy().flatten()
        
        self.model = RandomForestRegressor(**self.config)
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:  # ← HIER IST ES SCHON
            x = x.view(-1, x.shape[-1])
        
        X = x.cpu().numpy()
        pred = self.model.predict(X)
        return torch.tensor(pred, dtype=torch.float64).view(-1, 1)

class GradientBoostingGroundTruth(GroundTruthFunction):
    def __init__(self, train_x, train_y, config=None):
        super().__init__(train_x, train_y, config)
        self.default_config = {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'random_state': 42
        }
        self.config = {**self.default_config, **(config or {})}
    
    def fit(self):
        X = self.train_x.cpu().numpy()
        y = self.train_y.cpu().numpy().flatten()
        
        self.model = GradientBoostingRegressor(**self.config)
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:  # ← HIER IST ES SCHON
            x = x.view(-1, x.shape[-1])
        
        X = x.cpu().numpy()
        pred = self.model.predict(X)
        return torch.tensor(pred, dtype=torch.float64).view(-1, 1)

class NeuralNetworkGroundTruth(GroundTruthFunction):
    def __init__(self, train_x, train_y, config=None):
        super().__init__(train_x, train_y, config)
        self.default_config = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'random_state': 42
        }
        self.config = {**self.default_config, **(config or {})}
    
    def fit(self):
        X = self.train_x.cpu().numpy()
        y = self.train_y.cpu().numpy().flatten()
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPRegressor(**self.config)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:  # ← HIER IST ES SCHON
            x = x.view(-1, x.shape[-1])
        
        X = x.cpu().numpy()
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return torch.tensor(pred, dtype=torch.float64).view(-1, 1)

class GPGroundTruth(GroundTruthFunction):
    """Wrapper for existing GP ground truth mit observation noise"""
    def __init__(self, gp_model):
        self.gp_model = gp_model
        self.is_fitted = True
    
    def fit(self):
        # Already fitted
        pass
    
    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:  # ← HIER IST ES SCHON
            x = x.view(-1, x.shape[-1])
        
        self.gp_model.eval()
        with torch.no_grad():
            # Deterministischer posterior mean (NICHT sampling!)
            return self.gp_model.posterior(x).mean

# FACTORY FUNCTIONS
def create_ground_truth_function(method, train_x, train_y, config=None):
    """Factory function to create different ground truth functions"""
    if method == 'rf' or method == 'random_forest':
        return RandomForestGroundTruth(train_x, train_y, config)
    elif method == 'gbm' or method == 'gradient_boosting':
        return GradientBoostingGroundTruth(train_x, train_y, config)
    elif method == 'nn' or method == 'neural_network':
        return NeuralNetworkGroundTruth(train_x, train_y, config)
    elif method == 'gp':
        # Keep GP option for comparison - aber mit deterministischem predict!
        from bo_utils.bo_model import build_groundtruth_gp_model, fit_model
        model, _ = build_groundtruth_gp_model(train_x, train_y, train_x.shape[1], config)
        fit_model(model)
        return GPGroundTruth(model)
    elif method == 'linear':
        return LinearModelGroundTruth(train_x, train_y, config)
    else:
        raise ValueError(f"Unknown method: {method}")

def sample_random_ground_truth_config(method, seed=None):
    """Generate random configurations for different ground truth methods"""
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'rf' or method == 'random_forest':
        return {
            # Moderate increase in trees for stability
            'n_estimators': np.random.choice([300, 400, 500]),
            
            # Moderate depth - still allow some complexity
            'max_depth': np.random.choice([4, 5, 6, 7]),  # Not too shallow!
            
            # Moderate sample requirements - balanced smoothness
            'min_samples_split': np.random.choice([15, 20, 25]),  # Reasonable
            'min_samples_leaf': np.random.choice([8, 10, 12]),    # Not too high
            
            # Balanced feature selection
            'max_features': np.random.choice([0.6, 0.7, 0.8]),   # Allow more features
            'min_impurity_decrease': np.random.choice([0.001, 0.003, 0.005]),  # Moderate threshold
            
            'bootstrap': True,
            'random_state': seed or 42
        }
    elif method == 'linear':
        return {
            'use_8_features': True
        }
    elif method == 'gbm' or method == 'gradient_boosting':
        return {
         # Sehr viele schwache Learner
        'n_estimators': 1000,
        
        # NUR STUMPS - jeder Baum macht nur eine simple Entscheidung
        'max_depth': 1,  # Nur Stumps!
        
        # Extrem langsame Learning Rate für sanfte Updates
        'learning_rate': 0.005,  # Sehr, sehr langsam
        
        # Extrem hohe Sample-Anforderungen
        'min_samples_split': 100,  # Braucht 100+ Samples für Split
        'min_samples_leaf': 50,    # Jedes Blatt muss 50+ Samples haben
        
        # Sehr starke Regularisierung
        'subsample': 0.6,  # Nur 60% der Samples pro Baum
        
        # Sehr hoher Threshold für Splits
        'min_impurity_decrease': 0.01,  # Nur bei signifikanter Verbesserung
        
        # Frühes Stoppen bei Plateau
        'validation_fraction': 0.2,
        'n_iter_no_change': 100,  # Sehr geduldig
        'tol': 1e-5,
        
        'random_state': seed or 42}
        
    elif method == 'nn' or method == 'neural_network':
        # Moderate networks - not too small
        hidden_layer_sizes = [
            (64,), (100,), (64, 32), (100, 50), (80, 40, 20)  # Reasonable sizes
        ]
        
        return {
            'hidden_layer_sizes': hidden_layer_sizes[np.random.randint(len(hidden_layer_sizes))],
            'activation': 'tanh',  # Smooth activation
            'learning_rate_init': np.random.choice([0.001, 0.005]),  # Moderate LR
            'alpha': np.random.choice([0.01, 0.05, 0.1]),  # Moderate regularization
            'max_iter': 1500,  # Sufficient iterations
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 30,
            'tol': 1e-4,
            'random_state': seed or 42
        }
    elif method == 'gp':
        # Use your existing sample_random_model_config for GPs
        from bo_utils.bo_utils import sample_random_model_config
        return sample_random_model_config(seed=seed)
    else:
        return {}

# GROUND TRUTH EVALUATOR
class GroundTruthEvaluator:
    def __init__(self, ground_truth_func, noise_strategy="observation_noise", noise_level=0.03):
        self.gt_func = ground_truth_func
        self.noise_strategy = noise_strategy
        self.noise_level = noise_level
        
        if not self.gt_func.is_fitted:
            self.gt_func.fit()
    
    def _normalize_input(self, x):
        """IDENTISCHE Input-Normalisierung für beide Methoden"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        return x
    
    def _base_predict(self, x):
        """EINZIGE Quelle für Predictions - eliminiert Inkonsistenzen"""
        x = self._normalize_input(x)
        return self.gt_func.predict(x)
    
    def evaluate(self, x, add_noise=True):
        """Evaluate ground truth function at x"""
        pred = self._base_predict(x)  # IDENTISCHE BASIS
        
        if self.noise_strategy == "deterministic":
            return pred
            
        elif self.noise_strategy == "observation_noise":
            if add_noise and self.noise_level > 0:
                noise = torch.randn_like(pred) * self.noise_level
                pred += noise
            return pred
            
        else:
            raise ValueError(f"Unknown noise strategy: {self.noise_strategy}")
    
    def get_true_value(self, x):
        """Get true function value without any noise - IDENTISCH zu evaluate(..., add_noise=False)"""
        return self._base_predict(x)  # IDENTISCHE BASIS

# UPDATED FIND_TRUE_OPTIMUM FUNCTION
def find_true_optimum_general(gt_evaluator, input_dim, limit=-100000, n_candidates=500000):
    """
    Das wahre Optimum ist immer die reine Funktion ohne Outlier-Zufälle
    """
    
    # Wenn Outliers enabled sind, finde das funktionale Optimum
    if hasattr(gt_evaluator.gt_func, 'outlier_config') and gt_evaluator.gt_func.outlier_config.get('enabled', False):
        
        print("Finding functional optimum (outlier-free)...")
        
        # Deaktiviere Outliers temporär
        original_enabled = gt_evaluator.gt_func.outlier_config['enabled']
        gt_evaluator.gt_func.outlier_config['enabled'] = False
        
        try:
            candidates = generate_random_samples_unit_cube(input_dim, n_candidates, seed=42)
            values = gt_evaluator.get_true_value(candidates).squeeze(-1)
            best_idx = values.argmax()
            best_x = candidates[best_idx:best_idx+1]
            best_value = values[best_idx].item()
            
            print(f"Functional optimum: {best_value:.4f} (outlier-free)")
            return best_x
            
        finally:
            # Outliers wieder aktivieren
            gt_evaluator.gt_func.outlier_config['enabled'] = original_enabled
    
    # Standard Fall: Normale Optimum-Suche
    candidates = generate_random_samples_unit_cube(input_dim, n_candidates, seed=42)
    values = gt_evaluator.get_true_value(candidates).squeeze(-1)
    best_idx = values.argmax()
    best_x = candidates[best_idx:best_idx+1]
    best_value = values[best_idx].item()
    
    print(f"True optimum: {best_value:.4f}")
    return best_x

# UPDATED EVALUATION FUNCTIONS
def evaluate_model_vs_groundtruth_general(model, gt_evaluator, X_test, threshold=0.9):
    """
    Compare posterior mean of BO surrogate model with ANY ground-truth function on test grid
    """
    model.eval()
    
    with torch.no_grad():
        mu_surrogate = model.posterior(X_test).mean.squeeze(-1).cpu().numpy()
        
    # Get true function values (without noise)
    mu_true = gt_evaluator.get_true_value(X_test).squeeze(-1).cpu().numpy()
    
    # Define high-value region based on true function
    f_max = np.max(mu_true)
    mask = mu_true > f_max * threshold
        
    # Calculate metrics
    mse = mean_squared_error(mu_true, mu_surrogate)
    r2 = r2_score(mu_true, mu_surrogate)
    
    # RMSE in high-value region (where it matters most for optimization)
    if np.any(mask):
        rmse_h = np.sqrt(mean_squared_error(mu_true[mask], mu_surrogate[mask]))
    else:
        rmse_h = np.nan

    return mse, r2, rmse_h

def evaluate_model_crps_general(model, gt_evaluator, X_test, threshold=0.9):
    """
    Calculate CRPS between BO surrogate model and ANY ground-truth function on test grid
    """
    model.eval()

    with torch.no_grad():
        posterior_surrogate = model.posterior(X_test)
        mu_surrogate = posterior_surrogate.mean.squeeze(-1).cpu().numpy()
        sigma_surrogate = posterior_surrogate.variance.sqrt().squeeze(-1).cpu().numpy()
        
    # Get true function values (without noise)
    y_true = gt_evaluator.get_true_value(X_test).squeeze(-1).cpu().numpy()

    # High-value region based on true function
    f_max = np.max(y_true)
    mask = y_true > f_max * threshold

    # Calculate CRPS (how well surrogate uncertainty matches true function)
    crps = crps_gaussian(y_true, mu_surrogate, sigma_surrogate)
    crps_mean = np.mean(crps)

    # Coverage in high-value region (95% CI)
    if np.any(mask):
        lower = mu_surrogate[mask] - 1.96 * sigma_surrogate[mask]
        upper = mu_surrogate[mask] + 1.96 * sigma_surrogate[mask]
        within_ci = (y_true[mask] >= lower) & (y_true[mask] <= upper)
        coverage_h = np.mean(within_ci)
    else:
        coverage_h = np.nan

    return crps_mean, coverage_h

# UTILITY FUNCTIONS
def generate_normalized_testgrid(d, n_points=512):
    """
    Generate consistent test grid in normalized space [0,1]^d
    """
    if d <= 2:
        # Grid for low dimensions
        grid_points = int(n_points ** (1 / d))
        grids = [torch.linspace(0.0, 1.0, grid_points) for _ in range(d)]
        mesh = torch.meshgrid(*grids, indexing="ij")
        X_test = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    else:
        # Sobol sampling for higher dimensions
        unit_bounds = torch.tensor([[0.0] * d, [1.0] * d])
        X_test = draw_sobol_samples(bounds=unit_bounds, n=n_points, q=1).squeeze(1)
    
    return X_test.double()

def generate_random_samples_unit_cube(
    input_dim,
    num_points,
    seed=None,
    constraint_dims=None,
    constraint_coeffs=None,
    constraint_limit=None,
    max_attempts=10000,
):
    """
    Generate random samples in [0, 1]^d space with optional linear constraint
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    samples = []
    attempts = 0
    rejected = 0

    while len(samples) < num_points and attempts < max_attempts:
        point = np.random.rand(input_dim)
        
        # Check constraint if specified
        if constraint_dims and constraint_limit is not None:
            if constraint_coeffs is None:
                # Simple sum
                constraint_value = point[constraint_dims].sum()
            else:
                # Weighted sum
                constraint_value = np.dot(point[constraint_dims], constraint_coeffs)
                
            if constraint_value > constraint_limit:
                rejected += 1
                attempts += 1
                continue
                
        samples.append(point)
        attempts += 1

    if len(samples) < num_points:
        print(f"Warning: Only {len(samples)} valid samples generated after {max_attempts} attempts.")
        print(f"Rejected {rejected} samples due to constraints.")

    return torch.tensor(np.array(samples), dtype=torch.double)

class FlexibleMetricsCollector:
    def __init__(self):
        self.metrics_data = []
    
    def add_metrics(self, iteration=None, **kwargs):
        metric_dict = {}
        if iteration is not None:
            metric_dict["iteration"] = iteration
        metric_dict.update(kwargs)
        self.metrics_data.append(metric_dict)
    
    def to_dataframe(self):
        if not self.metrics_data:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_data)

def calculate_both_regrets(all_x, all_y, bo_model, true_optimum_value, evaluate_true_function, input_dim, limit=-100000):
    """
    Berechne beide Regret-Metriken:
    1. Model Understanding Regret: Wie gut versteht das Modell wo das Optimum liegt?
    2. Exploitative Regret: Wie gut ist der beste bereits gesehene Punkt?
    """
    
    # 1. Model Understanding Regret (Hauptmetrik - wird "regret" genannt)
    try:
        # Wo denkt das Modell ist das Optimum?
        model_optimum_x = optimize_posterior_mean(bo_model, input_dim, limit=limit)
        true_value_at_model_optimum = evaluate_true_function(model_optimum_x.unsqueeze(0), add_noise=False).item()
        model_regret = abs(true_optimum_value - true_value_at_model_optimum)
    except Exception as e:
        print(f"Warning: Could not calculate model regret: {e}")
        model_regret = np.nan
        true_value_at_model_optimum = np.nan
    
    # 2. Exploitative Regret (wie gut ist der beste beobachtete Punkt?)
    best_idx = torch.cat(all_y, dim=0).argmax()
    best_x = torch.cat(all_x, dim=0)[best_idx]
    true_value_at_best_observed = evaluate_true_function(best_x.unsqueeze(0), add_noise=False).item()
    exploitative_regret = abs(true_optimum_value - true_value_at_best_observed)
    
    return {
        'regret': model_regret,  # Dein Ansatz als Hauptmetrik
        'exploitative_regret': exploitative_regret,  # Mein Ansatz als Zusatzmetrik
        'true_value_at_model_optimum': true_value_at_model_optimum,
        'true_value_at_best_observed': true_value_at_best_observed,
        'best_observed_x': best_x
    }

# LEGACY FUNCTIONS (for compatibility)
def find_true_optimum(truth_model, input_dim, limit=-100000, n_candidates=1000):
    """
    Find the true optimum of the ground truth function using dense sampling + optimization.
    Legacy function for GP-only ground truth.
    """
    truth_model.eval()
    
    # Generate many candidate points
    candidates = generate_random_samples_unit_cube(input_dim, n_candidates, seed=42)
    
    with torch.no_grad():
        # Evaluate ground truth at all candidates
        posterior = truth_model.posterior(candidates)
        values = posterior.mean.squeeze(-1)
        
        # Find the best candidate
        best_idx = values.argmax()
        best_x = candidates[best_idx:best_idx+1]  # Keep batch dimension
        
    # Optionally: refine with gradient-based optimization
    try:
        refined_x = optimize_posterior_mean(truth_model, input_dim, limit=limit)
        with torch.no_grad():
            refined_val = truth_model.posterior(refined_x.unsqueeze(0)).mean
            candidate_val = truth_model.posterior(best_x).mean
        
        if refined_val > candidate_val:
            best_x = refined_x.unsqueeze(0)
    except Exception as e:
        print(f"Warning: Gradient-based refinement failed: {e}, using best candidate.")
    
    return best_x

def evaluate_model_vs_groundtruth(model, ground_truth_model, X_test, threshold=0.9):
    """
    Compare posterior mean of BO surrogate model with ground-truth GP on test grid
    Legacy function for GP-only ground truth.
    """
    model.eval()
    ground_truth_model.eval()
    
    with torch.no_grad():
        mu_surrogate = model.posterior(X_test).mean.squeeze(-1).cpu().numpy()
        mu_true = ground_truth_model.posterior(X_test).mean.squeeze(-1).cpu().numpy()
        
        # Define high-value region based on true function
        f_max = np.max(mu_true)
        mask = mu_true > f_max * threshold
        
    # Calculate metrics
    mse = mean_squared_error(mu_true, mu_surrogate)
    r2 = r2_score(mu_true, mu_surrogate)
    
    # RMSE in high-value region (where it matters most for optimization)
    if np.any(mask):
        rmse_h = np.sqrt(mean_squared_error(mu_true[mask], mu_surrogate[mask]))
    else:
        rmse_h = np.nan

    return mse, r2, rmse_h

def evaluate_model_crps(model, ground_truth_model, X_test, threshold=0.9):
    """
    Calculate CRPS between BO surrogate model and ground-truth GP on test grid
    Legacy function for GP-only ground truth.
    """
    model.eval()
    ground_truth_model.eval()

    with torch.no_grad():
        posterior_surrogate = model.posterior(X_test)
        posterior_true = ground_truth_model.posterior(X_test)
        
        mu_surrogate = posterior_surrogate.mean.squeeze(-1).cpu().numpy()
        sigma_surrogate = posterior_surrogate.variance.sqrt().squeeze(-1).cpu().numpy()
        y_true = posterior_true.mean.squeeze(-1).cpu().numpy()  # "True" function values

        # High-value region based on true function
        f_max = np.max(y_true)
        mask = y_true > f_max * threshold

    # Calculate CRPS (how well surrogate uncertainty matches true function)
    crps = crps_gaussian(y_true, mu_surrogate, sigma_surrogate)
    crps_mean = np.mean(crps)

    # Coverage in high-value region (95% CI)
    if np.any(mask):
        lower = mu_surrogate[mask] - 1.96 * sigma_surrogate[mask]
        upper = mu_surrogate[mask] + 1.96 * sigma_surrogate[mask]
        within_ci = (y_true[mask] >= lower) & (y_true[mask] <= upper)
        coverage_h = np.mean(within_ci)
    else:
        coverage_h = np.nan

    return crps_mean, coverage_h


    """
    Berechne beide Regret-Metriken:
    1. Model Understanding Regret: Wie gut versteht das Modell wo das Optimum liegt?
    2. Exploitative Regret: Wie gut ist der beste bereits gesehene Punkt?
    """
    
    # 1. Model Understanding Regret (Hauptmetrik - wird "regret" genannt)
    try:
        # Wo denkt das Modell ist das Optimum?
        model_optimum_x = optimize_posterior_mean(bo_model, input_dim, limit=limit)
        true_value_at_model_optimum = evaluate_true_function(model_optimum_x.unsqueeze(0), add_noise=False).item()
        model_regret = abs(true_optimum_value - true_value_at_model_optimum)
    except Exception as e:
        print(f"Warning: Could not calculate model regret: {e}")
        model_regret = np.nan
        true_value_at_model_optimum = np.nan
    
    # 2. Exploitative Regret (wie gut ist der beste beobachtete Punkt?)
    best_idx = torch.cat(all_y, dim=0).argmax()
    best_x = torch.cat(all_x, dim=0)[best_idx]
    true_value_at_best_observed = evaluate_true_function(best_x.unsqueeze(0), add_noise=False).item()
    exploitative_regret = abs(true_optimum_value - true_value_at_best_observed)
    
    return {
        'regret': model_regret,  # Dein Ansatz als Hauptmetrik
        'exploitative_regret': exploitative_regret,  # Mein Ansatz als Zusatzmetrik
        'true_value_at_model_optimum': true_value_at_model_optimum,
        'true_value_at_best_observed': true_value_at_best_observed,
        'best_observed_x': best_x
    }


class EnhancedLinearModelGroundTruth:
    """
    Enhanced Linear Model with TRUE RANDOM outliers for fair BO evaluation
    """
    def __init__(self, train_x, train_y, config=None):
        self.train_x = train_x
        self.train_y = train_y
        self.config = config or {}
        self.is_fitted = False
        
        # Feature setup
        self.T_mean = train_x[:, 0].mean().item()
        self.use_8_features = self.config.get('use_8_features', False)
        
        # Outlier configuration
        self.outlier_config = self._setup_outlier_config()
        
        # *** NEW: True random outlier tracking ***
        self.evaluation_counter = 0  # Track how many evaluations we've done
        self.outlier_decisions = {}  # Cache outlier decisions per evaluation
        
    def _setup_outlier_config(self):
        """Setup outlier configuration from config dict"""
        outlier_config = {
            'enabled': self.config.get('outliers_enabled', False),
            'fraction': self.config.get('outlier_fraction', 0.05),
            'type': self.config.get('outlier_type', 'true_random'),  # NEW: true_random
            'magnitude': self.config.get('outlier_magnitude', 3.0),
            'seed': self.config.get('outlier_seed', 42),
            'generation_method': self.config.get('outlier_generation_method', 'additive_noise'),
        }
        return outlier_config
    
    def fit(self):
        """Fit linear model on training data"""
        features = self._create_features(self.train_x)
        
        X = features.cpu().numpy()
        y = self.train_y.cpu().numpy().flatten()
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Calculate residual statistics for outlier generation
        y_pred = self.model.predict(X)
        self.residual_std = np.std(y - y_pred)
        self.y_range = np.max(y) - np.min(y)
        
        print(f"Linear GT fitted with R² = {self.model.score(X, y):.3f}")
        print(f"Residual std: {self.residual_std:.4f}, Y range: {self.y_range:.4f}")
        
        if self.outlier_config['enabled']:
            print(f"Outliers enabled: {self.outlier_config['fraction']*100:.1f}% "
                  f"TRUE RANDOM outliers with magnitude {self.outlier_config['magnitude']}")
    
    def _create_features(self, x):
        """Create polynomial features from input"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        T, RT, CO, CC = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        T_centered = T - self.T_mean
        TT_centered = T_centered ** 2  
        RTTT = T * RT
        RT_squared = RT ** 2
        
        if self.use_8_features:
            RT_T_squared = RT * T_centered**2
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared, RT_T_squared], dim=1)
        else:
            features = torch.stack([RT, CO, CC, T_centered, TT_centered, RTTT, RT_squared], dim=1)
        
        return features
    
    def _should_be_outlier_true_random(self, evaluation_id):
        """
        TRUE RANDOM outlier decision - independent of x-coordinates
        Each evaluation gets a fresh random decision
        """
        if not self.outlier_config['enabled']:
            return False
        
        # Check if we already decided for this evaluation
        if evaluation_id in self.outlier_decisions:
            return self.outlier_decisions[evaluation_id]
        
        # Make a fresh random decision
        # Use evaluation_id + seed for reproducibility across runs
        eval_seed = self.outlier_config['seed'] + evaluation_id
        eval_rng = np.random.RandomState(eval_seed)
        
        is_outlier = eval_rng.random() < self.outlier_config['fraction']
        
        # Cache the decision
        self.outlier_decisions[evaluation_id] = is_outlier
        
        return is_outlier
    
    def _generate_outlier_value_true_random(self, base_prediction, evaluation_id):
        """Generate an outlier value using true random approach"""
        eval_seed = self.outlier_config['seed'] + evaluation_id + 10000
        eval_rng = np.random.RandomState(eval_seed)
        
        magnitude = self.outlier_config['magnitude']
        method = self.outlier_config['generation_method']
        
        outlier_types = {
            'additive_noise': lambda: base_prediction + magnitude * self.residual_std * eval_rng.randn(),
            'multiplicative': lambda: base_prediction * (1 + magnitude * 0.3 * eval_rng.randn()),
            'extreme_shift': lambda: base_prediction + magnitude * self.y_range * (eval_rng.choice([-1, 1]) * eval_rng.uniform(0.2, 0.5)),
            'random_value': lambda: eval_rng.uniform(
                base_prediction - magnitude * self.residual_std,
                base_prediction + magnitude * self.residual_std
            ),
            'systematic_bias': lambda: base_prediction + magnitude * self.residual_std * eval_rng.choice([-1, 1])
        }
        
        # Choose outlier generation method
        if method not in outlier_types:
            method = 'additive_noise'
            
        outlier_value = outlier_types[method]()
        
        return outlier_value
    
    def predict(self, x):
        """Predict with TRUE RANDOM outliers"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Normalize input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        # Base prediction
        features = self._create_features(x)
        X = features.cpu().numpy()
        base_predictions = self.model.predict(X)
    
        # Add TRUE RANDOM outliers if enabled
        if self.outlier_config['enabled']:
            final_predictions = []
            
            for i, base_pred in enumerate(base_predictions):
                # Each point gets a unique evaluation ID
                evaluation_id = self.evaluation_counter
                self.evaluation_counter += 1
                
                if self._should_be_outlier_true_random(evaluation_id):
                    outlier_value = self._generate_outlier_value_true_random(base_pred, evaluation_id)
                    final_predictions.append(outlier_value)
                    # Optional: Log outlier for debugging
                    # print(f"Outlier at eval {evaluation_id}: {base_pred:.4f} → {outlier_value:.4f}")
                else:
                    final_predictions.append(base_pred)
            
            predictions = np.array(final_predictions)
        else:
            predictions = base_predictions
        
        return torch.tensor(predictions, dtype=torch.float64).view(-1, 1)

    def reset_evaluation_counter(self):
        """Reset counter for new simulation run"""
        self.evaluation_counter = 0
        self.outlier_decisions = {}

# Updated configuration function for true random outliers
def sample_random_linear_config_with_true_random_outliers(seed=None):
    """Generate random configuration for linear model with TRUE RANDOM outliers"""
    if seed is not None:
        np.random.seed(seed)
    
    config = {
        'use_8_features': True,
        
        # Outlier configuration - now with true random option
        'outliers_enabled': np.random.choice([True, False], p=[0.7, 0.3]),
    }
    
    if config['outliers_enabled']:
        config.update({
            'outlier_fraction': np.random.choice([0.02, 0.05, 0.08, 0.1]),
            'outlier_type': 'true_random',  # Always use true random
            'outlier_magnitude': np.random.choice([1.5, 2.0, 3.0, 4.0]),
            'outlier_generation_method': np.random.choice([
                'additive_noise', 'multiplicative', 'extreme_shift'
            ]),
            'outlier_seed': seed or 42,
        })
    
    return config

def sample_random_linear_config_with_outliers(seed=None):
    """Generate random configuration for linear model with outliers"""
    if seed is not None:
        np.random.seed(seed)
    
    config = {
        'use_8_features': True,
        
        # Outlier configuration
        'outliers_enabled': np.random.choice([True, False], p=[0.7, 0.3]),  # 70% chance of outliers
    }
    
    if config['outliers_enabled']:
        config.update({
            'outlier_fraction': np.random.choice([0.02, 0.05, 0.08, 0.1]),  # 2-10% outliers
            'outlier_type': np.random.choice(['random', 'systematic', 'extreme_values']),
            'outlier_magnitude': np.random.choice([1.5, 2.0, 3.0, 4.0]),  # Outlier strength
            'outlier_generation_method': np.random.choice([
                'additive_noise', 'multiplicative', 'extreme_shift', 'systematic_bias'
            ]),
            'outlier_seed': seed or 42,
        })
        
        # Add specific regions for systematic outliers
        if config['outlier_type'] == 'systematic':
            config['outlier_regions'] = [
                {
                    'bounds': [(0.0, 0.3), (0.0, 0.3), (0.7, 1.0), (0.0, 1.0)],  # Corner region
                    'probability': config['outlier_fraction'] * 3
                },
                {
                    'bounds': [(0.7, 1.0), (0.7, 1.0), (0.0, 0.3), (0.0, 1.0)],  # Opposite corner
                    'probability': config['outlier_fraction'] * 2
                }
            ]
    
    return config

# Update the main factory function
def create_ground_truth_function_with_outliers(method, train_x, train_y, config=None):
    """Enhanced factory function with outlier support for linear models"""
    if method == 'linear' or method == 'linear_with_outliers':
        return EnhancedLinearModelGroundTruth(train_x, train_y, config)
    elif method == 'rf' or method == 'random_forest':
        return RandomForestGroundTruth(train_x, train_y, config)
    elif method == 'gbm' or method == 'gradient_boosting':
        return GradientBoostingGroundTruth(train_x, train_y, config)
    elif method == 'nn' or method == 'neural_network':
        return NeuralNetworkGroundTruth(train_x, train_y, config)
    elif method == 'gp':
        from bo_utils.bo_model import build_groundtruth_gp_model, fit_model
        model, _ = build_groundtruth_gp_model(train_x, train_y, train_x.shape[1], config)
        fit_model(model)
        return GPGroundTruth(model)
    else:
        raise ValueError(f"Unknown method: {method}")

def sample_random_ground_truth_config_with_outliers(method, seed=None):
    """Enhanced config sampling with outlier support"""
    if method == 'linear' or method == 'linear_with_outliers':
        return sample_random_linear_config_with_outliers(seed)
    elif method == 'rf' or method == 'random_forest':
        return sample_random_rf_config(seed)  # Your existing function
    elif method == 'gbm' or method == 'gradient_boosting':
        return sample_random_gbm_config(seed)  # Your existing function
    elif method == 'nn' or method == 'neural_network':
        return sample_random_nn_config(seed)  # Your existing function
    elif method == 'gp':
        from bo_utils.bo_utils import sample_random_model_config
        return sample_random_model_config(seed=seed)
    else:
        return {}

# Example usage and testing
def test_linear_model_with_outliers():
    """Test the enhanced linear model with different outlier configurations"""
    
    # Generate test data
    torch.manual_seed(42)
    n_points = 200
    train_x = torch.rand(n_points, 4, dtype=torch.double)  # 4D input
    
    # Create "true" linear relationship
    T, RT, CO, CC = train_x[:, 0], train_x[:, 1], train_x[:, 2], train_x[:, 3]
    T_mean = T.mean()
    T_centered = T - T_mean
    
    # Your polynomial features
    true_coeffs = torch.tensor([0.5, -0.3, 0.8, 1.2, -0.4, 0.6, 0.2], dtype=torch.double)
    features = torch.stack([RT, CO, CC, T_centered, T_centered**2, T*RT, RT**2], dim=1)
    
    train_y = (features @ true_coeffs).unsqueeze(1) + 0.02 * torch.randn(n_points, 1, dtype=torch.double)
    
    print("Testing Linear Model with Different Outlier Configurations:")
    print("=" * 60)
    
    configs = [
        {'outliers_enabled': False, 'use_8_features': False},
        {
            'outliers_enabled': True,
            'outlier_fraction': 0.05,
            'outlier_type': 'random',
            'outlier_magnitude': 2.0,
            'outlier_generation_method': 'additive_noise',
            'outlier_seed': 42,
            'use_8_features': False
        },
        {
            'outliers_enabled': True,
            'outlier_fraction': 0.08,
            'outlier_type': 'systematic',
            'outlier_magnitude': 3.0,
            'outlier_generation_method': 'extreme_shift',
            'outlier_seed': 123,
            'use_8_features': False,
            'outlier_regions': [
                {'bounds': [(0.0, 0.3), (0.0, 0.3), (0.7, 1.0), (0.0, 1.0)], 'probability': 0.15}
            ]
        },
        {
            'outliers_enabled': True,
            'outlier_fraction': 0.03,
            'outlier_type': 'extreme_values',
            'outlier_magnitude': 4.0,
            'outlier_generation_method': 'systematic_bias',
            'outlier_seed': 456,
            'use_8_features': False
        }
    ]
    
    test_x = torch.rand(50, 4, dtype=torch.double)
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        # Create and fit model
        model = EnhancedLinearModelGroundTruth(train_x, train_y, config)
        model.fit()
        
        # Test predictions
        predictions = model.predict(test_x)
        
        # Count outliers in predictions (rough estimate)
        base_model = EnhancedLinearModelGroundTruth(train_x, train_y, {'outliers_enabled': False})
        base_model.fit()
        base_predictions = base_model.predict(test_x)
        
        diff = torch.abs(predictions - base_predictions)
        outlier_threshold = 2 * diff.std()
        estimated_outliers = (diff > outlier_threshold).sum().item()
        
        print(f"  Estimated outliers in test set: {estimated_outliers}/{len(test_x)} "
              f"({estimated_outliers/len(test_x)*100:.1f}%)")
        print(f"  Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        print(f"  Base prediction range: [{base_predictions.min().item():.4f}, {base_predictions.max().item():.4f}]")

if __name__ == "__main__":
    # Test the enhanced linear model
    test_linear_model_with_outliers()
    
    print("\n" + "="*60)
    print("Random Configuration Examples:")
    print("="*60)
    
    # Show some random configurations
    for i in range(3):
        config = sample_random_linear_config_with_outliers(seed=i*100)
        print(f"\nRandom Config {i+1}:")
        for key, value in config.items():
            print(f"  {key}: {value}")



def analyze_outlier_impact(df_results):
    """
    Analyze the impact of outliers on BO performance
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results dataframe with outlier metadata columns
    
    Returns:
    --------
    analysis : dict
        Analysis results comparing performance with/without outliers
    """
    if 'outliers_enabled' not in df_results.columns:
        return {"message": "No outlier data available for analysis"}
    
    # Separate results by outlier status
    no_outliers = df_results[df_results['outliers_enabled'] == False]
    with_outliers = df_results[df_results['outliers_enabled'] == True]
    
    if len(no_outliers) == 0 or len(with_outliers) == 0:
        return {"message": "Need both outlier and non-outlier results for comparison"}
    
    analysis = {}
    
    # Performance comparison
    if 'regret' in df_results.columns:
        # Get final regrets
        if 'iteration' in df_results.columns:
            final_no_outliers = no_outliers.groupby(['seed', 'acquisition_type'])['regret'].last()
            final_with_outliers = with_outliers.groupby(['seed', 'acquisition_type'])['regret'].last()
        else:
            final_no_outliers = no_outliers['regret']
            final_with_outliers = with_outliers['regret']
        
        analysis['performance_impact'] = {
            'mean_regret_no_outliers': final_no_outliers.mean(),
            'mean_regret_with_outliers': final_with_outliers.mean(),
            'performance_degradation_%': ((final_with_outliers.mean() - final_no_outliers.mean()) / final_no_outliers.mean()) * 100,
            'std_no_outliers': final_no_outliers.std(),
            'std_with_outliers': final_with_outliers.std(),
            'robustness_impact_%': ((final_with_outliers.std() - final_no_outliers.std()) / final_no_outliers.std()) * 100
        }
    
    # Outlier configuration analysis
    if len(with_outliers) > 0:
        outlier_configs = with_outliers.groupby(['outlier_type', 'outlier_fraction', 'outlier_magnitude']).size().reset_index()
        outlier_configs.columns = ['type', 'fraction', 'magnitude', 'count']
        analysis['outlier_configurations'] = outlier_configs.to_dict('records')
        
        # Impact by outlier type
        if 'regret' in with_outliers.columns:
            type_impact = with_outliers.groupby('outlier_type')['regret'].agg(['mean', 'std', 'count']).round(4)
            analysis['impact_by_type'] = type_impact.to_dict('index')
    
    # Transfer learning resilience to outliers
    if 'use_transfer_learning' in df_results.columns:
        transfer_resilience = {}
        
        for transfer_status in [False, True]:
            subset = df_results[df_results['use_transfer_learning'] == transfer_status]
            
            if len(subset) > 0:
                no_out = subset[subset['outliers_enabled'] == False]['regret'].mean() if 'regret' in subset.columns else np.nan
                with_out = subset[subset['outliers_enabled'] == True]['regret'].mean() if 'regret' in subset.columns else np.nan
                
                method_name = 'Transfer Learning' if transfer_status else 'Standard BO'
                transfer_resilience[method_name] = {
                    'no_outliers_regret': no_out,
                    'with_outliers_regret': with_out,
                    'degradation_%': ((with_out - no_out) / no_out * 100) if not np.isnan(no_out) and not np.isnan(with_out) else np.nan
                }
        
        analysis['transfer_learning_resilience'] = transfer_resilience
    
    return analysis

def create_outlier_visualization(df_results):
    """
    Create visualizations comparing performance with/without outliers
    """
    import matplotlib.pyplot as plt
    
    if 'outliers_enabled' not in df_results.columns or 'regret' not in df_results.columns:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Box plot comparison
    ax = axes[0, 0]
    outlier_data = []
    outlier_labels = []
    
    for outlier_status in [False, True]:
        subset = df_results[df_results['outliers_enabled'] == outlier_status]
        if len(subset) > 0 and 'regret' in subset.columns:
            if 'iteration' in subset.columns:
                final_regrets = subset.groupby(['seed', 'acquisition_type'])['regret'].last()
            else:
                final_regrets = subset['regret']
            
            outlier_data.append(final_regrets)
            outlier_labels.append('Without Outliers' if not outlier_status else 'With Outliers')
    
    if len(outlier_data) == 2:
        bp = ax.boxplot(outlier_data, labels=outlier_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        
        ax.set_title('Final Regret: With vs Without Outliers')
        ax.set_ylabel('Final Regret')
        ax.grid(True, alpha=0.3)
    
    # 2. Performance by outlier type
    ax = axes[0, 1]
    outlier_subset = df_results[df_results['outliers_enabled'] == True]
    
    if len(outlier_subset) > 0 and 'outlier_type' in outlier_subset.columns:
        type_performance = outlier_subset.groupby('outlier_type')['regret'].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(type_performance['outlier_type'], type_performance['mean'], 
                     yerr=type_performance['std'], alpha=0.7, capsize=5)
        ax.set_title('Performance by Outlier Type')
        ax.set_ylabel('Mean Final Regret')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 3. Transfer learning resilience
    ax = axes[1, 0]
    if 'use_transfer_learning' in df_results.columns:
        resilience_data = []
        resilience_labels = []
        colors = []
        
        for transfer_status in [False, True]:
            for outlier_status in [False, True]:
                subset = df_results[
                    (df_results['use_transfer_learning'] == transfer_status) & 
                    (df_results['outliers_enabled'] == outlier_status)
                ]
                
                if len(subset) > 0:
                    if 'iteration' in subset.columns:
                        final_regrets = subset.groupby(['seed', 'acquisition_type'])['regret'].last()
                    else:
                        final_regrets = subset['regret']
                    
                    resilience_data.append(final_regrets.mean())
                    
                    method = 'Transfer' if transfer_status else 'Standard'
                    outlier_info = 'w/ Outliers' if outlier_status else 'Clean'
                    resilience_labels.append(f'{method}\n{outlier_info}')
                    
                    # Color coding
                    if transfer_status and outlier_status:
                        colors.append('#d62728')  # Transfer + Outliers (red)
                    elif transfer_status:
                        colors.append('#ff7f0e')  # Transfer only (orange) 
                    elif outlier_status:
                        colors.append('#2ca02c')  # Standard + Outliers (green)
                    else:
                        colors.append('#1f77b4')  # Standard clean (blue)
        
        if resilience_data:
            bars = ax.bar(resilience_labels, resilience_data, color=colors, alpha=0.7)
            ax.set_title('Transfer Learning Outlier Resilience')
            ax.set_ylabel('Mean Final Regret')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    # 4. Convergence comparison
    ax = axes[1, 1]
    if 'iteration' in df_results.columns:
        for outlier_status in [False, True]:
            subset = df_results[df_results['outliers_enabled'] == outlier_status]
            
            if len(subset) > 0:
                convergence_stats = subset.groupby('iteration')['regret'].agg(['mean', 'std']).reset_index()
                
                label = 'With Outliers' if outlier_status else 'Without Outliers'
                color = '#ff7f0e' if outlier_status else '#1f77b4'
                
                ax.plot(convergence_stats['iteration'], convergence_stats['mean'], 
                       'o-', label=label, color=color, linewidth=2)
                ax.fill_between(convergence_stats['iteration'], 
                               convergence_stats['mean'] - convergence_stats['std'],
                               convergence_stats['mean'] + convergence_stats['std'],
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Regret')
        ax.set_title('Convergence: With vs Without Outliers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Example usage and integration with existing analysis
def enhanced_outlier_analysis_ui(df_results):
    """
    Streamlit UI component for outlier impact analysis
    """
    import streamlit as st
    
    st.markdown("### 🎯 Outlier Impact Analysis")
    
    # Check if outlier data is available
    if 'outliers_enabled' not in df_results.columns:
        st.warning("⚠️ No outlier information found in results data")
        return
    
    # Basic outlier statistics
    outlier_stats_cols = st.columns(4)
    
    with outlier_stats_cols[0]:
        n_total = len(df_results)
        st.metric("Total Runs", n_total)
    
    with outlier_stats_cols[1]:
        n_clean = len(df_results[df_results['outliers_enabled'] == False])
        st.metric("Clean Runs", n_clean)
    
    with outlier_stats_cols[2]:
        n_outliers = len(df_results[df_results['outliers_enabled'] == True])
        st.metric("Outlier Runs", n_outliers)
    
    with outlier_stats_cols[3]:
        if n_clean > 0 and n_outliers > 0:
            st.metric("Comparison Available", "✅ Yes")
        else:
            st.metric("Comparison Available", "❌ No")
    
    if n_clean > 0 and n_outliers > 0:
        # Detailed analysis
        analysis = analyze_outlier_impact(df_results)
        
        if 'performance_impact' in analysis:
            st.markdown("**📊 Performance Impact Summary:**")
            
            impact = analysis['performance_impact']
            impact_cols = st.columns(3)
            
            with impact_cols[0]:
                degradation = impact['performance_degradation_%']
                st.metric(
                    "Performance Degradation", 
                    f"{degradation:+.1f}%",
                    delta=f"Outliers {'hurt' if degradation > 0 else 'helped'} performance"
                )
            
            with impact_cols[1]:
                robustness_impact = impact['robustness_impact_%']
                st.metric(
                    "Robustness Impact", 
                    f"{robustness_impact:+.1f}%",
                    delta=f"Variability {'increased' if robustness_impact > 0 else 'decreased'}"
                )
            
            with impact_cols[2]:
                clean_regret = impact['mean_regret_no_outliers']
                outlier_regret = impact['mean_regret_with_outliers']
                st.metric(
                    "Regret Comparison", 
                    f"{outlier_regret:.4f}",
                    delta=f"{outlier_regret - clean_regret:+.4f} vs clean"
                )
        
        # Transfer learning resilience
        if 'transfer_learning_resilience' in analysis:
            st.markdown("**🔄 Transfer Learning Outlier Resilience:**")
            
            resilience = analysis['transfer_learning_resilience']
            
            for method, stats in resilience.items():
                if not np.isnan(stats['degradation_%']):
                    degradation = stats['degradation_%']
                    
                    if degradation < 5:
                        resilience_level = "🛡️ Highly Resilient"
                        color = "success"
                    elif degradation < 15:
                        resilience_level = "⚖️ Moderately Resilient" 
                        color = "warning"
                    else:
                        resilience_level = "⚠️ Sensitive to Outliers"
                        color = "error"
                    
                    if color == "success":
                        st.success(f"**{method}**: {resilience_level} ({degradation:+.1f}% degradation)")
                    elif color == "warning":
                        st.warning(f"**{method}**: {resilience_level} ({degradation:+.1f}% degradation)")
                    else:
                        st.error(f"**{method}**: {resilience_level} ({degradation:+.1f}% degradation)")
        
        # Visualizations
        st.markdown("**📈 Outlier Impact Visualizations:**")
        
        fig = create_outlier_visualization(df_results)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        
        # Outlier type breakdown
        if 'impact_by_type' in analysis:
            st.markdown("**🎲 Impact by Outlier Type:**")
            
            type_df = pd.DataFrame(analysis['impact_by_type']).T
            type_df.index.name = 'Outlier Type'
            st.dataframe(type_df.round(4), use_container_width=True)
    
    else:
        if n_outliers == 0:
            st.info("💡 No outlier runs found. Enable outliers in the simulation configuration to see outlier impact analysis.")
        elif n_clean == 0:
            st.info("💡 No clean runs found. Include non-outlier runs for comparison analysis.")