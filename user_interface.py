import streamlit as st
import pandas as pd
import torch
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_validation import loocv_gp_custom
from bo_utils.bo_optimization import optimize_qEI, optimize_qUCB
from botorch.test_functions import Hartmann
from bo_utils.bo_data import prepare_training_data
from config import COLUMNS_CONFIG
from bo_utils.bo_data import load_csv_experiment_data
from bo_utils.bo_model import build_gp_model, fit_model, prepare_training_tensors
from config import INPUT_DIM
from config import MODEL_CONFIG
from bo_utils.bo_run_grid_search import run_grid_search
from bo_utils.bo_validation import log_loocv_result
from botorch.acquisition.utils import prune_inferior_points
from bo_utils.bo_test import run_bo_test
from bo_utils.bo_utils import append_and_display_experiment
from bo_utils.bo_utils import rescale_single_point, rescale_batch
from bo_utils.bo_orthogonal_sampling import build_parameter_space, generate_orthogonal_samples, normalize_data, analyze_sample
from streamlit_app.sampling_section import show_sampling_ui
from streamlit_app.model_setup import model_setup_tab
from streamlit_app.sidebar_config import get_model_config_from_sidebar
from streamlit_app.filtering import (
    remove_loocv_outliers,
    remove_points_by_noise_influence,
    remove_by_looph_loss,
    remove_by_gradient_pairs,
)
from streamlit_app.training_loocv import loocv_and_training_section
import numpy as np

st.title("Bayesian Optimization for CO₂ Sequestration")

from streamlit_app.sampling_section import show_sampling_ui
df_raw, df_norm, param_defs = show_sampling_ui()




# Upload data
st.sidebar.header("Upload Experiment Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", 
    type=["csv", "xlsx", "xls", "tsv", "txt"]
)

if uploaded_file:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['xlsx', 'xls']:
            # Excel files
            data = pd.read_excel(uploaded_file)
            
        elif file_extension in ['csv', 'tsv', 'txt']:
            # Text files - auto-detect separator
            uploaded_file.seek(0)
            sample = uploaded_file.read(1024).decode('utf-8')
            uploaded_file.seek(0)
            
            # Count potential separators in sample
            separators = [';', ',', '\t', '|', ' ']
            sep_counts = {sep: sample.count(sep) for sep in separators}
            detected_sep = max(sep_counts, key=sep_counts.get)
            
            # Fallback to comma if no clear separator
            if sep_counts[detected_sep] == 0:
                detected_sep = ','
            
            data = pd.read_csv(uploaded_file, sep=detected_sep)
            st.sidebar.info(f"Detected separator: '{detected_sep}'")
        
        else:
            st.error("❌ Unsupported file format")
            data = None
        
        if data is not None:
            st.subheader("Uploaded Data")
            st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Column selection
            st.write("**Select columns to use:**")
            selected_columns = st.multiselect(
                "Choose columns for analysis:",
                options=data.columns.tolist(),
                default=data.columns.tolist(),
                key="column_selector"
            )
            
            if selected_columns:
                # Filter data to selected columns
                data = data[selected_columns]
                st.success(f"✅ Using {len(selected_columns)} of {len(data.columns)} columns")
            else:
                st.warning("⚠️ No columns selected")
                data = None
            
            if data is not None:
                st.dataframe(data, use_container_width=True)
                
                # Quick data info
                with st.expander("Data Info"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Column types:**")
                        for col, dtype in data.dtypes.items():
                            st.write(f"• {col}: {dtype}")
                    
                    with col2:
                        st.write("**Data summary:**")
                        st.write(f"• File: {uploaded_file.name}")
                        st.write(f"• Size: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
                        
                        missing_count = data.isnull().sum().sum()
                        if missing_count > 0:
                            st.warning(f"⚠️ {missing_count} missing values")
                            missing_cols = data.isnull().sum()[data.isnull().sum() > 0]
                            for col, count in missing_cols.items():
                                st.write(f"  • {col}: {count}")
                        else:
                            st.success("✅ No missing values")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.write("Try checking the file format or separator.")
        data = None
else:
    data = None
# Ensure data is loaded

st.session_state["data"] = data
### Parameter Configuration Section ###
if 'data' in locals() and data is not None:
    
    st.subheader("Select Output Column")
    output_col_name = st.selectbox(
        "Choose target variable:",
        options=data.columns,
        key="output_col"
    )

    if output_col_name:
        st.write(f"Selected: **{output_col_name}** ({len(data)} data points)")

        st.subheader("Parameter Bounds")
        param_columns = [col for col in data.columns if col != output_col_name]
        
        if param_columns:
            bounds = []
            
            for col in param_columns:
                col_min = float(data[col].min())
                col_max = float(data[col].max())
                
                st.write(f"**{col}** (range: {col_min:.3f} - {col_max:.3f})")
                
                min_col, max_col = st.columns(2)
                with min_col:
                    min_val = st.number_input(
                        "Min", value=col_min, key=f"min_{col}", format="%.6f"
                    )
                with max_col:
                    max_val = st.number_input(
                        "Max", value=col_max, key=f"max_{col}", format="%.6f"
                    )
                
                if min_val >= max_val:
                    st.error(f"❌ Min must be < Max for '{col}'")
                
                bounds.append((min_val, max_val))
            
            # Summary
            bounds_summary = dict(zip(param_columns, bounds))
            st.write("**Current bounds:**", bounds_summary)
            
        else:
            st.warning("⚠️ No parameters available")

else:
    st.info("Upload data to configure parameters")


# --- MODEL CONFIGURATION ---
from bo_utils.bo_validation import compute_loocv_k_diagnostics
if 'data' in locals():
    MODEL_CONFIG, acq_function, BATCH_SIZE, iterations, initial_points, noise_std = get_model_config_from_sidebar(data, MODEL_CONFIG)
    
    # --- TRAINING UND LOOCV ---
    train_x_new, train_y_new = loocv_and_training_section(uploaded_file=uploaded_file, bounds=bounds, output_col_name=output_col_name, MODEL_CONFIG=MODEL_CONFIG, COLUMNS_CONFIG=COLUMNS_CONFIG, data=data)
   
   # --- TRAINING --- 
    if train_x_new is None or train_y_new is None:
        st.warning("No data available for training.")
    else:
        input_dim = train_x_new.shape[1]
        model, likelihood = build_gp_model(train_x_new, train_y_new, MODEL_CONFIG)
        fit_model(model)

        preds_new, truths_new, mse_new, ave_noise_new = loocv_gp_custom(
            train_x_new, train_y_new, input_dim, model_config=MODEL_CONFIG
        )
        influence_scores, elpd, elpd_dif= compute_loocv_k_diagnostics(train_x_new, train_y_new, input_dim, model_config=MODEL_CONFIG)


        # Save model and LOOCV results
        st.session_state["model"] = model
        st.session_state["train_x"] = train_x_new
        st.session_state["train_y"] = train_y_new
        st.session_state["bounds"] = bounds
        st.session_state["loocv_preds"] = preds_new
        st.session_state["loocv_truths"] = truths_new
        st.session_state["loocv_mse"] = mse_new
        st.session_state["loocv_noise"] = ave_noise_new
        

        st.success(f"LOOCV completed. Final MSE/MAE: {mse_new:.4f} / {mse_new ** 0.5:.4f}")
        #st.success(f"ELPD: {elpd:.4f}")
        st.session_state["elpd"] = elpd
        #st.success(f"ELPD Difference: {elpd_dif:.4f}")

        from bo_utils.bo_utils import compute_pairwise_gradients
        df_gradient_pairs = compute_pairwise_gradients(train_x_new, train_y_new, top_k=10)
        with st.expander("Top 10 Point Pairs with Highest Local Gradient"):
            st.dataframe(df_gradient_pairs)

from bo_utils.bo_validation import show_model_evaluation
if "loocv_preds" in st.session_state:
    show_model_evaluation()

import matplotlib.pyplot as plt
# --- DISPLAY LOOCV RESULTS BLOCK ---
from streamlit_app.loocv_results import show_loocv_results
if "loocv_preds" in st.session_state:
    show_loocv_results(bounds)

# # --- NOISE INFLUENCE ANALYSE ---
# st.subheader("Leave-One-Out Influence on Learned Noise (σ) in %")
# from streamlit_app.noise_delta import show_noise_delta
# if "model" in st.session_state and "train_x" in st.session_state:
#     show_noise_delta(st.session_state, MODEL_CONFIG)

import torch
import pandas as pd

# --- MODEL COMPARISON ---
from streamlit_app.model_comparison import streamlit_model_comparison
streamlit_model_comparison()

# #--- POSTERIOR PREDICTIONS ---
# if "model" in st.session_state and "train_x" in st.session_state:
#     from streamlit_app.posterior_pred import display_posterior_predictions
#     display_posterior_predictions(st.session_state)



if st.sidebar.checkbox("Ca Constraint"):
    limit = -2
else:
    limit = -1000000

st.session_state["limit"] = limit

# from bo_utils.bo_optimization import find_promising_uncertain_point
# from bo_utils.bo_utils import rescale_single_point
# # --- FIND PROMISING UNCERTAIN POINT ---
# if "model" in st.session_state and "train_x" in st.session_state:
#     st.subheader("Find Promising Uncertain Point")
#     ratio = st.slider("Ratio of qEI to Uncertainty", min_value=0.0, max_value=1.0, value=0.5, key="ratio")
#     if st.button("Find Promising Point"):
#         try:
#             # Unpack the tuple correctly
#             candidates, info = find_promising_uncertain_point(
#                 model=st.session_state["model"],
#                 train_x=st.session_state["train_x"],
#                 train_y=st.session_state["train_y"],
#                 bounds=bounds,
#                 limit=limit,
#                 best_n=3 , # Start with just 1 point
#                 ratio = ratio
#             )
            
#             # Rescale to original space
#             rescaled_point = rescale_single_point(candidates, bounds)
            
#             st.success("✅ Promising uncertain point found!")
#             st.write(f"**Point (original space):** {rescaled_point.numpy()}")
            
#             if isinstance(info, dict):
#                 st.write(f"**qEI Score:** {info['ei_score']:.4f}")
#                 st.write(f"**Uncertainty:** {info['uncertainty']:.4f}")
#                 st.write(f"**GP Prediction:** {info['mean_prediction']:.4f}")
        
#         except Exception as e:
#             st.error(f"Error: {e}")

# --- BEST POSTERIOR MEAN ---
st.subheader("Best Posterior Mean")
if "model" in st.session_state and "train_x" in st.session_state:
    from bo_utils.bo_optimization import show_best_posterior_mean
    input_dim = st.session_state["train_x"].shape[1]
    best_x, best_y = show_best_posterior_mean(model=st.session_state["model"], input_dim=input_dim, bounds=bounds, limit= limit)
    st.write(f"Best x (rescaled): {best_x}")
    st.write(f"Predicted output: {best_y:.4f}")


if st.button("Retrospective Best Posterior Mean Evolution"):
    from streamlit_app.posterior_pred import retrospective_best_posterior_evolution
    if "model" in st.session_state and "train_x" in st.session_state:
        df_best_posterior = retrospective_best_posterior_evolution(
            train_x=st.session_state["train_x"],
            train_y=st.session_state["train_y"],
            model_config=MODEL_CONFIG,
            bounds=bounds,
            limit = limit
        )
        st.subheader("Best Posterior Mean Evolution")
        st.dataframe(df_best_posterior)

# --- BAYESIAN OPTIMIZATION ---

from streamlit_app.bayesian_optimization_step import bayesian_optimization_app
if "model" in st.session_state and "train_x" in st.session_state:
    bayesian_optimization_app(
        uploaded_file=uploaded_file,
        output_col_name=output_col_name,
        batch_size=BATCH_SIZE,
        acq_function=acq_function,
        bounds=bounds,
        columns_config=COLUMNS_CONFIG,
        data=data,
        model_config=MODEL_CONFIG,
        limit=limit
    )

# Get parameter names from uploaded CSV (excluding output column)
param_names = [col for col in data.columns if col != output_col_name]

# Check shape match
if "last_suggestion" in st.session_state:
    suggestion = st.session_state["last_suggestion"]
    if suggestion.shape[1] != len(param_names):
        st.warning(f"Shape mismatch: {suggestion.shape[1]} values, {len(param_names)} columns: {param_names}")

if "last_suggestion" in st.session_state:
    st.subheader("Last Suggested Experiment(s)")

    suggestion = st.session_state["last_suggestion"] 
    x_new = st.session_state['new_x'] if 'new_x' in st.session_state else None
    # Try to get correct parameter names from CSV
    if "data" in locals() and "output_col_name" in locals():
        param_names = [col for col in data.columns if col != output_col_name]
    else:
        param_names = [f"Var{i+1}" for i in range(suggestion.shape[1])]

    if suggestion.shape[1] != len(param_names):
        st.warning(f"Shape mismatch: {suggestion.shape[1]} values, {len(param_names)} columns.")
        # Fallback: use generic column names
        param_names = [f"Var{i+1}" for i in range(suggestion.shape[1])]

    # Show the suggestions
    st.dataframe(pd.DataFrame(suggestion, columns=param_names))
    st.dataframe(pd.DataFrame(x_new, columns=param_names))

if "last_updated_df" in st.session_state:
    st.subheader("Full Updated Experiment Table")
    st.dataframe(st.session_state["last_updated_df"])

from bo_utils.bo_optimization import find_promising_uncertain_point
from bo_utils.bo_utils import rescale_single_point

# if "model" in st.session_state and "train_x" in st.session_state:
#     st.subheader("Find Promising Uncertain Point")
#     ratio = st.slider("Ratio of qEI to Uncertainty", min_value=0.0, max_value=1.0, value=0.5, key="ratio")
#     if st.button("Find Promising Point"):
#         try:
#             # Unpack the tuple correctly
#             candidates, info = find_promising_uncertain_point(
#                 model=st.session_state["model"],
#                 train_x=st.session_state["train_x"],
#                 train_y=st.session_state["train_y"],
#                 bounds=bounds,
#                 limit=limit,
#                 best_n=1 , # Start with just 1 point
#                 ratio = ratio
#             )
            


#             # Rescale to original space
#             rescaled_point = rescale_single_point(candidates, bounds)
            
#             st.success("✅ Promising uncertain point found!")
#             st.write(f"**Point (original space):** {rescaled_point.numpy()}")
            
#             if isinstance(info, dict):
#                 st.write(f"**qEI Score:** {info['ei_score']:.4f}")
#                 st.write(f"**Uncertainty:** {info['uncertainty']:.4f}")
#                 st.write(f"**GP Prediction:** {info['mean_prediction']:.4f}")
        
#         except Exception as e:
#             st.error(f"Error: {e}")


# --- VALIDATE BAYESIAN OPTIMIZATION LOOP ---
# from streamlit_app.run_bo_test import run_bo_test_loop
# if "model" in st.session_state and "train_x" in st.session_state:
#     run_bo_test_loop(
#         model_config=MODEL_CONFIG,
#         acquisition_type=acq_function,
#         batch_size=BATCH_SIZE,
#         iterations=iterations,
#         initial_points=initial_points,
#         noise_std=noise_std
#     )

from bo_utils.bo_plotting import plot_posterior_slices_streamlit, plot_interactive_slice
from streamlit_app.bo_plotting import plot_posterior_slices_streamlit, plot_interactive_slice
# --- POSTERIOR SLICES ---
# Beispiel im Hauptteil deiner App:
# nach Modelltraining:
if (
    "model" in st.session_state and
    "train_x" in st.session_state and
    "bounds" in st.session_state and
    # data exists
    "data" in locals() and
    "output_col_name" in locals()
):
    param_names = [col for col in data.columns if col != output_col_name]
    # Hier wird die Funktion aufgerufen, um die Posterior-Slices zu plotten
    # Du kannst auch den Parameter 'param_names' anpassen, wenn nötig
    plot_posterior_slices_streamlit(
        model=st.session_state.model,
        train_x=st.session_state.train_x,
        bounds=st.session_state.bounds,
        param_names=param_names
    )

if (
    "model" in st.session_state and
    "train_x" in st.session_state and
    "bounds" in st.session_state and
    # data exists
    "data" in locals() and
    "output_col_name" in locals()
):
    param_names = [col for col in data.columns if col != output_col_name]
    plot_interactive_slice(
        #model=st.session_state.model,
        train_x=st.session_state.train_x,
        train_y=st.session_state.train_y,
        bounds=st.session_state.bounds,
        param_names=param_names,
        model_config=MODEL_CONFIG,
    )
else:
    st.info("Please train the model and run BO at least once.")  

from botorch.acquisition import (
    qExpectedImprovement, qNoisyExpectedImprovement,
    qLogExpectedImprovement, qLogNoisyExpectedImprovement, qUpperConfidenceBound
) 


from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
bounds = torch.tensor(bounds, dtype=torch.double).T  # Shape: [2, d]
candidate_set = torch.rand(1000, bounds.shape[1], dtype=torch.double)
if acq_function == "qEI":
    acq_factory = lambda model: qExpectedImprovement(model, best_f=st.session_state["train_y"].max().item())
elif acq_function == "qNEI":
    acq_factory = lambda model: qNoisyExpectedImprovement(model, X_baseline=st.session_state["train_x"])
elif acq_function == "qGIBBON":
    acq_factory = lambda model: qLowerBoundMaxValueEntropy(model,candidate_set)
elif acq_function == "qUCB":
    acq_factory = lambda model: qUpperConfidenceBound(model, beta=2.0)


from bo_utils.bo_plotting import plot_2d_bo_contours_streamlit





if all(k in st.session_state for k in ["model", "train_x", "train_y", "bounds"]) and "data" in locals() and "output_col_name" in locals():
    model = st.session_state["model"]
    train_x = st.session_state["train_x"]
    train_y = st.session_state["train_y"]
    bounds = st.session_state["bounds"]
    param_names = [col for col in data.columns if col != output_col_name]

    st.subheader("📈 2D/3D GP Surface Visualization")

    if len(param_names) >= 2:
        x_name = st.selectbox("Select X-axis Parameter", param_names, key="vis_x")
        y_name = st.selectbox("Select Y-axis Parameter", [p for p in param_names if p != x_name], key="vis_y")

        x_idx = param_names.index(x_name)
        y_idx = param_names.index(y_name)


        # Acquisition Function Selection
        acq_func_name = st.selectbox("Acquisition Function", ["qEI", "qNEI", "qGIBBON"], key="vis_acq")

        if acq_func_name == "qEI":
            acq_factory = lambda model: qLogExpectedImprovement(model, best_f=train_y.max().item())
        elif acq_func_name == "qNEI":
            acq_factory = lambda model: qLogNoisyExpectedImprovement(model, X_baseline=train_x)
        elif acq_func_name == "qGIBBON":
            boundsI = torch.tensor(bounds, dtype=torch.double).T  # Shape: [2, d]
            st.write(boundsI)
            candidate_set = torch.rand(1000, boundsI.shape[1], dtype=torch.double)
            st.write(candidate_set)
            acq_factory = lambda model: qLowerBoundMaxValueEntropy(model, candidate_set)
        else:
            st.warning("Invalid acquisition function selected.")


        # Plot type
        plot_type = st.radio("What to visualize?", ["Posterior Mean", "Model Uncertainty", "Acquisition Function"], key="vis_plot_type")

        save_html = st.checkbox("Also save as interactive HTML file?", value=False)
        filename_input = st.text_input("Filename (no extension):", value="bo_2d_plot")

        fixed_values = torch.zeros(len(param_names))

        if len(bounds) != len(param_names):
            st.warning("Mismatch between number of bounds and parameters. Please check your data.")
        else:
            with st.expander("🔧 Fixed Values for Remaining Parameters", expanded=False):
                for j, name in enumerate(param_names):
                    if j not in [x_idx, y_idx]:
                        lb, ub = bounds[j]
                        val = st.slider(
                            f"{name}",
                            min_value=lb,
                            max_value=ub,
                            value=(lb + ub) / 2,
                            step=(ub - lb) / 100
                        )
                        fixed_values[j] = (val - lb) / (ub - lb)

        if st.button("Generate Visualization"):
            with st.spinner("Rendering plot..."):
                plot_2d_bo_contours_streamlit(
                    model=model,
                    train_x=train_x,
                    train_y=train_y,
                    bounds=bounds,
                    param_x_idx=x_idx,
                    param_y_idx=y_idx,
                    param_names=param_names,
                    fixed_values=fixed_values,
                    acquisition_func_factory=acq_factory,
                    plot_type=plot_type,
                    show_streamlit=True,
                    save_html=save_html,
                    html_filename=filename_input + ".html"
                )

            st.success("✅ Plot generated.")

            if save_html:
                with open(filename_input + ".html", "rb") as f:
                    st.download_button(
                        label="📥 Download HTML",
                        data=f,
                        file_name=filename_input + ".html",
                        mime="text/html"
                    )
    else:
        st.warning("You need at least two input parameters for this plot.")
else:
    st.info("Please run a BO step first.")
st.session_state["trained_model"] = st.session_state["model"]
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bo_utils.bo_simulation_loop import simulate_bo_loop
from bo_utils.bo_utils import sample_random_model_config  # sicherstellen, dass vorhanden




st.session_state["trained_model"] = st.session_state["model"]
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bo_utils.bo_simulation_loop import simulate_bo_loop
from bo_utils.bo_utils import sample_random_model_config  # sicherstellen, dass vorhanden
st.subheader("🔁 Retrospective BO Strategy Simulation")
from bo_utils.bo_robust import RobustnessAnalyzer, create_robustness_analysis_ui
import concurrent.futures
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from typing import List, Dict, Any, Optional
from bo_utils.bo_model import fit_transfer_model, build_transfer_gp_model, create_transfer_model_from_source, extract_hyperparameters_from_model
# Ensure required imports are available
try:
    from bo_utils.bo_utils import sample_random_model_config
    from bo_utils.bo_simulation_loop import (
        simulate_bo_loop, 
        sample_random_ground_truth_config,
        FlexibleMetricsCollector
    )
    # Import transfer learning functions
    from bo_utils.bo_model import build_gp_model, fit_model
except ImportError as e:
    st.error(f"❌ Required imports missing: {e}")
    st.error("Please ensure bo_utils modules and transfer_gp_model are properly installed and available.")
    st.stop()

st.session_state["data"] = data
st.session_state["output_col_name"] = output_col_name

def run_single_simulation_safe(
    current_seed,
    train_x_full,
    train_y_full,
    data,
    bounds,
    output_col_name,
    total_evals,
    acquisition_functions,
    initial_sizes,
    limit,
    noise_level,
    ground_truth_method="gbm",
    noise_strategy="observation_noise",
    # Transfer Learning Parameters
    use_transfer_learning=False,
    source_model=None,
    prior_strength=2.0,
    # NEW: Outlier Parameters
    outlier_config=None
):
    """
    Safe wrapper function with outlier support
    """
    try:
        return run_single_simulation_debug(
            current_seed=current_seed,
            train_x_full=train_x_full,
            train_y_full=train_y_full,
            data=data,
            bounds=bounds,
            output_col_name=output_col_name,
            total_evals=total_evals,
            acquisition_functions=acquisition_functions,
            initial_sizes=initial_sizes,
            limit=limit,
            noise_level=noise_level,
            ground_truth_method=ground_truth_method,
            noise_strategy=noise_strategy,
            use_transfer_learning=use_transfer_learning,
            source_model=source_model,
            prior_strength=prior_strength,
            outlier_config=outlier_config
        )
    except Exception as e:
        outlier_suffix = " + Outliers" if outlier_config and outlier_config.get('outliers_enabled', False) else ""
        print(f"Simulation failed for seed {current_seed}, GT={ground_truth_method}{outlier_suffix}, Transfer={use_transfer_learning}: {str(e)}")
        traceback.print_exc()
        # Return an empty DataFrame with the relevant columns
        return pd.DataFrame(columns=[
            "seed", "ground_truth_method", "acquisition_type", "num_init", "batch_size", "iteration",
            "mse", "rmse_h", "r2", "crps", "coverage_high_region", "regret", "exploitative_regret",
            "use_transfer_learning", "prior_strength", "outliers_enabled", "outlier_fraction"
        ])

bo_config = []
from bo_utils.bo_simulation_loop import sample_random_ground_truth_config, sample_random_linear_config_with_outliers, sample_random_linear_config_with_true_random_outliers
def run_single_simulation_debug(
    current_seed,
    train_x_full,
    train_y_full,
    data,
    bounds,
    output_col_name,
    total_evals,
    acquisition_functions,
    initial_sizes,
    limit,
    noise_level,
    ground_truth_method="gbm",
    noise_strategy="observation_noise",
    # Transfer Learning Parameters
    use_transfer_learning=False,
    source_model=None,
    prior_strength=2.0,
    # NEW: Outlier Configuration
    outlier_config=None
):
    """
    Updated simulation function with outlier support
    """
    outlier_config = outlier_config or {'outliers_enabled': False}
    
    transfer_suffix = " (Transfer)" if use_transfer_learning else " (Standard)"
    outlier_suffix = " + Outliers" if outlier_config.get('outliers_enabled', False) else ""
    print(f"\n=== Starting simulation for seed {current_seed} with GT method: {ground_truth_method}{transfer_suffix}{outlier_suffix} ===")

    all_metrics = []
    input_dim = train_x_full.shape[1] if hasattr(train_x_full, 'shape') else len(train_x_full[0])

    # Generate configurations WITH outlier support
    if ground_truth_method in ["rf", "random_forest", "gbm", "gradient_boosting", "nn", "neural_network"]:
        ground_truth_config = sample_random_ground_truth_config(ground_truth_method, seed=current_seed)
    elif ground_truth_method in ["linear", "linear_with_outliers"]:
        # Enhanced config for linear models with outliers
        base_config = sample_random_linear_config_with_outliers(seed=current_seed) if ground_truth_method == "linear_with_outliers" else {'use_8_features': True, 'outliers_enabled': False}
        
        # Override with user-specified outlier settings
        if outlier_config.get('outliers_enabled', False):
            base_config.update(outlier_config)
        
        ground_truth_config = base_config
    else:
        ground_truth_config = sample_random_model_config(seed=current_seed, input_dim=input_dim)
    
    # Log outlier configuration
    if ground_truth_config.get('outliers_enabled', False):
        print(f"Outliers enabled: {ground_truth_config['outlier_fraction']*100:.1f}% of type '{ground_truth_config['outlier_type']}'")
    
    # Fixed BO model config for consistent comparison
    bo_model_config = {
        'covar_module': 'Matern_2.5',  # Good default kernel
        'outputscale': 1.0,           # Standard scaling
        'noise': 0.01,                # Low noise prior
        'lengthscale': 0.2            # Reasonable lengthscale
    }
    bo_model_config = MODEL_CONFIG
    
    print(f"Generated GT model config: {ground_truth_config}")
    print(f"Using BO model config: {bo_model_config}")
    print(f"Transfer Learning: {use_transfer_learning}")
    if use_transfer_learning:
        print(f"Prior Strength: {prior_strength}")

    for num_init in initial_sizes:
        num_steps = total_evals - num_init

        if num_steps <= 0:
            print(f"Skipping num_init={num_init}: no BO steps possible (num_steps={num_steps})")
            continue

        for acq in acquisition_functions:
            print(f"\nTrying: num_init={num_init}, acq={acq}, num_steps={num_steps}, transfer={use_transfer_learning}")

            try:
                # Call the updated simulate_bo_loop function with transfer learning parameters
                result = simulate_bo_loop(
                    train_x_full=train_x_full,
                    train_y_full=train_y_full,
                    data=data,
                    seed=current_seed,
                    bounds=bounds,
                    acquisition_type=acq,
                    num_init=num_init,
                    bo_steps=num_steps,
                    batch_size=1,
                    num_restarts=20,
                    raw_samples=2000,
                    
                    # Ground truth parameters
                    ground_truth_method=ground_truth_method,
                    ground_truth_config=ground_truth_config,
                    noise_strategy=noise_strategy,
                    noise_level=noise_level,
                    
                    # Transfer Learning parameters
                    use_transfer_learning=use_transfer_learning,
                    source_model=source_model,
                    prior_strength=prior_strength,
                    
                    # BO model parameters
                    bo_model_config=bo_model_config,
                    output_col_name=output_col_name,
                    num_total=total_evals,
                    limit=limit
                )

                # Extract metrics from result
                if isinstance(result, dict) and "metrics_df" in result:
                    metrics = result["metrics_df"]
                elif isinstance(result, pd.DataFrame):
                    metrics = result
                elif isinstance(result, tuple) and len(result) >= 2:
                    metrics = result[1]
                else:
                    metrics = result

                if metrics is None or (hasattr(metrics, 'empty') and metrics.empty):
                    print(f"⚠️ No metrics for GT={ground_truth_method}, seed={current_seed}, acq={acq}, num_init={num_init}, transfer={use_transfer_learning}")
                    continue

                # Attach metadata including transfer learning info
                metrics = metrics.copy()
                metrics["seed"] = current_seed
                metrics["acquisition_type"] = acq
                metrics["num_init"] = num_init
                metrics["batch_size"] = 1
                metrics["ground_truth_method"] = ground_truth_method
                metrics["noise_level"] = noise_level
                metrics["noise_strategy"] = noise_strategy
                # Transfer learning metadata
                metrics["use_transfer_learning"] = use_transfer_learning
                metrics["prior_strength"] = prior_strength if use_transfer_learning else None
                # dict to store ground truth config
                metrics["gt_config"] = ground_truth_config

                all_metrics.append(metrics)
                success_msg = f"✅ Successfully processed {acq} with {num_init} init points, GT method {ground_truth_method}"
                if use_transfer_learning:
                    success_msg += f", Transfer Learning (α={prior_strength})"
                print(success_msg)

            except Exception as e:
                print(f"Error in BO loop: GT={ground_truth_method}, seed={current_seed}, acq={acq}, init={num_init}, transfer={use_transfer_learning}: {str(e)}")
                traceback.print_exc()
                continue

    if not all_metrics:
        print(f"No successful runs for seed {current_seed}, GT method {ground_truth_method}, transfer={use_transfer_learning}")
        return pd.DataFrame(columns=[
            "seed", "ground_truth_method", "acquisition_type", "num_init", "batch_size", "iteration",
            "mse", "rmse_h", "r2", "crps", "coverage_high_region", "regret", "exploitative_regret",
            "use_transfer_learning", "prior_strength"
        ])

    result_df = pd.concat(all_metrics, ignore_index=True)
    print(f"Seed {current_seed} with GT {ground_truth_method}, transfer={use_transfer_learning} completed with {len(result_df)} rows")
    return result_df

if "train_x" in st.session_state and "train_y" in st.session_state and "bounds" in st.session_state:
    
    # Create tabs for simulation and analysis
    tab1, tab2 = st.tabs(["🚀 Run Simulation", "🎯 Robustness Analysis"])
    
    with tab1:
        # Debug information display
        st.subheader("🔍 Debug Information")
        debug_expander = st.expander("Show Debug Info")
        with debug_expander:
            st.write("**Session State Keys:**", list(st.session_state.keys()))
            
            if "data" in st.session_state:
                data_debug = st.session_state["data"]
                st.write("**Data Info:**")
                st.write(f"- Type: {type(data_debug)}")
                st.write(f"- Is None: {data_debug is None}")
                if data_debug is not None and hasattr(data_debug, 'shape'):
                    st.write(f"- Shape: {data_debug.shape}")
                if data_debug is not None and hasattr(data_debug, 'columns'):
                    st.write(f"- Columns: {list(data_debug.columns)}")
            
            output_col_debug = st.session_state.get("output_col_name", "Not set")
            st.write(f"**Output Column Name**: {output_col_debug}")
        
        # Configuration section
        st.subheader("⚙️ Simulation Configuration")
        col1, col2 = st.columns(2)
        with col1:
            n_seeds = st.slider("Number of seeds (Ground Truth variants)", min_value=1, max_value=20, value=5, 
                               help="Each seed creates a different plausible ground truth function")
            total_evals = st.slider("Total evaluations per run", min_value=10, max_value=100, value=25)
        
        with col2:
            use_parallel = st.checkbox("Use parallel processing", value=False, 
                                     help="Parallel processing uses multiple threads for faster execution")
            if use_parallel:
                max_cores = multiprocessing.cpu_count()
                selected_cores = st.slider(
                    "Number of parallel threads", 
                    min_value=1, 
                    max_value=min(max_cores, 8),
                    value=min(4, max_cores), 
                    help=f"Available CPU cores: {max_cores}. More threads = faster but more CPU usage."
                )
                st.info(f"⚡ Will use {selected_cores} parallel threads out of {max_cores} available cores.")
            
            acquisition_functions = st.multiselect(
                "Acquisition functions to test", 
                ["qEI", "Random", "qNEI", "qGIBBON","qUCB"], 
                default=["qEI", "Random"],
                help="Compare different BO strategies"
            )
        
        initial_sizes = st.multiselect(
            "Initial training sizes", 
            [3, 5, 10, 15, 20], 
            default=[5, 10],
            help="Different starting dataset sizes"
        )
        
        # Ground Truth Configuration
        st.subheader("🎲 Ground Truth Configuration")
        col3, col4 = st.columns(2)

        with col3:
            gt_methods = st.multiselect(
                "Ground Truth Method(s)",
                ["linear", "linear_with_outliers", "rf", "gbm", "nn", "gp"],  # ← ERWEITERT um linear_with_outliers
                default=["linear"],
                help="Select which ground truth models will act as the 'true function' in the simulation."
            )
            
            noise_strategy = st.selectbox(
                "Noise strategy",
                ["observation_noise", "deterministic"],
                index=0,
                help="Whether to simulate only deterministic ground truth or add stochastic observation noise."
            )

        with col4:
            noise_level = st.slider(
                "Observation noise level",
                min_value=0.0, max_value=0.2, value=0.02, step=0.01,
                help="Standard deviation of Gaussian observation noise added to the ground truth."
            )
            
            # === OUTLIER CONFIGURATION ===
            enable_outliers = st.checkbox(
                "🎯 Enable Outliers", 
                value=False,
                help="Add realistic outliers to linear ground truth models"
            )

        # === OUTLIER CONFIGURATION SECTION (Conditional) ===
        if enable_outliers and any("linear" in method for method in gt_methods):
            st.subheader("⚠️ Outlier Configuration")
            
            outlier_col1, outlier_col2 = st.columns(2)
            
            with outlier_col1:
                outlier_fraction = st.slider(
                    "Outlier Fraction (%)", 
                    min_value=1, 
                    max_value=15, 
                    value=5, 
                    step=1,
                    help="Percentage of points that will be outliers"
                ) / 100.0
                
                outlier_type = st.selectbox(
                    "Outlier Distribution Type",
                    ["random", "systematic", "extreme_values"],
                    index=0,
                    help="How outliers are distributed in the parameter space"
                )
            
            with outlier_col2:
                outlier_magnitude = st.slider(
                    "Outlier Magnitude", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=2.5, 
                    step=0.5,
                    help="How strong the outliers are (multiplier for deviation)"
                )
                
                outlier_generation_method = st.selectbox(
                    "Outlier Generation Method",
                    ["additive_noise", "multiplicative", "extreme_shift", "systematic_bias"],
                    index=0,
                    help="How outlier values are calculated"
                )
            
            # Advanced outlier settings
            with st.expander("🔧 Advanced Outlier Settings"):
                outlier_seed = st.number_input(
                    "Outlier Seed", 
                    min_value=1, 
                    max_value=1000, 
                    value=42,
                    help="Seed for reproducible outlier generation"
                )
                
                if outlier_type == "systematic":
                    st.info("💡 **Systematic Outliers**: Outliers will appear more frequently in corner regions of the parameter space")
                    corner_bias = st.slider(
                        "Corner Bias Multiplier", 
                        min_value=1.0, 
                        max_value=5.0, 
                        value=2.0, 
                        step=0.5,
                        help="How much more likely outliers are in corner regions"
                    )
                else:
                    corner_bias = 2.0
                
                if outlier_type == "extreme_values":
                    st.info("💡 **Extreme Values**: Points with high/low predicted values more likely to be outliers")
                    extreme_multiplier = st.slider(
                        "Extreme Value Multiplier", 
                        min_value=1.0, 
                        max_value=4.0, 
                        value=2.0, 
                        step=0.5,
                        help="How much more likely extreme predictions are to be outliers"
                    )
                else:
                    extreme_multiplier = 2.0
            
            # Create outlier configuration dict
            outlier_config = {
                'outliers_enabled': True,
                'outlier_fraction': outlier_fraction,
                'outlier_type': outlier_type,
                'outlier_magnitude': outlier_magnitude,
                'outlier_generation_method': outlier_generation_method,
                'outlier_seed': outlier_seed,
            }
            
            # Add type-specific parameters
            if outlier_type == "systematic":
                outlier_config['outlier_regions'] = [
                    {
                        'bounds': [(0.0, 0.3), (0.0, 0.3), (0.7, 1.0), (0.0, 1.0)],  # Corner region
                        'probability': outlier_fraction * corner_bias
                    },
                    {
                        'bounds': [(0.7, 1.0), (0.7, 1.0), (0.0, 0.3), (0.0, 1.0)],  # Opposite corner
                        'probability': outlier_fraction * (corner_bias * 0.8)
                    }
                ]
            
            # Preview outlier configuration
            st.info(f"""
            🎯 **Outlier Preview**: 
            - **{outlier_fraction*100:.1f}%** of points will be outliers
            - **Type**: {outlier_type.replace('_', ' ').title()}
            - **Magnitude**: {outlier_magnitude}x deviation
            - **Method**: {outlier_generation_method.replace('_', ' ').title()}
            """)

        else:
            # Default: no outliers
            outlier_config = {'outliers_enabled': False}

        # === TRANSFER LEARNING CONFIGURATION ===
        st.subheader("🔄 Transfer Learning Configuration")
        
        # Transfer Learning Enable/Disable
        col_transfer1, col_transfer2 = st.columns(2)
        
        with col_transfer1:
            enable_transfer_learning = st.checkbox(
                "🎯 Enable Transfer Learning", 
                value=False,
                help="Compare standard BO vs Transfer Learning BO using a source model"
            )
            
            if enable_transfer_learning:
                transfer_comparison_mode = st.radio(
                    "Comparison Mode:",
                    ["Both Standard and Transfer", "Transfer Learning Only"],
                    index=0,
                    help="Run both methods for comparison, or only Transfer Learning"
                )
            else:
                transfer_comparison_mode = "Standard BO Only"
        
        with col_transfer2:
            if enable_transfer_learning:
                prior_strength = st.slider(
                    "Prior Strength (α)",
                    min_value=0.5, 
                    max_value=10.0, 
                    value=2.0, 
                    step=0.5,
                    help="Strength of transfer priors. Higher = stronger influence from source model"
                )
                
                source_model_option = st.selectbox(
                    "Source Model",
                    ["Use session state model", "Generate synthetic source"],
                    index=0,
                    help="Choose between existing trained model or generate a synthetic one"
                )
            else:
                prior_strength = 2.0
                source_model_option = "Use session state model"

        # Source Model Setup
        source_model = None
        if enable_transfer_learning:
            st.subheader("🏗️ Source Model Setup")
            
            if source_model_option == "Use session state model":
                # Check if there's a trained model in session state
                if "trained_model" in st.session_state and st.session_state["trained_model"] is not None:
                    source_model = st.session_state["trained_model"]
                    st.success("✅ Using trained model from session state")
                    
                    # Show source model info
                    with st.expander("📊 Source Model Information"):
                        try:
                            source_hyperparams = extract_hyperparameters_from_model(source_model)
                            st.write("**Source Model Hyperparameters:**")
                            for key, value in source_hyperparams.items():
                                if key not in ['ard', 'input_dim']:
                                    if hasattr(value, 'numpy'):
                                        st.write(f"- {key}: {value.numpy()}")
                                    else:
                                        st.write(f"- {key}: {value}")
                        except Exception as e:
                            st.warning(f"Could not extract source model info: {e}")
                
                else:
                    st.warning("⚠️ No trained model found in session state. Please train a model first or use synthetic source.")
                    if st.button("🎲 Generate Synthetic Source Model"):
                        try:
                            # Generate synthetic source model
                            input_dim = st.session_state["train_x"].shape[1]
                            synthetic_x = torch.randn(50, input_dim).double()
                            synthetic_y = torch.randn(50, 1).double()
                            
                            source_model, _ = build_gp_model(
                                synthetic_x, synthetic_y, 
                                input_dim=input_dim, 
                                config={"kernel_type": "Matern", "kernel_nu": 2.5}
                            )
                            source_model = fit_model(source_model)
                            
                            st.session_state["synthetic_source_model"] = source_model
                            source_model = st.session_state["synthetic_source_model"]
                            st.success("✅ Generated synthetic source model")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to generate synthetic source model: {e}")
            
            elif source_model_option == "Generate synthetic source":
                if "synthetic_source_model" not in st.session_state:
                    if st.button("🎲 Generate Synthetic Source Model"):
                        try:
                            input_dim = st.session_state["train_x"].shape[1]
                            synthetic_x = torch.randn(50, input_dim).double()
                            synthetic_y = torch.randn(50, 1).double()
                            
                            source_model, _ = build_gp_model(
                                synthetic_x, synthetic_y, 
                                input_dim=input_dim, 
                                config={"kernel_type": "Matern", "kernel_nu": 2.5}
                            )
                            source_model = fit_model(source_model)
                            
                            st.session_state["synthetic_source_model"] = source_model
                            st.success("✅ Generated synthetic source model")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to generate synthetic source model: {e}")
                else:
                    source_model = st.session_state["synthetic_source_model"]
                    st.success("✅ Using previously generated synthetic source model")

            # Validation check
            if source_model is None:
                st.error("❌ No source model available. Transfer Learning will be disabled.")
                enable_transfer_learning = False
        
        # Robustness testing info with transfer learning
        if enable_transfer_learning and source_model is not None:
            if transfer_comparison_mode == "Both Standard and Transfer":
                n_conditions = 2  # Standard + Transfer
                condition_desc = "Standard BO and Transfer Learning BO"
            else:
                n_conditions = 1  # Transfer only
                condition_desc = "Transfer Learning BO"
        else:
            n_conditions = 1  # Standard only
            condition_desc = "Standard BO"

        # Count linear methods with outliers
        linear_methods_with_outliers = [m for m in gt_methods if "linear" in m and outlier_config.get('outliers_enabled', False)]
        outlier_info = f" (with {outlier_config['outlier_fraction']*100:.0f}% outliers)" if linear_methods_with_outliers else ""

        st.info(f"""
        🎯 **Robustness Testing Setup{outlier_info}**

        This simulation will create **{n_seeds} different ground truth variants** representing uncertainty in the true function.
        Each variant tests **{len(acquisition_functions)} acquisition functions** with **{len(initial_sizes)} different initial sizes**.

        **Conditions to test**: {condition_desc}
        **Per GT method**: {n_seeds * len(acquisition_functions) * len(initial_sizes) * n_conditions} runs
        **Selected GT methods**: {len(gt_methods)} ({', '.join(gt_methods)})
        **Linear methods with outliers**: {len(linear_methods_with_outliers)}
        **Total simulation runs**: {n_seeds * len(acquisition_functions) * len(initial_sizes) * n_conditions * len(gt_methods)} runs
        **Expected runtime**: ~{n_seeds * len(acquisition_functions) * len(initial_sizes) * n_conditions * len(gt_methods) * 0.5:.1f} minutes
        """)
        # Pre-run validation
        st.subheader("📋 Pre-run Validation")
        validation_passed = True
        
        # Check required data
        required_keys = ["train_x", "train_y", "bounds", "data", "output_col_name"]
        validation_cols = st.columns(len(required_keys))
        
        for i, key in enumerate(required_keys):
            with validation_cols[i]:
                if key not in st.session_state:
                    st.error(f"❌ {key}")
                    validation_passed = False
                elif st.session_state[key] is None:
                    st.error(f"❌ {key} (None)")
                    validation_passed = False
                else:
                    st.success(f"✅ {key}")
        
        # Additional validation checks
        if validation_passed:
            data_check = st.session_state.get("data")
            output_col_check = st.session_state.get("output_col_name")
            
            if data_check is not None and hasattr(data_check, 'columns'):
                if output_col_check not in data_check.columns:
                    st.error(f"❌ Output column '{output_col_check}' not found in data. Available: {list(data_check.columns)}")
                    validation_passed = False
                else:
                    st.success(f"✅ Output column '{output_col_check}' found in data")
            
            # Check ground truth methods
            if not gt_methods:
                st.error("❌ Please select at least one ground truth method")
                validation_passed = False
            
            # Check acquisition functions
            if not acquisition_functions:
                st.error("❌ Please select at least one acquisition function")
                validation_passed = False
            
            # Check transfer learning setup
            if enable_transfer_learning and source_model is None:
                st.error("❌ Transfer Learning enabled but no source model available")
                validation_passed = False
        
        run_simulation = st.button("🚀 Run Monte Carlo BO Evaluation", 
                                  disabled=not validation_passed,
                                  help="Start the robustness simulation")

        if run_simulation and validation_passed:
            train_x_full = st.session_state["train_x"]
            train_y_full = st.session_state["train_y"]
            bounds = st.session_state["bounds"]
            output_col_name = st.session_state["output_col_name"]
            data = st.session_state["data"]
            limit = st.session_state.get("limit", -100000)

            # Set default for cores if parallel processing is not used
            if not use_parallel:
                selected_cores = 1

            # Initialize evaluation storage if not present
            if 'eval' not in st.session_state:
                st.session_state['eval'] = []

            all_metrics = []

            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_preview = st.empty()

            start_time = pd.Timestamp.now()
            
            # Determine which conditions to run
               # Determine which conditions to run
            conditions_to_run = []
            if enable_transfer_learning and source_model is not None:
                if transfer_comparison_mode == "Both Standard and Transfer":
                    conditions_to_run = [
                        {"use_transfer_learning": False, "source_model": None, "prior_strength": None},
                        {"use_transfer_learning": True, "source_model": source_model, "prior_strength": prior_strength}
                    ]
                else:  # Transfer Learning Only
                    conditions_to_run = [
                        {"use_transfer_learning": True, "source_model": source_model, "prior_strength": prior_strength}
                    ]
            else:  # Standard BO Only
                conditions_to_run = [
                    {"use_transfer_learning": False, "source_model": None, "prior_strength": None}
                ]
            
# FIXED PARALLEL PROCESSING SECTION - with outlier support

            if use_parallel:
                # PARALLEL PROCESSING MODE - FIXED with outlier support
                st.info("🚀 **Using parallel threading** - multiple simulations running simultaneously")
                
                # Prepare all simulation tasks
                simulation_tasks = []
                for gt_method in gt_methods:
                    for condition in conditions_to_run:
                        for seed in range(n_seeds):
                            simulation_tasks.append({
                                'current_seed': seed,
                                'train_x_full': train_x_full,
                                'train_y_full': train_y_full,
                                'data': data,
                                'bounds': bounds,
                                'output_col_name': output_col_name,
                                'total_evals': total_evals,
                                'acquisition_functions': acquisition_functions,
                                'initial_sizes': initial_sizes,
                                'limit': limit,
                                'noise_level': noise_level,
                                'ground_truth_method': gt_method,
                                'noise_strategy': noise_strategy,
                                'use_transfer_learning': condition['use_transfer_learning'],
                                'source_model': condition['source_model'],
                                'prior_strength': condition['prior_strength'],
                                'outlier_config': outlier_config  # ← FIXED: Jetzt korrekt übergeben
                            })
                
                total_tasks = len(simulation_tasks)
                completed_tasks = 0
                
                # Execute in parallel using threads
                max_workers = selected_cores
                status_text.text(f"🔄 Starting {total_tasks} parallel simulations (using {max_workers} threads)...")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks using the safe function directly
                    future_to_task = {}
                    for task in simulation_tasks:
                        future = executor.submit(
                            run_single_simulation_safe,
                            current_seed=task['current_seed'],
                            train_x_full=task['train_x_full'],
                            train_y_full=task['train_y_full'],
                            data=task['data'],
                            bounds=task['bounds'],
                            output_col_name=task['output_col_name'],
                            total_evals=task['total_evals'],
                            acquisition_functions=task['acquisition_functions'],
                            initial_sizes=task['initial_sizes'],
                            limit=task['limit'],
                            noise_level=task['noise_level'],
                            ground_truth_method=task['ground_truth_method'],
                            noise_strategy=task['noise_strategy'],
                            use_transfer_learning=task['use_transfer_learning'],
                            source_model=task['source_model'],
                            prior_strength=task['prior_strength'],
                            outlier_config=task['outlier_config']  # ← FIXED: Jetzt auch im submit call
                        )
                        future_to_task[future] = task
                    
                    # Process completed tasks
                    for future in concurrent.futures.as_completed(future_to_task):
                        task = future_to_task[future]
                        completed_tasks += 1
                        
                        try:
                            result = future.result()
                            
                            # Process result
                            if result is not None and not result.empty:
                                result["ground_truth_method"] = task['ground_truth_method']
                                all_metrics.append(result)
                                
                                # Update progress
                                progress_bar.progress(completed_tasks / total_tasks)
                                elapsed = pd.Timestamp.now() - start_time
                                transfer_info = " (Transfer)" if task['use_transfer_learning'] else " (Standard)"
                                
                                # *** ENHANCED STATUS MESSAGE with outlier info ***
                                outlier_info = ""
                                if task['outlier_config'] and task['outlier_config'].get('outliers_enabled', False):
                                    outlier_fraction = task['outlier_config'].get('outlier_fraction', 0.0)
                                    outlier_info = f" + {outlier_fraction*100:.0f}% outliers"
                                
                                status_text.text(
                                    f"✅ Completed {completed_tasks}/{total_tasks} "
                                    f"(GT={task['ground_truth_method']}{transfer_info}{outlier_info}, seed={task['current_seed']}) "
                                    f"- Elapsed: {elapsed.total_seconds()/60:.1f}m"
                                )
                                
                                # Live preview with outlier information
                                if len(all_metrics) >= 1:
                                    with metrics_preview:
                                        combined_preview = pd.concat(all_metrics, ignore_index=True)
                                        n_standard = len(combined_preview[combined_preview['use_transfer_learning'] == False])
                                        n_transfer = len(combined_preview[combined_preview['use_transfer_learning'] == True])
                                        
                                        # Count outlier runs if data available
                                        outlier_info_preview = ""
                                        if 'outliers_enabled' in combined_preview.columns:
                                            n_with_outliers = len(combined_preview[combined_preview['outliers_enabled'] == True])
                                            outlier_info_preview = f", {n_with_outliers} with outliers"
                                        
                                        st.write(f"📊 **Results so far**: {len(combined_preview)} rows ({n_standard} Standard, {n_transfer} Transfer{outlier_info_preview})")
                            
                            else:
                                transfer_info = " (Transfer)" if task['use_transfer_learning'] else " (Standard)"
                                outlier_info = ""
                                if task['outlier_config'] and task['outlier_config'].get('outliers_enabled', False):
                                    outlier_fraction = task['outlier_config'].get('outlier_fraction', 0.0)
                                    outlier_info = f" + {outlier_fraction*100:.0f}% outliers"
                                
                                st.warning(f"⚠️ GT={task['ground_truth_method']}{transfer_info}{outlier_info}, seed={task['current_seed']} returned empty results")
                                
                        except Exception as e:
                            transfer_info = " (Transfer)" if task['use_transfer_learning'] else " (Standard)"
                            outlier_info = ""
                            if task['outlier_config'] and task['outlier_config'].get('outliers_enabled', False):
                                outlier_fraction = task['outlier_config'].get('outlier_fraction', 0.0)
                                outlier_info = f" + {outlier_fraction*100:.0f}% outliers"
                            
                            st.error(f"❌ Task failed (GT={task['ground_truth_method']}{transfer_info}{outlier_info}, seed={task['current_seed']}): {str(e)}")

            else:
                # SEQUENTIAL PROCESSING MODE - already fixed with outlier support  
                st.info("🐌 **Using sequential processing** - one simulation at a time")
                
                total_runs = n_seeds * len(gt_methods) * len(conditions_to_run)
                run_counter = 0

                # Loop over all GT methods, conditions, and seeds
                for gt_method in gt_methods:
                    for condition in conditions_to_run:
                        condition_name = "Transfer" if condition['use_transfer_learning'] else "Standard"
                        outlier_info = ""
                        if outlier_config.get('outliers_enabled', False):
                            outlier_fraction = outlier_config.get('outlier_fraction', 0.0)
                            outlier_info = f" + {outlier_fraction*100:.0f}% outliers"
                        
                        for seed in range(n_seeds):
                            elapsed = pd.Timestamp.now() - start_time
                            status_text.text(
                                f"⏳ Processing: GT={gt_method}{outlier_info} ({condition_name}), seed {seed + 1}/{n_seeds} "
                                f"({run_counter+1}/{total_runs}) (Elapsed: {elapsed.total_seconds()/60:.1f}m)"
                            )
                            try:
                                # Main simulation call with outlier support
                                result = run_single_simulation_safe(
                                    current_seed=seed,
                                    train_x_full=train_x_full,
                                    train_y_full=train_y_full,
                                    data=data,
                                    bounds=bounds,
                                    output_col_name=output_col_name,
                                    total_evals=total_evals,
                                    acquisition_functions=acquisition_functions,
                                    initial_sizes=initial_sizes,
                                    limit=limit,
                                    noise_level=noise_level,
                                    ground_truth_method=gt_method,
                                    noise_strategy=noise_strategy,
                                    use_transfer_learning=condition['use_transfer_learning'],
                                    source_model=condition['source_model'],
                                    prior_strength=condition['prior_strength'],
                                    outlier_config=outlier_config  # ← Already correctly passed here
                                )

                                # Attach metadata and collect
                                if result is not None and not result.empty:
                                    result["ground_truth_method"] = gt_method
                                    all_metrics.append(result)

                                    # Live preview in Streamlit
                                    if len(all_metrics) >= 1:
                                        combined_preview = pd.concat(all_metrics, ignore_index=True)
                                        with metrics_preview:
                                            n_standard = len(combined_preview[combined_preview['use_transfer_learning'] == False])
                                            n_transfer = len(combined_preview[combined_preview['use_transfer_learning'] == True])
                                            
                                            # Count outlier runs if data available
                                            outlier_info_preview = ""
                                            if 'outliers_enabled' in combined_preview.columns:
                                                n_with_outliers = len(combined_preview[combined_preview['outliers_enabled'] == True])
                                                outlier_info_preview = f", {n_with_outliers} with outliers"
                                            
                                            st.write(
                                                f"📊 **Current Results**: {len(combined_preview)} total rows "
                                                f"({n_standard} Standard, {n_transfer} Transfer{outlier_info_preview})"
                                            )
                                            if len(all_metrics) > 1:
                                                preview_stats = combined_preview.groupby(
                                                    ['ground_truth_method', 'acquisition_type', 'num_init', 'use_transfer_learning']
                                                ).size()
                                                st.write(preview_stats)
                                else:
                                    st.warning(f"⚠️ GT={gt_method} ({condition_name}){outlier_info}, seed {seed} returned empty results")
                            except Exception as e:
                                st.error(f"❌ GT={gt_method} ({condition_name}){outlier_info}, seed {seed} failed: {str(e)}")
                                traceback.print_exc()
                            
                            run_counter += 1
                            progress_bar.progress(run_counter / total_runs)

            # ADDITIONAL DEBUGGING HELPER for parallel mode
            def debug_parallel_outlier_config(outlier_config):
                """Helper function to debug outlier config in parallel mode"""
                if outlier_config is None:
                    print("DEBUG: outlier_config is None")
                    return False
                
                if not isinstance(outlier_config, dict):
                    print(f"DEBUG: outlier_config wrong type: {type(outlier_config)}")
                    return False
                
                print(f"DEBUG: outlier_config keys: {list(outlier_config.keys())}")
                print(f"DEBUG: outliers_enabled: {outlier_config.get('outliers_enabled', 'NOT_FOUND')}")
                
                if outlier_config.get('outliers_enabled', False):
                    print(f"DEBUG: outlier_fraction: {outlier_config.get('outlier_fraction', 'NOT_FOUND')}")
                    print(f"DEBUG: outlier_type: {outlier_config.get('outlier_type', 'NOT_FOUND')}")
                    print(f"DEBUG: outlier_magnitude: {outlier_config.get('outlier_magnitude', 'NOT_FOUND')}")
                    return True
                
                return False

            # Enhanced error logging for debugging
            def run_single_simulation_safe_with_debug(
                current_seed,
                train_x_full,
                train_y_full,
                data,
                bounds,
                output_col_name,
                total_evals,
                acquisition_functions,
                initial_sizes,
                limit,
                noise_level,
                ground_truth_method="gbm",
                noise_strategy="observation_noise",
                use_transfer_learning=False,
                source_model=None,
                prior_strength=2.0,
                outlier_config=None
            ):
                """
                Enhanced safe wrapper with outlier debugging
                """
                try:
                    # Debug outlier config
                    has_outliers = debug_parallel_outlier_config(outlier_config)
                    outlier_suffix = f" + {outlier_config.get('outlier_fraction', 0)*100:.0f}% outliers" if has_outliers else ""
                    
                    print(f"Starting simulation: seed={current_seed}, GT={ground_truth_method}{outlier_suffix}, Transfer={use_transfer_learning}")
                    
                    return run_single_simulation_debug(
                        current_seed=current_seed,
                        train_x_full=train_x_full,
                        train_y_full=train_y_full,
                        data=data,
                        bounds=bounds,
                        output_col_name=output_col_name,
                        total_evals=total_evals,
                        acquisition_functions=acquisition_functions,
                        initial_sizes=initial_sizes,
                        limit=limit,
                        noise_level=noise_level,
                        ground_truth_method=ground_truth_method,
                        noise_strategy=noise_strategy,
                        use_transfer_learning=use_transfer_learning,
                        source_model=source_model,
                        prior_strength=prior_strength,
                        outlier_config=outlier_config
                    )
                except Exception as e:
                    has_outliers = outlier_config and outlier_config.get('outliers_enabled', False) if outlier_config else False
                    outlier_suffix = f" + {outlier_config.get('outlier_fraction', 0)*100:.0f}% outliers" if has_outliers else ""
                    
                    print(f"Simulation FAILED: seed={current_seed}, GT={ground_truth_method}{outlier_suffix}, Transfer={use_transfer_learning}")
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                    
                    # Return an empty DataFrame with the relevant columns
                    return pd.DataFrame(columns=[
                        "seed", "ground_truth_method", "acquisition_type", "num_init", "batch_size", "iteration",
                        "mse", "rmse_h", "r2", "crps", "coverage_high_region", "regret", "exploitative_regret",
                        "use_transfer_learning", "prior_strength", "outliers_enabled", "outlier_fraction"
                    ])
            # Clear progress indicators after completion
            progress_container.empty()
            total_time = pd.Timestamp.now() - start_time

            if not all_metrics:
                st.error("❌ All simulations failed. Check the error messages above.")
                st.stop()

            # Combine all results into a single DataFrame
            try:
                df_all_metrics = pd.concat(all_metrics, ignore_index=True)
                st.success(
                    f"🎉 Simulation complete! Processed {len(all_metrics)} successful runs "
                    f"in {total_time.total_seconds()/60:.1f} minutes."
                )

                # Store results for later analysis (session state)
                st.session_state['bo_results'] = df_all_metrics

                # Show summary stats with transfer learning breakdown
                st.subheader("📊 Simulation Summary")
                if not df_all_metrics.empty:
                    # Overall summary with outlier information
                    if 'use_transfer_learning' in df_all_metrics.columns:
                        summary_cols = st.columns(4)  # ← Changed from 3 to 4
                        with summary_cols[0]:
                            st.metric("Total Runs", len(df_all_metrics))
                        with summary_cols[1]:
                            n_standard = len(df_all_metrics[df_all_metrics['use_transfer_learning'] == False])
                            st.metric("Standard BO Runs", n_standard)
                        with summary_cols[2]:
                            n_transfer = len(df_all_metrics[df_all_metrics['use_transfer_learning'] == True])
                            st.metric("Transfer BO Runs", n_transfer)
                        with summary_cols[3]:
                            if 'outliers_enabled' in df_all_metrics.columns:
                                n_with_outliers = len(df_all_metrics[df_all_metrics['outliers_enabled'] == True])
                                st.metric("Runs with Outliers", n_with_outliers)
                            else:
                                st.metric("Outlier Data", "N/A")
                    
                    # Detailed breakdown
                    groupby_cols = ['ground_truth_method', 'acquisition_type', 'num_init']
                    if 'use_transfer_learning' in df_all_metrics.columns:
                        groupby_cols.append('use_transfer_learning')
                    
                    summary_stats = df_all_metrics.groupby(groupby_cols).agg({
                        'mse': ['count', 'mean', 'std', 'min'] if 'mse' in df_all_metrics.columns else ['count'],
                        'regret': ['mean', 'std', 'min'] if 'regret' in df_all_metrics.columns else ['count'],
                        'seed': 'nunique'
                    }).round(4)
                    st.dataframe(summary_stats, use_container_width=True)

                    # Quick comparison if both methods were run
                    if 'use_transfer_learning' in df_all_metrics.columns and len(df_all_metrics['use_transfer_learning'].unique()) > 1:
                        st.subheader("🔄 Quick Transfer Learning Comparison")
                        
                        # Final regret comparison
                        if 'regret' in df_all_metrics.columns and 'iteration' in df_all_metrics.columns:
                            final_regrets = df_all_metrics.groupby(['seed', 'acquisition_type', 'use_transfer_learning'])['regret'].last().reset_index()
                            
                            comparison_stats = final_regrets.groupby(['acquisition_type', 'use_transfer_learning'])['regret'].agg([
                                'mean', 'std', 'min', 'max'
                            ]).round(4)
                            
                            st.write("**Final Regret Comparison:**")
                            st.dataframe(comparison_stats.reset_index(), use_container_width=True)
                            
                            # Calculate improvement
                            for acq in final_regrets['acquisition_type'].unique():
                                acq_data = final_regrets[final_regrets['acquisition_type'] == acq]
                                
                                if len(acq_data['use_transfer_learning'].unique()) == 2:
                                    standard_mean = acq_data[acq_data['use_transfer_learning'] == False]['regret'].mean()
                                    transfer_mean = acq_data[acq_data['use_transfer_learning'] == True]['regret'].mean()
                                    
                                    improvement = ((standard_mean - transfer_mean) / standard_mean) * 100
                                    
                                    if improvement > 0:
                                        st.success(f"🎯 **{acq}**: Transfer Learning improved by {improvement:.1f}% (lower regret)")
                                    else:
                                        st.warning(f"⚠️ **{acq}**: Standard BO performed {-improvement:.1f}% better")

                    # Show sample of data
                    st.subheader("📋 Sample Results")
                    st.dataframe(df_all_metrics.head(10), use_container_width=True)
                else:
                    st.warning("No data to display")

            except Exception as e:
                st.error(f"❌ Failed to combine results: {str(e)}")
                traceback.print_exc()
                st.stop()

            # Save results and offer download
            try:
                if not df_all_metrics.empty:
                    filename_suffix = "_transfer" if enable_transfer_learning else "_standard"
                    df_all_metrics.to_csv(f"bo_all_metrics_mc{filename_suffix}.csv", index=False, sep=";")
                    st.success(f"💾 Results saved to 'bo_all_metrics_mc{filename_suffix}.csv'")

                    csv_data = df_all_metrics.to_csv(index=False, sep=";")
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_data,
                        file_name=f"bo_simulation_results{filename_suffix}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results to save")
            except Exception as e:
                st.error(f"❌ Failed to save results: {str(e)}")

            # Enhanced visualization with transfer learning comparison
            if not df_all_metrics.empty and any(col in df_all_metrics.columns for col in ['mse', 'regret']):
                st.subheader("📈 Enhanced Convergence Analysis")
                
                # Choose metric to plot
                metric_col = 'regret' if 'regret' in df_all_metrics.columns else 'mse'
                
                # Enhanced convergence plot comparing Standard vs Transfer
                if 'use_transfer_learning' in df_all_metrics.columns and 'iteration' in df_all_metrics.columns:
                    transfer_methods = df_all_metrics['use_transfer_learning'].unique()
                    acquisition_types = df_all_metrics['acquisition_type'].unique()
                    
                    if len(transfer_methods) > 1 and len(acquisition_types) <= 3:
                        fig, axes = plt.subplots(1, len(acquisition_types), figsize=(5*len(acquisition_types), 5))
                        if len(acquisition_types) == 1:
                            axes = [axes]
                        
                        colors = {'Standard': '#1f77b4', 'Transfer': '#ff7f0e'}
                        
                        for idx, acq_type in enumerate(acquisition_types):
                            for use_transfer in transfer_methods:
                                subset = df_all_metrics[
                                    (df_all_metrics['acquisition_type'] == acq_type) & 
                                    (df_all_metrics['use_transfer_learning'] == use_transfer)
                                ]
                                
                                if len(subset) > 0 and 'iteration' in subset.columns:
                                    convergence_stats = subset.groupby('iteration')[metric_col].agg(['mean', 'std']).reset_index()
                                    
                                    method_name = 'Transfer' if use_transfer else 'Standard'
                                    color = colors.get(method_name, '#2ca02c')
                                    
                                    axes[idx].plot(
                                        convergence_stats['iteration'], 
                                        convergence_stats['mean'], 
                                        'o-', 
                                        label=f'{method_name} (Mean)', 
                                        color=color,
                                        linewidth=2, 
                                        markersize=4
                                    )
                                    axes[idx].fill_between(
                                        convergence_stats['iteration'], 
                                        convergence_stats['mean'] - convergence_stats['std'],
                                        convergence_stats['mean'] + convergence_stats['std'],
                                        alpha=0.2, 
                                        color=color,
                                        label=f'{method_name} (±1σ)'
                                    )
                            
                            axes[idx].set_title(f'{acq_type}\nStandard vs Transfer Learning')
                            axes[idx].set_xlabel('Iteration')
                            axes[idx].set_ylabel(f'{metric_col.title()}')
                            axes[idx].legend()
                            axes[idx].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                
                # Fallback: Simple convergence plot
                else:
                    strategies = df_all_metrics.groupby(['acquisition_type', 'num_init']).size().index
                    
                    if len(strategies) <= 4:  # Only show if manageable number
                        fig, axes = plt.subplots(1, len(strategies), figsize=(4*len(strategies), 4))
                        if len(strategies) == 1:
                            axes = [axes]
                        
                        for idx, (acq_type, num_init) in enumerate(strategies):
                            subset = df_all_metrics[
                                (df_all_metrics['acquisition_type'] == acq_type) & 
                                (df_all_metrics['num_init'] == num_init)
                            ]
                            
                            if 'iteration' in subset.columns:
                                convergence_stats = subset.groupby('iteration')[metric_col].agg(['mean', 'std']).reset_index()
                                
                                axes[idx].plot(convergence_stats['iteration'], convergence_stats['mean'], 'o-', label='Mean')
                                axes[idx].fill_between(
                                    convergence_stats['iteration'], 
                                    convergence_stats['mean'] - convergence_stats['std'],
                                    convergence_stats['mean'] + convergence_stats['std'],
                                    alpha=0.3, label='±1 Std'
                                )
                                axes[idx].set_title(f'{acq_type}\n(init={num_init})')
                                axes[idx].set_xlabel('Iteration')
                                axes[idx].set_ylabel(f'{metric_col.title()}')
                                axes[idx].legend()
                                axes[idx].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        elif run_simulation and not validation_passed:
            st.error("❌ Cannot run simulation - validation failed. Please check the issues above.")
    
    with tab2:
        # Robustness Analysis Tab - Enhanced for Transfer Learning
        st.subheader("🎯 Advanced Robustness Analysis")
        
        # === DATA SOURCE SELECTION ===
        st.subheader("📁 Data Source Selection")
        data_source = st.radio(
            "Choose your data source:",
            ["Use results from current session", "Upload results file"],
            help="You can either use results from the simulation you just ran, or upload a CSV file with previous results"
        )
        
        # === DATA LOADING ===
        df_results = None
        
        if data_source == "Use results from current session":
            if 'bo_results' not in st.session_state or st.session_state['bo_results'] is None:
                st.warning("⚠️ No simulation results found. Please run the simulation first in the 'Run Simulation' tab.")
            else:
                df_results = st.session_state['bo_results']
                
                # Check if transfer learning data is present
                has_transfer = 'use_transfer_learning' in df_results.columns and len(df_results['use_transfer_learning'].unique()) > 1
                transfer_info = " (includes Transfer Learning comparison)" if has_transfer else ""
                st.success(f"✅ Using current session results ({len(df_results)} rows){transfer_info}")
        
        else:  # Upload results file
            st.info("📤 Upload a CSV file containing BO simulation results")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type=['csv'],
                help="Upload a CSV file with columns like: ground_truth_method, acquisition_type, num_init, seed, regret, mse, iteration, use_transfer_learning, etc."
            )
            
            if uploaded_file is not None:
                try:
                    # Try different separators
                    separators_to_try = [';', ',', '\t']
                    df_uploaded = None
                    
                    for sep in separators_to_try:
                        try:
                            df_uploaded = pd.read_csv(uploaded_file, sep=sep)
                            if len(df_uploaded.columns) > 1:  # Successfully parsed multiple columns
                                break
                        except:
                            continue
                        # Reset file pointer for next attempt
                        uploaded_file.seek(0)
                    
                    if df_uploaded is not None and len(df_uploaded.columns) > 1:
                        df_results = df_uploaded
                        
                        # Check for transfer learning columns
                        has_transfer = 'use_transfer_learning' in df_results.columns
                        transfer_info = " (includes Transfer Learning data)" if has_transfer else ""
                        
                        st.success(f"✅ Successfully loaded file with {len(df_results)} rows and {len(df_results.columns)} columns{transfer_info}")
                        
                        # Show file info
                        with st.expander("📊 Uploaded File Information"):
                            st.write(f"**Shape**: {df_results.shape}")
                            st.write(f"**Columns**: {list(df_results.columns)}")
                            st.write("**Sample data**:")
                            st.dataframe(df_results.head(), use_container_width=True)
                            
                            # Check for required columns
                            required_columns = ['acquisition_type', 'regret', 'seed']
                            transfer_columns = ['use_transfer_learning', 'prior_strength']
                            
                            missing_cols = [col for col in required_columns if col not in df_results.columns]
                            transfer_cols_present = [col for col in transfer_columns if col in df_results.columns]
                            
                            if missing_cols:
                                st.warning(f"⚠️ Missing recommended columns: {missing_cols}")
                            else:
                                st.success("✅ All key columns found!")
                            
                            if transfer_cols_present:
                                st.info(f"🔄 Transfer Learning columns found: {transfer_cols_present}")
                    else:
                        st.error("❌ Could not parse the uploaded file. Please check the format.")
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
            else:
                st.info("👆 Please upload a CSV file to proceed with the analysis")
        
        # === PROCEED WITH ENHANCED ANALYSIS IF WE HAVE DATA ===
        if df_results is not None and not df_results.empty:
            
            # Check if transfer learning data is available
            has_transfer_data = 'use_transfer_learning' in df_results.columns
            
            # === DATA VALIDATION (Enhanced) ===
            st.subheader("🔍 Enhanced Data Validation")
            
            # Check data structure
            validation_info = st.container()
            with validation_info:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", len(df_results))
                
                with col2:
                    if 'seed' in df_results.columns:
                        n_seeds = df_results['seed'].nunique()
                        st.metric("Unique Seeds", n_seeds)
                    else:
                        st.metric("Unique Seeds", "N/A")
                
                with col3:
                    if 'acquisition_type' in df_results.columns:
                        n_strategies = df_results['acquisition_type'].nunique()
                        st.metric("Strategies", n_strategies)
                    else:
                        st.metric("Strategies", "N/A")
                
                with col4:
                    if has_transfer_data:
                        n_standard = len(df_results[df_results['use_transfer_learning'] == False])
                        n_transfer = len(df_results[df_results['use_transfer_learning'] == True])
                        st.metric("Standard/Transfer", f"{n_standard}/{n_transfer}")
                    else:
                        st.metric("Transfer Data", "Not Available")
            
            # === COLUMN MAPPING (for uploaded files) ===
            if data_source == "Upload results file":
                st.subheader("🔧 Column Mapping")
                with st.expander("Map columns if needed"):
                    available_cols = list(df_results.columns)
                    
                    # Key column mappings
                    col_mappings = {}
                    mapping_cols = st.columns(5)
                    
                    with mapping_cols[0]:
                        regret_col = st.selectbox("Regret column", options=['regret'] + available_cols, 
                                                index=0 if 'regret' in available_cols else len(['regret']))
                        if regret_col != 'regret' and regret_col in available_cols:
                            col_mappings['regret'] = regret_col
                    
                    with mapping_cols[1]:
                        acq_col = st.selectbox("Acquisition type column", options=['acquisition_type'] + available_cols,
                                             index=0 if 'acquisition_type' in available_cols else len(['acquisition_type']))
                        if acq_col != 'acquisition_type' and acq_col in available_cols:
                            col_mappings['acquisition_type'] = acq_col
                    
                    with mapping_cols[2]:
                        seed_col = st.selectbox("Seed column", options=['seed'] + available_cols,
                                              index=0 if 'seed' in available_cols else len(['seed']))
                        if seed_col != 'seed' and seed_col in available_cols:
                            col_mappings['seed'] = seed_col
                    
                    with mapping_cols[3]:
                        gt_col = st.selectbox("Ground truth method column", options=['ground_truth_method'] + available_cols,
                                            index=0 if 'ground_truth_method' in available_cols else len(['ground_truth_method']))
                        if gt_col != 'ground_truth_method' and gt_col in available_cols:
                            col_mappings['ground_truth_method'] = gt_col
                    
                    with mapping_cols[4]:
                        transfer_col = st.selectbox("Transfer learning column", options=['use_transfer_learning'] + available_cols,
                                                  index=0 if 'use_transfer_learning' in available_cols else len(['use_transfer_learning']))
                        if transfer_col != 'use_transfer_learning' and transfer_col in available_cols:
                            col_mappings['use_transfer_learning'] = transfer_col
                    
                    # Apply mappings
                    if col_mappings:
                        df_results = df_results.rename(columns=col_mappings)
                        st.info(f"Applied column mappings: {col_mappings}")
                        has_transfer_data = 'use_transfer_learning' in df_results.columns

            # === ENHANCED ANALYSIS WITH TRANSFER LEARNING SUPPORT ===
            st.subheader("📊 Enhanced Robustness Analysis with Transfer Learning")
            
            # Analysis mode selection
            analysis_mode = st.radio(
                "Select analysis mode:",
                ["🎯 Original Advanced RobustnessAnalyzer", "📊 Enhanced Transfer Learning Analysis", "📋 Basic Analysis Only"],
                index=1 if has_transfer_data else 0,
                help="Choose the type of robustness analysis to perform"
            )
            
            if analysis_mode == "🎯 Original Advanced RobustnessAnalyzer":
                try:
                    # Use the original advanced system with uploaded data support
                    if data_source == "Use results from current session":
                        # Use original function without parameters (reads from session state)
                        create_robustness_analysis_ui()
                    else:
                        # Pass uploaded data to the modified function
                        create_robustness_analysis_ui(df_results)
                    
                    st.success("✅ Advanced RobustnessAnalyzer completed successfully!")
                    
                except (NameError, AttributeError, ImportError) as e:
                    st.error(f"❌ Advanced RobustnessAnalyzer not available: {str(e)}")
                    st.info("🔄 Falling back to Enhanced Transfer Learning Analysis...")
                    analysis_mode = "📊 Enhanced Transfer Learning Analysis"
                
                except Exception as e:
                    st.error(f"❌ Advanced RobustnessAnalyzer failed: {str(e)}")
                    st.info("🔄 Falling back to Enhanced Transfer Learning Analysis...")
                    analysis_mode = "📊 Enhanced Transfer Learning Analysis"
            
            if analysis_mode == "📊 Enhanced Transfer Learning Analysis":
                # === ENHANCED TRANSFER LEARNING ANALYSIS ===
                st.info("🚀 **Running Enhanced Transfer Learning Robustness Analysis**")
                from bo_utils.bo_simulation_loop import enhanced_outlier_analysis_ui
                try:
                    # Import required libraries
                    import scipy.stats as stats
                    
                    # === 1. TRANSFER LEARNING OVERVIEW ===
                    if has_transfer_data:
                        st.markdown("### 🔄 Transfer Learning Overview")
                        
                        transfer_overview_cols = st.columns(4)
                        
                        with transfer_overview_cols[0]:
                            n_standard = len(df_results[df_results['use_transfer_learning'] == False])
                            st.metric("Standard BO Runs", n_standard)
                        
                        with transfer_overview_cols[1]:
                            n_transfer = len(df_results[df_results['use_transfer_learning'] == True])
                            st.metric("Transfer BO Runs", n_transfer)
                        
                        with transfer_overview_cols[2]:
                            if 'prior_strength' in df_results.columns:
                                prior_strengths = df_results[df_results['use_transfer_learning'] == True]['prior_strength'].unique()
                                st.metric("Prior Strengths", f"{len(prior_strengths)} levels")
                            else:
                                st.metric("Prior Strengths", "N/A")
                        
                        with transfer_overview_cols[3]:
                            total_comparisons = len(df_results.groupby(['seed', 'acquisition_type']).size())
                            st.metric("Total Comparisons", total_comparisons)
                        if 'outliers_enabled' in df_results.columns:
                            enhanced_outlier_analysis_ui(df_results)
                    else:
                        st.warning("⚠️ No Transfer Learning data detected. Running standard analysis...")
                    
                    # === 2. COMPREHENSIVE PERFORMANCE COMPARISON ===
                    st.markdown("### 📊 Comprehensive Performance Analysis")
                    
                    # Check for required columns
                    required_cols = ['acquisition_type', 'regret', 'seed']
                    missing_cols = [col for col in required_cols if col not in df_results.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                        st.info("Available columns: " + ", ".join(df_results.columns))
                    else:
                        # Get final performance for each run
                        if 'iteration' in df_results.columns:
                            groupby_cols = ['acquisition_type', 'seed']
                            if 'ground_truth_method' in df_results.columns:
                                groupby_cols.append('ground_truth_method')
                            if has_transfer_data:
                                groupby_cols.append('use_transfer_learning')
                            final_performance = df_results.groupby(groupby_cols)['regret'].last().reset_index()
                        else:
                            final_performance = df_results.copy()
                        
                        # Group by strategy (including transfer learning if available)
                        strategy_groupby = ['acquisition_type']
                        if 'ground_truth_method' in df_results.columns:
                            strategy_groupby.append('ground_truth_method')
                        if has_transfer_data:
                            strategy_groupby.append('use_transfer_learning')
                        
                        # Calculate comprehensive statistics
                        strategy_stats = final_performance.groupby(strategy_groupby)['regret'].agg([
                            'count',           # Number of runs
                            'mean',           # Mean performance
                            'std',            # Standard deviation
                            'min',            # Best case
                            'max',            # Worst case
                            'median',         # Median
                            lambda x: np.percentile(x, 10),   # 10th percentile
                            lambda x: np.percentile(x, 25),   # 25th percentile  
                            lambda x: np.percentile(x, 75),   # 75th percentile
                            lambda x: np.percentile(x, 90),   # 90th percentile
                            lambda x: np.percentile(x, 95),   # 95th percentile
                            lambda x: stats.skew(x),          # Skewness
                            lambda x: stats.kurtosis(x),      # Kurtosis
                        ]).round(4)
                        
                        strategy_stats.columns = ['n_runs', 'mean', 'std', 'best', 'worst', 'median', 
                                                 'p10', 'p25', 'p75', 'p90', 'p95', 'skewness', 'kurtosis']
                        
                        # Calculate advanced robustness metrics
                        strategy_stats['range'] = (strategy_stats['worst'] - strategy_stats['best']).round(4)
                        strategy_stats['iqr'] = (strategy_stats['p75'] - strategy_stats['p25']).round(4)
                        strategy_stats['cv'] = (strategy_stats['std'] / strategy_stats['mean']).round(4)  # Coefficient of Variation
                        
                        # Median Absolute Deviation (more robust than std)
                        strategy_stats['mad'] = final_performance.groupby(strategy_groupby)['regret'].apply(
                            lambda x: np.median(np.abs(x - np.median(x)))
                        ).round(4)
                        
                        # Robustness scores (higher = more robust)
                        strategy_stats['consistency_score'] = (1 / (1 + strategy_stats['cv'])).round(4)
                        strategy_stats['reliability_score'] = (1 / (1 + strategy_stats['range'])).round(4)
                        strategy_stats['overall_robustness'] = ((strategy_stats['consistency_score'] + strategy_stats['reliability_score']) / 2).round(4)
                        
                        # Display comprehensive table
                        st.dataframe(strategy_stats.reset_index(), use_container_width=True)
                        
                        # === 3. TRANSFER LEARNING SPECIFIC ANALYSIS ===
                        if has_transfer_data and len(df_results['use_transfer_learning'].unique()) > 1:
                            st.markdown("### 🔄 Transfer Learning Impact Analysis")
                            
                            # Direct comparison between Standard and Transfer for each acquisition function
                            st.markdown("**Head-to-Head Performance Comparison**")
                            
                            comparison_results = []
                            acquisition_types = final_performance['acquisition_type'].unique()
                            
                            for acq_type in acquisition_types:
                                acq_data = final_performance[final_performance['acquisition_type'] == acq_type]
                                
                                if len(acq_data['use_transfer_learning'].unique()) == 2:
                                    standard_regrets = acq_data[acq_data['use_transfer_learning'] == False]['regret']
                                    transfer_regrets = acq_data[acq_data['use_transfer_learning'] == True]['regret']
                                    
                                    # Statistical test
                                    try:
                                        stat, p_value = stats.wilcoxon(standard_regrets, transfer_regrets, alternative='greater')
                                        significant = p_value < 0.05
                                    except:
                                        stat, p_value = np.nan, np.nan
                                        significant = False
                                    
                                    # Calculate improvements
                                    mean_improvement = ((standard_regrets.mean() - transfer_regrets.mean()) / standard_regrets.mean()) * 100
                                    median_improvement = ((standard_regrets.median() - transfer_regrets.median()) / standard_regrets.median()) * 100
                                    
                                    # Win rate (how often transfer is better)
                                    wins = sum(s > t for s, t in zip(standard_regrets, transfer_regrets))
                                    win_rate = (wins / len(standard_regrets)) * 100
                                    
                                    comparison_results.append({
                                        'acquisition_type': acq_type,
                                        'mean_improvement_%': mean_improvement,
                                        'median_improvement_%': median_improvement,
                                        'win_rate_%': win_rate,
                                        'p_value': p_value,
                                        'significant': significant,
                                        'standard_mean': standard_regrets.mean(),
                                        'transfer_mean': transfer_regrets.mean(),
                                        'standard_std': standard_regrets.std(),
                                        'transfer_std': transfer_regrets.std()
                                    })
                            
                            if comparison_results:
                                comparison_df = pd.DataFrame(comparison_results).round(4)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Highlight best improvements
                                st.markdown("**Key Findings:**")
                                
                                # Best improvement
                                best_improvement = comparison_df.loc[comparison_df['mean_improvement_%'].idxmax()]
                                if best_improvement['mean_improvement_%'] > 0:
                                    significance_text = " (statistically significant)" if best_improvement['significant'] else " (not significant)"
                                    st.success(f"🏆 **Best Transfer Learning Performance**: {best_improvement['acquisition_type']} "
                                             f"improved by {best_improvement['mean_improvement_%']:.1f}%{significance_text}")
                                
                                # Most consistent improvement
                                best_win_rate = comparison_df.loc[comparison_df['win_rate_%'].idxmax()]
                                st.info(f"🎯 **Most Consistent**: {best_win_rate['acquisition_type']} "
                                       f"wins {best_win_rate['win_rate_%']:.1f}% of comparisons")
                                
                                # Statistical significance summary
                                n_significant = comparison_df['significant'].sum()
                                st.write(f"📊 **Statistical Significance**: {n_significant}/{len(comparison_df)} "
                                        f"acquisition functions show significant improvement with Transfer Learning")
                        
                        # === 4. ENHANCED VISUALIZATIONS ===
                        st.markdown("### 📊 Enhanced Performance Visualizations")
                        
                        if has_transfer_data and len(df_results['use_transfer_learning'].unique()) > 1:
                            # Transfer Learning comparison plots
                            acquisition_types = final_performance['acquisition_type'].unique()
                            
                            if len(acquisition_types) <= 4:
                                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                                axes = axes.flatten()
                                
                                # 1. Box plot comparison
                                ax = axes[0]
                                box_data = []
                                box_labels = []
                                colors = []
                                
                                for acq_type in acquisition_types:
                                    acq_data = final_performance[final_performance['acquisition_type'] == acq_type]
                                    
                                    standard_data = acq_data[acq_data['use_transfer_learning'] == False]['regret']
                                    transfer_data = acq_data[acq_data['use_transfer_learning'] == True]['regret']
                                    
                                    if len(standard_data) > 0:
                                        box_data.append(standard_data)
                                        box_labels.append(f'{acq_type}\n(Standard)')
                                        colors.append('#1f77b4')
                                    
                                    if len(transfer_data) > 0:
                                        box_data.append(transfer_data)
                                        box_labels.append(f'{acq_type}\n(Transfer)')
                                        colors.append('#ff7f0e')
                                
                                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                                for patch, color in zip(bp['boxes'], colors):
                                    patch.set_facecolor(color)
                                    patch.set_alpha(0.7)
                                
                                ax.set_title('Final Regret Distribution Comparison')
                                ax.set_ylabel('Regret')
                                ax.tick_params(axis='x', rotation=45)
                                ax.grid(True, alpha=0.3)
                                
                                # 2. Improvement scatter plot
                                ax = axes[1]
                                for i, acq_type in enumerate(acquisition_types):
                                    acq_data = final_performance[final_performance['acquisition_type'] == acq_type]
                                    
                                    if len(acq_data['use_transfer_learning'].unique()) == 2:
                                        standard_mean = acq_data[acq_data['use_transfer_learning'] == False]['regret'].mean()
                                        transfer_mean = acq_data[acq_data['use_transfer_learning'] == True]['regret'].mean()
                                        standard_std = acq_data[acq_data['use_transfer_learning'] == False]['regret'].std()
                                        transfer_std = acq_data[acq_data['use_transfer_learning'] == True]['regret'].std()
                                        
                                        improvement = ((standard_mean - transfer_mean) / standard_mean) * 100
                                        
                                        color = '#2ca02c' if improvement > 0 else '#d62728'
                                        ax.scatter(standard_mean, transfer_mean, s=100, alpha=0.7, c=color)
                                        ax.annotate(acq_type, (standard_mean, transfer_mean), 
                                                   xytext=(5,5), textcoords='offset points', fontsize=9)
                                
                                # Add diagonal line (no improvement)
                                min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
                                max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
                                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Improvement')
                                
                                ax.set_xlabel('Standard BO Mean Regret')
                                ax.set_ylabel('Transfer BO Mean Regret')
                                ax.set_title('Standard vs Transfer Performance')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                # 3. Convergence comparison
                                if 'iteration' in df_results.columns:
                                    ax = axes[2]
                                    
                                    for use_transfer in [False, True]:
                                        subset = df_results[df_results['use_transfer_learning'] == use_transfer]
                                        convergence_stats = subset.groupby('iteration')['regret'].agg(['mean', 'std']).reset_index()
                                        
                                        method_name = 'Transfer' if use_transfer else 'Standard'
                                        color = '#ff7f0e' if use_transfer else '#1f77b4'
                                        
                                        ax.plot(convergence_stats['iteration'], convergence_stats['mean'], 
                                               'o-', label=f'{method_name} (Mean)', color=color, linewidth=2)
                                        ax.fill_between(convergence_stats['iteration'], 
                                                       convergence_stats['mean'] - convergence_stats['std'],
                                                       convergence_stats['mean'] + convergence_stats['std'],
                                                       alpha=0.2, color=color, label=f'{method_name} (±1σ)')
                                    
                                    ax.set_xlabel('Iteration')
                                    ax.set_ylabel('Regret')
                                    ax.set_title('Convergence Comparison')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                
                                # 4. Win rate analysis
                                ax = axes[3]
                                if comparison_results:
                                    acq_names = [r['acquisition_type'] for r in comparison_results]
                                    win_rates = [r['win_rate_%'] for r in comparison_results]
                                    improvements = [r['mean_improvement_%'] for r in comparison_results]
                                    
                                    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
                                    
                                    bars = ax.bar(acq_names, win_rates, color=colors, alpha=0.7)
                                    ax.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Random (50%)')
                                    
                                    # Add improvement percentages as text
                                    for bar, imp in zip(bars, improvements):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                               f'{imp:+.1f}%', ha='center', va='bottom', fontsize=9)
                                    
                                    ax.set_ylabel('Win Rate (%)')
                                    ax.set_title('Transfer Learning Win Rates')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    ax.set_ylim(0, 100)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        
                        # === 5. FINAL RECOMMENDATION (Enhanced for Transfer Learning) ===
                        st.markdown("### 💡 Enhanced Recommendation System")
                        
                        if has_transfer_data and len(df_results['use_transfer_learning'].unique()) > 1:
                            # Multi-criteria analysis for both Standard and Transfer
                            st.markdown("**Transfer Learning Decision Matrix:**")
                            
                            # User-adjustable criteria weights
                            with st.expander("🔧 Adjust Recommendation Criteria"):
                                st.write("**Adjust the importance of each criterion:**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    mean_weight = st.slider("Mean Performance Weight", 0.0, 1.0, 0.3, 0.05)
                                    robustness_weight = st.slider("Robustness Weight", 0.0, 1.0, 0.3, 0.05)
                                
                                with col2:
                                    consistency_weight = st.slider("Consistency Weight", 0.0, 1.0, 0.2, 0.05)
                                    improvement_weight = st.slider("Transfer Improvement Weight", 0.0, 1.0, 0.2, 0.05)
                                
                                # Normalize weights
                                total_weight = mean_weight + robustness_weight + consistency_weight + improvement_weight
                                if total_weight > 0:
                                    criteria_weights = {
                                        'mean_performance': mean_weight / total_weight,
                                        'robustness': robustness_weight / total_weight,
                                        'consistency': consistency_weight / total_weight,
                                        'transfer_improvement': improvement_weight / total_weight
                                    }
                                else:
                                    criteria_weights = {'mean_performance': 0.25, 'robustness': 0.25, 'consistency': 0.25, 'transfer_improvement': 0.25}
                            
                            # Create recommendation matrix
                            recommendation_data = []
                            
                            for acq_type in acquisition_types:
                                acq_data = final_performance[final_performance['acquisition_type'] == acq_type]
                                
                                for use_transfer in [False, True]:
                                    method_data = acq_data[acq_data['use_transfer_learning'] == use_transfer]
                                    
                                    if len(method_data) > 0:
                                        method_name = f"{acq_type}_{'Transfer' if use_transfer else 'Standard'}"
                                        
                                        # Calculate metrics
                                        mean_regret = method_data['regret'].mean()
                                        std_regret = method_data['regret'].std()
                                        cv = std_regret / mean_regret if mean_regret > 0 else 0
                                        
                                        # Transfer improvement (only for transfer methods)
                                        if use_transfer and len(acq_data['use_transfer_learning'].unique()) == 2:
                                            standard_mean = acq_data[acq_data['use_transfer_learning'] == False]['regret'].mean()
                                            improvement = ((standard_mean - mean_regret) / standard_mean) * 100
                                        else:
                                            improvement = 0
                                        
                                        recommendation_data.append({
                                            'method': method_name,
                                            'acquisition_type': acq_type,
                                            'use_transfer': use_transfer,
                                            'mean_regret': mean_regret,
                                            'cv': cv,
                                            'improvement': improvement
                                        })
                            
                            if recommendation_data:
                                rec_df = pd.DataFrame(recommendation_data)
                                
                                # Normalize metrics (0-1 scale where 1 is best)
                                rec_df['mean_norm'] = 1 - (rec_df['mean_regret'] - rec_df['mean_regret'].min()) / (rec_df['mean_regret'].max() - rec_df['mean_regret'].min()) if rec_df['mean_regret'].max() > rec_df['mean_regret'].min() else 1
                                rec_df['cv_norm'] = 1 - (rec_df['cv'] - rec_df['cv'].min()) / (rec_df['cv'].max() - rec_df['cv'].min()) if rec_df['cv'].max() > rec_df['cv'].min() else 1
                                rec_df['improvement_norm'] = (rec_df['improvement'] - rec_df['improvement'].min()) / (rec_df['improvement'].max() - rec_df['improvement'].min()) if rec_df['improvement'].max() > rec_df['improvement'].min() else 0
                                
                                # Calculate final scores
                                rec_df['final_score'] = (
                                    criteria_weights['mean_performance'] * rec_df['mean_norm'] +
                                    criteria_weights['robustness'] * rec_df['cv_norm'] +
                                    criteria_weights['consistency'] * rec_df['cv_norm'] +
                                    criteria_weights['transfer_improvement'] * rec_df['improvement_norm']
                                ).round(4)
                                
                                # Get best recommendation
                                best_method = rec_df.loc[rec_df['final_score'].idxmax()]
                                
                                # Display recommendation
                                method_desc = f"{best_method['acquisition_type']} with {'Transfer Learning' if best_method['use_transfer'] else 'Standard BO'}"
                                st.success(f"🏆 **Recommended Method**: {method_desc}")
                                
                                rec_cols = st.columns(4)
                                with rec_cols[0]:
                                    st.metric("Final Score", f"{best_method['final_score']:.4f}")
                                with rec_cols[1]:
                                    st.metric("Mean Regret", f"{best_method['mean_regret']:.4f}")
                                with rec_cols[2]:
                                    st.metric("Consistency (CV)", f"{best_method['cv']:.4f}")
                                with rec_cols[3]:
                                    if best_method['use_transfer']:
                                        st.metric("Improvement", f"{best_method['improvement']:.1f}%")
                                    else:
                                        st.metric("Transfer", "N/A")
                                
                                # Show top 3 recommendations
                                st.markdown("**Top 3 Recommendations:**")
                                top_3 = rec_df.nlargest(3, 'final_score')[['method', 'final_score', 'mean_regret', 'cv', 'improvement']].round(4)
                                st.dataframe(top_3, use_container_width=True)
                        
                        else:
                            # Standard recommendation (no transfer learning data)
                            st.markdown("**Standard BO Recommendation:**")
                            
                            # Find best strategy based on mean regret and consistency
                            best_strategy_idx = strategy_stats.loc[
                                strategy_stats['overall_robustness'].idxmax() if 'overall_robustness' in strategy_stats.columns 
                                else strategy_stats['mean'].idxmin()
                            ]
                            
                            strategy_name = f"{best_strategy_idx.name}" if isinstance(best_strategy_idx.name, str) else f"{best_strategy_idx.name[0]}"
                            st.success(f"🏆 **Recommended Strategy**: {strategy_name}")
                            
                            rec_cols = st.columns(4)
                            with rec_cols[0]:
                                st.metric("Mean Regret", f"{best_strategy_idx['mean']:.4f}")
                            with rec_cols[1]:
                                st.metric("Std Dev", f"{best_strategy_idx['std']:.4f}")
                            with rec_cols[2]:
                                st.metric("Best Case", f"{best_strategy_idx['best']:.4f}")
                            with rec_cols[3]:
                                st.metric("Worst Case", f"{best_strategy_idx['worst']:.4f}")
                        
                        # === 6. EXPORT ENHANCED RESULTS ===
                        st.markdown("### 💾 Export Enhanced Analysis Results")
                        
                        # Prepare comprehensive results for export
                        export_data = {
                            'strategy_statistics': strategy_stats.reset_index(),
                            'has_transfer_learning': has_transfer_data,
                            'analysis_timestamp': pd.Timestamp.now().isoformat()
                        }
                        
                        if has_transfer_data and comparison_results:
                            export_data['transfer_comparison'] = comparison_results
                        
                        # Create summary CSV
                        summary_csv = strategy_stats.reset_index().to_csv(index=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            filename_suffix = "_with_transfer" if has_transfer_data else "_standard_only"
                            st.download_button(
                                "📥 Download Strategy Statistics",
                                summary_csv,
                                file_name=f"enhanced_robustness_analysis{filename_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Save detailed results as JSON
                            import json
                            json_results = json.dumps(export_data, indent=2, default=str)
                            st.download_button(
                                "📥 Download Detailed Analysis (JSON)",
                                json_results,
                                file_name=f"detailed_enhanced_analysis{filename_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        st.success("✅ Enhanced Transfer Learning Robustness Analysis completed!")
                
                except ImportError as e:
                    st.error(f"❌ Missing required library: {e}")
                    st.info("Please install: pip install scipy matplotlib")
                    
                except Exception as e:
                    st.error(f"❌ Error in enhanced analysis: {str(e)}")
                    st.info("🔄 Falling back to basic analysis...")
                    analysis_mode = "📋 Basic Analysis Only"
            
            if analysis_mode == "📋 Basic Analysis Only":
                # === BASIC ANALYSIS (FALLBACK) ===
                st.info("🔧 Using basic analysis...")
                
                if 'regret' in df_results.columns:
                    # Get final regret for each run (last iteration)
                    if 'iteration' in df_results.columns:
                        groupby_cols = ['acquisition_type', 'seed']
                        if has_transfer_data:
                            groupby_cols.append('use_transfer_learning')
                        final_regrets = df_results.groupby(groupby_cols)['regret'].last().reset_index()
                    else:
                        final_regrets = df_results[['acquisition_type', 'seed', 'regret']].copy()
                        if has_transfer_data:
                            final_regrets['use_transfer_learning'] = df_results['use_transfer_learning']
                    
                    # Statistics by strategy
                    strategy_groupby = ['acquisition_type']
                    if has_transfer_data:
                        strategy_groupby.append('use_transfer_learning')
                    
                    strategy_stats = final_regrets.groupby(strategy_groupby)['regret'].agg([
                        'count', 'mean', 'std', 'min', 'max',
                        lambda x: np.percentile(x, 25),
                        lambda x: np.percentile(x, 50),  # median
                        lambda x: np.percentile(x, 75),
                        lambda x: np.percentile(x, 95)
                    ]).round(4)
                    
                    strategy_stats.columns = ['n_runs', 'mean', 'std', 'best', 'worst', 'q25', 'median', 'q75', 'q95']
                    
                    # Calculate robustness metrics
                    strategy_stats['cv'] = (strategy_stats['std'] / strategy_stats['mean']).round(4)
                    strategy_stats['iqr'] = (strategy_stats['q75'] - strategy_stats['q25']).round(4)
                    strategy_stats['robustness_score'] = (1 / (1 + strategy_stats['cv'])).round(4)
                    
                    st.dataframe(strategy_stats.reset_index(), use_container_width=True)
                    
                    # Best strategy
                    best_strategy = strategy_stats.loc[strategy_stats['robustness_score'].idxmax()]
                    
                    if has_transfer_data and len(strategy_stats.index.names) > 1:
                        strategy_desc = f"{best_strategy.name[0]} ({'Transfer' if best_strategy.name[1] else 'Standard'})"
                    else:
                        strategy_desc = f"{best_strategy.name}"
                    
                    st.success(f"🏆 **Most Robust Strategy**: {strategy_desc}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Regret", f"{best_strategy['mean']:.4f}")
                    with col2:
                        st.metric("Robustness Score", f"{best_strategy['robustness_score']:.4f}")
                    with col3:
                        st.metric("Worst Case", f"{best_strategy['worst']:.4f}")
                
                else:
                    st.warning("No 'regret' column found for analysis. Available columns:")
                    st.write(list(df_results.columns))

else:
    st.warning("⚠️ Required session state data not found. Please run the data preparation steps first.")
    missing_keys = []
    for key in ["train_x", "train_y", "bounds"]:
        if key not in st.session_state:
            missing_keys.append(key)
    if missing_keys:
        st.write(f"Missing keys: {missing_keys}")