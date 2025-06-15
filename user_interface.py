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
