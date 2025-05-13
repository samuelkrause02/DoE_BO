import streamlit as st
import pandas as pd
import torch
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_validation import loocv_gp_custom
from bo_utils.bo_optimization import optimize_qEI
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

st.title("Bayesian Optimization for CO₂ Sequestration")

# Upload data
st.sidebar.header("Upload Experiment Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
print("type(uploaded_file):", type(uploaded_file))
if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=";")
    st.subheader("Uploaded Data")
    st.dataframe(data)

# Ensure data is loaded
if 'data' in locals():

    # SECTION: Output Column first
    st.subheader("Select Output Column")
    output_col_name = st.selectbox("Choose the output column", data.columns, key="output_col")

    if output_col_name:
        st.write(f"Selected output column: `{output_col_name}`")

        # SECTION: Parameter Bounds (excluding output column)
        st.subheader("Specify Parameter Bounds")
        bounds = []  # fresh list

        param_columns = [col for col in data.columns if col != output_col_name]
        for col in param_columns:
            min_val = st.number_input(
                f"Min value for parameter '{col}'",
                value=float(data[col].min()),
                key=f"min_{col}"
            )
            max_val = st.number_input(
                f"Max value for parameter '{col}'",
                value=float(data[col].max()),
                key=f"max_{col}"
            )
            bounds.append((min_val, max_val))

        st.write("Current parameter bounds:", dict(zip(param_columns, bounds)))


# Model settings
st.sidebar.header("Model Settings")
kernel_type = st.sidebar.selectbox("Kernel Type", ["Matern", "RBF"])
ard = st.sidebar.checkbox("ARD", value=True)
st.sidebar.subheader("Noise Prior (LogNormal)")
mean_noise = st.sidebar.number_input(
    "Mean Noise (μ, e.g. 0.01 = 1%)",
    min_value=1e-6,
    max_value=1.0,
    value=0.01,
    step=0.001,
    format="%.5f"
)

log_scale = st.sidebar.number_input(
    "Log Std (σ in log-space)",
    min_value=0.01,
    max_value=3.0,
    value=0.5,
    step=0.05,
    format="%.2f"
)

st.sidebar.write(f"Noise Prior: LogNormal(loc=log({mean_noise:.5f}), scale={log_scale:.2f})")

MODEL_CONFIG["mean_noise"] = mean_noise
MODEL_CONFIG["scale_log"] = log_scale
MODEL_CONFIG["noise_prior_type"] = "LogNormal"

prior_type = st.sidebar.selectbox("Prior Type", [None, "Gamma", "SmoothedBox", "LogNormal"])
if prior_type != None and prior_type != "LogNormal":
    prior_param1 = st.sidebar.number_input("Prior Param 1", value=2.0)
    prior_param2 = st.sidebar.number_input("Prior Param 2", value=0.5)
else:
    prior_param1, prior_param2 = None, None

acq_function = st.sidebar.selectbox(
    "Acquisition Function",
    ["qEI", "qNEI"]
)

# Batch size
BATCH_SIZE = st.sidebar.slider(
    "Batch Size",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

iterations = st.sidebar.slider("Number of iterations:", min_value=10, max_value=100, value=50, step=10)
initial_points = st.sidebar.slider("Number of initial points:", min_value=5, max_value=20, value=10)
noise_std = st.sidebar.slider("Noise standard deviation:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)


#Build config
MODEL_CONFIG["name"] = "Matern" if kernel_type == "Matern" else "RBF"
MODEL_CONFIG["kernel_type"] = kernel_type
MODEL_CONFIG["kernel_nu"] = 2.5 if kernel_type == "Matern" else None
MODEL_CONFIG["ard"] = ard

MODEL_CONFIG["lengthscale_prior"] = prior_type 
MODEL_CONFIG["lengthscale_prior_params"] = (prior_param1, prior_param2) 

# --- TRAINING UND LOOCV ---
if st.button("Train Model and Run LOOCV"):
    if uploaded_file is None:
        st.warning("Please upload experiment data first.")
    else:
        uploaded_file.seek(0)
        raw_inputs, raw_outputs, df = load_csv_experiment_data(
            uploaded_file,
            columns_config=COLUMNS_CONFIG,
            output_column=output_col_name
        )
        scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds)
        input_dim = scaled_x.shape[1] 
        train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)

        model, likelihood = build_gp_model(train_x, train_y, input_dim, MODEL_CONFIG)
        fit_model(model)

        # Save model state
        st.session_state["model"] = model
        st.session_state["train_x"] = train_x
        st.session_state["train_y"] = train_y
        st.session_state["bounds"] = bounds

        # LOOCV Evaluation
        preds, truths, mse, ave_noise = loocv_gp_custom(
            train_x, train_y, train_x.shape[1], model_config=MODEL_CONFIG
        )

        # Save LOOCV results
        st.session_state["loocv_preds"] = preds
        st.session_state["loocv_truths"] = truths
        st.session_state["loocv_mse"] = mse
        st.session_state["loocv_noise"] = ave_noise

        st.success("LOOCV completed and saved.")

# --- DISPLAY LOOCV RESULTS BLOCK ---
if all(k in st.session_state for k in ["loocv_preds", "loocv_truths", "loocv_mse", "loocv_noise"]):
    st.subheader("LOOCV Results")
    st.success(f"LOOCV MSE: {st.session_state['loocv_mse']:.4f}")
    st.success(f"Average Learned Noise: {st.session_state['loocv_noise']:.6f}")
    learned_noise_var = st.session_state["model"].likelihood.noise.item()
    noise = st.session_state["model"].posterior(st.session_state["train_x"]).variance.sqrt().mean().item()

    learned_noise_std = learned_noise_var ** 0.5
    st.write(f"Learned Noise Variance (σ²): {learned_noise_var:.6f}")
    st.write(f"Learned Noise Std Dev (σ): {learned_noise_std:.4%}")
    st.write(f"Posterior Std Dev (σ²): {noise:.6f}")
    preds = st.session_state["loocv_preds"]
    truths = st.session_state["loocv_truths"]

    lengthscale = st.session_state["model"].covar_module.base_kernel.lengthscale

    # Falls die Lengthscale ein Tensor der Form [1, D] ist
    if lengthscale.ndim == 2:
        for i in range(lengthscale.shape[1]):
            st.write(f"Learned Lengthscale {i+1}: {lengthscale[0, i].item():.4f}")
    # Falls isotrop (ein einzelner Wert)
    else:
        st.write(f"Learned Lengthscale: {lengthscale.item():.4f}")

    st.line_chart({"True": truths, "Pred": preds})

    # Optional: CSV-Download
    import pandas as pd
    import io

    df_loocv = pd.DataFrame({
        "True": truths,
        "Predicted": preds
    })

    csv_loocv = df_loocv.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download LOOCV Predictions",
        data=csv_loocv,
        file_name="loocv_results.csv",
        mime="text/csv"
    )
if st.button("Run Full Grid Search"):
    if uploaded_file is None:
        st.warning("Please upload data first.")
    else:
        st.info("Grid search running ... this might take a while.")
        results_df = run_grid_search(data, output_col_name, bounds, base_config=MODEL_CONFIG)

        # Optional: Clean formatting
        if "Lengthscale_Prior" in results_df.columns:
            results_df["Lengthscale_Prior"] = results_df["Lengthscale_Prior"].apply(
                lambda x: str(x) if pd.notnull(x) else ""
            )

        # Save results
        st.session_state["grid_search_results"] = results_df
        st.session_state["best_result"] = results_df.sort_values(by="Mean_MSE_LOOCV").iloc[0]

        st.success("Grid search completed. Results stored.")

# --- DISPLAY RESULTS BLOCK ---
if "grid_search_results" in st.session_state:
    st.subheader("Grid Search Results")
    results_df = st.session_state["grid_search_results"]
    st.dataframe(results_df.sort_values(by="Mean_MSE_LOOCV").reset_index(drop=True))

    # CSV download
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Grid Search Results as CSV",
        data=csv_data,
        file_name="grid_search_results.csv",
        mime="text/csv"
    )

import re
# --- BEST MODEL BLOCK ---
if "best_result" in st.session_state:
    best_result = st.session_state["best_result"]

    st.subheader("Best Model Configuration")

    # 1. to dataframe
    best_df = pd.DataFrame([best_result])

    # Optional: Clean formatting
    column_order = [
        "Model_Name", "Input_Dim", "Kernel_Type",
        "Mean_MSE_LOOCV", "ARD", "Number_trainingpoints",
        "Prior_type", "Lengthscale_Prior", "Noise_value", "Notes"
    ]
    best_df = best_df[column_order]

    # 2. Show
    st.dataframe(best_df, use_container_width=True)


    # Optionaler Kommentar für Dateiname
    filename_note = st.text_input("Optional comment for export filename:", value="")

    # Sicherer Dateiname
    def sanitize_filename(comment):
        return re.sub(r"[^\w\-]+", "_", comment.strip())

    safe_note = sanitize_filename(filename_note)
    base_name = "best_model_config"
    file_prefix = f"{base_name}_{safe_note}" if safe_note else base_name

    # Download als CSV
    csv_data = best_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Best Config (CSV)",
        data=csv_data,
        file_name=f"{file_prefix}.csv",
        mime="text/csv"
    )

    # Download als JSON
    json_data = best_df.to_json(orient="records", indent=2)
    st.download_button(
        label="📥 Download Best Config (JSON)",
        data=json_data,
        file_name=f"{file_prefix}.json",
        mime="application/json"
    )
st.subheader("Run Bayesian Optimization Step")
st.session_state["uploaded_file"] = uploaded_file
if "uploaded_file" not in st.session_state:

    st.session_state.uploaded_file = None
prune_inf = st.checkbox("Prune Inferior Points", value=False)
show_full_table = st.checkbox("Show full updated experiment table", value=False)
if st.button("Suggest Next Experiment"):

    if st.session_state.uploaded_file is None:
        st.warning("Please upload data first.")
    else:
        # Reset file pointer
        st.session_state.uploaded_file.seek(0)

        # Load and process the uploaded experimental data
        uploaded_file.seek(0)
        raw_inputs, raw_outputs, df = load_csv_experiment_data(
            uploaded_file,
            columns_config=COLUMNS_CONFIG,
            output_column=output_col_name
        )


        # Prepare scaled data and tensors
        scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds)
        input_dim = scaled_x.shape[1]
        train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)

        # Build and fit the GP model
        model, likelihood = build_gp_model(
            train_x, train_y,
            input_dim=input_dim,
            config=MODEL_CONFIG
        )
        fit_model(model)
        st.session_state["model"] = model
        st.session_state["model"] = model
        st.session_state["train_x"] = train_x
        st.session_state["train_y"] = train_y
        st.session_state["bounds"] = bounds  # falls als List[Tuple[float, float]]

        # Prune inferior points if selected
        if prune_inf:
            x_baseline = prune_inferior_points(model, train_x)
            st.write("Pruned inferior points.")
        else:
            x_baseline = train_x
        st.session_state["x_baseline"] = x_baseline

        best_f = train_y.max().item()
        st.write(f"Best observed value: {best_f*100:.2f}%")
        st.write(f"Best training point: {train_x[train_y.argmax()].cpu().numpy()}")

        st.write("Suggesting next experiment...")
        try:
            new_x = optimize_qEI(
                model=st.session_state.model,
                input_dim=input_dim,
                best_f=best_f,
                batch_size=BATCH_SIZE,
                x_baseline=st.session_state.x_baseline,
                acquisition_type=acq_function,
                limit=-2
            )
            
            new_x = rescale_batch(new_x, bounds)
            st.success("Next experiment suggested!")

            st.session_state["last_suggestion"] = new_x

            updated_df = append_and_display_experiment(
                new_x_tensor=new_x,
                columns_config=COLUMNS_CONFIG,
                original_filename=st.session_state.uploaded_file,
                show_full_table=show_full_table,
                existing_df=data
            )

            st.session_state["last_updated_df"] = updated_df

         
        except Exception as e:
            st.error(f"Error during optimization: {e}")

if "last_suggestion" in st.session_state:
    st.subheader("Last Suggested Experiment(s)")
    st.dataframe(pd.DataFrame(
        st.session_state["last_suggestion"],
        columns=[v for k, v in COLUMNS_CONFIG.items() if not k.startswith("yield")]
    ))

if show_full_table and "last_updated_df" in st.session_state:
    st.subheader("Full Updated Experiment Table")
    st.dataframe(st.session_state["last_updated_df"])
        
st.subheader("Validate Bayesian Optimization Loop")

if st.button("Run BO Test"):
    with st.spinner("Running Bayesian Optimization..."):
        best_values, plot_path = run_bo_test(
            model_config=MODEL_CONFIG,
            acquisition_type=acq_function,
            batch_size=BATCH_SIZE,
            iterations=iterations,
            initial_points=initial_points,
            noise_std=noise_std
        )

    # Speichern der Ergebnisse
    st.session_state["bo_test_values"] = best_values
    st.session_state["bo_test_plot"] = plot_path

    st.success("Optimization completed!")

# Anzeige auch außerhalb des Buttons
if "bo_test_values" in st.session_state and "bo_test_plot" in st.session_state:
    st.subheader("Live Chart")
    st.line_chart(st.session_state["bo_test_values"])

    st.image(
        st.session_state["bo_test_plot"],
        caption=f"Convergence Plot ({acq_function})",
        use_column_width=True
    )

    # Optionaler Download-Link für das Plot-Bild
    with open(st.session_state["bo_test_plot"], "rb") as f:
        img_bytes = f.read()

    st.download_button(
        label="📥 Download Convergence Plot",
        data=img_bytes,
        file_name="convergence_plot.png",
        mime="image/png"
    )


from bo_utils.bo_plotting import plot_posterior_slices_streamlit, plot_interactive_slice

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
    st.subheader("Posterior Slices")
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
        model=st.session_state.model,
        train_x=st.session_state.train_x,
        bounds=st.session_state.bounds,
        param_names=param_names
    )
else:
    st.info("Please train the model and run BO at least once.")  

from botorch.acquisition import (
    qExpectedImprovement, qNoisyExpectedImprovement,
    qLogExpectedImprovement, qLogNoisyExpectedImprovement
) 



if acq_function == "qEI":
    acq_factory = lambda model: qExpectedImprovement(model, best_f=st.session_state["train_y"].max().item())
elif acq_function == "qNEI":
    acq_factory = lambda model: qNoisyExpectedImprovement(model, X_baseline=st.session_state["train_x"])

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
        acq_func_name = st.selectbox("Acquisition Function", ["qEI", "qNEI", "qLogEI", "qLogNEI"], key="vis_acq")

        if acq_func_name == "qEI":
            acq_factory = lambda model: qExpectedImprovement(model, best_f=train_y.max().item())
        elif acq_func_name == "qNEI":
            acq_factory = lambda model: qNoisyExpectedImprovement(model, X_baseline=train_x)
        elif acq_func_name == "qLogEI":
            acq_factory = lambda model: qLogExpectedImprovement(model, best_f=train_y.max().item())
        else:
            acq_factory = lambda model: qLogNoisyExpectedImprovement(model, X_baseline=train_x)

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