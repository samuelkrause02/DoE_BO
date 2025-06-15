from bo_utils.bo_plotting import plot_posterior_slices_streamlit, plot_interactive_slice
import streamlit as st
import torch
from botorch.acquisition import (
    qExpectedImprovement, qNoisyExpectedImprovement,
    qLogExpectedImprovement, qLogNoisyExpectedImprovement
) 
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

def show_posterior_slices_streamlit(bounds, data, output_col_name, acq_function,model_config):
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
            #model=st.session_state.model,
            train_x=st.session_state.train_x,
            train_y=st.session_state.train_y,
            bounds=st.session_state.bounds,
            param_names=param_names,
            model_config=model_config,
        )
    else:
        st.info("Please train the model and run BO at least once.")  






from bo_utils.bo_plotting import plot_2d_bo_contours_streamlit



def threeD_plotting(data, output_col_name):
    """
    Plotting function for 3D visualization of the data.
    """
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