import streamlit as st
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_optimization import optimize_qEI, optimize_qUCB  # qUCB hinzufügen
from bo_utils.bo_utils import rescale_batch, append_and_display_experiment
import pandas as pd
from botorch.acquisition.utils import prune_inferior_points
import torch

def bayesian_optimization_app(uploaded_file, output_col_name, batch_size, acq_function, bounds, columns_config, data, model_config, limit=-100000):
    st.subheader("Run Bayesian Optimization Step")
    st.session_state["uploaded_file"] = uploaded_file

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # 🆕 Option: Pending Experiments hochladen
    use_pending = st.checkbox("Upload pending experiments before next suggestion")
    if use_pending:
        pending_file = st.file_uploader("Upload CSV with pending experiments", type="csv", key="pending_uploader")
        if pending_file:
            pending_df = pd.read_csv(pending_file, index_col=0)
            # Use all columns except the output column and first column
            
            input_cols = [col for col in pending_df.columns if col != output_col_name]
            pending_x = torch.tensor(pending_df[input_cols].values, dtype=torch.double)
            st.session_state["pending_x"] = pending_x
            st.dataframe(pending_df[input_cols])
        else:
            pending_x = None
            st.session_state["pending_x"] = None
    else:
        pending_x = None
        st.session_state["pending_x"] = None
    
    
    prune_inf = st.checkbox("Prune Inferior Points", value=False)
    show_full_table = st.checkbox("Show full updated experiment table", value=False)

    if st.button("Suggest Next Experiment"):
        if st.session_state.uploaded_file is None:
            st.warning("Please upload data first.")
        else:
            st.session_state.uploaded_file.seek(0)
            train_x = st.session_state["train_x"]
            train_y = st.session_state["train_y"]
            input_dim = train_x.shape[1]

            model, likelihood = build_gp_model(train_x, train_y, config=model_config)
            fit_model(model)
            st.session_state["model"] = model
            st.session_state["bounds"] = bounds

            x_baseline = prune_inferior_points(model, train_x) if prune_inf else train_x

            # 🆕 Falls pending_x vorhanden ist → an Baseline anhängen
            if st.session_state["pending_x"]is not None:
                x_baseline = torch.cat([x_baseline, st.session_state["pending_x"]], dim=0)
                st.info(f"Including {st.session_state['pending_x'].shape[0]} pending points in acquisition baseline.")

            st.session_state["x_baseline"] = x_baseline

            best_f = train_y.max().item()
            st.write(f"Best observed value: {best_f*100:.2f}%")
            st.write(f"Best training point: {train_x[train_y.argmax()].cpu().numpy()}")

            st.write("Suggesting next experiment...")
            try:
                if acq_function in ["qEI", "qNEI", "qGIBBON"]:
                    new_x = optimize_qEI(
                        model=st.session_state.model,
                        input_dim=input_dim,
                        best_f=best_f,
                        batch_size=batch_size,
                        x_baseline=st.session_state.x_baseline,
                        acquisition_type=acq_function,
                        limit=limit,
                        X_pending=st.session_state.get("pending_x", None)
                    )
                elif acq_function == "qUCB":
                    new_x = optimize_qUCB(
                        model=st.session_state.model,
                        input_dim=input_dim,
                        batch_size=batch_size,
                        limit=limit,
                        beta = st.session_state['beta']  # oder als Parameter übergeben
                    )
                st.session_state["new_x"] = new_x
                new_x = rescale_batch(new_x, bounds)
                st.success("Next experiment suggested!")

                st.session_state["last_suggestion"] = new_x
                updated_df = append_and_display_experiment(
                    new_x_tensor=new_x,
                    columns_config=columns_config,
                    original_filename=st.session_state.uploaded_file,
                    show_full_table=show_full_table,
                    existing_df=data
                )
                st.session_state["last_updated_df"] = updated_df

            except Exception as e:
                st.error(f"Error during optimization: {e}")

    param_names = [col for col in data.columns if col != output_col_name]

    if "last_suggestion" in st.session_state:
        suggestion = st.session_state["last_suggestion"]
        if suggestion.shape[1] != len(param_names):
            st.warning(f"Shape mismatch: {suggestion.shape[1]} values, {len(param_names)} columns: {param_names}")
        st.subheader("Last Suggested Experiment(s)")
        st.dataframe(pd.DataFrame(suggestion, columns=param_names))

    if show_full_table and "last_updated_df" in st.session_state:
        st.subheader("Full Updated Experiment Table")
        st.dataframe(st.session_state["last_updated_df"])