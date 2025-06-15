
import streamlit as st
import pandas as pd
from bo_utils.bo_model import build_gp_model, fit_model, prepare_training_tensors
from bo_utils.bo_validation import loocv_gp_custom
from streamlit_app.filtering import apply_loocv_based_filtering, show_model_summary
from bo_utils.bo_data import load_csv_experiment_data, prepare_training_data
from config import COLUMNS_CONFIG
def loocv_tab():
    st.subheader("Model Training and LOOCV")

    # 1. Konfiguration
    percent_remove = st.slider("Remove top x% worst LOOCV points (MSE)", 0, 50, 0)
    percent_noise_remove = st.slider("Remove x% by strongest negative Noise influence", 0, 50, 0)
    percent_looph_remove = st.slider("Remove x% by highest Standardized Residuals Points", 0, 50, 0)
    percent_gradient_remove = st.slider("Remove x% of closest high-gradient pairs", 0, 50, 0)

    # 2. Lade Inputs
    uploaded_file = st.session_state.get("uploaded_file")
    output_col = st.session_state.get("output_col_name")
    bounds = st.session_state.get("bounds")
    model_config = st.session_state.get("MODEL_CONFIG")

    if not uploaded_file or not output_col or not bounds:
        st.warning("Missing data or configuration.")
        return

    # 3. Button Trigger
    if st.button("Train Model and Run LOOCV"):

        # --- Vorbereitung
        
        uploaded_file.seek(0)
        raw_inputs, raw_outputs, _ = load_csv_experiment_data(uploaded_file, output_column=output_col, columns_config=COLUMNS_CONFIG)
        scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds)
        train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)
        input_dim = train_x.shape[1]

        # 4. LOOCV + Entfernen
        train_x, train_y, removal_logs = apply_loocv_based_filtering(
            train_x, train_y,
            model_config,
            bounds=bounds,
            percent_mse=percent_remove,
            percent_noise=percent_noise_remove,
            percent_looph=percent_looph_remove,
            percent_gradient=percent_gradient_remove
        )

        # 5. Finales Modell
        model, _ = build_gp_model(train_x, train_y, model_config)
        fit_model(model)
        preds, truths, mse, ave_noise = loocv_gp_custom(train_x, train_y, input_dim, model_config)

        # 6. Speicherung
        st.session_state.update({
            "model": model,
            "train_x": train_x,
            "train_y": train_y,
            "loocv_preds": preds,
            "loocv_truths": truths,
            "loocv_mse": mse,
            "loocv_noise": ave_noise,
            "bounds": bounds
        })

        st.success(f"Final MSE/MAE: {mse:.4f} / {mse**0.5*100:.2f}%")
        for label, df in removal_logs.items():
            st.subheader(label)
            st.dataframe(df)

    # 7. Ergebnisse anzeigen
    if "loocv_preds" in st.session_state:
        show_model_summary(st.session_state)


