import streamlit as st
from bo_utils.bo_utils import rescale_single_point
import pandas as pd
import numpy as np


def show_loocv_results(bounds):
    """Display LOOCV results with compact layout"""
    
    required_keys = ["loocv_preds", "loocv_truths", "loocv_mse", "loocv_noise", "model", "train_x"]
    
    if not all(k in st.session_state for k in required_keys):
        st.info("Run LOOCV training first to see results")
        return
    
    st.subheader("LOOCV Results")
    
    # Get data from session state
    mse = st.session_state["loocv_mse"]
    preds = st.session_state["loocv_preds"]
    truths = st.session_state["loocv_truths"]
    model = st.session_state["model"]
    
    # Convert to numpy arrays (handle tensors)
    preds_np = np.array([p.item() if hasattr(p, 'item') else p for p in preds])
    truths_np = np.array([t.item() if hasattr(t, 'item') else t for t in truths])
    
    # Main metrics
    mae = np.mean(np.abs(preds_np - truths_np))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((truths_np - preds_np)**2) / 
              np.sum((truths_np - np.mean(truths_np))**2))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", f"{mse:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.metric("R²", f"{r2:.3f}")
    with col4:
        noise_var = model.likelihood.noise.mean().item() * (model.outcome_transform.stdvs.item() ** 2)
        st.metric("Noise σ²", f"{noise_var:.4f}")
    
    # Prediction chart
    chart_data = pd.DataFrame({
        'True': truths_np,
        'Predicted': preds_np
    })
    st.line_chart(chart_data)
    
    # Advanced details in expanders
    with st.expander("Model Details"):
        y_std = model.outcome_transform.stdvs.item()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Noise Analysis:**")
            noise_with_obs = model.posterior(st.session_state["train_x"], observation_noise=True).variance.sqrt().mean().item()
            noise_without_obs = model.posterior(st.session_state["train_x"], observation_noise=False).variance.sqrt().mean().item()
            st.write(f"• With observation noise: {noise_with_obs:.4f}")
            st.write(f"• Without observation noise: {noise_without_obs:.4f}")
            st.session_state["noise_with_obs"] = noise_with_obs
        
        with col2:
            st.write("**Lengthscales:**")
            lengthscale = model.covar_module.base_kernel.lengthscale
            lengthscale_rescaled = rescale_single_point(lengthscale.detach().numpy(), bounds)
            
            if lengthscale_rescaled.ndim == 2 and lengthscale_rescaled.shape[1] > 1:
                for i in range(lengthscale_rescaled.shape[1]):
                    st.write(f"• Dim {i+1}: {lengthscale_rescaled[0, i].item():.4f}")
            else:
                st.write(f"• Single: {lengthscale_rescaled.item():.4f}")
            
            if hasattr(model.covar_module, 'outputscale'):
                st.write(f"• Output scale: {model.covar_module.outputscale.item():.4f}")
    
    with st.expander("Export Data"):
        results_df = pd.DataFrame({
            'True': truths_np,
            'Predicted': preds_np,
            'Residual': preds_np - truths_np
        })
        
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="loocv_results.csv",
            mime="text/csv"
        )