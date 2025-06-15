import streamlit as st
import pandas as pd
import numpy as np
import math
from config import MODEL_CONFIG  # Base config

def model_setup_tab():
    st.sidebar.header("Upload Experiment Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file, sep=";")
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["data"] = data
        
        st.subheader("Uploaded Data")
        st.dataframe(data)
        
        st.subheader("Select Output Column")
        output_col_name = st.selectbox("Choose the output column", data.columns, key="output_col")
        st.session_state["output_col_name"] = output_col_name
        
        if output_col_name:
            st.write(f"Selected output column: `{output_col_name}`")
            
            st.subheader("Specify Parameter Bounds")
            bounds = []
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
            st.session_state["bounds"] = bounds
            st.session_state["param_columns"] = param_columns
    
    # =============================================================================
    # MODEL CONFIGURATION SIDEBAR
    # =============================================================================
    
    st.sidebar.header("🔧 Model Configuration")
    
    # Initialize config if not exists
    if "MODEL_CONFIG" not in st.session_state:
        st.session_state["MODEL_CONFIG"] = {
            "kernel": {"type": "RBF"},
            "ard": True,
            "lengthscale_prior": None,
            "outputscale": {"use": True, "fixed": False},
            "noise_prior": None
        }
    
    config = st.session_state["MODEL_CONFIG"]
    
    # === KERNEL SETTINGS ===
    with st.sidebar.expander("🎯 Kernel Settings", expanded=True):
        kernel_type = st.selectbox("Kernel Type", ["RBF", "Matern", "Linear", "Polynomial"], 
                                 index=0 if config["kernel"]["type"] == "RBF" else 
                                       1 if config["kernel"]["type"] == "Matern" else
                                       2 if config["kernel"]["type"] == "Linear" else 3)
        
        # Kernel-specific parameters
        kernel_config = {"type": kernel_type}
        
        if kernel_type == "Matern":
            nu = st.selectbox("Matern ν parameter", [0.5, 1.5, 2.5], index=2)
            kernel_config["nu"] = nu
        elif kernel_type == "Polynomial":
            power = st.number_input("Polynomial power", min_value=1, max_value=5, value=2)
            kernel_config["power"] = power
        
        # ARD setting
        ard = st.checkbox("Enable ARD (Automatic Relevance Determination)", 
                         value=config.get("ard", True))
        
        config["kernel"] = kernel_config
        config["ard"] = ard
    
    # === LENGTHSCALE PRIOR ===
    with st.sidebar.expander("📏 Lengthscale Prior", expanded=False):
        use_ls_prior = st.checkbox("Use lengthscale prior", value=config["lengthscale_prior"] is not None)
        
        if use_ls_prior:
            ls_prior_type = st.selectbox("Prior Type", ["gamma", "lognormal"])
            
            if ls_prior_type == "gamma":
                concentration = st.number_input("Concentration (α)", min_value=0.1, max_value=10.0, 
                                              value=2.0, step=0.1, format="%.2f")
                rate = st.number_input("Rate (β)", min_value=0.01, max_value=5.0, 
                                     value=0.5, step=0.01, format="%.3f")
                
                config["lengthscale_prior"] = {
                    "type": "gamma",
                    "concentration": concentration,
                    "rate": rate
                }
                
                # Show expected lengthscale
                expected_ls = concentration / rate
                st.write(f"Expected lengthscale: {expected_ls:.3f}")
                
            elif ls_prior_type == "lognormal":
                loc = st.number_input("Location (μ)", min_value=-3.0, max_value=3.0, 
                                    value=0.0, step=0.1, format="%.2f")
                scale = st.number_input("Scale (σ)", min_value=0.1, max_value=3.0, 
                                      value=1.0, step=0.1, format="%.2f")
                
                config["lengthscale_prior"] = {
                    "type": "lognormal",
                    "loc": loc,
                    "scale": scale
                }
                
                # Show expected lengthscale
                expected_ls = math.exp(loc + 0.5 * scale**2)
                st.write(f"Expected lengthscale: {expected_ls:.3f}")
        else:
            config["lengthscale_prior"] = None
    
    # === OUTPUTSCALE SETTINGS ===
    with st.sidebar.expander("📊 Outputscale Settings", expanded=False):
        use_outputscale = st.checkbox("Use outputscale (ScaleKernel)", 
                                    value=config["outputscale"].get("use", True))
        
        if use_outputscale:
            fix_outputscale = st.checkbox("Fix outputscale", 
                                        value=config["outputscale"].get("fixed", False))
            
            if fix_outputscale:
                fixed_value = st.number_input("Fixed outputscale value", 
                                            min_value=0.01, max_value=10.0, 
                                            value=config["outputscale"].get("init_value", 1.0), 
                                            step=0.01, format="%.3f")
                config["outputscale"] = {
                    "use": True,
                    "fixed": True,
                    "init_value": fixed_value
                }
            else:
                init_value = st.number_input("Initial outputscale value (optional)", 
                                           min_value=0.01, max_value=10.0, 
                                           value=1.0, step=0.01, format="%.3f")
                config["outputscale"] = {
                    "use": True,
                    "fixed": False,
                    "init_value": init_value
                }
        else:
            config["outputscale"] = {"use": False}
    
    # === NOISE PRIOR ===
    with st.sidebar.expander("🔊 Noise Prior", expanded=False):
        use_noise_prior = st.checkbox("Use noise prior", value=config["noise_prior"] is not None)
        
        if use_noise_prior:
            noise_prior_type = st.selectbox("Noise Prior Type", ["lognormal", "gamma"])
            
            if noise_prior_type == "lognormal":
                st.write("**LogNormal Prior**: noise ~ LogNormal(loc, scale)")
                
                # Option 1: Direct parameters
                col1, col2 = st.columns(2)
                with col1:
                    param_mode = st.radio("Parameter mode", ["Direct", "From mean noise"])
                
                if param_mode == "Direct":
                    loc = st.number_input("Location (μ, log-space)", 
                                        min_value=-10.0, max_value=0.0, 
                                        value=-6.13, step=0.1, format="%.2f")
                    scale = st.number_input("Scale (σ, log-space)", 
                                          min_value=0.01, max_value=2.0, 
                                          value=0.25, step=0.01, format="%.3f")
                else:
                    mean_noise = st.number_input("Mean noise level", 
                                               min_value=1e-6, max_value=0.1, 
                                               value=0.0016, step=0.0001, format="%.6f")
                    scale = st.number_input("Uncertainty (σ, log-space)", 
                                          min_value=0.01, max_value=2.0, 
                                          value=0.25, step=0.01, format="%.3f")
                    loc = math.log(mean_noise)
                
                config["noise_prior"] = {
                    "type": "lognormal",
                    "loc": loc,
                    "scale": scale
                }
                
                # Show expected noise
                expected_noise = math.exp(loc + 0.5 * scale**2)
                st.write(f"Expected noise: {expected_noise:.6f}")
                
            elif noise_prior_type == "gamma":
                st.write("**Gamma Prior**: noise ~ Gamma(α, β)")
                
                concentration = st.number_input("Concentration (α)", 
                                              min_value=0.1, max_value=10.0, 
                                              value=2.0, step=0.1, format="%.2f")
                rate = st.number_input("Rate (β)", 
                                     min_value=0.1, max_value=1000.0, 
                                     value=10.0, step=0.1, format="%.2f")
                
                config["noise_prior"] = {
                    "type": "gamma",
                    "concentration": concentration,
                    "rate": rate
                }
                
                # Show expected noise
                expected_noise = concentration / rate
                st.write(f"Expected noise: {expected_noise:.6f}")
        else:
            config["noise_prior"] = None
    
    # === ACQUISITION SETTINGS ===
    with st.sidebar.expander("🎯 Acquisition Settings", expanded=False):
        acq_function = st.selectbox("Acquisition Function", 
                                   ["qNEI", "qEI", "qUCB", "qPI"], 
                                   index=0)
        batch_size = st.slider("Batch Size", min_value=1, max_value=10, value=3, step=1)
        
        st.session_state["acq_function"] = acq_function
        st.session_state["BATCH_SIZE"] = batch_size
    
    # Update session state
    st.session_state["MODEL_CONFIG"] = config
    
    # === CONFIGURATION PREVIEW ===
    st.sidebar.header("📋 Current Configuration")
    
    with st.sidebar.expander("View Current Config", expanded=False):
        st.json(config)
    
    # Configuration summary
    st.sidebar.write("**Summary:**")
    st.sidebar.write(f"- Kernel: {config['kernel']['type']}")
    if config['kernel']['type'] == 'Matern':
        st.sidebar.write(f"  - ν: {config['kernel']['nu']}")
    st.sidebar.write(f"- ARD: {config['ard']}")
    st.sidebar.write(f"- Lengthscale prior: {config['lengthscale_prior']['type'] if config['lengthscale_prior'] else 'None'}")
    st.sidebar.write(f"- Outputscale: {'Fixed' if config['outputscale'].get('fixed') else 'Learnable' if config['outputscale'].get('use') else 'Disabled'}")
    st.sidebar.write(f"- Noise prior: {config['noise_prior']['type'] if config['noise_prior'] else 'None'}")
    
    # === CONFIGURATION EXPORT/IMPORT ===
    st.sidebar.header("💾 Config Management")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📥 Export Config"):
            config_str = str(config).replace("'", '"')
            st.sidebar.download_button(
                label="Download Config",
                data=config_str,
                file_name="model_config.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🔄 Reset Config"):
            st.session_state["MODEL_CONFIG"] = {
                "kernel": {"type": "RBF"},
                "ard": True,
                "lengthscale_prior": None,
                "outputscale": {"use": True, "fixed": False},
                "noise_prior": None
            }
            st.experimental_rerun()


def show_config_summary():
    """Show a summary of the current model configuration"""
    if "MODEL_CONFIG" not in st.session_state:
        st.warning("No model configuration found. Please set up the model first.")
        return
    
    config = st.session_state["MODEL_CONFIG"]
    
    st.subheader("🔧 Current Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kernel Configuration:**")
        st.write(f"- Type: {config['kernel']['type']}")
        if config['kernel']['type'] == 'Matern':
            st.write(f"- ν parameter: {config['kernel']['nu']}")
        elif config['kernel']['type'] == 'Polynomial':
            st.write(f"- Power: {config['kernel']['power']}")
        st.write(f"- ARD: {config['ard']}")
        
        st.write("**Outputscale:**")
        if config['outputscale']['use']:
            if config['outputscale']['fixed']:
                st.write(f"- Fixed at: {config['outputscale']['init_value']}")
            else:
                st.write("- Learnable")
        else:
            st.write("- Disabled")
    
    with col2:
        st.write("**Lengthscale Prior:**")
        if config['lengthscale_prior']:
            prior = config['lengthscale_prior']
            st.write(f"- Type: {prior['type']}")
            if prior['type'] == 'gamma':
                st.write(f"- Concentration: {prior['concentration']}")
                st.write(f"- Rate: {prior['rate']}")
            elif prior['type'] == 'lognormal':
                st.write(f"- Location: {prior['loc']}")
                st.write(f"- Scale: {prior['scale']}")
        else:
            st.write("- None")
        
        st.write("**Noise Prior:**")
        if config['noise_prior']:
            prior = config['noise_prior']
            st.write(f"- Type: {prior['type']}")
            if prior['type'] == 'gamma':
                st.write(f"- Concentration: {prior['concentration']}")
                st.write(f"- Rate: {prior['rate']}")
            elif prior['type'] == 'lognormal':
                st.write(f"- Location: {prior['loc']:.3f}")
                st.write(f"- Scale: {prior['scale']:.3f}")
        else:
            st.write("- None")
    
    # Show acquisition settings if available
    if "acq_function" in st.session_state:
        st.write("**Acquisition Function:**")
        st.write(f"- Function: {st.session_state['acq_function']}")
        st.write(f"- Batch Size: {st.session_state['BATCH_SIZE']}")


# Example usage in main streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="GP Model Setup", layout="wide")
    
    tab1, tab2 = st.tabs(["Model Setup", "Configuration Summary"])
    
    with tab1:
        model_setup_tab()
    
    with tab2:
        show_config_summary()