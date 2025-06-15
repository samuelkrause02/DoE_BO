import streamlit as st
import numpy as np
import pandas as pd
import math


def get_model_config_from_sidebar(data, model_config):
    """
    Updated sidebar function to generate new model config format
    
    Args:
        data: Uploaded data (for reference)
        model_config: Base model config (will be updated)
        
    Returns:
        model_config: Updated model configuration in new format
        acq_function: Selected acquisition function
        BATCH_SIZE: Batch size for acquisition
        iterations: Number of BO iterations
        initial_points: Number of initial points
        noise_std: Noise standard deviation for simulation
    """
    
    st.sidebar.header("🔧 Model Settings")

    # === KERNEL SETTINGS ===
    with st.sidebar.expander("🎯 Kernel Settings", expanded=False):
        kernel_type = st.selectbox("Kernel Type", ["RBF", "Matern", "Linear", "Polynomial"])
        
        # Kernel-specific parameters
        kernel_config = {"type": kernel_type}
        
        if kernel_type == "Matern":
            nu = st.selectbox("Matern ν parameter", [0.5, 1.5, 2.5], index=2)
            kernel_config["nu"] = nu
        elif kernel_type == "Polynomial":
            power = st.number_input("Polynomial power", min_value=1, max_value=5, value=2)
            kernel_config["power"] = power
            
        ard = st.checkbox("ARD (Automatic Relevance Determination)", value=True)

    # === NOISE PRIOR SETTINGS ===
    with st.sidebar.expander("🔊 Noise Prior (LogNormal)", expanded=False):
        # Option to choose parameter input method
        param_mode = st.radio("Parameter input method:", ["From mean noise", "Direct parameters"])
        
        if param_mode == "From mean noise":
            mean_noise = st.number_input(
                "Mean Noise (e.g. 0.01 = 1%)",
                min_value=1e-6,
                max_value=1.0,
                value=0.018316,
                step=0.001,
                format="%.6f"
            )
            log_scale = st.number_input(
                "Uncertainty (σ in log-space)",
                min_value=0.01,
                max_value=3.0,
                value=1.0,
                step=0.05,
                format="%.2f"
            )
            
            # Convert to LogNormal parameters
            loc = math.log(mean_noise)
            scale = log_scale
            
            st.write(f"**Resulting LogNormal Prior:**")
            st.write(f"- loc = log({mean_noise:.6f}) = {loc:.3f}")
            st.write(f"- scale = {scale:.2f}")
            
        else:  # Direct parameters
            loc = st.number_input(
                "Location (μ, log-space)",
                min_value=-10.0,
                max_value=0.0,
                value=-4.0,
                step=0.1,
                format="%.2f"
            )
            scale = st.number_input(
                "Scale (σ, log-space)",
                min_value=0.01,
                max_value=3.0,
                value=1.0,
                step=0.05,
                format="%.2f"
            )
            
            # Show expected noise level
            expected_noise = math.exp(loc + 0.5 * scale**2)
            st.write(f"**Expected noise level:** {expected_noise:.6f}")
        
        # Option to disable noise prior
        use_noise_prior = st.checkbox("Use noise prior", value=True)
        
        if use_noise_prior:
            noise_prior_config = {
                "type": "lognormal",
                "loc": loc,
                "scale": scale
            }
        else:
            noise_prior_config = None

    # === LENGTHSCALE PRIOR ===
    with st.sidebar.expander("📏 Lengthscale Prior", expanded=False):
        prior_type = st.selectbox("Prior Type", [None, "Gamma", "LogNormal"])
        
        if prior_type == "Gamma":
            concentration = st.number_input("Concentration (α)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
            rate = st.number_input("Rate (β)", min_value=0.1, max_value=20.0, value=6.0, step=0.1)
            
            lengthscale_prior_config = {
                "type": "gamma",
                "concentration": concentration,
                "rate": rate
            }
            
            # Show expected lengthscale
            expected_ls = concentration / rate
            st.write(f"**Expected lengthscale:** {expected_ls:.3f}")
            
        elif prior_type == "LogNormal":
            loc = st.number_input("Location (μ, log-space)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            scale = st.number_input("Scale (σ, log-space)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
            
            lengthscale_prior_config = {
                "type": "lognormal",
                "loc": loc,
                "scale": scale
            }
            
            # Show expected lengthscale
            expected_ls = math.exp(loc + 0.5 * scale**2)
            st.write(f"**Expected lengthscale:** {expected_ls:.3f}")
            
        else:
            lengthscale_prior_config = None

    # === OUTPUTSCALE SETTINGS ===
    with st.sidebar.expander("📊 Outputscale Settings", expanded=False):
        use_outputscale = st.checkbox("Use outputscale (ScaleKernel)", value=True)
        
        if use_outputscale:
            fix_outputscale = st.checkbox("Fix outputscale", value=False)
            
            if fix_outputscale:
                fixed_value = st.number_input("Fixed outputscale value", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                outputscale_config = {
                    "use": True,
                    "fixed": True,
                    "init_value": fixed_value
                }
            else:
                init_value = st.number_input("Initial outputscale value", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                outputscale_config = {
                    "use": True,
                    "fixed": False,
                    "init_value": init_value
                }
        else:
            outputscale_config = {"use": False}

    # === ACQUISITION FUNCTION ===
    with st.sidebar.expander("🎯 Acquisition Settings", expanded=False):
        acq_function = st.selectbox("Acquisition Function", ["qNEI", "qEI", "qGIBBON","qUCB"])
        BATCH_SIZE = st.slider("Batch Size", min_value=1, max_value=12, value=3, step=1)
        if acq_function == "qUCB":
            beta = st.number_input("Beta parameter for qUCB", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            st.write(f"Using Beta = {beta} for qUCB acquisition function.")
            st.session_state["beta"] = beta

    # === BO SIMULATION SETTINGS ===
    with st.sidebar.expander("🔄 Bayesian Optimization Test", expanded=False):
        iterations = st.slider("Number of iterations:", min_value=10, max_value=100, value=50, step=10)
        initial_points = st.slider("Number of initial points:", min_value=5, max_value=20, value=10)
        noise_std = st.slider("Noise standard deviation:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    # === BUILD NEW MODEL CONFIG ===
    new_model_config = {
        "kernel": kernel_config,
        "ard": ard,
        "lengthscale_prior": lengthscale_prior_config,
        "outputscale": outputscale_config,
        "noise_prior": noise_prior_config
    }

    # === CONFIGURATION SUMMARY ===
    st.sidebar.header("📋 Configuration Summary")
    
    with st.sidebar.expander("Current Settings", expanded=False):
        st.write(f"**Kernel:** {kernel_config['type']}")
        if kernel_config['type'] == 'Matern':
            st.write(f"  - ν: {kernel_config['nu']}")
        elif kernel_config['type'] == 'Polynomial':
            st.write(f"  - Power: {kernel_config['power']}")
            
        st.write(f"**ARD:** {ard}")
        
        if lengthscale_prior_config:
            st.write(f"**Lengthscale Prior:** {lengthscale_prior_config['type']}")
        else:
            st.write("**Lengthscale Prior:** None")
            
        if outputscale_config['use']:
            if outputscale_config['fixed']:
                st.write(f"**Outputscale:** Fixed at {outputscale_config['init_value']}")
            else:
                st.write("**Outputscale:** Learnable")
        else:
            st.write("**Outputscale:** Disabled")
            
        if noise_prior_config:
            st.write(f"**Noise Prior:** {noise_prior_config['type']}")
        else:
            st.write("**Noise Prior:** None")
        
        st.write(f"**Acquisition:** {acq_function}")
        st.write(f"**Batch Size:** {BATCH_SIZE}")

    return new_model_config, acq_function, BATCH_SIZE, iterations, initial_points, noise_std


def convert_legacy_to_new_config(legacy_config):
    """
    Convert old model config format to new format for backward compatibility
    
    Args:
        legacy_config: Old model configuration dictionary
        
    Returns:
        new_config: New model configuration dictionary
    """
    
    new_config = {
        "kernel": {"type": legacy_config.get("kernel_type", "RBF")},
        "ard": legacy_config.get("ard", True),
        "lengthscale_prior": None,
        "outputscale": {"use": True, "fixed": False},
        "noise_prior": None
    }
    
    # Handle Matern kernel
    if legacy_config.get("kernel_type") == "Matern":
        new_config["kernel"]["nu"] = legacy_config.get("kernel_nu", 2.5)
    
    # Handle lengthscale prior
    if legacy_config.get("lengthscale_prior"):
        if legacy_config["lengthscale_prior"] == "Gamma":
            params = legacy_config.get("lengthscale_prior_params", (2.0, 0.5))
            new_config["lengthscale_prior"] = {
                "type": "gamma",
                "concentration": params[0],
                "rate": params[1]
            }
        elif legacy_config["lengthscale_prior"] == "LogNormal":
            new_config["lengthscale_prior"] = {
                "type": "lognormal",
                "loc": 0.0,
                "scale": 1.0
            }
    
    # Handle noise prior
    if legacy_config.get("noise_prior_type") == "LogNormal":
        mean_noise = legacy_config.get("mean_noise", 0.01)
        scale_log = legacy_config.get("scale_log", 1.0)
        
        new_config["noise_prior"] = {
            "type": "lognormal",
            "loc": math.log(mean_noise),
            "scale": scale_log
        }
    
    return new_config


# Example usage with streamlit session state
def update_session_state_config():
    """Example of how to use this in your streamlit app"""
    
    # Initialize base config if not exists
    if "MODEL_CONFIG" not in st.session_state:
        st.session_state["MODEL_CONFIG"] = {
            "kernel": {"type": "RBF"},
            "ard": True,
            "lengthscale_prior": None,
            "outputscale": {"use": True, "fixed": False},
            "noise_prior": None
        }
    
    # Get data (placeholder)
    data = st.session_state.get("data", pd.DataFrame())
    
    # Update config from sidebar
    new_config, acq_func, batch_size, iterations, init_points, noise_std = get_model_config_from_sidebar(
        data, st.session_state["MODEL_CONFIG"]
    )
    
    # Store in session state
    st.session_state["MODEL_CONFIG"] = new_config
    st.session_state["acq_function"] = acq_func
    st.session_state["BATCH_SIZE"] = batch_size
    st.session_state["iterations"] = iterations
    st.session_state["initial_points"] = init_points
    st.session_state["noise_std"] = noise_std
    
    return new_config, acq_func, batch_size, iterations, init_points, noise_std


if __name__ == "__main__":
    # Test the function
    st.set_page_config(page_title="Model Config Test")
    
    # Create dummy data
    data = pd.DataFrame({
        'x1': np.random.randn(10),
        'x2': np.random.randn(10),
        'y': np.random.randn(10)
    })
    
    # Base config
    base_config = {
        "kernel": {"type": "RBF"},
        "ard": True,
        "lengthscale_prior": None,
        "outputscale": {"use": True, "fixed": False},
        "noise_prior": None
    }
    
    st.title("Model Configuration Test")
    
    # Get updated config
    config, acq_func, batch_size, iters, init_pts, noise = get_model_config_from_sidebar(data, base_config)
    
    # Display results
    st.subheader("Updated Configuration")
    st.json(config)
    
    st.subheader("Other Settings")
    st.write(f"Acquisition Function: {acq_func}")
    st.write(f"Batch Size: {batch_size}")
    st.write(f"Iterations: {iters}")
    st.write(f"Initial Points: {init_pts}")
    st.write(f"Noise Std: {noise}")