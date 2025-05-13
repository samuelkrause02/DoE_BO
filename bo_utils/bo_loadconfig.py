import json
import streamlit as st

import json
import ast
import streamlit as st

def load_model_config_from_json(json_file):
    """
    Reads a saved best_model_config.json and updates Streamlit session state
    with key model config values.
    """
    try:
        content = json_file.read().decode("utf-8")
        config_data = json.loads(content)

        if isinstance(config_data, list):
            config_data = config_data[0]

        st.session_state["restored_model_config"] = config_data

        # Optional: Apply parsed config to MODEL_CONFIG if needed
        from config import MODEL_CONFIG
        MODEL_CONFIG["name"] = config_data.get("Model_Name")
        MODEL_CONFIG["kernel_type"] = config_data.get("Kernel_Type")
        MODEL_CONFIG["ard"] = config_data.get("ARD")
        MODEL_CONFIG["noise_prior_range"] = tuple(config_data.get("Noise_Prior_Range", (0.0001, 0.2)))
        MODEL_CONFIG["lengthscale_prior"] = config_data.get("Prior_type")

        # --- HERE: safely parse Lengthscale_Prior ---
        raw_val = config_data.get("Lengthscale_Prior", None)
        if isinstance(raw_val, str):
            try:
                prior_tuple = ast.literal_eval(raw_val)
                p1, p2 = float(prior_tuple[0]), float(prior_tuple[1])
            except Exception as e:
                st.warning(f"Could not parse Lengthscale_Prior from string: {e}")
                p1, p2 = None, None
        elif isinstance(raw_val, list) or isinstance(raw_val, tuple):
            try:
                p1, p2 = float(raw_val[0]), float(raw_val[1])
            except:
                p1, p2 = None, None
        else:
            p1, p2 = None, None

        MODEL_CONFIG["lengthscale_prior_params"] = (p1, p2) if p1 and p2 else None
        
        raw_range = config_data.get("Noise_Prior_Range", (0.0001, 0.2))
        try:
            a, b = float(raw_range[0]), float(raw_range[1])
            if a < b:
                MODEL_CONFIG["noise_prior_range"] = (a, b)
            else:
                st.warning(f"Ignoring invalid noise_prior_range: {raw_range} (must have a < b)")
                MODEL_CONFIG["noise_prior_range"] = (0.0001, 0.2)  # fallback
        except Exception as e:
            st.warning(f"Could not parse noise_prior_range: {e}")
            MODEL_CONFIG["noise_prior_range"] = (0.0001, 0.2)
        
        return config_data

    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None

def init_ui_from_config(config):
    """
    Applies config values to Streamlit widget defaults via st.session_state.
    Should be called before widget definitions.
    """
    # Kernel Type
    if "Kernel_Type" in config:
        st.session_state["model_kernel_type"] = config["Kernel_Type"]

    # ARD
    if "ARD" in config:
        st.session_state["model_ard"] = config["ARD"]

    # Noise Prior Range
    if "Noise_Prior_Range" in config:
        min_noise, max_noise = config["Noise_Prior_Range"]
        st.session_state["noise_min"] = min_noise
        st.session_state["noise_max"] = max_noise

    # Prior Type
    if "Prior_type" in config and config["Prior_type"] is not None:
        st.session_state["prior_type"] = config["Prior_type"]

    # Prior Parameters
    if "Lengthscale_Prior" in config and isinstance(config["Lengthscale_Prior"], str):
        try:
            p1, p2 = map(float, config["Lengthscale_Prior"].split(","))
            st.session_state["prior_param1"] = p1
            st.session_state["prior_param2"] = p2
        except:
            pass