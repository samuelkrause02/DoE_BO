import streamlit as st
import pandas as pd
from bo_utils.bo_orthogonal_sampling import (
    build_parameter_space,
    generate_orthogonal_samples,
    analyze_sample
)

def normalize_data_to_bounds(df, param_defs):
    """
    Normalize each column of df to [0, 1] using original user-defined bounds.
    """
    normalized_df = df.copy()
    for i, (name, low, high, _) in enumerate(param_defs):
        normalized_df[name] = (df[name] - low) / (high - low)
    return normalized_df

def show_sampling_ui():
    """
    Streamlit UI component for defining parameters and performing orthogonal sampling.
    Returns:
        df_raw (pd.DataFrame): Raw sampled values
        df_norm (pd.DataFrame): Normalized sampled values
        param_defs (List): List of (name, low, high, is_discrete) for further use
    """
    df_raw, df_norm, param_defs = None, None, []

    with st.expander("Orthogonal Sampling (click to expand)", expanded=False):
        st.header("1. Orthogonal Sampling")

        n_params = st.number_input(
            "How many variables do you want to sample?",
            min_value=1,
            max_value=20,
            value=3,
            step=1
        )

        with st.form("sampling_form"):
            names = []
            lowers = []
            uppers = []
            is_discretes = []

            for i in range(n_params):
                st.subheader(f"Variable {i+1}")
                name = st.text_input(f"Name of variable {i+1}", value=f"Var{i+1}", key=f"name_{i}")
                low = st.number_input(f"Lower bound of {name}", value=0.0, key=f"low_{i}")
                high = st.number_input(f"Upper bound of {name}", value=1.0, key=f"high_{i}")
                is_disc = st.checkbox(f"Is {name} discrete?", value=False, key=f"disc_{i}")

                names.append(name)
                lowers.append(low)
                uppers.append(high)
                is_discretes.append(is_disc)

            n_samples = st.number_input("Number of samples", min_value=2, max_value=100, value=10)
            run_sampling = st.form_submit_button("Generate samples")

        if run_sampling:
            param_defs = [(names[i], lowers[i], uppers[i], is_discretes[i]) for i in range(n_params)]
            try:
                space, col_names = build_parameter_space(param_defs)
                df_raw = generate_orthogonal_samples(space, n_samples, col_names)

                st.subheader("Orthogonal Samples (Raw)")
                st.dataframe(df_raw)

                df_norm = normalize_data_to_bounds(df_raw, param_defs)
                st.subheader("Normalized Samples (w.r.t. bounds)")
                st.dataframe(df_norm)

                st.subheader("Sample Analysis")
                coverage_1d, discrepancy, min_dist = analyze_sample(df_norm)
                st.markdown(f"""
                **1D Coverage**: {coverage_1d:.3f}  
                **L2-Star Discrepancy**: {discrepancy:.3f}  
                **Min. Distance (1-norm)**: {min_dist:.3f}
                """)

            except Exception as e:
                st.error(f"Sampling failed: {e}")

    return df_raw, df_norm, param_defs