import streamlit as st
from bo_utils.bo_test import run_bo_test    

def run_bo_test_loop( model_config, acquisition_type, batch_size, iterations, initial_points, noise_std):
    """
    Run a Bayesian Optimization test with the given parameters.
    """
    st.subheader("Validate Bayesian Optimization Loop")

    if st.button("Run BO Test"):
        with st.spinner("Running Bayesian Optimization..."):
            best_values, plot_path = run_bo_test(
                model_config=model_config,
                acquisition_type=acquisition_type,
                batch_size=batch_size,
                iterations=iterations,
                initial_points=initial_points,
                noise_std=noise_std
            )

        # Save results in session state
        st.session_state["bo_test_values"] = best_values
        st.session_state["bo_test_plot"] = plot_path

        st.success("Optimization completed!")

    # Display results
    if "bo_test_values" in st.session_state and "bo_test_plot" in st.session_state:
        st.subheader("Live Chart")
        st.line_chart(st.session_state["bo_test_values"])

        st.image(
            st.session_state["bo_test_plot"],
            caption=f"Convergence Plot ({acquisition_type})",
            use_column_width=True
        )

        # Download button for the plot
        with open(st.session_state["bo_test_plot"], "rb") as f:
            img_bytes = f.read()

        st.download_button(
            label="📥 Download Convergence Plot",
            data=img_bytes,
            file_name="convergence_plot.png",
            mime="image/png"
        )