import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_posterior_slices(model, train_x, bounds, param_names=None):
    """
    Plots 1D posterior slices for each parameter in the model.
    
    model: trained GP model
    train_x: tensor of training inputs (scaled)
    bounds: list of [min, max] pairs for each parameter
    param_names: optional list of parameter names
    """
    input_dim = train_x.shape[1]
    if param_names is None:
        param_names = [f"x{i+1}" for i in range(input_dim)]
    
    for k in range(input_dim):
        # Generate grid for this parameter
        grid = torch.linspace(0, 1, 100)
        
        # Fix other parameters at their mean
        fixed = train_x.mean(dim=0, keepdim=True).repeat(100, 1)
        fixed[:, k] = grid

        # Posterior prediction
        posterior = model.posterior(fixed)
        mean = posterior.mean.detach().numpy().flatten()
        std = posterior.variance.sqrt().detach().numpy().flatten()
        
        # Rescale x-axis to original bounds
        min_val, max_val = bounds[k]
        x_orig = grid.numpy() * (max_val - min_val) + min_val
        train_x_orig = train_x[:, k].numpy() * (max_val - min_val) + min_val

        # Rescale output (if you want %)
        y_mean = mean * 100
        err_upper = (mean + std) * 100
        err_lower = (mean - std) * 100

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(x_orig, y_mean, 'b-', label='Predicted Mean')
        plt.fill_between(x_orig, err_upper, err_lower, alpha=0.3, label='Uncertainty Band')
        sns.rugplot(train_x_orig, height=0.1, color='red', label='Training Points')
        
        plt.xlabel(f"{param_names[k]} (Original Scale)")
        plt.ylabel('Predicted Yield [%]')
        plt.title(f"Effect of {param_names[k]} on Yield")
        plt.legend()
        plt.tight_layout()
        plt.show()



import pandas as pd
import matplotlib.pyplot as plt

def plot_pairwise(df):
    """
    Creates a scatter matrix (pairplot) for the dataframe.
    df: pandas DataFrame including parameters + output
    """
    sns.pairplot(df)
    plt.suptitle("Pairwise Scatter Matrix", y=1.02)
    plt.show()


import plotly.graph_objs as go
import plotly.offline as pyo

def plot_3d_scatter_interactive(raw_inputs, raw_outputs, param_x_idx, param_y_idx, param_names, filename="3d_scatter.html"):
    """
    Creates an interactive 3D scatter plot and saves as HTML.
    """
    x = raw_inputs[:, param_x_idx]
    y = raw_inputs[:, param_y_idx]
    z = raw_outputs

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title=f"3D Scatter: {param_names[param_x_idx]} vs. {param_names[param_y_idx]} vs. Yield",
        scene=dict(
            xaxis_title=param_names[param_x_idx],
            yaxis_title=param_names[param_y_idx],
            zaxis_title='Yield'
        )
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Show in notebook or browser
    fig.show()

    # Save as HTML
    pyo.plot(fig, filename=filename, auto_open=False)
    print(f"Interactive plot saved as {filename}")



def plot_yield_distribution(raw_outputs):
    """
    Plots a histogram of the output yield values.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(raw_outputs, bins=15, kde=True)
    plt.xlabel("Yield")
    plt.title("Distribution of Yield Values")
    plt.tight_layout()
    plt.show()




# ----- 2D Contour Plots for GP Posterior Mean, Std, and Acquisition Function ----

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from botorch.acquisition import ExpectedImprovement
import plotly.graph_objs as go
import plotly.offline as pyo

def plot_2d_bo_contours(
    model,
    train_x,
    train_y,
    bounds,
    param_x_idx,
    param_y_idx,
    param_names=None,
    n_grid=50,
    acquisition_func_factory=None,
    save_html=False,
    html_filename="bo_2d_contour.html"
):
    """
    Plots 2D contour plots for GP Posterior Mean, Std, and Acquisition Function.

    acquisition_func_factory: function that takes (model) and returns an acquisition function.
    save_html: if True, saves an interactive Plotly version as HTML.
    """
    if param_names is None:
        param_names = [f"x{i+1}" for i in range(train_x.shape[1])]

    x1 = torch.linspace(0, 1, n_grid)
    x2 = torch.linspace(0, 1, n_grid)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    fixed = train_x.mean(dim=0)
    X_full = torch.zeros(X_grid.shape[0], train_x.shape[1])
    X_full[:, param_x_idx] = X_grid[:, 0]
    X_full[:, param_y_idx] = X_grid[:, 1]
    for j in range(train_x.shape[1]):
        if j not in [param_x_idx, param_y_idx]:
            X_full[:, j] = fixed[j]

    model.eval()
    posterior = model.posterior(X_full)
    mean = posterior.mean.detach().numpy()
    std = posterior.variance.sqrt().detach().numpy()

    if acquisition_func_factory is None:
        acq = ExpectedImprovement(model=model, best_f=train_y.max().item())
    else:
        acq = acquisition_func_factory(model)

    ei = acq(X_full.unsqueeze(1)).detach().numpy()

    mean_plot = mean.reshape(n_grid, n_grid)
    std_plot = std.reshape(n_grid, n_grid)
    ei_plot = ei.reshape(n_grid, n_grid)

    def unnormalize(vals, i):
        return vals * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    x1_orig = unnormalize(x1.numpy(), param_x_idx)
    x2_orig = unnormalize(x2.numpy(), param_y_idx)
    X1_orig, X2_orig = np.meshgrid(x1_orig, x2_orig)

    train_proj = train_x[:, [param_x_idx, param_y_idx]].numpy()

    def contour_plot(data, title, xlabel, ylabel):
        plt.figure(figsize=(6, 5))
        cp = plt.contourf(X1_orig, X2_orig, data, levels=20, cmap=cm.viridis)
        plt.scatter(
            unnormalize(train_proj[:, 0], param_x_idx),
            unnormalize(train_proj[:, 1], param_y_idx),
            c='red', s=25, label="Training Points", edgecolor='k'
        )
        plt.colorbar(cp)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    contour_plot(mean_plot, "GP Posterior Mean", param_names[param_x_idx], param_names[param_y_idx])
    contour_plot(std_plot, "Model Uncertainty (Std)", param_names[param_x_idx], param_names[param_y_idx])
    contour_plot(ei_plot, "Acquisition Function", param_names[param_x_idx], param_names[param_y_idx])

    if save_html:
        fig = go.Figure(data=[
            go.Surface(
                z=mean_plot,
                x=x1_orig,
                y=x2_orig,
                colorscale='Viridis',
                colorbar=dict(title='Posterior Mean')
            )
        ])
        fig.update_layout(
            title=f"GP Posterior Mean: {param_names[param_x_idx]} vs. {param_names[param_y_idx]}",
            scene=dict(
                xaxis_title=param_names[param_x_idx],
                yaxis_title=param_names[param_y_idx],
                zaxis_title='Posterior Mean'
            )
        )
        pyo.plot(fig, filename=html_filename, auto_open=False)
        print(f"Interactive HTML plot saved as {html_filename}")



def plot_posterior_slices_streamlit(model, train_x, bounds, param_names=None):
    import matplotlib.pyplot as plt
    import torch
    import seaborn as sns
    import numpy as np
    import streamlit as st

    input_dim = train_x.shape[1]
    if param_names is None:
        param_names = [f"x{i+1}" for i in range(input_dim)]

    st.subheader("Posterior Slice Visualization")

    selected = st.multiselect(
        "Select parameters to visualize:",
        options=param_names,
        default=param_names
    )

    for k, name in enumerate(param_names):
        if name not in selected:
            continue

        with st.expander(f"📈 Effect of {name}"):
            grid = torch.linspace(0, 1, 100)
            fixed = train_x.mean(dim=0, keepdim=True).repeat(100, 1)
            fixed[:, k] = grid

            posterior = model.posterior(fixed)
            mean = posterior.mean.detach().numpy().flatten()
            std = posterior.variance.sqrt().detach().numpy().flatten()

            min_val, max_val = bounds[k]
            x_orig = grid.numpy() * (max_val - min_val) + min_val
            train_x_orig = train_x[:, k].numpy() * (max_val - min_val) + min_val

            y_mean = mean * 100
            err_upper = (mean + std) * 100
            err_lower = (mean - std) * 100

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_orig, y_mean, 'b-', label='Predicted Mean')
            ax.fill_between(x_orig, err_upper, err_lower, alpha=0.3, label='Uncertainty Band')
            sns.rugplot(train_x_orig, height=0.1, color='red', label='Training Points', ax=ax)

            ax.set_xlabel(f"{name} (Original Scale)")
            ax.set_ylabel("Predicted Yield [%]")
            ax.set_title(f"Effect of {name} on Yield")
            ax.legend()
            fig.tight_layout()

            st.pyplot(fig)


import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_interactive_slice(model, train_x, bounds, param_names):
    st.subheader("Interactive Posterior Slice")

    input_dim = train_x.shape[1]

    # 1. Zielparameter auswählen
    target_param = st.selectbox("Parameter to analyze (x-axis):", param_names, key="selected_param")

    sweep_idx = param_names.index(target_param)

    # 2. Slider für alle anderen Parameter
    fixed_values = []

    st.markdown("### Fix other parameters:")

    for i, name in enumerate(param_names):
        min_val, max_val = bounds[i]
        slider_key = f"fix_{name}"

        if name != target_param:
            # Init default if not set
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (min_val + max_val) / 2

            val = st.slider(
                f"{name} ({min_val:.2f}–{max_val:.2f})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=st.session_state[slider_key],
                step=(max_val - min_val) / 100,
                key=slider_key
            )
            # Normalize
            fixed_values.append((val - min_val) / (max_val - min_val))
        else:
            fixed_values.append(None)

    # 3. Build tensor from fixed + grid
    grid = torch.linspace(0, 1, 100)
    X = []
    for g in grid:
        row_vals = [
            g.item() if v is None else v
            for v in fixed_values
        ]
        X.append(torch.tensor(row_vals, dtype=torch.float32))
    X_tensor = torch.stack(X)

    # 4. GP prediction
    posterior = model.posterior(X_tensor)
    mean = posterior.mean.detach().numpy().flatten()
    std = posterior.variance.sqrt().detach().numpy().flatten()

    min_val, max_val = bounds[sweep_idx]
    x_orig = grid.numpy() * (max_val - min_val) + min_val
    y_mean = mean * 100
    y_upper = (mean + std) * 100
    y_lower = (mean - std) * 100

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_orig, y_mean, 'b-', label='Predicted Mean')
    ax.fill_between(x_orig, y_upper, y_lower, alpha=0.3, label='Uncertainty')
    ax.set_xlabel(f"{target_param} (original scale)")
    ax.set_ylabel("Predicted Yield [%]")
    ax.set_title(f"Effect of {target_param} (others fixed)")
    ax.legend()
    st.pyplot(fig)



import numpy as np
import torch
import streamlit as st
import plotly.graph_objs as go
from botorch.acquisition import ExpectedImprovement

def plot_2d_bo_contours_streamlit(
    model,
    train_x,
    train_y,
    bounds,
    param_x_idx,
    param_y_idx,
    param_names=None,
    n_grid=50,
    acquisition_func_factory=None,
    plot_type="Posterior Mean",
    show_streamlit=True,
    save_html=False,
    html_filename="bo_2d_contour.html",
    fixed_values=None
):
    if param_names is None:
        param_names = [f"x{i+1}" for i in range(train_x.shape[1])]

    # --- 2. Create meshgrid for x/y ---
    x1 = torch.linspace(0, 1, n_grid)
    x2 = torch.linspace(0, 1, n_grid)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # --- 3. Fill full grid ---

    X_full = torch.zeros(X_grid.shape[0], train_x.shape[1])
    X_full[:, param_x_idx] = X_grid[:, 0]
    X_full[:, param_y_idx] = X_grid[:, 1]
    for j in range(train_x.shape[1]):
        if j not in [param_x_idx, param_y_idx]:
            X_full[:, j] = fixed_values[j]

    # --- 4. Posterior prediction ---
    model.eval()
    posterior = model.posterior(X_full)
    mean = posterior.mean.detach().numpy()
    std = posterior.variance.sqrt().detach().numpy()

    # --- 5. Acquisition ---
    if acquisition_func_factory is None:
        acq = ExpectedImprovement(model=model, best_f=train_y.max().item())
    else:
        acq = acquisition_func_factory(model)
    ei = acq(X_full.unsqueeze(1)).detach().numpy()

    # --- 6. Plot selection ---
    if plot_type == "Posterior Mean":
        z = mean.reshape(n_grid, n_grid)
        z_label = "Posterior Mean"
    elif plot_type == "Model Uncertainty":
        z = std.reshape(n_grid, n_grid)
        z_label = "Model Std (Uncertainty)"
    else:
        z = ei.reshape(n_grid, n_grid)
        z_label = "Acquisition Function"

    # --- 7. Unnormalize for plot ---
    def unnormalize(vals, i):
        return vals * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    x1_orig = unnormalize(x1.numpy(), param_x_idx)
    x2_orig = unnormalize(x2.numpy(), param_y_idx)
    X1_orig, X2_orig = np.meshgrid(x1_orig, x2_orig, indexing="ij")

    # --- 8. Plotly surface ---
    fig = go.Figure(data=[
        go.Surface(
            z=z,
            x=X1_orig,
            y=X2_orig,
            colorscale='Viridis',
            colorbar=dict(title=z_label)
        )
    ])
    fig.update_layout(
        title=f"{z_label}: {param_names[param_x_idx]} vs. {param_names[param_y_idx]}",
        scene=dict(
            xaxis_title=param_names[param_x_idx],
            yaxis_title=param_names[param_y_idx],
            zaxis_title=z_label
        )
    )

    # --- 9. Show / Save ---
    if save_html and html_filename:
        from plotly.offline import plot as plotly_plot
        plotly_plot(fig, filename=html_filename, auto_open=False)
        st.info(f"Saved interactive plot as: {html_filename}")

    if show_streamlit:
        st.plotly_chart(fig, use_container_width=True)
    