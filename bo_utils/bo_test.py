# bo_test.py

import torch
from bo_utils.bo_model import build_gp_model, fit_model
from bo_utils.bo_optimization import optimize_qEI, optimize_posterior_mean
from botorch.test_functions import Hartmann
import matplotlib.pyplot as plt
import tempfile
import os
from bo_utils.bo_optimization import greedy_gibbon_batch
torch.set_default_dtype(torch.double)

def run_bo_test(
    model_config,
    acquisition_type="qEI",
    batch_size=1,
    iterations=50,
    initial_points=10,
    noise_std=0.0
):
    hartmann = Hartmann(dim=6)
    bounds = torch.stack([torch.zeros(6), torch.ones(6)])

    train_x = torch.rand(initial_points, 6)
    train_y = -hartmann(train_x).unsqueeze(-1)
    best_values = []

    for iter in range(iterations):
        print(f"Iteration {iter+1}")

        # Build and fit model
        model, likelihood = build_gp_model(
            train_x, train_y,
            config=model_config
        )
        fit_model(model)

        # Choose acquisition function
        if acquisition_type == "qEI" or acquisition_type == "qNEI":
            new_x = optimize_qEI(
                model=model,
                input_dim=6,
                best_f=train_y.max(),
                bounds=bounds,
                acquisition_type=acquisition_type,
                batch_size=batch_size,
                x_baseline=train_x,
            )
        elif acquisition_type == "qGIBBON":
            new_x = greedy_gibbon_batch(
                model=model,
                bounds=bounds,
                batch_size=batch_size,
                num_restarts=10,
                raw_samples=256,
                inequality_constraints=None
            )
            
        elif acquisition_type == "posterior_mean":
            new_x = optimize_posterior_mean(
                model=model,
                input_dim=6,
                bounds=bounds,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unsupported acquisition type: {acquisition_type}")

        new_y_clean = -hartmann(new_x).unsqueeze(-1)
        noise = noise_std * torch.randn_like(new_y_clean)
        new_y = new_y_clean + noise

        # Update training set
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        best_val = train_y.max().item()
        best_values.append(best_val)

        print(f"New sample: {new_x}, New value: {new_y.mean().item():.4f}, Best: {train_y.max().item():.4f}")

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(best_values) + 1), best_values, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Observed Value')
    ax.set_title(f'Convergence of Bayesian Optimization ({acquisition_type})')
    ax.grid(True)

    tmp_dir = tempfile.gettempdir()
    plot_path = os.path.join(tmp_dir, "convergence_plot_qEI.png")
    fig.savefig(plot_path, dpi=1000)
    plt.close(fig)

    return best_values, plot_path