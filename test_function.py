from config import MODEL_CONFIG
import torch

from bo_utils.bo_model import build_gp_model, fit_model, prepare_training_tensors
from bo_utils.bo_optimization import optimize_posterior_mean, optimize_qEI
from bo_utils.bo_utils import rescale_single_point, rescale_batch

from botorch.test_functions import Hartmann

# Problem setup
hartmann = Hartmann(dim=6)
bounds = torch.stack([torch.zeros(6), torch.ones(6)])

noise_std = 0.00


# Initial data
train_x = torch.rand(10, 6)
train_y = - hartmann(train_x).unsqueeze(-1)
best_values = []

for iter in range(50):
    print(f"Iteration {iter+1}")

    # baue GP mit deinem Modell
    model, likelihood = build_gp_model(
        train_x, train_y,
        input_dim=6,
        config=MODEL_CONFIG
    )
    fit_model(model)

    # BO step
    new_x = optimize_qEI(
        model=model,
        input_dim=6,
        best_f=train_y.max(),
        bounds=bounds,
        acquisition_type="qEI",  # besser klassisch zum Start
        batch_size=1,
        x_baseline=train_x,
    )

    new_y_clean = -hartmann(new_x).unsqueeze(-1)
    noise = noise_std * torch.randn_like(new_y_clean)
    new_y = new_y_clean + noise
    # ... nach dem Hinzufügen des neuen Punktes:
    # Update train set
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    # Jetzt Bestwert berechnen
    best_val = train_y.max().item()
    best_values.append(best_val)

    print(f"New sample: {new_x}, New value: {new_y.item():.4f}, Best: {train_y.max().item():.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(range(1, len(best_values)+1), best_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Best Observed Value')
plt.title('Convergence of Bayesian Optimization')
plt.grid(True)
plt.savefig("convergence_plot_qEI.png", dpi=1000)
plt.show()

