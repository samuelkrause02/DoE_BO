import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from sklearn.metrics import mean_squared_error
from .bo_model import build_gp_model
from config import MODEL_CONFIG

def loocv_gp_custom(train_x, train_y, input_dim, verbose=True, model_config=MODEL_CONFIG):
    """
    Perform Leave-One-Out Cross-Validation for your custom GP model.
    """
    import numpy as np
    from botorch.exceptions import ModelFittingError

    n_points = train_x.shape[0]
    preds = []
    truths = []
    noise_values = []

    for i in range(n_points):
        if verbose:
            print(f"LOOCV iteration {i+1}/{n_points}")

        mask = torch.arange(n_points) != i
        x_train_cv = train_x[mask]
        y_train_cv = train_y[mask]
        x_val = train_x[i].unsqueeze(0)
        y_val = train_y[i].unsqueeze(0)

        try:
            model_cv, likelihood_cv = build_gp_model(x_train_cv, y_train_cv, input_dim, model_config)
            mll = ExactMarginalLogLikelihood(likelihood_cv, model_cv)
            fit_gpytorch_mll(mll)

            model_cv.eval()
            likelihood_cv.eval()

            with torch.no_grad():
                posterior = model_cv.posterior(x_val)
                mean_pred = posterior.mean.item()
                noise_value = model_cv.likelihood.noise.detach().cpu().item()

            preds.append(mean_pred)
            truths.append(y_val.item())
            noise_values.append(noise_value)

        except Exception as e:
            print(f"[WARNING] Skipping iteration {i+1}: {e}")
            continue  # Skip failed fit

    if not preds:
        raise RuntimeError("LOOCV failed: all fits unsuccessful.")

    preds = torch.tensor(preds)
    truths = torch.tensor(truths)

    mse = mean_squared_error(truths.numpy(), preds.numpy())
    avg_noise = np.mean(noise_values)

    print(f"\nLOOCV Mean Squared Error: {mse:.4f}")
    print(f"Average Learned Noise: {avg_noise:.6f}")

    return preds, truths, mse, avg_noise
import pandas as pd
import datetime
import os

def log_loocv_result(log_file, model_name, input_dim, kernel_type, noise_prior_range, ard, mse, num_points, prior, prior_lengthscale, noise_value, notes=""):
    """
    Log LOOCV results to a CSV file.
    """
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    log_entry = {
        'Timestamp': time_stamp,
        'Model_Name': model_name,
        'Input_Dim': input_dim,
        'Kernel_Type': kernel_type,
        'Noise_Prior_Range': str(noise_prior_range),
        'Mean_MSE_LOOCV': round(mse, 6),
        'ARD': ard,
        'Number_trainingpoints': num_points,
        'Prior_type': prior,
        'Lengthscale_Prior': str(prior_lengthscale),
        'Noise_value': noise_value,
        'Notes': notes
    }

    df_entry = pd.DataFrame([log_entry])

    # Check if the log file exists
    file_exists = os.path.isfile(log_file)

    # Save to CSV
    df_entry.to_csv(log_file, mode='a', header=not file_exists, index=False, sep=';')
    print(f"Logged LOOCV result to {log_file}")