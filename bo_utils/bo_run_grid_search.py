import torch
import pandas as pd
from bo_utils.bo_validation import loocv_gp_custom, log_loocv_result
from bo_utils.bo_data import prepare_training_data
from bo_utils.bo_model import prepare_training_tensors

def run_grid_search(dataframe, output_col_name, bounds, log_file="loocv_results_log.csv", base_config=None):
    import copy

    raw_inputs = dataframe[[col for col in dataframe.columns if col != output_col_name]].values
    raw_outputs = dataframe[output_col_name].values

    scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds=bounds)
    train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)
    input_dim = train_x.shape[1]
    n_train = train_x.shape[0]

    results_list = []
    kernels = ["Matern", "RBF"]

    for kernel in kernels:
        config = copy.deepcopy(base_config or {})
        config.update({
            "kernel_type": kernel,
            "name": f"GP_{kernel}",
            "kernel_nu": 2.5 if kernel == "Matern" else None,
        })

        try:
            preds, truths, mse, avg_noise = loocv_gp_custom(
                train_x, train_y, input_dim, model_config=config, verbose=True
            )
        except Exception as e:
            print(f"[ERROR] Skipping config {config['name']}: {e}")
            continue

        result = {
            'Model_Name': config["name"],
            'Input_Dim': input_dim,
            'Kernel_Type': kernel,
            'Mean_MSE_LOOCV': round(mse, 5),
            'ARD': config.get("ard", False),
            'Number_trainingpoints': n_train,
            'Prior_type': config.get("prior_type", ""),
            'Lengthscale_Prior': config.get("prior_params", ""),
            'Noise_value': round(avg_noise, 5),
            'Notes': "Kernel-only grid search"
        }
        results_list.append(result)

        # log_loocv_result(
        #     log_file=log_file,
        #     model_name=config["name"],
        #     input_dim=input_dim,
        #     kernel_type=kernel,
        #     noise_prior_range=config.get("noise_prior_range", ""),
        #     mse=mse,
        #     ard=config.get("ard", False),
        #     num_points=n_train,
        #     prior=config.get("prior_type", ""),
        #     prior_lengthscale=config.get("prior_params", ""),
        #     noise_value=avg_noise,
        #     notes="Kernel-only grid search"
        # )

    return pd.DataFrame(results_list)