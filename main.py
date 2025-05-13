# main.py

from config import (
    EXCEL_FILE, DATA_SHEET, BOUNDS_SHEET, LOI, COLUMNS_CONFIG,
    INPUT_DIM, BATCH_SIZE,
    COLUMNS_MAP, META_COLUMNS
)

from bo_utils.bo_data import load_experiment_data, load_bounds, prepare_training_data
from bo_utils.bo_model import build_gp_model, fit_model, prepare_training_tensors
from bo_utils.bo_optimization import optimize_posterior_mean, optimize_qEI
from bo_utils.bo_utils import rescale_single_point, rescale_batch
from bo_utils.bo_save_csv import write_candidates_to_excel

# 1. Load experimental data
print("Loading experimental data...")
raw_inputs, raw_outputs, df = load_experiment_data(
    EXCEL_FILE,
    sheet_name=DATA_SHEET,
    loi=LOI,
    columns_config=COLUMNS_CONFIG
)

# 2. Load parameter bounds
print("Loading parameter bounds...")
bounds = load_bounds(EXCEL_FILE, bounds_sheet=BOUNDS_SHEET)

# 3. Scale data
print("Preparing scaled training data...")
scaled_x, scaled_y = prepare_training_data(raw_inputs, raw_outputs, bounds)
print(f"Number of training points: {scaled_x.shape[0]}")

# 4. Convert data to torch tensors
train_x, train_y = prepare_training_tensors(scaled_x, scaled_y)

# 5. Build and fit Gaussian Process model
print("Building and fitting Gaussian Process model...")
model, likelihood = build_gp_model(train_x, train_y, INPUT_DIM)
fit_model(model)

# 6. Optimize posterior mean (find surrogate optimum)
print("Optimizing posterior mean...")
best_x_scaled = optimize_posterior_mean(model, INPUT_DIM)
best_x = rescale_single_point(best_x_scaled.detach().cpu().numpy()[0], bounds)
print("Best point (rescaled):", best_x)

# 7. Optimize qEI acquisition function to get next candidates
print("Optimizing qEI acquisition function...")
best_f = train_y.max().item()
cand = optimize_qEI(model, INPUT_DIM, best_f, batch_size=BATCH_SIZE)
x_next = rescale_batch(cand, bounds)
print("Next candidates (rescaled):", x_next)

# 8. Write results to Excel file
print("Writing candidates to Excel...")
write_candidates_to_excel(
    EXCEL_FILE,
    candidates_list=x_next,
    max_objective=scaled_y.max(),
    loi=LOI,
    sheet_name=DATA_SHEET,
    columns_map=COLUMNS_MAP,
    meta_columns=META_COLUMNS,
    n_existing_points=scaled_x.shape[0]
)

print("Finished.")