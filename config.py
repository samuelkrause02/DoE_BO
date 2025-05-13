# config.py

EXCEL_FILE = "orthogonal_samplesI.xlsx"
DATA_SHEET = "Experiments"
BOUNDS_SHEET = "Bounds"
LOI = True
PSD = True

COLUMNS_CONFIG = {
    "temperature": "Temperature [°C]",
    "reaction_time": "Reaction Time [min]",
    "cao_w": "CaO [w%]",
    "cacl2_w": "CaCl2 [w%]",
    "yield_loi": "Conversion [%]",
    "yield_mass": "Reaction yield (mass)"
}

BATCH_SIZE = 3
INPUT_DIM = 5

COLUMNS_MAP = {
    'temperature': 2,
    'reaction_time': 3,
    'cao_w': 4,
    'cacl2_w': 8,
    'psd': 9 # optional
}

META_COLUMNS = {
    'max_objective': 29,
    'comment': 27,
    'date': 28
}

# Path to the experiment CSV file (input + output data)
CSV_FILE = "experiments.csv"

# Path to the bounds CSV file (each row: [min, max] for one parameter)
BOUNDS_CSV = "bounds.csv"

# Batch size for Bayesian optimization
BATCH_SIZE = 3

# Optionally: other configs you might need in your workflow
# e.g., whether you work with LOI or Mass (for comments etc.)
LOI = True

# You no longer need COLUMNS_CONFIG here because the CSV
# is now general (all except last col = inputs, last col = output)


# Example for 4 parameters: [min, max]
BOUNDS = [
    [100, 200.0],   # Temperature
    [30, 180.0],   # Reaction Time
    [0.0, 2.0],     # CaO w%
    [0.0, 0.45414],     # CaCl2 w%
    
]

if PSD == True:
    BOUNDS.append([0.0, 200])  # PSD
    COLUMNS_CONFIG['psd'] = "PSD [µm]"


MODEL_CONFIG = {
    "name": "RBF",
    "kernel_type": "RBF",  # oder "Matern"
    "kernel_nu": 2.5,
    "ard": True,
    "lengthscale_prior": "LogNormal",  # oder "Gamma", "SmoothedBox"
    "lengthscale_prior_params": (2.0, 0.5),

    # LogNormal für Noise
    "noise_prior_type": "LogNormal",
    "mean_noise": 0.01,     # entspricht z. B. 1% auf normalisierter Skala
    "scale_log": 0.5,        # Streuung im Log-Raum
    "mean_noise": 0.0016,
    "scale_log": 0.25,
    "noise_prior_type": "LogNormal",
    "noise_constraint_type": "interval",
    "noise_constraint_bounds": (0.0004, 0.04),
}