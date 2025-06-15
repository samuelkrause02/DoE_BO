# config.py

EXCEL_FILE = "orthogonal_samplesI.xlsx"
DATA_SHEET = "Experiments"
BOUNDS_SHEET = "Bounds"
LOI = True
PSD = False

COLUMNS_CONFIG = {
    "temperature": "Temperature [°C]",
    "reaction_time": "Reaction Time [min]",
    "cao_w": "CaO [w%]",
    "cacl2_w": "CaCl2 [w%]",
    "yield_loi": "Conversion [%]",
    "yield_mass": "Reaction yield (mass)"
}

BATCH_SIZE = 3
INPUT_DIM = 4

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
    "kernel": {"type": "RBF"},
    "ard": True,
    "lengthscale_prior": {
        "type": "lognormal",
        "loc": 2.0,
        "scale": 0.5
    },
    "outputscale": {"use": True, "fixed": False},
    "noise_prior": {
        "type": "lognormal",
        "loc": -6.13,  # log(0.0016) 
        "scale": 0.25
    }
}