import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, ConstantKernel
from gpytorch.priors import SmoothedBoxPrior, GammaPrior, LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan, Interval
import numpy as np
import math
import gpytorch
from botorch.models.transforms import Standardize
from botorch.settings import debug
debug(True)

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, LinearKernel, PolynomialKernel
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan, Interval
import numpy as np
import math
from itertools import product


# =============================================================================
# KERNEL BUILDING
# =============================================================================

def build_kernel(kernel_config, input_dim, use_ard=False):
    """Build kernel based on configuration"""
    ard_dims = input_dim if use_ard else None
    
    if kernel_config["type"] == "Matern":
        return MaternKernel(
            nu=kernel_config.get("nu", 2.5),
            ard_num_dims=ard_dims
        )
    elif kernel_config["type"] == "RBF":
        return RBFKernel(ard_num_dims=ard_dims)
    elif kernel_config["type"] == "Linear+Matern":
            linear = LinearKernel(num_dimensions=input_dim, ard_num_dims=ard_dims)
            matern = MaternKernel(nu=kernel_config.get("nu", 2.5), ard_num_dims=ard_dims)
            return linear + matern
            
    elif kernel_config["type"] == "Linear+RBF":
        linear = LinearKernel(num_dimensions=input_dim, ard_num_dims=ard_dims)
        rbf = RBFKernel(ard_num_dims=ard_dims)
        return linear + rbf
        
    elif kernel_config["type"] == "Linear*Matern":
        linear = LinearKernel(num_dimensions=input_dim, ard_num_dims=ard_dims)
        matern = MaternKernel(nu=kernel_config.get("nu", 2.5), ard_num_dims=ard_dims)
        return linear * matern
    elif kernel_config["type"] == "Polynomial":
        return PolynomialKernel(
            power=kernel_config.get("power", 2),
            ard_num_dims=ard_dims
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_config['type']}")


def apply_lengthscale_prior(kernel, prior_config):
    """Apply lengthscale prior to kernel"""
    if prior_config is None:
        return
    
    # Handle legacy string format
    if isinstance(prior_config, str):
        if prior_config.lower() == "lognormal":
            prior_config = {"type": "lognormal", "loc": 0.0, "scale": 1.0}
        elif prior_config.lower() == "gamma":
            prior_config = {"type": "gamma", "concentration": 2.0, "rate": 0.5}
        else:
            print(f"Warning: Unknown legacy prior type '{prior_config}', skipping prior")
            return
    
    if prior_config.get("type") is None:
        return
    
    # FIX: Check for None values in parameters
    if prior_config["type"] == "gamma":
        concentration = prior_config.get("concentration", 2.0)
        rate = prior_config.get("rate", 0.5)
        
        # Skip if None values
        if concentration is None or rate is None:
            print(f"Warning: None values in gamma prior parameters, skipping")
            return
            
        prior = GammaPrior(float(concentration), float(rate))
    elif prior_config["type"] == "lognormal":
        loc = prior_config.get("loc", 0.0)
        scale = prior_config.get("scale", 1.0)
        
        # Skip if None values
        if loc is None or scale is None:
            print(f"Warning: None values in lognormal prior parameters, skipping")
            return
            
        prior = LogNormalPrior(float(loc), float(scale))
    else:
        print(f"Warning: Unknown lengthscale prior type: {prior_config['type']}")
        return
    
    kernel.register_prior("lengthscale_prior", prior, "lengthscale")


def apply_noise_prior(likelihood, prior_config):
    """Apply noise prior to likelihood"""
    if prior_config is None:
        return
    
    # Handle legacy string format
    if isinstance(prior_config, str):
        if prior_config.lower() == "lognormal":
            prior_config = {"type": "lognormal", "loc": -3.0, "scale": 1.0}
        elif prior_config.lower() == "gamma":
            prior_config = {"type": "gamma", "concentration": 2.0, "rate": 10.0}
        else:
            print(f"Warning: Unknown legacy noise prior type '{prior_config}', skipping prior")
            return
    
    if prior_config.get("type") is None:
        return
    
    # FIX: Check for None values in parameters
    if prior_config["type"] == "gamma":
        concentration = prior_config.get("concentration", 2.0)
        rate = prior_config.get("rate", 10.0)
        
        # Skip if None values
        if concentration is None or rate is None:
            print(f"Warning: None values in gamma noise prior parameters, skipping")
            return
            
        prior = GammaPrior(float(concentration), float(rate))
    elif prior_config["type"] == "lognormal":
        loc = prior_config.get("loc", -3.0)
        scale = prior_config.get("scale", 1.0)
        
        # Skip if None values
        if loc is None or scale is None:
            print(f"Warning: None values in lognormal noise prior parameters, skipping")
            return
            
        prior = LogNormalPrior(float(loc), float(scale))
    else:
        print(f"Warning: Unknown noise prior type: {prior_config['type']}")
        return
    
    likelihood.register_prior("noise_prior", prior, "noise")

def setup_outputscale(kernel, outputscale_config):
    """Setup outputscale (ScaleKernel wrapper)"""
    if outputscale_config.get("use", True):
        covar_module = ScaleKernel(kernel)
        
        # Initialize if specified
        if "init_value" in outputscale_config:
            with torch.no_grad():
                covar_module.outputscale = torch.tensor(outputscale_config["init_value"])
        
        # Fix if specified - must be done BEFORE model creation
        if outputscale_config.get("fixed", False):
            # Create a new parameter that doesn't require gradients
            fixed_value = outputscale_config.get("init_value", 1.0)
            with torch.no_grad():
                covar_module.raw_outputscale.fill_(covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(fixed_value)))
                covar_module.raw_outputscale.requires_grad_(False)
        
        return covar_module
    else:
        return kernel


# =============================================================================
# MAIN MODEL BUILDER
# =============================================================================

def build_gp_model(train_x, train_y, config):
    """
    Build a SingleTaskGP model based on configuration dictionary.
    
    Args:
        train_x: Training inputs [N x D]
        train_y: Training outputs [N x 1] 
        config: Configuration dictionary with structure:
        {
            "kernel": {
                "type": "Matern" | "RBF" | "Linear" | "Polynomial",
                "nu": 2.5,  # for Matern
                "power": 2,  # for Polynomial
            },
            "ard": True | False,
            "lengthscale_prior": {
                "type": "gamma" | "lognormal" | None,
                "concentration": 2.0, "rate": 0.5,  # for gamma
                "loc": 0.0, "scale": 1.0,  # for lognormal
            },
            "outputscale": {
                "use": True | False,
                "init_value": 1.0,  # optional
                "fixed": True | False,
            },
            "noise_prior": {
                "type": "gamma" | "lognormal" | None,
                "concentration": 2.0, "rate": 10.0,  # for gamma  
                "loc": -3.0, "scale": 1.0,  # for lognormal
            }
        }
    
    Returns:
        model: SingleTaskGP model
        likelihood: GaussianLikelihood
    """
    
    input_dim = train_x.shape[1]
    
    # Build kernel
    kernel = build_kernel(
        config["kernel"], 
        input_dim, 
        config.get("ard", False)
    )
    
    # Apply lengthscale prior
    apply_lengthscale_prior(kernel, config.get("lengthscale_prior"))
    
    # Setup covariance module (with or without outputscale)
    covar_module = setup_outputscale(kernel, config.get("outputscale", {"use": True}))
    
    # Setup likelihood
    likelihood = GaussianLikelihood()
    apply_noise_prior(likelihood, config.get("noise_prior"))
    
    # Build model
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        likelihood=likelihood,
    )
    
    return model, likelihood


# =============================================================================
# CONFIGURATION GENERATORS
# =============================================================================

def get_standard_kernel_configs():
    """Standard kernel configurations to try"""
    return [
        {"type": "Matern", "nu": 2.5},
        #{"type": "Matern", "nu": 1.5},
        #{"type": "Matern", "nu": 0.5},
        {"type": "RBF"},
        #{"type": "Linear+Matern", "nu": 2.5},  # Additiv
        #{"type": "Linear+RBF"},                # Additiv
        #{"type": "Linear*Matern", "nu": 2.5},  # Multiplikativ
    ]


def get_standard_lengthscale_priors():
    """Standard lengthscale prior configurations"""
    return [
        None,  # No prior
        #{"type": "gamma", "concentration": 1.0, "rate": 0.1},    # Weak, large lengthscales
        {"type": "gamma", "concentration": 2.0, "rate": 0.5},    # Medium
        {"type": "gamma", "concentration": 3.0, "rate": 6.0},    # Strong, small lengthscales
        {"type": "lognormal", "loc": 0.0, "scale": 1.0},         # Around 1.0
        {"type": "lognormal", "loc": 1.0, "scale": 0.5},         # Around 2.7
    ]


def get_standard_noise_priors():
    """Standard noise prior configurations"""
    return [
        None,  # No prior
        #{"type": "gamma", "concentration": 2.0, "rate": 10.0},   # Low noise
        {"type": "gamma", "concentration": 1.5, "rate": 5.0},    # Medium noise
        {"type": "lognormal", "loc": -4.0, "scale": 1.0},        # Around 0.05
        {"type": "lognormal", "loc": -2.0, "scale": 0.5},        # Around 0.14
    ]


def get_standard_outputscale_configs():
    """Standard outputscale configurations"""
    return [
        {"use": True, "fixed": False},                    # Learn outputscale
        {"use": True, "fixed": True, "init_value": 1.0},  # Fix to 1.0
       # {"use": True, "fixed": True, "init_value": 0.5},  # Fix to 0.5
        {"use": False},                                   # No outputscale
    ]


def generate_model_configs(subset=True, max_configs=24):
    """
    Generate model configurations for comparison.
    
    Args:
        subset: If True, generate reasonable subset. If False, generate all combinations.
        max_configs: Maximum number of configs if subset=True
    
    Returns:
        List of configuration dictionaries
    """
    
    if subset:
        # Focused subset for quick testing
        kernels = [
            {"type": "Matern", "nu": 2.5},
            {"type": "RBF"},
            {"type": "Linear"},
        ]
        lengthscale_priors = [
            None,
            {"type": "gamma", "concentration": 2.0, "rate": 0.5},
        ]
        noise_priors = [
            None,
            {"type": "gamma", "concentration": 2.0, "rate": 10.0},
        ]
        outputscales = [
            {"use": True, "fixed": False},
            {"use": True, "fixed": True, "init_value": 1.0},
        ]
        ard_options = [False, True]
    else:
        # Full combinations
        kernels = get_standard_kernel_configs()
        lengthscale_priors = get_standard_lengthscale_priors()
        noise_priors = get_standard_noise_priors()
        outputscales = get_standard_outputscale_configs()
        ard_options = [True]
    
    configs = []
    for kernel, ls_prior, noise_prior, outputscale, ard in product(
        kernels, lengthscale_priors, noise_priors, outputscales, ard_options
    ):
        config = {
            "kernel": kernel,
            "ard": ard,
            "lengthscale_prior": ls_prior,
            "outputscale": outputscale,
            "noise_prior": noise_prior,
        }
        configs.append(config)
        
        if subset and len(configs) >= max_configs:
            break
    
    return configs


# =============================================================================
# FITTING AND UTILITY FUNCTIONS
# =============================================================================

def fit_model(model, max_iter=100):
    """Fit GP model and return MLL score"""
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    try:
        fit_gpytorch_mll(mll, max_iter=max_iter)
        model.eval()
        # Return the scalar MLL value, not the MLL object
        with torch.no_grad():
            mll_value = mll(model(model.train_inputs[0]), model.train_targets).item()
        return mll_value
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return float('-inf')


def prepare_training_tensors(scaled_x, scaled_y):
    """Convert numpy arrays to PyTorch tensors for GP training"""
    train_x = torch.from_numpy(scaled_x).double()
    train_y = torch.from_numpy(scaled_y).double().unsqueeze(-1)
    print(f"Training data shapes: X={train_x.shape}, Y={train_y.shape}")
    return train_x, train_y


def print_config_summary(config):
    """Print readable summary of model configuration"""
    print(f"Kernel: {config['kernel']['type']}", end="")
    if config['kernel']['type'] == 'Matern':
        print(f" (nu={config['kernel']['nu']})", end="")
    elif config['kernel']['type'] == 'Polynomial':
        print(f" (power={config['kernel']['power']})", end="")
    
    print(f", ARD: {config['ard']}", end="")
    
    if config.get('lengthscale_prior'):
        print(f", LS Prior: {config['lengthscale_prior']['type']}", end="")
    
    if config.get('noise_prior'):
        print(f", Noise Prior: {config['noise_prior']['type']}", end="")
    
    if config['outputscale']['fixed']:
        print(f", OS: fixed({config['outputscale']['init_value']})", end="")
    elif not config['outputscale']['use']:
        print(f", OS: disabled", end="")
    
    print()






def extract_hyperparameters_from_model(trained_model):
    """
    Extrahiert die gelernten Hyperparameter aus einem trainierten GP-Modell.
    
    Args:
        trained_model: Trainiertes SingleTaskGP Modell
        
    Returns:
        dict: Dictionary mit allen relevanten Hyperparametern
    """
    hyperparams = {}
    
    # Lengthscale(s) - kann skalar oder Vektor sein (ARD)
    lengthscale = trained_model.covar_module.base_kernel.lengthscale.detach()
    hyperparams['lengthscale'] = lengthscale
    
    # Outputscale (Signal-Varianz)
    outputscale = trained_model.covar_module.outputscale.detach()
    hyperparams['outputscale'] = outputscale
    
    # Noise level
    noise = trained_model.likelihood.noise.detach()
    hyperparams['noise'] = noise
    
    # Kernel-Typ identifizieren
    base_kernel = trained_model.covar_module.base_kernel
    if isinstance(base_kernel, MaternKernel):
        hyperparams['kernel_type'] = 'Matern'
        hyperparams['kernel_nu'] = base_kernel.nu
    elif isinstance(base_kernel, RBFKernel):
        hyperparams['kernel_type'] = 'RBF'
    else:
        hyperparams['kernel_type'] = 'Unknown'
    
    # ARD Information
    hyperparams['ard'] = len(lengthscale.shape) > 1 and lengthscale.shape[1] > 1
    hyperparams['input_dim'] = lengthscale.shape[1] if hyperparams['ard'] else 1
    
    print("Extrahierte Hyperparameter:")
    print(f"  Lengthscale: {lengthscale.numpy()}")
    print(f"  Outputscale: {outputscale.item():.4f}")
    print(f"  Noise: {noise.item():.4f}")
    print(f"  Kernel: {hyperparams['kernel_type']}")
    print(f"  ARD: {hyperparams['ard']}")
    
    return hyperparams

def create_informed_priors(source_hyperparams, prior_strength=2.0):
    """
    Erstellt informierte Priors basierend auf Source-Hyperparametern.
    
    Args:
        source_hyperparams: Dict mit Source-Hyperparametern
        prior_strength: Stärke der Priors (höher = enger um Source-Werte)
        
    Returns:
        dict: Dictionary mit Prior-Objekten
    """
    priors = {}
    
    # === Lengthscale Prior ===
    lengthscale_values = source_hyperparams['lengthscale'].flatten()
    lengthscale_priors = []
    
    for ls_val in lengthscale_values:
        # Gamma-Prior: mean = concentration/rate
        concentration = prior_strength
        rate = concentration / ls_val.item()
        
        lengthscale_prior = GammaPrior(
            concentration=concentration,
            rate=rate
        )
        lengthscale_priors.append(lengthscale_prior)
    
    priors['lengthscale'] = lengthscale_priors[0] if len(lengthscale_priors) == 1 else lengthscale_priors
    
    # === Outputscale Prior ===
    outputscale_val = source_hyperparams['outputscale'].item()
    concentration = prior_strength
    rate = concentration / outputscale_val
    
    priors['outputscale'] = GammaPrior(
        concentration=concentration,
        rate=rate
    )
    
    # === Noise Prior ===
    noise_val = source_hyperparams['noise'].item()
    log_noise = math.log(noise_val)
    scale = 0.5  # Feste Unsicherheit
    
    priors['noise'] = LogNormalPrior(
        loc=log_noise,
        scale=scale
    )
    
    print(f"Erstelle Priors mit Stärke {prior_strength:.2f}")
    
    return priors

def build_transfer_gp_model(train_x, train_y, source_hyperparams, prior_strength=2.0):
    """
    Transfer Learning GP Model Builder - Fixed für ARD Kernels
    """
    from gpytorch.priors import GammaPrior, LogNormalPrior
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from botorch.models import SingleTaskGP
    import torch
    import math
    
    # === Lengthscale Prior (FIXED für ARD) ===
    lengthscale_values = source_hyperparams['lengthscale'].flatten()
    
    if len(lengthscale_values) == 1:
        # Single lengthscale
        ls_val = lengthscale_values[0].item()
        lengthscale_prior = GammaPrior(prior_strength, prior_strength / ls_val)
    else:
        # ARD: Use mean lengthscale for prior
        mean_ls = lengthscale_values.mean().item()
        lengthscale_prior = GammaPrior(prior_strength, prior_strength / mean_ls)
    
    # === Kernel Setup ===
    input_dim = train_x.shape[1]
    
    if source_hyperparams['kernel_type'] == 'Matern':
        base_kernel = MaternKernel(
            nu=source_hyperparams.get('kernel_nu', 2.5),
            ard_num_dims=input_dim if source_hyperparams.get('ard', False) else None,
            lengthscale_prior=lengthscale_prior
        )
    else:  # RBF or fallback
        base_kernel = RBFKernel(
            ard_num_dims=input_dim if source_hyperparams.get('ard', False) else None,
            lengthscale_prior=lengthscale_prior
        )
    
    # === Wrap in ScaleKernel ===
    outputscale_val = source_hyperparams['outputscale'].item()
    outputscale_prior = GammaPrior(prior_strength, prior_strength / outputscale_val)
    
    covar_module = ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
    
    # === Likelihood ===
    noise_val = source_hyperparams['noise'].item()
    noise_prior = LogNormalPrior(math.log(noise_val), 0.5)
    
    likelihood = GaussianLikelihood(noise_prior=noise_prior)
    
    # === Initialization ===
    with torch.no_grad():
        # Smart lengthscale initialization
        if source_hyperparams.get('ard', False) and len(lengthscale_values) > 1:
            covar_module.base_kernel.lengthscale = source_hyperparams['lengthscale']
        else:
            init_ls = lengthscale_values[0] if len(lengthscale_values) == 1 else lengthscale_values.mean()
            covar_module.base_kernel.lengthscale = init_ls
        
        covar_module.outputscale = source_hyperparams['outputscale']
        likelihood.noise = source_hyperparams['noise']
    
    # === Create Model ===
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        likelihood=likelihood
    )
    
    return model, likelihood

def fit_transfer_model(model, max_iter=1000):
    """
    Trainiert das Transfer-Modell mit den informierten Priors.
    
    Args:
        model: Transfer GP Modell
        max_iter: Maximale Iterationen für Optimierung
        
    Returns:
        model: Trainiertes Modell
    """
    model.train()
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, max_iter=max_iter)
    
    model.eval()
    
    print("Transfer-Modell Training abgeschlossen.")
    print("Finale Hyperparameter:")
    print(f"  Lengthscale: {model.covar_module.base_kernel.lengthscale.detach().numpy()}")
    print(f"  Outputscale: {model.covar_module.outputscale.detach().item():.4f}")
    print(f"  Noise: {model.likelihood.noise.detach().item():.4f}")
    
    return model

def create_transfer_model_from_source(source_model, target_train_x, target_train_y, prior_strength=2.0):
    """
    Kompletter Workflow: Extrahiert Hyperparameter aus Source-Modell und erstellt 
    Transfer-Modell für Target-Daten.
    
    Args:
        source_model: Trainiertes Source GP-Modell
        target_train_x: Target Trainingsdaten X
        target_train_y: Target Trainingsdaten Y
        prior_strength: Stärke der Prior-Information (höher = enger um Source-Werte)
        
    Returns:
        model: Trainiertes Transfer GP-Modell
    """
    
    print("=== TRANSFER LEARNING WORKFLOW ===")
    
    # Schritt 1: Extrahiere Source-Hyperparameter
    print("\n1. Extrahiere Source-Hyperparameter:")
    source_hyperparams = extract_hyperparameters_from_model(source_model)
    
    # Schritt 2: Erstelle Transfer-Modell mit informierten Priors
    print(f"\n2. Erstelle Transfer-Modell (Prior-Stärke: {prior_strength:.2f}):")
    transfer_model, likelihood = build_transfer_gp_model(
        target_train_x, target_train_y, 
        source_hyperparams, prior_strength
    )
    
    # Schritt 3: Trainiere Transfer-Modell
    print("\n3. Trainiere Transfer-Modell:")
    trained_transfer_model = fit_transfer_model(transfer_model)
    
    print("\n=== TRANSFER ABGESCHLOSSEN ===")
    
    return trained_transfer_model

def prepare_training_tensors(scaled_x, scaled_y):
    train_x = torch.from_numpy(scaled_x).double()
    train_y = torch.from_numpy(scaled_y).double().unsqueeze(-1)
    print(f"Training data shapes: X={train_x.shape}, Y={train_y.shape}")
    return train_x, train_y


def create_noise_mask(total_points, n_inaccurate_last):
    """
    Gibt ein bool-Array der Länge total_points zurück:
    - True für präzise Messung
    - False für unsichere (z. B. Mass-Gain-basierte) Messung am Ende
    """
    mask = np.ones(total_points, dtype=bool)
    if n_inaccurate_last > 0:
        mask[-n_inaccurate_last:] = False
    return mask


def build_gp_model_simple(train_x, train_y, input_dim, config):

    model = SingleTaskGP(train_x, train_y)

    likelihood = model.likelihood
    return model, likelihood


import torch
import math
from gpytorch.priors import LogNormalPrior, GammaPrior, SmoothedBoxPrior
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, LinearKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP

def build_groundtruth_gp_model(train_x, train_y, input_dim, config):
    """
    Build a GP model for simulation/ground-truth purposes using explicit hyperparameters
    from a provided config dictionary (no learning of hyperparameters).
    """
    # === Kernel-Auswahl
    if config["covar_module"].startswith("Matern"):
        nu_val = float(config["covar_module"].split("_")[1])
        base_kernel = MaternKernel(nu=nu_val)
    elif config["covar_module"] == "RBF":
        base_kernel = RBFKernel()
    elif config["covar_module"] == "Linear":
        base_kernel = LinearKernel()
    else:
        raise ValueError(f"Unknown covar_module: {config['covar_module']}")

    # === Lengthscale setzen
    if base_kernel.__class__ in [MaternKernel, RBFKernel]:
        if "lengthscale" not in config:
            raise KeyError("Missing 'lengthscale' in config for kernel with lengthscale.")
        base_kernel.lengthscale = torch.tensor(config["lengthscale"]).view(1, 1)

    # === Outputscale setzen via ScaleKernel
    covar_module = ScaleKernel(base_kernel)
    covar_module.outputscale = torch.tensor(config["outputscale"])
    covar_module.raw_outputscale.requires_grad = False  # Keine Optimierung

    # === Likelihood mit fixem Noise
    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor(config["noise"])
    likelihood.raw_noise.requires_grad = False

    # === GP Model bauen
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        likelihood=likelihood,
    )

    return model, likelihood