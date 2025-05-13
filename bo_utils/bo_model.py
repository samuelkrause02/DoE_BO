import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.priors import SmoothedBoxPrior, GammaPrior, LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan, Interval
import numpy as np
import math

def build_gp_model(train_x, train_y, input_dim, config):
    # === Optional: Initialwerte ===
    init_noise = config.get("init_noise", None)
    init_lengthscale = config.get("init_lengthscale", None)




    # === Noise Prior (LogNormal) ===
    noise_prior = LogNormalPrior(
        loc=np.log(config["mean_noise"]),
        scale=config["scale_log"]
    )

    # === Noise Constraint ===
    constraint_type = config.get("noise_constraint_type", "interval")
    constraint_bounds = config.get("noise_constraint_bounds", (0.0, 0.5))

    if constraint_type == "interval":
        noise_constraint = Interval(*constraint_bounds)
    elif constraint_type == "greater_than":
        noise_constraint = GreaterThan(constraint_bounds[0])
    elif constraint_type == "none":
        noise_constraint = None
    else:
        raise ValueError(f"Unknown noise constraint type: {constraint_type}")

    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=noise_constraint
    )

    if init_noise is not None:
        likelihood.noise = torch.tensor(init_noise)

    # === Lengthscale Prior ===
    lengthscale_prior = None
    if config.get("lengthscale_prior") is not None:
        prior_params = config.get("lengthscale_prior_params", (2.0, 0.5))
        if config["lengthscale_prior"] == "Gamma":
            lengthscale_prior = GammaPrior(*prior_params)
        elif config["lengthscale_prior"] == "SmoothedBox":
            lengthscale_prior = SmoothedBoxPrior(*prior_params)
        elif config.get("lengthscale_prior") == "LogNormal":
        # Dimension-scaled default (siehe BoTorch paper)
            loc = math.sqrt(2) + math.log(input_dim)
            scale = math.sqrt(3)
            lengthscale_prior = LogNormalPrior(loc=loc, scale=scale)

    # === Kernel Setup ===
    kernel_cls = MaternKernel if config["kernel_type"] == "Matern" else RBFKernel
    kernel = kernel_cls(
        nu=config.get("kernel_nu", 2.5) if config["kernel_type"] == "Matern" else None,
        ard_num_dims=input_dim if config.get("ard", False) else None,
        lengthscale_prior=lengthscale_prior,
    )

    if init_lengthscale is not None:
        with torch.no_grad():
            kernel.lengthscale = torch.tensor(init_lengthscale).reshape(1, -1)

    covar = ScaleKernel(kernel)

    model = SingleTaskGP(train_x, train_y, covar_module=covar, likelihood=likelihood)
    #resc
    return model, likelihood

def fit_model(model):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    print("Model fitting complete.")
    return model

def prepare_training_tensors(scaled_x, scaled_y):
    train_x = torch.from_numpy(scaled_x).double()
    train_y = torch.from_numpy(scaled_y).double().unsqueeze(-1)
    print(f"Training data shapes: X={train_x.shape}, Y={train_y.shape}")
    return train_x, train_y