import torch
from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement, qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler



def optimize_posterior_mean(model, input_dim, bounds=None, num_restarts=5, raw_samples=50):
    if bounds is None:
        bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)])

    model.eval()
    acq_mean = PosteriorMean(model)

    best_x_scaled, _ = optimize_acqf(
        acq_function=acq_mean,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    print(f"Best x (scaled): {best_x_scaled}")
    return best_x_scaled


def optimize_qEI(
    model,
    input_dim,
    best_f,
    batch_size=3,
    bounds=None,
    num_restarts=30,
    raw_samples=1024,
    num_sqmc=1024,
    limit=-100000,
    x_baseline=None,
    acquisition_type="qEI",
):
    if bounds is None:
        bounds = torch.stack([
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double)
        ])
    
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_sqmc]))
    if acquisition_type == "qEI":
        qEI = qLogExpectedImprovement(model, best_f=best_f, sampler=sampler)
    elif acquisition_type == "qNEI":
        if x_baseline is None:
            raise ValueError("x_baseline must be provided for qNEI.")
        else:
            qEI = qLogNoisyExpectedImprovement(model, X_baseline=x_baseline, sampler=sampler)
    elif acquisition_type == "ucb":
        qEI = qUpperConfidenceBound(model, beta=0.1,sampler=sampler)
    elif acquisition_type == "ei":
        qEI = qExpectedImprovement(model, best_f=best_f, sampler=sampler)


    # Achtung: Dezimalpunkt statt Komma!
    if limit > -1e2:
        inequality_constraints = [
            (
                torch.tensor([2, 3], dtype=torch.long),
                torch.tensor([-2.0, -0.45414], dtype=torch.double),
                limit
            )
        ]
    else:
        inequality_constraints = None  # Kein Constraint

    cand, _ = optimize_acqf(
        acq_function=qEI,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints
    )

    print(f"Candidates (scaled):\n{cand}")
    return cand
