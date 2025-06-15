import torch
from botorch.acquisition import (
    qLogExpectedImprovement, qLogNoisyExpectedImprovement, 
    qUpperConfidenceBound, PosteriorMean
)
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from bo_utils.bo_utils import rescale_single_point


def _get_constraint_config(limit):
    """Generate constraint configuration and optimization parameters."""
    if limit > -1e2:
        return {
            'constraints': [(
                torch.tensor([2, 3], dtype=torch.long),
                torch.tensor([-2.0, -0.45414], dtype=torch.double),
                limit
            )],
            'num_restarts': 10,
            'raw_samples': 256,
            'num_sqmc': 128
        }
    return {
        'constraints': None,
        'num_restarts': 40,
        'raw_samples': 1024,
        'num_sqmc': 512
    }


def _get_default_bounds(input_dim):
    """Generate default unit bounds."""
    return torch.stack([
        torch.zeros(input_dim, dtype=torch.double),
        torch.ones(input_dim, dtype=torch.double)
    ])


def clip_min_values(x_scaled, min_threshold=1e-4):
    """Clip small values to zero."""
    x_clipped = x_scaled.clone()
    x_clipped[x_clipped < min_threshold] = 0.0
    return x_clipped


def optimize_posterior_mean(model, input_dim, bounds=None, num_restarts=5, raw_samples=50, limit=-100000):
    """Optimize posterior mean for exploitation."""
    if bounds is None:
        bounds = _get_default_bounds(input_dim)
    
    model.eval()
    config = _get_constraint_config(limit)
    
    best_x_scaled, _ = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=config['constraints']
    )
    
    best_x_scaled = clip_min_values(best_x_scaled, min_threshold=1e-4)
    print(f"Best x (scaled): {best_x_scaled}")
    return best_x_scaled


def optimize_qEI(
    model, input_dim, best_f, batch_size=3, bounds=None, num_restarts=40,
    raw_samples=1024, num_sqmc=512, limit=-100000, x_baseline=None,
    acquisition_type="qEI", X_pending=None
):
    """Optimize qEI, qNEI, or qGIBBON acquisition functions."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if bounds is None:
        bounds = _get_default_bounds(input_dim)
    
    config = _get_constraint_config(limit)
    num_restarts = min(num_restarts, config['num_restarts'])
    raw_samples = min(raw_samples, config['raw_samples'])
    num_sqmc = min(num_sqmc, config['num_sqmc'])
    
    if acquisition_type == "qGIBBON":
        return greedy_gibbon_batch(
            model, bounds, batch_size, num_restarts, raw_samples,
            config['constraints'], X_pending, limit
        )
    
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_sqmc]))
    
    if acquisition_type == "qEI":
        acq_func = qLogExpectedImprovement(model, best_f=best_f, sampler=sampler)
    elif acquisition_type == "qNEI":
        if x_baseline is None:
            raise ValueError("x_baseline must be provided for qNEI.")
        acq_func = qLogNoisyExpectedImprovement(
            model, X_baseline=x_baseline, sampler=sampler, X_pending=X_pending
        )
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition_type}")
    
    cand, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=config['constraints']
    )
    
    cand = clip_min_values(cand, min_threshold=1e-4)
    print(f"Candidates (scaled):\n{cand}")
    return cand


def optimize_qUCB(
    model, input_dim, batch_size=3, bounds=None, num_restarts=40,
    raw_samples=1024, num_sqmc=512, limit=-100000, beta=2.0
):
    """Optimize qUCB acquisition function."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if bounds is None:
        bounds = _get_default_bounds(input_dim)
    
    config = _get_constraint_config(limit)
    num_restarts = min(num_restarts, config['num_restarts'])
    raw_samples = min(raw_samples, config['raw_samples'])
    num_sqmc = min(num_sqmc, config['num_sqmc'])
    
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_sqmc]))
    acq_func = qUpperConfidenceBound(model, beta=beta, sampler=sampler)
    
    cand, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=config['constraints']
    )
    
    cand = clip_min_values(cand, min_threshold=1e-4)
    print(f"Candidates (scaled, qUCB):\n{cand}")
    return cand


def greedy_gibbon_batch(model, bounds, batch_size=3, num_restarts=30, raw_samples=512, 
                       inequality_constraints=None, X_pending=None, constraint_threshold=-100000):
    """GIBBON batch optimization with constraint-aware candidate generation."""
    input_dim = bounds.size(1)
    
    # Generate constraint-aware candidates
    if constraint_threshold > -100 and input_dim >= 4:
        candidates = torch.rand(5000, input_dim, dtype=torch.double)
        candidates_orig = candidates * (bounds[1] - bounds[0]) + bounds[0]
        constraint_sum = -candidates_orig[:, 2] * 2 - candidates_orig[:, 3] * 0.45414
        feasible_mask = constraint_sum < -constraint_threshold
        candidate_set = candidates[feasible_mask]
        
        if len(candidate_set) < 1000:
            print(f"⚠️ Only {len(candidate_set)} feasible candidates, relaxing constraint...")
            relaxed_mask = constraint_sum < (constraint_threshold * 1.1)
            candidate_set = candidates[relaxed_mask]
        
        candidate_set = candidate_set[:1000]
        print(f"🔍 qGIBBON: {len(candidate_set)} feasible candidates from 5000")
    else:
        candidate_set = torch.rand(1000, input_dim, dtype=torch.double)
        print(f"🔍 qGIBBON: 1000 unconstrained candidates")
    
    # Handle X_pending
    selected = []
    if X_pending is not None:
        X_pending_std = (X_pending - bounds[0]) / (bounds[1] - bounds[0])
        selected = [X_pending_std[i] for i in range(X_pending_std.size(0))]
        print(f"🔍 qGIBBON: {len(selected)} pending points")
    
    # Greedy selection
    for i in range(batch_size):
        X_pending_tensor = torch.stack(selected) if selected else None
        
        try:
            acq = qLowerBoundMaxValueEntropy(
                model=model, candidate_set=candidate_set,
                X_pending=X_pending_tensor, use_gumbel=True
            )
            
            constraints_for_acqf = None if constraint_threshold > -100 else inequality_constraints
            
            x_next, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=torch.tensor([[0.0] * input_dim, [1.0] * input_dim], dtype=torch.double),
                q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                inequality_constraints=constraints_for_acqf
            )
            
            selected.append(x_next.squeeze(0))
            print(f"   Point {i+1}: acq_value = {acq_value:.6f}")
            
        except Exception as e:
            print(f"❌ qGIBBON failed at point {i+1}: {e}")
            # Fallback to random feasible point
            if len(candidate_set) > 0:
                fallback_idx = torch.randint(0, len(candidate_set), (1,))
                selected.append(candidate_set[fallback_idx])
            else:
                selected.append(torch.rand(input_dim, dtype=torch.double))
    
    return torch.stack(selected)


def show_best_posterior_mean(model, input_dim, bounds=None, limit=-100000):
    """Show best posterior mean prediction."""
    best_x = optimize_posterior_mean(
        model=model, input_dim=input_dim, 
        num_restarts=10, raw_samples=100, limit=limit
    )
    
    if best_x is not None:
        model.eval()
        with torch.no_grad():
            best_x_rescaled = rescale_single_point(best_x, bounds)
            best_x_rescaled = clip_min_values(best_x_rescaled, min_threshold=1e-4)
            pred = model.posterior(torch.tensor(best_x).unsqueeze(0)).mean.item()
    
    return best_x_rescaled, pred


def find_promising_uncertain_point(model, train_x, train_y, bounds, n_candidates=5000, limit=-100000, best_n=1, ratio=0.5,diversity_threshold=0.5):
    """
    Finde Punkt(e) der vielversprechend (hohe qEI) UND unsicher (hohe Varianz) ist
    Mit Constraint: x3 + x4 < 2 (auf Original-Skala)
    """
    if isinstance(bounds, list):
        bounds = torch.tensor(bounds, dtype=torch.double).T  # Shape: [2, d]
    elif isinstance(bounds, torch.Tensor) and bounds.shape[0] != 2:
        bounds = bounds.T
    
    print(f"🔍 Debug: bounds shape = {bounds.shape}, type = {type(bounds)}")
    
    # 1. Generiere Kandidaten
    input_dim = bounds.shape[1]
    candidates = torch.rand(n_candidates, input_dim, dtype=torch.double)

    # FIX: Constraint auf Original-Skala prüfen
    if limit > -1e2:  # Wenn Constraint aktiv ist
        # Konvertiere normalisierte Kandidaten [0,1] zu Original-Skala
        candidates_original = candidates * (bounds[1] - bounds[0]) + bounds[0]
        
        # Constraint: x3 + x4 < 2 (Parameter-Indizes 2 und 3 für x3 und x4)
        constraint_sum = candidates_original[:, 2] + candidates_original[:, 3]
        feasible_mask = constraint_sum < -limit
        
        # Filtere feasible Kandidaten
        candidates = candidates[feasible_mask]
        
        if candidates.shape[0] == 0:
            print("⚠️ Keine feasible Kandidaten gefunden. Verwende weniger strikte Constraint...")
            # Fallback: Entspanne Constraint leicht
            constraint_sum = candidates_original[:, 2] + candidates_original[:, 3]
            feasible_mask = constraint_sum < -limit * 1.1  # Leicht entspannt
            candidates = torch.rand(n_candidates, input_dim, dtype=torch.double)[feasible_mask]
            
            if candidates.shape[0] == 0:
                raise ValueError("Keine Kandidaten erfüllen den Constraint x3 + x4 < 2")
        
        print(f"🔍 Nach Constraint-Filtering (x3 + x4 < 2): {candidates.shape[0]} von {n_candidates} Kandidaten")

    # 2. Berechne GP Posterior
    with torch.no_grad():
        posterior = model.posterior(candidates)
        means = posterior.mean.squeeze()
        variances = posterior.variance.squeeze()
        stds = variances.sqrt()
    
    # 3. Berechne qEI für alle Kandidaten
    from botorch.acquisition import qExpectedImprovement
    best_f = train_y.max().item()
    qei = qExpectedImprovement(model, best_f=0.5335)
    
    with torch.no_grad():
        ei_values = qei(candidates.unsqueeze(1)).squeeze()
    
    # 4. Normalisiere beide Scores (0-1)
    ei_normalized = (ei_values - ei_values.min()) / (ei_values.max() - ei_values.min() + 1e-8)
    uncertainty_normalized = (stds - stds.min()) / (stds.max() - stds.min() + 1e-8)
    
    # 5. Kombiniere: Beide Kriterien gleich gewichtet
    combined_score = ratio * ei_normalized + (1-ratio) * uncertainty_normalized
    
        # 6. Diversified Selection
    if best_n == 1:
        best_indices = torch.topk(combined_score, k=1).indices
        best_candidates = candidates[best_indices]
    else:
        # Greedy diversified selection
        selected_indices = []
        
        # Sortiere nach Score (beste zuerst)
        sorted_indices = torch.argsort(combined_score, descending=True)
        
        for candidate_idx in sorted_indices:
            if len(selected_indices) >= best_n:
                break
                
            candidate = candidates[candidate_idx]
            
            # Check diversity: Mindestabstand zu bereits gewählten Punkten
            too_close = False
            for selected_idx in selected_indices:
                selected_point = candidates[selected_idx]
                distance = torch.norm(candidate - selected_point).item()
                if distance < diversity_threshold:
                    too_close = True
                    break
            
            if not too_close:
                selected_indices.append(candidate_idx.item())  # .item() hinzugefügt
        
        # Falls zu wenige diverse Punkte: fülle mit besten verfügbaren auf
        if len(selected_indices) < best_n:
            remaining = torch.topk(combined_score, k=best_n).indices
            for idx in remaining:
                if len(selected_indices) >= best_n:
                    break
                if idx.item() not in selected_indices:
                    selected_indices.append(idx.item())
        
        best_indices = torch.tensor(selected_indices[:best_n])  # Sicherstellen dass wir genau best_n haben
        best_candidates = candidates[best_indices]
    
    best_candidates = clip_min_values(best_candidates, min_threshold=1e-4)
    
    # Debug-Info mit Constraint-Check
    print(f"🎯 {len(best_indices)} vielversprechende aber unsichere Punkte gefunden:")
    
    for i, idx in enumerate(best_indices):
        idx_val = idx.item()
        candidate_original = best_candidates[i] * (bounds[1] - bounds[0]) + bounds[0]
        constraint_value = candidate_original[2].item() + candidate_original[3].item()
        
        print(f"   Kandidat {i+1}:")
        print(f"     qEI Score: {ei_values[idx_val].item():.4f}")
        print(f"     Uncertainty: {stds[idx_val].item():.4f}")
        print(f"     GP Mean: {means[idx_val].item():.4f}")
        print(f"     Combined Score: {combined_score[idx_val].item():.4f}")
        print(f"     Coordinates (normalized): {best_candidates[i].numpy()}")
        print(f"     Coordinates (original): {candidate_original.numpy()}")
        print(f"     Constraint Check: x3 + x4 = {constraint_value:.4f} < 2.0 ✓" if constraint_value < 2.0 else f"     Constraint Check: x3 + x4 = {constraint_value:.4f} ≥ 2.0 ❌")
    
    # Return structure
    if best_n == 1:
        idx = best_indices[0].item()
        return best_candidates[0], {
            'ei_score': ei_values[idx].item(),
            'uncertainty': stds[idx].item(),
            'mean_prediction': means[idx].item(),
            'combined_score': combined_score[idx].item()
        }
    else:
        candidates_info = []
        for i, idx in enumerate(best_indices):
            idx_val = idx.item()
            candidates_info.append({
                'ei_score': ei_values[idx_val].item(),
                'uncertainty': stds[idx_val].item(),
                'mean_prediction': means[idx_val].item(),
                'combined_score': combined_score[idx_val].item()
            })
        
        return best_candidates, candidates_info