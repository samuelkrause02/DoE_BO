import numpy as np
import pandas as pd
from scipy import stats
from experiment_design import ParameterSpace, OrthogonalSamplingDesigner
from SampleVis import mann_kendall, pearson, util, projection_properties, space_filling_measures


def build_parameter_space(param_defs):
    """
    Create the parameter space from user-defined variable definitions.

    Args:
        param_defs (list): List of tuples (name, lower_bound, upper_bound, is_discrete)

    Returns:
        ParameterSpace object, list of column names
    """
    dists = []
    col_names = []
    for name, low, high, is_disc in param_defs:
        if is_disc:
            dist = stats.randint(int(low), int(high + 1))
        else:
            dist = stats.uniform(loc=low, scale=high - low)
        dists.append(dist)
        col_names.append(name)
    return ParameterSpace(dists), col_names


def generate_orthogonal_samples(space, n_samples, col_names, seed=42):
    """
    Perform orthogonal sampling using the provided parameter space.

    Args:
        space (ParameterSpace): The defined parameter space
        n_samples (int): Number of samples to generate
        col_names (list): Names for the resulting DataFrame columns
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with orthogonal samples
    """
    np.random.seed(seed)
    designer = OrthogonalSamplingDesigner()
    samples = designer.design(space, int(n_samples))
    return pd.DataFrame(samples, columns=col_names)


def normalize_data(df, param_defs):
    """
    Normalize all DataFrame columns to the [0, 1] range.

    Args:
        df (pd.DataFrame): DataFrame to normalize

    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    normalized_df = df.copy()
    for i, (name, low, high, _) in enumerate(param_defs):
        normalized_df[name] = (df[name] - low) / (high - low)
    return normalized_df


def analyze_sample(sample_df):
    """
    Run correlation tests, projections, and space-filling measures on the sample.

    Args:
        sample_df (pd.DataFrame): Normalized sample data

    Returns:
        tuple: (1D coverage, L2-star discrepancy, min. distance)
    """
    sample = sample_df.to_numpy()
    names = sample_df.columns.tolist()

    # Correlation analysis
    z_mk, pval_mk = mann_kendall.test_sample(sample)
    rho, pval_pr = pearson.test_sample(sample)
    util.correlation_plots(z_mk, pval_mk, 'Mann-Kendall', names)
    util.correlation_plots(rho, pval_pr, 'Pearson', names)

    # 2D projection plots (if dimensionality is low)
    if sample.shape[1] < 11:
        projection_properties.projection_2D(sample, names)

    # 1D projection and space-filling metrics
    coverage_1d = projection_properties.projection_1D(sample, names)
    discrepancy = space_filling_measures.discrepancy(sample)
    min_dist = space_filling_measures.min_distance(sample, 1)

    return coverage_1d, discrepancy, min_dist

import torch
def build_param_defs(data, bounds, discrete=None, output_col_name=None):
    """
    Build parameter definitions from data and bounds.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        bounds (list or tensor): Bounds as list of (low, high) or 2xD torch.Tensor
        discrete (list of bool): List indicating which parameters are discrete
        output_col_name (str): Name of the output column to exclude

    Returns:
        list of tuples: [(param_name, lower_bound, upper_bound, is_discrete), ...]
    """
    # Entferne Output-Spalte
    if output_col_name is not None:
        data = data.drop(columns=[output_col_name], errors='ignore')

    param_columns = list(data.columns)

    # Konvertiere torch.Tensor bounds → List of tuples
    if isinstance(bounds, torch.Tensor) and bounds.shape[0] == 2:
        bounds = list(zip(bounds[0].tolist(), bounds[1].tolist()))

    if len(bounds) != len(param_columns):
        raise ValueError("Length of bounds must match number of input parameters.")

    if discrete is None:
        discrete = [False] * len(param_columns)
    if len(discrete) != len(param_columns):
        raise ValueError("Length of `discrete` must match number of input parameters.")

    if len(data) == 0:
        raise ValueError("Input data is empty.")

    param_defs = []
    for i, col in enumerate(param_columns):
        low, high = bounds[i]
        param_defs.append((col, low, high, bool(discrete[i])))

    return param_defs