import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class RobustnessAnalyzer:
    """
    Comprehensive robustness analysis for Bayesian Optimization strategies
    Including Transfer Learning support
    """
    
    def __init__(self, df_results: pd.DataFrame):
        """
        Initialize with simulation results DataFrame
        
        Expected columns:
        - acquisition_type, num_init, seed, iteration, mse, regret, 
        - observed_best, true_best, posterior_mean, use_transfer_learning, etc.
        """
        self.df = df_results.copy()
        self.validate_data()
        
    def validate_data(self):
        """Validate required columns exist"""
        required_cols = ['acquisition_type', 'num_init', 'seed', 'iteration']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for transfer learning column and standardize
        if 'use_transfer_learning' in self.df.columns:
            # Convert to boolean and handle None values
            self.df['use_transfer_learning'] = self.df['use_transfer_learning'].fillna(False).astype(bool)
            print(f"✅ Transfer learning column found and standardized")
        else:
            # Add default column if missing
            self.df['use_transfer_learning'] = False
            print(f"⚠️ No transfer learning column found, assuming False for all strategies")
        
        # Add regret if not present
        if 'regret' not in self.df.columns and 'observed_best' in self.df.columns and 'true_best' in self.df.columns:
            self.df['regret'] = (self.df['true_best'] - self.df['observed_best']) ** 2
        
        # Create comprehensive strategy identifier
        self.df['strategy_full'] = self.df.apply(
            lambda x: f"{x['acquisition_type']} (init={x['num_init']}) (TL={x['use_transfer_learning']})", 
            axis=1
        )
        
        print(f"✅ Data validated. Shape: {self.df.shape}")
        print(f"📊 Available metrics: {[col for col in self.df.columns if col not in ['acquisition_type', 'num_init', 'seed', 'iteration', 'use_transfer_learning', 'strategy_full']]}")
        print(f"🔄 Transfer learning strategies: {self.df['use_transfer_learning'].sum()}/{len(self.df)} rows")
        print(f"🎯 Unique strategies: {self.df['strategy_full'].nunique()}")
        
    def compute_robustness_metrics(self, metric_col: str = 'regret') -> pd.DataFrame:
        """
        Compute basic robustness metrics for each strategy including transfer learning
        """
        if metric_col not in self.df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in data")
        
        # Get final values (last iteration for each run) - now includes transfer learning
        final_values = self.df.groupby(['acquisition_type', 'num_init', 'use_transfer_learning', 'seed'])[metric_col].last().reset_index()
        
        # Compute statistics across seeds (ground truth variants) - grouped by transfer learning too
        robustness_stats = final_values.groupby(['acquisition_type', 'num_init', 'use_transfer_learning'])[metric_col].agg([
            'count',   # Number of successful runs
            'mean',    # Average performance
            'std',     # Variability (robustness indicator)
            'min',     # Best case
            'max',     # Worst case
            lambda x: np.percentile(x, 25),  # Q1
            lambda x: np.percentile(x, 75),  # Q3
            lambda x: np.percentile(x, 95),  # Near worst-case
        ]).round(4)
        
        robustness_stats.columns = ['n_runs', 'mean', 'std', 'best_case', 'worst_case', 'q25', 'q75', 'q95']
        
        # Compute derived metrics
        robustness_stats['cv'] = robustness_stats['std'] / robustness_stats['mean']  # Coefficient of variation
        robustness_stats['iqr'] = robustness_stats['q75'] - robustness_stats['q25']  # Interquartile range
        robustness_stats['robustness_score'] = 1 / (1 + robustness_stats['std'])    # Higher = more robust
        robustness_stats['performance_score'] = 1 / (1 + robustness_stats['mean'])  # Higher = better performance
        
        # Success rate (fraction below median performance)
        median_performance = final_values[metric_col].median()
        success_rates = final_values.groupby(['acquisition_type', 'num_init', 'use_transfer_learning'])[metric_col].apply(
            lambda x: (x <= median_performance).mean()
        )
        robustness_stats['success_rate'] = success_rates
        
        # Add strategy label
        robustness_stats = robustness_stats.reset_index()
        robustness_stats['strategy_full'] = robustness_stats.apply(
            lambda x: f"{x['acquisition_type']} (init={x['num_init']}) (TL={x['use_transfer_learning']})", 
            axis=1
        )
        
        return robustness_stats

    def compute_comprehensive_robustness_metrics(self, primary_metric: str = 'regret') -> pd.DataFrame:
        """
        Compute comprehensive robustness metrics including surrogate quality metrics and transfer learning
        """
        if primary_metric not in self.df.columns:
            raise ValueError(f"Primary metric '{primary_metric}' not found in data")
        
        # Get final values for optimization-focused metrics (last iteration for each run)
        final_values = self.df.groupby(['acquisition_type', 'num_init', 'use_transfer_learning', 'seed'])[primary_metric].last().reset_index()
        
        # Get all iterations for surrogate quality metrics
        all_iterations = self.df.copy()
        
        results = []
        
        # Group by acquisition type, num_init, AND use_transfer_learning
        for (acq_type, num_init, use_tl), group in final_values.groupby(['acquisition_type', 'num_init', 'use_transfer_learning']):
            strategy_data = all_iterations[
                (all_iterations['acquisition_type'] == acq_type) & 
                (all_iterations['num_init'] == num_init) &
                (all_iterations['use_transfer_learning'] == use_tl)
            ]
            
            # 1. OPTIMIZATION PERFORMANCE METRICS (based on primary metric)
            primary_values = group[primary_metric]
            
            opt_metrics = {
                'acquisition_type': acq_type,
                'num_init': num_init,
                'use_transfer_learning': use_tl,
                'strategy_full': f"{acq_type} (init={num_init}) (TL={use_tl})",
                'n_runs': len(primary_values),
                
                # Performance metrics
                f'{primary_metric}_mean': primary_values.mean(),
                f'{primary_metric}_std': primary_values.std(),
                f'{primary_metric}_min': primary_values.min(),
                f'{primary_metric}_max': primary_values.max(),
                f'{primary_metric}_q25': primary_values.quantile(0.25),
                f'{primary_metric}_q75': primary_values.quantile(0.75),
                f'{primary_metric}_q95': primary_values.quantile(0.95),
                
                # Derived metrics
                f'{primary_metric}_cv': primary_values.std() / primary_values.mean() if primary_values.mean() != 0 else np.inf,
                f'{primary_metric}_iqr': primary_values.quantile(0.75) - primary_values.quantile(0.25),
            }
            
            # 2. SURROGATE QUALITY METRICS (aggregated across all iterations)
            surrogate_metrics = {}
            
            # Collect available surrogate metrics
            available_surrogate_metrics = [
                'mse', 'r2', 'rmse_h', 'crps', 'coverage_95', 'coverage_high_region',
                'CI_size_avg'
            ]
            
            for metric in available_surrogate_metrics:
                if metric in strategy_data.columns:
                    values = strategy_data[metric].dropna()
                    if len(values) > 0:
                        surrogate_metrics.update({
                            f'{metric}_mean': values.mean(),
                            f'{metric}_std': values.std(),
                            f'{metric}_final': strategy_data.groupby('seed')[metric].last().mean(),  # Final iteration average
                        })
            
            # 3. MOVING WINDOW CONVERGENCE METRICS  
            convergence_metrics = {}
            
            if 'iteration' in strategy_data.columns:
                # Initialize convergence parameters if not set
                if not hasattr(self, '_convergence_params'):
                    self._calculate_moving_window_convergence(primary_metric)
                
                convergence_data = []
                convergence_details = []
                early_convergence_data = []
                
                # Analyze each seed individually
                for seed in strategy_data['seed'].unique():
                    seed_data = strategy_data[strategy_data['seed'] == seed].sort_values('iteration')
                    if len(seed_data) > 1 and primary_metric in seed_data.columns:
                        values = seed_data[primary_metric].values
                        total_iterations = len(seed_data)
                        
                        # Use moving window convergence detection
                        try:
                            conv_result = detect_moving_window_convergence(
                                values=values,
                                window_size=self._convergence_params.get('window_size', 5),
                                tolerance_type=self._convergence_params.get('tolerance_type', 'relative'),
                                tolerance_threshold=self._convergence_params.get('tolerance_threshold', 0.02),
                                min_iterations=10
                            )
                            
                            if conv_result['converged']:
                                convergence_iter = conv_result['convergence_iteration']
                                
                                # BOUNDS CHECK to prevent iteration > total_iterations
                                if convergence_iter >= total_iterations:
                                    convergence_iter = total_iterations - 1
                                
                                convergence_data.append(convergence_iter)
                                
                                # Check for early convergence (before 80% of iterations)
                                early_threshold = int(0.8 * total_iterations)
                                if convergence_iter < early_threshold:
                                    early_convergence_data.append(convergence_iter)
                                
                                # Store detailed info
                                convergence_details.append({
                                    'seed': seed,
                                    'convergence_iteration': convergence_iter,
                                    'early_convergence': convergence_iter < early_threshold,
                                    'final_regret': conv_result['final_value'],
                                    'convergence_value': conv_result['convergence_value'],
                                    'total_iterations': total_iterations,
                                    'efficiency_ratio': convergence_iter / total_iterations,
                                    'convergence_metric': conv_result.get('convergence_metric', np.nan),
                                    'convergence_window': conv_result.get('convergence_window', [])
                                })
                            else:
                                # Did not converge
                                convergence_details.append({
                                    'seed': seed,
                                    'convergence_iteration': None,
                                    'early_convergence': False,
                                    'final_regret': conv_result['final_value'],
                                    'convergence_value': None,
                                    'total_iterations': total_iterations,
                                    'efficiency_ratio': 1.0,
                                    'reason': conv_result.get('reason', 'No convergence detected')
                                })
                        
                        except Exception as e:
                            # Fallback if moving window function not available
                            print(f"Warning: Moving window convergence failed for {acq_type} (TL={use_tl}): {e}")
                            convergence_details.append({
                                'seed': seed,
                                'convergence_iteration': None,
                                'early_convergence': False,
                                'final_regret': values[-1] if len(values) > 0 else np.nan,
                                'convergence_value': None,
                                'total_iterations': total_iterations,
                                'efficiency_ratio': 1.0,
                                'reason': 'Moving window function failed'
                            })
                
                # Calculate convergence metrics
                if convergence_data:
                    convergence_metrics.update({
                        'convergence_speed_mean': np.mean(convergence_data),
                        'convergence_speed_std': np.std(convergence_data),
                        'convergence_speed_min': np.min(convergence_data),
                        'convergence_speed_max': np.max(convergence_data),
                        'convergence_speed_median': np.median(convergence_data),
                        'convergence_reliability': len(convergence_data) / len(strategy_data['seed'].unique()),
                        'early_convergence_rate': len(early_convergence_data) / len(strategy_data['seed'].unique()),
                        'convergence_success_count': len(convergence_data),
                        'early_convergence_count': len(early_convergence_data),
                        'convergence_total_seeds': len(strategy_data['seed'].unique()),
                        'avg_efficiency_ratio': np.mean([d['efficiency_ratio'] for d in convergence_details if d['convergence_iteration'] is not None]) if convergence_details else 1.0
                    })
                else:
                    convergence_metrics.update({
                        'convergence_speed_mean': np.nan,
                        'convergence_speed_std': np.nan,
                        'convergence_speed_min': np.nan,
                        'convergence_speed_max': np.nan,
                        'convergence_speed_median': np.nan,
                        'convergence_reliability': 0.0,
                        'early_convergence_rate': 0.0,
                        'convergence_success_count': 0,
                        'early_convergence_count': 0,
                        'convergence_total_seeds': len(strategy_data['seed'].unique()),
                        'avg_efficiency_ratio': 1.0
                    })
                
                # Store detailed convergence info
                convergence_metrics['convergence_details'] = convergence_details
            
            # 4. COMBINE ALL METRICS
            combined_metrics = {**opt_metrics, **surrogate_metrics, **convergence_metrics}
            results.append(combined_metrics)
        
        df_results = pd.DataFrame(results)
        
        # 5. COMPUTE COMPOSITE SCORES
        if len(df_results) > 0:
            # Performance score (lower primary metric is better)
            if f'{primary_metric}_mean' in df_results.columns:
                df_results['performance_score'] = 1 / (1 + df_results[f'{primary_metric}_mean'])
            
            # Robustness score (lower std is better)  
            if f'{primary_metric}_std' in df_results.columns:
                df_results['robustness_score'] = 1 / (1 + df_results[f'{primary_metric}_std'])
            
            # Safety score (lower q95 is better)
            if f'{primary_metric}_q95' in df_results.columns:
                min_q95 = df_results[f'{primary_metric}_q95'].min()
                max_q95 = df_results[f'{primary_metric}_q95'].max()
                if max_q95 > min_q95:
                    df_results['safety_score'] = 1 - (df_results[f'{primary_metric}_q95'] - min_q95) / (max_q95 - min_q95)
                else:
                    df_results['safety_score'] = 1.0
            
            # Surrogate quality score (higher r2, lower mse is better)
            surrogate_components = []
            
            if 'r2_mean' in df_results.columns:
                surrogate_components.append(df_results['r2_mean'].fillna(0))
            
            if 'mse_mean' in df_results.columns:
                mse_vals = df_results['mse_mean'].fillna(df_results['mse_mean'].max())
                surrogate_components.append(1 / (1 + mse_vals))
                
            if 'rmse_h_mean' in df_results.columns:
                rmse_vals = df_results['rmse_h_mean'].fillna(df_results['rmse_h_mean'].max())
                surrogate_components.append(1 / (1 + rmse_vals))
            
            if 'crps_mean' in df_results.columns:
                crps_vals = df_results['crps_mean'].fillna(df_results['crps_mean'].max())
                surrogate_components.append(1 / (1 + crps_vals))
            
            if surrogate_components:
                df_results['surrogate_quality_score'] = np.mean(surrogate_components, axis=0)
            else:
                df_results['surrogate_quality_score'] = 0.5
            
            # Convergence score
            if 'convergence_reliability' in df_results.columns and 'convergence_speed_mean' in df_results.columns:
                # 1. Zuverlässigkeit (0-1)
                reliability = df_results['convergence_reliability'].fillna(0)
                
                # 2. BO-Effizienz Score (nur BO Iterationen)
                bo_iterations = df_results['convergence_speed_mean'].fillna(100)
                bo_efficiency_score = 1 / (1 + bo_iterations)
                
                # 3. Gesamtkosten Score (BO + Initial)
                if 'num_init' in df_results.columns:
                    total_experiments = bo_iterations + df_results['num_init']
                    total_cost_score = 1 / (1 + total_experiments)
                else:
                    total_cost_score = bo_efficiency_score  # Fallback
                
                # 4. Kombinierter Speed Score
                # 60% BO-Effizienz, 40% Gesamtkosten
                combined_speed_score = 0.2 * bo_efficiency_score + 0.8 * total_cost_score
                
                # 5. Finaler Convergence Score
                # 70% Zuverlässigkeit, 30% Geschwindigkeit
                df_results['convergence_score'] = 0.5 * reliability + 0.5 * combined_speed_score
                
                # Strategien die nie konvergieren bekommen Score 0
                df_results.loc[reliability == 0, 'convergence_score'] = 0.0
                
                # Debug Info
                print(f"Speed Score Berechnung:")
                print(f"- BO Effizienz: {bo_efficiency_score.mean():.3f} (Durchschnitt)")
                print(f"- Gesamtkosten: {total_cost_score.mean():.3f} (Durchschnitt)")
                print(f"- Kombiniert: {combined_speed_score.mean():.3f} (Durchschnitt)")
                
            else:
                # Fallback
                df_results['convergence_score'] = df_results['convergence_reliability'].fillna(0)
        
        return df_results

    def get_experiment_planning_analysis(self, primary_metric: str = 'regret') -> pd.DataFrame:
        """
        Extract practical experiment planning analysis focused on real BO strategies.
        Reports total experiments needed (initial + BO iterations), not just BO iterations.
        Now includes transfer learning distinction.
        """
        comprehensive_metrics = self.compute_comprehensive_robustness_metrics(primary_metric)
        
        # Filter out random strategies - they don't represent real BO planning
        bo_strategies = comprehensive_metrics[
            comprehensive_metrics['acquisition_type'].str.upper() != 'RANDOM'
        ].copy()
        
        if bo_strategies.empty:
            return pd.DataFrame()
        
        convergence_results = []
        
        for _, row in bo_strategies.iterrows():
            if 'convergence_details' in row and row['convergence_details']:
                details = row['convergence_details']
                
                strategy_name = row['strategy_full']  # Already includes TL info
                
                # Extract convergence statistics
                converged_seeds = [d for d in details if d['convergence_iteration'] is not None]
                failed_seeds = [d for d in details if d['convergence_iteration'] is None]
                
                if converged_seeds:
                    # CRITICAL: Convert BO iterations to TOTAL experiments
                    # Total experiments = initial_points + BO_iterations
                    initial_experiments = row['num_init']
                    
                    total_experiments_list = []
                    
                    for d in converged_seeds:
                        bo_iterations = d['convergence_iteration']
                        total_experiments = initial_experiments + bo_iterations
                        total_experiments_list.append(total_experiments)
                    
                    convergence_results.append({
                        'strategy': strategy_name,
                        'acquisition_type': row['acquisition_type'],
                        'num_init': row['num_init'],
                        'use_transfer_learning': row['use_transfer_learning'],
                        'success_rate': len(converged_seeds) / len(details),
                        'successful_seeds': len(converged_seeds),
                        'failed_seeds': len(failed_seeds),
                        'total_seeds': len(details),
                        
                        # Total experiments needed (what practitioners actually care about)
                        'min_total_experiments': min(total_experiments_list),
                        'max_total_experiments': max(total_experiments_list),
                        'median_total_experiments': np.median(total_experiments_list),
                        'mean_total_experiments': np.mean(total_experiments_list),
                        'std_total_experiments': np.std(total_experiments_list),
                        'total_experiments_90th_percentile': np.percentile(total_experiments_list, 90),
                        'total_experiments_95th_percentile': np.percentile(total_experiments_list, 95),
                        
                        # For reference: BO-only metrics
                        'min_bo_iterations': min([d['convergence_iteration'] for d in converged_seeds]),
                        'median_bo_iterations': np.median([d['convergence_iteration'] for d in converged_seeds]),
                        'max_bo_iterations': max([d['convergence_iteration'] for d in converged_seeds]),
                        
                        'max_available_iterations': details[0]['total_iterations']
                    })
                else:
                    # No seeds converged
                    initial_experiments = row['num_init']
                    max_total_experiments = initial_experiments + details[0]['total_iterations']
                    
                    convergence_results.append({
                        'strategy': strategy_name,
                        'acquisition_type': row['acquisition_type'],
                        'num_init': row['num_init'],
                        'use_transfer_learning': row['use_transfer_learning'],
                        'success_rate': 0.0,
                        'successful_seeds': 0,
                        'failed_seeds': len(failed_seeds),
                        'total_seeds': len(details),
                        'min_total_experiments': np.nan,
                        'max_total_experiments': np.nan,
                        'median_total_experiments': np.nan,
                        'mean_total_experiments': np.nan,
                        'std_total_experiments': np.nan,
                        'total_experiments_90th_percentile': np.nan,
                        'total_experiments_95th_percentile': np.nan,
                        'min_bo_iterations': np.nan,
                        'median_bo_iterations': np.nan,
                        'max_bo_iterations': np.nan,
                        'max_available_experiments': max_total_experiments
                    })
        
        return pd.DataFrame(convergence_results)

    def plot_convergence_robustness(self, metric_col: str = 'regret', max_strategies: int = 8):
        """
        Plot convergence with robustness bands - now includes transfer learning distinction
        """
        # Get strategies with transfer learning info
        strategies = self.df.groupby(['acquisition_type', 'num_init', 'use_transfer_learning']).size().index[:max_strategies]
        
        # Calculate number of rows and columns for subplots
        n_strategies = len(strategies)
        cols = 3
        rows = (n_strategies + cols - 1) // cols  # Ceiling division
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{acq} (init={init}) (TL={tl})" for acq, init, tl in strategies],
            shared_yaxes=True
        )
        
        # Use safe color palette
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        for idx, (acq_type, num_init, use_tl) in enumerate(strategies):
            row = idx // cols + 1
            col = idx % cols + 1
            
            subset = self.df[
                (self.df['acquisition_type'] == acq_type) & 
                (self.df['num_init'] == num_init) &
                (self.df['use_transfer_learning'] == use_tl)
            ]
            
            if subset.empty:
                continue
            
            # Calculate statistics per iteration
            convergence_stats = subset.groupby('iteration')[metric_col].agg([
                'mean', 'std', 'min', 'max',
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75)
            ]).reset_index()
            convergence_stats.columns = ['iteration', 'mean', 'std', 'min', 'max', 'q25', 'q75']
            
            iterations = convergence_stats['iteration']
            mean_vals = convergence_stats['mean']
            
            # Get color for this strategy
            color = colors[idx % len(colors)]
            
            # Convert hex to RGB
            rgb = mcolors.hex2color(color)
            rgba_str = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.2)"
            
            # Median line
            fig.add_trace(
                go.Scatter(
                    x=iterations, y=mean_vals,
                    mode='lines+markers',
                    name=f'{acq_type} (TL={use_tl})',
                    line=dict(color=color, width=2),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Confidence band (Q1-Q3)
            fig.add_trace(
                go.Scatter(
                    x=iterations.tolist() + iterations.tolist()[::-1],
                    y=convergence_stats['q25'].tolist() + convergence_stats['q75'].tolist()[::-1],
                    fill='tonexty' if idx > 0 else 'tozeroy',
                    fillcolor=rgba_str,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Q1-Q3'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'BO Convergence Robustness Analysis ({metric_col.upper()}) - Including Transfer Learning',
            height=200 * rows,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="BO Iteration")
        fig.update_yaxes(title_text=metric_col.upper())
        
        return fig
    
    def plot_robustness_comparison(self, metric_col: str = 'regret'):
        """
        Create comprehensive robustness comparison plots - now includes transfer learning
        """
        robustness_metrics = self.compute_robustness_metrics(metric_col)
        
        # Calculate combined score here if not present
        if 'combined_score' not in robustness_metrics.columns:
            # Normalize scores (0-1 scale) with safety checks
            max_perf = robustness_metrics['performance_score'].max()
            max_robust = robustness_metrics['robustness_score'].max()
            min_q95 = robustness_metrics['q95'].min()

            if max_perf > 0:
                perf_scores = robustness_metrics['performance_score'] / max_perf
            else:
                perf_scores = robustness_metrics['performance_score'] * 0 + 0.5  # Default to 0.5
                
            if max_robust > 0:
                robust_scores = robustness_metrics['robustness_score'] / max_robust
            else:
                robust_scores = robustness_metrics['robustness_score'] * 0 + 0.5
                
            if min_q95 > 0 and not np.isnan(min_q95):
                # Inverse relationship: lower q95 = higher safety score
                max_q95 = robustness_metrics['q95'].max()
                if max_q95 > min_q95:
                    safety_scores = 1 - (robustness_metrics['q95'] - min_q95) / (max_q95 - min_q95)
                else:
                    safety_scores = robustness_metrics['q95'] * 0 + 1.0
            else:
                safety_scores = robustness_metrics['q95'] * 0 + 0.5
            
            # Combined score with equal weights for visualization
            robustness_metrics['combined_score'] = (
                0.4 * perf_scores + 
                0.4 * robust_scores + 
                0.2 * safety_scores
            )
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Performance vs Robustness', 
                'Distribution of Final Performance',
                'Risk-Return Analysis',
                'Strategy Ranking (with Transfer Learning)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Use safe color palette
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Create strategy colors that distinguish TL
        strategy_colors = []
        for i, (_, row) in enumerate(robustness_metrics.iterrows()):
            base_color = colors[i % len(colors)]
            if row['use_transfer_learning']:
                # Darker shade for TL=True
                strategy_colors.append(base_color)
            else:
                # Lighter shade for TL=False
                rgb = mcolors.hex2color(base_color)
                lighter_rgb = [min(1.0, c + 0.3) for c in rgb]
                strategy_colors.append(mcolors.rgb2hex(lighter_rgb))
        
        # 1. Performance vs Robustness scatter
        fig.add_trace(
            go.Scatter(
                x=robustness_metrics['mean'],
                y=robustness_metrics['robustness_score'],
                mode='markers+text',
                text=robustness_metrics['strategy_full'],
                textposition="top center",
                marker=dict(
                    size=robustness_metrics['success_rate'] * 20 + 5,
                    color=strategy_colors,
                    line=dict(width=1, color='white')
                ),
                name='Strategies'
            ),
            row=1, col=1
        )
        
        # 2. Box plots of final performance
        final_values = self.df.groupby(['acquisition_type', 'num_init', 'use_transfer_learning', 'seed'])[metric_col].last().reset_index()
        final_values['strategy_full'] = final_values.apply(
            lambda x: f"{x['acquisition_type']} (init={x['num_init']}) (TL={x['use_transfer_learning']})", axis=1
        )
        
        for idx, strategy in enumerate(final_values['strategy_full'].unique()):
            strategy_data = final_values[final_values['strategy_full'] == strategy][metric_col]
            
            fig.add_trace(
                go.Box(
                    y=strategy_data,
                    name=strategy,
                    marker_color=colors[idx % len(colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Risk-Return (Mean vs Std)  
        fig.add_trace(
            go.Scatter(
                x=robustness_metrics['std'],
                y=robustness_metrics['mean'],
                mode='markers+text',
                text=robustness_metrics['strategy_full'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=robustness_metrics['combined_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Combined Score", x=1.1),
                    line=dict(width=1, color='white')
                ),
                name='Risk-Return'
            ),
            row=2, col=1
        )
        
        # 4. Strategy ranking bar chart
        top_strategies = robustness_metrics.sort_values('combined_score', ascending=True).tail(8)
        
        # Create color list for bars that shows TL distinction
        bar_colors = []
        for _, row in top_strategies.iterrows():
            if row['use_transfer_learning']:
                bar_colors.append('#2ca02c')  # Green for TL=True
            else:
                bar_colors.append('#ff7f0e')  # Orange for TL=False
        
        fig.add_trace(
            go.Bar(
                x=top_strategies['combined_score'],
                y=top_strategies['strategy_full'],
                orientation='h',
                marker_color=bar_colors,
                name='Combined Score'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text=f"Mean {metric_col.upper()}", row=1, col=1)
        fig.update_yaxes(title_text="Robustness Score", row=1, col=1)
        
        fig.update_yaxes(title_text=f"{metric_col.upper()}", row=1, col=2)
        
        fig.update_xaxes(title_text=f"Std {metric_col.upper()} (Risk)", row=2, col=1)
        fig.update_yaxes(title_text=f"Mean {metric_col.upper()}", row=2, col=1)
        
        fig.update_xaxes(title_text="Combined Score", row=2, col=2)
        
        fig.update_layout(
            title=f'Comprehensive Robustness Analysis ({metric_col.upper()}) - Including Transfer Learning',
            height=800,
            showlegend=False
        )
        
        return fig

    def recommend_comprehensive_strategy(self, 
                                       primary_metric: str = 'regret',
                                       performance_weight: float = 0.25,
                                       robustness_weight: float = 0.25, 
                                       safety_weight: float = 0.15,
                                       surrogate_weight: float = 0.20,
                                       convergence_weight: float = 0.15) -> Dict:
        """
        Multi-criteria strategy recommendation including surrogate quality and transfer learning
        """
        comprehensive_metrics = self.compute_comprehensive_robustness_metrics(primary_metric)
        
        if len(comprehensive_metrics) == 0:
            raise ValueError("No metrics computed - check your data")
        
        # Normalize all scores to 0-1 range
        score_columns = ['performance_score', 'robustness_score', 'safety_score', 
                        'surrogate_quality_score', 'convergence_score']
        
        normalized_scores = {}
        for col in score_columns:
            if col in comprehensive_metrics.columns:
                values = comprehensive_metrics[col].fillna(0)
                max_val = values.max()
                if max_val > 0:
                    normalized_scores[col] = values / max_val
                else:
                    normalized_scores[col] = values
            else:
                normalized_scores[col] = pd.Series([0.5] * len(comprehensive_metrics))
        
        # Combined score with weights
        combined_scores = (
            performance_weight * normalized_scores['performance_score'] + 
            robustness_weight * normalized_scores['robustness_score'] + 
            safety_weight * normalized_scores['safety_score'] +
            surrogate_weight * normalized_scores['surrogate_quality_score'] +
            convergence_weight * normalized_scores['convergence_score']
        )
        
        comprehensive_metrics['combined_score'] = combined_scores
        
        # Find best strategy
        best_idx = combined_scores.idxmax()
        best_strategy = comprehensive_metrics.iloc[best_idx]
        
        recommendation = {
            'strategy': best_strategy['strategy_full'],  # Now includes TL info
            'combined_score': best_strategy['combined_score'],
            'performance_score': normalized_scores['performance_score'].iloc[best_idx],
            'robustness_score': normalized_scores['robustness_score'].iloc[best_idx],
            'safety_score': normalized_scores['safety_score'].iloc[best_idx],
            'surrogate_quality_score': normalized_scores['surrogate_quality_score'].iloc[best_idx],
            'convergence_score': normalized_scores['convergence_score'].iloc[best_idx],
            'metrics': best_strategy.to_dict(),
            'all_strategies': comprehensive_metrics.sort_values('combined_score', ascending=False),
            'weights_used': {
                'performance': performance_weight,
                'robustness': robustness_weight,
                'safety': safety_weight,
                'surrogate_quality': surrogate_weight,
                'convergence': convergence_weight
            }
        }
        
        return recommendation

    def recommend_strategy(self, 
                          metric_col: str = 'regret',
                          performance_weight: float = 0.4,
                          robustness_weight: float = 0.4, 
                          safety_weight: float = 0.2) -> Dict:
        """
        Multi-criteria strategy recommendation - now includes transfer learning
        """
        robustness_metrics = self.compute_robustness_metrics(metric_col)
        
        # Normalize scores (0-1 scale)
        perf_scores = robustness_metrics['performance_score'] / robustness_metrics['performance_score'].max()
        robust_scores = robustness_metrics['robustness_score'] / robustness_metrics['robustness_score'].max()
        
        # Safety scores: inverse relationship with q95 (lower q95 = better = higher score)
        min_q95 = robustness_metrics['q95'].min()
        max_q95 = robustness_metrics['q95'].max()
        
        if max_q95 > min_q95:
            safety_scores = 1 - (robustness_metrics['q95'] - min_q95) / (max_q95 - min_q95)
        else:
            safety_scores = pd.Series([1.0] * len(robustness_metrics), index=robustness_metrics.index)
        
        # Combined score
        combined_scores = (
            performance_weight * perf_scores + 
            robustness_weight * robust_scores + 
            safety_weight * safety_scores
        )
        
        robustness_metrics['combined_score'] = combined_scores
        
        # Find best strategy
        best_idx = combined_scores.idxmax()
        best_strategy = robustness_metrics.iloc[best_idx]
        
        recommendation = {
            'strategy': best_strategy['strategy_full'],  # Now includes TL info
            'combined_score': best_strategy['combined_score'],
            'performance_score': perf_scores.iloc[best_idx],
            'robustness_score': robust_scores.iloc[best_idx], 
            'safety_score': safety_scores.iloc[best_idx],
            'metrics': best_strategy.to_dict(),
            'all_strategies': robustness_metrics.sort_values('combined_score', ascending=False)
        }
        
        return recommendation

    def create_practical_experiment_planning_report(self, primary_metric: str = 'regret') -> str:
        """
        Generate practical experiment planning report focused on total experiments needed
        Now includes transfer learning distinction
        """
        convergence_df = self.get_experiment_planning_analysis(primary_metric)
        
        if convergence_df.empty:
            return "# ❌ No BO strategies available for experiment planning\n\nOnly Random strategies found - these are not suitable for practical BO planning."
        
        # Sort by median total experiments needed
        try:
            convergence_df_sorted = convergence_df.sort_values('median_total_experiments', na_position='last')
        except TypeError:
            temp_col = convergence_df['median_total_experiments'].fillna(999999)
            sort_idx = temp_col.argsort()
            convergence_df_sorted = convergence_df.iloc[sort_idx].reset_index(drop=True)
        
        report = f"""
# 🧪 Practical BO Experiment Planning Report (Including Transfer Learning)

## 📊 **Executive Summary: Total Experiments Required**

This analysis focuses on **real Bayesian Optimization strategies** including **Transfer Learning variants** and reports **total experiments needed** (initial sampling + BO iterations) for practical deployment planning.

"""
        
        # Quick summary for top strategies
        for _, row in convergence_df_sorted.head(3).iterrows():
            if not pd.isna(row['median_total_experiments']):
                tl_status = "✅ With Transfer Learning" if row['use_transfer_learning'] else "❌ No Transfer Learning"
                report += f"""
### 🏆 {row['strategy']}
**{tl_status}**
- **Typical Budget**: {row['median_total_experiments']:.0f} total experiments
- **Conservative Budget**: {row['total_experiments_90th_percentile']:.0f} total experiments  
- **Success Rate**: {row['success_rate']:.1%} (converges to target performance)
- **Range**: {row['min_total_experiments']:.0f} - {row['max_total_experiments']:.0f} total experiments
- **Breakdown**: {row['num_init']} initial + {row['median_bo_iterations']:.0f} BO iterations
"""
            else:
                tl_status = "✅ With Transfer Learning" if row['use_transfer_learning'] else "❌ No Transfer Learning"
                report += f"""
### ❌ {row['strategy']}
**{tl_status}**
- **Status**: Did not converge within {row['max_available_experiments']:.0f} available experiments
- **Success Rate**: {row['success_rate']:.1%}
"""
        
        # Find best strategy for recommendations
        successful_strategies = convergence_df_sorted[convergence_df_sorted['success_rate'] > 0]
        if not successful_strategies.empty:
            best_strategy = successful_strategies.iloc[0]
            tl_recommendation = "**WITH Transfer Learning**" if best_strategy['use_transfer_learning'] else "**WITHOUT Transfer Learning**"
            
            report += f"""

## 🎯 **Recommended Experiment Budget**

### 💡 **Primary Recommendation: {best_strategy['strategy']}**
**{tl_recommendation}**

#### 📊 **Budget Planning by Confidence Level**
- **Optimistic (50% confidence)**: {best_strategy['median_total_experiments']:.0f} total experiments
- **Realistic (90% confidence)**: {best_strategy['total_experiments_90th_percentile']:.0f} total experiments  
- **Conservative (95% confidence)**: {best_strategy['total_experiments_95th_percentile']:.0f} total experiments

#### 🔍 **Detailed Breakdown**
- **Phase 1 - Initial Sampling**: {best_strategy['num_init']} experiments
- **Phase 2 - BO Optimization**: {best_strategy['median_bo_iterations']:.0f} additional experiments (median)
- **Transfer Learning**: {'Enabled' if best_strategy['use_transfer_learning'] else 'Disabled'}
- **Total Budget**: {best_strategy['median_total_experiments']:.0f} experiments

#### ⚠️ **Risk Assessment**
- **Success Probability**: {best_strategy['success_rate']:.1%}
- **Worst Case**: Up to {best_strategy['max_total_experiments']:.0f} experiments may be needed
- **Best Case**: As few as {best_strategy['min_total_experiments']:.0f} experiments might suffice

## 📋 **Strategy Comparison Table (Including Transfer Learning)**

| Strategy | Transfer Learning | Success Rate | Median Budget | 90% Conf. | Initial | BO Iters | Total Range |
|----------|------------------|-------------|---------------|-----------|---------|----------|-------------|
"""
            
            for _, row in convergence_df_sorted.iterrows():
                tl_symbol = "✅" if row['use_transfer_learning'] else "❌"
                if not pd.isna(row['median_total_experiments']):
                    report += f"| {row['acquisition_type']} (init={row['num_init']}) | {tl_symbol} | {row['success_rate']:.1%} | {row['median_total_experiments']:.0f} exp. | {row['total_experiments_90th_percentile']:.0f} exp. | {row['num_init']} | {row['median_bo_iterations']:.0f} | {row['min_total_experiments']:.0f}-{row['max_total_experiments']:.0f} |\n"
                else:
                    report += f"| {row['acquisition_type']} (init={row['num_init']}) | {tl_symbol} | {row['success_rate']:.1%} | Failed | Failed | {row['num_init']} | - | - |\n"
            
            # Transfer Learning Impact Analysis
            report += f"""

## 🔄 **Transfer Learning Impact Analysis**

### 📈 **Comparison: With vs Without Transfer Learning**
"""
            
            # Compare TL vs non-TL for same acquisition functions
            acq_functions = convergence_df['acquisition_type'].unique()
            for acq in acq_functions:
                acq_strategies = convergence_df[convergence_df['acquisition_type'] == acq]
                tl_strategies = acq_strategies[acq_strategies['use_transfer_learning'] == True]
                no_tl_strategies = acq_strategies[acq_strategies['use_transfer_learning'] == False]
                
                if len(tl_strategies) > 0 and len(no_tl_strategies) > 0:
                    report += f"""
#### {acq} Acquisition Function
"""
                    for init_size in acq_strategies['num_init'].unique():
                        tl_subset = tl_strategies[tl_strategies['num_init'] == init_size]
                        no_tl_subset = no_tl_strategies[no_tl_strategies['num_init'] == init_size]
                        
                        if len(tl_subset) > 0 and len(no_tl_subset) > 0:
                            tl_row = tl_subset.iloc[0]
                            no_tl_row = no_tl_subset.iloc[0]
                            
                            if not pd.isna(tl_row['median_total_experiments']) and not pd.isna(no_tl_row['median_total_experiments']):
                                improvement = ((no_tl_row['median_total_experiments'] - tl_row['median_total_experiments']) / no_tl_row['median_total_experiments']) * 100
                                report += f"""
**Init={init_size} points:**
- **With TL**: {tl_row['median_total_experiments']:.0f} total experiments ({tl_row['success_rate']:.1%} success)
- **Without TL**: {no_tl_row['median_total_experiments']:.0f} total experiments ({no_tl_row['success_rate']:.1%} success)
- **TL Improvement**: {improvement:+.1f}% experiments saved
"""
            
            report += f"""

## 🚀 **Implementation Roadmap**

### Phase 1: Initial Design ({best_strategy['num_init']} experiments)
1. **Method**: Orthogonal/Latin Hypercube Sampling  
2. **Goal**: Explore parameter space broadly
3. **Budget**: {best_strategy['num_init']} experiments
4. **Timeline**: Can be run in parallel if possible

### Phase 2: BO Optimization ({best_strategy['median_bo_iterations']:.0f} experiments)
1. **Method**: {best_strategy['acquisition_type']} acquisition function
2. **Transfer Learning**: {'Enabled - Use prior knowledge/data' if best_strategy['use_transfer_learning'] else 'Disabled - Fresh start'}
3. **Goal**: Iteratively find optimum
4. **Budget**: {best_strategy['median_bo_iterations']:.0f} additional experiments (median case)
5. **Timeline**: Sequential execution required
6. **Monitoring**: Check convergence every 3-5 iterations

### Phase 3: Contingency Planning
- **If no convergence after {best_strategy['median_total_experiments']:.0f} experiments**: 
  - Continue up to {best_strategy['total_experiments_90th_percentile']:.0f} total experiments
  - Consider switching strategy or adjusting convergence criteria
  - Re-evaluate problem formulation

## 💰 **Budget Recommendations**

### 🎯 **Minimum Viable Budget**
- **{best_strategy['median_total_experiments']:.0f} total experiments** (50% confidence)
- Suitable for: Research projects, proof-of-concept

### 🏭 **Production Deployment Budget**  
- **{best_strategy['total_experiments_90th_percentile']:.0f} total experiments** (90% confidence)
- Suitable for: Critical applications, commercial deployment

### 🛡️ **Risk-Averse Budget**
- **{best_strategy['total_experiments_95th_percentile']:.0f} total experiments** (95% confidence)  
- Suitable for: High-stakes applications, regulatory environments

## ✅ **Key Takeaways**

1. **Recommended Strategy**: {best_strategy['strategy']} 
2. **Transfer Learning**: {'Recommended - significant efficiency gains' if best_strategy['use_transfer_learning'] else 'Not required for this case'}
3. **Typical Budget**: {best_strategy['median_total_experiments']:.0f} total experiments
4. **Success Rate**: {best_strategy['success_rate']:.1%} probability of convergence
5. **Planning Horizon**: {best_strategy['num_init']} initial + up to {best_strategy['max_bo_iterations']:.0f} BO iterations

*Note: These estimates are based on achieving practically relevant performance levels across multiple realistic scenarios. Transfer Learning results depend on the quality and relevance of prior data. Actual results may vary depending on problem complexity and noise levels.*
"""
        
        return report

    # Add helper methods for convergence analysis (keeping original functionality)
    def _calculate_moving_window_convergence(self, primary_metric: str, 
                                        window_size: int = 5, 
                                        tolerance_threshold: float = 0.05,
                                        tolerance_type: str = 'relative'):
        """
        Calculate convergence using moving window instead of global target.
        """
        try:
            # Store parameters for later use
            self._convergence_params = {
                'window_size': window_size,
                'tolerance_threshold': tolerance_threshold,
                'tolerance_type': tolerance_type
            }
            print(f"Moving window convergence: {window_size} window, {tolerance_threshold:.2%} {tolerance_type} threshold")
            
        except Exception as e:
            print(f"Failed to setup moving window convergence: {e}")
            self._convergence_params = None

    def create_comprehensive_strategy_report(self, metric_col: str = 'regret') -> str:
        """
        Generate comprehensive text report including surrogate quality metrics and transfer learning
        """
        try:
            recommendation = self.recommend_comprehensive_strategy(metric_col)
            comprehensive_metrics = recommendation['all_strategies']
        except Exception as e:
            return f"# ❌ Report Generation Failed\n\nError: {str(e)}\n\nPlease check your data and try again."
        
        report = f"""
# 🎯 Comprehensive BO Strategy Analysis Report (Including Transfer Learning)

## 📊 **Recommended Strategy**
**{recommendation['strategy']}**

### 🏅 **Multi-Criteria Scores**
- **Overall Combined Score**: {recommendation['combined_score']:.4f}
- **Performance Score**: {recommendation['performance_score']:.3f} ({recommendation['weights_used']['performance']:.1%} weight)
- **Robustness Score**: {recommendation['robustness_score']:.3f} ({recommendation['weights_used']['robustness']:.1%} weight)
- **Safety Score**: {recommendation['safety_score']:.3f} ({recommendation['weights_used']['safety']:.1%} weight)
- **Surrogate Quality Score**: {recommendation['surrogate_quality_score']:.3f} ({recommendation['weights_used']['surrogate_quality']:.1%} weight)
- **Convergence Score**: {recommendation['convergence_score']:.3f} ({recommendation['weights_used']['convergence']:.1%} weight)

### 📈 **Detailed Performance Metrics**
"""
        
        # Add detailed metrics from the best strategy
        best_metrics = recommendation['metrics']
        
        # Show transfer learning status
        tl_status = "✅ Transfer Learning Enabled" if best_metrics.get('use_transfer_learning', False) else "❌ No Transfer Learning"
        report += f"""
#### 🔄 **Transfer Learning Status**
**{tl_status}**
"""
        
        # Optimization performance
        if f'{metric_col}_mean' in best_metrics:
            report += f"""
#### 🎯 Optimization Performance
- **Mean {metric_col.upper()}**: {best_metrics[f'{metric_col}_mean']:.4f}
- **Std {metric_col.upper()}**: {best_metrics[f'{metric_col}_std']:.4f}
- **Best Case**: {best_metrics[f'{metric_col}_min']:.4f}
- **Worst Case (95th percentile)**: {best_metrics[f'{metric_col}_q95']:.4f}
- **Coefficient of Variation**: {best_metrics[f'{metric_col}_cv']:.4f}
"""
        
        # Surrogate quality
        surrogate_metrics = [k for k in best_metrics.keys() if any(x in k for x in ['mse', 'r2', 'rmse_h', 'crps', 'coverage'])]
        if surrogate_metrics:
            report += f"""
#### 🔍 Surrogate Model Quality
"""
            for metric in surrogate_metrics:
                if '_mean' in metric:
                    report += f"- **{metric.replace('_', ' ').title()}**: {best_metrics[metric]:.4f}\n"
        
        # Convergence
        conv_metrics = [k for k in best_metrics.keys() if 'convergence' in k]
        if conv_metrics:
            report += f"""
#### ⚡ Convergence Behavior
"""
            for metric in conv_metrics:
                if 'speed' in metric and not pd.isna(best_metrics[metric]):
                    report += f"- **{metric.replace('_', ' ').title()}**: {best_metrics[metric]:.2f} iterations\n"
                elif 'reliability' in metric:
                    report += f"- **{metric.replace('_', ' ').title()}**: {best_metrics[metric]:.1%}\n"
        
        report += f"""

## 🏆 **Top 5 Strategies (Multi-Criteria)**
"""
        
        top_5 = comprehensive_metrics.head()
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            regret_val = row.get(f'{metric_col}_mean', np.nan)
            regret_std = row.get(f'{metric_col}_std', np.nan)
            regret_str = f"{regret_val:.4f} ± {regret_std:.4f}" if not pd.isna(regret_val) else "N/A"
            tl_symbol = "✅" if row.get('use_transfer_learning', False) else "❌"
            
            report += f"""
### {i}. {row['acquisition_type']} (init={row['num_init']}) (TL={tl_symbol})
- **Combined Score**: {row['combined_score']:.4f}
- **Optimization**: {regret_str}
- **Surrogate Quality**: {row.get('surrogate_quality_score', np.nan):.3f}
- **Convergence**: {row.get('convergence_score', np.nan):.3f}
- **Transfer Learning**: {'Enabled' if row.get('use_transfer_learning', False) else 'Disabled'}
"""

        report += f"""

## 🔍 **Category Winners**

### 🎯 Best Optimization Performance
"""
        
        if f'{metric_col}_mean' in comprehensive_metrics.columns:
            best_perf_idx = comprehensive_metrics[f'{metric_col}_mean'].idxmin()  # Lower is better for regret
            best_perf = comprehensive_metrics.loc[best_perf_idx]
            tl_symbol = "✅" if best_perf.get('use_transfer_learning', False) else "❌"
            report += f"{best_perf['acquisition_type']} (init={best_perf['num_init']}) (TL={tl_symbol}) - Mean {metric_col}: {best_perf[f'{metric_col}_mean']:.4f}\n"
        
        ### Most Robust
        if 'robustness_score' in comprehensive_metrics.columns:
            best_robust_idx = comprehensive_metrics['robustness_score'].idxmax()
            best_robust = comprehensive_metrics.loc[best_robust_idx]
            tl_symbol = "✅" if best_robust.get('use_transfer_learning', False) else "❌"
            report += f"""
### 🛡️ Most Robust Strategy
{best_robust['acquisition_type']} (init={best_robust['num_init']}) (TL={tl_symbol}) - Robustness Score: {best_robust['robustness_score']:.4f}
"""
        
        ### Best Surrogate Quality
        if 'surrogate_quality_score' in comprehensive_metrics.columns:
            best_surrogate_idx = comprehensive_metrics['surrogate_quality_score'].idxmax()
            best_surrogate = comprehensive_metrics.loc[best_surrogate_idx]
            tl_symbol = "✅" if best_surrogate.get('use_transfer_learning', False) else "❌"
            report += f"""
### 🔍 Best Surrogate Quality
{best_surrogate['acquisition_type']} (init={best_surrogate['num_init']}) (TL={tl_symbol}) - Surrogate Score: {best_surrogate['surrogate_quality_score']:.4f}
"""
        
        ### Fastest Convergence
        if 'convergence_score' in comprehensive_metrics.columns:
            best_conv_idx = comprehensive_metrics['convergence_score'].idxmax()
            best_conv = comprehensive_metrics.loc[best_conv_idx]
            tl_symbol = "✅" if best_conv.get('use_transfer_learning', False) else "❌"
            report += f"""
### ⚡ Fastest Convergence
{best_conv['acquisition_type']} (init={best_conv['num_init']}) (TL={tl_symbol}) - Convergence Score: {best_conv['convergence_score']:.4f}
"""

        # Transfer Learning Analysis
        tl_strategies = comprehensive_metrics[comprehensive_metrics['use_transfer_learning'] == True]
        no_tl_strategies = comprehensive_metrics[comprehensive_metrics['use_transfer_learning'] == False]
        
        if len(tl_strategies) > 0 and len(no_tl_strategies) > 0:
            report += f"""

## 🔄 **Transfer Learning Impact Analysis**

### 📊 **Overall Comparison**
- **Strategies with Transfer Learning**: {len(tl_strategies)}
- **Strategies without Transfer Learning**: {len(no_tl_strategies)}
- **Best TL Strategy**: {tl_strategies.iloc[0]['strategy_full']} (Score: {tl_strategies.iloc[0]['combined_score']:.4f})
- **Best Non-TL Strategy**: {no_tl_strategies.iloc[0]['strategy_full']} (Score: {no_tl_strategies.iloc[0]['combined_score']:.4f})

### 📈 **Performance Impact**
"""
            if f'{metric_col}_mean' in comprehensive_metrics.columns:
                tl_mean = tl_strategies[f'{metric_col}_mean'].mean()
                no_tl_mean = no_tl_strategies[f'{metric_col}_mean'].mean()
                improvement = ((no_tl_mean - tl_mean) / no_tl_mean) * 100 if no_tl_mean != 0 else 0
                
                report += f"""
- **Average {metric_col} with TL**: {tl_mean:.4f}
- **Average {metric_col} without TL**: {no_tl_mean:.4f}
- **Transfer Learning Improvement**: {improvement:+.1f}%
"""

        report += f"""

## 💡 **Strategic Recommendations**

### 🏭 For Production Deployment
**Use: {recommendation['strategy']}**
- Balanced performance across all criteria
- {'Leverage transfer learning for efficiency gains' if best_metrics.get('use_transfer_learning', False) else 'No transfer learning required'}
- Reliable surrogate model quality
- Consistent convergence behavior

### 🎯 For Performance-Critical Applications  
**Consider:** {comprehensive_metrics.loc[comprehensive_metrics[f'{metric_col}_mean'].idxmin(), 'strategy_full']}
- Best pure optimization performance
- Accept potential robustness trade-offs

### 🛡️ For Risk-Averse Applications
**Consider:** {comprehensive_metrics.loc[comprehensive_metrics['safety_score'].idxmax(), 'strategy_full']}
- Best worst-case protection
- Most predictable outcomes

### 🔬 For Research & Exploration
**Consider:** {comprehensive_metrics.loc[comprehensive_metrics['surrogate_quality_score'].idxmax(), 'strategy_full']}
- Best model understanding
- Most reliable uncertainty estimates

### 🔄 **Transfer Learning Recommendations**
"""
        
        if len(tl_strategies) > 0:
            best_tl = tl_strategies.iloc[0]
            report += f"""
- **When to use Transfer Learning**: If you have relevant prior data/knowledge
- **Best TL Strategy**: {best_tl['strategy_full']}
- **Expected Benefit**: Faster convergence and potentially better performance
"""
        
        if len(no_tl_strategies) > 0:
            best_no_tl = no_tl_strategies.iloc[0]
            report += f"""
- **When to avoid Transfer Learning**: Limited/poor quality prior data
- **Best Non-TL Strategy**: {best_no_tl['strategy_full']}
- **Advantage**: No dependency on external data quality
"""

        report += f"""

## 📊 **Analysis Summary**

This comprehensive analysis evaluated **{len(comprehensive_metrics)} strategies** across **5 key criteria** and **Transfer Learning variants**:

1. **Optimization Performance** ({recommendation['weights_used']['performance']:.1%}): How well does the strategy find good solutions?
2. **Robustness** ({recommendation['weights_used']['robustness']:.1%}): How consistent is performance across different scenarios?
3. **Safety** ({recommendation['weights_used']['safety']:.1%}): How good is worst-case performance?
4. **Surrogate Quality** ({recommendation['weights_used']['surrogate_quality']:.1%}): How accurate and well-calibrated is the GP model?
5. **Convergence** ({recommendation['weights_used']['convergence']:.1%}): How quickly and reliably does the strategy converge?
6. **Transfer Learning**: Whether the strategy leverages prior knowledge/data

The recommended strategy represents the best balance across all criteria for robust, production-ready Bayesian Optimization.

### 🔍 **Transfer Learning Summary**
- **Total strategies analyzed**: {len(comprehensive_metrics)}
- **With Transfer Learning**: {len(tl_strategies)} strategies
- **Without Transfer Learning**: {len(no_tl_strategies)} strategies
- **Transfer Learning appears**: {'beneficial' if len(tl_strategies) > 0 and tl_strategies.iloc[0]['combined_score'] > (no_tl_strategies.iloc[0]['combined_score'] if len(no_tl_strategies) > 0 else 0) else 'neutral or negative'} for this problem
"""
        
        return report

    def create_strategy_report(self, metric_col: str = 'regret') -> str:
        """
        Generate comprehensive text report - now includes transfer learning
        """
        recommendation = self.recommend_strategy(metric_col)
        robustness_metrics = recommendation['all_strategies']
        
        # Count TL vs non-TL strategies
        tl_count = robustness_metrics['use_transfer_learning'].sum()
        total_count = len(robustness_metrics)
        
        report = f"""
# 🎯 BO Strategy Robustness Report (Including Transfer Learning)

## 📊 **Recommended Strategy**
**{recommendation['strategy']}**
- **Combined Score**: {recommendation['combined_score']:.3f}
- **Performance Score**: {recommendation['performance_score']:.3f}
- **Robustness Score**: {recommendation['robustness_score']:.3f}
- **Safety Score**: {recommendation['safety_score']:.3f}

## 📈 **Key Metrics**
- **Mean {metric_col.upper()}**: {recommendation['metrics']['mean']:.4f}
- **Std {metric_col.upper()}**: {recommendation['metrics']['std']:.4f}
- **Best Case**: {recommendation['metrics']['best_case']:.4f}
- **Worst Case (95th percentile)**: {recommendation['metrics']['q95']:.4f}
- **Success Rate**: {recommendation['metrics']['success_rate']:.1%}
- **Transfer Learning**: {'Enabled' if recommendation['metrics']['use_transfer_learning'] else 'Disabled'}

## 🏆 **Top 5 Strategies**
"""
        
        top_5 = robustness_metrics.head()
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            tl_symbol = "✅" if row['use_transfer_learning'] else "❌"
            report += f"""
### {i}. {row['acquisition_type']} (init={row['num_init']}) (TL={tl_symbol})
- Combined Score: {row['combined_score']:.3f}
- Mean {metric_col}: {row['mean']:.4f} ± {row['std']:.4f}
- Success Rate: {row['success_rate']:.1%}
- Transfer Learning: {'Enabled' if row['use_transfer_learning'] else 'Disabled'}
"""

        report += f"""
## 🔍 **Robustness Insights**

### Most Robust Strategy
{robustness_metrics.loc[robustness_metrics['robustness_score'].idxmax(), 'strategy_full']}
- Lowest variability across ground truth variants

### Best Performance Strategy  
{robustness_metrics.loc[robustness_metrics['performance_score'].idxmax(), 'strategy_full']}
- Best average performance

### Safest Strategy
{robustness_metrics.loc[robustness_metrics['q95'].idxmin(), 'strategy_full']}
- Best worst-case performance

## 🔄 **Transfer Learning Analysis**

### Overview
- **Total Strategies**: {total_count}
- **With Transfer Learning**: {tl_count}
- **Without Transfer Learning**: {total_count - tl_count}

### Best Transfer Learning Strategy
{robustness_metrics[robustness_metrics['use_transfer_learning'] == True].iloc[0]['strategy_full'] if tl_count > 0 else 'None available'}

### Best Non-Transfer Learning Strategy  
{robustness_metrics[robustness_metrics['use_transfer_learning'] == False].iloc[0]['strategy_full'] if (total_count - tl_count) > 0 else 'None available'}

## 💡 **Recommendations**

1. **For Production Use**: Choose the recommended strategy for balanced performance-robustness
2. **For Risk-Averse Applications**: Consider the safest strategy
3. **For High-Performance Requirements**: Consider the best performance strategy if robustness is acceptable
4. **Transfer Learning Decision**: 
   - **Use TL** if you have relevant prior data and TL strategies show better performance
   - **Skip TL** if prior data quality is questionable or non-TL strategies perform comparably
5. **Further Testing**: Consider strategies with high potential but limited data
"""
        
        return report


# Updated moving window convergence functions to handle transfer learning
def detect_moving_window_convergence(
    values: np.ndarray,
    window_size: int = 2,
    tolerance_type: str = 'relative',
    tolerance_threshold: float = 0.50,
    min_iterations: int = 10,
    require_monotonic: bool = False
) -> Dict:
    """
    Detect convergence using moving window analysis of regret values.
    (Function remains the same as it works on individual value arrays)
    """
    # Input validation
    if len(values) < max(window_size, min_iterations):
        return {
            'converged': False,
            'convergence_iteration': None,
            'convergence_value': None,
            'final_value': values[-1] if len(values) > 0 else None,
            'total_iterations': len(values),
            'convergence_window': None,
            'convergence_metric': None,
            'reason': f'Insufficient iterations (need ≥{max(window_size, min_iterations)}, got {len(values)})'
        }
    
    # Convert to numpy array and handle NaN values
    values = np.array(values)
    if np.any(np.isnan(values)):
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < max(window_size, min_iterations):
            return {
                'converged': False,
                'convergence_iteration': None,
                'convergence_value': None,
                'final_value': values[-1] if len(values) > 0 else None,
                'total_iterations': len(values),
                'convergence_window': None,
                'convergence_metric': None,
                'reason': 'Too many NaN values'
            }
        values = values[valid_mask]
    
    # Start checking from min_iterations
    for i in range(min_iterations, len(values) - window_size + 1):
        window = values[i:i + window_size]
        
        # Calculate convergence metric based on tolerance type
        if tolerance_type == 'relative':
            if window[0] != 0:
                convergence_metric = abs(window[-1] - window[0]) / abs(window[0])
            else:
                convergence_metric = abs(window[-1] - window[0])
        elif tolerance_type == 'absolute':
            convergence_metric = abs(window[-1] - window[0])
        elif tolerance_type == 'coefficient_of_variation':
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_mean != 0:
                convergence_metric = window_std / abs(window_mean)
            else:
                convergence_metric = window_std
        else:
            raise ValueError(f"Unknown tolerance_type: {tolerance_type}")
        
        # Check monotonic requirement if specified
        if require_monotonic:
            if not np.all(np.diff(window) <= 0):
                continue
        
        # Check if converged
        if convergence_metric <= tolerance_threshold:
            return {
                'converged': True,
                'convergence_iteration': i + window_size - 1,
                'convergence_value': values[i + window_size - 1],
                'final_value': values[-1],
                'total_iterations': len(values),
                'convergence_window': window.tolist(),
                'convergence_metric': convergence_metric,
                'convergence_type': tolerance_type,
                'tolerance_threshold': tolerance_threshold,
                'window_size': window_size,
                'reason': f'Converged: {tolerance_type} change {convergence_metric:.4f} ≤ {tolerance_threshold:.4f}'
            }
    
    # Did not converge
    return {
        'converged': False,
        'convergence_iteration': None,
        'convergence_value': None,
        'final_value': values[-1],
        'total_iterations': len(values),
        'convergence_window': None,
        'convergence_metric': None,
        'convergence_type': tolerance_type,
        'tolerance_threshold': tolerance_threshold,
        'window_size': window_size,
        'reason': f'No convergence detected with {tolerance_type} threshold {tolerance_threshold:.4f}'
    }


# Updated Streamlit UI to handle transfer learning
def create_robustness_analysis_ui(df_results=None):
    """
    Create Streamlit UI for robustness analysis with Transfer Learning support
    """
    st.title("🎯 BO Strategy Robustness Analysis (Including Transfer Learning)")
    
    # Determine data source
    if df_results is not None:
        data_source = "uploaded"
        st.info(f"📁 **Analyzing uploaded data** ({len(df_results)} rows)")
    else:
        if 'bo_results' not in st.session_state or st.session_state['bo_results'] is None:
            st.warning("⚠️ No BO simulation results found. Please run the simulation first or upload a file.")
            return
        
        df_results = st.session_state['bo_results']
        data_source = "session"
        st.info(f"🎯 **Analyzing session data** ({len(df_results)} rows)")
    
    try:
        analyzer = RobustnessAnalyzer(df_results)
        
        # Show transfer learning info in sidebar
        st.sidebar.header("📊 Data Overview")
        total_strategies = df_results.groupby(['acquisition_type', 'num_init', 'use_transfer_learning']).size()
        tl_strategies = df_results[df_results['use_transfer_learning'] == True].groupby(['acquisition_type', 'num_init']).size()
        no_tl_strategies = df_results[df_results['use_transfer_learning'] == False].groupby(['acquisition_type', 'num_init']).size()
        
        st.sidebar.metric("Total Strategy Variants", len(total_strategies))
        st.sidebar.metric("With Transfer Learning", len(tl_strategies))
        st.sidebar.metric("Without Transfer Learning", len(no_tl_strategies))
        
        # Sidebar controls
        st.sidebar.header("🔧 Analysis Configuration")
        
        available_metrics = [col for col in df_results.columns 
                           if col not in ['acquisition_type', 'num_init', 'seed', 'iteration', 'use_transfer_learning'] 
                           and df_results[col].dtype in ['float64', 'int64']]
        
        if not available_metrics:
            st.error("❌ No numeric metrics found for analysis")
            return
        
        selected_metric = st.sidebar.selectbox(
            "Primary Optimization Metric",
            options=available_metrics,
            index=0 if 'regret' not in available_metrics else available_metrics.index('regret'),
            help="Main metric for optimization performance (e.g., regret, loss)"
        )
        
        # Analysis mode selection
        analysis_mode = st.sidebar.radio(
            "Analysis Mode",
            ["🎯 Comprehensive (Multi-Metric)", "📊 Simple (Regret-Only)"],
            help="Comprehensive mode includes surrogate quality, convergence, etc."
        )
        
        # Transfer Learning filter
        tl_filter = st.sidebar.selectbox(
            "Transfer Learning Filter",
            ["All Strategies", "Only Transfer Learning", "Only Non-Transfer Learning"],
            help="Filter strategies by transfer learning usage"
        )
        
        # Apply TL filter
        if tl_filter == "Only Transfer Learning":
            filtered_df = df_results[df_results['use_transfer_learning'] == True]
        elif tl_filter == "Only Non-Transfer Learning":
            filtered_df = df_results[df_results['use_transfer_learning'] == False]
        else:
            filtered_df = df_results
        
        if len(filtered_df) == 0:
            st.error(f"❌ No data found for filter: {tl_filter}")
            return
        
        # Update analyzer with filtered data if needed
        if tl_filter != "All Strategies":
            analyzer = RobustnessAnalyzer(filtered_df)
        
        # Weights configuration
        if analysis_mode.startswith("🎯"):
            st.sidebar.subheader("⚖️ Multi-Criteria Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.25, 0.05, 
                                           help="How important is average optimization performance?")
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.25, 0.05,
                                             help="How important is consistency across ground truths?")
            safety_weight = st.sidebar.slider("Safety Weight", 0.0, 1.0, 0.15, 0.05,
                                             help="How important is worst-case performance?")
            surrogate_weight = st.sidebar.slider("Surrogate Quality Weight", 0.0, 1.0, 0.20, 0.05,
                                                help="How important is GP model accuracy?")
            convergence_weight = st.sidebar.slider("Convergence Weight", 0.0, 1.0, 0.15, 0.05,
                                                  help="How important is fast convergence?")
            
            # Normalize weights
            total_weight = perf_weight + robust_weight + safety_weight + surrogate_weight + convergence_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight  
                safety_weight /= total_weight
                surrogate_weight /= total_weight
                convergence_weight /= total_weight
        else:
            # Simple mode weights
            st.sidebar.subheader("⚖️ Simple Mode Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.4, 0.1)
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.4, 0.1)
            safety_weight = st.sidebar.slider("Safety Weight", 0.0, 1.0, 0.2, 0.1)
            surrogate_weight = 0.0
            convergence_weight = 0.0
            
            # Normalize weights
            total_weight = perf_weight + robust_weight + safety_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight  
                safety_weight /= total_weight
        
        # Main analysis
        st.header("📊 Strategy Recommendation")
        
        try:
            if analysis_mode.startswith("🎯"):
                # Comprehensive analysis
                recommendation = analyzer.recommend_comprehensive_strategy(
                    selected_metric, perf_weight, robust_weight, safety_weight,
                    surrogate_weight, convergence_weight
                )
                
                # Display comprehensive recommendation
                col1, col2, col3, col4, col5 = st.columns(5)
                
                strategy_parts = recommendation['strategy'].split(' (')
                acq_func = strategy_parts[0]
                details = ' ('.join(strategy_parts[1:]) if len(strategy_parts) > 1 else ""
                
                with col1:
                    st.metric("🏆 Strategy", acq_func, details)
                with col2:
                    st.metric("🎯 Performance", f"{recommendation['performance_score']:.3f}")
                with col3:
                    st.metric("🛡️ Robustness", f"{recommendation['robustness_score']:.3f}")
                with col4:
                    st.metric("🔍 Surrogate Quality", f"{recommendation['surrogate_quality_score']:.3f}")
                with col5:
                    st.metric("⚡ Convergence", f"{recommendation['convergence_score']:.3f}")
                
                # Overall score and TL status
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏅 Overall Combined Score", f"{recommendation['combined_score']:.4f}")
                with col2:
                    tl_status = recommendation['metrics'].get('use_transfer_learning', False)
                    st.metric("🔄 Transfer Learning", "✅ Enabled" if tl_status else "❌ Disabled")
                
                # Weight visualization
                with st.expander("⚖️ Weights Used"):
                    weights_df = pd.DataFrame(list(recommendation['weights_used'].items()), 
                                            columns=['Criterion', 'Weight'])
                    fig_weights = px.pie(weights_df, values='Weight', names='Criterion', 
                                       title="Criteria Weights")
                    st.plotly_chart(fig_weights, use_container_width=True)
                
            else:
                # Simple analysis
                recommendation = analyzer.recommend_strategy(
                    selected_metric, perf_weight, robust_weight, safety_weight
                )
                
                # Display simple recommendation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Recommended Strategy",
                        recommendation['strategy'],
                        f"Score: {recommendation['combined_score']:.3f}"
                    )
                
                with col2:
                    mean_val = recommendation['metrics'].get('mean', 'N/A')
                    std_val = recommendation['metrics'].get('std', 'N/A')
                    if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                        st.metric(
                            f"📈 Mean {selected_metric.upper()}",
                            f"{mean_val:.4f}",
                            f"±{std_val:.4f}"
                        )
                    else:
                        st.metric(f"📈 Mean {selected_metric.upper()}", str(mean_val))
                
                with col3:
                    tl_status = recommendation['metrics'].get('use_transfer_learning', False)
                    success_rate = recommendation['metrics'].get('success_rate', 'N/A')
                    st.metric(
                        "🔄 Transfer Learning",
                        "✅ Enabled" if tl_status else "❌ Disabled",
                        f"Success: {success_rate:.1%}" if isinstance(success_rate, (int, float)) else ""
                    )
                        
        except Exception as e:
            st.error(f"Strategy recommendation failed: {str(e)}")
            st.exception(e)
        
        # Visualizations and planning tabs
        viz_tab, planning_tab = st.tabs(["📈 Visualizations", "🧪 Experiment Planning"])
        
        with viz_tab:
            # Convergence plot
            st.subheader("🔄 Convergence Robustness (Including Transfer Learning)")
            try:
                conv_fig = analyzer.plot_convergence_robustness(selected_metric)
                st.plotly_chart(conv_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Convergence plot failed: {str(e)}")
            
            # Comparison plot
            st.subheader("🔍 Strategy Comparison (Including Transfer Learning)")
            try:
                comp_fig = analyzer.plot_robustness_comparison(selected_metric)
                st.plotly_chart(comp_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Comparison plot failed: {str(e)}")
        
        with planning_tab:
            st.subheader("🧪 Practical Experiment Planning (Including Transfer Learning)")
            
            try:
                # Get experiment planning analysis
                convergence_df = analyzer.get_experiment_planning_analysis(selected_metric)
                
                if not convergence_df.empty:
                    # Key insights at the top
                    st.subheader("⚡ Practical BO Planning Insights")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Best BO strategy for minimum total experiments
                    converged_strategies = convergence_df[convergence_df['success_rate'] > 0]
                    if not converged_strategies.empty:
                        best_strategy = converged_strategies.loc[converged_strategies['median_total_experiments'].idxmin()]
                        
                        with col1:
                            tl_icon = "✅" if best_strategy['use_transfer_learning'] else "❌"
                            st.metric(
                                "🏆 Most Efficient BO Strategy",
                                f"{best_strategy['acquisition_type']} (init={best_strategy['num_init']})",
                                f"TL: {tl_icon} | {best_strategy['median_total_experiments']:.0f} total exp."
                            )
                        
                        with col2:
                            st.metric(
                                "⚡ Best Case Total",
                                f"{best_strategy['min_total_experiments']:.0f} experiments",
                                f"Success: {best_strategy['success_rate']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "🎯 Realistic Budget",
                                f"{best_strategy['median_total_experiments']:.0f} experiments",
                                f"{best_strategy['num_init']} init + {best_strategy['median_bo_iterations']:.0f} BO"
                            )
                        
                        with col4:
                            st.metric(
                                "🛡️ Conservative Budget", 
                                f"{best_strategy['total_experiments_90th_percentile']:.0f} experiments",
                                "90% confidence"
                            )
                    
                    # Detailed experiment planning table
                    st.subheader("📊 Detailed BO Experiment Planning (Transfer Learning Included)")
                    
                    # Format the dataframe for display
                    display_conv_df = convergence_df.copy()
                    
                    # Round numeric columns
                    numeric_cols = ['median_total_experiments', 'mean_total_experiments', 
                                   'total_experiments_90th_percentile', 'total_experiments_95th_percentile',
                                   'min_total_experiments', 'max_total_experiments',
                                   'median_bo_iterations']
                    
                    for col in numeric_cols:
                        if col in display_conv_df.columns:
                            display_conv_df[col] = display_conv_df[col].round(1)
                    
                    display_conv_df['success_rate'] = (display_conv_df['success_rate'] * 100).round(1)
                    display_conv_df['tl_status'] = display_conv_df['use_transfer_learning'].apply(lambda x: "✅" if x else "❌")
                    
                    # Select key columns for display
                    key_cols = ['strategy', 'tl_status', 'success_rate', 'median_total_experiments', 
                               'total_experiments_90th_percentile', 'num_init', 'median_bo_iterations',
                               'min_total_experiments', 'max_total_experiments']
                    
                    display_cols = [col for col in key_cols if col in display_conv_df.columns]
                    
                    st.dataframe(
                        display_conv_df[display_cols].rename(columns={
                            'strategy': 'BO Strategy',
                            'tl_status': 'Transfer Learning',
                            'success_rate': 'Success Rate (%)',
                            'median_total_experiments': 'Median Total Exp.',
                            'total_experiments_90th_percentile': '90% Conf. Total Exp.',
                            'num_init': 'Initial Exp.',
                            'median_bo_iterations': 'Median BO Iters',
                            'min_total_experiments': 'Min Total',
                            'max_total_experiments': 'Max Total'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Transfer Learning Impact Visualization
                    st.subheader("🔄 Transfer Learning Impact on Experiment Budget")
                    
                    # Compare TL vs non-TL strategies
                    tl_comparison_data = []
                    
                    # Group by acquisition function and initial points
                    for acq_type in convergence_df['acquisition_type'].unique():
                        for init_size in convergence_df['num_init'].unique():
                            tl_subset = convergence_df[
                                (convergence_df['acquisition_type'] == acq_type) &
                                (convergence_df['num_init'] == init_size) &
                                (convergence_df['use_transfer_learning'] == True)
                            ]
                            no_tl_subset = convergence_df[
                                (convergence_df['acquisition_type'] == acq_type) &
                                (convergence_df['num_init'] == init_size) &
                                (convergence_df['use_transfer_learning'] == False)
                            ]
                            
                            if len(tl_subset) > 0 and len(no_tl_subset) > 0:
                                tl_row = tl_subset.iloc[0]
                                no_tl_row = no_tl_subset.iloc[0]
                                
                                if not pd.isna(tl_row['median_total_experiments']) and not pd.isna(no_tl_row['median_total_experiments']):
                                    tl_comparison_data.append({
                                        'Strategy': f"{acq_type} (init={init_size})",
                                        'With TL': tl_row['median_total_experiments'],
                                        'Without TL': no_tl_row['median_total_experiments'],
                                        'TL Improvement': no_tl_row['median_total_experiments'] - tl_row['median_total_experiments']
                                    })
                    
                    if tl_comparison_data:
                        tl_comp_df = pd.DataFrame(tl_comparison_data)
                        
                        fig_tl_comp = go.Figure()
                        
                        fig_tl_comp.add_trace(go.Bar(
                            name='Without Transfer Learning',
                            x=tl_comp_df['Strategy'],
                            y=tl_comp_df['Without TL'],
                            marker_color='orange'
                        ))
                        
                        fig_tl_comp.add_trace(go.Bar(
                            name='With Transfer Learning',
                            x=tl_comp_df['Strategy'],
                            y=tl_comp_df['With TL'],
                            marker_color='green'
                        ))
                        
                        fig_tl_comp.update_layout(
                            title="Transfer Learning Impact on Total Experiments Needed",
                            yaxis_title="Total Experiments (Initial + BO)",
                            barmode='group',
                            height=500
                        )
                        
                        st.plotly_chart(fig_tl_comp, use_container_width=True)
                        
                        # Show improvement table
                        st.subheader("📊 Transfer Learning Efficiency Gains")
                        tl_comp_df['TL Improvement %'] = (tl_comp_df['TL Improvement'] / tl_comp_df['Without TL'] * 100).round(1)
                        st.dataframe(tl_comp_df, use_container_width=True)
                    
                    # Generate practical planning report
                    st.subheader("📋 Practical Experiment Planning Report")
                    
                    try:
                        planning_report = analyzer.create_practical_experiment_planning_report(selected_metric)
                        st.markdown(planning_report)
                        
                        # Download button for planning report
                        st.download_button(
                            "📥 Download Practical Planning Report",
                            planning_report,
                            f"practical_experiment_planning_{selected_metric}_with_TL.md",
                            "text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Planning report generation failed: {str(e)}")
                
                else:
                    st.warning("No BO strategies available for practical experiment planning.")
                    st.write("**Possible reasons:**")
                    st.write("- Only Random strategies in data (not suitable for BO planning)")
                    st.write("- No strategies converged to target performance")
                    st.write("- Missing required data columns")
            
            except Exception as e:
                st.error(f"Practical experiment planning analysis failed: {str(e)}")
                st.exception(e)
        
        # Detailed metrics table
        st.header("📋 Detailed Metrics (Including Transfer Learning)")
        
        robustness_metrics = analyzer.compute_robustness_metrics(selected_metric)
        
        # Add combined_score if not present
        if 'combined_score' not in robustness_metrics.columns:
            # Normalize scores (0-1 scale) with safety checks
            max_perf = robustness_metrics['performance_score'].max()
            max_robust = robustness_metrics['robustness_score'].max()
            min_q95 = robustness_metrics['q95'].min()
            
            if max_perf > 0:
                perf_scores = robustness_metrics['performance_score'] / max_perf
            else:
                perf_scores = robustness_metrics['performance_score'] * 0 + 0.5
                
            if max_robust > 0:
                robust_scores = robustness_metrics['robustness_score'] / max_robust
            else:
                robust_scores = robustness_metrics['robustness_score'] * 0 + 0.5
                
            if min_q95 > 0 and not np.isnan(min_q95):
                max_q95 = robustness_metrics['q95'].max()
                if max_q95 > min_q95:
                    safety_scores = 1 - (robustness_metrics['q95'] - min_q95) / (max_q95 - min_q95)
                else:
                    safety_scores = robustness_metrics['q95'] * 0 + 1.0
            else:
                safety_scores = robustness_metrics['q95'] * 0 + 0.5
            
            # Combined score with weights from UI
            robustness_metrics['combined_score'] = (
                perf_weight * perf_scores + 
                robust_weight * robust_scores + 
                safety_weight * safety_scores
            )
        
        robustness_metrics = robustness_metrics.sort_values('combined_score', ascending=False)
        
        # Format for display - move strategy_full to front and add TL indicator
        display_metrics = robustness_metrics.copy()
        display_metrics = display_metrics.round(4)
        
        # Reorder columns to show strategy info first
        cols = ['strategy_full', 'acquisition_type', 'num_init', 'use_transfer_learning'] + \
               [col for col in display_metrics.columns if col not in ['strategy_full', 'acquisition_type', 'num_init', 'use_transfer_learning']]
        display_metrics = display_metrics[cols]
        
        st.dataframe(
            display_metrics,
            use_container_width=True,
            height=400
        )
        
        # Transfer Learning Summary
        st.header("🔄 Transfer Learning Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tl_strategies = robustness_metrics[robustness_metrics['use_transfer_learning'] == True]
            st.metric("Strategies with TL", len(tl_strategies))
            
        with col2:
            no_tl_strategies = robustness_metrics[robustness_metrics['use_transfer_learning'] == False]
            st.metric("Strategies without TL", len(no_tl_strategies))
            
        with col3:
            if len(tl_strategies) > 0 and len(no_tl_strategies) > 0:
                best_tl_score = tl_strategies['combined_score'].max()
                best_no_tl_score = no_tl_strategies['combined_score'].max()
                tl_advantage = ((best_tl_score - best_no_tl_score) / best_no_tl_score * 100) if best_no_tl_score > 0 else 0
                st.metric("TL Advantage", f"{tl_advantage:+.1f}%")
        
        # Best performers in each category
        if len(tl_strategies) > 0 and len(no_tl_strategies) > 0:
            st.subheader("🏆 Category Leaders")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best with Transfer Learning:**")
                best_tl = tl_strategies.iloc[0]
                st.write(f"- {best_tl['strategy_full']}")
                st.write(f"- Combined Score: {best_tl['combined_score']:.4f}")
                st.write(f"- Mean {selected_metric}: {best_tl['mean']:.4f}")
                
            with col2:
                st.write("**Best without Transfer Learning:**")
                best_no_tl = no_tl_strategies.iloc[0]
                st.write(f"- {best_no_tl['strategy_full']}")
                st.write(f"- Combined Score: {best_no_tl['combined_score']:.4f}")
                st.write(f"- Mean {selected_metric}: {best_no_tl['mean']:.4f}")
        
        # Text report
        st.header("📄 Analysis Report (Including Transfer Learning)")
        
        try:
            if analysis_mode.startswith("🎯"):
                report = analyzer.create_comprehensive_strategy_report(selected_metric)
            else:
                report = analyzer.create_strategy_report(selected_metric)
            st.markdown(report)
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            st.write("Basic information available in the metrics table above.")
        
        # Download options
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download metrics CSV
            csv_data = robustness_metrics.to_csv(index=False)
            st.download_button(
                "📥 Download Metrics CSV",
                csv_data,
                f"robustness_metrics_{selected_metric}_with_TL.csv",
                "text/csv"
            )
        
        with col2:
            # Download report
            st.download_button(
                "📄 Download Report",
                report,
                f"robustness_report_{selected_metric}_with_TL.md",
                "text/markdown"
            )
            
        with col3:
            # Download experiment planning data
            if 'convergence_df' in locals() and not convergence_df.empty:
                planning_csv = convergence_df.to_csv(index=False)
                st.download_button(
                    "🧪 Download Planning Data",
                    planning_csv,
                    f"experiment_planning_{selected_metric}_with_TL.csv",
                    "text/csv"
                )
        
        # Store results (only if using session data, not uploaded data)
        if data_source == "session":
            st.session_state['robustness_analyzer'] = analyzer
            st.session_state['robustness_recommendation'] = recommendation
        
        # Success message
        st.success("✅ Robustness analysis completed successfully! Transfer Learning variants have been properly analyzed and compared.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)


# Example usage for testing
if __name__ == "__main__":
    # This would be called from your main Streamlit app
    create_robustness_analysis_ui()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

def detect_moving_window_convergence(
    values: np.ndarray,
    window_size: int = 2,
    tolerance_type: str = 'relative',
    tolerance_threshold: float = 0.50,
    min_iterations: int = 10,
    require_monotonic: bool = False
) -> Dict:
    """
    Detect convergence using moving window analysis of regret values.
    
    Parameters:
    -----------
    values : np.ndarray
        Array of regret values (or other metric) over iterations
    window_size : int, default=5
        Size of the moving window to analyze
    tolerance_type : str, default='relative'
        Type of tolerance: 'relative', 'absolute', or 'coefficient_of_variation'
    tolerance_threshold : float, default=0.02
        Threshold for convergence detection:
        - For 'relative': maximum relative change in window (e.g., 0.02 = 2%)
        - For 'absolute': maximum absolute change in window
        - For 'coefficient_of_variation': maximum CV in window (std/mean)
    min_iterations : int, default=10
        Minimum iterations before convergence can be detected
    require_monotonic : bool, default=False
        Whether to require monotonic improvement in the window
    
    Returns:
    --------
    Dict with convergence information
    """
    print(f"{len(values)} values")
    # Input validation
    if len(values) < max(window_size, min_iterations):
        return {
            'converged': False,
            'convergence_iteration': None,
            'convergence_value': None,
            'final_value': values[-1] if len(values) > 0 else None,
            'total_iterations': len(values),
            'convergence_window': None,
            'convergence_metric': None,
            'reason': f'Insufficient iterations (need ≥{max(window_size, min_iterations)}, got {len(values)})'
        }
    
    # Convert to numpy array and handle NaN values
    values = np.array(values)
    if np.any(np.isnan(values)):
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < max(window_size, min_iterations):
            return {
                'converged': False,
                'convergence_iteration': None,
                'convergence_value': None,
                'final_value': values[-1] if len(values) > 0 else None,
                'total_iterations': len(values),
                'convergence_window': None,
                'convergence_metric': None,
                'reason': 'Too many NaN values'
            }
        values = values[valid_mask]
    
    # Start checking from min_iterations
    for i in range(min_iterations, len(values) - window_size + 1):
        window = values[i:i + window_size]
        
        # Calculate convergence metric based on tolerance type
        if tolerance_type == 'relative':
            if window[0] != 0:
                convergence_metric = abs(window[-1] - window[0]) / abs(window[0])
            else:
                # Handle zero initial value
                convergence_metric = abs(window[-1] - window[0])
        
        elif tolerance_type == 'absolute':
            convergence_metric = abs(window[-1] - window[0])
        
        elif tolerance_type == 'coefficient_of_variation':
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_mean != 0:
                convergence_metric = window_std / abs(window_mean)
            else:
                convergence_metric = window_std
        
        else:
            raise ValueError(f"Unknown tolerance_type: {tolerance_type}")
        
        # Check monotonic requirement if specified
        if require_monotonic:
            # For regret: should be decreasing (improving)
            if not np.all(np.diff(window) <= 0):
                continue
        
        # Check if converged
        if convergence_metric <= tolerance_threshold:
            return {
                'converged': True,
                'convergence_iteration': i + window_size - 1,  # End of convergence window
                'convergence_value': values[i + window_size - 1],
                'final_value': values[-1],
                'total_iterations': len(values),
                'convergence_window': window.tolist(),
                'convergence_metric': convergence_metric,
                'convergence_type': tolerance_type,
                'tolerance_threshold': tolerance_threshold,
                'window_size': window_size,
                'reason': f'Converged: {tolerance_type} change {convergence_metric:.4f} ≤ {tolerance_threshold:.4f}'
            }
    print(f"Debug: values = {values}")
    print(f"Debug: window_size = {window_size}")
    print(f"Debug: tolerance = {tolerance_threshold}")
    print(f"Debug: final convergence_metric = {convergence_metric}")
    
    # Did not converge
    return {
        'converged': False,
        'convergence_iteration': None,
        'convergence_value': None,
        'final_value': values[-1],
        'total_iterations': len(values),
        'convergence_window': None,
        'convergence_metric': None,
        'convergence_type': tolerance_type,
        'tolerance_threshold': tolerance_threshold,
        'window_size': window_size,
        'reason': f'No convergence detected with {tolerance_type} threshold {tolerance_threshold:.4f}'
    }

def analyze_convergence_for_bo_results(
    df_results: pd.DataFrame,
    primary_metric: str = 'regret',
    window_size: int = 2,
    tolerance_type: str = 'relative',
    tolerance_threshold: float = 0.10,
    min_iterations: int = 10,
    require_monotonic: bool = False
) -> pd.DataFrame:
    """
    Analyze convergence for all BO runs in the results DataFrame.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        BO simulation results with columns: acquisition_type, num_init, seed, iteration, [primary_metric]
    primary_metric : str
        Column name for the metric to analyze (e.g., 'regret')
    window_size : int
        Size of moving window for convergence detection
    tolerance_type : str
        Type of tolerance: 'relative', 'absolute', or 'coefficient_of_variation'
    tolerance_threshold : float
        Convergence threshold
    min_iterations : int
        Minimum iterations before convergence can be detected
    require_monotonic : bool
        Whether to require monotonic improvement
    
    Returns:
    --------
    pd.DataFrame with convergence analysis for each strategy
    """
    
    if primary_metric not in df_results.columns:
        raise ValueError(f"Metric '{primary_metric}' not found in data")
    
    convergence_results = []
    
    # Group by strategy and seed
    for (acq_type, num_init, seed), group in df_results.groupby(['acquisition_type', 'num_init', 'seed']):
        # Sort by iteration
        group_sorted = group.sort_values('iteration')
        values = group_sorted[primary_metric].values
        
        # Detect convergence
        conv_result = detect_moving_window_convergence(
            values=values,
            window_size=window_size,
            tolerance_type=tolerance_type,
            tolerance_threshold=tolerance_threshold,
            min_iterations=min_iterations,
            require_monotonic=require_monotonic
        )
        
        # Add strategy information
        conv_result.update({
            'acquisition_type': acq_type,
            'num_init': num_init,
            'seed': seed,
            'strategy': f"{acq_type} (init={num_init})"
        })
        
        convergence_results.append(conv_result)
    
    return pd.DataFrame(convergence_results)

def compute_convergence_robustness_metrics(
    convergence_df: pd.DataFrame,
    include_total_experiments: bool = True
) -> pd.DataFrame:
    """
    Compute robustness metrics based on moving window convergence analysis.
    
    Parameters:
    -----------
    convergence_df : pd.DataFrame
        Output from analyze_convergence_for_bo_results()
    include_total_experiments : bool
        Whether to compute total experiments (initial + BO iterations)
    
    Returns:
    --------
    pd.DataFrame with convergence robustness metrics by strategy
    """
    
    results = []
    
    for (acq_type, num_init, transfer), group in convergence_df.groupby(['acquisition_type', 'num_init','use_transfer_learning']):
        if transfer is None: 
            transfer = False
        else:
            transfer = True
        strategy_name = f"{acq_type} (init={num_init}) (transfer={transfer})"
        
        # Basic statistics
        total_runs = len(group)
        converged_runs = group['converged'].sum()
        convergence_rate = converged_runs / total_runs if total_runs > 0 else 0
        
        # Convergence iteration statistics (only for converged runs)
        converged_group = group[group['converged']]
        
        if len(converged_group) > 0:
            conv_iterations = converged_group['convergence_iteration'].values
            
            # BO iterations until convergence
            conv_stats = {
                'convergence_rate': convergence_rate,
                'successful_runs': converged_runs,
                'failed_runs': total_runs - converged_runs,
                'total_runs': total_runs,
                
                # Convergence speed (BO iterations)
                'mean_convergence_iteration': np.mean(conv_iterations),
                'median_convergence_iteration': np.median(conv_iterations),
                'std_convergence_iteration': np.std(conv_iterations),
                'min_convergence_iteration': np.min(conv_iterations),
                'max_convergence_iteration': np.max(conv_iterations),
                'q25_convergence_iteration': np.percentile(conv_iterations, 25),
                'q75_convergence_iteration': np.percentile(conv_iterations, 75),
                'q90_convergence_iteration': np.percentile(conv_iterations, 90),
                'q95_convergence_iteration': np.percentile(conv_iterations, 95),
            }
            
            # Total experiments if requested
            if include_total_experiments:
                total_experiments = num_init + conv_iterations
                conv_stats.update({
                    'mean_total_experiments': np.mean(total_experiments),
                    'median_total_experiments': np.median(total_experiments),
                    'std_total_experiments': np.std(total_experiments),
                    'min_total_experiments': np.min(total_experiments),
                    'max_total_experiments': np.max(total_experiments),
                    'q25_total_experiments': np.percentile(total_experiments, 25),
                    'q75_total_experiments': np.percentile(total_experiments, 75),
                    'q90_total_experiments': np.percentile(total_experiments, 90),
                    'q95_total_experiments': np.percentile(total_experiments, 95),
                })
            
            # Performance at convergence
            conv_values = converged_group['convergence_value'].values
            conv_stats.update({
                'mean_convergence_value': np.mean(conv_values),
                'median_convergence_value': np.median(conv_values),
                'std_convergence_value': np.std(conv_values),
                'best_convergence_value': np.min(conv_values),  # Lower is better for regret
                'worst_convergence_value': np.max(conv_values),
            })
            
            # Final performance (all runs, including non-converged)
            final_values = group['final_value'].values
            conv_stats.update({
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'best_final_value': np.min(final_values),
                'worst_final_value': np.max(final_values),
            })
            
            # Robustness metrics
            conv_stats.update({
                'convergence_efficiency': convergence_rate,  # Same as convergence_rate
                'convergence_consistency': 1 / (1 + np.std(conv_iterations)) if len(conv_iterations) > 1 else 1.0,
                'convergence_speed_score': 1 / (1 + np.mean(conv_iterations)),  # Lower iterations = better
                'convergence_robustness_score': convergence_rate * (1 / (1 + np.std(conv_iterations)) if len(conv_iterations) > 1 else 1.0)
            })
            
        else:
            # No converged runs
            final_values = group['final_value'].values
            conv_stats = {
                'convergence_rate': 0.0,
                'successful_runs': 0,
                'failed_runs': total_runs,
                'total_runs': total_runs,
                
                'mean_convergence_iteration': np.nan,
                'median_convergence_iteration': np.nan,
                'std_convergence_iteration': np.nan,
                'min_convergence_iteration': np.nan,
                'max_convergence_iteration': np.nan,
                'q25_convergence_iteration': np.nan,
                'q75_convergence_iteration': np.nan,
                'q90_convergence_iteration': np.nan,
                'q95_convergence_iteration': np.nan,
                
                'mean_convergence_value': np.nan,
                'median_convergence_value': np.nan,
                'std_convergence_value': np.nan,
                'best_convergence_value': np.nan,
                'worst_convergence_value': np.nan,
                
                'mean_final_value': np.mean(final_values) if len(final_values) > 0 else np.nan,
                'median_final_value': np.median(final_values) if len(final_values) > 0 else np.nan,
                'std_final_value': np.std(final_values) if len(final_values) > 0 else np.nan,
                'best_final_value': np.min(final_values) if len(final_values) > 0 else np.nan,
                'worst_final_value': np.max(final_values) if len(final_values) > 0 else np.nan,
                
                'convergence_efficiency': 0.0,
                'convergence_consistency': 0.0,
                'convergence_speed_score': 0.0,
                'convergence_robustness_score': 0.0,
            }
            
            if include_total_experiments:
                max_iterations = group['total_iterations'].max()
                max_total_exp = num_init + max_iterations
                
                conv_stats.update({
                    'mean_total_experiments': max_total_exp,
                    'median_total_experiments': max_total_exp,
                    'std_total_experiments': 0.0,
                    'min_total_experiments': max_total_exp,
                    'max_total_experiments': max_total_exp,
                    'q25_total_experiments': max_total_exp,
                    'q75_total_experiments': max_total_exp,
                    'q90_total_experiments': max_total_exp,
                    'q95_total_experiments': max_total_exp,
                })
        
        # Add strategy info
        conv_stats.update({
            'acquisition_type': acq_type,
            'num_init': num_init,
            'strategy': strategy_name
        })
        
        results.append(conv_stats)
    
    return pd.DataFrame(results)

def create_convergence_analysis_report(
    convergence_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    window_size: int,
    tolerance_type: str,
    tolerance_threshold: float,
    primary_metric: str = 'regret'
) -> str:
    """
    Create a comprehensive report of the moving window convergence analysis.
    """
    
    # Sort by convergence robustness score
    top_strategies = robustness_df.sort_values('convergence_robustness_score', ascending=False)
    
    report = f"""
# 🎯 Moving Window Convergence Analysis Report

## 📊 **Convergence Definition**
- **Window Size**: {window_size} consecutive iterations
- **Tolerance Type**: {tolerance_type}
- **Tolerance Threshold**: {tolerance_threshold:.4f}
- **Metric**: {primary_metric}

### 📝 **Interpretation**
Convergence is detected when the {primary_metric} shows **{tolerance_type} change ≤ {tolerance_threshold:.2%}** over {window_size} consecutive iterations.

This means the optimization has **practically stopped improving** and further iterations yield diminishing returns.

## 🏆 **Best Strategies by Convergence Robustness**

"""
    
    for i, (_, row) in enumerate(top_strategies.head(5).iterrows(), 1):
        if row['convergence_rate'] > 0:
            report += f"""
### {i}. {row['strategy']}
- **Convergence Rate**: {row['convergence_rate']:.1%} ({row['successful_runs']}/{row['total_runs']} runs)
- **Median Convergence**: {row['median_convergence_iteration']:.1f} BO iterations
- **Total Budget**: {row['median_total_experiments']:.1f} experiments (median)
- **Consistency**: {row['convergence_consistency']:.3f} (lower std = more consistent)
- **Speed Score**: {row['convergence_speed_score']:.3f} (higher = faster)
- **Robustness Score**: {row['convergence_robustness_score']:.3f}
"""
        else:
            report += f"""
### {i}. {row['strategy']} ❌
- **Convergence Rate**: 0% - Did not converge in any runs
- **Final Performance**: {row['median_final_value']:.4f} (median)
"""
    
    # Find best overall strategy
    best_strategy = top_strategies.iloc[0] if len(top_strategies) > 0 and top_strategies.iloc[0]['convergence_rate'] > 0 else None
    
    if best_strategy is not None:
        report += f"""

## 🎯 **Recommended Strategy: {best_strategy['strategy']}**

### 📊 **Budget Planning**
- **Optimistic (25th percentile)**: {best_strategy['q25_total_experiments']:.0f} total experiments
- **Realistic (median)**: {best_strategy['median_total_experiments']:.0f} total experiments  
- **Conservative (90th percentile)**: {best_strategy['q90_total_experiments']:.0f} total experiments
- **Worst case observed**: {best_strategy['max_total_experiments']:.0f} total experiments

### ⚡ **Convergence Characteristics**
- **Success Rate**: {best_strategy['convergence_rate']:.1%} probability of convergence
- **Typical BO Iterations**: {best_strategy['median_convergence_iteration']:.0f} iterations until convergence
- **Range**: {best_strategy['min_convergence_iteration']:.0f} - {best_strategy['max_convergence_iteration']:.0f} BO iterations
- **Consistency**: ±{best_strategy['std_convergence_iteration']:.1f} iterations std deviation

### 🎯 **Performance at Convergence**
- **Converged {primary_metric}**: {best_strategy['median_convergence_value']:.4f} (median)
- **Best case**: {best_strategy['best_convergence_value']:.4f}
- **Worst case**: {best_strategy['worst_convergence_value']:.4f}

## 📊 **Convergence Comparison Table**

| Strategy | Success Rate | Median BO Iters | Median Total Exp | Robustness Score |
|----------|-------------|------------------|------------------|------------------|
"""
        
        for _, row in top_strategies.iterrows():
            if row['convergence_rate'] > 0:
                report += f"| {row['strategy']} | {row['convergence_rate']:.1%} | {row['median_convergence_iteration']:.0f} | {row['median_total_experiments']:.0f} | {row['convergence_robustness_score']:.3f} |\n"
            else:
                report += f"| {row['strategy']} | 0% | - | - | 0.000 |\n"
        
        report += f"""

## 💡 **Key Insights**

### ✅ **Converging Strategies**
{top_strategies[top_strategies['convergence_rate'] > 0]['strategy'].tolist()}

### ❌ **Non-Converging Strategies**  
{top_strategies[top_strategies['convergence_rate'] == 0]['strategy'].tolist() if any(top_strategies['convergence_rate'] == 0) else 'None'}

### 🚀 **Practical Recommendations**

1. **For Budget Planning**: Use {best_strategy['median_total_experiments']:.0f} experiments as baseline budget
2. **For Risk Management**: Plan for up to {best_strategy['q90_total_experiments']:.0f} experiments (90% confidence)
3. **For Early Stopping**: Monitor convergence after {best_strategy['median_convergence_iteration']:.0f} BO iterations
4. **For Success Probability**: Expect {best_strategy['convergence_rate']:.1%} chance of achieving practical convergence

### ⚠️ **Convergence Detection in Practice**

Monitor these indicators during your BO run:
- Track {primary_metric} over the last {window_size} iterations
- Calculate {tolerance_type} change: should be ≤ {tolerance_threshold:.2%}
- If achieved: you can likely stop the optimization
- If not achieved after {best_strategy['q90_convergence_iteration']:.0f} iterations: consider continuing or adjusting parameters

## 🔧 **Technical Details**

### Moving Window Analysis
- **Window Size**: {window_size} iterations chosen to balance responsiveness vs. noise resistance
- **Tolerance Type**: {tolerance_type} - measures practical improvement significance
- **Threshold**: {tolerance_threshold:.4f} - represents "good enough" improvement level

### Robustness Metrics
- **Convergence Rate**: Fraction of scenarios achieving convergence
- **Consistency**: Inverse of iteration standard deviation (1/(1+std))
- **Speed Score**: Inverse of mean iterations (1/(1+mean))
- **Robustness Score**: Rate × Consistency (overall reliability)

*Note: This analysis uses practical convergence detection based on diminishing returns rather than arbitrary performance targets.*
"""
    
    else:
        report += f"""

## ❌ **No Strategies Achieved Convergence**

Based on the moving window analysis with:
- Window size: {window_size} iterations
- Tolerance: {tolerance_threshold:.2%} {tolerance_type} change

**Possible reasons:**
1. **Tolerance too strict**: Consider increasing threshold to {tolerance_threshold * 2:.4f}
2. **Window too large**: Consider reducing window size to {max(3, window_size - 2)}
3. **Insufficient iterations**: Strategies may need more than {robustness_df['total_runs'].max()} iterations
4. **Noisy objective**: Consider using coefficient_of_variation tolerance type

**Alternative Analysis**: Look at final performance trends instead of strict convergence.
"""
    
    return report

# Example usage and integration
def integrate_moving_window_convergence_analysis(
    df_results: pd.DataFrame,
    primary_metric: str = 'regret',
    window_size: int = 2,
    tolerance_type: str = 'relative',
    tolerance_threshold: float = 0.10,
    min_iterations: int = 10,
    require_monotonic: bool = False
) -> Dict:
    """
    Complete moving window convergence analysis pipeline.
    
    Returns:
    --------
    Dict containing:
    - convergence_df: Individual run analysis
    - robustness_df: Strategy-level metrics  
    - report: Text report
    - best_strategy: Recommended strategy info
    """
    
    # Step 1: Analyze convergence for all runs
    convergence_df = analyze_convergence_for_bo_results(
        df_results=df_results,
        primary_metric=primary_metric,
        window_size=window_size,
        tolerance_type=tolerance_type,
        tolerance_threshold=tolerance_threshold,
        min_iterations=min_iterations,
        require_monotonic=require_monotonic
    )
    
    # Step 2: Compute robustness metrics by strategy
    robustness_df = compute_convergence_robustness_metrics(
        convergence_df=convergence_df,
        include_total_experiments=True
    )
    
    # Step 3: Generate report
    report = create_convergence_analysis_report(
        convergence_df=convergence_df,
        robustness_df=robustness_df,
        window_size=window_size,
        tolerance_type=tolerance_type,
        tolerance_threshold=tolerance_threshold,
        primary_metric=primary_metric
    )
    
    # Step 4: Extract best strategy
    best_strategy = None
    if len(robustness_df) > 0:
        top_strategies = robustness_df.sort_values('convergence_robustness_score', ascending=False)
        if len(top_strategies) > 0 and top_strategies.iloc[0]['convergence_rate'] > 0:
            best_strategy = top_strategies.iloc[0].to_dict()
    
    return {
        'convergence_df': convergence_df,
        'robustness_df': robustness_df,
        'report': report,
        'best_strategy': best_strategy,
        'analysis_params': {
            'window_size': window_size,
            'tolerance_type': tolerance_type,
            'tolerance_threshold': tolerance_threshold,
            'min_iterations': min_iterations,
            'require_monotonic': require_monotonic,
            'primary_metric': primary_metric
        }
    }
# Streamlit Integration
def create_robustness_analysis_ui(df_results=None):
    """
    Create Streamlit UI for robustness analysis
    
    Args:
        df_results: Optional DataFrame. If provided, uses this data instead of session state.
                   This allows the function to work with uploaded files.
    """
    st.title("🎯 BO Strategy Robustness Analysis")
    
    # Determine data source
    if df_results is not None:
        # Use provided data (from upload)
        data_source = "uploaded"
        st.info(f"📁 **Analyzing uploaded data** ({len(df_results)} rows)")
    else:
        # Check if results exist in session state
        if 'bo_results' not in st.session_state or st.session_state['bo_results'] is None:
            st.warning("⚠️ No BO simulation results found. Please run the simulation first or upload a file.")
            return
        
        df_results = st.session_state['bo_results']
        data_source = "session"
        st.info(f"🎯 **Analyzing session data** ({len(df_results)} rows)")
    
    try:
        analyzer = RobustnessAnalyzer(df_results)
        
        # Rest of the function remains exactly the same...
        # Sidebar controls
        st.sidebar.header("🔧 Analysis Configuration")
        
        available_metrics = [col for col in df_results.columns 
                           if col not in ['acquisition_type', 'num_init', 'seed', 'iteration'] 
                           and df_results[col].dtype in ['float64', 'int64']]
        
        if not available_metrics:
            st.error("❌ No numeric metrics found for analysis")
            return
        
        selected_metric = st.sidebar.selectbox(
            "Primary Optimization Metric",
            options=available_metrics,
            index=0 if 'regret' not in available_metrics else available_metrics.index('regret'),
            help="Main metric for optimization performance (e.g., regret, loss)"
        )
        
        # Analysis mode selection
        analysis_mode = st.sidebar.radio(
            "Analysis Mode",
            ["🎯 Comprehensive (Multi-Metric)", "📊 Simple (Regret-Only)"],
            help="Comprehensive mode includes surrogate quality, convergence, etc."
        )
        
        # Weights for comprehensive recommendation
        if analysis_mode.startswith("🎯"):
            st.sidebar.subheader("⚖️ Multi-Criteria Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.25, 0.05, 
                                           help="How important is average optimization performance?")
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.25, 0.05,
                                             help="How important is consistency across ground truths?")
            safety_weight = st.sidebar.slider("Safety Weight", 0.0, 1.0, 0.15, 0.05,
                                             help="How important is worst-case performance?")
            surrogate_weight = st.sidebar.slider("Surrogate Quality Weight", 0.0, 1.0, 0.20, 0.05,
                                                help="How important is GP model accuracy?")
            convergence_weight = st.sidebar.slider("Convergence Weight", 0.0, 1.0, 0.15, 0.05,
                                                  help="How important is fast convergence?")
            
            # Normalize weights
            total_weight = perf_weight + robust_weight + safety_weight + surrogate_weight + convergence_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight  
                safety_weight /= total_weight
                surrogate_weight /= total_weight
                convergence_weight /= total_weight
        else:
            # Simple mode weights
            st.sidebar.subheader("⚖️ Simple Mode Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.4, 0.1)
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.4, 0.1)
            safety_weight = st.sidebar.slider("Safety Weight", 0.0, 1.0, 0.2, 0.1)
            surrogate_weight = 0.0
            convergence_weight = 0.0
            
            # Normalize weights
            total_weight = perf_weight + robust_weight + safety_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight  
                safety_weight /= total_weight
        
        # Main analysis
        st.header("📊 Strategy Recommendation")
        
        try:
            if analysis_mode.startswith("🎯"):
                # Comprehensive analysis
                recommendation = analyzer.recommend_comprehensive_strategy(
                    selected_metric, perf_weight, robust_weight, safety_weight,
                    surrogate_weight, convergence_weight
                )
                
                # Display comprehensive recommendation
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("🏆 Strategy", recommendation['strategy'].split(' (')[0], 
                             f"Init: {recommendation['strategy'].split('init=')[1].rstrip(')')}")
                with col2:
                    st.metric("🎯 Performance", f"{recommendation['performance_score']:.3f}")
                with col3:
                    st.metric("🛡️ Robustness", f"{recommendation['robustness_score']:.3f}")
                with col4:
                    st.metric("🔍 Surrogate Quality", f"{recommendation['surrogate_quality_score']:.3f}")
                with col5:
                    st.metric("⚡ Convergence", f"{recommendation['convergence_score']:.3f}")
                
                # Overall score
                st.metric("🏅 Overall Combined Score", f"{recommendation['combined_score']:.4f}")
                
                # Weight visualization
                with st.expander("⚖️ Weights Used"):
                    weights_df = pd.DataFrame(list(recommendation['weights_used'].items()), 
                                            columns=['Criterion', 'Weight'])
                    fig_weights = px.pie(weights_df, values='Weight', names='Criterion', 
                                       title="Criteria Weights")
                    st.plotly_chart(fig_weights, use_container_width=True)
                
            else:
                # Simple analysis (regret-only)
                recommendation = analyzer.recommend_strategy(
                    selected_metric, perf_weight, robust_weight, safety_weight
                )
                
                # Display simple recommendation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Recommended Strategy",
                        recommendation['strategy'],
                        f"Score: {recommendation['combined_score']:.3f}"
                    )
                
                with col2:
                    mean_val = recommendation['metrics'].get('mean', 'N/A')
                    std_val = recommendation['metrics'].get('std', 'N/A')
                    if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                        st.metric(
                            f"📈 Mean {selected_metric.upper()}",
                            f"{mean_val:.4f}",
                            f"±{std_val:.4f}"
                        )
                    else:
                        st.metric(f"📈 Mean {selected_metric.upper()}", str(mean_val))
                
                with col3:
                    success_rate = recommendation['metrics'].get('success_rate', 'N/A')
                    worst_case = recommendation['metrics'].get('q95', 'N/A')
                    if isinstance(success_rate, (int, float)):
                        st.metric(
                            "✅ Success Rate",
                            f"{success_rate:.1%}",
                            f"Worst: {worst_case:.4f}" if isinstance(worst_case, (int, float)) else f"Worst: {worst_case}"
                        )
                    else:
                        st.metric("✅ Success Rate", str(success_rate))
                        
        except Exception as e:
            st.error(f"Strategy recommendation failed: {str(e)}")
            st.exception(e)
        
        # Add new tab for experiment planning
        viz_tab, planning_tab = st.tabs(["📈 Visualizations", "🧪 Experiment Planning"])
        
        with viz_tab:
            # Convergence plot
            st.subheader("🔄 Convergence Robustness")
            try:
                conv_fig = analyzer.plot_convergence_robustness(selected_metric)
                st.plotly_chart(conv_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Convergence plot failed: {str(e)}")
                st.write("Debug info:", str(e))
            
            # Comparison plot
            st.subheader("🔍 Strategy Comparison")
            try:
                comp_fig = analyzer.plot_robustness_comparison(selected_metric)
                st.plotly_chart(comp_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Comparison plot failed: {str(e)}")
                st.write("Debug info:", str(e))
        
        with planning_tab:
            st.subheader("🧪 Practical Experiment Planning")
            
            try:
                # Get PRACTICAL experiment planning analysis (BO strategies only)
                convergence_df = analyzer.get_experiment_planning_analysis(selected_metric)
                
                if not convergence_df.empty:
                    # Key insights at the top
                    st.subheader("⚡ Practical BO Planning Insights")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Best BO strategy for minimum total experiments
                    converged_strategies = convergence_df[convergence_df['success_rate'] > 0]
                    if not converged_strategies.empty:
                        best_strategy = converged_strategies.loc[converged_strategies['median_total_experiments'].idxmin()]
                        
                        with col1:
                            st.metric(
                                "🏆 Most Efficient BO Strategy",
                                best_strategy['strategy'],
                                f"{best_strategy['median_total_experiments']:.0f} total exp."
                            )
                        
                        with col2:
                            st.metric(
                                "⚡ Best Case Total",
                                f"{best_strategy['min_total_experiments']:.0f} experiments",
                                f"Success: {best_strategy['success_rate']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "🎯 Realistic Budget",
                                f"{best_strategy['median_total_experiments']:.0f} experiments",
                                f"{best_strategy['num_init']} init + {best_strategy['median_bo_iterations']:.0f} BO"
                            )
                        
                        with col4:
                            st.metric(
                                "🛡️ Conservative Budget", 
                                f"{best_strategy['total_experiments_90th_percentile']:.0f} experiments",
                                "90% confidence"
                            )
                    
                    # Detailed experiment planning table
                    st.subheader("📊 Detailed BO Experiment Planning")
                    
                    # Format the dataframe for display
                    display_conv_df = convergence_df.copy()
                    
                    # Round numeric columns
                    numeric_cols = ['median_total_experiments', 'mean_total_experiments', 
                                   'total_experiments_90th_percentile', 'total_experiments_95th_percentile',
                                   'min_total_experiments', 'max_total_experiments',
                                   'median_bo_iterations']
                    
                    for col in numeric_cols:
                        if col in display_conv_df.columns:
                            display_conv_df[col] = display_conv_df[col].round(1)
                    
                    display_conv_df['success_rate'] = (display_conv_df['success_rate'] * 100).round(1)
                    
                    # Select key columns for display
                    key_cols = ['strategy', 'success_rate', 'median_total_experiments', 
                               'total_experiments_90th_percentile', 'num_init', 'median_bo_iterations',
                               'min_total_experiments', 'max_total_experiments']
                    
                    display_cols = [col for col in key_cols if col in display_conv_df.columns]
                    
                    st.dataframe(
                        display_conv_df[display_cols].rename(columns={
                            'strategy': 'BO Strategy',
                            'success_rate': 'Success Rate (%)',
                            'median_total_experiments': 'Median Total Exp.',
                            'total_experiments_90th_percentile': '90% Conf. Total Exp.',
                            'num_init': 'Initial Exp.',
                            'median_bo_iterations': 'Median BO Iters',
                            'min_total_experiments': 'Min Total',
                            'max_total_experiments': 'Max Total'
                        }),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Total experiments visualization
                    st.subheader("📈 Total Experiment Requirements")
                    
                    # Create comparison plot
                    fig_exp = go.Figure()
                    
                    for idx, row in converged_strategies.iterrows():
                        if not pd.isna(row['median_total_experiments']):
                            fig_exp.add_trace(go.Bar(
                                name=row['strategy'],
                                x=['Best Case', 'Median', '90% Confidence'],
                                y=[row['min_total_experiments'], 
                                   row['median_total_experiments'],
                                   row['total_experiments_90th_percentile']],
                                text=[f"{row['min_total_experiments']:.0f}", 
                                     f"{row['median_total_experiments']:.0f}",
                                     f"{row['total_experiments_90th_percentile']:.0f}"],
                                textposition='auto'
                            ))
                    
                    fig_exp.update_layout(
                        title="Total Experiments Needed by BO Strategy",
                        yaxis_title="Total Experiments (Initial + BO)",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig_exp, use_container_width=True)
                    
                    # Initial sampling comparison
                    if len(convergence_df) > 1:
                        st.subheader("🔄 Initial Sampling Strategy Impact")
                        
                        # Group by acquisition function to compare init sizes
                        acq_functions = convergence_df['acquisition_type'].unique()
                        
                        for acq in acq_functions:
                            acq_data = convergence_df[convergence_df['acquisition_type'] == acq]
                            if len(acq_data) > 1:
                                st.write(f"**{acq} Acquisition Function:**")
                                
                                comparison_data = []
                                for _, row in acq_data.iterrows():
                                    if not pd.isna(row['median_total_experiments']):
                                        efficiency = row['median_bo_iterations'] / row['num_init'] if row['num_init'] > 0 else float('inf')
                                        comparison_data.append({
                                            'Initial Points': row['num_init'],
                                            'Total Experiments': row['median_total_experiments'],
                                            'BO Iterations': row['median_bo_iterations'],
                                            'Efficiency Ratio': efficiency,
                                            'Success Rate': f"{row['success_rate']:.1%}"
                                        })
                                
                                if comparison_data:
                                    comp_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Generate practical planning report
                    st.subheader("📋 Practical Experiment Planning Report")
                    
                    try:
                        planning_report = analyzer.create_practical_experiment_planning_report(selected_metric)
                        st.markdown(planning_report)
                        
                        # Download button for planning report
                        st.download_button(
                            "📥 Download Practical Planning Report",
                            planning_report,
                            f"practical_experiment_planning_{selected_metric}.md",
                            "text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Planning report generation failed: {str(e)}")
                
                else:
                    st.warning("No BO strategies available for practical experiment planning.")
                    st.write("**Possible reasons:**")
                    st.write("- Only Random strategies in data (not suitable for BO planning)")
                    st.write("- No strategies converged to target performance")
                    st.write("- Missing required data columns")
            
            except Exception as e:
                st.error(f"Practical experiment planning analysis failed: {str(e)}")
                st.exception(e)
        
        # Detailed metrics table
        st.header("📋 Detailed Metrics")
        
        robustness_metrics = analyzer.compute_robustness_metrics(selected_metric)
        
        # Add combined_score if not present (same logic as in plot function)
        if 'combined_score' not in robustness_metrics.columns:
            # Normalize scores (0-1 scale) with safety checks
            max_perf = robustness_metrics['performance_score'].max()
            max_robust = robustness_metrics['robustness_score'].max()
            min_q95 = robustness_metrics['q95'].min()
            
            if max_perf > 0:
                perf_scores = robustness_metrics['performance_score'] / max_perf
            else:
                perf_scores = robustness_metrics['performance_score'] * 0 + 0.5
                
            if max_robust > 0:
                robust_scores = robustness_metrics['robustness_score'] / max_robust
            else:
                robust_scores = robustness_metrics['robustness_score'] * 0 + 0.5
                
            if min_q95 > 0 and not np.isnan(min_q95):
                # Inverse relationship: lower q95 (better worst-case) = higher safety score
                # Transform so best case gets ~1.0, worst case gets ~0.0
                max_q95 = robustness_metrics['q95'].max()
                if max_q95 > min_q95:  # Avoid division by zero
                    safety_scores = 1 - (robustness_metrics['q95'] - min_q95) / (max_q95 - min_q95)
                else:
                    safety_scores = robustness_metrics['q95'] * 0 + 1.0  # All equal, give max score
            else:
                safety_scores = robustness_metrics['q95'] * 0 + 0.5
            
            # Combined score with weights from UI
            robustness_metrics['combined_score'] = (
                perf_weight * perf_scores + 
                robust_weight * robust_scores + 
                safety_weight * safety_scores
            )
        
        robustness_metrics = robustness_metrics.sort_values('combined_score', ascending=False)
        
        # Format for display
        display_metrics = robustness_metrics.copy()
        display_metrics = display_metrics.round(4)
        
        st.dataframe(
            display_metrics,
            use_container_width=True,
            height=400
        )
        
        # Text report
        st.header("📄 Analysis Report")
        
        try:
            if analysis_mode.startswith("🎯"):
                report = analyzer.create_comprehensive_strategy_report(selected_metric)
            else:
                report = analyzer.create_strategy_report(selected_metric)
            st.markdown(report)
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            st.write("Basic information available in the metrics table above.")
        
        # Download options
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download metrics CSV
            csv_data = robustness_metrics.to_csv(index=False)
            st.download_button(
                "📥 Download Metrics CSV",
                csv_data,
                f"robustness_metrics_{selected_metric}.csv",
                "text/csv"
            )
        
        with col2:
            # Download report
            st.download_button(
                "📄 Download Report",
                report,
                f"robustness_report_{selected_metric}.md",
                "text/markdown"
            )
        
        # Store results (only if using session data, not uploaded data)
        if data_source == "session":
            st.session_state['robustness_analyzer'] = analyzer
            st.session_state['robustness_recommendation'] = recommendation
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

# Example usage for testing
if __name__ == "__main__":
    # This would be called from your main Streamlit app
    create_robustness_analysis_ui()