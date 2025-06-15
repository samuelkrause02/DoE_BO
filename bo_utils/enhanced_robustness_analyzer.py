# Part 2: Enhanced RobustnessAnalyzer
# File: enhanced_robustness_analyzer.py

# Import Part 1
from bo_utils.scale_invariant_convergence import ScaleInvariantConvergenceDetector
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class RobustnessAnalyzer:
    """
    Enhanced robustness analysis for Bayesian Optimization strategies
    with scale-invariant convergence detection.
    
    This class replaces the old global convergence target approach with
    relative, scale-invariant methods that work for any problem magnitude.
    """
    
    def __init__(self, 
                 df_results: pd.DataFrame, 
                 convergence_method: str = 'dual_strategy',
                 relative_tolerance: float = 0.05,
                 cv_threshold: float = 0.02):
        """
        Initialize with simulation results DataFrame and enhanced convergence detection.
        
        Parameters:
        -----------
        df_results : pd.DataFrame
            BO simulation results with columns: acquisition_type, num_init, seed, iteration, regret, etc.
        convergence_method : str, default='dual_strategy'
            Convergence detection method: 'dual_strategy', 'relative_tolerance', 'coefficient_of_variation'
        relative_tolerance : float, default=0.05
            Relative tolerance for convergence (5% = 0.05) - scales automatically
        cv_threshold : float, default=0.02
            Coefficient of variation threshold (2% = 0.02) - dimensionless
        """
        self.df = df_results.copy()
        self.convergence_method = convergence_method
        
        # Initialize enhanced scale-invariant convergence detector
        self.convergence_detector = ScaleInvariantConvergenceDetector(
            relative_tolerance=relative_tolerance,
            cv_threshold=cv_threshold,
            window_size=5,
            min_iterations=10
        )
        
        self.validate_data()
        
    def validate_data(self):
        """Validate required columns exist and add computed metrics if needed."""
        required_cols = ['acquisition_type', 'num_init', 'seed', 'iteration']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add regret if not present but can be computed
        if 'regret' not in self.df.columns and 'observed_best' in self.df.columns and 'true_best' in self.df.columns:
            self.df['regret'] = (self.df['true_best'] - self.df['observed_best']) ** 2
        
        print(f"✅ Data validated. Shape: {self.df.shape}")
        available_metrics = [col for col in self.df.columns 
                           if col not in ['acquisition_type', 'num_init', 'seed', 'iteration']]
        print(f"📊 Available metrics: {available_metrics}")
    
    def _enhanced_convergence_analysis(self, 
                                     strategy_data: pd.DataFrame, 
                                     primary_metric: str) -> List[Dict]:
        """
        Enhanced convergence analysis using scale-invariant methods.
        
        Replaces the old global convergence target approach with relative methods
        that automatically adapt to any problem magnitude.
        
        Parameters:
        -----------
        strategy_data : pd.DataFrame
            Data for one strategy (all seeds)
        primary_metric : str
            Primary metric column name (e.g., 'regret')
            
        Returns:
        --------
        List[Dict] : Convergence details for each seed with enhanced scale-invariant analysis
        """
        convergence_details = []
        
        for seed in strategy_data['seed'].unique():
            seed_data = strategy_data[strategy_data['seed'] == seed].sort_values('iteration')
            
            if len(seed_data) < 10 or primary_metric not in seed_data.columns:
                # Insufficient data for convergence analysis
                convergence_details.append({
                    'seed': seed,
                    'convergence_iteration': None,
                    'convergence_method': self.convergence_method,
                    'converged': False,
                    'reason': 'insufficient_data',
                    'total_iterations': len(seed_data),
                    'final_regret': seed_data[primary_metric].iloc[-1] if len(seed_data) > 0 else None,
                    'scale_invariant': True
                })
                continue
            
            regret_history = seed_data[primary_metric].tolist()
            
            # Test convergence at each iteration using enhanced scale-invariant methods
            convergence_iteration = None
            convergence_result = None
            
            for i in range(self.convergence_detector.min_iterations, len(regret_history)):
                current_history = regret_history[:i+1]
                
                # Apply selected convergence method
                if self.convergence_method == 'dual_strategy':
                    # Recommended: Dual strategy with consensus between relative tolerance & CV
                    result = self.convergence_detector.check_dual_strategy_convergence(
                        current_history,
                        primary_method='relative_tolerance',
                        secondary_method='coefficient_of_variation',
                        consensus_required=True  # Both methods must agree for robustness
                    )
                elif self.convergence_method == 'relative_tolerance':
                    # Scale-invariant relative tolerance method
                    result = self.convergence_detector.check_relative_stabilization(current_history)
                elif self.convergence_method == 'coefficient_of_variation':
                    # Dimensionless CV method
                    result = self.convergence_detector.check_cv_stabilization(current_history)
                elif self.convergence_method == 'comprehensive':
                    # Majority vote across all methods
                    result = self.convergence_detector.analyze_convergence_comprehensive(current_history)
                    result['converged'] = result['overall_converged']
                else:
                    raise ValueError(f"Unknown convergence method: {self.convergence_method}")
                
                if result['converged']:
                    convergence_iteration = i
                    convergence_result = result
                    break
            
            # Record enhanced convergence details
            convergence_details.append({
                'seed': seed,
                'convergence_iteration': convergence_iteration,
                'convergence_method': self.convergence_method,
                'converged': convergence_iteration is not None,
                'convergence_details': convergence_result,
                'total_iterations': len(seed_data),
                'final_regret': seed_data[primary_metric].iloc[-1],
                'initial_regret': seed_data[primary_metric].iloc[0],
                'total_improvement': seed_data[primary_metric].iloc[0] - seed_data[primary_metric].iloc[-1],
                'efficiency_ratio': (convergence_iteration / len(seed_data)) if convergence_iteration else 1.0,
                'scale_invariant': True,
                'early_convergence': convergence_iteration < int(0.8 * len(seed_data)) if convergence_iteration else False
            })
        
        return convergence_details
    
    def compute_comprehensive_robustness_metrics(self, primary_metric: str = 'regret') -> pd.DataFrame:
        """
        Compute comprehensive robustness metrics including enhanced scale-invariant convergence analysis.
        
        Parameters:
        -----------
        primary_metric : str, default='regret'
            Primary optimization metric to analyze
            
        Returns:
        --------
        pd.DataFrame : Comprehensive metrics for each strategy including scale-invariant convergence
        """
        if primary_metric not in self.df.columns:
            raise ValueError(f"Primary metric '{primary_metric}' not found in data")
        
        # Get final values for optimization-focused metrics (last iteration for each run)
        final_values = self.df.groupby(['acquisition_type', 'num_init', 'seed'])[primary_metric].last().reset_index()
        
        # Get all iterations for surrogate quality metrics
        all_iterations = self.df.copy()
        
        results = []
        
        for (acq_type, num_init), group in final_values.groupby(['acquisition_type', 'num_init']):
            strategy_data = all_iterations[
                (all_iterations['acquisition_type'] == acq_type) & 
                (all_iterations['num_init'] == num_init)
            ]
            
            # 1. OPTIMIZATION PERFORMANCE METRICS (based on primary metric)
            primary_values = group[primary_metric]
            
            opt_metrics = {
                'acquisition_type': acq_type,
                'num_init': num_init,
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
                            f'{metric}_final': strategy_data.groupby('seed')[metric].last().mean(),
                        })
            
            # 3. ENHANCED CONVERGENCE METRICS with Scale-Invariant Methods
            convergence_metrics = {}
            
            if 'iteration' in strategy_data.columns:
                # Use enhanced scale-invariant convergence analysis
                convergence_details = self._enhanced_convergence_analysis(
                    strategy_data, primary_metric
                )
                
                convergence_data = []
                early_convergence_data = []
                
                for detail in convergence_details:
                    if detail['converged'] and detail['convergence_iteration'] is not None:
                        convergence_data.append(detail['convergence_iteration'])
                        
                        if detail['early_convergence']:
                            early_convergence_data.append(detail['convergence_iteration'])
                
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
                        'avg_efficiency_ratio': np.mean([d['efficiency_ratio'] for d in convergence_details if d['converged']]),
                        'convergence_method_used': self.convergence_method,
                        'scale_invariant_analysis': True
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
                        'avg_efficiency_ratio': 1.0,
                        'convergence_method_used': self.convergence_method,
                        'scale_invariant_analysis': True
                    })
                
                # Store detailed convergence info for further analysis
                convergence_metrics['convergence_details'] = convergence_details
            
            # 4. COMBINE ALL METRICS
            combined_metrics = {**opt_metrics, **surrogate_metrics, **convergence_metrics}
            results.append(combined_metrics)
        
        df_results = pd.DataFrame(results)
        
        # 5. COMPUTE COMPOSITE SCORES
        if len(df_results) > 0:
            # Performance score (lower primary metric is better for regret-like metrics)
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
            
            # Convergence score based on enhanced scale-invariant analysis
            if 'convergence_reliability' in df_results.columns:
                df_results['convergence_score'] = df_results['convergence_reliability'].fillna(0)
            else:
                df_results['convergence_score'] = 0.5
        
        return df_results
    
    def recommend_comprehensive_strategy(self, 
                                       primary_metric: str = 'regret',
                                       performance_weight: float = 0.25,
                                       robustness_weight: float = 0.25, 
                                       safety_weight: float = 0.15,
                                       surrogate_weight: float = 0.20,
                                       convergence_weight: float = 0.15) -> Dict:
        """
        Multi-criteria strategy recommendation including enhanced scale-invariant convergence analysis.
        
        Parameters:
        -----------
        primary_metric : str, default='regret'
            Primary optimization metric
        performance_weight : float, default=0.25
            Weight for optimization performance
        robustness_weight : float, default=0.25
            Weight for robustness (consistency across runs)
        safety_weight : float, default=0.15
            Weight for safety (worst-case performance)
        surrogate_weight : float, default=0.20
            Weight for surrogate model quality
        convergence_weight : float, default=0.15
            Weight for convergence behavior
            
        Returns:
        --------
        Dict : Comprehensive recommendation with best strategy and all analysis details
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
            'strategy': f"{best_strategy['acquisition_type']} (init={best_strategy['num_init']})",
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
            },
            'convergence_method': self.convergence_method,
            'scale_invariant': True
        }
        
        return recommendation
    
    def get_experiment_planning_analysis(self, primary_metric: str = 'regret') -> pd.DataFrame:
        """
        Extract practical experiment planning analysis with enhanced scale-invariant convergence.
        Reports total experiments needed (initial + BO iterations), not just BO iterations.
        
        Parameters:
        -----------
        primary_metric : str, default='regret'
            Primary optimization metric
            
        Returns:
        --------
        pd.DataFrame : Experiment planning analysis with total experiments and scale-invariant convergence info
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
                
                strategy_name = f"{row['acquisition_type']} (init={row['num_init']})"
                
                # Extract convergence statistics from enhanced scale-invariant analysis
                converged_seeds = [d for d in details if d['converged'] and d['convergence_iteration'] is not None]
                failed_seeds = [d for d in details if not d['converged']]
                
                if converged_seeds:
                    # CRITICAL: Convert BO iterations to TOTAL experiments
                    # Total experiments = initial_points + BO_iterations
                    initial_experiments = row['num_init']
                    
                    total_experiments_list = []
                    bo_iterations_list = []
                    for d in converged_seeds:
                        bo_iterations = d['convergence_iteration']
                        total_experiments = initial_experiments + bo_iterations
                        total_experiments_list.append(total_experiments)
                        bo_iterations_list.append(bo_iterations)
                    
                    # Add scale-invariant method info
                    convergence_method_info = converged_seeds[0].get('convergence_method', 'unknown')
                    scale_invariant = all(d.get('scale_invariant', False) for d in converged_seeds)
                    
                    convergence_results.append({
                        'strategy': strategy_name,
                        'acquisition_type': row['acquisition_type'],
                        'num_init': row['num_init'],
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
                        'min_bo_iterations': min(bo_iterations_list),
                        'median_bo_iterations': np.median(bo_iterations_list),
                        'max_bo_iterations': max(bo_iterations_list),
                        'mean_bo_iterations': np.mean(bo_iterations_list),
                        
                        # Enhanced convergence info
                        'convergence_method': convergence_method_info,
                        'scale_invariant_analysis': scale_invariant,
                        'early_convergence_rate': sum(1 for d in converged_seeds if d.get('early_convergence', False)) / len(converged_seeds),
                        'max_available_iterations': details[0]['total_iterations']
                    })
                else:
                    # No seeds converged
                    initial_experiments = row['num_init']
                    max_total_experiments = initial_experiments + details[0]['total_iterations']
                    convergence_method_info = details[0].get('convergence_method', 'unknown')
                    
                    convergence_results.append({
                        'strategy': strategy_name,
                        'acquisition_type': row['acquisition_type'],
                        'num_init': row['num_init'],
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
                        'mean_bo_iterations': np.nan,
                        'convergence_method': convergence_method_info,
                        'scale_invariant_analysis': True,
                        'early_convergence_rate': 0.0,
                        'max_available_experiments': max_total_experiments
                    })
        
        return pd.DataFrame(convergence_results)


# Test function for Part 2
def test_enhanced_robustness_analyzer():
    """
    Test function to verify enhanced robustness analyzer works with scale-invariant convergence.
    """
    print("🔬 Testing Enhanced RobustnessAnalyzer")
    print("=" * 40)
    
    # Create sample data for testing
    np.random.seed(42)
    
    # Sample BO simulation data
    data = []
    strategies = [('EI', 5), ('UCB', 5), ('PI', 10)]
    seeds = [1, 2, 3]
    iterations = list(range(50))
    
    for acq_type, num_init in strategies:
        for seed in seeds:
            # Simulate convergence pattern (decreasing regret)
            base_regret = np.random.exponential(1000) if acq_type == 'EI' else np.random.exponential(0.1)
            regret_values = base_regret * np.exp(-np.array(iterations) * 0.1) + np.random.normal(0, base_regret * 0.02, len(iterations))
            
            for i, regret in enumerate(regret_values):
                data.append({
                    'acquisition_type': acq_type,
                    'num_init': num_init,
                    'seed': seed,
                    'iteration': i,
                    'regret': max(regret, 0.001)  # Ensure positive
                })
    
    df_test = pd.DataFrame(data)
    print(f"📊 Created test data: {df_test.shape}")
    
    # Test enhanced analyzer
    analyzer = RobustnessAnalyzer(
        df_test, 
        convergence_method='dual_strategy',
        relative_tolerance=0.05,
        cv_threshold=0.02
    )
    
    # Test comprehensive metrics
    metrics = analyzer.compute_comprehensive_robustness_metrics('regret')
    print(f"✅ Computed metrics for {len(metrics)} strategies")
    
    # Test recommendation
    recommendation = analyzer.recommend_comprehensive_strategy('regret')
    print(f"🏆 Best strategy: {recommendation['strategy']}")
    print(f"🎯 Combined score: {recommendation['combined_score']:.3f}")
    print(f"🔬 Scale-invariant: {recommendation['scale_invariant']}")
    
    # Test experiment planning
    planning = analyzer.get_experiment_planning_analysis('regret')
    print(f"🧪 Experiment planning for {len(planning)} strategies")
    
    print("\n✅ Enhanced RobustnessAnalyzer test complete!")
    return analyzer


if __name__ == "__main__":
    print("🚀 Part 2: Enhanced RobustnessAnalyzer Ready!")
    print("\n🔧 Features:")
    print("✅ Scale-invariant convergence detection integration")
    print("✅ Comprehensive robustness metrics")
    print("✅ Multi-criteria strategy recommendation")
    print("✅ Experiment planning with total experiments")
    print("✅ Enhanced surrogate quality analysis")
    print("✅ Automatic adaptation to problem magnitude")
    
    print("\n📏 Scale-Invariant Benefits:")
    print("• Same convergence criteria for €1000s and 0.001% problems")
    print("• Automatic threshold adaptation")
    print("• Robust multi-method consensus")
    print("• Universal applicability")
    
    # Run test
    test_analyzer = test_enhanced_robustness_analyzer()
    print("\n✅ Part 2 complete - ready for Part 3 (UI Function)!")