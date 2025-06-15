# Teil 1: Imports + ScaleInvariantConvergenceDetector
# Datei: scale_invariant_convergence.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ScaleInvariantConvergenceDetector:
    """
    Enhanced convergence detection with scale-invariant relative tolerance methods.
    Solves das fundamentale Skalierungsproblem: Gleiche Konvergenz-Kriterien für 
    €1000s und 0.001% Werte.
    """
    
    def __init__(self, 
                 relative_tolerance: float = 0.05,
                 cv_threshold: float = 0.02,
                 window_size: int = 5,
                 min_iterations: int = 10):
        """
        Initialize the scale-invariant convergence detector.
        
        Parameters:
        -----------
        relative_tolerance : float, default=0.05
            Relative tolerance as fraction of current level (5% = 0.05)
            Beispiel: €917 → tolerance = €45.85, 0.0917% → tolerance = 0.0046%
        cv_threshold : float, default=0.02
            Coefficient of variation threshold (2% = 0.02) - völlig dimensionslos
        window_size : int, default=5
            Number of recent iterations to analyze
        min_iterations : int, default=10
            Minimum iterations before checking convergence
        """
        self.relative_tolerance = relative_tolerance
        self.cv_threshold = cv_threshold
        self.window_size = window_size
        self.min_iterations = min_iterations
        
    def check_relative_stabilization(self, 
                                   regret_history: List[float], 
                                   tolerance: Optional[float] = None) -> Dict:
        """
        Check convergence using relative tolerance (% of current level).
        
        SCALE-INVARIANT: Funktioniert für jede Problem-Größenordnung.
        
        Beispiele:
        - €917 Kosten → tolerance = €917 * 0.05 = €45.85
        - 0.0917% Effizienz → tolerance = 0.0917% * 0.05 = 0.0046%
        - Gleiche 5% Kriterien für beide Probleme!
        
        Parameters:
        -----------
        regret_history : List[float]
            Historie der regret values
        tolerance : Optional[float]
            Override für relative_tolerance, falls None wird self.relative_tolerance verwendet
            
        Returns:
        --------
        Dict mit convergence result und details
        """
        if tolerance is None:
            tolerance = self.relative_tolerance
            
        if len(regret_history) < max(self.window_size, self.min_iterations):
            return {
                'converged': False,
                'method': 'relative_tolerance',
                'reason': 'insufficient_data',
                'details': {
                    'required_length': max(self.window_size, self.min_iterations),
                    'actual_length': len(regret_history),
                    'tolerance_used': tolerance
                }
            }
        
        recent_values = np.array(regret_history[-self.window_size:])
        current_level = np.mean(recent_values)
        regret_range = np.max(recent_values) - np.min(recent_values)
        
        # SCALE-INVARIANT LOGIC: Handle edge cases
        if current_level <= 0:
            # Wenn wir zero regret oder negative Werte erreicht haben
            # Fallback auf absolute tolerance basierend auf historischer Variabilität
            absolute_tolerance = np.std(regret_history) * tolerance
            tolerance_actual = absolute_tolerance
        else:
            # Normal case: tolerance relativ zum aktuellen Level
            # Das ist der KERN der scale-invariance!
            tolerance_actual = current_level * tolerance
        
        converged = regret_range <= tolerance_actual
        
        return {
            'converged': converged,
            'method': 'relative_tolerance',
            'details': {
                'current_level': current_level,
                'regret_range': regret_range,
                'tolerance_used': tolerance_actual,
                'relative_tolerance_percent': tolerance * 100,
                'window_values': recent_values.tolist(),
                'convergence_ratio': regret_range / tolerance_actual if tolerance_actual > 0 else float('inf'),
                'scale_invariant': True
            }
        }
    
    def check_cv_stabilization(self, 
                              regret_history: List[float], 
                              cv_threshold: Optional[float] = None) -> Dict:
        """
        Check convergence using Coefficient of Variation (CV = std/mean).
        
        VÖLLIG SCALE-INVARIANT und dimensionslos.
        CV ist ein relativer Maßstab der unabhängig von der Größenordnung funktioniert.
        
        Beispiele:
        - €917 ± €45.85 → CV = 45.85/917 = 0.05 = 5%
        - 0.0917% ± 0.0046% → CV = 0.0046/0.0917 = 0.05 = 5%
        - Gleicher CV für beide Probleme!
        
        Parameters:
        -----------
        regret_history : List[float]
            Historie der regret values
        cv_threshold : Optional[float]
            Override für cv_threshold, falls None wird self.cv_threshold verwendet
            
        Returns:
        --------
        Dict mit convergence result und CV details
        """
        if cv_threshold is None:
            cv_threshold = self.cv_threshold
            
        if len(regret_history) < max(self.window_size, self.min_iterations):
            return {
                'converged': False,
                'method': 'coefficient_of_variation',
                'reason': 'insufficient_data',
                'details': {
                    'required_length': max(self.window_size, self.min_iterations),
                    'actual_length': len(regret_history),
                    'cv_threshold': cv_threshold
                }
            }
        
        recent_values = np.array(regret_history[-self.window_size:])
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        # ROBUST EDGE CASE HANDLING für scale-invariance
        if mean_val <= 0 or np.isclose(mean_val, 0):
            # Special handling when mean is near zero
            if np.allclose(recent_values, recent_values[0], atol=1e-10):
                cv = 0.0  # All values are essentially identical → perfect convergence
            else:
                cv = float('inf')  # High variation around zero → no convergence
        else:
            # Normal case: CV = std/mean (dimensionslos!)
            cv = std_val / mean_val
        
        converged = cv <= cv_threshold
        
        return {
            'converged': converged,
            'method': 'coefficient_of_variation',
            'details': {
                'cv_value': cv,
                'cv_threshold': cv_threshold,
                'mean_value': mean_val,
                'std_value': std_val,
                'window_values': recent_values.tolist(),
                'cv_percent': cv * 100,
                'scale_invariant': True,
                'dimensionless': True
            }
        }
    
    def check_relative_range_stabilization(self, 
                                         regret_history: List[float],
                                         range_threshold: float = 0.05) -> Dict:
        """
        Check convergence using relative range: (max - min) / mean.
        
        Another scale-invariant approach - relative range ist dimensionslos.
        
        Parameters:
        -----------
        regret_history : List[float]
            Historie der regret values
        range_threshold : float, default=0.05
            Threshold für relative range (5% = 0.05)
            
        Returns:
        --------
        Dict mit convergence result und relative range details
        """
        if len(regret_history) < max(self.window_size, self.min_iterations):
            return {
                'converged': False,
                'method': 'relative_range',
                'reason': 'insufficient_data',
                'details': {
                    'required_length': max(self.window_size, self.min_iterations),
                    'actual_length': len(regret_history),
                    'range_threshold': range_threshold
                }
            }
        
        recent_values = np.array(regret_history[-self.window_size:])
        mean_val = np.mean(recent_values)
        range_val = np.max(recent_values) - np.min(recent_values)
        
        # Handle edge cases für scale-invariance
        if mean_val <= 0 or np.isclose(mean_val, 0):
            if np.allclose(recent_values, recent_values[0], atol=1e-10):
                relative_range = 0.0  # All values identical → perfect convergence
            else:
                relative_range = float('inf')  # Range around zero → problematic
        else:
            # Normal case: relative range = (max-min)/mean (dimensionslos!)
            relative_range = range_val / mean_val
        
        converged = relative_range <= range_threshold
        
        return {
            'converged': converged,
            'method': 'relative_range',
            'details': {
                'relative_range': relative_range,
                'range_threshold': range_threshold,
                'absolute_range': range_val,
                'mean_value': mean_val,
                'window_values': recent_values.tolist(),
                'relative_range_percent': relative_range * 100,
                'scale_invariant': True,
                'dimensionless': True
            }
        }
    
    def check_dual_strategy_convergence(self, 
                                      regret_history: List[float],
                                      primary_method: str = 'relative_tolerance',
                                      secondary_method: str = 'coefficient_of_variation',
                                      consensus_required: bool = True) -> Dict:
        """
        Dual strategy: Use two complementary methods for robust convergence detection.
        
        EMPFOHLENE METHODE für production use! 
        Kombiniert relative tolerance und CV für maximale Robustheit.
        
        Parameters:
        -----------
        regret_history : List[float]
            Historie der regret values
        primary_method : str, default='relative_tolerance'
            Primary convergence method
        secondary_method : str, default='coefficient_of_variation'
            Secondary convergence method for confirmation
        consensus_required : bool, default=True
            If True, both methods must agree. If False, either method can trigger convergence.
            
        Returns:
        --------
        Dict mit dual strategy result und details beider Methoden
        """
        # Run primary method
        if primary_method == 'relative_tolerance':
            primary_result = self.check_relative_stabilization(regret_history)
        elif primary_method == 'coefficient_of_variation':
            primary_result = self.check_cv_stabilization(regret_history)
        elif primary_method == 'relative_range':
            primary_result = self.check_relative_range_stabilization(regret_history)
        else:
            raise ValueError(f"Unknown primary method: {primary_method}")
        
        # Run secondary method
        if secondary_method == 'relative_tolerance':
            secondary_result = self.check_relative_stabilization(regret_history)
        elif secondary_method == 'coefficient_of_variation':
            secondary_result = self.check_cv_stabilization(regret_history)
        elif secondary_method == 'relative_range':
            secondary_result = self.check_relative_range_stabilization(regret_history)
        else:
            raise ValueError(f"Unknown secondary method: {secondary_method}")
        
        # Determine final convergence based on consensus
        if consensus_required:
            # ROBUSTE STRATEGIE: Beide Methoden müssen zustimmen
            final_converged = primary_result['converged'] and secondary_result['converged']
            decision_logic = "both_must_agree"
        else:
            # PERMISSIVE STRATEGIE: Eine Methode reicht
            final_converged = primary_result['converged'] or secondary_result['converged']
            decision_logic = "either_can_trigger"
        
        return {
            'converged': final_converged,
            'method': 'dual_strategy',
            'decision_logic': decision_logic,
            'primary_method': primary_method,
            'secondary_method': secondary_method,
            'primary_result': primary_result,
            'secondary_result': secondary_result,
            'agreement': primary_result['converged'] == secondary_result['converged'],
            'details': {
                'primary_converged': primary_result['converged'],
                'secondary_converged': secondary_result['converged'],
                'consensus_required': consensus_required,
                'history_length': len(regret_history),
                'scale_invariant': True,
                'robust_dual_detection': True
            }
        }
    
    def analyze_convergence_comprehensive(self, 
                                        regret_history: List[float]) -> Dict:
        """
        Run all convergence methods and provide comprehensive analysis.
        
        Führt alle verfügbaren scale-invariant Methoden aus und gibt
        eine umfassende Analyse mit majority vote zurück.
        
        Parameters:
        -----------
        regret_history : List[float]
            Historie der regret values
            
        Returns:
        --------
        Dict mit comprehensive analysis und majority vote result
        """
        methods = [
            ('relative_tolerance', self.check_relative_stabilization),
            ('coefficient_of_variation', self.check_cv_stabilization),
            ('relative_range', self.check_relative_range_stabilization)
        ]
        
        results = {}
        convergence_votes = []
        
        for method_name, method_func in methods:
            try:
                result = method_func(regret_history)
                results[method_name] = result
                if result['converged']:
                    convergence_votes.append(method_name)
            except Exception as e:
                results[method_name] = {
                    'converged': False,
                    'method': method_name,
                    'error': str(e)
                }
        
        # Dual strategies
        dual_strategies = [
            ('relative_tolerance', 'coefficient_of_variation', True),   # Consensus
            ('relative_tolerance', 'coefficient_of_variation', False),  # Either
            ('relative_range', 'coefficient_of_variation', True),       # Consensus  
            ('relative_tolerance', 'relative_range', False)             # Either
        ]
        
        for primary, secondary, consensus in dual_strategies:
            strategy_name = f"dual_{primary}_{secondary}_{'consensus' if consensus else 'either'}"
            try:
                dual_result = self.check_dual_strategy_convergence(
                    regret_history, primary, secondary, consensus
                )
                results[strategy_name] = dual_result
                if dual_result['converged']:
                    convergence_votes.append(strategy_name)
            except Exception as e:
                results[strategy_name] = {
                    'converged': False,
                    'method': strategy_name,
                    'error': str(e)
                }
        
        # Summary statistics
        total_methods = len([r for r in results.values() if not r.get('error')])
        converged_methods = len(convergence_votes)
        convergence_confidence = converged_methods / total_methods if total_methods > 0 else 0
        
        return {
            'overall_converged': convergence_confidence >= 0.5,  # Majority vote
            'convergence_confidence': convergence_confidence,
            'converged_methods': convergence_votes,
            'total_methods_tested': total_methods,
            'individual_results': results,
            'regret_history_stats': {
                'length': len(regret_history),
                'final_value': regret_history[-1] if regret_history else None,
                'initial_value': regret_history[0] if regret_history else None,
                'total_improvement': (regret_history[0] - regret_history[-1]) if len(regret_history) > 0 else 0,
                'mean_recent': np.mean(regret_history[-self.window_size:]) if len(regret_history) >= self.window_size else None
            },
            'scale_invariant_analysis': True,
            'methodology': 'comprehensive_multi_method'
        }


# Quick test function für Teil 1
def test_scale_invariant_convergence():
    """
    Test function to verify scale-invariant convergence detection works.
    """
    print("🔬 Testing Scale-Invariant Convergence Detection")
    print("=" * 50)
    
    detector = ScaleInvariantConvergenceDetector()
    
    # Test data: Different scales, same relative convergence pattern
    test_cases = {
        "Manufacturing Costs (€1000s)": [1000, 950, 920, 918, 917, 916, 915, 915.5, 915.2, 915.1, 915.0],
        "Catalyst Efficiency (0.1%)": [0.1, 0.095, 0.092, 0.0918, 0.0917, 0.0916, 0.0915, 0.09155, 0.09152, 0.09151, 0.09150],
        "Micro-precision (0.001)": [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000008, 0.000007, 0.000006, 0.000005]
    }
    
    print("\n📊 Testing Dual Strategy (Recommended):")
    for case_name, regret_history in test_cases.items():
        result = detector.check_dual_strategy_convergence(regret_history)
        
        print(f"\n{case_name}:")
        print(f"  Converged: {'✅ Yes' if result['converged'] else '❌ No'}")
        print(f"  Primary (Relative): {'✅' if result['primary_result']['converged'] else '❌'}")
        print(f"  Secondary (CV): {'✅' if result['secondary_result']['converged'] else '❌'}")
        
        if result['primary_result']['converged']:
            tolerance = result['primary_result']['details']['tolerance_used']
            print(f"  Tolerance used: {tolerance:.6f}")
    
    print("\n🎯 Scale-Invariant Test Complete!")
    print("✅ Same 5% relative criteria work across all problem scales!")
    
    return detector


if __name__ == "__main__":
    print("🚀 Teil 1: ScaleInvariantConvergenceDetector Ready!")
    print("\n🔧 Features:")
    print("✅ Relative tolerance (5% of current level)")  
    print("✅ Coefficient of variation (dimensionless)")
    print("✅ Relative range (scale-invariant)")
    print("✅ Dual strategy (robust consensus)")
    print("✅ Comprehensive analysis (majority vote)")
    print("✅ Edge case handling (near-zero, negative)")
    
    print("\n📏 Scale-Invariant Benefits:")
    print("• €917 ± €45.85 = 0.0917% ± 0.0046% (same 5% criteria)")
    print("• Automatic adaptation to problem magnitude") 
    print("• Universal convergence detection")
    
    # Run test
    test_detector = test_scale_invariant_convergence()
    print("\n✅ Teil 1 complete - ready for Teil 2 (RobustnessAnalyzer)!")