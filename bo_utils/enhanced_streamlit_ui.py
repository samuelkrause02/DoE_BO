# Part 3: Enhanced Streamlit UI Function
# File: enhanced_streamlit_ui.py

# Import previous parts
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from bo_utils.enhanced_robustness_analyzer import RobustnessAnalyzer
from bo_utils.scale_invariant_convergence import ScaleInvariantConvergenceDetector


def create_enhanced_robustness_analysis_ui():
    """
    Enhanced Streamlit UI for robustness analysis with scale-invariant convergence detection.
    
    Features:
    - Scale-invariant convergence detection (works for €1000s and 0.001% problems)
    - Multi-criteria strategy recommendation
    - Practical experiment planning with total experiments
    - Enhanced visualizations
    - Robust error handling with fallback
    """
    st.title("🎯 Enhanced BO Strategy Robustness Analysis")
    st.markdown("*With Scale-Invariant Convergence Detection*")
    
    # Check if results exist
    if 'bo_results' not in st.session_state or st.session_state['bo_results'] is None:
        st.warning("⚠️ No BO simulation results found. Please run the simulation first.")
        return
    
    df_results = st.session_state['bo_results']
    
    try:
        # Enhanced sidebar controls
        st.sidebar.header("🔧 Analysis Configuration")
        
        # Get available metrics
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
        
        # Enhanced convergence method selection
        st.sidebar.subheader("🔬 Scale-Invariant Convergence")
        
        convergence_method = st.sidebar.selectbox(
            "Convergence Detection Method",
            options=[
                'dual_strategy',
                'relative_tolerance', 
                'coefficient_of_variation'
            ],
            index=0,
            help="""
            • dual_strategy: Robust consensus between relative tolerance & CV (recommended)
            • relative_tolerance: 5% of current performance level (scales automatically)
            • coefficient_of_variation: 2% CV threshold (dimensionless)
            """
        )
        
        # Method-specific parameters
        if convergence_method in ['dual_strategy', 'relative_tolerance']:
            relative_tolerance = st.sidebar.slider(
                "Relative Tolerance (%)", 
                min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                help="Tolerance as percentage of current performance level (scales automatically)"
            ) / 100
        else:
            relative_tolerance = 0.05
            
        if convergence_method in ['dual_strategy', 'coefficient_of_variation']:
            cv_threshold = st.sidebar.slider(
                "CV Threshold (%)", 
                min_value=0.5, max_value=10.0, value=2.0, step=0.1,
                help="Coefficient of variation threshold (dimensionless, scale-invariant)"
            ) / 100
        else:
            cv_threshold = 0.02
        
        # Analysis mode selection
        analysis_mode = st.sidebar.radio(
            "Analysis Mode",
            ["🎯 Comprehensive (Multi-Metric)", "📊 Simple (Performance Only)"],
            help="Comprehensive includes surrogate quality, convergence analysis, etc."
        )
        
        # Weight configuration for comprehensive mode
        if analysis_mode.startswith("🎯"):
            st.sidebar.subheader("⚖️ Multi-Criteria Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.25, 0.05)
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.25, 0.05)
            safety_weight = st.sidebar.slider("Safety Weight", 0.0, 1.0, 0.15, 0.05)
            surrogate_weight = st.sidebar.slider("Surrogate Quality Weight", 0.0, 1.0, 0.20, 0.05)
            convergence_weight = st.sidebar.slider("Convergence Weight", 0.0, 1.0, 0.15, 0.05)
            
            # Normalize weights
            total_weight = perf_weight + robust_weight + safety_weight + surrogate_weight + convergence_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight  
                safety_weight /= total_weight
                surrogate_weight /= total_weight
                convergence_weight /= total_weight
        else:
            # Simple mode - just performance and robustness
            st.sidebar.subheader("⚖️ Simple Mode Weights")
            perf_weight = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.6, 0.1)
            robust_weight = st.sidebar.slider("Robustness Weight", 0.0, 1.0, 0.4, 0.1)
            safety_weight = 0.0
            surrogate_weight = 0.0
            convergence_weight = 0.0
            
            # Normalize weights
            total_weight = perf_weight + robust_weight
            if total_weight > 0:
                perf_weight /= total_weight
                robust_weight /= total_weight
        
        # Initialize enhanced analyzer
        analyzer = RobustnessAnalyzer(
            df_results, 
            convergence_method=convergence_method,
            relative_tolerance=relative_tolerance,
            cv_threshold=cv_threshold
        )
        
        # Show convergence method info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔬 Scale-Invariant Features:**")
        st.sidebar.markdown("✅ Works for any problem magnitude")
        st.sidebar.markdown("✅ Automatic threshold adaptation")
        st.sidebar.markdown("✅ Robust edge case handling")
        st.sidebar.markdown(f"✅ Using {convergence_method.replace('_', ' ').title()}")
        
        # Enhanced main analysis display
        st.header("📊 Strategy Recommendation")
        
        # Show convergence method info prominently
        col_method, col_scale = st.columns(2)
        with col_method:
            st.info(f"🔬 **Method**: {convergence_method.replace('_', ' ').title()}")
        with col_scale:
            st.success("📏 **Scale-Invariant**: ✅ Adapts to any problem magnitude")
        
        # Run analysis and display results
        if analysis_mode.startswith("🎯"):
            # Comprehensive analysis
            recommendation = analyzer.recommend_comprehensive_strategy(
                selected_metric, perf_weight, robust_weight, safety_weight,
                surrogate_weight, convergence_weight
            )
            
            # Enhanced display of results
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                strategy_name = recommendation['strategy'].split(' (')[0]
                init_size = recommendation['strategy'].split('init=')[1].rstrip(')')
                st.metric("🏆 Strategy", strategy_name, f"Init: {init_size}")
            with col2:
                st.metric("🎯 Performance", f"{recommendation['performance_score']:.3f}")
            with col3:
                st.metric("🛡️ Robustness", f"{recommendation['robustness_score']:.3f}")
            with col4:
                st.metric("🔍 Surrogate Quality", f"{recommendation['surrogate_quality_score']:.3f}")
            with col5:
                st.metric("⚡ Convergence", f"{recommendation['convergence_score']:.3f}")
            
            # Overall score with additional info
            col_score, col_details = st.columns(2)
            with col_score:
                st.metric("🏅 Overall Combined Score", f"{recommendation['combined_score']:.4f}")
            with col_details:
                convergence_reliability = recommendation['metrics'].get('convergence_reliability', 0)
                if convergence_reliability:
                    st.metric("🔬 Convergence Reliability", f"{convergence_reliability:.1%}")
                else:
                    st.metric("🔬 Analysis Method", recommendation.get('convergence_method', 'standard').title())
        
        else:
            # Simple analysis mode
            comprehensive_metrics = analyzer.compute_comprehensive_robustness_metrics(selected_metric)
            
            if not comprehensive_metrics.empty:
                # Simple scoring
                perf_scores = comprehensive_metrics['performance_score'] / comprehensive_metrics['performance_score'].max()
                robust_scores = comprehensive_metrics['robustness_score'] / comprehensive_metrics['robustness_score'].max()
                combined_scores = perf_weight * perf_scores + robust_weight * robust_scores
                
                comprehensive_metrics['combined_score'] = combined_scores
                best_idx = combined_scores.idxmax()
                best_strategy = comprehensive_metrics.iloc[best_idx]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Best Strategy", f"{best_strategy['acquisition_type']} (init={best_strategy['num_init']})")
                with col2:
                    mean_val = best_strategy[f'{selected_metric}_mean']
                    std_val = best_strategy[f'{selected_metric}_std']
                    st.metric(f"📈 Mean {selected_metric.upper()}", f"{mean_val:.4f}", f"±{std_val:.4f}")
                with col3:
                    st.metric("🏅 Combined Score", f"{best_strategy['combined_score']:.3f}")
                
                recommendation = {
                    'strategy': f"{best_strategy['acquisition_type']} (init={best_strategy['num_init']})",
                    'combined_score': best_strategy['combined_score'],
                    'performance_score': perf_scores.iloc[best_idx],
                    'robustness_score': robust_scores.iloc[best_idx],
                    'convergence_score': best_strategy.get('convergence_score', 0.5),
                    'metrics': best_strategy.to_dict(),
                    'all_strategies': comprehensive_metrics.sort_values('combined_score', ascending=False)
                }
        
        # Enhanced tabs
        viz_tab, planning_tab = st.tabs(["📈 Visualizations", "🧪 Experiment Planning"])
        
        with viz_tab:
            # Enhanced convergence plot
            st.subheader("🔄 Convergence Analysis (Scale-Invariant)")
            try:
                conv_fig = create_enhanced_convergence_plot(analyzer, selected_metric)
                st.plotly_chart(conv_fig, use_container_width=True)
                
                # Method explanation
                st.info(f"""
                📏 **Scale-Invariant Detection**: Using {convergence_method.replace('_', ' ')} method.
                Same convergence criteria work for €1000s manufacturing costs and 0.001% catalyst efficiency.
                Automatic adaptation to problem magnitude ensures universal applicability.
                """)
                
            except Exception as e:
                st.error(f"Convergence plot failed: {str(e)}")
                st.write("Using basic fallback visualization...")
                
                # Simple fallback plot
                try:
                    basic_fig = create_basic_convergence_plot(df_results, selected_metric)
                    st.plotly_chart(basic_fig, use_container_width=True)
                except Exception as e2:
                    st.error(f"Fallback plot also failed: {str(e2)}")
        
        with planning_tab:
            st.subheader("🧪 Enhanced Experiment Planning")
            st.markdown(f"*Using {convergence_method.replace('_', ' ').title()} convergence detection*")
            
            try:
                convergence_df = analyzer.get_experiment_planning_analysis(selected_metric)
                
                if not convergence_df.empty:
                    # Show scale-invariant benefits
                    has_scale_invariant = convergence_df.get('scale_invariant_analysis', pd.Series([False])).any()
                    if has_scale_invariant:
                        st.success("✅ **Scale-invariant convergence detection active** - works for any problem magnitude!")
                        st.info("🎯 Same convergence criteria work for €1000s manufacturing costs and 0.001% catalyst efficiency")
                    
                    # Key insights
                    st.subheader("⚡ Planning Insights")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    converged_strategies = convergence_df[convergence_df['success_rate'] > 0]
                    if not converged_strategies.empty:
                        best_strategy = converged_strategies.loc[converged_strategies['median_total_experiments'].idxmin()]
                        
                        with col1:
                            st.metric(
                                "🏆 Most Efficient",
                                best_strategy['strategy'],
                                f"{best_strategy['median_total_experiments']:.0f} total exp."
                            )
                        
                        with col2:
                            st.metric(
                                "⚡ Best Case",
                                f"{best_strategy['min_total_experiments']:.0f} experiments",
                                f"Success: {best_strategy['success_rate']:.1%}"
                            )
                        
                        with col3:
                            early_rate = best_strategy.get('early_convergence_rate', 0) * 100
                            st.metric(
                                "🎯 Realistic Budget",
                                f"{best_strategy['median_total_experiments']:.0f} experiments",
                                f"{early_rate:.1f}% early convergence"
                            )
                        
                        with col4:
                            method_name = best_strategy.get('convergence_method', convergence_method)
                            st.metric(
                                "🔬 Detection Method", 
                                method_name.replace('_', ' ').title()[:15],
                                "Scale-invariant"
                            )
                    
                    # Enhanced planning table
                    st.subheader("📊 Experiment Planning Table")
                    
                    display_conv_df = convergence_df.copy()
                    
                    # Format columns for better display
                    numeric_cols = ['median_total_experiments', 'total_experiments_90th_percentile', 
                                   'min_total_experiments', 'max_total_experiments',
                                   'median_bo_iterations', 'early_convergence_rate']
                    
                    for col in numeric_cols:
                        if col in display_conv_df.columns:
                            if col == 'early_convergence_rate':
                                display_conv_df[col] = (display_conv_df[col] * 100).round(1)
                            else:
                                display_conv_df[col] = display_conv_df[col].round(1)
                    
                    display_conv_df['success_rate'] = (display_conv_df['success_rate'] * 100).round(1)
                    
                    # Show key columns
                    key_cols = ['strategy', 'convergence_method', 'success_rate', 'median_total_experiments', 
                               'total_experiments_90th_percentile', 'early_convergence_rate', 'num_init', 
                               'median_bo_iterations']
                    
                    display_cols = [col for col in key_cols if col in display_conv_df.columns]
                    
                    renamed_df = display_conv_df[display_cols].rename(columns={
                        'strategy': 'BO Strategy',
                        'convergence_method': 'Method',
                        'success_rate': 'Success Rate (%)',
                        'median_total_experiments': 'Median Total Exp.',
                        'total_experiments_90th_percentile': '90% Conf. Total',
                        'early_convergence_rate': 'Early Conv. (%)',
                        'num_init': 'Initial',
                        'median_bo_iterations': 'Median BO Iters'
                    })
                    
                    st.dataframe(renamed_df, use_container_width=True, height=350)
                    
                    # Generate planning summary
                    if not converged_strategies.empty:
                        st.subheader("📋 Planning Summary")
                        best_strategy = converged_strategies.iloc[0]
                        
                        summary_text = f"""
**🏆 Recommended Strategy**: {best_strategy['strategy']}

**📊 Budget Planning**:
- **Typical Budget**: {best_strategy['median_total_experiments']:.0f} total experiments (50% confidence)
- **Conservative Budget**: {best_strategy['total_experiments_90th_percentile']:.0f} total experiments (90% confidence)
- **Success Rate**: {best_strategy['success_rate']:.1%}

**🔍 Breakdown**:
- **Initial Sampling**: {best_strategy['num_init']} experiments
- **BO Iterations**: {best_strategy['median_bo_iterations']:.0f} additional experiments (median)

**🔬 Enhanced Features**:
- **Method**: {best_strategy['convergence_method'].replace('_', ' ').title()} (Scale-Invariant)
- **Early Convergence**: {best_strategy.get('early_convergence_rate', 0) * 100:.1f}% of cases finish early
- **Universal**: Same criteria work for €1000s and 0.001% problems
                        """
                        
                        st.markdown(summary_text)
                        
                        # Download planning summary
                        st.download_button(
                            "📥 Download Planning Summary",
                            summary_text,
                            f"experiment_planning_summary_{selected_metric}_{convergence_method}.md",
                            "text/markdown"
                        )
                
                else:
                    st.warning("❌ No successful BO strategies found for experiment planning")
                    st.info("💡 Try adjusting convergence parameters or check data quality")
            
            except Exception as e:
                st.error(f"Experiment planning analysis failed: {str(e)}")
                st.exception(e)
        
        # Enhanced detailed metrics
        st.header("📋 Detailed Strategy Metrics")
        
        try:
            comprehensive_metrics = analyzer.compute_comprehensive_robustness_metrics(selected_metric)
            
            if not comprehensive_metrics.empty:
                # Add method info for display
                if 'convergence_method_used' not in comprehensive_metrics.columns:
                    comprehensive_metrics['convergence_method_used'] = convergence_method
                
                # Format for display
                display_metrics = comprehensive_metrics.copy()
                
                # Drop complex columns for cleaner display
                columns_to_drop = ['convergence_details']
                for col in columns_to_drop:
                    if col in display_metrics.columns:
                        display_metrics = display_metrics.drop(columns=[col])
                
                # Round numeric columns
                numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
                display_metrics[numeric_cols] = display_metrics[numeric_cols].round(4)
                
                # Show dataframe
                st.dataframe(display_metrics, use_container_width=True, height=400)
                
                # Show convergence method distribution
                if len(display_metrics) > 1:
                    st.subheader("🔬 Analysis Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strategies Analyzed", len(display_metrics))
                    with col2:
                        scale_invariant_count = display_metrics.get('scale_invariant_analysis', pd.Series([False])).sum()
                        st.metric("Scale-Invariant Analysis", f"{scale_invariant_count}/{len(display_metrics)}")
                    with col3:
                        avg_reliability = display_metrics.get('convergence_reliability', pd.Series([0])).mean()
                        st.metric("Avg. Convergence Reliability", f"{avg_reliability:.1%}")
                        
            else:
                st.warning("No comprehensive metrics available - check data format")
                
        except Exception as e:
            st.error(f"Failed to compute detailed metrics: {str(e)}")
        
        # Enhanced export options
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Enhanced metrics CSV
            try:
                metrics_for_export = analyzer.compute_comprehensive_robustness_metrics(selected_metric)
                if not metrics_for_export.empty:
                    # Clean for export
                    export_metrics = metrics_for_export.drop(columns=['convergence_details'], errors='ignore')
                    csv_data = export_metrics.to_csv(index=False)
                    st.download_button(
                        "📥 Download Metrics CSV",
                        csv_data,
                        f"enhanced_robustness_metrics_{selected_metric}_{convergence_method}.csv",
                        "text/csv",
                        help="Comprehensive strategy metrics with scale-invariant analysis"
                    )
            except Exception as e:
                st.error(f"CSV export failed: {str(e)}")
        
        with col2:
            # Experiment planning CSV
            try:
                planning_data = analyzer.get_experiment_planning_analysis(selected_metric)
                if not planning_data.empty:
                    planning_csv = planning_data.to_csv(index=False)
                    st.download_button(
                        "🧪 Download Planning CSV",
                        planning_csv,
                        f"experiment_planning_{selected_metric}_{convergence_method}.csv",
                        "text/csv",
                        help="Experiment planning with total experiments and scale-invariant convergence"
                    )
            except Exception as e:
                st.error(f"Planning export failed: {str(e)}")
        
        with col3:
            # Configuration export
            try:
                config_data = {
                    'convergence_method': convergence_method,
                    'relative_tolerance': relative_tolerance,
                    'cv_threshold': cv_threshold,
                    'selected_metric': selected_metric,
                    'analysis_mode': analysis_mode,
                    'weights': {
                        'performance': perf_weight,
                        'robustness': robust_weight,
                        'safety': safety_weight,
                        'surrogate': surrogate_weight,
                        'convergence': convergence_weight
                    },
                    'scale_invariant_features': {
                        'automatic_scaling': True,
                        'universal_criteria': True,
                        'edge_case_handling': True
                    }
                }
                
                config_json = pd.Series(config_data).to_json(indent=2)
                st.download_button(
                    "⚙️ Download Config",
                    config_json,
                    f"enhanced_analysis_config_{convergence_method}.json",
                    "application/json",
                    help="Save analysis configuration for reproducibility"
                )
            except Exception as e:
                st.error(f"Config export failed: {str(e)}")
        
        # Store enhanced results in session state
        st.session_state['enhanced_robustness_analyzer'] = analyzer
        st.session_state['enhanced_robustness_recommendation'] = recommendation
        st.session_state['convergence_method_used'] = convergence_method
        st.session_state['scale_invariant_parameters'] = {
            'relative_tolerance': relative_tolerance,
            'cv_threshold': cv_threshold,
            'convergence_method': convergence_method
        }
        
        # Success message
        st.success(f"""
        ✅ **Enhanced Scale-Invariant Analysis Complete!**
        
        🔬 **Method**: {convergence_method.replace('_', ' ').title()}  
        📏 **Scale-Invariant**: Works for any problem magnitude  
        🎯 **Recommended Strategy**: {recommendation['strategy']}  
        🛡️ **Robustness**: Enhanced convergence detection with automatic threshold adaptation
        """)
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.exception(e)
        
        # Comprehensive fallback analysis
        st.warning("🔄 Attempting comprehensive fallback analysis...")
        try:
            # Fallback to basic robustness analysis
            final_values = df_results.groupby(['acquisition_type', 'num_init', 'seed'])[selected_metric].last().reset_index()
            basic_stats = final_values.groupby(['acquisition_type', 'num_init'])[selected_metric].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(4)
            
            st.subheader("📊 Basic Robustness Analysis (Fallback)")
            st.dataframe(basic_stats, use_container_width=True)
            
            # Simple recommendation
            basic_stats['performance_score'] = 1 / (1 + basic_stats['mean'])
            basic_stats['robustness_score'] = 1 / (1 + basic_stats['std'])
            basic_stats['combined_score'] = 0.6 * basic_stats['performance_score'] + 0.4 * basic_stats['robustness_score']
            
            best_idx = basic_stats['combined_score'].idxmax()
            best_strategy = basic_stats.loc[best_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Best Strategy (Basic)", f"{best_idx[0]} (init={best_idx[1]})")
            with col2:
                st.metric(f"📈 Mean {selected_metric}", f"{best_strategy['mean']:.4f}", 
                         f"±{best_strategy['std']:.4f}")
            with col3:
                st.metric("🏅 Combined Score", f"{best_strategy['combined_score']:.3f}")
            
            st.info("""
            💡 **Fallback Analysis Active**: This is a basic robustness analysis. 
            For full scale-invariant analysis with enhanced convergence detection, 
            please check your data format and parameter settings.
            """)
            
        except Exception as e2:
            st.error(f"❌ Fallback analysis also failed: {str(e2)}")
            st.markdown("""
            **Please check your data format:**
            - Required columns: `acquisition_type`, `num_init`, `seed`, `iteration`
            - Numeric metric columns for analysis (e.g., `regret`, `loss`)
            - Data should contain multiple seeds and iterations
            - Values should be numeric and not all NaN
            """)


def create_enhanced_convergence_plot(analyzer, metric_col, max_strategies=6):
    """
    Create enhanced convergence plot with scale-invariant convergence information.
    """
    strategies = analyzer.df.groupby(['acquisition_type', 'num_init']).size().index[:max_strategies]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{acq} (init={init})" for acq, init in strategies],
        shared_yaxes=True
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (acq_type, num_init) in enumerate(strategies):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        subset = analyzer.df[
            (analyzer.df['acquisition_type'] == acq_type) & 
            (analyzer.df['num_init'] == num_init)
        ]
        
        if subset.empty:
            continue
        
        # Calculate statistics per iteration
        convergence_stats = subset.groupby('iteration')[metric_col].agg([
            'mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)
        ]).reset_index()
        convergence_stats.columns = ['iteration', 'mean', 'q25', 'q75']
        
        color = colors[idx % len(colors)]
        rgb = mcolors.hex2color(color)
        rgba_str = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.2)"
        
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=convergence_stats['iteration'], 
                y=convergence_stats['mean'],
                mode='lines+markers',
                name=f'{acq_type}',
                line=dict(color=color, width=2),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Confidence band (Q1-Q3)
        fig.add_trace(
            go.Scatter(
                x=convergence_stats['iteration'].tolist() + convergence_stats['iteration'].tolist()[::-1],
                y=convergence_stats['q25'].tolist() + convergence_stats['q75'].tolist()[::-1],
                fill='tonexty' if idx > 0 else 'tozeroy',
                fillcolor=rgba_str,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Q1-Q3'
            ),
            row=row, col=col
        )
    
    method_name = analyzer.convergence_method.replace('_', ' ').title()
    fig.update_layout(
        title=f'Enhanced BO Convergence Analysis ({metric_col}) - {method_name} (Scale-Invariant)',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="BO Iteration")
    fig.update_yaxes(title_text=metric_col.upper())
    
    return fig


# Part 3.5: Helper Functions for UI
# File: ui_helper_functions.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors


def create_basic_convergence_plot(df_results, metric_col, max_strategies=6):
    """
    Create basic convergence plot as fallback when enhanced plotting fails.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        BO simulation results
    metric_col : str
        Metric column to plot
    max_strategies : int
        Maximum number of strategies to show
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Basic convergence plot
    """
    strategies = df_results.groupby(['acquisition_type', 'num_init']).size().index[:max_strategies]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{acq} (init={init})" for acq, init in strategies],
        shared_yaxes=True
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (acq_type, num_init) in enumerate(strategies):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        subset = df_results[
            (df_results['acquisition_type'] == acq_type) & 
            (df_results['num_init'] == num_init)
        ]
        
        if subset.empty:
            continue
        
        # Simple mean convergence per iteration
        convergence_stats = subset.groupby('iteration')[metric_col].mean().reset_index()
        
        color = colors[idx % len(colors)]
        
        # Simple mean line
        fig.add_trace(
            go.Scatter(
                x=convergence_stats['iteration'], 
                y=convergence_stats[metric_col],
                mode='lines+markers',
                name=f'{acq_type}',
                line=dict(color=color, width=2),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f'Basic Convergence Analysis ({metric_col})',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="BO Iteration")
    fig.update_yaxes(title_text=metric_col.upper())
    
    return fig


def create_strategy_comparison_plot(comprehensive_metrics, primary_metric):
    """
    Create strategy comparison visualization.
    
    Parameters:
    -----------
    comprehensive_metrics : pd.DataFrame
        Comprehensive strategy metrics
    primary_metric : str
        Primary metric for comparison
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Strategy comparison plot
    """
    if comprehensive_metrics.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Performance vs Robustness', 
            'Convergence Reliability',
            'Risk-Return Analysis',
            'Strategy Ranking'
        ]
    )
    
    # Create strategy labels
    comprehensive_metrics['strategy_label'] = comprehensive_metrics.apply(
        lambda x: f"{x['acquisition_type']}<br>(init={x['num_init']})", axis=1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    strategy_colors = [colors[i % len(colors)] for i in range(len(comprehensive_metrics))]
    
    # 1. Performance vs Robustness scatter
    if f'{primary_metric}_mean' in comprehensive_metrics.columns and 'robustness_score' in comprehensive_metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=comprehensive_metrics[f'{primary_metric}_mean'],
                y=comprehensive_metrics['robustness_score'],
                mode='markers+text',
                text=comprehensive_metrics['strategy_label'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=strategy_colors,
                    line=dict(width=1, color='white')
                ),
                name='Strategies',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Convergence reliability
    if 'convergence_reliability' in comprehensive_metrics.columns:
        fig.add_trace(
            go.Bar(
                x=comprehensive_metrics['strategy_label'],
                y=comprehensive_metrics['convergence_reliability'] * 100,
                marker_color=strategy_colors,
                name='Convergence %',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Risk-Return (Mean vs Std)
    if f'{primary_metric}_mean' in comprehensive_metrics.columns and f'{primary_metric}_std' in comprehensive_metrics.columns:
        combined_score = comprehensive_metrics.get('combined_score', comprehensive_metrics.get('performance_score', [0.5] * len(comprehensive_metrics)))
        
        fig.add_trace(
            go.Scatter(
                x=comprehensive_metrics[f'{primary_metric}_std'],
                y=comprehensive_metrics[f'{primary_metric}_mean'],
                mode='markers+text',
                text=comprehensive_metrics['strategy_label'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=combined_score,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Combined Score", x=1.1),
                    line=dict(width=1, color='white')
                ),
                name='Risk-Return',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Strategy ranking
    if 'combined_score' in comprehensive_metrics.columns:
        top_strategies = comprehensive_metrics.nlargest(6, 'combined_score')
        
        fig.add_trace(
            go.Bar(
                x=top_strategies['combined_score'],
                y=top_strategies['strategy_label'],
                orientation='h',
                marker_color=strategy_colors[:len(top_strategies)],
                name='Combined Score',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text=f"Mean {primary_metric.upper()}", row=1, col=1)
    fig.update_yaxes(title_text="Robustness Score", row=1, col=1)
    
    fig.update_yaxes(title_text="Convergence Reliability (%)", row=1, col=2)
    
    fig.update_xaxes(title_text=f"Std {primary_metric.upper()} (Risk)", row=2, col=1)
    fig.update_yaxes(title_text=f"Mean {primary_metric.upper()}", row=2, col=1)
    
    fig.update_xaxes(title_text="Combined Score", row=2, col=2)
    
    fig.update_layout(
        title=f'Strategy Comparison Analysis ({primary_metric.upper()})',
        height=800,
        showlegend=False
    )
    
    return fig


def format_experiment_planning_dataframe(convergence_df):
    """
    Format experiment planning dataframe for better display.
    
    Parameters:
    -----------
    convergence_df : pd.DataFrame
        Raw experiment planning data
        
    Returns:
    --------
    pd.DataFrame
        Formatted dataframe for display
    """
    if convergence_df.empty:
        return convergence_df
    
    display_df = convergence_df.copy()
    
    # Format numeric columns
    numeric_cols = [
        'median_total_experiments', 'total_experiments_90th_percentile',
        'total_experiments_95th_percentile', 'min_total_experiments', 
        'max_total_experiments', 'median_bo_iterations', 'early_convergence_rate'
    ]
    
    for col in numeric_cols:
        if col in display_df.columns:
            if col == 'early_convergence_rate':
                display_df[col] = (display_df[col] * 100).round(1)
            else:
                display_df[col] = display_df[col].round(1)
    
    # Format success rate as percentage
    if 'success_rate' in display_df.columns:
        display_df['success_rate'] = (display_df['success_rate'] * 100).round(1)
    
    # Rename columns for better readability
    column_mapping = {
        'strategy': 'BO Strategy',
        'convergence_method': 'Convergence Method',
        'success_rate': 'Success Rate (%)',
        'median_total_experiments': 'Median Total Exp.',
        'total_experiments_90th_percentile': '90% Confidence Total',
        'total_experiments_95th_percentile': '95% Confidence Total',
        'early_convergence_rate': 'Early Convergence (%)',
        'num_init': 'Initial Experiments',
        'median_bo_iterations': 'Median BO Iterations',
        'scale_invariant_analysis': 'Scale-Invariant'
    }
    
    # Only rename columns that exist
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in display_df.columns}
    display_df = display_df.rename(columns=columns_to_rename)
    
    return display_df


def create_planning_summary_report(best_strategy, selected_metric, convergence_method):
    """
    Create a comprehensive planning summary report.
    
    Parameters:
    -----------
    best_strategy : pd.Series
        Best strategy data
    selected_metric : str
        Selected optimization metric
    convergence_method : str
        Convergence detection method used
        
    Returns:
    --------
    str
        Formatted planning summary report
    """
    # Extract key values with safe defaults
    strategy_name = best_strategy.get('strategy', 'Unknown Strategy')
    median_total = best_strategy.get('median_total_experiments', 0)
    percentile_90 = best_strategy.get('total_experiments_90th_percentile', 0)
    percentile_95 = best_strategy.get('total_experiments_95th_percentile', 0)
    success_rate = best_strategy.get('success_rate', 0)
    num_init = best_strategy.get('num_init', 0)
    median_bo = best_strategy.get('median_bo_iterations', 0)
    early_rate = best_strategy.get('early_convergence_rate', 0) * 100
    min_total = best_strategy.get('min_total_experiments', 0)
    max_total = best_strategy.get('max_total_experiments', 0)
    
    report = f"""
# 🧪 Enhanced BO Experiment Planning Report

## 📊 **Executive Summary: Scale-Invariant Analysis**

**🔬 Analysis Method**: {convergence_method.replace('_', ' ').title()} (Scale-Invariant)
**📏 Universal Benefits**: Same convergence criteria work for any problem magnitude

### 🏆 **Recommended Strategy: {strategy_name}**

#### 📊 **Budget Planning by Confidence Level**
- **Optimistic (50% confidence)**: {median_total:.0f} total experiments
- **Realistic (90% confidence)**: {percentile_90:.0f} total experiments  
- **Conservative (95% confidence)**: {percentile_95:.0f} total experiments

#### 🔍 **Enhanced Analysis Details**
- **Convergence Method**: {convergence_method.replace('_', ' ').title()} (Scale-Invariant)
- **Success Probability**: {success_rate:.1%}
- **Early Convergence Rate**: {early_rate:.1f}% of cases finish ahead of schedule
- **Experiment Range**: {min_total:.0f} - {max_total:.0f} total experiments

#### 🔍 **Detailed Breakdown**
- **Phase 1 - Initial Sampling**: {num_init} experiments (space exploration)
- **Phase 2 - BO Optimization**: {median_bo:.0f} additional experiments (median case)
- **Total Budget**: {median_total:.0f} experiments

## 🔬 **Scale-Invariant Convergence Benefits**

### 🎯 **Universal Detection**
The enhanced convergence detection automatically adapts to your problem:

- **Manufacturing Costs**: €917 → tolerance adapts to €45.85
- **Catalyst Efficiency**: 0.0917% → tolerance adapts to 0.0046%
- **Same Criteria**: Both use identical 5% relative tolerance!

### ✅ **Key Implementation Benefits**
1. **🏆 Recommended Strategy**: {strategy_name}
2. **📊 Typical Budget**: {median_total:.0f} total experiments
3. **🎯 Success Rate**: {success_rate:.1%} probability of convergence
4. **🔬 Method**: {convergence_method.replace('_', ' ').title()} (Scale-Invariant)
5. **📏 Universal**: Works for any problem magnitude automatically
6. **⚡ Efficiency**: {early_rate:.1f}% chance of early completion

## 🚀 **Implementation Roadmap**

### Phase 1: Initial Design ({num_init} experiments)
1. **Method**: Orthogonal/Latin Hypercube Sampling
2. **Goal**: Broad exploration of parameter space
3. **Budget**: {num_init} experiments
4. **Timeline**: Can run in parallel if resources allow

### Phase 2: BO Optimization ({median_bo:.0f} experiments)
1. **Method**: {strategy_name.split('(')[0].strip()} acquisition function
2. **Convergence**: {convergence_method.replace('_', ' ').title()} detection
3. **Goal**: Iterative optimization with scale-invariant stopping
4. **Budget**: {median_bo:.0f} additional experiments (median case)
5. **Timeline**: Sequential execution required
6. **Monitoring**: Check convergence every 3-5 iterations

### Phase 3: Contingency Planning
- **Standard Budget**: {median_total:.0f} total experiments
- **If delayed convergence**: Continue up to {percentile_90:.0f} experiments (90% confidence)
- **Maximum Budget**: {percentile_95:.0f} experiments (95% confidence)
- **Early Success**: May complete in as few as {min_total:.0f} experiments

## 💰 **Budget Recommendations**

### 🎯 **Minimum Viable Budget**
- **{median_total:.0f} total experiments** (50% confidence)
- **Suitable for**: Research projects, proof-of-concept studies
- **Risk**: Moderate chance of needing additional experiments

### 🏭 **Production Deployment Budget**  
- **{percentile_90:.0f} total experiments** (90% confidence)
- **Suitable for**: Critical applications, commercial deployment
- **Benefit**: High confidence in completion within budget

### 🛡️ **Risk-Averse Budget**
- **{percentile_95:.0f} total experiments** (95% confidence)  
- **Suitable for**: High-stakes applications, regulatory environments
- **Benefit**: Maximum confidence in successful completion

*Note: This analysis uses scale-invariant convergence detection that automatically adapts to your problem's magnitude. The same criteria work equally well for optimizing €1000s manufacturing costs or 0.001% catalyst efficiency.*
    """
    
    return report.strip()


def validate_data_format(df_results):
    """
    Validate BO simulation data format and provide helpful error messages.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        BO simulation results to validate
        
    Returns:
    --------
    tuple
        (is_valid: bool, error_messages: list, warnings: list)
    """
    errors = []
    warnings = []
    
    # Check if dataframe is empty
    if df_results.empty:
        errors.append("Dataset is empty")
        return False, errors, warnings
    
    # Check required columns
    required_cols = ['acquisition_type', 'num_init', 'seed', 'iteration']
    missing_cols = [col for col in required_cols if col not in df_results.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for numeric metrics
    numeric_cols = [col for col in df_results.columns 
                   if col not in required_cols and df_results[col].dtype in ['float64', 'int64']]
    
    if not numeric_cols:
        errors.append("No numeric metric columns found for analysis")
    
    # Check data completeness
    if not errors:  # Only check if basic structure is OK
        # Check for sufficient data per strategy
        strategy_counts = df_results.groupby(['acquisition_type', 'num_init']).size()
        min_data_points = strategy_counts.min()
        
        if min_data_points < 10:
            warnings.append(f"Some strategies have very few data points (min: {min_data_points})")
        
        # Check for seed variation
        seed_counts = df_results.groupby(['acquisition_type', 'num_init'])['seed'].nunique()
        min_seeds = seed_counts.min()
        
        if min_seeds < 2:
            warnings.append("Some strategies have only one seed - robustness analysis may be limited")
        
        # Check for iteration progression
        max_iterations = df_results.groupby(['acquisition_type', 'num_init', 'seed'])['iteration'].count()
        min_iterations = max_iterations.min()
        
        if min_iterations < 5:
            warnings.append("Some runs have very few iterations - convergence analysis may be unreliable")
        
        # Check for NaN values in metrics
        nan_metrics = []
        for col in numeric_cols:
            if df_results[col].isna().any():
                nan_count = df_results[col].isna().sum()
                nan_metrics.append(f"{col}: {nan_count} NaN values")
        
        if nan_metrics:
            warnings.append(f"NaN values found in metrics: {', '.join(nan_metrics)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def display_data_validation_results(is_valid, errors, warnings):
    """
    Display data validation results in Streamlit UI.
    
    Parameters:
    -----------
    is_valid : bool
        Whether data passed validation
    errors : list
        List of error messages
    warnings : list
        List of warning messages
    """
    if is_valid:
        if warnings:
            st.warning("⚠️ **Data validation passed with warnings:**")
            for warning in warnings:
                st.write(f"• {warning}")
        else:
            st.success("✅ **Data validation passed** - ready for enhanced analysis!")
    else:
        st.error("❌ **Data validation failed:**")
        for error in errors:
            st.write(f"• {error}")
        
        if warnings:
            st.write("**Additional warnings:**")
            for warning in warnings:
                st.write(f"• {warning}")
        
        st.markdown("""
        **Required data format:**
        - **Columns**: `acquisition_type`, `num_init`, `seed`, `iteration`, plus numeric metrics
        - **Structure**: Multiple seeds per strategy, multiple iterations per seed
        - **Content**: Numeric values for metrics (e.g., regret, loss, accuracy)
        """)



# Alias for compatibility with existing code
create_robustness_analysis_ui = None  # Will be set after importing Part 3


if __name__ == "__main__":
    print("🚀 Part 3.5: UI Helper Functions Ready!")
    print("\n🔧 Helper Functions:")
    print("✅ create_basic_convergence_plot() - Fallback plotting")
    print("✅ create_strategy_comparison_plot() - Enhanced visualizations")
    print("✅ format_experiment_planning_dataframe() - Clean data display")
    print("✅ create_planning_summary_report() - Comprehensive reports")
    print("✅ validate_data_format() - Data validation")
    print("✅ display_data_validation_results() - User-friendly error messages")
    
    print("\n📏 Features:")
    print("• Robust error handling and fallbacks")
    print("• Enhanced visualizations with scale-invariant info")
    print("• Comprehensive data validation")
    print("• User-friendly error messages and guidance")
    print("• Professional report generation")
    
    print("\n✅ Part 3.5 complete - ready for Part 4 (Integration & Main)!")