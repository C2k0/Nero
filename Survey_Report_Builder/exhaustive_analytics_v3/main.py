"""
Main orchestrator for Exhaustive Analytics v3

This is the primary entry point that defines and executes the analytics pipeline.
Modify the pipeline_steps list to customize the analysis flow.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from . import config
from .pipeline import run_pipeline, get_intermediate_state, list_intermediate_states
from .steps import validation, statistics, reporting


def run_exhaustive_analytics_v3(
    df: pd.DataFrame,
    current_yrmo: str,
    satisfaction_columns: List[Dict[str, str]],
    demographic_pivot_columns: List[str],
    product_column: str = "product",
    capture_intermediates: bool = True,
    verbose: bool = True,
    output_format: str = "standard"
) -> Dict[str, Any]:
    """
    Run the complete exhaustive analytics pipeline v3.
    
    This is the main function that orchestrates the entire analysis process
    using a modular pipeline approach. Each step can be easily modified,
    added, or removed by editing the pipeline_steps list.
    
    Args:
        df: DataFrame containing survey data with required fields
        current_yrmo: Current year-month in YYYYMM format (e.g., "202505")
        satisfaction_columns: List of dictionaries with column info:
            [{"column": "COST_SAT", "calculation": "1-10 AVG"}, ...]
        demographic_pivot_columns: List of demographic columns to analyze by
        product_column: Name of product column (default: "product")
        capture_intermediates: Whether to save intermediate pipeline states
        verbose: Whether to print progress messages
        output_format: Output format style ("standard", "detailed", "summary")
    
    Returns:
        Dictionary containing:
        {
            'final_df': Final processed DataFrame,
            'intermediates': Dictionary of intermediate DataFrames,
            'reports': {
                'consolidated': Product-level satisfaction report,
                'consolidated_significant': Significant changes only,
                'demographic': Demographic breakdown report,
                'demographic_significant': Significant demographic changes,
                'summary_stats': High-level summary statistics
            },
            'monthly_stats': Pre-calculated monthly statistics,
            'ma_results': Moving average calculations by period
        }
    
    Example:
        >>> results = run_exhaustive_analytics_v3(
        ...     df=survey_df,
        ...     current_yrmo="202505",
        ...     satisfaction_columns=[
        ...         {"column": "COST_SAT", "calculation": "1-10 AVG"},
        ...         {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
        ...     ],
        ...     demographic_pivot_columns=["AGE_GROUP", "REGION"]
        ... )
        >>> results['reports']['consolidated'].to_csv('output.csv')
    """
    
    # --- ARCHITECTURAL WARNING ---
    # The pipeline state is passed between steps using the pandas DataFrame's
    # experimental `.attrs` dictionary.
    # WARNING: Not all pandas operations preserve attributes. Any new pipeline
    # step must be carefully checked to ensure it doesn't drop attributes,
    # which would break subsequent steps.
    # Future versions should consider using a dedicated state class instead.
    df.attrs['current_yrmo'] = current_yrmo
    df.attrs['satisfaction_columns'] = satisfaction_columns
    df.attrs['demographic_columns'] = demographic_pivot_columns
    df.attrs['product_column'] = product_column
    
    # Define the pipeline steps
    # Each tuple contains: (step_name, function, parameters)
    pipeline_steps = [
        # ========== VALIDATION STEPS ==========
        ('validate_columns', validation.check_required_columns, {
            'satisfaction_columns': satisfaction_columns,
            'demographic_columns': demographic_pivot_columns,
            'required_columns': ['YRMO', product_column]
        }),
        
        ('validate_values', validation.validate_column_values, {
            'satisfaction_columns': satisfaction_columns
        }),
        
        ('validate_yrmo', validation.validate_yrmo_format, {}),
        
        ('check_sample_sizes', validation.check_sample_sizes, {
            'grouping_columns': [product_column] + demographic_pivot_columns
        }),
        
        # ========== STATISTICS STEPS ==========
        ('calculate_monthly_stats', statistics.calculate_monthly_stats, {
            'satisfaction_columns': satisfaction_columns,
            'demographic_columns': demographic_pivot_columns,
            'product_column': product_column
        }),
        
        ('calculate_1ma', statistics.add_ma_calculations, {
            'ma_period': 1,
            'current_yrmo': current_yrmo
        }),
        
        ('calculate_3ma', statistics.add_ma_calculations, {
            'ma_period': 3,
            'current_yrmo': current_yrmo
        }),
        
        ('calculate_6ma', statistics.add_ma_calculations, {
            'ma_period': 6,
            'current_yrmo': current_yrmo
        }),
        
        ('add_comparisons_1ma', statistics.add_period_comparisons, {
            'ma_period': 1,
            'current_yrmo': current_yrmo
        }),
        
        ('add_comparisons_3ma', statistics.add_period_comparisons, {
            'ma_period': 3,
            'current_yrmo': current_yrmo
        }),
        
        ('add_comparisons_6ma', statistics.add_period_comparisons, {
            'ma_period': 6,
            'current_yrmo': current_yrmo
        }),
        
        # ========== REPORTING STEPS ==========
        ('build_consolidated_report', reporting.build_consolidated_report, {
            'ma_period': 3,
            'product_column': product_column
        }),
        
        ('build_demographic_report', reporting.build_demographic_wide_report, {
            'demographic_columns': demographic_pivot_columns,
            'product_column': product_column
        }),
        
        ('filter_significant', reporting.filter_significant_changes, {
            'report_type': 'all'
        }),
        
        ('create_summary', reporting.create_summary_statistics, {}),
        
        ('format_output', reporting.format_final_output, {
            'output_format': output_format
        })
    ]
    
    # Execute the pipeline
    final_df, intermediates = run_pipeline(
        df=df,
        steps=pipeline_steps,
        capture_intermediates=capture_intermediates,
        verbose=verbose
    )
    
    # Extract reports from DataFrame attributes
    reports = {}
    
    # Get consolidated report
    if 'consolidated_report' in final_df.attrs:
        reports['consolidated'] = final_df.attrs['consolidated_report']
    
    # Get significant items
    if 'consolidated_significant' in final_df.attrs:
        reports['consolidated_significant'] = final_df.attrs['consolidated_significant']
    
    # Get demographic reports
    if 'demographic_report' in final_df.attrs:
        reports['demographic'] = final_df.attrs['demographic_report']
    
    if 'demographic_significant' in final_df.attrs:
        reports['demographic_significant'] = final_df.attrs['demographic_significant']
    
    # Get summary statistics
    if 'summary_statistics' in final_df.attrs:
        reports['summary_stats'] = final_df.attrs['summary_statistics']
    
    # Get the formatted final output
    if 'final_output' in final_df.attrs:
        reports['final_output'] = final_df.attrs['final_output']
    
    # Build return dictionary
    results = {
        'final_df': final_df,
        'intermediates': intermediates,
        'reports': reports
    }
    
    # Add direct access to key intermediate results
    if 'monthly_stats' in final_df.attrs:
        results['monthly_stats'] = final_df.attrs['monthly_stats']
    
    if 'ma_results' in final_df.attrs:
        results['ma_results'] = final_df.attrs['ma_results']
    
    return results


def add_custom_report_step(
    pipeline_steps: List[tuple],
    step_name: str,
    function: callable,
    parameters: Dict[str, Any],
    position: str = "before_output"
) -> List[tuple]:
    """
    Helper function to add a custom report step to the pipeline.
    
    This makes it easy to extend the pipeline with new report types
    without modifying the main code.
    
    Args:
        pipeline_steps: Current pipeline steps list
        step_name: Name for the new step
        function: Function to execute
        parameters: Parameters to pass to the function
        position: Where to insert ("before_output" or "after_stats")
    
    Returns:
        Modified pipeline steps list
    
    Example:
        >>> from my_custom_reports import generate_trend_report
        >>> pipeline_steps = add_custom_report_step(
        ...     pipeline_steps,
        ...     'trend_analysis',
        ...     generate_trend_report,
        ...     {'window': 12}
        ... )
    """
    new_step = (step_name, function, parameters)
    
    if position == "before_output":
        # Insert before the format_output step
        for i, (name, _, _) in enumerate(pipeline_steps):
            if name == 'format_output':
                pipeline_steps.insert(i, new_step)
                break
    elif position == "after_stats":
        # Insert after the last statistics step
        for i in range(len(pipeline_steps) - 1, -1, -1):
            if 'comparison' in pipeline_steps[i][0]:
                pipeline_steps.insert(i + 1, new_step)
                break
    else:
        # Append to end
        pipeline_steps.append(new_step)
    
    return pipeline_steps


# Convenience function for backward compatibility with v2
def run_exhaustive_analytics_v2_compatible(
    df: pd.DataFrame,
    current_yrmo: str,
    satisfaction_columns: List[Dict[str, str]],
    demographic_pivot_columns: List[str],
    product_column: str = "product"
) -> Dict[str, pd.DataFrame]:
    """
    Backward-compatible wrapper that mimics v2 output format.
    
    Use this if you need to maintain compatibility with existing code
    that expects the v2 output structure.
    """
    # Run v3 pipeline
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo=current_yrmo,
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=demographic_pivot_columns,
        product_column=product_column,
        capture_intermediates=False,  # v2 didn't capture intermediates
        verbose=False  # v2 had different logging
    )
    
    # Map to v2 output structure
    v2_output = {
        'precalc_stats': results.get('monthly_stats', pd.DataFrame()),
        'demographic_ma_wide': results['reports'].get('demographic', pd.DataFrame()),
        'demographic_significant': results['reports'].get('demographic_significant', pd.DataFrame()),
        'consolidated_3ma': results['reports'].get('consolidated', pd.DataFrame()),
        'consolidated_3ma_significant': results['reports'].get('consolidated_significant', pd.DataFrame())
    }
    
    return v2_output


if __name__ == "__main__":
    # Example usage - replace with your actual data
    print("Exhaustive Analytics v3 - Main Module")
    print("=====================================")
    print("This module should be imported and used in your analysis scripts.")
    print("\nExample usage:")
    print("""
    from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3
    import pandas as pd
    
    # Load your data
    df = pd.read_csv('survey_data.csv')
    
    # Configure satisfaction columns
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
    ]
    
    # Run analysis
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"]
    )
    
    # Save results
    results['reports']['consolidated'].to_csv('output.csv', index=False)
    
    # Access intermediate states
    monthly_stats = results['monthly_stats']
    ma_data = results['intermediates']['calculate_3ma']
    """)