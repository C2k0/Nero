"""
Report generation steps for the analytics pipeline

This module contains functions for creating final output reports
in various formats (demographic, consolidated, significant items).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .. import config
from ..utils.errors import print_error_block


def build_demographic_wide_report(
    df: pd.DataFrame,
    demographic_columns: Optional[List[str]] = None,
    product_column: str = "product"
) -> pd.DataFrame:
    """
    Build wide-format demographic report with MA analysis.
    
    Creates a report with one row per product/demographic combination
    and columns for each satisfaction metric's MA values and comparisons.
    
    Args:
        df: Input DataFrame with MA calculations in attributes
        demographic_columns: Demographic columns to include
        product_column: Name of the product column
    
    Returns:
        Wide-format DataFrame with demographic breakdowns
    """
    if demographic_columns is None:
        demographic_columns = config.DEFAULT_DEMOGRAPHIC_COLUMNS
    
    # Check for required attributes with validation
    if not hasattr(df, 'attrs') or 'comparison_results' not in df.attrs:
        raise ValueError("Comparison results not found. Run period comparisons first.")
    
    results = []
    
    # Process each demographic column separately
    for demo_col in demographic_columns:
        if demo_col not in df.columns:
            print_error_block(
                f"Demographic column '{demo_col}' not found",
                [f"Available columns: {list(df.columns[:10])}..."]
            )
            continue
        
        # Get unique combinations from the comparison results
        for ma_period in config.MA_PERIODS:
            if ma_period not in df.attrs['comparison_results']:
                continue
            
            comp_df = df.attrs['comparison_results'][ma_period]
            
            # Filter for this demographic
            demo_data = comp_df[comp_df.columns[comp_df.columns.str.contains(demo_col, case=False)]]
            
            if len(demo_data) == 0:
                # Use the full comparison data and add demographic grouping
                for _, row in comp_df.iterrows():
                    # Find demographic value from original data
                    product = row.get(product_column, 'Unknown')
                    
                    # Get unique demographic values for this product
                    product_df = df[df[product_column] == product]
                    if len(product_df) == 0:
                        continue
                    
                    for demo_value in product_df[demo_col].unique():
                        result_row = {
                            'product': product,
                            'demographic_field': demo_col,
                            'demographic_value': demo_value,
                            'ma_period': ma_period
                        }
                        
                        # Add all MA-related columns from the comparison results
                        for col in comp_df.columns:
                            if any(suffix in col for suffix in ['_ma', '_diff', '_sig', '_statsig']):
                                result_row[col] = row[col]
                        
                        results.append(result_row)
    
    if not results:
        # Create empty DataFrame with expected structure
        results_df = pd.DataFrame(columns=['product', 'demographic_field', 'demographic_value'])
    else:
        # Convert to DataFrame and pivot to wide format
        results_df = pd.DataFrame(results)
    
    # Add the demographic report to attributes for easy access
    df.attrs['demographic_report'] = results_df
    
    # Create multi-level columns for better organization
    # This is a simplified version - in production you might want more sophisticated pivoting
    return df


def build_consolidated_report(
    df: pd.DataFrame,
    ma_period: int = 3,
    product_column: str = "product"
) -> pd.DataFrame:
    """
    Build consolidated product-level satisfaction report.
    
    Creates a summary report with one row per product/satisfaction metric
    showing current and previous period values with significance testing.
    
    Args:
        df: Input DataFrame with comparison results
        ma_period: MA period to report (default: 3)
        product_column: Name of the product column
    
    Returns:
        Consolidated report DataFrame
    """
    if not hasattr(df, 'attrs') or 'comparison_results' not in df.attrs:
        raise ValueError("Comparison results not found. Run period comparisons first.")
    
    if ma_period not in df.attrs['comparison_results']:
        raise ValueError(f"No comparison results found for {ma_period}MA")
    
    comp_df = df.attrs['comparison_results'][ma_period]
    
    # Select relevant columns for consolidated report
    report_columns = [
        product_column,
        'satisfaction_column',
        'calculation_type',
        f'current_{ma_period}ma',
        f'current_{ma_period}ma_count',
        f'previous_{ma_period}ma',
        f'pop_diff_{ma_period}ma',
        f'pop_statsig_{ma_period}ma',
        f'pop_sig95_{ma_period}ma',
        f'pop_sig90_{ma_period}ma'
    ]
    
    # Filter to columns that exist
    available_columns = [col for col in report_columns if col in comp_df.columns]
    
    # Create the consolidated report
    consolidated = comp_df[available_columns].copy()
    
    # Rename columns for clarity
    rename_map = {
        'satisfaction_column': 'satisfaction_metric',
        f'current_{ma_period}ma': f'current_{ma_period}MA',
        f'current_{ma_period}ma_count': 'sample_size',
        f'previous_{ma_period}ma': f'previous_{ma_period}MA',
        f'pop_diff_{ma_period}ma': 'difference',
        f'pop_statsig_{ma_period}ma': 'confidence_level',
        f'pop_sig95_{ma_period}ma': 'sig_95',
        f'pop_sig90_{ma_period}ma': 'sig_90'
    }
    
    consolidated = consolidated.rename(columns=rename_map)
    
    # Sort by product and satisfaction metric
    if 'product' in consolidated.columns and 'satisfaction_metric' in consolidated.columns:
        consolidated = consolidated.sort_values(['product', 'satisfaction_metric'])
    
    # Add the consolidated report to attributes for easy access
    df.attrs['consolidated_report'] = consolidated
    
    return df


def filter_significant_changes(
    df: pd.DataFrame,
    threshold: float = None,
    report_type: str = "all"
) -> pd.DataFrame:
    """
    Filter reports to show only statistically significant changes.
    
    Args:
        df: Input DataFrame with reports in attributes
        threshold: Significance threshold (default: from config)
        report_type: Which report to filter ("demographic", "consolidated", "all")
    
    Returns:
        DataFrame with filtered significant changes added to attributes
    """
    if threshold is None:
        threshold = config.SIGNIFICANCE_FILTER_THRESHOLD
    
    sig_column = 'sig_95' if threshold >= 0.95 else 'sig_90'
    
    # Filter demographic report
    if report_type in ["demographic", "all"] and 'demographic_report' in df.attrs:
        demo_report = df.attrs['demographic_report']
        sig_columns = [col for col in demo_report.columns if col.endswith(sig_column)]
        
        if sig_columns:
            # Keep rows where any significance flag is 1
            sig_mask = demo_report[sig_columns].sum(axis=1) > 0
            df.attrs['demographic_significant'] = demo_report[sig_mask].copy()
    
    # Filter consolidated report
    if report_type in ["consolidated", "all"] and 'consolidated_report' in df.attrs:
        consol_report = df.attrs['consolidated_report']
        if sig_column in consol_report.columns:
            sig_mask = consol_report[sig_column] == 1
            df.attrs['consolidated_significant'] = consol_report[sig_mask].copy()
    
    return df


def format_final_output(
    df: pd.DataFrame,
    output_format: str = "standard"
) -> pd.DataFrame:
    """
    Format the final output reports for export.
    
    This function prepares the reports for saving to CSV or other formats
    by applying formatting, column ordering, and other presentation details.
    
    Args:
        df: Input DataFrame with reports in attributes
        output_format: Output format style ("standard", "detailed", "summary")
    
    Returns:
        DataFrame ready for export (the main consolidated report)
    """
    # Get the consolidated report as the primary output
    if 'consolidated_report' not in df.attrs:
        raise ValueError("No consolidated report found. Run build_consolidated_report first.")
    
    output_df = df.attrs['consolidated_report'].copy()
    
    # Apply formatting based on output format
    if output_format == "detailed":
        # Include all columns
        pass
    elif output_format == "summary":
        # Include only key columns
        summary_columns = [
            'product', 'satisfaction_metric', 'current_3MA',
            'difference', 'sig_95'
        ]
        available_summary = [col for col in summary_columns if col in output_df.columns]
        output_df = output_df[available_summary]
    else:  # standard
        # Remove technical columns
        drop_columns = ['calculation_type', 'confidence_level']
        output_df = output_df.drop(columns=[col for col in drop_columns if col in output_df.columns])
    
    # Round numeric columns
    numeric_columns = output_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if 'count' in col or 'size' in col:
            continue  # Don't round counts
        elif any(x in col for x in ['MA', 'difference']):
            # Round based on calculation type - this is simplified
            output_df[col] = output_df[col].round(config.DECIMAL_PLACES)
    
    # Sort by product and metric
    if 'product' in output_df.columns:
        output_df = output_df.sort_values(['product', 'satisfaction_metric'])
    
    # Store the formatted output
    df.attrs['final_output'] = output_df
    
    return df


def create_summary_statistics(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create summary statistics across all products and metrics.
    
    This generates high-level insights about the overall trends
    and significant changes in the data.
    
    Args:
        df: Input DataFrame with completed analysis
    
    Returns:
        DataFrame with summary statistics added to attributes
    """
    summary_stats = {}
    
    # Count significant changes
    if 'consolidated_significant' in df.attrs:
        sig_df = df.attrs['consolidated_significant']
        summary_stats['total_significant_changes'] = len(sig_df)
        summary_stats['products_with_changes'] = sig_df['product'].nunique() if 'product' in sig_df.columns else 0
        summary_stats['metrics_with_changes'] = sig_df['satisfaction_metric'].nunique() if 'satisfaction_metric' in sig_df.columns else 0
    
    # Calculate average changes
    if 'consolidated_report' in df.attrs:
        consol_df = df.attrs['consolidated_report']
        if 'difference' in consol_df.columns:
            summary_stats['avg_change'] = consol_df['difference'].mean()
            summary_stats['max_positive_change'] = consol_df['difference'].max()
            summary_stats['max_negative_change'] = consol_df['difference'].min()
    
    # Store summary statistics
    df.attrs['summary_statistics'] = pd.Series(summary_stats)
    
    return df