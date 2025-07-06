"""
Statistical calculation steps for the analytics pipeline

This module contains functions for calculating monthly statistics,
moving averages, and statistical significance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .. import config
from ..utils.stats_helpers import (
    calculate_metric, get_ma_periods, get_previous_yrmo,
    get_yoy_yrmo, calculate_statistical_significance_ma
)
from ..utils.errors import print_error_block


def calculate_monthly_stats(
    df: pd.DataFrame,
    satisfaction_columns: Optional[List[Dict[str, str]]] = None,
    demographic_columns: Optional[List[str]] = None,
    product_column: str = "product"
) -> pd.DataFrame:
    """
    Calculate monthly statistics for all metrics grouped by specified columns.
    
    This creates a long-format DataFrame with statistics for each
    satisfaction metric, broken down by product and demographics.
    
    Args:
        df: Input DataFrame
        satisfaction_columns: List of satisfaction column definitions
        demographic_columns: Demographic columns to include in grouping
        product_column: Name of the product column
    
    Returns:
        DataFrame with monthly statistics for each group
    """
    if satisfaction_columns is None:
        satisfaction_columns = config.DEFAULT_SATISFACTION_COLUMNS
    if demographic_columns is None:
        demographic_columns = config.DEFAULT_DEMOGRAPHIC_COLUMNS
    
    results = []
    
    # Build grouping columns
    group_by_columns = ["YRMO", product_column] + demographic_columns
    
    # Group by specified columns
    for group_values, group_df in df.groupby(group_by_columns):
        # Handle single vs multiple grouping columns
        if len(group_by_columns) == 1:
            group_values = [group_values]
        
        # Create base row with grouping column values
        base_row = {col: val for col, val in zip(group_by_columns, group_values)}
        
        # Calculate metrics for each satisfaction column
        for sat_col_info in satisfaction_columns:
            col_name = sat_col_info["column"]
            calc_type = sat_col_info["calculation"]
            
            if col_name not in group_df.columns:
                continue
            
            metric_value, count, std_dev = calculate_metric(group_df[col_name], calc_type)
            
            result_row = base_row.copy()
            result_row.update({
                'satisfaction_column': col_name,
                'calculation_type': calc_type,
                'mean_or_proportion': metric_value,
                'count': count,
                'std_dev': std_dev
            })
            
            results.append(result_row)
    
    # Create the stats DataFrame
    stats_df = pd.DataFrame(results)
    
    # Add the precalc stats to the original DataFrame for pipeline compatibility
    df = df.copy()
    df.attrs['monthly_stats'] = stats_df
    
    return df


def add_ma_calculations(
    df: pd.DataFrame,
    ma_period: int,
    current_yrmo: Optional[str] = None
) -> pd.DataFrame:
    """
    Add moving average calculations for a specific period.
    
    This function calculates MA values and their statistical significance
    compared to the previous period.
    
    Args:
        df: Input DataFrame (must have monthly_stats in attrs)
        ma_period: Moving average period (1, 3, or 6)
        current_yrmo: Current YRMO to calculate for (auto-detected if None)
    
    Returns:
        DataFrame with MA columns added
    """
    # Get monthly stats from attributes with validation
    if not hasattr(df, 'attrs') or 'monthly_stats' not in df.attrs:
        raise ValueError("Monthly stats not found. Run calculate_monthly_stats first.")
    
    monthly_stats_df = df.attrs['monthly_stats']
    
    # Auto-detect current YRMO if not provided
    if current_yrmo is None:
        current_yrmo = str(df['YRMO'].max())
    
    # Get required YRMOs for this MA period
    required_yrmos = get_ma_periods(current_yrmo, ma_period)
    
    # Filter for required YRMOs
    available_data = monthly_stats_df[monthly_stats_df['YRMO'].isin(required_yrmos)]
    
    # Group by all columns except YRMO for MA calculation
    group_columns = [col for col in monthly_stats_df.columns 
                    if col not in ['YRMO', 'mean_or_proportion', 'count', 'std_dev']]
    
    ma_results = []
    
    for group_values, group_df in available_data.groupby(group_columns):
        # Check if we have enough data
        available_months = len(group_df)
        
        if available_months < ma_period:
            # Log missing data warning
            if 'product' in group_columns:
                product_idx = group_columns.index('product')
                product = group_values[product_idx] if isinstance(group_values, tuple) else group_values
                sat_col_idx = group_columns.index('satisfaction_column')
                sat_col = group_values[sat_col_idx] if isinstance(group_values, tuple) else "Unknown"
                
                details = [
                    f"Product: {product}",
                    f"Satisfaction Column: {sat_col}",
                    f"MA Period: {ma_period}",
                    f"Available months: {available_months} (need {ma_period})",
                    f"Missing YRMOs: {set(required_yrmos) - set(group_df['YRMO'].tolist())}"
                ]
                print_error_block(f"Insufficient data for {ma_period}MA calculation", details)
            continue
        
        # Calculate MA
        ma_value = group_df['mean_or_proportion'].mean()
        total_count = group_df['count'].sum()
        ma_std = group_df['mean_or_proportion'].std() if len(group_df) > 1 else 0.0
        
        # Create result row
        if isinstance(group_values, tuple):
            result_row = {col: val for col, val in zip(group_columns, group_values)}
        else:
            result_row = {group_columns[0]: group_values}
        
        result_row.update({
            'YRMO': current_yrmo,
            f'{config.MA_COLUMN_PREFIX}{ma_period}_value': ma_value,
            f'{config.MA_COLUMN_PREFIX}{ma_period}_count': total_count,
            f'{config.MA_COLUMN_PREFIX}{ma_period}_std': ma_std
        })
        
        ma_results.append(result_row)
    
    # Convert to DataFrame
    ma_df = pd.DataFrame(ma_results)
    
    # Store MA results in attributes
    if 'ma_results' not in df.attrs:
        df.attrs['ma_results'] = {}
    df.attrs['ma_results'][ma_period] = ma_df
    
    return df


def add_period_comparisons(
    df: pd.DataFrame,
    ma_period: int = 3,
    current_yrmo: Optional[str] = None
) -> pd.DataFrame:
    """
    Add period-over-period and year-over-year comparisons with significance testing.
    
    Args:
        df: Input DataFrame with MA calculations
        ma_period: MA period to use for comparisons
        current_yrmo: Current YRMO (auto-detected if None)
    
    Returns:
        DataFrame with comparison columns added
    """
    if not hasattr(df, 'attrs') or 'monthly_stats' not in df.attrs:
        raise ValueError("Monthly stats not found. Run calculate_monthly_stats first.")
    
    monthly_stats_df = df.attrs['monthly_stats']
    
    # Auto-detect current YRMO if not provided
    if current_yrmo is None:
        current_yrmo = str(df['YRMO'].max())
    
    # Get comparison periods
    previous_yrmo = get_previous_yrmo(current_yrmo)
    yoy_yrmo = get_yoy_yrmo(current_yrmo)
    
    # Filter data for relevant periods
    current_data = monthly_stats_df[monthly_stats_df['YRMO'] == current_yrmo]
    previous_data = monthly_stats_df[monthly_stats_df['YRMO'] == previous_yrmo]
    yoy_data = monthly_stats_df[monthly_stats_df['YRMO'] == yoy_yrmo]
    
    # Group columns for matching
    group_columns = [col for col in monthly_stats_df.columns 
                    if col not in ['YRMO', 'mean_or_proportion', 'count', 'std_dev']]
    
    comparison_results = []
    
    # Process each current period group
    for _, current_row in current_data.iterrows():
        # Build match criteria
        match_criteria = {col: current_row[col] for col in group_columns}
        
        # Find matching previous period data
        prev_match = previous_data
        for col, val in match_criteria.items():
            prev_match = prev_match[prev_match[col] == val]
        
        # Find matching YoY data
        yoy_match = yoy_data
        for col, val in match_criteria.items():
            yoy_match = yoy_match[yoy_match[col] == val]
        
        # Build result row
        result_row = match_criteria.copy()
        result_row['YRMO'] = current_yrmo
        
        # Add current values
        result_row[f'current_{ma_period}ma'] = current_row['mean_or_proportion']
        result_row[f'current_{ma_period}ma_count'] = current_row['count']
        
        # Add previous period comparison
        if len(prev_match) > 0:
            prev_row = prev_match.iloc[0]
            difference = current_row['mean_or_proportion'] - prev_row['mean_or_proportion']
            
            # Calculate significance
            confidence = calculate_statistical_significance_ma(
                current_row['mean_or_proportion'],
                prev_row['mean_or_proportion'],
                current_row['std_dev'],
                prev_row['std_dev'],
                current_row['count'],
                prev_row['count'],
                ma_period,
                current_row['calculation_type']
            )
            
            result_row[f'previous_{ma_period}ma'] = prev_row['mean_or_proportion']
            result_row[f'pop_diff_{ma_period}ma'] = difference
            result_row[f'pop_statsig_{ma_period}ma'] = confidence
            result_row[f'pop_sig95_{ma_period}ma'] = 1 if (confidence and confidence >= config.SIG_95_THRESHOLD) else 0
            result_row[f'pop_sig90_{ma_period}ma'] = 1 if (confidence and confidence >= config.SIG_90_THRESHOLD) else 0
        
        # Add YoY comparison
        if len(yoy_match) > 0:
            yoy_row = yoy_match.iloc[0]
            yoy_difference = current_row['mean_or_proportion'] - yoy_row['mean_or_proportion']
            
            # Calculate YoY significance
            yoy_confidence = calculate_statistical_significance_ma(
                current_row['mean_or_proportion'],
                yoy_row['mean_or_proportion'],
                current_row['std_dev'],
                yoy_row['std_dev'],
                current_row['count'],
                yoy_row['count'],
                ma_period,
                current_row['calculation_type']
            )
            
            result_row[f'yoy_{ma_period}ma'] = yoy_row['mean_or_proportion']
            result_row[f'yoy_diff_{ma_period}ma'] = yoy_difference
            result_row[f'yoy_statsig_{ma_period}ma'] = yoy_confidence
            result_row[f'yoy_sig95_{ma_period}ma'] = 1 if (yoy_confidence and yoy_confidence >= config.SIG_95_THRESHOLD) else 0
            result_row[f'yoy_sig90_{ma_period}ma'] = 1 if (yoy_confidence and yoy_confidence >= config.SIG_90_THRESHOLD) else 0
        
        comparison_results.append(result_row)
    
    # Store comparison results
    if 'comparison_results' not in df.attrs:
        df.attrs['comparison_results'] = {}
    df.attrs['comparison_results'][ma_period] = pd.DataFrame(comparison_results)
    
    return df


def calculate_all_ma_periods(
    df: pd.DataFrame,
    current_yrmo: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate all configured MA periods in one step.
    
    This is a convenience function that runs MA calculations
    for all periods defined in config.MA_PERIODS.
    
    Args:
        df: Input DataFrame with monthly stats
        current_yrmo: Current YRMO (auto-detected if None)
    
    Returns:
        DataFrame with all MA calculations complete
    """
    for ma_period in config.MA_PERIODS:
        df = add_ma_calculations(df, ma_period, current_yrmo)
    
    return df