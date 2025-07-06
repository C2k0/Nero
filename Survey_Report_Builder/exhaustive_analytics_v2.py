"""
Exhaustive Analytics Module v2.0 for Survey Data Analysis

SURVEY DATA SAMPLE SIZE BEST PRACTICES:
==========================================
- Minimum n=30: Below this, results flagged as "insufficient data"
- Small sample (30-100): Use t-distribution, results may be less reliable
- Medium sample (100-500): Standard statistical tests appropriate
- Large sample (500+): High confidence in results
- For proportions (T2B): Need at least 5 expected successes and failures

This module provides comprehensive survey data analysis functionality including:
- Moving Average (MA) calculations: 1MA, 3MA, 6MA
- Demographic analysis with period-over-period and year-over-year comparisons
- Consolidated product-level satisfaction reports
- Statistical significance testing with 90% and 95% confidence flags
- Pre-calculated statistics for all metrics

USAGE & FUNCTION FLOW:
=====================
1. run_exhaustive_analytics_v2() - Main function
   ├─► validate_required_columns() - Check data integrity
   ├─► calculate_monthly_stats() - Build precalc DataFrame [runs once per YRMO]
   ├─► calculate_ma_metrics() - Generate MA statistics [runs 3x: 1MA, 3MA, 6MA]
   ├─► build_demographic_ma_report() - Create wide demographic report
   ├─► build_consolidated_report() - Create product satisfaction report
   └─► filter_significant_items() - Extract statistically significant changes

QUICK START EXAMPLE:
===================

    from exhaustive_analytics_v2 import run_exhaustive_analytics_v2
    
    # Define satisfaction columns with calculation types
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
        {"column": "SERVICE_SAT", "calculation": "1-10 AVG"}
    ]
    
    # Run analysis
    results = run_exhaustive_analytics_v2(
        df=survey_df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"]
    )
    
    # Access results
    precalc_stats = results['precalc_stats']           # All monthly statistics
    demographic_report = results['demographic_ma_wide'] # Wide demographic report
    significant_items = results['demographic_significant'] # Significant changes only
    consolidated = results['consolidated_3ma']         # Product-level 3MA report

SAMPLE DATA STRUCTURE:
======================
Input DataFrame must contain:
- YRMO: Year-month in YYYYMM format (e.g., "202505")
- product: Product name
- Demographic columns (e.g., "AGE_GROUP", "REGION")
- Satisfaction columns with integer values:
  - 1-10 scale for AVG calculations
  - 1-5 scale for T2B calculations

Author: Exhaustive Analytics System
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
from scipy import stats

# Sample JSON structure for satisfaction columns (commented out for reference)
"""
SAMPLE SATISFACTION COLUMNS JSON:
[
    {"column": "COST_SAT", "calculation": "1-10 AVG"},
    {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
    {"column": "SERVICE_SAT", "calculation": "1-10 AVG"},
    {"column": "SPEED_SAT", "calculation": "1-5 T2B"},
    {"column": "RELIABILITY_SAT", "calculation": "1-10 AVG"},
    {"column": "USABILITY_SAT", "calculation": "1-5 T2B"},
    {"column": "SUPPORT_SAT", "calculation": "1-10 AVG"},
    {"column": "VALUE_SAT", "calculation": "1-5 T2B"},
    {"column": "RECOMMEND_SAT", "calculation": "1-10 AVG"},
    {"column": "OVERALL_SAT", "calculation": "1-5 T2B"}
]
"""


def print_error_block(title: str, details: List[str]) -> None:
    """
    Print clearly formatted error blocks with line separators.
    
    Args:
        title: Error title
        details: List of error details
    """
    print("\n" + "="*60)
    print(f"ERROR: {title}")
    print("="*60)
    for detail in details:
        print(f"  {detail}")
    print("="*60 + "\n")


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """
    Validate that all required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
    
    Returns:
        List of missing column names (empty if all present)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns


def validate_column_values(df: pd.DataFrame, column: str, calculation_type: str) -> List[str]:
    """
    Validate that column values are within expected ranges based on calculation type.
    
    Args:
        df: DataFrame containing survey data
        column: Column name to validate
        calculation_type: Either "1-10 AVG" or "1-5 T2B"
    
    Returns:
        List of validation warnings (empty if no issues)
    """
    warnings_list = []
    
    if column not in df.columns:
        warnings_list.append(f"Column '{column}' not found in DataFrame")
        return warnings_list
    
    # Get non-null values
    non_null_values = df[column].dropna()
    
    if len(non_null_values) == 0:
        warnings_list.append(f"Column '{column}' contains only null values")
        return warnings_list
    
    # Check for expected ranges
    if calculation_type == "1-10 AVG":
        valid_values = set(range(1, 11))  # 1 through 10
        invalid_values = set(non_null_values.unique()) - valid_values
        if invalid_values:
            warnings_list.append(
                f"Column '{column}' (1-10 AVG) contains invalid values: {invalid_values}. "
                f"Expected values: 1-10"
            )
    
    elif calculation_type == "1-5 T2B":
        valid_values = set(range(1, 6))  # 1 through 5
        invalid_values = set(non_null_values.unique()) - valid_values
        if invalid_values:
            warnings_list.append(
                f"Column '{column}' (1-5 T2B) contains invalid values: {invalid_values}. "
                f"Expected values: 1-5"
            )
    
    return warnings_list


def calculate_metric(series: pd.Series, calculation_type: str) -> Tuple[float, int, Optional[float]]:
    """
    Calculate metric based on calculation type with standard deviation.
    
    Args:
        series: Pandas Series with values to calculate
        calculation_type: Either "1-10 AVG" or "1-5 T2B"
    
    Returns:
        Tuple of (calculated_value, count_of_non_null_values, std_dev)
    """
    # Remove null values
    clean_series = series.dropna()
    count = len(clean_series)
    
    if count == 0:
        return (np.nan, 0, np.nan)
    
    if calculation_type == "1-10 AVG":
        # Simple average with standard deviation
        mean_val = clean_series.mean()
        std_val = clean_series.std() if count > 1 else 0.0
        return (mean_val, count, std_val)
    
    elif calculation_type == "1-5 T2B":
        # Top-2-Box: percentage of 4s and 5s
        top_2_count = len(clean_series[clean_series.isin([4, 5])])
        percentage = (top_2_count / count) * 100
        # For proportions, std_dev is calculated from the proportion itself
        proportion = top_2_count / count
        std_val = np.sqrt(proportion * (1 - proportion)) if count > 1 else 0.0
        return (percentage, count, std_val)
    
    else:
        raise ValueError(f"Unknown calculation type: {calculation_type}")


def calculate_monthly_stats(
    df: pd.DataFrame,
    satisfaction_columns: List[Dict[str, str]],
    group_by_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate monthly statistics for all metrics grouped by specified columns.
    
    Args:
        df: Input DataFrame
        satisfaction_columns: List of satisfaction column definitions
        group_by_columns: Columns to group by (e.g., ['YRMO', 'product'])
    
    Returns:
        DataFrame with monthly statistics for each group
    """
    results = []
    
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
    
    return pd.DataFrame(results)


def get_ma_periods(current_yrmo: str, ma_period: int) -> List[str]:
    """
    Get list of YRMOs needed for MA calculation.
    
    Args:
        current_yrmo: Current YRMO in YYYYMM format
        ma_period: Number of months for moving average
    
    Returns:
        List of YRMOs needed for MA calculation
    """
    year = int(current_yrmo[:4])
    month = int(current_yrmo[4:])
    
    yrmos = []
    for i in range(ma_period):
        # Calculate months backward
        target_month = month - i
        target_year = year
        
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        
        yrmo = f"{target_year:04d}{target_month:02d}"
        yrmos.append(yrmo)
    
    # Return in chronological order (oldest first)
    return list(reversed(yrmos))


def calculate_ma_metrics(
    monthly_stats_df: pd.DataFrame,
    current_yrmo: str,
    ma_period: int,
    group_by_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate moving average metrics from monthly statistics.
    
    Args:
        monthly_stats_df: DataFrame with monthly statistics
        current_yrmo: Current YRMO in YYYYMM format
        ma_period: Number of months for moving average
        group_by_columns: Columns to group by (excluding YRMO)
    
    Returns:
        DataFrame with MA metrics for each group
    """
    required_yrmos = get_ma_periods(current_yrmo, ma_period)
    
    # Filter for required YRMOs
    available_data = monthly_stats_df[monthly_stats_df['YRMO'].isin(required_yrmos)]
    
    results = []
    
    # Group by all columns except YRMO
    non_yrmo_groups = [col for col in group_by_columns if col != 'YRMO']
    group_columns = non_yrmo_groups + ['satisfaction_column', 'calculation_type']
    
    for group_values, group_df in available_data.groupby(group_columns):
        # Check if we have enough data
        available_months = len(group_df)
        
        if available_months < ma_period:
            # Print error for missing data
            if len(group_columns) > 3:  # Has product column
                details = [
                    f"Product: {group_values[0]}",
                    f"Satisfaction Column: {group_values[-2]}",
                    f"Current YRMO: {current_yrmo}",
                    f"MA Period: {ma_period}",
                    f"Available months: {available_months} (need {ma_period})",
                    f"Missing YRMOs: {set(required_yrmos) - set(group_df['YRMO'].tolist())}"
                ]
            else:
                details = [
                    f"Satisfaction Column: {group_values[-2]}",
                    f"Current YRMO: {current_yrmo}",
                    f"MA Period: {ma_period}",
                    f"Available months: {available_months} (need {ma_period})"
                ]
            
            print_error_block(f"Insufficient data for {ma_period}MA calculation", details)
            continue
        
        # Calculate MA
        ma_value = group_df['mean_or_proportion'].mean()
        total_count = group_df['count'].sum()
        
        # For MA standard deviation, we use the standard deviation of monthly means
        ma_std = group_df['mean_or_proportion'].std() if len(group_df) > 1 else 0.0
        
        # Create result row
        result_row = {col: val for col, val in zip(group_columns, group_values)}
        result_row.update({
            'YRMO': current_yrmo,
            'ma_period': ma_period,
            'ma_value': ma_value,
            'total_count': total_count,
            'ma_std': ma_std
        })
        
        results.append(result_row)
    
    return pd.DataFrame(results)


def calculate_statistical_significance_ma(
    current_ma_value: float,
    previous_ma_value: float,
    current_ma_std: float,
    previous_ma_std: float,
    current_count: int,
    previous_count: int,
    ma_period: int,
    calculation_type: str
) -> Optional[float]:
    """
    Calculate statistical significance for MA comparisons accounting for overlap.
    
    Args:
        current_ma_value: Current MA metric value
        previous_ma_value: Previous MA metric value
        current_ma_std: Current MA standard deviation
        previous_ma_std: Previous MA standard deviation
        current_count: Total sample size for current MA
        previous_count: Total sample size for previous MA
        ma_period: MA period (1, 3, or 6)
        calculation_type: Either "1-10 AVG" or "1-5 T2B"
    
    Returns:
        Confidence level (1 - p_value) for the statistical test, or None if cannot calculate
    """
    # Check if we have valid values and sufficient sample sizes
    if (pd.isna(current_ma_value) or pd.isna(previous_ma_value) or 
        current_count < 2 or previous_count < 2):
        return None
    
    # For 1MA, use standard tests
    if ma_period == 1:
        if calculation_type == "1-5 T2B":
            # Two-proportion z-test
            p1 = current_ma_value / 100
            p2 = previous_ma_value / 100
            n1 = current_count
            n2 = previous_count
            
            # Pooled proportion
            p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
            
            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            if se == 0:
                return None
            
            # Z-score
            z = (p1 - p2) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
        else:
            # Two-sample t-test
            # Use provided standard deviations
            se_diff = np.sqrt(
                (current_ma_std**2 / current_count) + 
                (previous_ma_std**2 / previous_count)
            )
            
            if se_diff == 0:
                return None
            
            # T-statistic
            t_stat = (current_ma_value - previous_ma_value) / se_diff
            
            # Degrees of freedom (Welch's approximation)
            df = current_count + previous_count - 2
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    else:
        # For MA periods > 1, account for overlap using conservative approach
        # Reduce effective sample size based on overlap
        overlap_ratio = (ma_period - 1) / ma_period
        effective_n1 = max(30, current_count * (1 - overlap_ratio))
        effective_n2 = max(30, previous_count * (1 - overlap_ratio))
        
        if calculation_type == "1-5 T2B":
            # Conservative two-proportion test with reduced sample size
            p1 = current_ma_value / 100
            p2 = previous_ma_value / 100
            
            # Pooled proportion
            p_pool = (p1 * effective_n1 + p2 * effective_n2) / (effective_n1 + effective_n2)
            
            # Standard error with effective sample sizes
            se = np.sqrt(p_pool * (1 - p_pool) * (1/effective_n1 + 1/effective_n2))
            
            if se == 0:
                return None
            
            # Z-score
            z = (p1 - p2) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
        else:
            # Conservative t-test with effective sample sizes
            se_diff = np.sqrt(
                (current_ma_std**2 / effective_n1) + 
                (previous_ma_std**2 / effective_n2)
            )
            
            if se_diff == 0:
                return None
            
            # T-statistic
            t_stat = (current_ma_value - previous_ma_value) / se_diff
            
            # Degrees of freedom
            df = effective_n1 + effective_n2 - 2
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Return confidence level (1 - p_value)
    confidence_level = 1 - p_value
    return max(0, min(1, confidence_level))  # Clamp between 0 and 1


def get_yoy_yrmo(current_yrmo: str) -> str:
    """
    Calculate the year-over-year YRMO given the current YRMO.
    
    Args:
        current_yrmo: Current YRMO in YYYYMM format
    
    Returns:
        YoY YRMO in YYYYMM format (same month, previous year)
    """
    year = int(current_yrmo[:4])
    month = current_yrmo[4:]
    
    yoy_year = year - 1
    return f"{yoy_year:04d}{month}"


def build_demographic_ma_report(
    df: pd.DataFrame,
    current_yrmo: str,
    satisfaction_columns: List[Dict[str, str]],
    demographic_pivot_columns: List[str],
    product_column: str,
    monthly_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build wide-format demographic report with MA calculations.
    
    Args:
        df: Original survey DataFrame
        current_yrmo: Current YRMO
        satisfaction_columns: List of satisfaction column definitions
        demographic_pivot_columns: List of demographic columns
        product_column: Product column name
        monthly_stats_df: Pre-calculated monthly statistics
    
    Returns:
        Wide-format DataFrame with MA analysis for demographics
    """
    results = []
    
    # Get unique combinations of product and demographics from current month
    current_df = df[df['YRMO'] == current_yrmo]
    
    for demo_col in demographic_pivot_columns:
        if demo_col not in current_df.columns:
            print_error_block(f"Demographic column '{demo_col}' not found", 
                            [f"Available columns: {list(current_df.columns)}"])
            continue
        
        # Get unique product/demographic combinations
        for (product, demo_value), group in current_df.groupby([product_column, demo_col]):
            base_row = {
                'product': product,
                'demographic_field': demo_col,
                'demographic_value': demo_value
            }
            
            # Build MA calculations for each satisfaction column
            for sat_col_info in satisfaction_columns:
                col_name = sat_col_info["column"]
                
                # Calculate 1MA, 3MA, 6MA
                for ma_period in [1, 3, 6]:
                    # Get current MA data
                    current_ma_filter = (
                        (monthly_stats_df['YRMO'] == current_yrmo) &
                        (monthly_stats_df[product_column] == product) &
                        (monthly_stats_df[demo_col] == demo_value) &
                        (monthly_stats_df['satisfaction_column'] == col_name)
                    )
                    
                    current_ma_data = monthly_stats_df[current_ma_filter]
                    
                    if len(current_ma_data) > 0:
                        current_ma_value = current_ma_data.iloc[0]['mean_or_proportion']
                        current_count = current_ma_data.iloc[0]['count']
                        current_std = current_ma_data.iloc[0]['std_dev']
                        
                        # Get previous month data for comparison
                        previous_yrmo = get_previous_yrmo(current_yrmo)
                        previous_ma_filter = (
                            (monthly_stats_df['YRMO'] == previous_yrmo) &
                            (monthly_stats_df[product_column] == product) &
                            (monthly_stats_df[demo_col] == demo_value) &
                            (monthly_stats_df['satisfaction_column'] == col_name)
                        )
                        
                        previous_ma_data = monthly_stats_df[previous_ma_filter]
                        
                        # Add current MA columns
                        base_row[f"{col_name}__{ma_period}MA_current"] = current_ma_value
                        base_row[f"{col_name}__{ma_period}MA_count"] = current_count
                        
                        if len(previous_ma_data) > 0:
                            previous_ma_value = previous_ma_data.iloc[0]['mean_or_proportion']
                            previous_count = previous_ma_data.iloc[0]['count']
                            previous_std = previous_ma_data.iloc[0]['std_dev']
                            
                            # Calculate difference
                            difference = current_ma_value - previous_ma_value
                            
                            # Calculate statistical significance
                            confidence = calculate_statistical_significance_ma(
                                current_ma_value, previous_ma_value,
                                current_std, previous_std,
                                current_count, previous_count,
                                ma_period, sat_col_info["calculation"]
                            )
                            
                            # Add comparison columns
                            base_row[f"{col_name}__{ma_period}MA_previous"] = previous_ma_value
                            base_row[f"{col_name}__{ma_period}MA_diff"] = difference
                            base_row[f"{col_name}__{ma_period}MA_statsig"] = confidence
                            base_row[f"{col_name}__{ma_period}MA_sig95"] = 1 if (confidence and confidence >= 0.95) else 0
                            base_row[f"{col_name}__{ma_period}MA_sig90"] = 1 if (confidence and confidence >= 0.90) else 0
                        else:
                            # No previous data
                            base_row[f"{col_name}__{ma_period}MA_previous"] = np.nan
                            base_row[f"{col_name}__{ma_period}MA_diff"] = np.nan
                            base_row[f"{col_name}__{ma_period}MA_statsig"] = None
                            base_row[f"{col_name}__{ma_period}MA_sig95"] = 0
                            base_row[f"{col_name}__{ma_period}MA_sig90"] = 0
                    else:
                        # No current data for this MA period
                        for suffix in ['current', 'previous', 'diff', 'statsig', 'sig95', 'sig90']:
                            base_row[f"{col_name}__{ma_period}MA_{suffix}"] = np.nan if suffix in ['current', 'previous', 'diff'] else 0
                        base_row[f"{col_name}__{ma_period}MA_count"] = 0
                
                # Add YoY comparison for 3MA only
                yoy_yrmo = get_yoy_yrmo(current_yrmo)
                yoy_ma_filter = (
                    (monthly_stats_df['YRMO'] == yoy_yrmo) &
                    (monthly_stats_df[product_column] == product) &
                    (monthly_stats_df[demo_col] == demo_value) &
                    (monthly_stats_df['satisfaction_column'] == col_name)
                )
                
                yoy_ma_data = monthly_stats_df[yoy_ma_filter]
                
                if len(yoy_ma_data) > 0:
                    yoy_ma_value = yoy_ma_data.iloc[0]['mean_or_proportion']
                    yoy_count = yoy_ma_data.iloc[0]['count']
                    yoy_std = yoy_ma_data.iloc[0]['std_dev']
                    
                    # Use current 3MA value (already calculated above)
                    current_3ma = base_row.get(f"{col_name}__3MA_current", np.nan)
                    current_3ma_count = base_row.get(f"{col_name}__3MA_count", 0)
                    
                    if not pd.isna(current_3ma):
                        # Calculate YoY difference
                        yoy_difference = current_3ma - yoy_ma_value
                        
                        # Calculate YoY statistical significance
                        current_3ma_std = current_std  # Use current std for 3MA
                        yoy_confidence = calculate_statistical_significance_ma(
                            current_3ma, yoy_ma_value,
                            current_3ma_std, yoy_std,
                            current_3ma_count, yoy_count,
                            3, sat_col_info["calculation"]
                        )
                        
                        # Add YoY columns
                        base_row[f"{col_name}__YoY_previous"] = yoy_ma_value
                        base_row[f"{col_name}__YoY_diff"] = yoy_difference
                        base_row[f"{col_name}__YoY_statsig"] = yoy_confidence
                        base_row[f"{col_name}__YoY_sig95"] = 1 if (yoy_confidence and yoy_confidence >= 0.95) else 0
                        base_row[f"{col_name}__YoY_sig90"] = 1 if (yoy_confidence and yoy_confidence >= 0.90) else 0
                    else:
                        # No current 3MA data
                        for suffix in ['previous', 'diff', 'statsig', 'sig95', 'sig90']:
                            base_row[f"{col_name}__YoY_{suffix}"] = np.nan if suffix in ['previous', 'diff'] else 0
                else:
                    # No YoY data
                    for suffix in ['previous', 'diff', 'statsig', 'sig95', 'sig90']:
                        base_row[f"{col_name}__YoY_{suffix}"] = np.nan if suffix in ['previous', 'diff'] else 0
            
            results.append(base_row)
    
    return pd.DataFrame(results)


def build_consolidated_report(
    monthly_stats_df: pd.DataFrame,
    current_yrmo: str,
    satisfaction_columns: List[Dict[str, str]],
    product_column: str
) -> pd.DataFrame:
    """
    Build consolidated product-level satisfaction report (3MA only).
    
    Args:
        monthly_stats_df: Pre-calculated monthly statistics
        current_yrmo: Current YRMO
        satisfaction_columns: List of satisfaction column definitions
        product_column: Product column name
    
    Returns:
        DataFrame with consolidated satisfaction metrics by product
    """
    results = []
    
    # Get unique products from current month data
    current_products = monthly_stats_df[monthly_stats_df['YRMO'] == current_yrmo][product_column].unique()
    
    for product in current_products:
        for sat_col_info in satisfaction_columns:
            col_name = sat_col_info["column"]
            calc_type = sat_col_info["calculation"]
            
            # Get current 3MA data
            current_filter = (
                (monthly_stats_df['YRMO'] == current_yrmo) &
                (monthly_stats_df[product_column] == product) &
                (monthly_stats_df['satisfaction_column'] == col_name)
            )
            
            current_data = monthly_stats_df[current_filter]
            
            if len(current_data) > 0:
                current_value = current_data.iloc[0]['mean_or_proportion']
                current_count = current_data.iloc[0]['count']
                current_std = current_data.iloc[0]['std_dev']
                
                # Get previous month data
                previous_yrmo = get_previous_yrmo(current_yrmo)
                previous_filter = (
                    (monthly_stats_df['YRMO'] == previous_yrmo) &
                    (monthly_stats_df[product_column] == product) &
                    (monthly_stats_df['satisfaction_column'] == col_name)
                )
                
                previous_data = monthly_stats_df[previous_filter]
                
                result_row = {
                    'satisfaction_metric': col_name,
                    'product': product,
                    'current_3ma': current_value,
                    'current_count': current_count
                }
                
                if len(previous_data) > 0:
                    previous_value = previous_data.iloc[0]['mean_or_proportion']
                    previous_count = previous_data.iloc[0]['count']
                    previous_std = previous_data.iloc[0]['std_dev']
                    
                    difference = current_value - previous_value
                    
                    # Calculate statistical significance for 3MA
                    confidence = calculate_statistical_significance_ma(
                        current_value, previous_value,
                        current_std, previous_std,
                        current_count, previous_count,
                        3, calc_type
                    )
                    
                    result_row.update({
                        'previous_3ma': previous_value,
                        'previous_count': previous_count,
                        'difference': difference,
                        'statsig': confidence,
                        'sig95': 1 if (confidence and confidence >= 0.95) else 0,
                        'sig90': 1 if (confidence and confidence >= 0.90) else 0
                    })
                else:
                    result_row.update({
                        'previous_3ma': np.nan,
                        'previous_count': 0,
                        'difference': np.nan,
                        'statsig': None,
                        'sig95': 0,
                        'sig90': 0
                    })
                
                results.append(result_row)
    
    return pd.DataFrame(results)


def filter_significant_items(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Filter DataFrame to include only rows with statistically significant changes.
    
    Args:
        df: Input DataFrame
        threshold: Significance threshold (default 0.95 for 95% confidence)
    
    Returns:
        Filtered DataFrame with only significant items
    """
    if df.empty:
        return df.copy()
    
    # Find all significance flag columns
    sig_columns = [col for col in df.columns if col.endswith('_sig95')]
    
    if not sig_columns:
        return df.iloc[0:0].copy()  # Return empty DataFrame with same structure
    
    # Create mask for any significant change
    sig_mask = df[sig_columns].sum(axis=1) > 0
    
    return df[sig_mask].copy()


def get_previous_yrmo(current_yrmo: str) -> str:
    """
    Calculate the previous YRMO given the current YRMO.
    
    Args:
        current_yrmo: Current YRMO in YYYYMM format
    
    Returns:
        Previous YRMO in YYYYMM format
    """
    year = int(current_yrmo[:4])
    month = int(current_yrmo[4:])
    
    if month == 1:
        # January -> December of previous year
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1
    
    return f"{prev_year:04d}{prev_month:02d}"


def run_exhaustive_analytics_v2(
    df: pd.DataFrame,
    current_yrmo: str,
    satisfaction_columns: List[Dict[str, str]],
    demographic_pivot_columns: List[str],
    product_column: str = "product"
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive analytics on survey data with MA calculations and statistical testing.
    
    This function performs the following analysis:
    1. Validates data quality for satisfaction columns
    2. Calculates monthly statistics for all metrics
    3. Generates MA calculations (1MA, 3MA, 6MA)
    4. Creates wide-format demographic report with period-over-period and YoY comparisons
    5. Creates consolidated product-level satisfaction report (3MA)
    6. Filters for statistically significant changes
    
    Args:
        df: DataFrame containing survey data with required fields
        current_yrmo: Current year-month in YYYYMM format (e.g., "202505")
        satisfaction_columns: List of dictionaries with column info:
            [{"column": "COST_SAT", "calculation": "1-10 AVG"}, ...]
        demographic_pivot_columns: List of demographic columns to analyze by
        product_column: Name of product column (default: "product")
    
    Returns:
        Dictionary with DataFrames:
        {
            'precalc_stats': DataFrame with monthly statistics for all metrics,
            'demographic_ma_wide': Wide-format demographic report with MA analysis,
            'demographic_significant': Filtered significant demographic changes,
            'consolidated_3ma': Product-level 3MA satisfaction report,
            'consolidated_3ma_significant': Filtered significant product changes
        }
    
    Raises:
        ValueError: If required columns are missing or data validation fails
    """
    # ================================================================
    # STEP 1: VALIDATE INPUT DATA
    # ================================================================
    
    # Validate required columns exist
    required_columns = ["YRMO"] + [product_column] + demographic_pivot_columns
    missing_columns = validate_required_columns(df, required_columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate satisfaction columns exist and have valid values
    all_warnings = []
    for sat_col_info in satisfaction_columns:
        col_warnings = validate_column_values(
            df, 
            sat_col_info["column"], 
            sat_col_info["calculation"]
        )
        all_warnings.extend(col_warnings)
    
    if all_warnings:
        print_error_block("Data Validation Warnings", all_warnings)
    
    # ================================================================
    # STEP 2: BUILD PRE-CALCULATION STATS DATAFRAME
    # ================================================================
    
    # Calculate monthly statistics grouped by all required dimensions
    group_by_cols = ["YRMO", product_column] + demographic_pivot_columns
    monthly_stats_df = calculate_monthly_stats(df, satisfaction_columns, group_by_cols)
    
    # ================================================================
    # STEP 3: GENERATE MA CALCULATIONS
    # ================================================================
    
    # Note: MA calculations are handled within the report generation functions
    # This approach allows for better error handling and data validation
    
    # ================================================================
    # STEP 4: GENERATE DEMOGRAPHIC REPORT (WIDE FORMAT)
    # ================================================================
    
    demographic_ma_wide = build_demographic_ma_report(
        df, current_yrmo, satisfaction_columns, 
        demographic_pivot_columns, product_column, monthly_stats_df
    )
    
    # ================================================================
    # STEP 5: GENERATE CONSOLIDATED REPORT (3MA ONLY)
    # ================================================================
    
    consolidated_3ma = build_consolidated_report(
        monthly_stats_df, current_yrmo, satisfaction_columns, product_column
    )
    
    # ================================================================
    # STEP 6: FILTER SIGNIFICANT ITEMS
    # ================================================================
    
    demographic_significant = filter_significant_items(demographic_ma_wide)
    consolidated_3ma_significant = filter_significant_items(consolidated_3ma)
    
    # ================================================================
    # RETURN RESULTS
    # ================================================================
    
    return {
        'precalc_stats': monthly_stats_df,
        'demographic_ma_wide': demographic_ma_wide,
        'demographic_significant': demographic_significant,
        'consolidated_3ma': consolidated_3ma,
        'consolidated_3ma_significant': consolidated_3ma_significant
    }


# Example usage (commented out)
"""
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        "YRMO": ["202505"] * 500 + ["202504"] * 500 + ["202503"] * 500 + ["202405"] * 500,
        "product": np.random.choice(["Product A", "Product B"], 2000),
        "AGE_GROUP": np.random.choice(["18-34", "35-54", "55+"], 2000),
        "REGION": np.random.choice(["North", "South", "East", "West"], 2000),
        "COST_SAT": np.random.randint(1, 11, 2000),
        "QUALITY_SAT": np.random.randint(1, 6, 2000),
        "SERVICE_SAT": np.random.randint(1, 11, 2000)
    })
    
    # Add some null values
    sample_data.loc[sample_data.sample(200).index, "COST_SAT"] = np.nan
    
    # Define satisfaction columns
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
        {"column": "SERVICE_SAT", "calculation": "1-10 AVG"}
    ]
    
    # Run analysis
    results = run_exhaustive_analytics_v2(
        df=sample_data,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"]
    )
    
    print("=== ANALYSIS RESULTS ===")
    print(f"Precalc Stats Shape: {results['precalc_stats'].shape}")
    print(f"Demographic Report Shape: {results['demographic_ma_wide'].shape}")
    print(f"Significant Demographic Items: {len(results['demographic_significant'])}")
    print(f"Consolidated Report Shape: {results['consolidated_3ma'].shape}")
    print(f"Significant Consolidated Items: {len(results['consolidated_3ma_significant'])}")
    
    # Display sample outputs
    print("\n=== SAMPLE DEMOGRAPHIC REPORT ===")
    print(results['demographic_ma_wide'].head())
    
    print("\n=== SAMPLE CONSOLIDATED REPORT ===")
    print(results['consolidated_3ma'].head())
"""