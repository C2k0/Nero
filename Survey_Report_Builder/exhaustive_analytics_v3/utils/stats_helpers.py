"""
Statistical helper functions for Exhaustive Analytics v3

Low-level statistical calculations used by the pipeline steps.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy import stats
from .. import config


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
        top_values = config.CALCULATION_TYPES["1-5 T2B"]["top_values"]
        top_2_count = len(clean_series[clean_series.isin(top_values)])
        percentage = (top_2_count / count) * 100
        # For proportions, std_dev is calculated from the proportion itself
        proportion = top_2_count / count
        std_val = np.sqrt(proportion * (1 - proportion)) if count > 1 else 0.0
        return (percentage, count, std_val)
    
    else:
        raise ValueError(f"Unknown calculation type: {calculation_type}")


def get_ma_periods(current_yrmo: str, ma_period: int) -> List[str]:
    """
    Get list of YRMOs needed for MA calculation.
    
    Args:
        current_yrmo: Current YRMO in YYYYMM format
        ma_period: Number of months for moving average
    
    Returns:
        List of YRMOs needed for MA calculation (in chronological order)
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
        effective_n1 = max(config.MIN_SAMPLE_SIZE, current_count * (1 - overlap_ratio))
        effective_n2 = max(config.MIN_SAMPLE_SIZE, previous_count * (1 - overlap_ratio))
        
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


def format_percentage(value: float, decimal_places: Optional[int] = None) -> str:
    """
    Format a percentage value with consistent decimal places.
    
    Args:
        value: The percentage value
        decimal_places: Number of decimal places (default from config)
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    
    if decimal_places is None:
        decimal_places = config.PERCENTAGE_DECIMAL_PLACES
    
    return f"{value:.{decimal_places}f}%"


def format_average(value: float, decimal_places: Optional[int] = None) -> str:
    """
    Format an average value with consistent decimal places.
    
    Args:
        value: The average value
        decimal_places: Number of decimal places (default from config)
        
    Returns:
        Formatted average string
    """
    if pd.isna(value):
        return "N/A"
    
    if decimal_places is None:
        decimal_places = config.DECIMAL_PLACES
    
    return f"{value:.{decimal_places}f}"