"""
Data validation steps for the analytics pipeline

This module contains functions for validating input data quality,
checking required columns, and ensuring data integrity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .. import config
from ..utils.errors import print_error_block


def handle_validation_issue(issue_type: str, details: List[str], 
                           should_fail: bool = None) -> None:
    """
    Centralized validation issue handler for consistent behavior.
    
    Args:
        issue_type: Type of validation issue
        details: List of detail messages
        should_fail: Whether to raise exception (uses config.STRICT_VALIDATION if None)
    """
    if should_fail is None:
        should_fail = config.STRICT_VALIDATION
    
    if should_fail:
        print_error_block(f"VALIDATION ERROR: {issue_type}", details)
        raise ValueError(f"Validation failed: {issue_type}")
    else:
        print_error_block(f"VALIDATION WARNING: {issue_type}", details)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    satisfaction_columns: Optional[List[Dict[str, str]]] = None,
    demographic_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Validate that all required columns exist in the DataFrame.
    
    This is typically the first step in the pipeline to ensure all
    necessary data is present before processing begins.
    
    Args:
        df: Input DataFrame to validate
        required_columns: Base required columns (default: from config)
        satisfaction_columns: Satisfaction column definitions (default: from config)
        demographic_columns: Demographic columns to check (default: from config)
        
    Returns:
        The input DataFrame unchanged (for pipeline compatibility)
        
    Raises:
        ValueError: If any required columns are missing
    """
    # Use defaults from config if not provided
    if required_columns is None:
        required_columns = config.REQUIRED_BASE_COLUMNS.copy()
    else:
        required_columns = required_columns.copy()
    
    if satisfaction_columns is None:
        satisfaction_columns = config.DEFAULT_SATISFACTION_COLUMNS
        
    if demographic_columns is None:
        demographic_columns = config.DEFAULT_DEMOGRAPHIC_COLUMNS
    
    # Build complete list of required columns
    all_required = set(required_columns)
    
    # Add satisfaction columns
    for sat_col in satisfaction_columns:
        all_required.add(sat_col["column"])
    
    # Add demographic columns
    all_required.update(demographic_columns)
    
    # Check for missing columns
    missing_columns = all_required - set(df.columns)
    
    if missing_columns:
        error_details = [
            f"Missing columns: {sorted(missing_columns)}",
            f"Available columns: {sorted(df.columns)}",
            f"Total required: {len(all_required)}",
            f"Total missing: {len(missing_columns)}"
        ]
        
        handle_validation_issue(
            "Required columns not found in DataFrame",
            error_details,
            should_fail=config.FAIL_ON_MISSING_COLUMNS
        )
    
    return df


def validate_column_values(
    df: pd.DataFrame,
    satisfaction_columns: Optional[List[Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Validate that satisfaction column values are within expected ranges.
    
    Checks each satisfaction column to ensure values match the expected
    scale (1-10 for averages, 1-5 for Top-2-Box).
    
    Args:
        df: Input DataFrame to validate
        satisfaction_columns: Column definitions with calculation types
        
    Returns:
        The input DataFrame unchanged (for pipeline compatibility)
        
    Prints warnings for any invalid values found.
    """
    if satisfaction_columns is None:
        satisfaction_columns = config.DEFAULT_SATISFACTION_COLUMNS
    
    all_warnings = []
    
    for sat_col_info in satisfaction_columns:
        column = sat_col_info["column"]
        calc_type = sat_col_info["calculation"]
        
        if column not in df.columns:
            all_warnings.append(f"Column '{column}' not found in DataFrame")
            continue
        
        # Get non-null values
        non_null_values = df[column].dropna()
        
        if len(non_null_values) == 0:
            all_warnings.append(f"Column '{column}' contains only null values")
            continue
        
        # Get expected range from config
        if calc_type in config.CALCULATION_TYPES:
            valid_range = config.CALCULATION_TYPES[calc_type]["valid_range"]
            invalid_values = set(non_null_values.unique()) - set(valid_range)
            
            if invalid_values:
                all_warnings.append(
                    f"Column '{column}' ({calc_type}) contains invalid values: "
                    f"{sorted(invalid_values)}. Expected: {list(valid_range)}"
                )
    
    if all_warnings and config.WARN_ON_INVALID_VALUES:
        print_error_block("Data Validation Warnings", all_warnings)
    
    return df


def validate_yrmo_format(
    df: pd.DataFrame,
    yrmo_column: str = "YRMO"
) -> pd.DataFrame:
    """
    Validate YRMO column format and values.
    
    Ensures YRMO values are in YYYYMM format with valid year/month combinations.
    
    Args:
        df: Input DataFrame
        yrmo_column: Name of the YRMO column
        
    Returns:
        The input DataFrame unchanged (for pipeline compatibility)
    """
    if yrmo_column not in df.columns:
        raise ValueError(f"YRMO column '{yrmo_column}' not found in DataFrame")
    
    # Convert to string for validation
    yrmo_values = df[yrmo_column].astype(str)
    
    invalid_formats = []
    invalid_months = []
    
    for yrmo in yrmo_values.unique():
        # Check format (should be 6 digits)
        if len(yrmo) != 6 or not yrmo.isdigit():
            invalid_formats.append(yrmo)
            continue
        
        # Check month validity
        month = int(yrmo[4:])
        if month < 1 or month > 12:
            invalid_months.append(yrmo)
    
    # Report any issues
    if invalid_formats or invalid_months:
        error_details = []
        if invalid_formats:
            error_details.append(f"Invalid format (expected YYYYMM): {invalid_formats}")
        if invalid_months:
            error_details.append(f"Invalid month values: {invalid_months}")
        
        handle_validation_issue(
            "YRMO Validation Issues",
            error_details,
            should_fail=config.FAIL_ON_INVALID_YRMO
        )
    
    return df


def check_sample_sizes(
    df: pd.DataFrame,
    grouping_columns: List[str],
    min_sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Check and report on sample sizes for different groupings.
    
    This helps identify groups with insufficient data for reliable analysis.
    
    Args:
        df: Input DataFrame
        grouping_columns: Columns to group by for sample size check
        min_sample_size: Minimum required sample size (default: from config)
        
    Returns:
        The input DataFrame unchanged (for pipeline compatibility)
    """
    if min_sample_size is None:
        min_sample_size = config.MIN_SAMPLE_SIZE
    
    # Calculate group sizes
    group_sizes = df.groupby(grouping_columns).size()
    
    # Find groups below threshold
    small_groups = group_sizes[group_sizes < min_sample_size]
    
    if len(small_groups) > 0:
        warnings = [
            f"Found {len(small_groups)} groups with sample size < {min_sample_size}",
            f"Smallest group: {group_sizes.min()} observations",
            f"These groups may have unreliable statistics"
        ]
        
        # Show a few examples
        examples = small_groups.head(5)
        for idx, size in examples.items():
            warnings.append(f"  {idx}: n={size}")
        
        if len(small_groups) > 5:
            warnings.append(f"  ... and {len(small_groups) - 5} more")
        
        print_error_block("Small Sample Size Warning", warnings)
    
    return df


def remove_invalid_data(
    df: pd.DataFrame,
    satisfaction_columns: Optional[List[Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Remove rows with invalid satisfaction values.
    
    This is an optional cleaning step that removes rows containing
    out-of-range values in satisfaction columns.
    
    Args:
        df: Input DataFrame
        satisfaction_columns: Column definitions with calculation types
        
    Returns:
        DataFrame with invalid rows removed
    """
    if satisfaction_columns is None:
        satisfaction_columns = config.DEFAULT_SATISFACTION_COLUMNS
    
    df_clean = df.copy()
    total_removed = 0
    
    for sat_col_info in satisfaction_columns:
        column = sat_col_info["column"]
        calc_type = sat_col_info["calculation"]
        
        if column not in df_clean.columns:
            continue
        
        # Get valid range
        if calc_type in config.CALCULATION_TYPES:
            valid_range = list(config.CALCULATION_TYPES[calc_type]["valid_range"])
            
            # Remove invalid values
            initial_len = len(df_clean)
            df_clean = df_clean[
                df_clean[column].isna() | df_clean[column].isin(valid_range)
            ]
            
            removed = initial_len - len(df_clean)
            if removed > 0:
                total_removed += removed
                print(f"Removed {removed} rows with invalid {column} values")
    
    if total_removed > 0:
        print(f"Total rows removed: {total_removed}")
        print(f"Remaining rows: {len(df_clean)}")
    
    return df_clean