"""
Pipeline runner module for Exhaustive Analytics v3

This module provides the core pipeline execution functionality with
intermediate state tracking and error handling.
"""

import pandas as pd
from typing import List, Tuple, Callable, Dict, Any, Optional
from datetime import datetime
from . import config


class PipelineError(Exception):
    """Custom exception for pipeline execution errors"""
    pass


def run_pipeline(
    df: pd.DataFrame,
    steps: List[Tuple[str, Callable[..., pd.DataFrame], Dict[str, Any]]],
    capture_intermediates: bool = None,
    verbose: bool = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Executes a pipeline of transformation functions on a DataFrame.
    
    This is the core function that orchestrates the entire analysis pipeline.
    Each step transforms the DataFrame and optionally captures intermediate states
    for debugging and analysis.
    
    Args:
        df: The initial DataFrame to process
        steps: List of tuples containing:
            - step_name (str): Descriptive name for the step
            - func (Callable): Function that takes a DataFrame and returns a DataFrame
            - params (Dict): Parameters to pass to the function
        capture_intermediates: Whether to save DataFrame state after each step
            Defaults to config.CAPTURE_INTERMEDIATES_DEFAULT
        verbose: Whether to print progress messages
            Defaults to config.VERBOSE_PIPELINE
    
    Returns:
        Tuple containing:
            - Final transformed DataFrame
            - Dictionary of intermediate DataFrames (empty if capture_intermediates=False)
    
    Raises:
        PipelineError: If any step fails, with details about which step and why
        
    Example:
        >>> steps = [
        ...     ('validate', validation.check_columns, {'required': ['YRMO']}),
        ...     ('calculate_ma', statistics.add_moving_average, {'period': 3})
        ... ]
        >>> final_df, intermediates = run_pipeline(df, steps)
        >>> print(f"Pipeline completed with {len(intermediates)} intermediate states")
    """
    # Use config defaults if not specified
    if capture_intermediates is None:
        capture_intermediates = config.CAPTURE_INTERMEDIATES_DEFAULT
    if verbose is None:
        verbose = config.VERBOSE_PIPELINE
    
    # Initialize intermediate results storage
    intermediate_results = {}
    
    # Capture initial state
    if capture_intermediates:
        intermediate_results['00_initial'] = df.copy() if config.COPY_DATAFRAMES else df
    
    # Track pipeline execution time
    start_time = datetime.now()
    
    if verbose:
        print(f"\n{config.ERROR_SEPARATOR}")
        print(f"Starting pipeline execution at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input DataFrame shape: {df.shape}")
        print(config.ERROR_SEPARATOR)
    
    # Execute each step in sequence
    current_df = df.copy() if config.COPY_DATAFRAMES else df
    
    for idx, (step_name, func, params) in enumerate(steps, 1):
        step_start = datetime.now()
        
        if verbose:
            print(f"\nStep {idx}/{len(steps)}: {step_name}")
            print(f"  Function: {func.__name__}")
            if params:
                print(f"  Parameters: {params}")
        
        try:
            # Execute the transformation
            current_df = func(current_df, **params)
            
            # Validate output
            if not isinstance(current_df, pd.DataFrame):
                raise PipelineError(
                    f"Step '{step_name}' did not return a DataFrame. "
                    f"Got {type(current_df).__name__} instead."
                )
            
            # Capture intermediate state
            if capture_intermediates:
                intermediate_key = f"{idx:02d}_{step_name}"
                intermediate_results[intermediate_key] = (
                    current_df.copy() if config.COPY_DATAFRAMES else current_df
                )
            
            # Report step completion
            step_duration = (datetime.now() - step_start).total_seconds()
            if verbose:
                print(f"  ✓ Completed in {step_duration:.2f}s")
                print(f"  Output shape: {current_df.shape}")
                
        except Exception as e:
            # Enhanced error reporting
            error_msg = (
                f"\nPipeline failed at step {idx}: '{step_name}'\n"
                f"Function: {func.__name__}\n"
                f"Parameters: {params}\n"
                f"Error: {type(e).__name__}: {str(e)}"
            )
            
            if verbose:
                print(f"\n{config.ERROR_SEPARATOR}")
                print("ERROR: Pipeline Execution Failed")
                print(config.ERROR_SEPARATOR)
                print(error_msg)
                print(config.ERROR_SEPARATOR)
            
            # Re-raise with context
            raise PipelineError(error_msg) from e
    
    # Report pipeline completion
    total_duration = (datetime.now() - start_time).total_seconds()
    if verbose:
        print(f"\n{config.ERROR_SEPARATOR}")
        print(f"Pipeline completed successfully!")
        print(f"Total execution time: {total_duration:.2f}s")
        print(f"Final DataFrame shape: {current_df.shape}")
        if capture_intermediates:
            print(f"Captured {len(intermediate_results)} intermediate states")
        print(config.ERROR_SEPARATOR)
    
    return current_df, intermediate_results


def get_intermediate_state(
    intermediate_results: Dict[str, pd.DataFrame],
    step_name: str
) -> Optional[pd.DataFrame]:
    """
    Retrieve a specific intermediate state by step name.
    
    Args:
        intermediate_results: Dictionary of intermediate DataFrames from run_pipeline
        step_name: Name of the step to retrieve
        
    Returns:
        DataFrame for the specified step, or None if not found
        
    Example:
        >>> ma_data = get_intermediate_state(intermediates, 'calculate_ma')
    """
    # Try exact match first
    for key, df in intermediate_results.items():
        if key.endswith(f"_{step_name}"):
            return df
    
    # Try partial match
    for key, df in intermediate_results.items():
        if step_name in key:
            return df
    
    return None


def list_intermediate_states(
    intermediate_results: Dict[str, pd.DataFrame],
    show_shapes: bool = True
) -> None:
    """
    Print a summary of all captured intermediate states.
    
    Args:
        intermediate_results: Dictionary of intermediate DataFrames
        show_shapes: Whether to include DataFrame shapes in the output
    """
    print(f"\nCaptured Intermediate States ({len(intermediate_results)} total):")
    print("-" * 50)
    
    for key, df in intermediate_results.items():
        if show_shapes:
            print(f"{key}: {df.shape} (rows: {df.shape[0]}, cols: {df.shape[1]})")
        else:
            print(key)
    
    print("-" * 50)


def safe_get_attr(df: pd.DataFrame, key: str, default=None):
    """
    Safely get attribute from DataFrame with validation.
    
    Args:
        df: DataFrame to get attribute from
        key: Attribute key to retrieve
        default: Default value if attribute doesn't exist
        
    Returns:
        Attribute value or default
        
    Raises:
        ValueError: If critical attribute is missing and no default provided
    """
    if hasattr(df, 'attrs') and key in df.attrs:
        return df.attrs[key]
    elif default is not None:
        return default
    else:
        raise ValueError(f"Critical pipeline attribute '{key}' not found in DataFrame.attrs")


def safe_set_attr(df: pd.DataFrame, key: str, value):
    """
    Safely set attribute on DataFrame.
    
    Args:
        df: DataFrame to set attribute on
        key: Attribute key
        value: Value to set
    """
    if not hasattr(df, 'attrs'):
        df.attrs = {}
    df.attrs[key] = value