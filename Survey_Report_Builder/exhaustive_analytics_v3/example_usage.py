"""
Example usage script for Exhaustive Analytics v3

This script demonstrates how to use the v3 analytics pipeline
with sample data and various configuration options.
"""

import pandas as pd
import numpy as np
from .main import run_exhaustive_analytics_v3, run_exhaustive_analytics_v2_compatible
from .pipeline import get_intermediate_state, list_intermediate_states
from . import config


def create_sample_data(n_months: int = 6, n_products: int = 2, 
                      n_responses_per_month: int = 500) -> pd.DataFrame:
    """
    Create sample survey data for testing.
    
    Args:
        n_months: Number of months of data to generate
        n_products: Number of products to include
        n_responses_per_month: Approximate responses per month
    
    Returns:
        Sample DataFrame with survey data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate YRMOs (last n_months ending in current month)
    base_year = 2025
    base_month = 5
    yrmos = []
    
    for i in range(n_months):
        month = base_month - i
        year = base_year
        
        while month <= 0:
            month += 12
            year -= 1
        
        yrmo = f"{year:04d}{month:02d}"
        yrmos.append(yrmo)
    
    yrmos.reverse()  # Chronological order
    
    # Products and demographics
    products = [f"Product {chr(65+i)}" for i in range(n_products)]  # Product A, B, etc.
    age_groups = ["18-34", "35-54", "55+"]
    regions = ["North", "South", "East", "West"]
    
    data = []
    
    for yrmo in yrmos:
        for _ in range(n_responses_per_month):
            # Add some realistic trends
            trend_factor = yrmos.index(yrmo) * 0.1  # Slight improvement over time
            product = np.random.choice(products)
            
            # Simulate different satisfaction levels by product
            if product == "Product A":
                base_satisfaction = 7.5 + trend_factor
            else:
                base_satisfaction = 6.8 + trend_factor
            
            row = {
                'YRMO': yrmo,
                'product': product,
                'AGE_GROUP': np.random.choice(age_groups),
                'REGION': np.random.choice(regions),
                
                # 1-10 AVG satisfaction scores
                'COST_SAT': np.clip(int(np.random.normal(base_satisfaction, 1.5)), 1, 10),
                'SERVICE_SAT': np.clip(int(np.random.normal(base_satisfaction + 0.5, 1.2)), 1, 10),
                'RELIABILITY_SAT': np.clip(int(np.random.normal(base_satisfaction - 0.2, 1.3)), 1, 10),
                
                # 1-5 T2B satisfaction scores
                'QUALITY_SAT': np.clip(int(np.random.normal(3.8, 1.0)), 1, 5),
                'SPEED_SAT': np.clip(int(np.random.normal(3.5, 1.1)), 1, 5),
                'VALUE_SAT': np.clip(int(np.random.normal(3.6, 1.0)), 1, 5)
            }
            
            data.append(row)
    
    return pd.DataFrame(data)


def example_basic_usage():
    """Demonstrate basic usage of the v3 pipeline."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create sample data
    df = create_sample_data(n_months=6, n_products=2, n_responses_per_month=300)
    print(f"Created sample data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['YRMO'].min()} to {df['YRMO'].max()}")
    
    # Define satisfaction columns
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
        {"column": "SERVICE_SAT", "calculation": "1-10 AVG"},
        {"column": "SPEED_SAT", "calculation": "1-5 T2B"}
    ]
    
    # Run analysis
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"]
    )
    
    # Display results
    print(f"\nResults generated:")
    print(f"- Monthly stats: {results['monthly_stats'].shape}")
    print(f"- Consolidated report: {results['reports']['consolidated'].shape}")
    print(f"- Significant changes: {len(results['reports'].get('consolidated_significant', []))}")
    print(f"- Intermediate states captured: {len(results['intermediates'])}")
    
    # Show sample of consolidated report
    print(f"\nSample consolidated report:")
    print(results['reports']['consolidated'].head())
    
    return results


def example_intermediate_access():
    """Demonstrate accessing intermediate pipeline states."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Accessing Intermediate States")
    print("="*60)
    
    # Create data and run analysis
    df = create_sample_data(n_months=4, n_products=1, n_responses_per_month=200)
    
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
    ]
    
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP"],
        capture_intermediates=True,
        verbose=False
    )
    
    # List all intermediate states
    print("Available intermediate states:")
    list_intermediate_states(results['intermediates'])
    
    # Access specific states
    print(f"\nMonthly statistics sample:")
    monthly_stats = get_intermediate_state(results['intermediates'], 'monthly_stats')
    if monthly_stats is not None:
        print(monthly_stats.head(3))
    
    print(f"\nAfter 3MA calculation:")
    ma3_state = get_intermediate_state(results['intermediates'], 'calculate_3ma')
    if ma3_state is not None:
        print(f"DataFrame shape: {ma3_state.shape}")
        if 'ma_results' in ma3_state.attrs:
            print(f"MA results keys: {list(ma3_state.attrs['ma_results'].keys())}")


def example_custom_configuration():
    """Demonstrate custom configuration and error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration & Error Handling")
    print("="*60)
    
    # Create smaller dataset to trigger warnings
    df = create_sample_data(n_months=2, n_products=3, n_responses_per_month=50)
    
    # Custom satisfaction columns with edge cases
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "NONEXISTENT_COL", "calculation": "1-5 T2B"},  # This will cause warnings
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
    ]
    
    # Try with invalid configuration
    try:
        results = run_exhaustive_analytics_v3(
            df=df,
            current_yrmo="202505",
            satisfaction_columns=satisfaction_columns,
            demographic_pivot_columns=["AGE_GROUP", "MISSING_DEMO"],  # This will cause errors
            capture_intermediates=False,
            verbose=True
        )
    except Exception as e:
        print(f"\nExpected error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    
    # Run with valid configuration
    print(f"\nRunning with corrected configuration...")
    satisfaction_columns_fixed = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
    ]
    
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns_fixed,
        demographic_pivot_columns=["AGE_GROUP"],
        capture_intermediates=False,
        verbose=False
    )
    
    print(f"Analysis completed with {len(results['reports']['consolidated'])} results")


def example_v2_compatibility():
    """Demonstrate backward compatibility with v2 format."""
    print("\n" + "="*60)
    print("EXAMPLE 4: v2 Compatibility Mode")
    print("="*60)
    
    # Create data
    df = create_sample_data(n_months=4, n_products=2, n_responses_per_month=250)
    
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
        {"column": "SERVICE_SAT", "calculation": "1-10 AVG"}
    ]
    
    # Run in v2 compatibility mode
    v2_results = run_exhaustive_analytics_v2_compatible(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"]
    )
    
    # Display v2-style output
    print("v2-compatible results structure:")
    for key, value in v2_results.items():
        if isinstance(value, pd.DataFrame):
            print(f"- {key}: {value.shape}")
        else:
            print(f"- {key}: {type(value).__name__}")
    
    print(f"\nSample from consolidated_3ma:")
    if len(v2_results['consolidated_3ma']) > 0:
        print(v2_results['consolidated_3ma'].head(3))


def example_summary_and_export():
    """Demonstrate summary statistics and data export."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Summary Statistics & Export")
    print("="*60)
    
    # Create data with more variation
    df = create_sample_data(n_months=6, n_products=3, n_responses_per_month=400)
    
    satisfaction_columns = [
        {"column": "COST_SAT", "calculation": "1-10 AVG"},
        {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
        {"column": "SERVICE_SAT", "calculation": "1-10 AVG"},
        {"column": "SPEED_SAT", "calculation": "1-5 T2B"},
        {"column": "VALUE_SAT", "calculation": "1-5 T2B"}
    ]
    
    # Run analysis
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo="202505",
        satisfaction_columns=satisfaction_columns,
        demographic_pivot_columns=["AGE_GROUP", "REGION"],
        output_format="detailed"
    )
    
    # Display summary statistics
    if 'summary_stats' in results['reports']:
        print("Summary Statistics:")
        summary = results['reports']['summary_stats']
        for key, value in summary.items():
            print(f"- {key}: {value}")
    
    # Show export example (would normally save to files)
    print(f"\nExport examples:")
    print(f"Main report: {results['reports']['consolidated'].shape[0]} rows")
    print(f"Significant only: {len(results['reports'].get('consolidated_significant', []))} rows")
    
    # Demonstrate different output formats
    for fmt in ['standard', 'detailed', 'summary']:
        test_results = run_exhaustive_analytics_v3(
            df=df.sample(100),  # Smaller sample for speed
            current_yrmo="202505",
            satisfaction_columns=satisfaction_columns[:2],
            demographic_pivot_columns=["AGE_GROUP"],
            output_format=fmt,
            verbose=False
        )
        print(f"Format '{fmt}': {test_results['reports']['consolidated'].shape[1]} columns")


if __name__ == "__main__":
    print("Exhaustive Analytics v3 - Example Usage")
    print("======================================")
    print("This script demonstrates various features of the v3 pipeline.")
    print("Note: This creates sample data for demonstration purposes.\n")
    
    # Run all examples
    try:
        example_basic_usage()
        example_intermediate_access()
        example_custom_configuration()
        example_v2_compatibility()
        example_summary_and_export()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Replace sample data with your actual survey data")
        print("2. Modify satisfaction_columns to match your survey structure")
        print("3. Adjust demographic_pivot_columns as needed")
        print("4. Customize config.py for your requirements")
        print("5. Add custom report steps if needed")
        
    except Exception as e:
        print(f"\nExample failed with error: {type(e).__name__}: {str(e)}")
        print("This might be expected for demonstration purposes.")