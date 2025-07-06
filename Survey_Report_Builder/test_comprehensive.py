#!/usr/bin/env python3
"""
Comprehensive Test Script for Exhaustive Analytics v3

This script generates synthetic survey data and runs it through the v3 pipeline
to validate the output formats, data integrity, and statistical calculations.

Features:
- Generates fake survey data with 4 demographics, 3 products, 10 satisfaction metrics
- Tests both 1-10 AVG and 1-5 T2B calculation types
- Validates wide demographic report format (100+ columns)
- Spot checks calculations by manually verifying averages
- Tests multiple months of data for MA calculations
- Comprehensive output validation
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3


def generate_test_data(n_months: int = 6, n_responses_per_month: int = 800) -> pd.DataFrame:
    """
    Generate synthetic survey data for testing.
    
    Args:
        n_months: Number of months of data to generate
        n_responses_per_month: Approximate responses per month per product
    
    Returns:
        DataFrame with synthetic survey data
    """
    print(f"🔄 Generating test data: {n_months} months, ~{n_responses_per_month} responses/month")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate YRMOs (ending in current month 202505)
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
    print(f"📅 Date range: {yrmos[0]} to {yrmos[-1]}")
    
    # Define test structure
    products = ["ProductA", "ProductB", "ProductC"]
    
    # 4 demographic columns as requested
    demographics = {
        'demo1': ['Value1A', 'Value1B', 'Value1C'],
        'demo2': ['Value2A', 'Value2B'],
        'demo3': ['Value3A', 'Value3B', 'Value3C', 'Value3D'],
        'demo4': ['Value4A', 'Value4B', 'Value4C']
    }
    
    # 10 satisfaction metrics: 6 x "1-10 AVG" + 4 x "1-5 T2B"
    satisfaction_metrics = {
        # 1-10 AVG metrics
        'avg1': {'type': '1-10 AVG', 'base_score': 7.2},
        'avg2': {'type': '1-10 AVG', 'base_score': 6.8},
        'avg3': {'type': '1-10 AVG', 'base_score': 7.5},
        'avg4': {'type': '1-10 AVG', 'base_score': 6.9},
        'avg5': {'type': '1-10 AVG', 'base_score': 7.1},
        'avg6': {'type': '1-10 AVG', 'base_score': 7.3},
        
        # 1-5 T2B metrics (we'll store raw 1-5 scores, T2B calculated later)
        't2b1': {'type': '1-5 T2B', 'base_score': 3.2},
        't2b2': {'type': '1-5 T2B', 'base_score': 3.4},
        't2b3': {'type': '1-5 T2B', 'base_score': 3.1},
        't2b4': {'type': '1-5 T2B', 'base_score': 3.3}
    }
    
    data = []
    
    for yrmo_idx, yrmo in enumerate(yrmos):
        print(f"  📊 Generating data for {yrmo}")
        
        for product in products:
            # Add some realistic trends and product differences
            product_modifier = {'ProductA': 0.3, 'ProductB': 0.0, 'ProductC': -0.2}[product]
            trend_modifier = yrmo_idx * 0.05  # Slight improvement over time
            
            for _ in range(n_responses_per_month):
                # Random demographic values
                demo_values = {demo: np.random.choice(values) 
                              for demo, values in demographics.items()}
                
                # Generate satisfaction scores with realistic variance
                sat_scores = {}
                for metric, config in satisfaction_metrics.items():
                    base = config['base_score'] + product_modifier + trend_modifier
                    
                    if config['type'] == '1-10 AVG':
                        # Generate 1-10 scale scores
                        score = np.random.normal(base, 1.2)
                        sat_scores[metric] = int(np.clip(score, 1, 10))
                    else:  # 1-5 T2B
                        # Generate 1-5 scale scores
                        score = np.random.normal(base, 0.8)
                        sat_scores[metric] = int(np.clip(score, 1, 5))
                
                # Create row
                row = {
                    'YRMO': yrmo,
                    'productColX': product,  # Custom product column name as requested
                    **demo_values,
                    **sat_scores
                }
                
                data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df):,} total responses")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🏢 Products: {df['productColX'].unique().tolist()}")
    
    return df


def define_satisfaction_columns() -> List[Dict[str, str]]:
    """Define satisfaction column configurations."""
    return [
        # 1-10 AVG metrics
        {"column": "avg1", "calculation": "1-10 AVG"},
        {"column": "avg2", "calculation": "1-10 AVG"},
        {"column": "avg3", "calculation": "1-10 AVG"},
        {"column": "avg4", "calculation": "1-10 AVG"},
        {"column": "avg5", "calculation": "1-10 AVG"},
        {"column": "avg6", "calculation": "1-10 AVG"},
        
        # 1-5 T2B metrics
        {"column": "t2b1", "calculation": "1-5 T2B"},
        {"column": "t2b2", "calculation": "1-5 T2B"},
        {"column": "t2b3", "calculation": "1-5 T2B"},
        {"column": "t2b4", "calculation": "1-5 T2B"}
    ]


def run_pipeline_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the v3 pipeline and return results."""
    print("\n🚀 Running Exhaustive Analytics v3 Pipeline")
    print("=" * 60)
    
    satisfaction_columns = define_satisfaction_columns()
    demographic_columns = ['demo1', 'demo2', 'demo3', 'demo4']
    
    try:
        results = run_exhaustive_analytics_v3(
            df=df,
            current_yrmo="202505",
            satisfaction_columns=satisfaction_columns,
            demographic_pivot_columns=demographic_columns,
            product_column="productColX",  # Custom column name
            capture_intermediates=True,
            verbose=True
        )
        
        print("✅ Pipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {type(e).__name__}: {str(e)}")
        raise


def validate_output_structure(results: Dict[str, Any]) -> None:
    """Validate the structure and format of pipeline outputs."""
    print("\n🔍 Validating Output Structure")
    print("=" * 60)
    
    # Check main structure
    expected_keys = ['final_df', 'intermediates', 'reports', 'monthly_stats', 'ma_results']
    for key in expected_keys:
        if key in results:
            print(f"✅ {key}: Present")
        else:
            print(f"❌ {key}: Missing")
    
    # Check reports structure
    reports = results.get('reports', {})
    expected_reports = ['consolidated', 'demographic', 'consolidated_significant', 
                       'demographic_significant', 'summary_stats']
    
    print(f"\n📊 Report Validation:")
    for report_name in expected_reports:
        if report_name in reports:
            report = reports[report_name]
            if isinstance(report, pd.DataFrame):
                print(f"✅ {report_name}: DataFrame with shape {report.shape}")
            else:
                print(f"⚠️  {report_name}: {type(report).__name__} (not DataFrame)")
        else:
            print(f"❌ {report_name}: Missing")
    
    # Validate demographic wide report specifically
    if 'demographic' in reports:
        demo_report = reports['demographic']
        print(f"\n🏗️  Demographic Wide Report Analysis:")
        print(f"   Shape: {demo_report.shape}")
        print(f"   Columns: {demo_report.shape[1]} (should be 100+)")
        
        # Check for expected column patterns
        column_patterns = ['__1MA_current', '__3MA_current', '__6MA_current', 
                          '__3MA_diff', '__3MA_sig95']
        pattern_counts = {}
        for pattern in column_patterns:
            count = len([col for col in demo_report.columns if pattern in col])
            pattern_counts[pattern] = count
            print(f"   Columns with '{pattern}': {count}")
        
        # Sample of column names
        print(f"   Sample columns: {demo_report.columns[:10].tolist()}")
        
        return demo_report
    else:
        print("❌ No demographic report found for validation")
        return None


def spot_check_calculations(df: pd.DataFrame, results: Dict[str, Any]) -> None:
    """Perform spot checks on calculations to verify accuracy."""
    print("\n🎯 Spot Check Calculations")
    print("=" * 60)
    
    # Get the demographic report
    demo_report = results['reports'].get('demographic')
    if demo_report is None:
        print("❌ No demographic report available for spot checking")
        return
    
    # Check a few random combinations
    current_yrmo = "202505"
    test_cases = [
        {'product': 'ProductA', 'demo_col': 'demo1', 'demo_val': 'Value1A', 'metric': 'avg1'},
        {'product': 'ProductB', 'demo_col': 'demo2', 'demo_val': 'Value2B', 'metric': 't2b1'},
        {'product': 'ProductC', 'demo_col': 'demo3', 'demo_val': 'Value3C', 'metric': 'avg3'}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['product']} × {case['demo_col']}={case['demo_val']} × {case['metric']}")
        
        # Filter raw data for this combination
        filtered_data = df[
            (df['YRMO'] == current_yrmo) &
            (df['productColX'] == case['product']) &
            (df[case['demo_col']] == case['demo_val'])
        ][case['metric']]
        
        if len(filtered_data) > 0:
            # Calculate manual average/T2B
            if case['metric'].startswith('avg'):
                manual_calc = filtered_data.mean()
                calc_type = "Average"
            else:  # T2B
                t2b_count = len(filtered_data[filtered_data.isin([4, 5])])
                manual_calc = (t2b_count / len(filtered_data)) * 100
                calc_type = "T2B %"
            
            # Find in demographic report
            report_row = demo_report[
                (demo_report['product'] == case['product']) &
                (demo_report['demographic_field'] == case['demo_col']) &
                (demo_report['demographic_value'] == case['demo_val'])
            ]
            
            if len(report_row) > 0:
                # Check 1MA current value (should match our calculation)
                column_name = f"{case['metric']}__1MA_current"
                if column_name in demo_report.columns:
                    report_value = report_row.iloc[0][column_name]
                    difference = abs(manual_calc - report_value) if not pd.isna(report_value) else float('inf')
                    
                    print(f"   📊 Raw data: {len(filtered_data)} responses")
                    print(f"   🧮 Manual {calc_type}: {manual_calc:.2f}")
                    print(f"   📈 Report 1MA: {report_value:.2f}" if not pd.isna(report_value) else "   📈 Report 1MA: NaN")
                    print(f"   🎯 Difference: {difference:.4f}")
                    
                    if difference < 0.01:  # Allow small floating point differences
                        print(f"   ✅ PASS: Values match within tolerance")
                    else:
                        print(f"   ❌ FAIL: Values differ by {difference:.4f}")
                else:
                    print(f"   ❌ Column '{column_name}' not found in report")
            else:
                print(f"   ❌ No matching row found in demographic report")
        else:
            print(f"   ⚠️  No raw data found for this combination")


def validate_wide_format_columns(demo_report: pd.DataFrame) -> None:
    """Validate that the wide format has the expected column structure."""
    print("\n🏗️  Wide Format Column Structure Validation")
    print("=" * 60)
    
    satisfaction_metrics = ['avg1', 'avg2', 'avg3', 'avg4', 'avg5', 'avg6', 
                           't2b1', 't2b2', 't2b3', 't2b4']
    ma_periods = [1, 3, 6]
    suffixes = ['current', 'count', 'previous', 'diff', 'statsig', 'sig95', 'sig90']
    
    # Calculate expected columns
    base_columns = ['product', 'demographic_field', 'demographic_value']
    expected_metric_columns = []
    
    for metric in satisfaction_metrics:
        for ma_period in ma_periods:
            for suffix in suffixes:
                expected_metric_columns.append(f"{metric}__{ma_period}MA_{suffix}")
    
    total_expected = len(base_columns) + len(expected_metric_columns)
    print(f"📊 Expected total columns: {total_expected}")
    print(f"📊 Actual columns: {demo_report.shape[1]}")
    
    # Check base columns
    missing_base = [col for col in base_columns if col not in demo_report.columns]
    if missing_base:
        print(f"❌ Missing base columns: {missing_base}")
    else:
        print(f"✅ All base columns present: {base_columns}")
    
    # Check metric columns
    present_metric_columns = [col for col in expected_metric_columns if col in demo_report.columns]
    missing_metric_columns = [col for col in expected_metric_columns if col not in demo_report.columns]
    
    print(f"✅ Metric columns present: {len(present_metric_columns)}/{len(expected_metric_columns)}")
    
    if missing_metric_columns:
        print(f"❌ Missing metric columns: {len(missing_metric_columns)}")
        print(f"   First 10 missing: {missing_metric_columns[:10]}")
    
    # Sample some specific expected columns
    sample_expected = [
        'avg1__3MA_current', 'avg1__3MA_diff', 'avg1__3MA_sig95',
        't2b1__1MA_current', 't2b1__6MA_previous', 't2b4__3MA_statsig'
    ]
    
    print(f"\n🔍 Sample Column Check:")
    for col in sample_expected:
        if col in demo_report.columns:
            print(f"   ✅ {col}: Present")
        else:
            print(f"   ❌ {col}: Missing")


def main():
    """Main test execution function."""
    print("🧪 Exhaustive Analytics v3 - Comprehensive Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Generate test data
        df = generate_test_data(n_months=6, n_responses_per_month=800)
        
        # Step 2: Run pipeline
        results = run_pipeline_test(df)
        
        # Step 3: Validate structure
        demo_report = validate_output_structure(results)
        
        # Step 4: Validate wide format columns
        if demo_report is not None:
            validate_wide_format_columns(demo_report)
        
        # Step 5: Spot check calculations
        spot_check_calculations(df, results)
        
        # Summary
        print(f"\n🎉 Test Summary")
        print("=" * 60)
        print(f"✅ Data Generation: PASSED")
        print(f"✅ Pipeline Execution: PASSED")
        
        if results.get('reports', {}).get('demographic') is not None:
            demo_shape = results['reports']['demographic'].shape
            print(f"✅ Demographic Report: PASSED ({demo_shape[0]} rows × {demo_shape[1]} columns)")
            
            if demo_shape[1] >= 100:
                print(f"✅ Wide Format: PASSED (100+ columns achieved)")
            else:
                print(f"⚠️  Wide Format: PARTIAL ({demo_shape[1]} columns, expected 100+)")
        else:
            print(f"❌ Demographic Report: FAILED")
        
        print(f"\n🏁 All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()