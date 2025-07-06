#!/usr/bin/env python3
"""
Enhanced Count Validation Test for Demographic Reports

This script specifically tests that demographic counts follow the sum rule:
For each product, the sum of counts across all values within a demographic field
should be consistent across different demographic fields.

Example: Product1 + Demo1(all values) = Product1 + Demo2(all values)
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3


def test_count_sum_rule():
    """Test that demographic counts follow the sum rule across different demographic fields."""
    print("🧮 COUNT SUM RULE VALIDATION TEST")
    print("=" * 60)
    
    # Create controlled test data
    np.random.seed(42)
    data = []
    
    # ProductA: 300 responses with specific distribution
    demographics = {
        'demo1': ['D1_Val1', 'D1_Val2', 'D1_Val3'],  # 100, 120, 80
        'demo2': ['D2_ValA', 'D2_ValB'],             # 150, 150  
        'demo3': ['D3_X', 'D3_Y', 'D3_Z'],           # 75, 100, 125
        'demo4': ['D4_Alpha', 'D4_Beta']             # 180, 120
    }
    
    # Generate data with controlled distributions
    response_id = 0
    for demo1_val in demographics['demo1']:
        for demo2_val in demographics['demo2']:
            for demo3_val in demographics['demo3']:
                for demo4_val in demographics['demo4']:
                    # Create specific counts to test the sum rule
                    if demo1_val == 'D1_Val1':
                        count = 8 if demo2_val == 'D2_ValA' else 9
                    elif demo1_val == 'D1_Val2':
                        count = 10 if demo2_val == 'D2_ValA' else 10
                    else:  # D1_Val3
                        count = 7 if demo2_val == 'D2_ValA' else 6
                    
                    for _ in range(count):
                        data.append({
                            'YRMO': '202505',
                            'productColX': 'ProductA',
                            'demo1': demo1_val,
                            'demo2': demo2_val,
                            'demo3': demo3_val,
                            'demo4': demo4_val,
                            'avg1': 7.0 + np.random.normal(0, 0.5),
                            't2b1': np.random.choice([1, 2, 3, 4, 5])
                        })
    
    df = pd.DataFrame(data)
    total_responses = len(df)
    print(f"📊 Generated {total_responses} total responses for ProductA")
    
    # Calculate expected counts manually
    expected_counts = {}
    for demo_field, values in demographics.items():
        expected_counts[demo_field] = {}
        for value in values:
            count = len(df[df[demo_field] == value])
            expected_counts[demo_field][value] = count
            print(f"   Expected {demo_field}={value}: {count}")
    
    print()
    
    # Run pipeline
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo='202505',
        satisfaction_columns=[
            {'column': 'avg1', 'calculation': '1-10 AVG'},
            {'column': 't2b1', 'calculation': '1-5 T2B'}
        ],
        demographic_pivot_columns=['demo1', 'demo2', 'demo3', 'demo4'],
        product_column='productColX',
        verbose=False
    )
    
    demo_report = results['reports']['demographic']
    print(f"🔍 Pipeline Results:")
    
    # Validate counts and check sum rule
    pipeline_counts = {}
    total_sums = {}
    
    for demo_field in demographics.keys():
        pipeline_counts[demo_field] = {}
        demo_rows = demo_report[demo_report['demographic_field'] == demo_field]
        
        field_sum = 0
        for _, row in demo_rows.iterrows():
            demo_value = row['demographic_value']
            pipeline_count = row['avg1__1MA_count']
            expected_count = expected_counts[demo_field][demo_value]
            
            pipeline_counts[demo_field][demo_value] = pipeline_count
            field_sum += pipeline_count
            
            # Check individual count
            if pipeline_count == expected_count:
                status = "✅"
            else:
                status = f"❌ (expected {expected_count})"
            
            print(f"   {demo_field}={demo_value}: {pipeline_count} {status}")
        
        total_sums[demo_field] = field_sum
        print(f"   {demo_field} TOTAL SUM: {field_sum}")
        print()
    
    # Validate sum rule: all demographic fields should sum to same total
    print("🎯 SUM RULE VALIDATION:")
    reference_sum = total_sums[list(demographics.keys())[0]]
    all_match = True
    
    for demo_field, field_sum in total_sums.items():
        if field_sum == reference_sum:
            print(f"✅ {demo_field} sum: {field_sum} = {reference_sum}")
        else:
            print(f"❌ {demo_field} sum: {field_sum} ≠ {reference_sum}")
            all_match = False
    
    print()
    if all_match:
        print("🎉 SUCCESS: All demographic fields have consistent sums!")
        print(f"📊 Total responses per product: {reference_sum}")
    else:
        print("💥 FAILURE: Demographic field sums are inconsistent!")
    
    return all_match


def test_multiple_products():
    """Test count validation across multiple products."""
    print("\n🏢 MULTIPLE PRODUCTS COUNT TEST")
    print("=" * 60)
    
    # Create data for multiple products
    np.random.seed(123)
    data = []
    
    products = ['ProductX', 'ProductY', 'ProductZ']
    product_sizes = [200, 150, 100]  # Different sizes for each product
    
    for i, (product, size) in enumerate(zip(products, product_sizes)):
        for j in range(size):
            data.append({
                'YRMO': '202505',
                'productColX': product,
                'demo1': np.random.choice(['A1', 'A2']),
                'demo2': np.random.choice(['B1', 'B2', 'B3']),
                'avg1': 7.0 + i * 0.5 + np.random.normal(0, 0.3)  # Slight product differences
            })
    
    df = pd.DataFrame(data)
    print(f"📊 Generated {len(df)} total responses across {len(products)} products")
    
    # Calculate expected counts per product
    for product in products:
        product_df = df[df['productColX'] == product]
        print(f"   {product}: {len(product_df)} responses")
    
    # Run pipeline
    results = run_exhaustive_analytics_v3(
        df=df,
        current_yrmo='202505',
        satisfaction_columns=[{'column': 'avg1', 'calculation': '1-10 AVG'}],
        demographic_pivot_columns=['demo1', 'demo2'],
        product_column='productColX',
        verbose=False
    )
    
    demo_report = results['reports']['demographic']
    
    # Validate sum rule for each product
    print("\n🔍 Per-Product Sum Rule Validation:")
    all_products_valid = True
    
    for product in products:
        product_rows = demo_report[demo_report['product'] == product]
        
        # Calculate sums for each demographic field
        demo1_sum = product_rows[product_rows['demographic_field'] == 'demo1']['avg1__1MA_count'].sum()
        demo2_sum = product_rows[product_rows['demographic_field'] == 'demo2']['avg1__1MA_count'].sum()
        
        expected_count = len(df[df['productColX'] == product])
        
        print(f"   {product}:")
        print(f"     demo1 sum: {demo1_sum}")
        print(f"     demo2 sum: {demo2_sum}")
        print(f"     expected: {expected_count}")
        
        if demo1_sum == demo2_sum == expected_count:
            print(f"     ✅ PASS: All sums match")
        else:
            print(f"     ❌ FAIL: Sums don't match")
            all_products_valid = False
    
    return all_products_valid


def main():
    """Run all count validation tests."""
    print("🧪 DEMOGRAPHIC COUNT VALIDATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Sum rule validation
        test1_pass = test_count_sum_rule()
        
        # Test 2: Multiple products
        test2_pass = test_multiple_products()
        
        # Summary
        print(f"\n🏁 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Sum Rule Test: {'PASSED' if test1_pass else 'FAILED'}")
        print(f"✅ Multiple Products Test: {'PASSED' if test2_pass else 'FAILED'}")
        
        if test1_pass and test2_pass:
            print(f"\n🎉 ALL COUNT VALIDATION TESTS PASSED!")
            print(f"✅ Demographic counts are now calculated correctly")
            print(f"✅ Sum rule is properly enforced across all demographic fields")
        else:
            print(f"\n💥 SOME TESTS FAILED - Count calculation still has issues")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()