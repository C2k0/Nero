#!/usr/bin/env python3
"""
Test script to verify the v3 fixes work correctly.

This script tests:
1. Import structure fixes
2. DataFrame.attrs reliability
3. Error handling standardization

Run from the parent directory of exhaustive_analytics_v3/
"""

import pandas as pd
import numpy as np

def test_imports():
    """Test that all imports work correctly."""
    print("Testing import structure fixes...")
    
    try:
        from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3
        from exhaustive_analytics_v3.pipeline import run_pipeline, safe_get_attr, safe_set_attr
        from exhaustive_analytics_v3.steps import validation, statistics, reporting
        from exhaustive_analytics_v3 import config
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_attrs_safety():
    """Test DataFrame.attrs safety functions."""
    print("Testing DataFrame.attrs safety...")
    
    from exhaustive_analytics_v3.pipeline import safe_get_attr, safe_set_attr
    
    # Create test DataFrame
    df = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test safe setting
    safe_set_attr(df, 'test_key', 'test_value')
    
    # Test safe getting
    value = safe_get_attr(df, 'test_key')
    assert value == 'test_value', f"Expected 'test_value', got {value}"
    
    # Test default value
    default_value = safe_get_attr(df, 'missing_key', default='default')
    assert default_value == 'default', f"Expected 'default', got {default_value}"
    
    # Test error on missing critical attribute
    try:
        safe_get_attr(df, 'missing_critical')
        print("❌ Should have raised ValueError")
        return False
    except ValueError:
        pass  # Expected
    
    print("✅ DataFrame.attrs safety functions work correctly")
    return True

def test_error_handling():
    """Test standardized error handling."""
    print("Testing error handling standardization...")
    
    from exhaustive_analytics_v3.steps.validation import handle_validation_issue
    
    # Test warning mode (should not raise)
    try:
        handle_validation_issue("Test Issue", ["Detail 1", "Detail 2"], should_fail=False)
        print("✅ Warning mode works correctly")
    except Exception as e:
        print(f"❌ Warning mode failed: {e}")
        return False
    
    # Test error mode (should raise)
    try:
        handle_validation_issue("Test Issue", ["Detail 1", "Detail 2"], should_fail=True)
        print("❌ Error mode should have raised ValueError")
        return False
    except ValueError:
        print("✅ Error mode works correctly")
    
    return True

def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("Testing basic pipeline functionality...")
    
    try:
        from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3
        
        # Create minimal sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            "YRMO": ["202505"] * 100 + ["202504"] * 100 + ["202503"] * 100,
            "product": np.random.choice(["Product A", "Product B"], 300),
            "AGE_GROUP": np.random.choice(["18-34", "35-54"], 300),
            "COST_SAT": np.random.randint(1, 11, 300),
            "QUALITY_SAT": np.random.randint(1, 6, 300)
        })
        
        satisfaction_columns = [
            {"column": "COST_SAT", "calculation": "1-10 AVG"},
            {"column": "QUALITY_SAT", "calculation": "1-5 T2B"}
        ]
        
        # Run analysis
        results = run_exhaustive_analytics_v3(
            df=sample_data,
            current_yrmo="202505",
            satisfaction_columns=satisfaction_columns,
            demographic_pivot_columns=["AGE_GROUP"],
            verbose=False,
            capture_intermediates=False
        )
        
        # Basic validation
        assert 'reports' in results, "Missing reports in results"
        assert 'consolidated' in results['reports'], "Missing consolidated report"
        
        print("✅ Basic pipeline functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running v3 fixes verification tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_attrs_safety,
        test_error_handling,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All fixes verified successfully!")
        print("\nThe v3 implementation is ready for use.")
        print("\nNext steps:")
        print("1. Run with your actual survey data")
        print("2. Customize config.py for your requirements")
        print("3. Add custom report steps if needed")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)