"""
Configuration module for Exhaustive Analytics v3

This module contains all configuration constants, settings, and parameters
used throughout the analytics pipeline. Modify these values to customize
the analysis behavior without changing the core logic.
"""

# ============================================================================
# SAMPLE SIZE REQUIREMENTS
# ============================================================================
MIN_SAMPLE_SIZE = 30  # Below this, results flagged as "insufficient data"
SMALL_SAMPLE_THRESHOLD = 100  # Use t-distribution below this
MEDIUM_SAMPLE_THRESHOLD = 500  # Standard tests appropriate
# Large sample: 500+, high confidence

# ============================================================================
# MOVING AVERAGE PERIODS
# ============================================================================
MA_PERIODS = [1, 3, 6]  # Moving average periods to calculate

# ============================================================================
# STATISTICAL SIGNIFICANCE LEVELS
# ============================================================================
CONFIDENCE_LEVELS = {
    'high': 0.95,    # 95% confidence level
    'medium': 0.90   # 90% confidence level
}

# Default significance thresholds for flagging
SIG_95_THRESHOLD = 0.95
SIG_90_THRESHOLD = 0.90

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================
# Required columns in input DataFrame
REQUIRED_BASE_COLUMNS = ['YRMO', 'product']

# Default satisfaction columns configuration
# Each dict must have 'column' and 'calculation' keys
DEFAULT_SATISFACTION_COLUMNS = [
    {"column": "COST_SAT", "calculation": "1-10 AVG"},
    {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
    {"column": "SERVICE_SAT", "calculation": "1-10 AVG"},
    {"column": "SPEED_SAT", "calculation": "1-5 T2B"},
    {"column": "RELIABILITY_SAT", "calculation": "1-10 AVG"}
]

# Default demographic pivot columns
DEFAULT_DEMOGRAPHIC_COLUMNS = ["AGE_GROUP", "REGION"]

# ============================================================================
# CALCULATION TYPES
# ============================================================================
# Valid calculation types and their value ranges
CALCULATION_TYPES = {
    "1-10 AVG": {
        "type": "average",
        "valid_range": range(1, 11),  # 1 through 10
        "description": "Average of 1-10 scale"
    },
    "1-5 T2B": {
        "type": "top2box",
        "valid_range": range(1, 6),   # 1 through 5
        "top_values": [4, 5],          # Values considered "top 2"
        "description": "Percentage of 4s and 5s on 1-5 scale"
    }
}

# ============================================================================
# ERROR HANDLING
# ============================================================================
# Error message formatting
ERROR_SEPARATOR = "=" * 60

# Data validation settings
WARN_ON_INVALID_VALUES = True  # Warn about out-of-range values
FAIL_ON_MISSING_COLUMNS = True  # Raise error if required columns missing
FAIL_ON_INVALID_YRMO = True  # Raise error if YRMO format is invalid
STRICT_VALIDATION = True  # Use strict validation throughout pipeline

# ============================================================================
# OUTPUT FORMATTING
# ============================================================================
# Number formatting
DECIMAL_PLACES = 2  # Decimal places for averages
PERCENTAGE_DECIMAL_PLACES = 1  # Decimal places for percentages

# Column naming conventions
MA_COLUMN_PREFIX = "MA"  # Prefix for moving average columns
SIGNIFICANCE_COLUMN_SUFFIX = "_sig"  # Suffix for significance flags
STATSIG_COLUMN_SUFFIX = "_statsig"  # Suffix for statistical significance values

# ============================================================================
# PIPELINE SETTINGS
# ============================================================================
# Pipeline execution settings
CAPTURE_INTERMEDIATES_DEFAULT = True  # Save intermediate DataFrames by default
VERBOSE_PIPELINE = True  # Print progress messages during pipeline execution

# ============================================================================
# FILE PATHS (Override these in your script)
# ============================================================================
INPUT_FILE = None  # Set this to your input CSV path
OUTPUT_FILE = None  # Set this to your output CSV path

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Memory optimization
COPY_DATAFRAMES = True  # Create copies to preserve original data
CHUNK_SIZE = None  # Set to integer to process large files in chunks

# ============================================================================
# REPORT SETTINGS
# ============================================================================
# Significant items filtering
FILTER_SIGNIFICANT_ONLY = True  # Generate filtered reports with significant changes
SIGNIFICANCE_FILTER_THRESHOLD = 0.95  # Default threshold for filtering

# Report column ordering
REPORT_COLUMN_ORDER = [
    'product',
    'demographic_field',
    'demographic_value',
    'satisfaction_metric'
]