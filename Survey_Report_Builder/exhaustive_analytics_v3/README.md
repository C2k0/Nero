# Exhaustive Analytics v3.0.0

A modular, maintainable survey data analysis pipeline with statistical significance testing.

## Quick Start

```python
# Import the module (run from parent directory of exhaustive_analytics_v3)
from exhaustive_analytics_v3.main import run_exhaustive_analytics_v3
import pandas as pd

# Load your survey data
df = pd.read_csv('survey_data.csv')

# Define satisfaction metrics
satisfaction_columns = [
    {"column": "COST_SAT", "calculation": "1-10 AVG"},
    {"column": "QUALITY_SAT", "calculation": "1-5 T2B"},
    {"column": "SERVICE_SAT", "calculation": "1-10 AVG"}
]

# Run the analysis
results = run_exhaustive_analytics_v3(
    df=df,
    current_yrmo="202505",
    satisfaction_columns=satisfaction_columns,
    demographic_pivot_columns=["AGE_GROUP", "REGION"]
)

# Save the main report
results['reports']['consolidated'].to_csv('satisfaction_report.csv', index=False)

# Access significant changes only
significant_changes = results['reports']['consolidated_significant']
print(f"Found {len(significant_changes)} significant changes")
```

## Project Structure

```
exhaustive_analytics_v3/
│
├── __init__.py                 # Package initialization
├── config.py                   # All configuration constants
├── pipeline.py                 # Core pipeline runner with state tracking
├── main.py                     # Main orchestrator and entry point
│
├── steps/                      # Pipeline transformation steps
│   ├── __init__.py
│   ├── validation.py          # Data validation and quality checks
│   ├── statistics.py          # MA calculations and significance testing
│   └── reporting.py           # Report generation and formatting
│
└── utils/                      # Helper functions
    ├── __init__.py
    ├── errors.py              # Error formatting utilities
    └── stats_helpers.py       # Statistical calculation helpers
```

## How It Works

### Pipeline Architecture

The system uses a functional pipeline pattern where each step:
1. Takes a DataFrame as input
2. Performs a specific transformation
3. Returns the modified DataFrame
4. Optionally stores results in DataFrame attributes

### Execution Flow

1. **Validation Phase**
   - Check required columns exist
   - Validate data ranges (1-10 for averages, 1-5 for T2B)
   - Verify YRMO format
   - Check sample sizes

2. **Statistics Phase**
   - Calculate monthly statistics by product/demographic groups
   - Compute moving averages (1MA, 3MA, 6MA)
   - Perform period-over-period comparisons
   - Calculate statistical significance

3. **Reporting Phase**
   - Build demographic breakout reports
   - Create consolidated product reports
   - Filter for significant changes
   - Generate summary statistics

### Data Flow Diagram

```
Input DataFrame
    ↓
[Validation Steps]
    ↓
[Monthly Statistics] → stored in df.attrs['monthly_stats']
    ↓
[MA Calculations] → stored in df.attrs['ma_results']
    ↓
[Comparisons] → stored in df.attrs['comparison_results']
    ↓
[Report Generation]
    ↓
Output Reports (in results['reports'])
```

## Adding New Reports

### Method 1: Add to Main Pipeline

Edit the `pipeline_steps` list in `main.py`:

```python
# In main.py, add your step to pipeline_steps:
pipeline_steps = [
    # ... existing steps ...
    
    # Add your custom report step
    ('custom_report', my_module.generate_custom_report, {
        'param1': value1,
        'param2': value2
    }),
    
    # ... remaining steps ...
]
```

### Method 2: Use Helper Function

```python
from exhaustive_analytics_v3.main import add_custom_report_step
from my_reports import generate_trend_report

# Get default pipeline steps
pipeline_steps = [...]  # Your pipeline

# Add custom report
pipeline_steps = add_custom_report_step(
    pipeline_steps,
    'trend_analysis',
    generate_trend_report,
    {'window': 12, 'min_periods': 6},
    position='before_output'  # or 'after_stats'
)
```

### Method 3: Create Custom Pipeline

```python
def run_custom_analysis(df, **kwargs):
    # Define your own pipeline
    custom_steps = [
        ('validate', validation.check_required_columns, {}),
        ('monthly_stats', statistics.calculate_monthly_stats, {}),
        ('custom_calc', my_calculations.special_metric, {'threshold': 0.8}),
        ('custom_report', my_reports.build_report, {})
    ]
    
    final_df, intermediates = run_pipeline(df, custom_steps)
    return final_df, intermediates
```

## Accessing Data Throughout

### 1. Intermediate Pipeline States

```python
# Run with capture_intermediates=True (default)
results = run_exhaustive_analytics_v3(df, ..., capture_intermediates=True)

# Access any intermediate state
intermediates = results['intermediates']

# List all captured states
from exhaustive_analytics_v3.pipeline import list_intermediate_states
list_intermediate_states(intermediates)

# Get specific state
validation_output = intermediates['01_validate_columns']
ma_calculated = intermediates['07_calculate_3ma']

# Or use helper function
from exhaustive_analytics_v3.pipeline import get_intermediate_state
monthly_stats_df = get_intermediate_state(intermediates, 'monthly_stats')
```

### 2. Statistical Results

```python
# Access pre-calculated monthly statistics
monthly_stats = results['monthly_stats']

# Access MA results by period
ma_results = results['ma_results']  # Dict with keys 1, 3, 6
ma_3_month = ma_results[3]
```

### 3. Generated Reports

```python
# All reports are in results['reports']
reports = results['reports']

# Main consolidated report
consolidated = reports['consolidated']

# Significant changes only
significant = reports['consolidated_significant']

# Demographic breakdown
demographic = reports.get('demographic', pd.DataFrame())

# Summary statistics
summary = reports.get('summary_stats', pd.Series())
```

### 4. Within Custom Functions

```python
def my_custom_step(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Access stored attributes
    monthly_stats = df.attrs.get('monthly_stats')
    current_yrmo = df.attrs.get('current_yrmo')
    
    # Your processing...
    
    # Store your results
    df.attrs['my_custom_results'] = custom_df
    
    return df
```

## Configuration

Edit `config.py` to customize:

- Sample size thresholds
- Confidence levels (90%, 95%)
- Column naming conventions
- Output formatting options
- Validation behavior

Key settings:
```python
MIN_SAMPLE_SIZE = 30
MA_PERIODS = [1, 3, 6]
CONFIDENCE_LEVELS = {'high': 0.95, 'medium': 0.90}
```

## Error Handling

The pipeline provides detailed error messages:

```
============================================================
ERROR: Required columns not found in DataFrame
============================================================
  Missing columns: ['AGE_GROUP', 'INCOME']
  Available columns: ['YRMO', 'product', 'COST_SAT', ...]
  Total required: 8
  Total missing: 2
============================================================
```

Pipeline errors show exactly where failures occur:
```
Pipeline failed at step 3: 'calculate_monthly_stats'
Function: calculate_monthly_stats
Parameters: {'satisfaction_columns': [...]}
Error: KeyError: 'COST_SAT'
```

## Backward Compatibility

For code expecting v2 output format:

```python
from exhaustive_analytics_v3.main import run_exhaustive_analytics_v2_compatible

# Returns v2-style dictionary
v2_results = run_exhaustive_analytics_v2_compatible(
    df, current_yrmo, satisfaction_columns, demographic_pivot_columns
)

# Access using v2 keys
precalc_stats = v2_results['precalc_stats']
consolidated_3ma = v2_results['consolidated_3ma']
```

## Best Practices

1. **Always validate first**: The pipeline validates data by default, but you can add custom validation steps

2. **Use configuration**: Don't hardcode values; add them to `config.py`

3. **Store in attributes**: Use `df.attrs` to pass data between steps without modifying the DataFrame

4. **Document your functions**: Follow the existing docstring format for consistency

5. **Handle missing data**: Check for required attributes before using them

6. **Test incrementally**: Use intermediate states to verify each step

## Performance Tips

- Set `capture_intermediates=False` for production runs to save memory
- Use `verbose=False` to reduce console output
- Consider chunking for very large datasets (>1M rows)
- Pre-filter data to required columns before processing

## Troubleshooting

**Missing monthly stats**: Ensure `calculate_monthly_stats` runs before MA calculations

**No significant results**: Check sample sizes and significance thresholds

**Memory issues**: Disable intermediate capture and process in chunks

**Slow performance**: Profile using intermediate timestamps; consider optimizing groupby operations

## Version History

- v3.0.0 (2025-01-06): Complete refactor with modular pipeline architecture
- v2.0.0: Original monolithic implementation