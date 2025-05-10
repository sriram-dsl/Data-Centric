from datetime import datetime
import pandas as pd

def format_value(value):
    """Format different data types consistently for text embedding."""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if pd.api.types.is_datetime64_any_dtype(type(value)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, float) and value == int(value) else str(value)
    return str(value)
