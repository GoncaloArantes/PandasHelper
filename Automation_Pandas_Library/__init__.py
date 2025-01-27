"""
YourPackageName
---------------
A package for advanced data manipulation and validation.

Author: Your Name
Email: your.email@example.com
Version: 0.1.0
"""

# Public functions
from .PandasHelper import (
    all_functions,
    data_compacting,
    datetime_handling,
    dummy_creation,
    duplicate_handling,
    highest_lowest_correlation,
    null_handling,
    summary,
    validate_date,
)

# Metadata
__version__ = "0.0.1"
__author__ = "Gonçalo Arantes"
__license__ = "MIT"

# Expose public API for convenience
__all__ = [
    "all_functions",
    "data_compacting",
    "datetime_handling",
    "dummy_creation",
    "duplicate_handling",
    "highest_lowest_correlation",
    "null_handling",
    "summary",
    "validate_date",
]
