"""
FLOWFINDER - Watershed Delineation Tool
======================================

A high-performance watershed delineation tool that takes lat/lon coordinates
and returns watershed boundaries with high spatial accuracy.

Key Features:
- Fast watershed delineation (<30s target)
- High spatial accuracy (95% IOU target)
- Command-line interface
- Python API
- Support for 10m DEM data

Author: FLOWFINDER Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FLOWFINDER Team"
__license__ = "MIT"

from .core import FlowFinder
from .exceptions import FlowFinderError, DEMError, WatershedError

__all__ = [
    "FlowFinder",
    "FlowFinderError", 
    "DEMError",
    "WatershedError"
] 