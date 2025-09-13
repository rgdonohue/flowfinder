"""
FLOWFINDER - Python Watershed Delineation Tool
===============================================

A Python implementation of watershed delineation algorithms using standard 
hydrological methods. Extracts watershed boundaries from Digital Elevation 
Models (DEMs) with performance monitoring and validation tools.

Key Features:
- D8 flow direction with priority-flood depression filling
- O(n) flow accumulation using topological sorting
- Watershed extraction from pour points (lat/lon coordinates)
- Real-time performance monitoring (runtime, memory usage)
- Topology validation and quality assessment
- Python API and command-line interface
- Support for GeoTIFF DEM data

Author: FLOWFINDER Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FLOWFINDER Team"
__license__ = "MIT"

from .core import FlowFinder
from .exceptions import FlowFinderError, DEMError, WatershedError

__all__ = ["FlowFinder", "FlowFinderError", "DEMError", "WatershedError"]
