"""
FLOWFINDER Exceptions
====================

Custom exception classes for FLOWFINDER watershed delineation tool.
"""


class FlowFinderError(Exception):
    """Base exception for FLOWFINDER errors."""

    pass


class DEMError(FlowFinderError):
    """Exception raised for DEM-related errors."""

    pass


class WatershedError(FlowFinderError):
    """Exception raised for watershed delineation errors."""

    pass


class ValidationError(FlowFinderError):
    """Exception raised for input validation errors."""

    pass


class PerformanceError(FlowFinderError):
    """Exception raised when performance targets are not met."""

    pass


class CRSError(FlowFinderError):
    """Exception raised for coordinate reference system errors."""

    pass
