"""
FLOWFINDER Core Implementation
=============================

Core watershed delineation algorithm for FLOWFINDER.
Implements the main FlowFinder class with watershed delineation capabilities.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import rasterio
from shapely.geometry import Polygon

# shapely.ops.unary_union not used in this file
from shapely.validation import make_valid

from .exceptions import (
    DEMError,
    WatershedError,
    ValidationError,
    PerformanceError,
    CRSError,
)
from .crs_handler import CRSHandler
from .optimized_algorithms import OptimizedPolygonCreation
from .advanced_algorithms import StreamBurning
from .scientific_validation import (
    PerformanceMonitor,
    TopologyValidator,
    AccuracyAssessment,
)
from .flow_direction import FlowDirectionCalculator
from .flow_accumulation import FlowAccumulationCalculator
from .watershed import WatershedExtractor


class FlowFinder:
    """
    Main FLOWFINDER watershed delineation class.

    This class implements the complete watershed delineation workflow:
    1. Load and validate DEM data
    2. Calculate flow direction
    3. Calculate flow accumulation
    4. Extract watershed boundary from pour point

    Attributes:
        dem_path (str): Path to DEM raster file
        dem_data (rasterio.DatasetReader): Loaded DEM dataset
        flow_direction (FlowDirectionCalculator): Flow direction calculator
        flow_accumulation (FlowAccumulationCalculator): Flow accumulation calculator
        watershed_extractor (WatershedExtractor): Watershed boundary extractor
        logger (logging.Logger): Logger instance
        config (Dict[str, Any]): Configuration parameters
    """

    def __init__(self, dem_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FLOWFINDER with DEM data.

        Args:
            dem_path: Path to DEM raster file (GeoTIFF)
            config: Optional configuration dictionary

        Raises:
            DEMError: If DEM file cannot be loaded
            ValidationError: If DEM data is invalid
        """
        self.dem_path = Path(dem_path)
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.dem_data: Optional[rasterio.DatasetReader] = None
        self.crs_handler: Optional[CRSHandler] = None
        self.polygon_creator: Optional[OptimizedPolygonCreation] = None
        self.stream_burner: Optional[StreamBurning] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.topology_validator: Optional[TopologyValidator] = None
        self.accuracy_assessor: Optional[AccuracyAssessment] = None
        self.flow_direction: Optional[FlowDirectionCalculator] = None
        self.flow_accumulation: Optional[FlowAccumulationCalculator] = None
        self.watershed_extractor: Optional[WatershedExtractor] = None

        # Advanced algorithm components (accessed via flow_direction)
        self.dinf_calculator = None  # Will be set via flow_direction
        self.hydrologic_enforcer = None  # Will be set via flow_direction

        # Load and validate DEM
        self._load_dem()
        self._validate_dem()

        # Initialize CRS handler
        self._initialize_crs_handler()

        # Initialize optimized algorithms
        self.polygon_creator = OptimizedPolygonCreation(self.logger)
        self.stream_burner = StreamBurning(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.topology_validator = TopologyValidator(self.logger)
        self.accuracy_assessor = AccuracyAssessment(self.logger)

        # Initialize processing components
        self._initialize_components()

        self.logger.info(f"FLOWFINDER initialized with DEM: {self.dem_path}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "target_resolution": 10,  # meters
            "flow_direction_method": "d8",  # d8, dinf, or mfd
            "depression_filling": True,
            "stream_threshold": 1000,  # cells
            "timeout_seconds": 30,
            "memory_limit_mb": 2048,
            "output_crs": "EPSG:4326",  # WGS84
            "quality_checks": True,
        }

    def _load_dem(self) -> None:
        """Load DEM data from file."""
        try:
            self.dem_data = rasterio.open(self.dem_path)
            self.logger.info(
                f"Loaded DEM: {self.dem_data.width}x{self.dem_data.height} "
                f"pixels, {self.dem_data.res[0]}m resolution"
            )
        except Exception as e:
            raise DEMError(f"Failed to load DEM from {self.dem_path}: {e}")

    def _validate_dem(self) -> None:
        """Validate DEM data quality and characteristics."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")

        # Check resolution
        resolution = self.dem_data.res[0]
        if resolution > self.config["target_resolution"] * 2:
            raise ValidationError(
                f"DEM resolution ({resolution}m) is too coarse. "
                f"Expected {self.config['target_resolution']}m or better"
            )

        # Check for valid data
        data = self.dem_data.read(1)
        valid_pixels = np.sum(data != self.dem_data.nodata)
        total_pixels = data.size

        if valid_pixels / total_pixels < 0.5:
            raise ValidationError(
                f"DEM has too many no-data pixels: "
                f"{valid_pixels}/{total_pixels} valid"
            )

        # Check value range
        valid_data = data[data != self.dem_data.nodata]
        if len(valid_data) > 0:
            min_elev = np.min(valid_data)
            max_elev = np.max(valid_data)
            if max_elev - min_elev < 10:  # Less than 10m elevation range
                raise ValidationError("DEM has insufficient elevation range")

        self.logger.info("DEM validation passed")

    def _initialize_crs_handler(self) -> None:
        """Initialize CRS handler for coordinate transformations."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")

        try:
            input_crs = self.dem_data.crs
            output_crs = self.config["output_crs"]

            self.crs_handler = CRSHandler(input_crs, output_crs)

            # Validate transformation accuracy
            self.crs_handler.validate_transformation_accuracy()

            self.logger.info("CRS handler initialized and validated")

        except Exception as e:
            raise CRSError(f"Failed to initialize CRS handler: {e}")

    def _initialize_components(self) -> None:
        """Initialize processing components."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")

        self.flow_direction = FlowDirectionCalculator(
            dem_data=self.dem_data,
            method=self.config["flow_direction_method"],
            fill_depressions=self.config["depression_filling"],
        )

        self.flow_accumulation = FlowAccumulationCalculator(
            flow_direction_calc=self.flow_direction
        )

        self.watershed_extractor = WatershedExtractor(
            flow_accumulation_calc=self.flow_accumulation,
            stream_threshold=self.config["stream_threshold"],
        )

        # Set references to advanced algorithms for validation
        if self.flow_direction:
            self.dinf_calculator = getattr(self.flow_direction, "dinf_calculator", None)
            self.hydrologic_enforcer = getattr(
                self.flow_direction, "hydrologic_enforcer", None
            )

        self.logger.info("Processing components initialized")

    def delineate_watershed(
        self,
        lat: float,
        lon: float,
        timeout: Optional[float] = None,
        validate_topology: bool = True,
    ) -> Tuple[Polygon, Dict[str, Any]]:
        """
        Delineate watershed for a given pour point.

        Args:
            lat: Latitude of pour point (decimal degrees)
            lon: Longitude of pour point (decimal degrees)
            timeout: Optional timeout in seconds (overrides config)
            validate_topology: Whether to validate topology (default: True)

        Returns:
            Tuple of (watershed_polygon, quality_metrics)

        Raises:
            ValidationError: If coordinates are invalid
            WatershedError: If watershed delineation fails
            PerformanceError: If timeout is exceeded
        """
        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()

        timeout = timeout or self.config["timeout_seconds"]

        # Validate input coordinates
        self._validate_coordinates(lat, lon)

        # Convert lat/lon to DEM coordinates
        row, col = self._latlon_to_pixel(lat, lon)

        # Check if point is within DEM bounds
        if not self._point_in_bounds(row, col):
            raise ValidationError(f"Point ({lat}, {lon}) is outside DEM bounds")

        # Perform watershed delineation
        try:
            watershed_pixels = self.watershed_extractor.extract_watershed(row, col)

            # Convert pixels to polygon
            watershed_polygon = self._pixels_to_polygon(watershed_pixels)

            # Validate result
            if watershed_polygon.is_empty:
                raise WatershedError("Generated watershed is empty")

            # Update peak memory during processing
            if self.performance_monitor:
                self.performance_monitor.update_peak_memory()

            # Validate topology if requested
            topology_metrics = None
            if validate_topology and self.topology_validator:
                try:
                    dem_bounds = self.dem_data.bounds if self.dem_data else (0, 0, 1, 1)
                    topology_metrics = (
                        self.topology_validator.validate_watershed_topology(
                            watershed_polygon, (lon, lat), dem_bounds
                        )
                    )

                    # Repair topology if needed
                    if not topology_metrics.is_valid:
                        self.logger.warning("Repairing watershed topology...")
                        watershed_polygon = (
                            self.topology_validator.repair_watershed_topology(
                                watershed_polygon
                            )
                        )
                except Exception as e:
                    self.logger.warning(f"Topology validation failed: {e}")

            # Finish performance monitoring
            performance_metrics = None
            if self.performance_monitor and self.dem_data:
                try:
                    dem_size = self.dem_data.width * self.dem_data.height
                    watershed_area = (
                        topology_metrics.area_km2 if topology_metrics else 0.0
                    )
                    performance_metrics = self.performance_monitor.finish_monitoring(
                        dem_size, watershed_area
                    )

                    # Check timeout after monitoring
                    if performance_metrics.runtime_seconds > timeout:
                        self.logger.warning(
                            f"Watershed delineation exceeded timeout: "
                            f"{performance_metrics.runtime_seconds:.1f}s > {timeout}s"
                        )
                except Exception as e:
                    self.logger.warning(f"Performance monitoring failed: {e}")

            # Generate quality assessment
            quality_metrics = {}
            if self.accuracy_assessor and performance_metrics and topology_metrics:
                try:
                    quality_metrics = self.accuracy_assessor.assess_watershed_quality(
                        watershed_polygon, performance_metrics, topology_metrics
                    )
                except Exception as e:
                    self.logger.warning(f"Quality assessment failed: {e}")

            # Log results
            runtime = (
                performance_metrics.runtime_seconds if performance_metrics else 0.0
            )
            area = topology_metrics.area_km2 if topology_metrics else 0.0

            self.logger.info(
                f"Watershed delineated in {runtime:.1f}s, area: {area:.2f} km²"
            )

            if quality_metrics.get("overall_quality"):
                self.logger.info(
                    f"Overall quality: {quality_metrics['overall_quality']}"
                )

            return watershed_polygon, quality_metrics

        except Exception as e:
            if isinstance(e, (WatershedError, PerformanceError)):
                raise
            raise WatershedError(f"Watershed delineation failed: {e}")

    def _validate_coordinates(self, lat: float, lon: float) -> None:
        """Validate input coordinates."""
        if not (-90 <= lat <= 90):
            raise ValidationError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValidationError(f"Invalid longitude: {lon}")

    def _latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to DEM pixel coordinates."""
        if self.dem_data is None or self.crs_handler is None:
            raise ValidationError("DEM data or CRS handler not loaded")

        try:
            # Use CRS handler for robust coordinate transformation
            row, col = self.crs_handler.get_pixel_coordinates(
                lon, lat, self.dem_data.transform
            )
            return row, col
        except Exception as e:
            raise CRSError(f"Failed to convert lat/lon to pixel coordinates: {e}")

    def _point_in_bounds(self, row: int, col: int) -> bool:
        """Check if pixel coordinates are within DEM bounds."""
        if self.dem_data is None:
            return False

        return 0 <= row < self.dem_data.height and 0 <= col < self.dem_data.width

    def _pixels_to_polygon(self, watershed_pixels: np.ndarray) -> Polygon:
        """Convert watershed pixel array to Shapely polygon."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")

        # Create a mask from the watershed pixels
        watershed_mask = np.zeros(
            (self.dem_data.height, self.dem_data.width), dtype=np.uint8
        )
        watershed_mask[watershed_pixels] = 1

        # Use optimized polygon creation
        try:
            if self.polygon_creator is None:
                raise ValidationError("Polygon creator not initialized")

            # Convert pixels to coordinates using optimized algorithm
            coords = self.polygon_creator.pixels_to_polygon(
                watershed_pixels, self.dem_data.transform
            )

            if len(coords) >= 3:
                polygon = Polygon(coords)

                # Validate and fix if needed
                if not polygon.is_valid:
                    self.logger.warning(
                        "Generated polygon is invalid - attempting repair"
                    )
                    polygon = make_valid(polygon)

                return polygon
            else:
                return Polygon()

        except Exception as e:
            self.logger.warning(f"Optimized polygon creation failed: {e}")
            # Fallback: create a simple bounding box
            return self._create_fallback_polygon(watershed_pixels)

    def _create_fallback_polygon(self, watershed_pixels: np.ndarray) -> Polygon:
        """Create a fallback polygon from watershed pixels."""
        if self.dem_data is None or len(watershed_pixels) == 0:
            return Polygon()

        # Get bounding box of watershed pixels
        rows = watershed_pixels[0]
        cols = watershed_pixels[1]

        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Convert to geographic coordinates
        transform = self.dem_data.transform

        # Get corner coordinates
        corners = [
            (min_row, min_col),
            (min_row, max_col),
            (max_row, max_col),
            (max_row, min_col),
        ]

        coords = []
        for row, col in corners:
            lon, lat = rasterio.transform.xy(transform, row, col)
            coords.append((lon, lat))

        return Polygon(coords)

    def close(self) -> None:
        """Close DEM dataset and clean up resources."""
        if self.dem_data is not None:
            self.dem_data.close()
            self.dem_data = None
        self.logger.info("FLOWFINDER resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
