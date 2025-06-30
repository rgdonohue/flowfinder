"""
FLOWFINDER Core Implementation
=============================

Core watershed delineation algorithm for FLOWFINDER.
Implements the main FlowFinder class with watershed delineation capabilities.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .exceptions import (
    FlowFinderError, DEMError, WatershedError, 
    ValidationError, PerformanceError, CRSError
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
        self.flow_direction: Optional[FlowDirectionCalculator] = None
        self.flow_accumulation: Optional[FlowAccumulationCalculator] = None
        self.watershed_extractor: Optional[WatershedExtractor] = None
        
        # Load and validate DEM
        self._load_dem()
        self._validate_dem()
        
        # Initialize processing components
        self._initialize_components()
        
        self.logger.info(f"FLOWFINDER initialized with DEM: {self.dem_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'target_resolution': 10,  # meters
            'flow_direction_method': 'd8',  # d8, dinf, or mfd
            'depression_filling': True,
            'stream_threshold': 1000,  # cells
            'timeout_seconds': 30,
            'memory_limit_mb': 2048,
            'output_crs': 'EPSG:4326',  # WGS84
            'quality_checks': True
        }
    
    def _load_dem(self) -> None:
        """Load DEM data from file."""
        try:
            self.dem_data = rasterio.open(self.dem_path)
            self.logger.info(f"Loaded DEM: {self.dem_data.width}x{self.dem_data.height} "
                           f"pixels, {self.dem_data.res[0]}m resolution")
        except Exception as e:
            raise DEMError(f"Failed to load DEM from {self.dem_path}: {e}")
    
    def _validate_dem(self) -> None:
        """Validate DEM data quality and characteristics."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")
        
        # Check resolution
        resolution = self.dem_data.res[0]
        if resolution > self.config['target_resolution'] * 2:
            raise ValidationError(f"DEM resolution ({resolution}m) is too coarse. "
                                f"Expected {self.config['target_resolution']}m or better")
        
        # Check for valid data
        data = self.dem_data.read(1)
        valid_pixels = np.sum(data != self.dem_data.nodata)
        total_pixels = data.size
        
        if valid_pixels / total_pixels < 0.5:
            raise ValidationError(f"DEM has too many no-data pixels: "
                                f"{valid_pixels}/{total_pixels} valid")
        
        # Check value range
        valid_data = data[data != self.dem_data.nodata]
        if len(valid_data) > 0:
            min_elev = np.min(valid_data)
            max_elev = np.max(valid_data)
            if max_elev - min_elev < 10:  # Less than 10m elevation range
                raise ValidationError("DEM has insufficient elevation range")
        
        self.logger.info("DEM validation passed")
    
    def _initialize_components(self) -> None:
        """Initialize processing components."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")
        
        self.flow_direction = FlowDirectionCalculator(
            dem_data=self.dem_data,
            method=self.config['flow_direction_method'],
            fill_depressions=self.config['depression_filling']
        )
        
        self.flow_accumulation = FlowAccumulationCalculator(
            flow_direction_calc=self.flow_direction
        )
        
        self.watershed_extractor = WatershedExtractor(
            flow_accumulation_calc=self.flow_accumulation,
            stream_threshold=self.config['stream_threshold']
        )
        
        self.logger.info("Processing components initialized")
    
    def delineate_watershed(self, lat: float, lon: float, 
                           timeout: Optional[float] = None) -> Polygon:
        """
        Delineate watershed for a given pour point.
        
        Args:
            lat: Latitude of pour point (decimal degrees)
            lon: Longitude of pour point (decimal degrees)
            timeout: Optional timeout in seconds (overrides config)
            
        Returns:
            Watershed boundary as Shapely Polygon
            
        Raises:
            ValidationError: If coordinates are invalid
            WatershedError: If watershed delineation fails
            PerformanceError: If timeout is exceeded
        """
        start_time = time.time()
        timeout = timeout or self.config['timeout_seconds']
        
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
            
            # Check performance
            runtime = time.time() - start_time
            if runtime > timeout:
                raise PerformanceError(f"Watershed delineation exceeded timeout: {runtime:.1f}s")
            
            self.logger.info(f"Watershed delineated in {runtime:.1f}s, "
                           f"area: {watershed_polygon.area:.2f} km²")
            
            return watershed_polygon
            
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
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")
        
        # Use rasterio's transform to convert coordinates
        row, col = self.dem_data.index(lon, lat)
        return int(row), int(col)
    
    def _point_in_bounds(self, row: int, col: int) -> bool:
        """Check if pixel coordinates are within DEM bounds."""
        if self.dem_data is None:
            return False
        
        return (0 <= row < self.dem_data.height and 
                0 <= col < self.dem_data.width)
    
    def _pixels_to_polygon(self, watershed_pixels: np.ndarray) -> Polygon:
        """Convert watershed pixel array to Shapely polygon."""
        if self.dem_data is None:
            raise ValidationError("DEM data not loaded")
        
        # Create a mask from the watershed pixels
        mask = np.zeros((self.dem_data.height, self.dem_data.width), dtype=np.uint8)
        mask[watershed_pixels] = 1
        
        # Convert mask to polygon using rasterio
        try:
            # Get the transform for the mask
            transform = self.dem_data.transform
            
            # Find contours in the mask
            from skimage import measure
            contours = measure.find_contours(mask, 0.5)
            
            if not contours:
                return Polygon()
            
            # Convert contours to polygons
            polygons = []
            for contour in contours:
                # Convert pixel coordinates to geographic coordinates
                coords = []
                for row, col in contour:
                    lon, lat = rasterio.transform.xy(transform, row, col)
                    coords.append((lon, lat))
                
                if len(coords) >= 3:  # Need at least 3 points for a polygon
                    polygons.append(Polygon(coords))
            
            # Union all polygons
            if polygons:
                result = unary_union(polygons)
                return make_valid(result) if hasattr(result, 'is_valid') else result
            else:
                return Polygon()
                
        except Exception as e:
            self.logger.warning(f"Failed to convert pixels to polygon: {e}")
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
            (max_row, min_col)
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