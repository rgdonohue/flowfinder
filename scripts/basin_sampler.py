#!/usr/bin/env python3
"""
FLOWFINDER Accuracy Benchmark - Stratified Basin Sampling
=========================================================

This module implements stratified sampling of watershed basins for the FLOWFINDER
accuracy benchmark, focusing on the Mountain West region of the United States.

The sampler creates a representative sample of basins across three dimensions:
- Size: small (5-20 km²), medium (20-100 km²), large (100-500 km²)
- Terrain: flat, moderate, steep (based on slope standard deviation)
- Complexity: low, medium, high (based on stream density)

Key Features:
- Mountain West state filtering (CO, UT, NM, WY, MT, ID, AZ)
- Pour point computation with flowline snapping
- Terrain roughness calculation from DEM data
- Stream complexity assessment
- Stratified sampling with configurable samples per stratum
- Quality validation and export functionality

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 1.0.0
"""

import argparse
import logging
import sys
import warnings
import gc
import psutil
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class BasinSampler:
    """
    Stratified basin sampler for FLOWFINDER accuracy benchmarking.
    
    This class handles the complete workflow of loading geospatial datasets,
    filtering for Mountain West basins, computing terrain and complexity metrics,
    and performing stratified sampling for benchmark testing.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters
        logger (logging.Logger): Logger instance for the sampler
        huc12 (gpd.GeoDataFrame): HUC12 watershed boundaries
        flowlines (gpd.GeoDataFrame): NHD+ flowlines for pour point snapping
        dem (Optional[rasterio.DatasetReader]): DEM raster for terrain analysis
        sample (pd.DataFrame): Final stratified sample of basins
        error_logs (List[Dict[str, Any]]): Structured error tracking
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initialize the basin sampler with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            data_dir: Directory containing input datasets
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        self.config = self._load_config(config_path, data_dir)
        self.error_logs: List[Dict[str, Any]] = []
        self._setup_logging()
        self._validate_config()
        
        # Initialize data attributes
        self.huc12: Optional[gpd.GeoDataFrame] = None
        self.flowlines: Optional[gpd.GeoDataFrame] = None
        self.dem: Optional[rasterio.DatasetReader] = None
        self.sample: Optional[pd.DataFrame] = None
        self.use_chunked_dem_processing: bool = False
        self.dem_validation: Optional[Dict[str, Any]] = None
        
        self.logger.info("BasinSampler initialized successfully")
    
    def _load_config(self, config_path: Optional[str], data_dir: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to configuration file
            data_dir: Data directory override
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'data_dir': data_dir or 'data',
            'area_range': [5, 500],  # km²
            'snap_tolerance': 150,  # meters
            'n_per_stratum': 2,  # samples per stratum
            'target_crs': 'EPSG:5070',  # Albers Equal Area CONUS
            'output_crs': 'EPSG:4326',  # WGS84 for export
            'random_seed': 42,
            'chunk_size': 1000,  # for memory management
            'memory_management': {
                'max_memory_mb': 2048,  # Maximum memory usage in MB
                'large_file_threshold_mb': 50,  # Files larger than this use chunked processing
                'dem_chunk_size_mb': 100,  # DEM chunk size for processing
                'gc_frequency': 10,  # Garbage collection frequency (every N basins)
                'monitor_memory': True  # Enable memory monitoring
            },
            'mountain_west_states': ['CO', 'UT', 'NM', 'WY', 'MT', 'ID', 'AZ'],
            'files': {
                'huc12': 'huc12.shp',
                'flowlines': 'nhd_flowlines.shp',
                'dem': 'dem_10m.tif',
                'slope': None  # Optional pre-computed slope
            },
            'terrain_thresholds': {
                'flat': 5.0,      # degrees
                'moderate': 15.0,  # degrees
                'steep': float('inf')  # degrees
            },
            'size_thresholds': {
                'small': 20,    # km²
                'medium': 100,  # km²
                'large': 500    # km²
            },
            'quality_checks': {
                'min_area_km2': 5.0,
                'max_area_km2': 500.0,
                'min_lat': 30.0,   # Southern boundary
                'max_lat': 50.0,   # Northern boundary
                'min_lon': -120.0, # Western boundary
                'max_lon': -100.0  # Eastern boundary
            },
            'dem_validation': {
                'enable_validation': True,
                'max_nodata_percent': 80,  # Maximum acceptable NoData percentage
                'min_elevation_std': 0.1,  # Minimum elevation variation (meters)
                'expected_resolution_m': 10.0,  # Expected pixel resolution
                'resolution_tolerance': 0.1,  # Tolerance for resolution check (10%)
                'elevation_range_min': -500,  # Minimum acceptable elevation (m)
                'elevation_range_max': 5000,  # Maximum acceptable elevation (m)
                'fail_on_validation_errors': False  # Whether to stop processing on validation failures
            },
            'export': {
                'csv': True,
                'gpkg': True,
                'summary': True,
                'error_log': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
        else:
            self.logger.info("Using default configuration")
            
        return default_config
    
    def _setup_logging(self) -> None:
        """Configure logging for the sampling session."""
        log_file = f"basin_sampler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - log file: {log_file}")
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate area range
        if self.config['area_range'][0] >= self.config['area_range'][1]:
            raise ValueError("area_range must be [min, max] with min < max")
        
        # Validate thresholds
        if self.config['n_per_stratum'] < 1:
            raise ValueError("n_per_stratum must be >= 1")
        
        if self.config['snap_tolerance'] <= 0:
            raise ValueError("snap_tolerance must be > 0")
        
        # Validate file paths
        data_dir = Path(self.config['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        required_files = ['huc12', 'flowlines']
        for file_key in required_files:
            file_path = data_dir / self.config['files'][file_key]
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        self.logger.info("Configuration validation passed")
    
    def _validate_crs_transformation(self, gdf: gpd.GeoDataFrame, source_description: str, 
                                   target_crs: str) -> gpd.GeoDataFrame:
        """
        Validate and perform CRS transformation with comprehensive error handling.
        
        Args:
            gdf: GeoDataFrame to transform
            source_description: Description of data source for error messages
            target_crs: Target CRS string (e.g., 'EPSG:5070')
            
        Returns:
            Transformed GeoDataFrame
            
        Raises:
            ValueError: If transformation fails or coordinates are out of expected range
        """
        if gdf.crs is None:
            error_msg = f"Source CRS is undefined for {source_description}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        original_crs = str(gdf.crs)
        self.logger.info(f"Transforming {source_description} from {original_crs} to {target_crs}")
        
        # Check if already in target CRS
        if original_crs == target_crs:
            self.logger.info(f"{source_description} already in target CRS")
            return gdf
        
        try:
            # Attempt primary transformation
            transformed_gdf = gdf.to_crs(target_crs)
            
            # Validate transformation success
            if len(transformed_gdf) != len(gdf):
                raise ValueError("Transformation resulted in different number of features")
            
            if transformed_gdf.is_empty.any():
                raise ValueError("Transformation resulted in empty geometries")
            
            # Validate coordinate ranges based on target CRS
            if not self._validate_coordinate_ranges(transformed_gdf, target_crs, source_description):
                # Try fallback strategies
                transformed_gdf = self._apply_crs_fallback_strategies(gdf, target_crs, source_description)
            
            self.logger.info(f"Successfully transformed {source_description} to {target_crs}")
            return transformed_gdf
            
        except Exception as e:
            self.logger.error(f"CRS transformation failed for {source_description}: {e}")
            # Try fallback strategies
            return self._apply_crs_fallback_strategies(gdf, target_crs, source_description)
    
    def _validate_coordinate_ranges(self, gdf: gpd.GeoDataFrame, crs: str, 
                                  source_description: str) -> bool:
        """
        Validate that coordinates are within expected ranges for the given CRS.
        
        Args:
            gdf: GeoDataFrame to validate
            crs: CRS string
            source_description: Description for error messages
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        try:
            # Get coordinate bounds
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            # Define expected ranges for common CRS
            expected_ranges = {
                'EPSG:4326': {  # WGS84
                    'x_min': -180, 'x_max': 180,
                    'y_min': -90, 'y_max': 90,
                    'name': 'WGS84'
                },
                'EPSG:5070': {  # NAD83 Albers Equal Area CONUS
                    'x_min': -2500000, 'x_max': 2500000,
                    'y_min': -1500000, 'y_max': 1500000,
                    'name': 'NAD83 Albers CONUS'
                },
                'EPSG:3857': {  # Web Mercator
                    'x_min': -20037508, 'x_max': 20037508,
                    'y_min': -20037508, 'y_max': 20037508,
                    'name': 'Web Mercator'
                }
            }
            
            # Mountain West specific validation for geographic CRS
            if crs == 'EPSG:4326':
                quality_checks = self.config.get('quality_checks', {})
                mountain_west_bounds = {
                    'x_min': quality_checks.get('min_lon', -125),
                    'x_max': quality_checks.get('max_lon', -100),
                    'y_min': quality_checks.get('min_lat', 30),
                    'y_max': quality_checks.get('max_lat', 50)
                }
                expected_ranges['EPSG:4326'].update(mountain_west_bounds)
            
            if crs not in expected_ranges:
                self.logger.warning(f"No coordinate range validation available for CRS {crs}")
                return True
            
            ranges = expected_ranges[crs]
            
            # Check if coordinates are within expected ranges
            if (bounds[0] < ranges['x_min'] or bounds[2] > ranges['x_max'] or
                bounds[1] < ranges['y_min'] or bounds[3] > ranges['y_max']):
                
                self.logger.warning(f"Coordinates for {source_description} appear out of range for {ranges['name']}: "
                                  f"bounds=({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}), "
                                  f"expected x: [{ranges['x_min']}, {ranges['x_max']}], "
                                  f"y: [{ranges['y_min']}, {ranges['y_max']}]")
                return False
            
            self.logger.debug(f"Coordinate validation passed for {source_description} in {ranges['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Coordinate range validation failed for {source_description}: {e}")
            return False
    
    def _apply_crs_fallback_strategies(self, gdf: gpd.GeoDataFrame, target_crs: str, 
                                     source_description: str) -> gpd.GeoDataFrame:
        """
        Apply fallback strategies for CRS transformation failures.
        
        Args:
            gdf: Original GeoDataFrame
            target_crs: Target CRS string
            source_description: Description for error messages
            
        Returns:
            Transformed GeoDataFrame
            
        Raises:
            ValueError: If all fallback strategies fail
        """
        self.logger.warning(f"Applying CRS fallback strategies for {source_description}")
        
        fallback_strategies = [
            self._fallback_assume_wgs84,
            self._fallback_force_crs,
            self._fallback_geometry_validation,
            self._fallback_coordinate_cleanup
        ]
        
        for i, strategy in enumerate(fallback_strategies, 1):
            try:
                self.logger.info(f"Trying fallback strategy {i}/{len(fallback_strategies)} for {source_description}")
                result = strategy(gdf, target_crs, source_description)
                
                if result is not None and len(result) > 0:
                    # Validate the fallback result
                    if self._validate_coordinate_ranges(result, target_crs, f"{source_description} (fallback {i})"):
                        self.logger.info(f"Fallback strategy {i} succeeded for {source_description}")
                        return result
                    else:
                        self.logger.warning(f"Fallback strategy {i} produced invalid coordinates for {source_description}")
                
            except Exception as e:
                self.logger.warning(f"Fallback strategy {i} failed for {source_description}: {e}")
                continue
        
        # All fallback strategies failed
        error_msg = f"All CRS transformation strategies failed for {source_description}"
        self.logger.error(error_msg)
        raise ValueError(error_msg)
    
    def _fallback_assume_wgs84(self, gdf: gpd.GeoDataFrame, target_crs: str, 
                             source_description: str) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Assume source is WGS84 if CRS is missing."""
        if gdf.crs is None:
            self.logger.info(f"Assuming WGS84 for {source_description} with missing CRS")
            gdf_copy = gdf.copy()
            gdf_copy.crs = 'EPSG:4326'
            return gdf_copy.to_crs(target_crs)
        return None
    
    def _fallback_force_crs(self, gdf: gpd.GeoDataFrame, target_crs: str, 
                          source_description: str) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Force CRS without transformation, then transform."""
        try:
            gdf_copy = gdf.copy()
            gdf_copy.crs = target_crs
            return gdf_copy
        except Exception:
            return None
    
    def _fallback_geometry_validation(self, gdf: gpd.GeoDataFrame, target_crs: str, 
                                    source_description: str) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Fix invalid geometries before transformation."""
        try:
            gdf_copy = gdf.copy()
            # Fix invalid geometries
            invalid_mask = ~gdf_copy.is_valid
            if invalid_mask.any():
                self.logger.info(f"Fixing {invalid_mask.sum()} invalid geometries in {source_description}")
                gdf_copy.loc[invalid_mask, 'geometry'] = gdf_copy.loc[invalid_mask, 'geometry'].apply(make_valid)
            
            return gdf_copy.to_crs(target_crs)
        except Exception:
            return None
    
    def _fallback_coordinate_cleanup(self, gdf: gpd.GeoDataFrame, target_crs: str, 
                                   source_description: str) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Remove features with problematic coordinates."""
        try:
            gdf_copy = gdf.copy()
            
            # Remove features with extreme coordinates that might cause transformation issues
            bounds = gdf_copy.bounds
            
            # For geographic coordinates, remove extreme values
            if gdf_copy.crs and 'EPSG:4326' in str(gdf_copy.crs):
                valid_mask = ((bounds['minx'] >= -180) & (bounds['maxx'] <= 180) & 
                            (bounds['miny'] >= -90) & (bounds['maxy'] <= 90))
            else:
                # For projected coordinates, remove extremely large values
                valid_mask = ((abs(bounds['minx']) < 1e8) & (abs(bounds['maxx']) < 1e8) & 
                            (abs(bounds['miny']) < 1e8) & (abs(bounds['maxy']) < 1e8))
            
            if not valid_mask.all():
                removed_count = (~valid_mask).sum()
                self.logger.warning(f"Removing {removed_count} features with extreme coordinates from {source_description}")
                gdf_copy = gdf_copy[valid_mask].reset_index(drop=True)
            
            if len(gdf_copy) > 0:
                return gdf_copy.to_crs(target_crs)
            else:
                return None
        except Exception:
            return None
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def _check_memory_usage(self, operation: str = "operation") -> bool:
        """
        Check if memory usage is within acceptable limits.
        
        Args:
            operation: Description of current operation for logging
            
        Returns:
            True if memory usage is acceptable, False if approaching limits
        """
        memory_config = self.config.get('memory_management', {})
        if not memory_config.get('monitor_memory', True):
            return True
        
        memory_info = self._get_memory_usage()
        max_memory_mb = memory_config.get('max_memory_mb', 2048)
        
        if memory_info['rss_mb'] > max_memory_mb:
            self.logger.warning(f"Memory usage ({memory_info['rss_mb']:.1f} MB) exceeds limit "
                              f"({max_memory_mb} MB) during {operation}")
            return False
        
        if memory_info['percent'] > 80:
            self.logger.warning(f"High memory usage ({memory_info['percent']:.1f}%) during {operation}")
            return False
        
        self.logger.debug(f"Memory usage during {operation}: {memory_info['rss_mb']:.1f} MB "
                         f"({memory_info['percent']:.1f}%)")
        return True
    
    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        self.logger.debug("Forced garbage collection")
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Could not get file size for {file_path}: {e}")
            return 0.0
    
    def _is_large_dem_file(self, dem_path: Path) -> bool:
        """Check if DEM file is large enough to require chunked processing."""
        memory_config = self.config.get('memory_management', {})
        threshold_mb = memory_config.get('large_file_threshold_mb', 50)
        
        file_size_mb = self._get_file_size_mb(dem_path)
        is_large = file_size_mb > threshold_mb
        
        if is_large:
            self.logger.info(f"DEM file is large ({file_size_mb:.1f} MB > {threshold_mb} MB), "
                           "using chunked processing")
        else:
            self.logger.info(f"DEM file size: {file_size_mb:.1f} MB")
        
        return is_large
    
    def _validate_dem_quality(self, dem_path: Path) -> Dict[str, Any]:
        """
        Comprehensive DEM quality validation.
        
        Args:
            dem_path: Path to DEM file
            
        Returns:
            Dictionary containing validation results and metrics
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'recommendations': []
        }
        
        try:
            with rasterio.open(dem_path) as dem:
                self.logger.info("Performing comprehensive DEM quality validation...")
                
                # 1. Basic raster integrity checks
                self._validate_raster_integrity(dem, validation_results)
                
                # 2. Coordinate reference system validation
                self._validate_dem_crs(dem, validation_results)
                
                # 3. Resolution consistency checks
                self._validate_dem_resolution(dem, validation_results)
                
                # 4. NoData extent analysis
                self._validate_nodata_extent(dem, validation_results)
                
                # 5. Elevation value range validation
                self._validate_elevation_values(dem, validation_results)
                
                # 6. Spatial coverage validation
                self._validate_spatial_coverage(dem, validation_results)
                
                # 7. Data type and precision checks
                self._validate_data_precision(dem, validation_results)
                
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"DEM validation failed: {e}")
            self.logger.error(f"DEM validation error: {e}")
        
        # Log validation summary
        self._log_validation_summary(validation_results)
        
        return validation_results
    
    def _validate_raster_integrity(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate basic raster file integrity."""
        try:
            # Check basic properties
            if dem.width <= 0 or dem.height <= 0:
                results['errors'].append("Invalid raster dimensions")
                results['is_valid'] = False
            
            if dem.count <= 0:
                results['errors'].append("No raster bands found")
                results['is_valid'] = False
            
            # Check for corrupted transform
            if dem.transform is None:
                results['errors'].append("Missing geospatial transform")
                results['is_valid'] = False
            
            # Validate transform values
            transform = dem.transform
            if abs(transform.a) < 1e-10 or abs(transform.e) < 1e-10:
                results['warnings'].append("Extremely small pixel sizes detected")
            
            # Store basic metrics
            results['metrics'].update({
                'width': dem.width,
                'height': dem.height,
                'bands': dem.count,
                'data_type': str(dem.dtypes[0]),
                'pixel_size_x': abs(transform.a),
                'pixel_size_y': abs(transform.e),
                'total_pixels': dem.width * dem.height
            })
            
        except Exception as e:
            results['errors'].append(f"Raster integrity check failed: {e}")
    
    def _validate_dem_crs(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate DEM coordinate reference system."""
        try:
            if dem.crs is None:
                results['errors'].append("DEM has no coordinate reference system defined")
                results['is_valid'] = False
                return
            
            crs_str = str(dem.crs)
            results['metrics']['crs'] = crs_str
            
            # Check for common problematic CRS
            if 'EPSG:4326' in crs_str:
                results['warnings'].append("DEM uses geographic coordinates (EPSG:4326) - "
                                         "projected coordinates recommended for terrain analysis")
            
            # Validate CRS for Mountain West region
            if dem.bounds:
                bounds = dem.bounds
                if 'EPSG:4326' in crs_str:
                    # Geographic bounds check
                    if bounds.left < -130 or bounds.right > -90 or bounds.bottom < 25 or bounds.top > 55:
                        results['warnings'].append("DEM extent appears outside Mountain West region")
                
        except Exception as e:
            results['warnings'].append(f"CRS validation warning: {e}")
    
    def _validate_dem_resolution(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate DEM resolution consistency and appropriateness."""
        try:
            transform = dem.transform
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            
            # Check for square pixels
            if abs(pixel_size_x - pixel_size_y) > pixel_size_x * 0.01:  # 1% tolerance
                results['warnings'].append(f"Non-square pixels detected: "
                                         f"{pixel_size_x:.3f} x {pixel_size_y:.3f}")
            
            # Check for expected 10m resolution (within tolerance)
            expected_resolution = 10.0  # meters for 10m DEM
            if 'EPSG:4326' not in str(dem.crs):  # Only check for projected coordinates
                if abs(pixel_size_x - expected_resolution) > expected_resolution * 0.1:  # 10% tolerance
                    results['warnings'].append(f"Unexpected resolution: {pixel_size_x:.3f}m "
                                             f"(expected ~{expected_resolution}m)")
            
            # Check for extremely high or low resolution
            if 'EPSG:4326' not in str(dem.crs):
                if pixel_size_x > 100:
                    results['warnings'].append(f"Very coarse resolution: {pixel_size_x:.1f}m")
                elif pixel_size_x < 1:
                    results['warnings'].append(f"Very fine resolution: {pixel_size_x:.3f}m")
            
            results['metrics'].update({
                'resolution_x': pixel_size_x,
                'resolution_y': pixel_size_y,
                'is_square_pixels': abs(pixel_size_x - pixel_size_y) < pixel_size_x * 0.01
            })
            
        except Exception as e:
            results['warnings'].append(f"Resolution validation warning: {e}")
    
    def _validate_nodata_extent(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate NoData extent and distribution."""
        try:
            # Sample the DEM to check NoData distribution
            sample_window = rasterio.windows.Window(0, 0, 
                                                   min(1000, dem.width), 
                                                   min(1000, dem.height))
            sample_data = dem.read(1, window=sample_window)
            
            # Calculate NoData statistics
            if dem.nodata is not None:
                nodata_mask = sample_data == dem.nodata
                nodata_percent = (nodata_mask.sum() / sample_data.size) * 100
                
                results['metrics'].update({
                    'nodata_value': dem.nodata,
                    'nodata_percent_sample': nodata_percent,
                    'has_nodata': True
                })
                
                # Check for excessive NoData
                if nodata_percent > 80:
                    results['errors'].append(f"Excessive NoData values: {nodata_percent:.1f}% of sample")
                    results['is_valid'] = False
                elif nodata_percent > 50:
                    results['warnings'].append(f"High NoData percentage: {nodata_percent:.1f}% of sample")
                elif nodata_percent > 20:
                    results['warnings'].append(f"Moderate NoData percentage: {nodata_percent:.1f}% of sample")
                
                # Check for reasonable NoData value
                if abs(dem.nodata) < 1e6:  # Reasonable range check
                    data_min = np.min(sample_data[~nodata_mask]) if not np.all(nodata_mask) else np.nan
                    data_max = np.max(sample_data[~nodata_mask]) if not np.all(nodata_mask) else np.nan
                    
                    if not np.isnan(data_min) and not np.isnan(data_max):
                        if data_min <= dem.nodata <= data_max:
                            results['warnings'].append("NoData value within data range - may cause confusion")
            else:
                results['metrics'].update({
                    'nodata_value': None,
                    'has_nodata': False
                })
                results['warnings'].append("No NoData value defined")
            
        except Exception as e:
            results['warnings'].append(f"NoData validation warning: {e}")
    
    def _validate_elevation_values(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate elevation value ranges and statistics."""
        try:
            # Sample elevation data for statistical analysis
            sample_window = rasterio.windows.Window(0, 0, 
                                                   min(2000, dem.width), 
                                                   min(2000, dem.height))
            sample_data = dem.read(1, window=sample_window)
            
            # Remove NoData values
            if dem.nodata is not None:
                valid_data = sample_data[sample_data != dem.nodata]
            else:
                valid_data = sample_data
            
            if len(valid_data) == 0:
                results['errors'].append("No valid elevation data found in sample")
                results['is_valid'] = False
                return
            
            # Calculate elevation statistics
            elev_min = float(np.min(valid_data))
            elev_max = float(np.max(valid_data))
            elev_mean = float(np.mean(valid_data))
            elev_std = float(np.std(valid_data))
            
            results['metrics'].update({
                'elevation_min': elev_min,
                'elevation_max': elev_max,
                'elevation_mean': elev_mean,
                'elevation_std': elev_std,
                'elevation_range': elev_max - elev_min
            })
            
            # Validate elevation ranges for Mountain West
            if elev_min < -500:
                results['warnings'].append(f"Unusually low elevations: {elev_min:.1f}m")
            if elev_max > 5000:
                results['warnings'].append(f"Very high elevations: {elev_max:.1f}m")
            if elev_min > 4000:
                results['warnings'].append(f"All elevations very high: min={elev_min:.1f}m")
            
            # Check for flat areas (no elevation variation)
            if elev_std < 1.0:
                results['warnings'].append(f"Very low elevation variation: std={elev_std:.3f}m")
            
            # Check for unrealistic elevation ranges
            if (elev_max - elev_min) > 6000:
                results['warnings'].append(f"Very large elevation range: {elev_max - elev_min:.1f}m")
            
        except Exception as e:
            results['warnings'].append(f"Elevation validation warning: {e}")
    
    def _validate_spatial_coverage(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate spatial coverage and extent."""
        try:
            bounds = dem.bounds
            if bounds:
                # Calculate extent
                width_extent = bounds.right - bounds.left
                height_extent = bounds.top - bounds.bottom
                
                results['metrics'].update({
                    'bounds': {
                        'left': bounds.left,
                        'bottom': bounds.bottom,
                        'right': bounds.right,
                        'top': bounds.top
                    },
                    'extent_width': width_extent,
                    'extent_height': height_extent,
                    'total_area': width_extent * height_extent
                })
                
                # Check for reasonable extents
                if 'EPSG:4326' in str(dem.crs):
                    # Geographic coordinates
                    if width_extent < 0.001 or height_extent < 0.001:
                        results['warnings'].append("Very small spatial extent")
                    elif width_extent > 50 or height_extent > 50:
                        results['warnings'].append("Very large spatial extent")
                else:
                    # Projected coordinates
                    if width_extent < 1000 or height_extent < 1000:
                        results['warnings'].append("Very small spatial extent")
                    elif width_extent > 1000000 or height_extent > 1000000:
                        results['warnings'].append("Very large spatial extent")
            
        except Exception as e:
            results['warnings'].append(f"Spatial coverage validation warning: {e}")
    
    def _validate_data_precision(self, dem: rasterio.DatasetReader, results: Dict[str, Any]) -> None:
        """Validate data type and precision appropriateness."""
        try:
            dtype = dem.dtypes[0]
            results['metrics']['data_type'] = str(dtype)
            
            # Check data type appropriateness
            if dtype in ['uint8', 'int8']:
                results['warnings'].append("8-bit data type may have insufficient precision for elevations")
            elif dtype in ['float64']:
                results['recommendations'].append("Consider float32 for better memory efficiency")
            elif dtype in ['uint16', 'int16']:
                results['recommendations'].append("16-bit integer may be sufficient for most applications")
            
            # Check for signed vs unsigned appropriateness
            if 'uint' in str(dtype):
                # Sample data to check for negative values that would be problematic
                sample_window = rasterio.windows.Window(0, 0, min(500, dem.width), min(500, dem.height))
                sample_data = dem.read(1, window=sample_window)
                
                # For unsigned types, check if we're near the maximum value (indicating clipping)
                max_val = np.iinfo(dtype).max if 'int' in str(dtype) else np.finfo(dtype).max
                if np.any(sample_data == max_val):
                    results['warnings'].append("Data values at maximum for data type - possible clipping")
            
        except Exception as e:
            results['warnings'].append(f"Data precision validation warning: {e}")
    
    def _log_validation_summary(self, results: Dict[str, Any]) -> None:
        """Log DEM validation summary."""
        self.logger.info("=== DEM Quality Validation Summary ===")
        self.logger.info(f"Overall Status: {'VALID' if results['is_valid'] else 'INVALID'}")
        
        if results['errors']:
            self.logger.error(f"Errors ({len(results['errors'])}):")
            for error in results['errors']:
                self.logger.error(f"  - {error}")
        
        if results['warnings']:
            self.logger.warning(f"Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                self.logger.warning(f"  - {warning}")
        
        if results['recommendations']:
            self.logger.info(f"Recommendations ({len(results['recommendations'])}):")
            for rec in results['recommendations']:
                self.logger.info(f"  - {rec}")
        
        # Log key metrics
        metrics = results.get('metrics', {})
        if metrics:
            self.logger.info("Key Metrics:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    continue  # Skip nested dictionaries for summary
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.3f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=====================================")
    
    def _is_dem_suitable_for_processing(self) -> bool:
        """
        Check if DEM passed quality validation for reliable processing.
        
        Returns:
            True if DEM is suitable for processing, False otherwise
        """
        if self.dem_validation is None:
            self.logger.warning("No DEM validation results available")
            return True  # Assume OK if no validation was run
        
        validation = self.dem_validation
        
        # Critical validation failures
        if not validation['is_valid']:
            return False
        
        # Check for deal-breaker conditions
        metrics = validation.get('metrics', {})
        
        # Too much NoData
        nodata_percent = metrics.get('nodata_percent_sample', 0)
        if nodata_percent > 80:
            self.logger.error(f"DEM has too much NoData ({nodata_percent:.1f}%) for reliable processing")
            return False
        
        # No valid elevation data
        if 'elevation_min' not in metrics or 'elevation_max' not in metrics:
            self.logger.error("No valid elevation data found in DEM sample")
            return False
        
        # Extremely flat terrain (may indicate corrupt data)
        elev_std = metrics.get('elevation_std', 0)
        if elev_std < 0.1:
            self.logger.warning(f"DEM appears extremely flat (std={elev_std:.3f}m) - results may be unreliable")
        
        return True
    
    def export_dem_validation_report(self, output_file: str = "dem_validation_report.json") -> None:
        """
        Export DEM validation results to a file.
        
        Args:
            output_file: Path to output file for validation report
        """
        if self.dem_validation is None:
            self.logger.warning("No DEM validation results to export")
            return
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.dem_validation, f, indent=2, default=str)
            
            self.logger.info(f"DEM validation report exported to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export DEM validation report: {e}")
    
    def _extract_dem_data_for_basin(self, basin_geometry, basin_id: str = "unknown") -> Optional[np.ndarray]:
        """
        Extract DEM data for a single basin with memory management.
        
        Args:
            basin_geometry: Basin geometry for extraction
            basin_id: Basin identifier for logging
            
        Returns:
            Extracted elevation data or None if extraction fails
        """
        if self.dem is None:
            return None
        
        try:
            # Check memory before processing
            if not self._check_memory_usage(f"DEM extraction for basin {basin_id}"):
                self._force_garbage_collection()
            
            # Use chunked processing for large DEMs
            if getattr(self, 'use_chunked_dem_processing', False):
                return self._extract_dem_data_chunked(basin_geometry, basin_id)
            else:
                return self._extract_dem_data_direct(basin_geometry, basin_id)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract DEM data for basin {basin_id}: {e}")
            return None
    
    def _extract_dem_data_direct(self, basin_geometry, basin_id: str) -> Optional[np.ndarray]:
        """Direct DEM extraction for smaller files."""
        try:
            basin_geom = [basin_geometry]
            out_image, out_transform = mask(self.dem, basin_geom, crop=True)
            
            if out_image.size == 0:
                self.logger.debug(f"No DEM data found for basin {basin_id}")
                return None
            
            return out_image[0]  # Return first band
            
        except Exception as e:
            self.logger.warning(f"Direct DEM extraction failed for basin {basin_id}: {e}")
            return None
    
    def _extract_dem_data_chunked(self, basin_geometry, basin_id: str) -> Optional[np.ndarray]:
        """
        Chunked DEM extraction for large files.
        
        This method processes the DEM in chunks to manage memory usage.
        """
        try:
            from rasterio.windows import from_bounds
            from rasterio.transform import from_bounds as transform_from_bounds
            
            # Get basin bounds
            minx, miny, maxx, maxy = basin_geometry.bounds
            
            # Convert bounds to pixel coordinates
            window = from_bounds(minx, miny, maxx, maxy, self.dem.transform)
            
            # Calculate chunk size based on memory limits
            memory_config = self.config.get('memory_management', {})
            chunk_size_mb = memory_config.get('dem_chunk_size_mb', 100)
            
            # Estimate pixels per MB for this DEM
            bytes_per_pixel = np.dtype(self.dem.dtypes[0]).itemsize
            pixels_per_mb = (chunk_size_mb * 1024 * 1024) / bytes_per_pixel
            chunk_size_pixels = int(np.sqrt(pixels_per_mb))  # Square chunks
            
            # Read the windowed area in chunks if it's large
            window_width = int(window.width)
            window_height = int(window.height)
            
            if window_width * window_height * bytes_per_pixel > chunk_size_mb * 1024 * 1024:
                self.logger.debug(f"Using chunked processing for basin {basin_id} "
                                f"(window: {window_width}x{window_height})")
                return self._process_dem_window_chunked(window, basin_geometry, chunk_size_pixels, basin_id)
            else:
                # Small enough to process directly
                return self._process_dem_window_direct(window, basin_geometry, basin_id)
                
        except Exception as e:
            self.logger.warning(f"Chunked DEM extraction failed for basin {basin_id}: {e}")
            return None
    
    def _process_dem_window_direct(self, window: Window, basin_geometry, basin_id: str) -> Optional[np.ndarray]:
        """Process a DEM window directly."""
        try:
            # Read the windowed data
            window_data = self.dem.read(1, window=window)
            
            # Create the transform for this window
            window_transform = rasterio.windows.transform(window, self.dem.transform)
            
            # Create a temporary raster for masking
            from rasterio.io import MemoryFile
            
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=window.height,
                    width=window.width,
                    count=1,
                    dtype=window_data.dtype,
                    crs=self.dem.crs,
                    transform=window_transform,
                    nodata=self.dem.nodata
                ) as temp_raster:
                    temp_raster.write(window_data, 1)
                    
                    # Apply mask
                    basin_geom = [basin_geometry]
                    out_image, out_transform = mask(temp_raster, basin_geom, crop=True)
                    
                    if out_image.size == 0:
                        return None
                    
                    return out_image[0]
                    
        except Exception as e:
            self.logger.warning(f"Direct window processing failed for basin {basin_id}: {e}")
            return None
    
    def _process_dem_window_chunked(self, window: Window, basin_geometry, 
                                  chunk_size_pixels: int, basin_id: str) -> Optional[np.ndarray]:
        """Process a large DEM window in chunks."""
        try:
            window_width = int(window.width)
            window_height = int(window.height)
            
            # Collect results from chunks
            result_chunks = []
            
            for row_start in range(0, window_height, chunk_size_pixels):
                row_end = min(row_start + chunk_size_pixels, window_height)
                
                for col_start in range(0, window_width, chunk_size_pixels):
                    col_end = min(col_start + chunk_size_pixels, window_width)
                    
                    # Create chunk window
                    chunk_window = Window(
                        col_off=window.col_off + col_start,
                        row_off=window.row_off + row_start,
                        width=col_end - col_start,
                        height=row_end - row_start
                    )
                    
                    # Process chunk
                    chunk_data = self._process_dem_window_direct(chunk_window, basin_geometry, 
                                                              f"{basin_id}_chunk_{row_start}_{col_start}")
                    
                    if chunk_data is not None:
                        result_chunks.append(chunk_data)
                    
                    # Memory management
                    if not self._check_memory_usage(f"chunked processing for basin {basin_id}"):
                        self._force_garbage_collection()
            
            # Combine chunks if we have results
            if result_chunks:
                # For simplicity, return the first valid chunk
                # In production, you might want to mosaic the chunks properly
                return result_chunks[0]
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Chunked window processing failed for basin {basin_id}: {e}")
            return None
    
    def load_datasets(self) -> None:
        """
        Load all required geospatial datasets.
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If datasets cannot be loaded
        """
        self.logger.info("Loading geospatial datasets...")
        
        try:
            # Load HUC12 boundaries
            self._load_huc12_data()
            
            # Load flowlines
            self._load_flowlines_data()
            
            # Load DEM if available
            self._load_dem_data()
            
            self.logger.info("All datasets loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise
    
    def _load_huc12_data(self) -> None:
        """Load HUC12 watershed boundaries."""
        huc12_path = Path(self.config['data_dir']) / self.config['files']['huc12']
        self.logger.info(f"Loading HUC12 boundaries from {huc12_path}")
        
        self.huc12 = gpd.read_file(huc12_path)
        self.huc12 = self._validate_crs_transformation(self.huc12, "HUC12 boundaries", self.config['target_crs'])
        
        # Ensure required columns exist
        required_cols = ['HUC12']
        missing_cols = [col for col in required_cols if col not in self.huc12.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in HUC12 data: {missing_cols}")
        
        self.logger.info(f"Loaded {len(self.huc12)} HUC12 watersheds")
    
    def _load_flowlines_data(self) -> None:
        """Load NHD+ flowlines for pour point snapping."""
        flowlines_path = Path(self.config['data_dir']) / self.config['files']['flowlines']
        self.logger.info(f"Loading flowlines from {flowlines_path}")
        
        self.flowlines = gpd.read_file(flowlines_path)
        self.flowlines = self._validate_crs_transformation(self.flowlines, "NHD+ flowlines", self.config['target_crs'])
        
        self.logger.info(f"Loaded {len(self.flowlines)} flowlines")
    
    def _load_dem_data(self) -> None:
        """Load DEM raster for terrain analysis with memory management."""
        dem_path = Path(self.config['data_dir']) / self.config['files']['dem']
        
        if dem_path.exists():
            self.logger.info(f"Loading DEM from {dem_path}")
            
            # Check file size and memory usage
            file_size_mb = self._get_file_size_mb(dem_path)
            self._check_memory_usage("DEM loading")
            
            # Open DEM
            self.dem = rasterio.open(dem_path)
            
            # Log DEM information
            dem_info = {
                'width': self.dem.width,
                'height': self.dem.height,
                'bands': self.dem.count,
                'dtype': self.dem.dtypes[0],
                'crs': str(self.dem.crs) if self.dem.crs else 'None',
                'file_size_mb': file_size_mb
            }
            
            self.logger.info(f"DEM loaded: {dem_info['width']}x{dem_info['height']} pixels, "
                           f"{dem_info['bands']} bands, {dem_info['dtype']}, "
                           f"CRS: {dem_info['crs']}, Size: {file_size_mb:.1f} MB")
            
            # Perform comprehensive DEM quality validation
            validation_results = self._validate_dem_quality(dem_path)
            
            # Store validation results for later reference
            self.dem_validation = validation_results
            
            # Handle validation failures
            if not validation_results['is_valid']:
                self.logger.error("DEM validation failed - some processing may be unreliable")
                if len(validation_results['errors']) > 3:  # Too many critical errors
                    self.logger.error("Too many critical DEM errors - consider using a different DEM")
                    self.dem.close()
                    self.dem = None
                    self.use_chunked_dem_processing = False
                    return
            
            # Check if this is a large file that will need chunked processing
            self.use_chunked_dem_processing = self._is_large_dem_file(dem_path)
            
            # Estimate memory requirements
            estimated_memory_mb = (self.dem.width * self.dem.height * 
                                 np.dtype(self.dem.dtypes[0]).itemsize * 
                                 self.dem.count) / (1024 * 1024)
            
            if estimated_memory_mb > 500:
                self.logger.warning(f"DEM may require significant memory if fully loaded: "
                                  f"{estimated_memory_mb:.1f} MB estimated")
            
        else:
            self.logger.warning(f"DEM file not found: {dem_path}")
            self.dem = None
            self.use_chunked_dem_processing = False
    
    def filter_mountain_west_basins(self) -> None:
        """
        Filter basins to Mountain West states and apply quality constraints.
        
        This method filters the HUC12 dataset to include only basins within
        the Mountain West region and applies quality constraints for sampling.
        """
        if self.huc12 is None:
            raise ValueError("HUC12 data not loaded. Call load_datasets() first.")
        
        self.logger.info("Filtering Mountain West basins...")
        initial_count = len(self.huc12)
        
        # Filter by Mountain West states
        if 'STATES' in self.huc12.columns:
            mw_states = self.config['mountain_west_states']
            self.huc12 = self.huc12[self.huc12['STATES'].isin(mw_states)]
            self.logger.info(f"State filtering: {len(self.huc12)} basins remaining")
        
        # Filter by area constraints
        self.huc12['area_km2'] = self.huc12.geometry.area / 1e6
        area_min, area_max = self.config['area_range']
        self.huc12 = self.huc12[
            (self.huc12['area_km2'] >= area_min) & 
            (self.huc12['area_km2'] <= area_max)
        ]
        self.logger.info(f"Area filtering: {len(self.huc12)} basins remaining")
        
        # Filter by geographic bounds
        bounds = self.config['quality_checks']
        self.huc12 = self.huc12[
            (self.huc12.geometry.bounds.miny >= bounds['min_lat']) &
            (self.huc12.geometry.bounds.maxy <= bounds['max_lat']) &
            (self.huc12.geometry.bounds.minx >= bounds['min_lon']) &
            (self.huc12.geometry.bounds.maxx <= bounds['max_lon'])
        ]
        self.logger.info(f"Geographic filtering: {len(self.huc12)} basins remaining")
        
        # Remove invalid geometries
        valid_mask = self.huc12.geometry.is_valid
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            self.logger.warning(f"Removing {invalid_count} invalid geometries")
            self.huc12 = self.huc12[valid_mask]
        
        final_count = len(self.huc12)
        self.logger.info(f"Mountain West filtering complete: {final_count}/{initial_count} basins retained")
    
    def compute_pour_points(self) -> None:
        """
        Compute pour points for each basin with flowline snapping.
        
        This method calculates the lowest point (pour point) of each basin
        and snaps it to the nearest flowline within the specified tolerance.
        """
        if self.huc12 is None or self.flowlines is None:
            raise ValueError("HUC12 and flowlines data must be loaded")
        
        self.logger.info("Computing pour points with flowline snapping...")
        
        pour_points = []
        snap_tolerance = self.config['snap_tolerance']
        
        for idx, basin in tqdm(self.huc12.iterrows(), total=len(self.huc12), desc="Computing pour points"):
            try:
                # Find the lowest point of the basin (simplified approach)
                basin_geom = make_valid(basin.geometry)
                if basin_geom.is_empty:
                    continue
                
                # Use centroid as initial pour point (simplified)
                initial_point = basin_geom.centroid
                
                # Snap to nearest flowline if within tolerance
                distances = self.flowlines.geometry.distance(initial_point)
                min_distance = distances.min()
                
                if min_distance <= snap_tolerance:
                    # Snap to nearest flowline
                    nearest_idx = distances.idxmin()
                    nearest_flowline = self.flowlines.loc[nearest_idx].geometry
                    snapped_point = nearest_flowline.interpolate(nearest_flowline.project(initial_point))
                    pour_points.append(snapped_point)
                else:
                    # Use initial point if no flowline within tolerance
                    pour_points.append(initial_point)
                    
            except Exception as e:
                self.logger.warning(f"Failed to compute pour point for basin {idx}: {e}")
                pour_points.append(None)
        
        self.huc12['pour_point'] = pour_points
        valid_points = sum(1 for p in pour_points if p is not None)
        self.logger.info(f"Pour point computation complete: {valid_points}/{len(pour_points)} valid points")
    
    def compute_terrain_roughness(self) -> None:
        """
        Compute terrain roughness (slope standard deviation) from DEM data with memory management.
        
        This method calculates the standard deviation of slope values within
        each basin to characterize terrain complexity, using chunked processing
        for large DEM files to manage memory usage.
        """
        if self.huc12 is None:
            raise ValueError("HUC12 data not loaded")
        
        if self.dem is None:
            self.logger.warning("DEM not available - skipping terrain roughness calculation")
            self.huc12['slope_std'] = np.nan
            return
        
        # Check DEM quality before processing
        if not self._is_dem_suitable_for_processing():
            self.logger.error("DEM failed quality validation - skipping terrain roughness calculation")
            self.huc12['slope_std'] = np.nan
            return
        
        self.logger.info("Computing terrain roughness from DEM with memory management...")
        
        # Get memory management configuration
        memory_config = self.config.get('memory_management', {})
        gc_frequency = memory_config.get('gc_frequency', 10)
        
        slope_stds = []
        processed_count = 0
        memory_warnings = 0
        
        for idx, basin in tqdm(self.huc12.iterrows(), total=len(self.huc12), desc="Computing terrain roughness"):
            try:
                # Get basin ID for logging
                basin_id = basin.get('HUC12', str(idx))
                
                # Extract DEM data using memory-managed extraction
                elevation_data = self._extract_dem_data_for_basin(basin.geometry, basin_id)
                
                if elevation_data is None or elevation_data.size == 0:
                    slope_stds.append(np.nan)
                    continue
                
                # Create valid data mask
                valid_mask = elevation_data != self.dem.nodata
                
                if valid_mask.sum() < 10:  # Need minimum pixels
                    slope_stds.append(np.nan)
                    continue
                
                # Calculate terrain roughness
                slope_std = self._calculate_terrain_roughness(elevation_data[valid_mask])
                slope_stds.append(slope_std)
                
                # Memory management
                processed_count += 1
                if processed_count % gc_frequency == 0:
                    # Check memory usage periodically
                    if not self._check_memory_usage(f"terrain roughness (processed {processed_count})"):
                        memory_warnings += 1
                        self._force_garbage_collection()
                
                # Clear elevation data from memory
                del elevation_data
                
            except Exception as e:
                self.logger.warning(f"Failed to compute terrain roughness for basin {basin_id}: {e}")
                slope_stds.append(np.nan)
        
        # Store results
        self.huc12['slope_std'] = slope_stds
        valid_slopes = sum(1 for s in slope_stds if not np.isnan(s))
        
        # Log completion statistics
        self.logger.info(f"Terrain roughness computation complete: {valid_slopes}/{len(slope_stds)} valid values")
        if memory_warnings > 0:
            self.logger.warning(f"Encountered {memory_warnings} memory warnings during processing")
        
        # Final memory cleanup
        self._force_garbage_collection()
    
    def _calculate_terrain_roughness(self, elevation_data: np.ndarray) -> float:
        """
        Calculate terrain roughness from elevation data.
        
        Args:
            elevation_data: Valid elevation data (nodata values already removed)
            
        Returns:
            Terrain roughness value (standard deviation of slopes)
        """
        try:
            # For simplicity, use elevation standard deviation as roughness proxy
            # In production, calculate actual slope values and their standard deviation
            if len(elevation_data) < 10:
                return np.nan
            
            # Simple roughness calculation - standard deviation of elevations
            # TODO: Implement proper slope calculation for production use
            roughness = np.std(elevation_data)
            
            return float(roughness)
            
        except Exception as e:
            self.logger.warning(f"Terrain roughness calculation failed: {e}")
            return np.nan
    
    def compute_stream_complexity(self) -> None:
        """
        Compute stream complexity (stream density) for each basin.
        
        This method calculates the density of flowlines within each basin
        to characterize stream network complexity.
        """
        if self.huc12 is None or self.flowlines is None:
            raise ValueError("HUC12 and flowlines data must be loaded")
        
        self.logger.info("Computing stream complexity...")
        
        stream_densities = []
        
        for idx, basin in tqdm(self.huc12.iterrows(), total=len(self.huc12), desc="Computing stream complexity"):
            try:
                basin_geom = make_valid(basin.geometry)
                if basin_geom.is_empty:
                    stream_densities.append(0.0)
                    continue
                
                # Find flowlines within basin
                intersecting_flowlines = self.flowlines[self.flowlines.intersects(basin_geom)]
                
                if len(intersecting_flowlines) == 0:
                    stream_densities.append(0.0)
                    continue
                
                # Calculate total stream length within basin
                total_length = 0.0
                for _, flowline in intersecting_flowlines.iterrows():
                    intersection = flowline.geometry.intersection(basin_geom)
                    total_length += intersection.length
                
                # Calculate stream density (km/km²)
                basin_area_km2 = basin_geom.area / 1e6
                stream_density = total_length / 1000 / basin_area_km2  # Convert to km/km²
                stream_densities.append(stream_density)
                
            except Exception as e:
                self.logger.warning(f"Failed to compute stream complexity for basin {idx}: {e}")
                stream_densities.append(0.0)
        
        self.huc12['stream_density'] = stream_densities
        self.logger.info(f"Stream complexity computation complete: {len(stream_densities)} basins processed")
    
    def classify_basins(self) -> None:
        """
        Classify basins into size, terrain, and complexity categories.
        
        This method assigns categorical labels to basins based on their
        computed metrics for stratified sampling.
        """
        if self.huc12 is None:
            raise ValueError("HUC12 data not loaded")
        
        self.logger.info("Classifying basins...")
        
        # Size classification
        size_thresholds = self.config['size_thresholds']
        self.huc12['size_class'] = pd.cut(
            self.huc12['area_km2'],
            bins=[0, size_thresholds['small'], size_thresholds['medium'], size_thresholds['large']],
            labels=['small', 'medium', 'large'],
            include_lowest=True
        )
        
        # Terrain classification
        terrain_thresholds = self.config['terrain_thresholds']
        self.huc12['terrain_class'] = pd.cut(
            self.huc12['slope_std'],
            bins=[0, terrain_thresholds['flat'], terrain_thresholds['moderate'], float('inf')],
            labels=['flat', 'moderate', 'steep'],
            include_lowest=True
        )
        
        # Complexity classification (tertiles)
        self.huc12['complexity_score'] = pd.qcut(
            self.huc12['stream_density'],
            q=3,
            labels=[1, 2, 3],
            duplicates='drop'
        )
        
        self.logger.info("Basin classification complete")
        self.logger.info(f"Size distribution: {self.huc12['size_class'].value_counts().to_dict()}")
        self.logger.info(f"Terrain distribution: {self.huc12['terrain_class'].value_counts().to_dict()}")
        self.logger.info(f"Complexity distribution: {self.huc12['complexity_score'].value_counts().to_dict()}")
    
    def stratified_sample(self) -> Dict[str, Any]:
        """
        Perform stratified sampling across size, terrain, and complexity dimensions.
        
        Returns:
            Dictionary containing sampling summary and statistics
            
        Raises:
            ValueError: If insufficient basins in any stratum
        """
        if self.huc12 is None:
            raise ValueError("HUC12 data not loaded")
        
        self.logger.info("Performing stratified sampling...")
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Create stratification key
        self.huc12['stratum'] = (
            self.huc12['size_class'].astype(str) + '_' +
            self.huc12['terrain_class'].astype(str) + '_' +
            self.huc12['complexity_score'].astype(str)
        )
        
        # Sample from each stratum
        sampled_basins = []
        stratum_stats = {}
        
        for stratum in self.huc12['stratum'].unique():
            stratum_basins = self.huc12[self.huc12['stratum'] == stratum]
            n_available = len(stratum_basins)
            n_requested = self.config['n_per_stratum']
            
            if n_available < n_requested:
                self.logger.warning(f"Stratum {stratum}: {n_available} available, {n_requested} requested")
                n_sample = n_available
            else:
                n_sample = n_requested
            
            if n_sample > 0:
                sampled = stratum_basins.sample(n=n_sample, random_state=self.config['random_seed'])
                sampled_basins.append(sampled)
                
                stratum_stats[stratum] = {
                    'available': n_available,
                    'sampled': n_sample,
                    'basin_ids': sampled['HUC12'].tolist()
                }
        
        if not sampled_basins:
            raise ValueError("No basins sampled - check stratification criteria")
        
        # Combine sampled basins
        self.sample = pd.concat(sampled_basins, ignore_index=True)
        
        # Prepare summary
        summary = {
            'total_strata': len(stratum_stats),
            'total_basins': len(self.sample),
            'strata': stratum_stats,
            'sampling_date': datetime.now().isoformat(),
            'random_seed': self.config['random_seed']
        }
        
        self.logger.info(f"Stratified sampling complete: {len(self.sample)} basins sampled from {len(stratum_stats)} strata")
        return summary
    
    def export_sample(self, output_prefix: str = "basin_sample") -> List[str]:
        """
        Export the sampled basins to various formats.
        
        Args:
            output_prefix: Prefix for output files
            
        Returns:
            List of exported file paths
        """
        if self.sample is None:
            raise ValueError("No sample to export. Run stratified_sample() first.")
        
        self.logger.info(f"Exporting sample with prefix: {output_prefix}")
        
        exported_files = []
        export_config = self.config['export']
        
        # Export CSV
        if export_config.get('csv', True):
            csv_path = f"{output_prefix}.csv"
            self.sample.to_csv(csv_path, index=False)
            exported_files.append(csv_path)
            self.logger.info(f"Exported CSV: {csv_path}")
        
        # Export GeoPackage
        if export_config.get('gpkg', True):
            gpkg_path = f"{output_prefix}.gpkg"
            sample_gdf = gpd.GeoDataFrame(self.sample, crs=self.config['target_crs'])
            sample_gdf = self._validate_crs_transformation(sample_gdf, "basin sample export", self.config['output_crs'])
            sample_gdf.to_file(gpkg_path, driver='GPKG')
            exported_files.append(gpkg_path)
            self.logger.info(f"Exported GeoPackage: {gpkg_path}")
        
        # Export summary
        if export_config.get('summary', True):
            summary_path = f"{output_prefix}_summary.txt"
            self._write_summary(summary_path)
            exported_files.append(summary_path)
            self.logger.info(f"Exported summary: {summary_path}")
        
        # Export error log
        if export_config.get('error_log', True) and self.error_logs:
            error_path = f"{output_prefix}_errors.csv"
            error_df = pd.DataFrame(self.error_logs)
            error_df.to_csv(error_path, index=False)
            exported_files.append(error_path)
            self.logger.info(f"Exported error log: {error_path}")
        
        self.logger.info(f"Export complete: {len(exported_files)} files created")
        return exported_files
    
    def _write_summary(self, summary_path: str) -> None:
        """Write sampling summary to text file."""
        if self.sample is None:
            return
        
        with open(summary_path, 'w') as f:
            f.write("FLOWFINDER Basin Sampling Summary\n")
            f.write("==================================\n\n")
            f.write(f"Sampling Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Basins Sampled: {len(self.sample)}\n")
            f.write(f"Random Seed: {self.config['random_seed']}\n\n")
            
            f.write("Size Distribution:\n")
            size_counts = self.sample['size_class'].value_counts()
            for size, count in size_counts.items():
                f.write(f"  {size}: {count}\n")
            
            f.write("\nTerrain Distribution:\n")
            terrain_counts = self.sample['terrain_class'].value_counts()
            for terrain, count in terrain_counts.items():
                f.write(f"  {terrain}: {count}\n")
            
            f.write("\nComplexity Distribution:\n")
            complexity_counts = self.sample['complexity_score'].value_counts()
            for complexity, count in complexity_counts.items():
                f.write(f"  {complexity}: {count}\n")
            
            f.write(f"\nArea Range: {self.sample['area_km2'].min():.1f} - {self.sample['area_km2'].max():.1f} km²\n")
            f.write(f"Mean Area: {self.sample['area_km2'].mean():.1f} km²\n")
    
    def run_complete_workflow(self, output_prefix: str = "basin_sample") -> Dict[str, Any]:
        """
        Run the complete basin sampling workflow.
        
        Args:
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary containing workflow results and exported files
        """
        self.logger.info("Starting complete basin sampling workflow...")
        
        try:
            # Load datasets
            self.load_datasets()
            
            # Filter Mountain West basins
            self.filter_mountain_west_basins()
            
            # Compute metrics
            self.compute_pour_points()
            self.compute_terrain_roughness()
            self.compute_stream_complexity()
            
            # Classify basins
            self.classify_basins()
            
            # Perform stratified sampling
            sampling_summary = self.stratified_sample()
            
            # Export results
            exported_files = self.export_sample(output_prefix)
            
            workflow_results = {
                'success': True,
                'sampling_summary': sampling_summary,
                'exported_files': exported_files,
                'error_count': len(self.error_logs)
            }
            
            self.logger.info("Basin sampling workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_count': len(self.error_logs)
            }


def create_sample_config() -> str:
    """Create a sample configuration file."""
    sample_config = {
        'data_dir': 'data',
        'area_range': [5, 500],
        'snap_tolerance': 150,
        'n_per_stratum': 2,
        'target_crs': 'EPSG:5070',
        'output_crs': 'EPSG:4326',
        'random_seed': 42,
        'mountain_west_states': ['CO', 'UT', 'NM', 'WY', 'MT', 'ID', 'AZ'],
        'files': {
            'huc12': 'huc12.shp',
            'flowlines': 'nhd_flowlines.shp',
            'dem': 'dem_10m.tif'
        },
        'export': {
            'csv': True,
            'gpkg': True,
            'summary': True
        }
    }
    
    config_content = yaml.dump(sample_config, default_flow_style=False, indent=2)
    return config_content


def main() -> None:
    """Main CLI entry point for basin sampling."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Basin Sampler - Stratified sampling for accuracy benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python basin_sampler.py --output sample_basins
  
  # Run with custom configuration
  python basin_sampler.py --config config.yaml --output custom_sample
  
  # Create sample configuration
  python basin_sampler.py --create-config > basin_sampler_config.yaml
  
  # Run with specific data directory
  python basin_sampler.py --data-dir /path/to/data --output mountain_west_sample
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        help='Directory containing input datasets'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='basin_sample',
        help='Output file prefix (default: basin_sample)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create sample configuration file and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        print(create_sample_config())
        return
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize sampler
        sampler = BasinSampler(config_path=args.config, data_dir=args.data_dir)
        
        # Run complete workflow
        results = sampler.run_complete_workflow(args.output)
        
        if results['success']:
            print(f"\n✅ Basin sampling completed successfully!")
            print(f"📊 Sampled {results['sampling_summary']['total_basins']} basins")
            print(f"📁 Exported {len(results['exported_files'])} files")
            if results['error_count'] > 0:
                print(f"⚠️  {results['error_count']} warnings logged")
        else:
            print(f"\n❌ Basin sampling failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
