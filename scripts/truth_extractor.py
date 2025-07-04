#!/usr/bin/env python3
"""
FLOWFINDER Accuracy Benchmark - Truth Dataset Extraction
========================================================

This module extracts USGS NHD+ HR watershed polygons as ground truth for sampled basins
in the FLOWFINDER accuracy benchmark system.

The truth extractor performs spatial joins between basin pour points and NHD+ catchments,
applies quality validation, and exports clean truth polygons ready for IOU calculation
against FLOWFINDER delineations.

Key Features:
- Spatial join between basin pour points and NHD+ HR catchments
- Multiple extraction strategies with priority ordering
- Quality validation (topology, area ratios, completeness)
- Terrain-specific extraction parameters for Mountain West
- Comprehensive error handling and logging
- Export to multiple formats (GeoPackage, CSV, summary)

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 1.0.0
"""

import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import warnings
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.validation import explain_validity, make_valid
from shapely.ops import unary_union
from tqdm import tqdm

# Import shared geometry utilities
try:
    from .geometry_utils import GeometryDiagnostics
except ImportError:
    # Fall back to absolute import when run as script
    from geometry_utils import GeometryDiagnostics

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyogrio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyproj")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*shapely.geos.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*CRS.*unsafe.*"
)


class TruthExtractor:
    """
    Truth dataset extractor for FLOWFINDER accuracy benchmarking.

    This class handles the extraction of ground truth watershed polygons from
    NHD+ HR catchments based on basin pour points, with comprehensive quality
    validation and error handling.

    Attributes:
        config (Dict[str, Any]): Configuration parameters
        logger (logging.Logger): Logger instance for the extractor
        basin_sample (gpd.GeoDataFrame): Basin sample points with pour points
        catchments (gpd.GeoDataFrame): NHD+ HR catchments
        flowlines (Optional[gpd.GeoDataFrame]): NHD+ flowlines for drainage validation
        truth_polygons (Optional[gpd.GeoDataFrame]): Extracted truth polygons
        error_logs (List[Dict[str, Any]]): Structured error tracking
    """

    def __init__(
        self, config_path: Optional[str] = None, data_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the truth extractor with configuration.

        Args:
            config_path: Path to YAML configuration file
            data_dir: Directory containing input datasets

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        # Set up logging first
        self._setup_logging()
        self.config = self._load_config(config_path, data_dir)
        self.error_logs: List[Dict[str, Any]] = []
        self._validate_config()

        # Initialize data attributes
        self.basin_sample: Optional[gpd.GeoDataFrame] = None
        self.catchments: Optional[gpd.GeoDataFrame] = None
        self.flowlines: Optional[gpd.GeoDataFrame] = None
        self.truth_polygons: Optional[gpd.GeoDataFrame] = None

        # Initialize geometry diagnostics
        self.geometry_diagnostics = GeometryDiagnostics(
            logger=self.logger, config=self.config
        )

        self.logger.info("TruthExtractor initialized successfully")

    def _load_config(
        self, config_path: Optional[str], data_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file or use defaults.

        Args:
            config_path: Path to configuration file
            data_dir: Data directory override

        Returns:
            Configuration dictionary
        """
        default_config = {
            "data_dir": data_dir or "data",
            "basin_sample_file": "basin_sample.csv",
            "buffer_tolerance": 500,  # meters for spatial join
            "min_area_ratio": 0.1,  # minimum truth area / sample area ratio
            "max_area_ratio": 10.0,  # maximum truth area / sample area ratio
            "target_crs": "EPSG:5070",  # Albers Equal Area CONUS
            "output_crs": "EPSG:4326",  # WGS84 for export
            "min_polygon_area": 1.0,  # km² minimum area
            "max_polygon_parts": 10,  # maximum polygon parts
            "max_attempts": 3,  # maximum extraction attempts
            "retry_with_different_strategies": True,
            "chunk_size": 100,  # for memory management
            "memory_management": {
                "max_memory_mb": 2048,  # Maximum memory usage in MB
                "large_file_threshold_mb": 50,  # Files larger than this get special handling
                "gc_frequency": 50,  # Garbage collection frequency (every N extractions)
                "monitor_memory": True,  # Enable memory monitoring
                "fail_on_validation_errors": False,  # Whether to fail on validation errors
            },
            "geometry_repair": {
                "enable_diagnostics": True,  # Enable detailed geometry diagnostics
                "enable_repair_attempts": True,  # Enable automatic repair attempts
                "invalid_geometry_action": "remove",  # 'remove', 'keep', 'convert_to_point'
                "max_repair_attempts": 3,  # Maximum repair attempts per geometry
                "detailed_logging": True,  # Enable detailed repair logging
                "repair_strategies": {
                    "buffer_fix": True,  # Use buffer(0) to fix self-intersections
                    "simplify": True,  # Use simplify() for duplicate points
                    "make_valid": True,  # Use make_valid() as fallback
                    "convex_hull": False,  # Use convex hull (changes geometry significantly)
                    "orient_fix": True,  # Fix orientation issues
                    "simplify_holes": True,  # Remove problematic holes
                },
            },
            "shapefile_schemas": {
                "basin_sample": {
                    "required_columns": ["ID", "Pour_Point_Lat", "Pour_Point_Lon"],
                    "column_types": {
                        "ID": "string",
                        "Pour_Point_Lat": "float",
                        "Pour_Point_Lon": "float",
                        "Area_km2": "float",
                        "Terrain_Class": "string",
                    },
                    "not_null_columns": ["ID", "Pour_Point_Lat", "Pour_Point_Lon"],
                    "value_ranges": {
                        "Pour_Point_Lat": {
                            "min": 25,
                            "max": 55,
                        },  # Mountain West latitudes
                        "Pour_Point_Lon": {
                            "min": -125,
                            "max": -100,
                        },  # Mountain West longitudes
                        "Area_km2": {"min": 0.01, "max": 10000},
                    },
                },
                "nhd_catchments": {
                    "required_columns": ["FEATUREID"],
                    "column_types": {
                        "FEATUREID": "integer",
                        "GRIDCODE": "integer",
                        "AREASQKM": "float",
                    },
                    "not_null_columns": ["FEATUREID"],
                    "geometry_type": "Polygon",
                    "value_ranges": {
                        "FEATUREID": {"min": 1},
                        "AREASQKM": {"min": 0.001, "max": 5000},
                    },
                    "consistency_rules": {"unique_values": {"columns": ["FEATUREID"]}},
                },
                "nhd_flowlines": {
                    "required_columns": ["COMID"],
                    "column_types": {"COMID": "integer", "LENGTHKM": "float"},
                    "not_null_columns": ["COMID"],
                    "geometry_type": "LineString",
                    "consistency_rules": {"unique_values": {"columns": ["COMID"]}},
                },
            },
            "extraction_priority": {
                1: "contains_point",
                2: "largest_containing",
                3: "nearest_centroid",
                4: "largest_intersecting",
            },
            "files": {
                "nhd_catchments": "nhd_hr_catchments.shp",
                "nhd_flowlines": "nhd_flowlines.shp",  # Optional for drainage validation
            },
            "quality_checks": {
                "topology_validation": True,
                "area_validation": True,
                "completeness_check": True,
                "drainage_check": False,  # Requires flowlines
            },
            "terrain_extraction": {
                "alpine": {
                    "max_parts": 15,
                    "min_drainage_density": 0.01,
                    "buffer_tolerance": 1000,
                },
                "desert": {
                    "max_parts": 3,
                    "min_drainage_density": 0.001,
                    "buffer_tolerance": 200,
                },
            },
            "export": {
                "gpkg": True,
                "csv": True,
                "summary": True,
                "failed_extractions": True,
                "error_log": True,
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
        else:
            self.logger.info("Using default configuration")

        return default_config

    def _setup_logging(self) -> None:
        """Configure logging for the extraction session."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs/truth_extractor")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (
            log_dir / f"truth_extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Set up handlers with rotation (daily rotation, keep 30 days)
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        console_handler = logging.StreamHandler(sys.stdout)

        # Set format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure logging with handlers
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[file_handler, console_handler],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - log file: {log_file}")

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate area ratios
        if self.config["min_area_ratio"] >= self.config["max_area_ratio"]:
            raise ValueError("min_area_ratio must be < max_area_ratio")

        if self.config["min_area_ratio"] <= 0 or self.config["max_area_ratio"] <= 0:
            raise ValueError("Area ratios must be positive")

        # Validate tolerance
        if self.config["buffer_tolerance"] <= 0:
            raise ValueError("buffer_tolerance must be > 0")

        # Validate file paths (more lenient for test environments)
        data_dir = Path(self.config["data_dir"])
        if not data_dir.exists():
            self.logger.warning(f"Data directory does not exist yet: {data_dir}")

        # Check required files (warn but don't fail - files may be created later)
        basin_file = data_dir / self.config["basin_sample_file"]
        if not basin_file.exists():
            self.logger.warning(f"Basin sample file not found yet: {basin_file}")

        catchments_file = data_dir / self.config["files"]["nhd_catchments"]
        if not catchments_file.exists():
            self.logger.warning(f"Catchments file not found yet: {catchments_file}")

        self.logger.info("Configuration validation passed")

    def _validate_crs_transformation(
        self, gdf: gpd.GeoDataFrame, source_description: str, target_crs: str
    ) -> gpd.GeoDataFrame:
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
        self.logger.info(
            f"Transforming {source_description} from {original_crs} to {target_crs}"
        )

        # Check if already in target CRS
        if original_crs == target_crs:
            self.logger.info(f"{source_description} already in target CRS")
            return gdf

        try:
            # Attempt primary transformation
            transformed_gdf = gdf.to_crs(target_crs)

            # Validate transformation success
            if len(transformed_gdf) != len(gdf):
                raise ValueError(
                    "Transformation resulted in different number of features"
                )

            if transformed_gdf.is_empty.any():
                raise ValueError("Transformation resulted in empty geometries")

            # Validate coordinate ranges based on target CRS
            if not self._validate_coordinate_ranges(
                transformed_gdf, target_crs, source_description
            ):
                # Try fallback strategies
                transformed_gdf = self._apply_crs_fallback_strategies(
                    gdf, target_crs, source_description
                )

            self.logger.info(
                f"Successfully transformed {source_description} to {target_crs}"
            )
            return transformed_gdf

        except Exception as e:
            self.logger.error(
                f"CRS transformation failed for {source_description}: {e}"
            )
            # Try fallback strategies
            return self._apply_crs_fallback_strategies(
                gdf, target_crs, source_description
            )

    def _validate_coordinate_ranges(
        self, gdf: gpd.GeoDataFrame, crs: str, source_description: str
    ) -> bool:
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
                "EPSG:4326": {  # WGS84
                    "x_min": -180,
                    "x_max": 180,
                    "y_min": -90,
                    "y_max": 90,
                    "name": "WGS84",
                },
                "EPSG:5070": {  # NAD83 Albers Equal Area CONUS
                    "x_min": -2500000,
                    "x_max": 2500000,
                    "y_min": -1500000,
                    "y_max": 1500000,
                    "name": "NAD83 Albers CONUS",
                },
                "EPSG:3857": {  # Web Mercator
                    "x_min": -20037508,
                    "x_max": 20037508,
                    "y_min": -20037508,
                    "y_max": 20037508,
                    "name": "Web Mercator",
                },
            }

            # Mountain West specific validation for geographic CRS
            if crs == "EPSG:4326":
                quality_checks = self.config.get("quality_checks", {})
                mountain_west_bounds = {
                    "x_min": quality_checks.get("min_lon", -125),
                    "x_max": quality_checks.get("max_lon", -100),
                    "y_min": quality_checks.get("min_lat", 30),
                    "y_max": quality_checks.get("max_lat", 50),
                }
                expected_ranges["EPSG:4326"].update(mountain_west_bounds)

            if crs not in expected_ranges:
                self.logger.warning(
                    f"No coordinate range validation available for CRS {crs}"
                )
                return True

            ranges = expected_ranges[crs]

            # Check if coordinates are within expected ranges
            if (
                bounds[0] < ranges["x_min"]
                or bounds[2] > ranges["x_max"]
                or bounds[1] < ranges["y_min"]
                or bounds[3] > ranges["y_max"]
            ):

                self.logger.warning(
                    f"Coordinates for {source_description} appear out of range for {ranges['name']}: "
                    f"bounds=({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}), "
                    f"expected x: [{ranges['x_min']}, {ranges['x_max']}], "
                    f"y: [{ranges['y_min']}, {ranges['y_max']}]"
                )
                return False

            self.logger.debug(
                f"Coordinate validation passed for {source_description} in {ranges['name']}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Coordinate range validation failed for {source_description}: {e}"
            )
            return False

    def _apply_crs_fallback_strategies(
        self, gdf: gpd.GeoDataFrame, target_crs: str, source_description: str
    ) -> gpd.GeoDataFrame:
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
        self.logger.warning(
            f"Applying CRS fallback strategies for {source_description}"
        )

        fallback_strategies = [
            self._fallback_assume_wgs84,
            self._fallback_force_crs,
            self._fallback_geometry_validation,
            self._fallback_coordinate_cleanup,
        ]

        for i, strategy in enumerate(fallback_strategies, 1):
            try:
                self.logger.info(
                    f"Trying fallback strategy {i}/{len(fallback_strategies)} for {source_description}"
                )
                result = strategy(gdf, target_crs, source_description)

                if result is not None and len(result) > 0:
                    # Validate the fallback result
                    if self._validate_coordinate_ranges(
                        result, target_crs, f"{source_description} (fallback {i})"
                    ):
                        self.logger.info(
                            f"Fallback strategy {i} succeeded for {source_description}"
                        )
                        return result
                    else:
                        self.logger.warning(
                            f"Fallback strategy {i} produced invalid coordinates for {source_description}"
                        )

            except Exception as e:
                self.logger.warning(
                    f"Fallback strategy {i} failed for {source_description}: {e}"
                )
                continue

        # All fallback strategies failed
        error_msg = f"All CRS transformation strategies failed for {source_description}"
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    def _fallback_assume_wgs84(
        self, gdf: gpd.GeoDataFrame, target_crs: str, source_description: str
    ) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Assume source is WGS84 if CRS is missing."""
        if gdf.crs is None:
            self.logger.info(
                f"Assuming WGS84 for {source_description} with missing CRS"
            )
            gdf_copy = gdf.copy()
            gdf_copy.crs = "EPSG:4326"
            return gdf_copy.to_crs(target_crs)
        return None

    def _fallback_force_crs(
        self, gdf: gpd.GeoDataFrame, target_crs: str, source_description: str
    ) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Force CRS without transformation, then transform."""
        try:
            gdf_copy = gdf.copy()
            gdf_copy.crs = target_crs
            return gdf_copy
        except Exception:
            return None

    def _fallback_geometry_validation(
        self, gdf: gpd.GeoDataFrame, target_crs: str, source_description: str
    ) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Fix invalid geometries before transformation using enhanced diagnostics."""
        try:
            gdf_copy = gdf.copy()

            # Use enhanced geometry diagnostics if enabled
            geometry_config = self.config.get("geometry_repair", {})
            if geometry_config.get("enable_diagnostics", True):
                self.logger.info(
                    f"Applying enhanced geometry repair for {source_description} before CRS transformation"
                )
                gdf_copy = self.geometry_diagnostics.diagnose_and_repair_geometries(
                    gdf_copy, f"{source_description} (CRS fallback)"
                )
            else:
                # Basic geometry fix (backward compatibility)
                invalid_mask = ~gdf_copy.is_valid
                if invalid_mask.any():
                    self.logger.info(
                        f"Fixing {invalid_mask.sum()} invalid geometries in {source_description}"
                    )
                    gdf_copy.loc[invalid_mask, "geometry"] = gdf_copy.loc[
                        invalid_mask, "geometry"
                    ].apply(make_valid)

            return gdf_copy.to_crs(target_crs)
        except Exception:
            return None

    def _fallback_coordinate_cleanup(
        self, gdf: gpd.GeoDataFrame, target_crs: str, source_description: str
    ) -> Optional[gpd.GeoDataFrame]:
        """Fallback: Remove features with problematic coordinates."""
        try:
            gdf_copy = gdf.copy()

            # Remove features with extreme coordinates that might cause transformation issues
            bounds = gdf_copy.bounds

            # For geographic coordinates, remove extreme values
            if gdf_copy.crs and "EPSG:4326" in str(gdf_copy.crs):
                valid_mask = (
                    (bounds["minx"] >= -180)
                    & (bounds["maxx"] <= 180)
                    & (bounds["miny"] >= -90)
                    & (bounds["maxy"] <= 90)
                )
            else:
                # For projected coordinates, remove extremely large values
                valid_mask = (
                    (abs(bounds["minx"]) < 1e8)
                    & (abs(bounds["maxx"]) < 1e8)
                    & (abs(bounds["miny"]) < 1e8)
                    & (abs(bounds["maxy"]) < 1e8)
                )

            if not valid_mask.all():
                removed_count = (~valid_mask).sum()
                self.logger.warning(
                    f"Removing {removed_count} features with extreme coordinates from {source_description}"
                )
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
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def _check_memory_usage(self, operation: str = "operation") -> bool:
        """
        Check if memory usage is within acceptable limits.

        Args:
            operation: Description of current operation for logging

        Returns:
            True if memory usage is acceptable, False if approaching limits
        """
        memory_config = self.config.get("memory_management", {})
        if not memory_config.get("monitor_memory", True):
            return True

        memory_info = self._get_memory_usage()
        max_memory_mb = memory_config.get("max_memory_mb", 2048)

        if memory_info["rss_mb"] > max_memory_mb:
            self.logger.warning(
                f"Memory usage ({memory_info['rss_mb']:.1f} MB) exceeds limit "
                f"({max_memory_mb} MB) during {operation}"
            )
            return False

        if memory_info["percent"] > 80:
            self.logger.warning(
                f"High memory usage ({memory_info['percent']:.1f}%) during {operation}"
            )
            return False

        self.logger.debug(
            f"Memory usage during {operation}: {memory_info['rss_mb']:.1f} MB "
            f"({memory_info['percent']:.1f}%)"
        )
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

    def _validate_basin_sample_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate basin sample CSV schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes, False otherwise
        """
        schema_config = self.config.get("shapefile_schemas", {}).get("basin_sample", {})
        if not schema_config:
            return True

        validation_passed = True

        # Check required columns
        required_columns = schema_config.get("required_columns", [])
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Basin sample missing required columns: {missing_cols}")
            validation_passed = False

        # Check data types and ranges
        for col_name in required_columns:
            if col_name not in df.columns:
                continue

            # Check for null values in required columns
            null_count = df[col_name].isnull().sum()
            if null_count > 0:
                self.logger.error(
                    f"Basin sample column '{col_name}' has {null_count} null values"
                )
                validation_passed = False

            # Check value ranges
            value_ranges = schema_config.get("value_ranges", {})
            if value_ranges and pd.api.types.is_numeric_dtype(df[col_name]):
                if "min" in value_ranges:
                    min_val = df[col_name].min()
                    if min_val < value_ranges["min"]:
                        self.logger.warning(
                            f"Basin sample column '{col_name}' has values below minimum: {min_val} < {value_ranges['min']}"
                        )

                if "max" in value_ranges:
                    max_val = df[col_name].max()
                    if max_val > value_ranges["max"]:
                        self.logger.warning(
                            f"Basin sample column '{col_name}' has values above maximum: {max_val} > {value_ranges['max']}"
                        )

        return validation_passed

    def _validate_catchments_schema(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Validate NHD+ catchments shapefile schema.

        Args:
            gdf: GeoDataFrame to validate

        Returns:
            True if validation passes, False otherwise
        """
        schema_config = self.config.get("shapefile_schemas", {}).get(
            "nhd_catchments", {}
        )
        if not schema_config:
            return True

        validation_passed = True

        # Check required columns
        required_columns = schema_config.get("required_columns", [])
        missing_cols = [col for col in required_columns if col not in gdf.columns]
        if missing_cols:
            self.logger.error(
                f"NHD+ catchments missing required columns: {missing_cols}"
            )
            validation_passed = False

        # Enhanced geometry validity checking with detailed diagnostics
        if len(gdf) > 0:
            geometry_config = self.config.get("geometry_repair", {})

            if geometry_config.get("enable_diagnostics", True):
                # Use enhanced geometry analysis
                geometry_stats = self.geometry_diagnostics.analyze_geometry_issues(
                    gdf, "NHD+ catchments validation"
                )

                # Log detailed diagnostics
                self.logger.info(
                    f"NHD+ catchments geometry analysis: {geometry_stats['total_invalid']}/{geometry_stats['total_features']} invalid"
                )

                if geometry_stats["total_invalid"] > 0:
                    invalid_percentage = (
                        geometry_stats["total_invalid"]
                        / geometry_stats["total_features"]
                    ) * 100

                    # Log issue type breakdown
                    if geometry_stats["issue_types"]:
                        issue_summary = ", ".join(
                            [
                                f"{issue}: {count}"
                                for issue, count in geometry_stats[
                                    "issue_types"
                                ].items()
                            ]
                        )
                        self.logger.info(f"Issue types: {issue_summary}")

                    # Log detailed diagnostics for first few invalid geometries
                    for diagnostic in geometry_stats["detailed_diagnostics"][:3]:
                        self.logger.warning(
                            f"Invalid geometry {diagnostic['index']}: {diagnostic['explanation']}"
                        )

                    if invalid_percentage > 10:
                        self.logger.error(
                            f"NHD+ catchments has high percentage of invalid geometries: {invalid_percentage:.1f}%"
                        )
                        validation_passed = False
                    else:
                        self.logger.warning(
                            f"NHD+ catchments has some invalid geometries: {geometry_stats['total_invalid']} ({invalid_percentage:.1f}%)"
                        )
            else:
                # Basic geometry validation (backward compatibility)
                invalid_geoms = ~gdf.is_valid
                invalid_count = invalid_geoms.sum()
                if invalid_count > 0:
                    invalid_percentage = (invalid_count / len(gdf)) * 100
                    if invalid_percentage > 10:
                        self.logger.error(
                            f"NHD+ catchments has high percentage of invalid geometries: {invalid_percentage:.1f}%"
                        )
                        validation_passed = False
                    else:
                        self.logger.warning(
                            f"NHD+ catchments has some invalid geometries: {invalid_count} ({invalid_percentage:.1f}%)"
                        )

        # Check for duplicate FEATUREID values
        if "FEATUREID" in gdf.columns:
            duplicates = gdf["FEATUREID"].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(
                    f"NHD+ catchments has {duplicates} duplicate FEATUREID values"
                )

        self.logger.info(
            f"NHD+ catchments schema validation: {'PASSED' if validation_passed else 'FAILED'}"
        )
        return validation_passed

    def _validate_flowlines_schema(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Validate NHD+ flowlines shapefile schema.

        Args:
            gdf: GeoDataFrame to validate

        Returns:
            True if validation passes, False otherwise
        """
        schema_config = self.config.get("shapefile_schemas", {}).get(
            "nhd_flowlines", {}
        )
        if not schema_config:
            return True

        validation_passed = True

        # Check required columns
        required_columns = schema_config.get("required_columns", [])
        missing_cols = [col for col in required_columns if col not in gdf.columns]
        if missing_cols:
            self.logger.error(
                f"NHD+ flowlines missing required columns: {missing_cols}"
            )
            validation_passed = False

        # Check geometry type
        expected_geom_type = schema_config.get("geometry_type", "LineString")
        if len(gdf) > 0:
            geom_types = gdf.geom_type.unique()
            if expected_geom_type not in geom_types:
                self.logger.error(
                    f"NHD+ flowlines expected geometry type {expected_geom_type}, found: {list(geom_types)}"
                )
                validation_passed = False

        # Check for duplicate COMID values
        if "COMID" in gdf.columns:
            duplicates = gdf["COMID"].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(
                    f"NHD+ flowlines has {duplicates} duplicate COMID values"
                )

        # Check value ranges
        value_ranges = schema_config.get("value_ranges", {})
        for col_name, ranges in value_ranges.items():
            if col_name not in gdf.columns:
                continue

            if pd.api.types.is_numeric_dtype(gdf[col_name]):
                if "min" in ranges:
                    min_val = gdf[col_name].min()
                    if min_val < ranges["min"]:
                        self.logger.warning(
                            f"NHD+ flowlines column '{col_name}' has values below minimum: {min_val} < {ranges['min']}"
                        )

                if "max" in ranges:
                    max_val = gdf[col_name].max()
                    if max_val > ranges["max"]:
                        self.logger.warning(
                            f"NHD+ flowlines column '{col_name}' has values above maximum: {max_val} > {ranges['max']}"
                        )

        self.logger.info(
            f"NHD+ flowlines schema validation: {'PASSED' if validation_passed else 'FAILED'}"
        )
        return validation_passed

    def load_datasets(self) -> None:
        """
        Load all required datasets for truth extraction.

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If datasets cannot be loaded
        """
        self.logger.info("Loading datasets for truth extraction...")

        try:
            # Load basin sample
            self._load_basin_sample()

            # Load NHD+ data
            self._load_nhd_data()

            self.logger.info("All datasets loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise

    def _load_basin_sample(self) -> None:
        """Load basin sample points from CSV."""
        sample_file = Path(self.config["data_dir"]) / self.config["basin_sample_file"]
        self.logger.info(f"Loading basin sample from {sample_file}")

        try:
            # Load CSV and create GeoDataFrame
            df = pd.read_csv(sample_file)

            # Perform schema validation
            if not self._validate_basin_sample_schema(df):
                if self.config.get("memory_management", {}).get(
                    "fail_on_validation_errors", False
                ):
                    raise ValueError("Basin sample CSV failed schema validation")
                else:
                    self.logger.warning(
                        "Basin sample CSV validation failed - proceeding with caution"
                    )

            # Create point geometries from lat/lon
            geometry = [
                Point(lon, lat)
                for lon, lat in zip(df["Pour_Point_Lon"], df["Pour_Point_Lat"])
            ]
            self.basin_sample = gpd.GeoDataFrame(
                df, geometry=geometry, crs=self.config["output_crs"]
            )

            # Transform to target CRS for spatial operations
            self.basin_sample = self._validate_crs_transformation(
                self.basin_sample, "basin sample points", self.config["target_crs"]
            )

            self.logger.info(f"Loaded {len(self.basin_sample)} basin sample points")

        except Exception as e:
            self.logger.error(f"Failed to load basin sample: {e}")
            raise

    def _load_nhd_data(self) -> None:
        """Load NHD+ HR catchments and optional flowlines with memory management."""
        data_dir = Path(self.config["data_dir"])
        files = self.config["files"]
        target_crs = self.config["target_crs"]

        try:
            # Load catchments with memory monitoring
            self.logger.info("Loading NHD+ HR catchments...")
            catchments_file = data_dir / files["nhd_catchments"]

            # Check file size and memory before loading
            file_size_mb = self._get_file_size_mb(catchments_file)
            self.logger.info(f"NHD+ catchments file size: {file_size_mb:.1f} MB")
            self._check_memory_usage("NHD+ catchments loading")

            self.catchments = gpd.read_file(catchments_file)
            self.catchments = self._validate_crs_transformation(
                self.catchments, "NHD+ catchments", target_crs
            )

            # Perform geometry diagnostics and repair
            geometry_config = self.config.get("geometry_repair", {})
            if geometry_config.get("enable_diagnostics", True) or geometry_config.get(
                "enable_repair_attempts", True
            ):
                self.catchments = (
                    self.geometry_diagnostics.diagnose_and_repair_geometries(
                        self.catchments, "NHD+ catchments"
                    )
                )

            # Perform schema validation
            if not self._validate_catchments_schema(self.catchments):
                if self.config.get("memory_management", {}).get(
                    "fail_on_validation_errors", False
                ):
                    raise ValueError("NHD+ catchments failed schema validation")
                else:
                    self.logger.warning(
                        "NHD+ catchments schema validation failed - proceeding with caution"
                    )

            # Ensure required columns exist
            if "FEATUREID" not in self.catchments.columns:
                self.logger.warning(
                    "FEATUREID column not found in catchments - using index"
                )
                self.catchments["FEATUREID"] = self.catchments.index.astype(str)

            self.logger.info(f"Loaded {len(self.catchments)} NHD+ HR catchments")

            # Load flowlines if drainage check enabled
            if self.config["quality_checks"]["drainage_check"] and files.get(
                "nhd_flowlines"
            ):
                self.logger.info("Loading NHD+ flowlines for drainage validation...")
                flowlines_file = data_dir / files["nhd_flowlines"]
                if flowlines_file.exists():
                    self.flowlines = gpd.read_file(flowlines_file)
                    self.flowlines = self._validate_crs_transformation(
                        self.flowlines, "NHD+ flowlines", target_crs
                    )

                    # Perform geometry diagnostics and repair
                    geometry_config = self.config.get("geometry_repair", {})
                    if geometry_config.get(
                        "enable_diagnostics", True
                    ) or geometry_config.get("enable_repair_attempts", True):
                        self.flowlines = (
                            self.geometry_diagnostics.diagnose_and_repair_geometries(
                                self.flowlines, "NHD+ flowlines"
                            )
                        )

                    # Perform schema validation
                    if not self._validate_flowlines_schema(self.flowlines):
                        if self.config.get("memory_management", {}).get(
                            "fail_on_validation_errors", False
                        ):
                            raise ValueError("NHD+ flowlines failed schema validation")
                    else:
                        self.logger.warning(
                            "NHD+ flowlines schema validation failed - proceeding with caution"
                        )

                    self.logger.info(f"Loaded {len(self.flowlines)} flowlines")
                else:
                    self.logger.warning(f"Flowlines file not found: {flowlines_file}")
                    self.flowlines = None
            else:
                self.flowlines = None

        except Exception as e:
            self.logger.error(f"Failed to load NHD+ data: {e}")
            raise

    def extract_truth_polygons(self) -> gpd.GeoDataFrame:
        """
        Extract truth watershed polygons for each basin sample point.

        Returns:
            GeoDataFrame containing extracted truth polygons with metadata

        Raises:
            ValueError: If datasets are not loaded
        """
        if self.basin_sample is None or self.catchments is None:
            raise ValueError("Basin sample and catchments must be loaded")

        self.logger.info(
            "Extracting truth polygons via spatial join with memory management..."
        )

        buffer_tolerance = self.config["buffer_tolerance"]
        truth_polygons = []

        # Memory management configuration
        memory_config = self.config.get("memory_management", {})
        gc_frequency = memory_config.get("gc_frequency", 50)
        processed_count = 0
        memory_warnings = 0

        for idx, basin in tqdm(
            self.basin_sample.iterrows(),
            total=len(self.basin_sample),
            desc="Extracting truth polygons",
        ):
            basin_id = str(basin["ID"])

            try:
                # Create buffer around pour point for spatial join tolerance
                point_buffer = basin.geometry.buffer(buffer_tolerance)

                # Find intersecting catchments
                intersecting = self.catchments[self.catchments.intersects(point_buffer)]

                if len(intersecting) == 0:
                    self._log_error(
                        basin_id,
                        "no_catchment",
                        f"No catchments within {buffer_tolerance}m",
                    )
                    continue
                elif len(intersecting) == 1:
                    # Single catchment - ideal case
                    truth_poly = intersecting.iloc[0].copy()
                    truth_poly["extraction_method"] = "single_catchment"
                else:
                    # Multiple catchments - need to choose best one
                    self.logger.debug(
                        f"Basin {basin_id}: {len(intersecting)} intersecting catchments"
                    )

                    # Apply extraction strategy priority
                    truth_poly = self._apply_extraction_strategy(
                        basin, intersecting, basin_id
                    )
                    if truth_poly is None:
                        continue

                # Add basin metadata
                truth_poly["ID"] = basin_id
                truth_poly["sample_area_km2"] = basin.get("Area_km2", np.nan)
                truth_poly["sample_terrain"] = basin.get("Terrain_Class", "unknown")
                truth_poly["sample_complexity"] = basin.get("Complexity_Score", np.nan)

                # Calculate truth polygon area
                truth_poly["truth_area_km2"] = truth_poly.geometry.area / 1e6

                truth_polygons.append(truth_poly)

                # Memory management
                processed_count += 1
                if processed_count % gc_frequency == 0:
                    # Check memory usage periodically
                    if not self._check_memory_usage(
                        f"truth extraction (processed {processed_count})"
                    ):
                        memory_warnings += 1
                        self._force_garbage_collection()

            except Exception as e:
                self._log_error(basin_id, "extraction_error", str(e))
                continue

        # Log memory management statistics
        if memory_warnings > 0:
            self.logger.warning(
                f"Encountered {memory_warnings} memory warnings during extraction"
            )

        # Final cleanup
        self._force_garbage_collection()

        if not truth_polygons:
            raise ValueError(
                "No truth polygons extracted - check spatial data and configuration"
            )

        # Create GeoDataFrame
        self.truth_polygons = gpd.GeoDataFrame(
            truth_polygons, crs=self.config["target_crs"]
        )

        self.logger.info(
            f"Truth extraction complete: {len(self.truth_polygons)} polygons extracted"
        )
        return self.truth_polygons

    def _apply_extraction_strategy(
        self, basin: pd.Series, intersecting: gpd.GeoDataFrame, basin_id: str
    ) -> Optional[pd.Series]:
        """
        Apply extraction strategy priority to select best catchment.

        Args:
            basin: Basin sample row
            intersecting: Intersecting catchments
            basin_id: Basin identifier for logging

        Returns:
            Selected catchment or None if extraction fails
        """
        priority_order = self.config["extraction_priority"]

        for priority, strategy in priority_order.items():
            try:
                if strategy == "contains_point":
                    # Choose catchment containing the pour point
                    containing = intersecting[intersecting.contains(basin.geometry)]
                    if len(containing) == 1:
                        truth_poly = containing.iloc[0].copy()
                        truth_poly["extraction_method"] = "contains_point"
                        return truth_poly
                    elif len(containing) > 1:
                        # Multiple containing - choose largest
                        largest_idx = containing.geometry.area.idxmax()
                        truth_poly = containing.loc[largest_idx].copy()
                        truth_poly["extraction_method"] = "largest_containing"
                        self._log_error(
                            basin_id,
                            "multiple_containing",
                            f"{len(containing)} catchments contain point",
                        )
                        return truth_poly

                elif strategy == "largest_containing":
                    # Choose largest catchment containing the point
                    containing = intersecting[intersecting.contains(basin.geometry)]
                    if len(containing) > 0:
                        largest_idx = containing.geometry.area.idxmax()
                        truth_poly = containing.loc[largest_idx].copy()
                        truth_poly["extraction_method"] = "largest_containing"
                        return truth_poly

                elif strategy == "nearest_centroid":
                    # Choose nearest catchment by centroid distance
                    distances = intersecting.geometry.centroid.distance(basin.geometry)
                    nearest_idx = distances.idxmin()
                    truth_poly = intersecting.loc[nearest_idx].copy()
                    truth_poly["extraction_method"] = "nearest_centroid"
                    self._log_error(
                        basin_id,
                        "point_not_contained",
                        "Pour point not contained in any catchment",
                    )
                    return truth_poly

                elif strategy == "largest_intersecting":
                    # Choose largest intersecting catchment
                    largest_idx = intersecting.geometry.area.idxmax()
                    truth_poly = intersecting.loc[largest_idx].copy()
                    truth_poly["extraction_method"] = "largest_intersecting"
                    return truth_poly

            except Exception as e:
                self.logger.warning(
                    f"Strategy {strategy} failed for basin {basin_id}: {e}"
                )
                continue

        return None

    def validate_truth_quality(self) -> Dict[str, Any]:
        """
        Validate quality of extracted truth polygons.

        Returns:
            Dictionary containing validation results and statistics
        """
        if self.truth_polygons is None:
            raise ValueError("No truth polygons to validate")

        self.logger.info("Validating truth polygon quality...")

        validation_results = {
            "total_polygons": len(self.truth_polygons),
            "valid_polygons": 0,
            "invalid_polygons": 0,
            "area_ratio_violations": 0,
            "topology_errors": 0,
            "completeness_issues": 0,
            "validation_details": [],
        }

        quality_checks = self.config["quality_checks"]

        for idx, row in self.truth_polygons.iterrows():
            basin_id = row["ID"]
            validation_detail = {"ID": basin_id, "issues": []}

            try:
                # Topology validation
                if quality_checks["topology_validation"]:
                    if not row.geometry.is_valid:
                        validation_detail["issues"].append("invalid_topology")
                        validation_results["topology_errors"] += 1
                    elif row.geometry.is_empty:
                        validation_detail["issues"].append("empty_geometry")
                        validation_results["completeness_issues"] += 1

                # Area validation
                if quality_checks["area_validation"]:
                    truth_area = row["truth_area_km2"]
                    sample_area = row["sample_area_km2"]

                    if not np.isnan(sample_area) and sample_area > 0:
                        area_ratio = truth_area / sample_area
                        min_ratio = self.config["min_area_ratio"]
                        max_ratio = self.config["max_area_ratio"]

                        if area_ratio < min_ratio or area_ratio > max_ratio:
                            validation_detail["issues"].append(
                                f"area_ratio_violation_{area_ratio:.2f}"
                            )
                            validation_results["area_ratio_violations"] += 1

                # Completeness check
                if quality_checks["completeness_check"]:
                    if truth_area < self.config["min_polygon_area"]:
                        validation_detail["issues"].append("area_too_small")
                        validation_results["completeness_issues"] += 1

                # Mark as valid if no issues
                if not validation_detail["issues"]:
                    validation_results["valid_polygons"] += 1
                else:
                    validation_results["invalid_polygons"] += 1

                validation_results["validation_details"].append(validation_detail)

            except Exception as e:
                self.logger.warning(f"Validation failed for basin {basin_id}: {e}")
                validation_detail["issues"].append("validation_error")
                validation_results["invalid_polygons"] += 1
                validation_results["validation_details"].append(validation_detail)

        # Log validation summary
        self.logger.info(f"Quality validation complete:")
        self.logger.info(f"  Valid polygons: {validation_results['valid_polygons']}")
        self.logger.info(
            f"  Invalid polygons: {validation_results['invalid_polygons']}"
        )
        self.logger.info(
            f"  Area ratio violations: {validation_results['area_ratio_violations']}"
        )
        self.logger.info(f"  Topology errors: {validation_results['topology_errors']}")
        self.logger.info(
            f"  Completeness issues: {validation_results['completeness_issues']}"
        )

        return validation_results

    def export_truth_dataset(self, output_prefix: str = "truth_polygons") -> List[str]:
        """
        Export truth dataset to various formats.

        Args:
            output_prefix: Prefix for output files

        Returns:
            List of exported file paths
        """
        if self.truth_polygons is None:
            raise ValueError("No truth polygons to export")

        self.logger.info(f"Exporting truth dataset with prefix: {output_prefix}")

        exported_files = []
        export_config = self.config["export"]

        # Export GeoPackage
        if export_config.get("gpkg", True):
            gpkg_path = f"{output_prefix}.gpkg"
            export_gdf = self._validate_crs_transformation(
                self.truth_polygons, "truth polygons export", self.config["output_crs"]
            )
            export_gdf.to_file(gpkg_path, driver="GPKG")
            exported_files.append(gpkg_path)
            self.logger.info(f"Exported GeoPackage: {gpkg_path}")

        # Export CSV (non-geometry attributes)
        if export_config.get("csv", True):
            csv_path = f"{output_prefix}.csv"
            # Remove geometry column for CSV export
            csv_df = self.truth_polygons.drop(columns=["geometry"])
            csv_df.to_csv(csv_path, index=False)
            exported_files.append(csv_path)
            self.logger.info(f"Exported CSV: {csv_path}")

        # Export summary
        if export_config.get("summary", True):
            summary_path = f"{output_prefix}_summary.txt"
            self._write_summary(summary_path)
            exported_files.append(summary_path)
            self.logger.info(f"Exported summary: {summary_path}")

        # Export failed extractions
        if export_config.get("failed_extractions", True):
            failed_path = f"{output_prefix}_failed.csv"
            self._export_failed_extractions(failed_path)
            if Path(failed_path).exists():
                exported_files.append(failed_path)
                self.logger.info(f"Exported failed extractions: {failed_path}")

        # Export error log
        if export_config.get("error_log", True) and self.error_logs:
            error_path = f"{output_prefix}_errors.csv"
            error_df = pd.DataFrame(self.error_logs)
            error_df.to_csv(error_path, index=False)
            exported_files.append(error_path)
            self.logger.info(f"Exported error log: {error_path}")

        self.logger.info(f"Export complete: {len(exported_files)} files created")
        return exported_files

    def _write_summary(self, summary_path: str) -> None:
        """Write extraction summary to text file."""
        if self.truth_polygons is None:
            return

        with open(summary_path, "w") as f:
            f.write("FLOWFINDER Truth Extraction Summary\n")
            f.write("===================================\n\n")
            f.write(
                f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total Basins Processed: {len(self.basin_sample)}\n")
            f.write(f"Successful Extractions: {len(self.truth_polygons)}\n")
            f.write(f"Failed Extractions: {len(self.error_logs)}\n\n")

            # Extraction method statistics
            if "extraction_method" in self.truth_polygons.columns:
                f.write("Extraction Methods:\n")
                method_counts = self.truth_polygons["extraction_method"].value_counts()
                for method, count in method_counts.items():
                    f.write(f"  {method}: {count}\n")
                f.write("\n")

            # Area statistics
            if "truth_area_km2" in self.truth_polygons.columns:
                areas = self.truth_polygons["truth_area_km2"]
                f.write(f"Area Statistics:\n")
                f.write(f"  Min: {areas.min():.1f} km²\n")
                f.write(f"  Max: {areas.max():.1f} km²\n")
                f.write(f"  Mean: {areas.mean():.1f} km²\n")
                f.write(f"  Median: {areas.median():.1f} km²\n\n")

            # Terrain distribution
            if "sample_terrain" in self.truth_polygons.columns:
                f.write("Terrain Distribution:\n")
                terrain_counts = self.truth_polygons["sample_terrain"].value_counts()
                for terrain, count in terrain_counts.items():
                    f.write(f"  {terrain}: {count}\n")

    def _export_failed_extractions(self, failed_path: str) -> None:
        """Export list of failed extractions."""
        if not self.error_logs:
            return

        # Get unique failed basin IDs
        failed_basins = set()
        for error in self.error_logs:
            if error["error_type"] in ["no_catchment", "extraction_error"]:
                failed_basins.add(error["ID"])

        if failed_basins:
            # Find corresponding basin data
            failed_data = self.basin_sample[self.basin_sample["ID"].isin(failed_basins)]
            if not failed_data.empty:
                failed_data.to_csv(failed_path, index=False)

    def _log_error(self, basin_id: str, error_type: str, message: str) -> None:
        """Log structured error for later analysis."""
        error_record = {
            "basin_id": basin_id,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.error_logs.append(error_record)
        self.logger.warning(f"Basin {basin_id} - {error_type}: {message}")

    def run_complete_workflow(
        self, output_prefix: str = "truth_polygons"
    ) -> Dict[str, Any]:
        """
        Run the complete truth extraction workflow.

        Args:
            output_prefix: Prefix for output files

        Returns:
            Dictionary containing workflow results and exported files
        """
        self.logger.info("Starting complete truth extraction workflow...")

        try:
            # Load datasets
            self.load_datasets()

            # Extract truth polygons
            truth_polygons = self.extract_truth_polygons()

            # Validate quality
            validation_results = self.validate_truth_quality()

            # Export results
            exported_files = self.export_truth_dataset(output_prefix)

            workflow_results = {
                "success": True,
                "extracted_count": len(truth_polygons),
                "validation_results": validation_results,
                "exported_files": exported_files,
                "error_count": len(self.error_logs),
            }

            self.logger.info("Truth extraction workflow completed successfully")
            return workflow_results

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_count": len(self.error_logs),
            }


def create_sample_config() -> str:
    """Create a sample configuration file."""
    sample_config = {
        "data_dir": "data",
        "basin_sample_file": "basin_sample.csv",
        "buffer_tolerance": 500,
        "min_area_ratio": 0.1,
        "max_area_ratio": 10.0,
        "target_crs": "EPSG:5070",
        "output_crs": "EPSG:4326",
        "min_polygon_area": 1.0,
        "max_polygon_parts": 10,
        "max_attempts": 3,
        "retry_with_different_strategies": True,
        "chunk_size": 100,
        "memory_management": {
            "max_memory_mb": 2048,
            "large_file_threshold_mb": 50,
            "gc_frequency": 50,
            "monitor_memory": True,
            "fail_on_validation_errors": False,
        },
        "geometry_repair": {
            "enable_diagnostics": True,
            "enable_repair_attempts": True,
            "invalid_geometry_action": "remove",
            "max_repair_attempts": 3,
            "detailed_logging": True,
            "repair_strategies": {
                "buffer_fix": True,
                "simplify": True,
                "make_valid": True,
                "convex_hull": False,
                "orient_fix": True,
                "simplify_holes": True,
            },
        },
        "shapefile_schemas": {
            "basin_sample": {
                "required_columns": ["ID", "Pour_Point_Lat", "Pour_Point_Lon"],
                "column_types": {
                    "ID": "string",
                    "Pour_Point_Lat": "float",
                    "Pour_Point_Lon": "float",
                    "Area_km2": "float",
                    "Terrain_Class": "string",
                },
                "not_null_columns": ["ID", "Pour_Point_Lat", "Pour_Point_Lon"],
                "value_ranges": {
                    "Pour_Point_Lat": {"min": 25, "max": 55},
                    "Pour_Point_Lon": {"min": -125, "max": -100},
                    "Area_km2": {"min": 0.01, "max": 10000},
                },
            },
            "nhd_catchments": {
                "required_columns": ["FEATUREID"],
                "column_types": {
                    "FEATUREID": "integer",
                    "GRIDCODE": "integer",
                    "AREASQKM": "float",
                },
                "not_null_columns": ["FEATUREID"],
                "geometry_type": "Polygon",
                "value_ranges": {
                    "FEATUREID": {"min": 1},
                    "AREASQKM": {"min": 0.001, "max": 5000},
                },
                "consistency_rules": {"unique_values": {"columns": ["FEATUREID"]}},
            },
            "nhd_flowlines": {
                "required_columns": ["COMID"],
                "column_types": {"COMID": "integer", "LENGTHKM": "float"},
                "not_null_columns": ["COMID"],
                "geometry_type": "LineString",
                "consistency_rules": {"unique_values": {"columns": ["COMID"]}},
            },
        },
        "extraction_priority": {
            1: "contains_point",
            2: "largest_containing",
            3: "nearest_centroid",
            4: "largest_intersecting",
        },
        "files": {
            "nhd_catchments": "nhd_hr_catchments.shp",
            "nhd_flowlines": "nhd_flowlines.shp",
        },
        "quality_checks": {
            "topology_validation": True,
            "area_validation": True,
            "completeness_check": True,
            "drainage_check": False,
        },
        "terrain_extraction": {
            "alpine": {
                "max_parts": 15,
                "min_drainage_density": 0.01,
                "buffer_tolerance": 1000,
            },
            "desert": {
                "max_parts": 3,
                "min_drainage_density": 0.001,
                "buffer_tolerance": 200,
            },
        },
        "export": {
            "gpkg": True,
            "csv": True,
            "summary": True,
            "failed_extractions": True,
        },
    }

    config_content = yaml.dump(sample_config, default_flow_style=False, indent=2)
    return config_content


def main() -> None:
    """Main CLI entry point for truth extraction."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Truth Extractor - Extract ground truth polygons for accuracy benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python truth_extractor.py --output truth_polygons

  # Run with custom configuration
  python truth_extractor.py --config config.yaml --output custom_truth

  # Create sample configuration
  python truth_extractor.py --create-config > truth_extractor_config.yaml

  # Run with specific data directory
  python truth_extractor.py --data-dir /path/to/data --output mountain_west_truth
        """,
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--data-dir", "-d", type=str, help="Directory containing input datasets"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="truth_polygons",
        help="Output file prefix (default: truth_polygons)",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create sample configuration file and exit",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
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
        # Initialize extractor
        extractor = TruthExtractor(config_path=args.config, data_dir=args.data_dir)

        # Run complete workflow
        results = extractor.run_complete_workflow(args.output)

        if results["success"]:
            print(f"\n✅ Truth extraction completed successfully!")
            print(f"📊 Extracted {results['extracted_count']} truth polygons")
            print(f"📁 Exported {len(results['exported_files'])} files")
            if results["error_count"] > 0:
                print(f"⚠️  {results['error_count']} warnings logged")
        else:
            print(f"\n❌ Truth extraction failed: {results['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
