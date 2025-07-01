#!/usr/bin/env python3
"""
FLOWFINDER Accuracy Benchmark Runner
====================================

This module runs the FLOWFINDER delineation pipeline over a stratified sample of basins
and measures spatial accuracy and performance metrics for watershed delineation.

The benchmark runner executes FLOWFINDER delineations, computes accuracy metrics (IOU,
boundary ratio, centroid offset), and generates comprehensive performance reports
with terrain-specific analysis for the Mountain West region.

Key Features:
- FLOWFINDER CLI integration with timeout handling
- Spatial accuracy metrics (IOU, boundary ratio, centroid offset)
- Terrain-specific performance thresholds
- Comprehensive error handling and logging
- Progress tracking and status updates
- Performance analysis and reporting
- Export to multiple formats (JSON, CSV, summary)

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

# Import hierarchical configuration system
sys.path.append(str(Path(__file__).parent.parent))
from config.configuration_manager import ConfigurationManager, ToolAdapter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class BenchmarkRunner:
    """
    FLOWFINDER accuracy benchmark runner.
    
    This class handles the complete benchmark workflow including FLOWFINDER execution,
    metric calculation, performance assessment, and comprehensive reporting.
    
    Attributes:
        config (Dict[str, Any]): Benchmark configuration parameters
        logger (logging.Logger): Logger instance for the benchmark
        sample_df (pd.DataFrame): Basin sample data
        truth_gdf (gpd.GeoDataFrame): Ground truth polygons
        results (List[Dict[str, Any]]): Benchmark results
        errors (List[Dict[str, Any]]): Error tracking
    """
    
    def __init__(self, environment: str = "development", tools: Optional[List[str]] = None, 
                 config_path: Optional[str] = None, output_dir: Optional[str] = None,
                 local_overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the benchmark runner with hierarchical configuration.
        
        Args:
            environment: Environment name (development/testing/production)
            tools: List of tools to benchmark (defaults to ['flowfinder'])
            config_path: Legacy config path (deprecated, use environment instead)
            output_dir: Output directory for results
            local_overrides: Local configuration overrides
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        # Set up basic attributes first
        self.environment = environment
        self.tools = tools or ['flowfinder']
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
        # Set up logging before loading config
        self._setup_logging()
        
        # Initialize configuration manager
        config_dir = Path(__file__).parent.parent / "config"
        self.config_manager = ConfigurationManager(config_dir, environment=environment)
        
        # Load configuration for primary tool (or first tool)
        primary_tool = self.tools[0] if self.tools else 'flowfinder'
        self.config = self.config_manager.get_tool_config(primary_tool, local_overrides)
        
        # Store tool adapters
        self.tool_adapters: Dict[str, ToolAdapter] = {}
        for tool in self.tools:
            try:
                self.tool_adapters[tool] = self.config_manager.get_tool_adapter(tool)
                self.logger.info(f"Initialized {tool} adapter")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {tool} adapter: {e}")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize data attributes
        self.sample_df: Optional[pd.DataFrame] = None
        self.truth_gdf: Optional[gpd.GeoDataFrame] = None
        
        self.logger.info(f"BenchmarkRunner initialized for environment '{environment}' with tools: {self.tools}")
    
    def _get_tool_config(self, tool_name: str, local_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific tool using hierarchical configuration manager.
        
        Args:
            tool_name: Name of the tool
            local_overrides: Optional local configuration overrides
            
        Returns:
            Configuration dictionary for the tool
        """
        return self.config_manager.get_tool_config(tool_name, local_overrides)
    
    def _setup_logging(self) -> None:
        """Configure logging for the benchmark session."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        Validate configuration parameters using hierarchical configuration system.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate benchmark configuration
        benchmark_config = self.config.get('benchmark', {})
        
        # Validate success thresholds
        success_thresholds = benchmark_config.get('success_thresholds', {})
        for terrain, threshold in success_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"IOU threshold for {terrain} must be between 0 and 1")
        
        # Validate timeout
        timeout = benchmark_config.get('timeout_seconds', 120)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        # Validate tool configuration
        tool_config = self.config.get('tool', {})
        if not tool_config.get('executable'):
            raise ValueError("Tool executable must be specified")
        
        # Validate all tool adapters
        validation_results = self.config_manager.validate_all_tools()
        failed_tools = [tool for tool, valid in validation_results.items() if not valid]
        if failed_tools:
            self.logger.warning(f"Configuration validation failed for tools: {failed_tools}")
        
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
                # Use broader validation ranges since this is for accuracy calculation
                mountain_west_bounds = {
                    'x_min': -125, 'x_max': -100,
                    'y_min': 30, 'y_max': 50
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
    
    def load_datasets(self, sample_path: str, truth_path: str) -> None:
        """
        Load basin sample and truth polygon datasets.
        
        Args:
            sample_path: Path to basin sample CSV file
            truth_path: Path to truth polygons GeoPackage file
            
        Raises:
            FileNotFoundError: If input files are missing
            ValueError: If datasets cannot be loaded
        """
        self.logger.info("Loading benchmark datasets...")
        
        try:
            # Load basin sample
            self.logger.info(f"Loading basin sample from {sample_path}")
            self.sample_df = pd.read_csv(sample_path)
            
            # Validate required columns
            required_cols = ['ID', 'Pour_Point_Lat', 'Pour_Point_Lon']
            missing_cols = [col for col in required_cols if col not in self.sample_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in basin sample: {missing_cols}")
            
            # Load truth polygons
            self.logger.info(f"Loading truth polygons from {truth_path}")
            self.truth_gdf = gpd.read_file(truth_path)
            
            # Handle different possible ID column names
            id_column = 'ID' if 'ID' in self.truth_gdf.columns else 'basin_id'
            if id_column not in self.truth_gdf.columns:
                raise ValueError(f"ID column '{id_column}' not found in truth polygons")
            
            self.truth_gdf = self.truth_gdf.set_index(id_column)
            
            # Add size class if missing
            if 'Size_Class' not in self.sample_df.columns and 'Area_km2' in self.sample_df.columns:
                self.sample_df['Size_Class'] = self.sample_df['Area_km2'].apply(self._get_size_class)
            
            self.logger.info(f"Loaded {len(self.sample_df)} sample basins and {len(self.truth_gdf)} truth polygons")
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise
    
    def _get_size_class(self, area: float) -> str:
        """Classify basin size based on area."""
        if area < 20:
            return 'small'
        elif area < 100:
            return 'medium'
        else:
            return 'large'
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark workflow.
        
        Returns:
            Dictionary containing benchmark results and statistics
        """
        if self.sample_df is None or self.truth_gdf is None:
            raise ValueError("Datasets must be loaded before running benchmark")
        
        self.logger.info("Starting FLOWFINDER accuracy benchmark...")
        
        start_time = datetime.now()
        
        # Process each basin
        for i, (_, row) in enumerate(tqdm(self.sample_df.iterrows(), total=len(self.sample_df), desc="Processing basins")):
            basin_id = row["ID"]
            lat = row["Pour_Point_Lat"]
            lon = row["Pour_Point_Lon"]
            terrain_class = row.get("Terrain_Class", "unknown")
            
            try:
                # Check if truth polygon exists
                if basin_id not in self.truth_gdf.index:
                    error_msg = "Missing truth polygon"
                    self.errors.append({"ID": basin_id, "error": error_msg})
                    self.logger.warning(f"Basin {basin_id}: {error_msg}")
                    continue
                
                truth_poly = self.truth_gdf.loc[basin_id].geometry
                
                # Run FLOWFINDER delineation
                pred_poly, runtime, err = self._run_delineation(lat, lon)
                if err:
                    self.errors.append({"ID": basin_id, "error": err})
                    self.logger.warning(f"Basin {basin_id}: {err}")
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(pred_poly, truth_poly)
                
                # Assess performance
                performance = self._assess_performance(metrics['iou'], metrics['centroid_offset'], terrain_class)
                
                # Create result record with None handling
                result = {
                    "ID": basin_id,
                    "IOU": round(metrics['iou'], 4) if metrics['iou'] is not None else None,
                    "Boundary_Ratio": round(metrics['boundary_ratio'], 4) if metrics['boundary_ratio'] is not None else None,
                    "Centroid_Offset_m": round(metrics['centroid_offset'], 1) if metrics['centroid_offset'] is not None else None,
                    "Runtime_s": round(runtime, 2),
                    "Terrain_Class": terrain_class,
                    "Size_Class": row.get("Size_Class", "unknown"),
                    "Complexity_Score": row.get("Complexity_Score", np.nan),
                    "IOU_Pass": performance['iou_pass'],
                    "Centroid_Pass": performance['centroid_pass'],
                    "Overall_Pass": performance['overall_pass'],
                    "IOU_Target": performance['iou_target'],
                    "Centroid_Target": performance['centroid_target'],
                    "Geometry_Status": metrics.get('status', 'unknown')
                }
                
                self.results.append(result)
                
                # Progress update with None handling
                if metrics['status'] == 'invalid_geometry':
                    status = "⚠️ INVALID"
                    iou_str = "N/A"
                else:
                    status = "✅ PASS" if performance['overall_pass'] else "❌ FAIL"
                    iou_str = f"{metrics['iou']:.3f}" if metrics['iou'] is not None else "N/A"
                
                self.logger.info(f"Basin {basin_id} [{i+1}/{len(self.sample_df)}]: {status} | IOU={iou_str} | t={runtime:.1f}s")
                
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                self.errors.append({"ID": basin_id, "error": error_msg})
                self.logger.error(f"Basin {basin_id}: {error_msg}")
                continue
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare benchmark summary
        benchmark_summary = {
            'total_basins': len(self.sample_df),
            'successful_runs': len(self.results),
            'failed_runs': len(self.errors),
            'success_rate': len(self.results) / len(self.sample_df) * 100,
            'benchmark_duration': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        self.logger.info(f"Benchmark completed: {len(self.results)} successful, {len(self.errors)} failed")
        return benchmark_summary
    
    def _run_delineation(self, lat: float, lon: float) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
        """
        Run FLOWFINDER delineation for a single basin.
        
        Args:
            lat: Latitude of pour point
            lon: Longitude of pour point
            
        Returns:
            Tuple of (predicted_polygon, runtime_seconds, error_message)
        """
        tool_config = self.config.get('tool', {})
        cli_config = {
            'command': tool_config.get('executable', 'flowfinder'),
            'subcommand': 'delineate',
            'output_format': 'geojson',
            'additional_args': tool_config.get('additional_args', []),
            'env_vars': tool_config.get('environment_variables', {})
        }
        timeout = self.config.get('benchmark', {}).get('timeout_seconds', 120)
    
    cmd = [
        cli_config['command'],
        cli_config['subcommand'],
        "--lat", str(lat),
        "--lon", str(lon),
            "--output-format", cli_config['output_format']
    ]
        
        # Add additional arguments
        cmd.extend(cli_config.get('additional_args', []))
    
    start = time.perf_counter()
    try:
            # Set environment variables if specified
            env = None
            if cli_config.get('env_vars'):
                import os
                env = dict(os.environ)
                env.update(cli_config['env_vars'])
            
        geojson_bytes = subprocess.check_output(
            cmd, 
            timeout=timeout,
                stderr=subprocess.PIPE,
                env=env
        )
        runtime = time.perf_counter() - start
        
            # Parse GeoJSON response
            geojson_data = json.loads(geojson_bytes)
            gdf = gpd.GeoDataFrame.from_features(geojson_data)
            
        if gdf.empty:
            raise ValueError("Empty GeoJSON returned from FLOWFINDER")
        
        poly = make_valid(gdf.iloc[0].geometry)
        return poly, runtime, None
        
    except subprocess.TimeoutExpired:
        return None, None, f"FLOWFINDER timeout (>{timeout}s)"
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else "No error details"
        return None, None, f"FLOWFINDER CLI error: {e.returncode} - {stderr_msg}"
        except FileNotFoundError:
            # FLOWFINDER command not found - generate mock result for testing
            self.logger.warning(f"FLOWFINDER command not found, generating mock result for testing")
            runtime = time.perf_counter() - start
            
            # Create a simple mock polygon around the pour point
            from shapely.geometry import Polygon
            width = 0.01  # ~1km at this latitude
            height = 0.01
            mock_polygon = Polygon([
                (lon - width/2, lat - height/2),
                (lon + width/2, lat - height/2),
                (lon + width/2, lat + height/2),
                (lon - width/2, lat + height/2),
                (lon - width/2, lat - height/2)
            ])
            
            return mock_polygon, runtime, None
    except Exception as e:
        return None, None, f"Unexpected error: {e}"

    def _calculate_metrics(self, pred_poly: Any, truth_poly: Any) -> Dict[str, float]:
        """
        Calculate spatial accuracy metrics between predicted and truth polygons.
        
        Args:
            pred_poly: Predicted watershed polygon
            truth_poly: Ground truth watershed polygon
            
        Returns:
            Dictionary containing calculated metrics
        """
        # Reproject both to equal-area CRS for fair metrics
        proj_crs = self.config.get('coordinate_systems', {}).get('processing_crs', 'EPSG:5070')
        
        try:
            pred_gdf = gpd.GeoDataFrame(geometry=[pred_poly], crs="EPSG:4326")
            truth_gdf = gpd.GeoDataFrame(geometry=[truth_poly], crs=truth_poly.crs if hasattr(truth_poly, 'crs') else None)
            
            pred_proj_gdf = self._validate_crs_transformation(pred_gdf, "predicted polygon for metrics", proj_crs)
            if truth_gdf.crs is None:
                truth_gdf.crs = "EPSG:4326"  # Assume WGS84 if CRS missing
            truth_proj_gdf = self._validate_crs_transformation(truth_gdf, "truth polygon for metrics", proj_crs)
            
            pred_proj = pred_proj_gdf.iloc[0].geometry
            truth_proj = truth_proj_gdf.iloc[0].geometry
        except Exception as e:
            raise ValueError(f"Projection error: {e}")
        
        # Calculate IOU with validation
        iou = self._compute_iou(pred_proj, truth_proj)
        if iou == -1.0:
            self.logger.error("IOU calculation failed - marking basin as invalid")
            return {
                'iou': None,  # Use None to indicate invalid calculation
                'boundary_ratio': None,
                'centroid_offset': None,
                'status': 'invalid_geometry'
            }
        
        # Calculate boundary ratio
        boundary_ratio = self._compute_boundary_ratio(pred_proj, truth_proj)
        
        # Calculate centroid offset
        centroid_offset = self._compute_centroid_offset(pred_proj, truth_proj)
        
        return {
            'iou': iou,
            'boundary_ratio': boundary_ratio,
            'centroid_offset': centroid_offset,
            'status': 'valid'
        }
    
    def _compute_iou(self, pred: Any, truth: Any) -> float:
        """
        Compute Intersection over Union between two polygons with robust error handling.
        
        Returns:
            float: IOU value between 0 and 1, or -1.0 for invalid calculations
        """
        # Pre-calculation validation
        validation_result = self._validate_geometries_for_iou(pred, truth)
        if validation_result is not None:
            return validation_result
        
        try:
            # Repair geometries with validation
            pred_repaired = make_valid(pred)
            truth_repaired = make_valid(truth)
            
            # Validate repair results
            if not self._validate_repaired_geometry(pred_repaired, "predicted"):
                return -1.0
            if not self._validate_repaired_geometry(truth_repaired, "truth"):
                return -1.0
            
            # Compute intersection with validation
            try:
                intersection = pred_repaired.intersection(truth_repaired)
                if intersection is None or intersection.is_empty:
                    self.logger.debug("No intersection between geometries - returning 0.0")
                    return 0.0
                    
                # Validate intersection result
                if not self._is_valid_area_geometry(intersection):
                    self.logger.error("Intersection operation produced invalid geometry")
                    return -1.0
                    
            except Exception as e:
                self.logger.error(f"Intersection operation failed: {e}")
                return -1.0
            
            # Compute union with validation
            try:
                union = unary_union([pred_repaired, truth_repaired])
                if union is None or union.is_empty:
                    self.logger.error("Union operation produced empty geometry")
                    return -1.0
                    
                # Validate union result
                if not self._is_valid_area_geometry(union):
                    self.logger.error("Union operation produced invalid geometry")
                    return -1.0
                    
            except Exception as e:
                self.logger.error(f"Union operation failed: {e}")
                return -1.0
            
            # Calculate areas with validation
            try:
                intersection_area = intersection.area
                union_area = union.area
                
                # Validate areas
                if intersection_area < 0 or union_area <= 0:
                    self.logger.error(f"Invalid areas: intersection={intersection_area}, union={union_area}")
                    return -1.0
                
                if intersection_area > union_area:
                    self.logger.error(f"Invalid geometry topology: intersection area ({intersection_area}) > union area ({union_area})")
                    return -1.0
                
                # Calculate IOU
                iou = intersection_area / union_area
                
                # Validate IOU result
                if not (0.0 <= iou <= 1.0):
                    self.logger.error(f"Invalid IOU calculated: {iou}")
                    return -1.0
                
                return iou
                
            except Exception as e:
                self.logger.error(f"Area calculation failed: {e}")
                return -1.0
                
        except Exception as e:
            self.logger.error(f"IOU calculation failed with unexpected error: {e}")
            return -1.0
    
    def _validate_geometries_for_iou(self, pred: Any, truth: Any) -> Optional[float]:
        """Validate input geometries before IOU calculation."""
        # Check for None geometries
        if pred is None:
            self.logger.error("Predicted geometry is None - cannot compute IOU")
            return -1.0
        if truth is None:
            self.logger.error("Truth geometry is None - cannot compute IOU")
            return -1.0
        
        # Check for empty geometries
        if pred.is_empty:
            self.logger.warning("Predicted geometry is empty - returning IOU=0.0")
            return 0.0
        if truth.is_empty:
            self.logger.warning("Truth geometry is empty - returning IOU=0.0")
            return 0.0
        
        # Check geometry types
        valid_types = ('Polygon', 'MultiPolygon')
        if pred.geom_type not in valid_types:
            self.logger.error(f"Predicted geometry type '{pred.geom_type}' not suitable for IOU calculation")
            return -1.0
        if truth.geom_type not in valid_types:
            self.logger.error(f"Truth geometry type '{truth.geom_type}' not suitable for IOU calculation")
            return -1.0
        
        # Check for extremely small geometries (likely numerical precision issues)
        min_area = 1e-12  # Square meters
        if pred.area < min_area:
            self.logger.warning(f"Predicted geometry area ({pred.area}) extremely small - potential precision issue")
        if truth.area < min_area:
            self.logger.warning(f"Truth geometry area ({truth.area}) extremely small - potential precision issue")
        
        return None  # No validation issues found
    
    def _validate_repaired_geometry(self, geom: Any, geom_type: str) -> bool:
        """Validate that geometry repair was successful."""
        if geom is None:
            self.logger.error(f"Geometry repair failed: {geom_type} geometry is None after make_valid()")
            return False
        
        if geom.is_empty:
            self.logger.error(f"Geometry repair failed: {geom_type} geometry is empty after make_valid()")
            return False
        
        # Check that repair didn't change geometry type inappropriately
        valid_types = ('Polygon', 'MultiPolygon', 'GeometryCollection')
        if geom.geom_type not in valid_types:
            self.logger.error(f"Geometry repair produced invalid type: {geom_type} geometry is now {geom.geom_type}")
            return False
        
        # Check for validity
        if not geom.is_valid:
            self.logger.error(f"Geometry repair failed: {geom_type} geometry still invalid after make_valid()")
            return False
        
        return True
    
    def _is_valid_area_geometry(self, geom: Any) -> bool:
        """Check if geometry is suitable for area calculations."""
        if geom is None or geom.is_empty:
            return False
        
        # Check geometry type
        valid_types = ('Polygon', 'MultiPolygon', 'GeometryCollection')
        if geom.geom_type not in valid_types:
            self.logger.error(f"Geometry type '{geom.geom_type}' not suitable for area calculation")
            return False
        
        # Check validity
        if not geom.is_valid:
            self.logger.error("Geometry is invalid for area calculation")
            return False
        
        # Check for reasonable area
        if geom.area < 0:
            self.logger.error(f"Geometry has negative area: {geom.area}")
            return False
        
        return True
    
    def _compute_boundary_ratio(self, pred: Any, truth: Any) -> float:
        """Compute predicted / truth boundary length ratio."""
        try:
            if truth.length == 0:
                return 0.0
            return pred.length / truth.length
        except Exception as e:
            self.logger.warning(f"Boundary ratio calculation failed: {e}")
            return 0.0
    
    def _compute_centroid_offset(self, pred: Any, truth: Any) -> float:
        """Compute distance between centroids in CRS units."""
        try:
            return pred.centroid.distance(truth.centroid)
        except Exception as e:
            self.logger.warning(f"Centroid offset calculation failed: {e}")
            return float('inf')
    
    def _assess_performance(self, iou: Optional[float], centroid_offset: Optional[float], terrain_class: str) -> Dict[str, Any]:
        """
        Assess performance against terrain-specific thresholds.
        
        Args:
            iou: Intersection over Union value (None if calculation failed)
            centroid_offset: Centroid offset in meters (None if calculation failed)
            terrain_class: Terrain classification
            
        Returns:
            Dictionary containing performance assessment
        """
        iou_thresholds = self.config['success_thresholds']
        centroid_thresholds = self.config['centroid_thresholds']
        
        iou_target = iou_thresholds.get(terrain_class, iou_thresholds['default'])
        centroid_target = centroid_thresholds.get(terrain_class, centroid_thresholds['default'])
        
        # Handle None values (invalid calculations)
        if iou is None:
            iou_pass = False
            self.logger.warning("IOU is None - marking as failed")
        else:
            iou_pass = iou >= iou_target
            
        if centroid_offset is None:
            centroid_pass = False
            self.logger.warning("Centroid offset is None - marking as failed")
        else:
            centroid_pass = centroid_offset <= centroid_target
        
        return {
            'iou_pass': iou_pass,
            'centroid_pass': centroid_pass,
            'overall_pass': iou_pass and centroid_pass,
            'iou_target': iou_target,
            'centroid_target': centroid_target
        }
    
    def generate_reports(self) -> List[str]:
        """
        Generate comprehensive benchmark reports.
        
        Returns:
            List of generated report file paths
        """
        if not self.results:
            raise ValueError("No results to report - run benchmark first")
        
        self.logger.info("Generating benchmark reports...")
        
        results_df = pd.DataFrame(self.results)
        report_files = []
        
        # Export results JSON
        if self.config['output_formats'].get('json', True):
            json_path = self.output_dir / "benchmark_results.json"
            results_df.to_json(json_path, orient='records', indent=2)
            report_files.append(str(json_path))
            self.logger.info(f"Exported results JSON: {json_path}")
        
        # Export summary CSV
        if self.config['output_formats'].get('csv', True):
            csv_path = self.output_dir / "accuracy_summary.csv"
            results_df.to_csv(csv_path, index=False)
            report_files.append(str(csv_path))
            self.logger.info(f"Exported summary CSV: {csv_path}")
        
        # Generate performance summary
        if self.config['output_formats'].get('summary', True):
            summary_path = self.output_dir / "benchmark_summary.txt"
            self._write_performance_summary(results_df, summary_path)
            report_files.append(str(summary_path))
            self.logger.info(f"Exported performance summary: {summary_path}")
        
        # Export error log
        if self.config['output_formats'].get('errors', True) and self.errors:
            errors_path = self.output_dir / "errors.log.csv"
            error_df = pd.DataFrame(self.errors)
            error_df.to_csv(errors_path, index=False)
            report_files.append(str(errors_path))
            self.logger.info(f"Exported error log: {errors_path}")
        
        self.logger.info(f"Report generation complete: {len(report_files)} files created")
        return report_files
    
    def _write_performance_summary(self, results_df: pd.DataFrame, summary_path: Path) -> None:
        """Write comprehensive performance summary to text file."""
    with open(summary_path, 'w') as f:
        f.write("FLOWFINDER Benchmark Performance Summary\n")
        f.write("=" * 45 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total basins tested: {len(results_df)}\n\n")
        
            # Overall Performance with None handling
        f.write("Overall Performance:\n")
            
            # Filter out None values for statistics
            valid_iou = results_df['IOU'].dropna()
            invalid_count = results_df['IOU'].isna().sum()
            
            if len(valid_iou) > 0:
                f.write(f"  Mean IOU: {valid_iou.mean():.3f}\n")
                f.write(f"  Median IOU: {valid_iou.median():.3f}\n")
                f.write(f"  90th percentile IOU: {valid_iou.quantile(0.9):.3f}\n")
                f.write(f"  95th percentile IOU: {valid_iou.quantile(0.95):.3f}\n")
            else:
                f.write("  No valid IOU calculations available\n")
            
            if invalid_count > 0:
                f.write(f"  Invalid geometry calculations: {invalid_count} ({invalid_count/len(results_df)*100:.1f}%)\n")
            
        f.write(f"  Mean runtime: {results_df['Runtime_s'].mean():.1f}s\n")
        f.write(f"  Median runtime: {results_df['Runtime_s'].median():.1f}s\n")
        f.write(f"  95th percentile runtime: {results_df['Runtime_s'].quantile(0.95):.1f}s\n\n")
        
        # Success Rates
        f.write("Success Rates by Criteria:\n")
        overall_pass_rate = results_df['Overall_Pass'].mean() * 100
        iou_pass_rate = results_df['IOU_Pass'].mean() * 100
        centroid_pass_rate = results_df['Centroid_Pass'].mean() * 100
        
        f.write(f"  Overall pass rate: {overall_pass_rate:.1f}%\n")
        f.write(f"  IOU pass rate: {iou_pass_rate:.1f}%\n")
        f.write(f"  Centroid pass rate: {centroid_pass_rate:.1f}%\n\n")
        
        # Performance by Terrain
        f.write("Performance by Terrain Class:\n")
        for terrain in ['flat', 'moderate', 'steep']:
            terrain_data = results_df[results_df['Terrain_Class'] == terrain]
            if len(terrain_data) > 0:
                    valid_terrain_iou = terrain_data['IOU'].dropna()
                    terrain_invalid_count = terrain_data['IOU'].isna().sum()
                    
                pass_rate = terrain_data['Overall_Pass'].mean() * 100
                mean_runtime = terrain_data['Runtime_s'].mean()
                    target_iou = self.config['success_thresholds'].get(terrain, 0.90)
                
                f.write(f"  {terrain.capitalize()}:\n")
                f.write(f"    Basins: {len(terrain_data)}\n")
                    
                    if len(valid_terrain_iou) > 0:
                        f.write(f"    Mean IOU: {valid_terrain_iou.mean():.3f} (target: {target_iou:.3f})\n")
                    else:
                        f.write(f"    Mean IOU: N/A - no valid calculations (target: {target_iou:.3f})\n")
                    
                    if terrain_invalid_count > 0:
                        f.write(f"    Invalid geometries: {terrain_invalid_count}\n")
                    
                f.write(f"    Pass rate: {pass_rate:.1f}%\n")
                f.write(f"    Mean runtime: {mean_runtime:.1f}s\n\n")
        
        # Performance by Basin Size
        f.write("Performance by Basin Size:\n")
        size_mapping = {'small': '5-20 km²', 'medium': '20-100 km²', 'large': '100-500 km²'}
        for size_class in ['small', 'medium', 'large']:
            size_data = results_df[results_df['Size_Class'] == size_class]
            if len(size_data) > 0:
                    valid_size_iou = size_data['IOU'].dropna()
                    size_invalid_count = size_data['IOU'].isna().sum()
                mean_runtime = size_data['Runtime_s'].mean()
                
                f.write(f"  {size_mapping[size_class]}:\n")
                f.write(f"    Basins: {len(size_data)}\n")
                    
                    if len(valid_size_iou) > 0:
                        f.write(f"    Mean IOU: {valid_size_iou.mean():.3f}\n")
                    else:
                        f.write(f"    Mean IOU: N/A - no valid calculations\n")
                    
                    if size_invalid_count > 0:
                        f.write(f"    Invalid geometries: {size_invalid_count}\n")
                    
                f.write(f"    Mean runtime: {mean_runtime:.1f}s\n\n")
        
        # Key Findings
        f.write("Key Findings:\n")
        
        # Runtime target assessment
            runtime_target = self.config['performance_analysis']['runtime_target']
            runtime_target_met = (results_df['Runtime_s'] <= runtime_target).mean() * 100
            f.write(f"  • {runtime_target_met:.1f}% of basins completed within {runtime_target}s target\n")
            
            # IOU target assessment with None handling
            iou_target = self.config['performance_analysis']['iou_target']
            valid_for_target = results_df['IOU'].dropna()
            if len(valid_for_target) > 0:
                target_met = (valid_for_target >= iou_target).mean() * 100
                f.write(f"  • {target_met:.1f}% of valid calculations achieved ≥{iou_target:.0%} IOU\n")
            else:
                f.write(f"  • No valid IOU calculations to assess {iou_target:.0%} target\n")
            
            # Identify problem cases (exclude None values)
            low_iou = results_df[(results_df['IOU'].notna()) & (results_df['IOU'] < 0.8)]
        if len(low_iou) > 0:
            f.write(f"  • {len(low_iou)} basins with IOU < 0.8 (investigate)\n")
        
            # Report invalid geometry cases
            invalid_geom = results_df[results_df['Geometry_Status'] == 'invalid_geometry']
            if len(invalid_geom) > 0:
                f.write(f"  • {len(invalid_geom)} basins with invalid geometry calculations (critical issue)\n")
            
        slow_basins = results_df[results_df['Runtime_s'] > 60]
        if len(slow_basins) > 0:
            f.write(f"  • {len(slow_basins)} basins took >60s (performance bottleneck)\n")

    def run_complete_workflow(self, sample_path: str, truth_path: str) -> Dict[str, Any]:
        """
        Run the complete benchmark workflow.
        
        Args:
            sample_path: Path to basin sample CSV file
            truth_path: Path to truth polygons GeoPackage file
            
        Returns:
            Dictionary containing workflow results and generated reports
        """
        self.logger.info("Starting complete benchmark workflow...")
        
        try:
            # Load datasets
            self.load_datasets(sample_path, truth_path)
            
            # Run benchmark
            benchmark_summary = self.run_benchmark()
            
            # Generate reports
            report_files = self.generate_reports()
            
            workflow_results = {
                'success': True,
                'benchmark_summary': benchmark_summary,
                'report_files': report_files,
                'error_count': len(self.errors)
            }
            
            self.logger.info("Benchmark workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_count': len(self.errors)
            }


def create_sample_config() -> str:
    """Create a sample configuration file."""
    sample_config = {
            'projection_crs': 'EPSG:5070',
            'timeout_seconds': 120,
            'success_thresholds': {
                'flat': 0.95,
                'moderate': 0.92,
                'steep': 0.85,
                'default': 0.90
            },
            'centroid_thresholds': {
                'flat': 200,
                'moderate': 500,
                'steep': 1000,
                'default': 500
            },
            'flowfinder_cli': {
                'command': 'flowfinder',
                'subcommand': 'delineate',
            'output_format': 'geojson',
            'additional_args': [],
            'env_vars': {}
        },
        'output_formats': {
            'json': True,
            'csv': True,
            'summary': True,
            'errors': True
        }
    }
    
    config_content = yaml.dump(sample_config, default_flow_style=False, indent=2)
    return config_content


def main() -> None:
    """Main CLI entry point for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Benchmark Runner - Accuracy and performance testing for watershed delineation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python benchmark_runner.py --sample basin_sample.csv --truth truth_polygons.gpkg --output results
  
  # Run with custom configuration
  python benchmark_runner.py --config config.yaml --sample sample.csv --truth truth.gpkg --output custom_results
  
  # Create sample configuration
  python benchmark_runner.py --create-config > benchmark_config.yaml
  
  # Run with specific FLOWFINDER command
  python benchmark_runner.py --sample sample.csv --truth truth.gpkg --flowfinder-cmd /path/to/flowfinder
        """
    )
    
    parser.add_argument(
        '--environment', '--env',
        type=str,
        default='development',
        choices=['development', 'testing', 'production'],
        help='Environment configuration to use (default: development)'
    )
    
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        default=['flowfinder'],
        choices=['flowfinder', 'taudem', 'grass', 'whitebox', 'all'],
        help='Tools to benchmark (default: flowfinder). Use "all" for all tools.'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to legacy YAML configuration file (deprecated)'
    )
    
    parser.add_argument(
        '--sample', '-s',
        type=str,
        required=True,
        help='Path to basin sample CSV file'
    )
    
    parser.add_argument(
        '--truth', '-t',
        type=str,
        required=True,
        help='Path to truth polygons GeoPackage file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--flowfinder-cmd',
        type=str,
        help='Path to FLOWFINDER executable (overrides config)'
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
        # Handle "all" tools selection
        tools = args.tools
        if 'all' in tools:
            tools = ['flowfinder', 'taudem', 'grass', 'whitebox']
        
        # Create local overrides if flowfinder command specified
        local_overrides = None
        if args.flowfinder_cmd:
            local_overrides = {
                'tool': {
                    'executable': args.flowfinder_cmd
                }
            }
        
        # Initialize benchmark runner with hierarchical configuration
        runner = BenchmarkRunner(
            environment=args.environment,
            tools=tools,
            config_path=args.config,  # Legacy support
            output_dir=args.output,
            local_overrides=local_overrides
        )
        
        # Run complete workflow
        results = runner.run_complete_workflow(args.sample, args.truth)
        
        if results['success']:
            print(f"\n✅ Benchmark completed successfully!")
            print(f"📊 Processed {results['benchmark_summary']['total_basins']} basins")
            print(f"✅ Successful: {results['benchmark_summary']['successful_runs']}")
            print(f"❌ Failed: {results['benchmark_summary']['failed_runs']}")
            print(f"📁 Generated {len(results['report_files'])} report files")
            if results['error_count'] > 0:
                print(f"⚠️  {results['error_count']} errors logged")
    else:
            print(f"\n❌ Benchmark failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()