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
    
    def __init__(self, config_path: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        """
        Initialize the benchmark runner with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Output directory for results
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self._setup_logging()
        self._validate_config()
        
        # Initialize data attributes
        self.sample_df: Optional[pd.DataFrame] = None
        self.truth_gdf: Optional[gpd.GeoDataFrame] = None
        
        self.logger.info("BenchmarkRunner initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load benchmark configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'projection_crs': 'EPSG:5070',  # NAD83 / CONUS Albers Equal Area
            'timeout_seconds': 120,
            'success_thresholds': {
                'flat': 0.95,
                'moderate': 0.92,
                'steep': 0.85,
                'default': 0.90
            },
            'centroid_thresholds': {
                'flat': 200,      # meters
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
            'metrics': {
                'iou': True,
                'boundary_ratio': True,
                'centroid_offset': True,
                'runtime': True
            },
            'output_formats': {
                'json': True,
                'csv': True,
                'summary': True,
                'errors': True
            },
            'performance_analysis': {
                'runtime_target': 30,  # seconds
                'iou_target': 0.95,
                'generate_plots': False,
                'detailed_analysis': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if 'benchmark' in user_config:
                        default_config.update(user_config['benchmark'])
                    else:
                        default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
        else:
            self.logger.info("Using default configuration")
            
        return default_config
    
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
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate thresholds
        for terrain, threshold in self.config['success_thresholds'].items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"IOU threshold for {terrain} must be between 0 and 1")
        
        for terrain, threshold in self.config['centroid_thresholds'].items():
            if threshold <= 0:
                raise ValueError(f"Centroid threshold for {terrain} must be positive")
        
        # Validate timeout
        if self.config['timeout_seconds'] <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        # Validate FLOWFINDER CLI configuration
        cli_config = self.config['flowfinder_cli']
        if not cli_config.get('command'):
            raise ValueError("FLOWFINDER command must be specified")
        
        self.logger.info("Configuration validation passed")
    
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
                
                # Create result record
                result = {
                    "ID": basin_id,
                    "IOU": round(metrics['iou'], 4),
                    "Boundary_Ratio": round(metrics['boundary_ratio'], 4),
                    "Centroid_Offset_m": round(metrics['centroid_offset'], 1),
                    "Runtime_s": round(runtime, 2),
                    "Terrain_Class": terrain_class,
                    "Size_Class": row.get("Size_Class", "unknown"),
                    "Complexity_Score": row.get("Complexity_Score", np.nan),
                    "IOU_Pass": performance['iou_pass'],
                    "Centroid_Pass": performance['centroid_pass'],
                    "Overall_Pass": performance['overall_pass'],
                    "IOU_Target": performance['iou_target'],
                    "Centroid_Target": performance['centroid_target']
                }
                
                self.results.append(result)
                
                # Progress update
                status = "✅ PASS" if performance['overall_pass'] else "❌ FAIL"
                self.logger.info(f"Basin {basin_id} [{i+1}/{len(self.sample_df)}]: {status} | IOU={metrics['iou']:.3f} | t={runtime:.1f}s")
                
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
        cli_config = self.config['flowfinder_cli']
        timeout = self.config['timeout_seconds']
        
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
        proj_crs = self.config['projection_crs']
        
        try:
            pred_proj = gpd.GeoSeries([pred_poly], crs="EPSG:4326").to_crs(proj_crs).iloc[0]
            truth_proj = gpd.GeoSeries([truth_poly]).to_crs(proj_crs).iloc[0]
        except Exception as e:
            raise ValueError(f"Projection error: {e}")
        
        # Calculate IOU
        iou = self._compute_iou(pred_proj, truth_proj)
        
        # Calculate boundary ratio
        boundary_ratio = self._compute_boundary_ratio(pred_proj, truth_proj)
        
        # Calculate centroid offset
        centroid_offset = self._compute_centroid_offset(pred_proj, truth_proj)
        
        return {
            'iou': iou,
            'boundary_ratio': boundary_ratio,
            'centroid_offset': centroid_offset
        }
    
    def _compute_iou(self, pred: Any, truth: Any) -> float:
        """Compute Intersection over Union between two polygons."""
        try:
            pred = make_valid(pred)
            truth = make_valid(truth)
            
            if pred.is_empty or truth.is_empty:
                return 0.0
                
            inter = pred.intersection(truth)
            union = unary_union([pred, truth])
            
            if union.area == 0:
                return 0.0
                
            return inter.area / union.area
        except Exception as e:
            self.logger.warning(f"IOU calculation failed: {e}")
            return 0.0
    
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
    
    def _assess_performance(self, iou: float, centroid_offset: float, terrain_class: str) -> Dict[str, Any]:
        """
        Assess performance against terrain-specific thresholds.
        
        Args:
            iou: Intersection over Union value
            centroid_offset: Centroid offset in meters
            terrain_class: Terrain classification
            
        Returns:
            Dictionary containing performance assessment
        """
        iou_thresholds = self.config['success_thresholds']
        centroid_thresholds = self.config['centroid_thresholds']
        
        iou_target = iou_thresholds.get(terrain_class, iou_thresholds['default'])
        centroid_target = centroid_thresholds.get(terrain_class, centroid_thresholds['default'])
        
        iou_pass = iou >= iou_target
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
            
            # Overall Performance
            f.write("Overall Performance:\n")
            f.write(f"  Mean IOU: {results_df['IOU'].mean():.3f}\n")
            f.write(f"  Median IOU: {results_df['IOU'].median():.3f}\n")
            f.write(f"  90th percentile IOU: {results_df['IOU'].quantile(0.9):.3f}\n")
            f.write(f"  95th percentile IOU: {results_df['IOU'].quantile(0.95):.3f}\n")
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
                    mean_iou = terrain_data['IOU'].mean()
                    pass_rate = terrain_data['Overall_Pass'].mean() * 100
                    mean_runtime = terrain_data['Runtime_s'].mean()
                    target_iou = self.config['success_thresholds'].get(terrain, 0.90)
                    
                    f.write(f"  {terrain.capitalize()}:\n")
                    f.write(f"    Basins: {len(terrain_data)}\n")
                    f.write(f"    Mean IOU: {mean_iou:.3f} (target: {target_iou:.3f})\n")
                    f.write(f"    Pass rate: {pass_rate:.1f}%\n")
                    f.write(f"    Mean runtime: {mean_runtime:.1f}s\n\n")
            
            # Performance by Basin Size
            f.write("Performance by Basin Size:\n")
            size_mapping = {'small': '5-20 km²', 'medium': '20-100 km²', 'large': '100-500 km²'}
            for size_class in ['small', 'medium', 'large']:
                size_data = results_df[results_df['Size_Class'] == size_class]
                if len(size_data) > 0:
                    mean_iou = size_data['IOU'].mean()
                    mean_runtime = size_data['Runtime_s'].mean()
                    
                    f.write(f"  {size_mapping[size_class]}:\n")
                    f.write(f"    Basins: {len(size_data)}\n")
                    f.write(f"    Mean IOU: {mean_iou:.3f}\n")
                    f.write(f"    Mean runtime: {mean_runtime:.1f}s\n\n")
            
            # Key Findings
            f.write("Key Findings:\n")
            
            # Runtime target assessment
            runtime_target = self.config['performance_analysis']['runtime_target']
            runtime_target_met = (results_df['Runtime_s'] <= runtime_target).mean() * 100
            f.write(f"  • {runtime_target_met:.1f}% of basins completed within {runtime_target}s target\n")
            
            # IOU target assessment
            iou_target = self.config['performance_analysis']['iou_target']
            target_met = (results_df['IOU'] >= iou_target).mean() * 100
            f.write(f"  • {target_met:.1f}% of basins achieved ≥{iou_target:.0%} IOU\n")
            
            # Identify problem cases
            low_iou = results_df[results_df['IOU'] < 0.8]
            if len(low_iou) > 0:
                f.write(f"  • {len(low_iou)} basins with IOU < 0.8 (investigate)\n")
            
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
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
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
        # Initialize benchmark runner
        runner = BenchmarkRunner(config_path=args.config, output_dir=args.output)
        
        # Override FLOWFINDER command if specified
        if args.flowfinder_cmd:
            runner.config['flowfinder_cli']['command'] = args.flowfinder_cmd
        
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