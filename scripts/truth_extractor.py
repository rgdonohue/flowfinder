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
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity, make_valid
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


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
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initialize the truth extractor with configuration.
        
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
        self.basin_sample: Optional[gpd.GeoDataFrame] = None
        self.catchments: Optional[gpd.GeoDataFrame] = None
        self.flowlines: Optional[gpd.GeoDataFrame] = None
        self.truth_polygons: Optional[gpd.GeoDataFrame] = None
        
        self.logger.info("TruthExtractor initialized successfully")
    
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
            'basin_sample_file': 'basin_sample.csv',
            'buffer_tolerance': 500,  # meters for spatial join
            'min_area_ratio': 0.1,   # minimum truth area / sample area ratio
            'max_area_ratio': 10.0,  # maximum truth area / sample area ratio
            'target_crs': 'EPSG:5070',  # Albers Equal Area CONUS
            'output_crs': 'EPSG:4326',  # WGS84 for export
            'min_polygon_area': 1.0,  # km² minimum area
            'max_polygon_parts': 10,  # maximum polygon parts
            'max_attempts': 3,  # maximum extraction attempts
            'retry_with_different_strategies': True,
            'chunk_size': 100,  # for memory management
            'extraction_priority': {
                1: 'contains_point',
                2: 'largest_containing', 
                3: 'nearest_centroid',
                4: 'largest_intersecting'
            },
            'files': {
                'nhd_catchments': 'nhd_hr_catchments.shp',
                'nhd_flowlines': 'nhd_flowlines.shp'  # Optional for drainage validation
            },
            'quality_checks': {
                'topology_validation': True,
                'area_validation': True,
                'completeness_check': True,
                'drainage_check': False  # Requires flowlines
            },
            'terrain_extraction': {
                'alpine': {
                    'max_parts': 15,
                    'min_drainage_density': 0.01,
                    'buffer_tolerance': 1000
                },
                'desert': {
                    'max_parts': 3,
                    'min_drainage_density': 0.001,
                    'buffer_tolerance': 200
                }
            },
            'export': {
                'gpkg': True,
                'csv': True,
                'summary': True,
                'failed_extractions': True,
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
        """Configure logging for the extraction session."""
        log_file = f"truth_extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        # Validate area ratios
        if self.config['min_area_ratio'] >= self.config['max_area_ratio']:
            raise ValueError("min_area_ratio must be < max_area_ratio")
        
        if self.config['min_area_ratio'] <= 0 or self.config['max_area_ratio'] <= 0:
            raise ValueError("Area ratios must be positive")
        
        # Validate tolerance
        if self.config['buffer_tolerance'] <= 0:
            raise ValueError("buffer_tolerance must be > 0")
        
        # Validate file paths
        data_dir = Path(self.config['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        # Check required files
        basin_file = data_dir / self.config['basin_sample_file']
        if not basin_file.exists():
            raise FileNotFoundError(f"Basin sample file not found: {basin_file}")
        
        catchments_file = data_dir / self.config['files']['nhd_catchments']
        if not catchments_file.exists():
            raise FileNotFoundError(f"Catchments file not found: {catchments_file}")
        
        self.logger.info("Configuration validation passed")
    
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
        sample_file = Path(self.config['data_dir']) / self.config['basin_sample_file']
        self.logger.info(f"Loading basin sample from {sample_file}")
        
        try:
            # Load CSV and create GeoDataFrame
            df = pd.read_csv(sample_file)
            
            # Validate required columns
            required_cols = ['ID', 'Pour_Point_Lat', 'Pour_Point_Lon']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in basin sample: {missing_cols}")
            
            # Create point geometries from lat/lon
            geometry = [Point(lon, lat) for lon, lat in zip(df['Pour_Point_Lon'], df['Pour_Point_Lat'])]
            self.basin_sample = gpd.GeoDataFrame(df, geometry=geometry, crs=self.config['output_crs'])
            
            # Transform to target CRS for spatial operations
            self.basin_sample = self.basin_sample.to_crs(self.config['target_crs'])
            
            self.logger.info(f"Loaded {len(self.basin_sample)} basin sample points")
            
        except Exception as e:
            self.logger.error(f"Failed to load basin sample: {e}")
            raise
    
    def _load_nhd_data(self) -> None:
        """Load NHD+ HR catchments and optional flowlines."""
        data_dir = Path(self.config['data_dir'])
        files = self.config['files']
        target_crs = self.config['target_crs']
        
        try:
            # Load catchments
            self.logger.info("Loading NHD+ HR catchments...")
            catchments_file = data_dir / files['nhd_catchments']
            self.catchments = gpd.read_file(catchments_file)
            self.catchments = self.catchments.to_crs(target_crs)
            
            # Ensure required columns exist
            if 'FEATUREID' not in self.catchments.columns:
                self.logger.warning("FEATUREID column not found in catchments - using index")
                self.catchments['FEATUREID'] = self.catchments.index.astype(str)
            
            self.logger.info(f"Loaded {len(self.catchments)} NHD+ HR catchments")
            
            # Load flowlines if drainage check enabled
            if self.config['quality_checks']['drainage_check'] and files.get('nhd_flowlines'):
                self.logger.info("Loading NHD+ flowlines for drainage validation...")
                flowlines_file = data_dir / files['nhd_flowlines']
                if flowlines_file.exists():
                    self.flowlines = gpd.read_file(flowlines_file)
                    self.flowlines = self.flowlines.to_crs(target_crs)
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
        
        self.logger.info("Extracting truth polygons via spatial join...")
        
        buffer_tolerance = self.config['buffer_tolerance']
        truth_polygons = []
        
        for idx, basin in tqdm(self.basin_sample.iterrows(), total=len(self.basin_sample), desc="Extracting truth polygons"):
            basin_id = str(basin['ID'])
            
            try:
                # Create buffer around pour point for spatial join tolerance
                point_buffer = basin.geometry.buffer(buffer_tolerance)
                
                # Find intersecting catchments
                intersecting = self.catchments[self.catchments.intersects(point_buffer)]
                
                if len(intersecting) == 0:
                    self._log_error(basin_id, 'no_catchment', f"No catchments within {buffer_tolerance}m")
                    continue
                elif len(intersecting) == 1:
                    # Single catchment - ideal case
                    truth_poly = intersecting.iloc[0].copy()
                    truth_poly['extraction_method'] = 'single_catchment'
                else:
                    # Multiple catchments - need to choose best one
                    self.logger.debug(f"Basin {basin_id}: {len(intersecting)} intersecting catchments")
                    
                    # Apply extraction strategy priority
                    truth_poly = self._apply_extraction_strategy(basin, intersecting, basin_id)
                    if truth_poly is None:
                        continue
                
                # Add basin metadata
                truth_poly['basin_id'] = basin_id
                truth_poly['sample_area_km2'] = basin.get('Area_km2', np.nan)
                truth_poly['sample_terrain'] = basin.get('Terrain_Class', 'unknown')
                truth_poly['sample_complexity'] = basin.get('Complexity_Score', np.nan)
                
                # Calculate truth polygon area
                truth_poly['truth_area_km2'] = truth_poly.geometry.area / 1e6
                
                truth_polygons.append(truth_poly)
                
            except Exception as e:
                self._log_error(basin_id, 'extraction_error', str(e))
                continue
        
        if not truth_polygons:
            raise ValueError("No truth polygons extracted - check spatial data and configuration")
        
        # Create GeoDataFrame
        self.truth_polygons = gpd.GeoDataFrame(truth_polygons, crs=self.config['target_crs'])
        
        self.logger.info(f"Truth extraction complete: {len(self.truth_polygons)} polygons extracted")
        return self.truth_polygons
    
    def _apply_extraction_strategy(self, basin: pd.Series, intersecting: gpd.GeoDataFrame, basin_id: str) -> Optional[pd.Series]:
        """
        Apply extraction strategy priority to select best catchment.
        
        Args:
            basin: Basin sample row
            intersecting: Intersecting catchments
            basin_id: Basin identifier for logging
            
        Returns:
            Selected catchment or None if extraction fails
        """
        priority_order = self.config['extraction_priority']
        
        for priority, strategy in priority_order.items():
            try:
                if strategy == 'contains_point':
                    # Choose catchment containing the pour point
                    containing = intersecting[intersecting.contains(basin.geometry)]
                    if len(containing) == 1:
                        truth_poly = containing.iloc[0].copy()
                        truth_poly['extraction_method'] = 'contains_point'
                        return truth_poly
                    elif len(containing) > 1:
                        # Multiple containing - choose largest
                        largest_idx = containing.geometry.area.idxmax()
                        truth_poly = containing.loc[largest_idx].copy()
                        truth_poly['extraction_method'] = 'largest_containing'
                        self._log_error(basin_id, 'multiple_containing', f"{len(containing)} catchments contain point")
                        return truth_poly
                
                elif strategy == 'largest_containing':
                    # Choose largest catchment containing the point
                    containing = intersecting[intersecting.contains(basin.geometry)]
                    if len(containing) > 0:
                        largest_idx = containing.geometry.area.idxmax()
                        truth_poly = containing.loc[largest_idx].copy()
                        truth_poly['extraction_method'] = 'largest_containing'
                        return truth_poly
                
                elif strategy == 'nearest_centroid':
                    # Choose nearest catchment by centroid distance
                    distances = intersecting.geometry.centroid.distance(basin.geometry)
                    nearest_idx = distances.idxmin()
                    truth_poly = intersecting.loc[nearest_idx].copy()
                    truth_poly['extraction_method'] = 'nearest_centroid'
                    self._log_error(basin_id, 'point_not_contained', "Pour point not contained in any catchment")
                    return truth_poly
                
                elif strategy == 'largest_intersecting':
                    # Choose largest intersecting catchment
                    largest_idx = intersecting.geometry.area.idxmax()
                    truth_poly = intersecting.loc[largest_idx].copy()
                    truth_poly['extraction_method'] = 'largest_intersecting'
                    return truth_poly
                    
            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed for basin {basin_id}: {e}")
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
            'total_polygons': len(self.truth_polygons),
            'valid_polygons': 0,
            'invalid_polygons': 0,
            'area_ratio_violations': 0,
            'topology_errors': 0,
            'completeness_issues': 0,
            'validation_details': []
        }
        
        quality_checks = self.config['quality_checks']
        
        for idx, row in self.truth_polygons.iterrows():
            basin_id = row['basin_id']
            validation_detail = {'basin_id': basin_id, 'issues': []}
            
            try:
                # Topology validation
                if quality_checks['topology_validation']:
                    if not row.geometry.is_valid:
                        validation_detail['issues'].append('invalid_topology')
                        validation_results['topology_errors'] += 1
                    elif row.geometry.is_empty:
                        validation_detail['issues'].append('empty_geometry')
                        validation_results['completeness_issues'] += 1
                
                # Area validation
                if quality_checks['area_validation']:
                    truth_area = row['truth_area_km2']
                    sample_area = row['sample_area_km2']
                    
                    if not np.isnan(sample_area) and sample_area > 0:
                        area_ratio = truth_area / sample_area
                        min_ratio = self.config['min_area_ratio']
                        max_ratio = self.config['max_area_ratio']
                        
                        if area_ratio < min_ratio or area_ratio > max_ratio:
                            validation_detail['issues'].append(f'area_ratio_violation_{area_ratio:.2f}')
                            validation_results['area_ratio_violations'] += 1
                
                # Completeness check
                if quality_checks['completeness_check']:
                    if truth_area < self.config['min_polygon_area']:
                        validation_detail['issues'].append('area_too_small')
                        validation_results['completeness_issues'] += 1
                
                # Mark as valid if no issues
                if not validation_detail['issues']:
                    validation_results['valid_polygons'] += 1
                else:
                    validation_results['invalid_polygons'] += 1
                
                validation_results['validation_details'].append(validation_detail)
                
            except Exception as e:
                self.logger.warning(f"Validation failed for basin {basin_id}: {e}")
                validation_detail['issues'].append('validation_error')
                validation_results['invalid_polygons'] += 1
                validation_results['validation_details'].append(validation_detail)
        
        # Log validation summary
        self.logger.info(f"Quality validation complete:")
        self.logger.info(f"  Valid polygons: {validation_results['valid_polygons']}")
        self.logger.info(f"  Invalid polygons: {validation_results['invalid_polygons']}")
        self.logger.info(f"  Area ratio violations: {validation_results['area_ratio_violations']}")
        self.logger.info(f"  Topology errors: {validation_results['topology_errors']}")
        self.logger.info(f"  Completeness issues: {validation_results['completeness_issues']}")
        
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
        export_config = self.config['export']
        
        # Export GeoPackage
        if export_config.get('gpkg', True):
            gpkg_path = f"{output_prefix}.gpkg"
            export_gdf = self.truth_polygons.to_crs(self.config['output_crs'])
            export_gdf.to_file(gpkg_path, driver='GPKG')
            exported_files.append(gpkg_path)
            self.logger.info(f"Exported GeoPackage: {gpkg_path}")
        
        # Export CSV (non-geometry attributes)
        if export_config.get('csv', True):
            csv_path = f"{output_prefix}.csv"
            # Remove geometry column for CSV export
            csv_df = self.truth_polygons.drop(columns=['geometry'])
            csv_df.to_csv(csv_path, index=False)
            exported_files.append(csv_path)
            self.logger.info(f"Exported CSV: {csv_path}")
        
        # Export summary
        if export_config.get('summary', True):
            summary_path = f"{output_prefix}_summary.txt"
            self._write_summary(summary_path)
            exported_files.append(summary_path)
            self.logger.info(f"Exported summary: {summary_path}")
        
        # Export failed extractions
        if export_config.get('failed_extractions', True):
            failed_path = f"{output_prefix}_failed.csv"
            self._export_failed_extractions(failed_path)
            if Path(failed_path).exists():
                exported_files.append(failed_path)
                self.logger.info(f"Exported failed extractions: {failed_path}")
        
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
        """Write extraction summary to text file."""
        if self.truth_polygons is None:
            return
        
        with open(summary_path, 'w') as f:
            f.write("FLOWFINDER Truth Extraction Summary\n")
            f.write("===================================\n\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Basins Processed: {len(self.basin_sample)}\n")
            f.write(f"Successful Extractions: {len(self.truth_polygons)}\n")
            f.write(f"Failed Extractions: {len(self.error_logs)}\n\n")
            
            # Extraction method statistics
            if 'extraction_method' in self.truth_polygons.columns:
                f.write("Extraction Methods:\n")
                method_counts = self.truth_polygons['extraction_method'].value_counts()
                for method, count in method_counts.items():
                    f.write(f"  {method}: {count}\n")
                f.write("\n")
            
            # Area statistics
            if 'truth_area_km2' in self.truth_polygons.columns:
                areas = self.truth_polygons['truth_area_km2']
                f.write(f"Area Statistics:\n")
                f.write(f"  Min: {areas.min():.1f} km²\n")
                f.write(f"  Max: {areas.max():.1f} km²\n")
                f.write(f"  Mean: {areas.mean():.1f} km²\n")
                f.write(f"  Median: {areas.median():.1f} km²\n\n")
            
            # Terrain distribution
            if 'sample_terrain' in self.truth_polygons.columns:
                f.write("Terrain Distribution:\n")
                terrain_counts = self.truth_polygons['sample_terrain'].value_counts()
                for terrain, count in terrain_counts.items():
                    f.write(f"  {terrain}: {count}\n")
    
    def _export_failed_extractions(self, failed_path: str) -> None:
        """Export list of failed extractions."""
        if not self.error_logs:
            return
        
        # Get unique failed basin IDs
        failed_basins = set()
        for error in self.error_logs:
            if error['error_type'] in ['no_catchment', 'extraction_error']:
                failed_basins.add(error['basin_id'])
        
        if failed_basins:
            # Find corresponding basin data
            failed_data = self.basin_sample[self.basin_sample['ID'].isin(failed_basins)]
            if not failed_data.empty:
                failed_data.to_csv(failed_path, index=False)
    
    def _log_error(self, basin_id: str, error_type: str, message: str) -> None:
        """Log structured error for later analysis."""
        error_record = {
            'basin_id': basin_id,
            'error_type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.error_logs.append(error_record)
        self.logger.warning(f"Basin {basin_id} - {error_type}: {message}")
    
    def run_complete_workflow(self, output_prefix: str = "truth_polygons") -> Dict[str, Any]:
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
                'success': True,
                'extracted_count': len(truth_polygons),
                'validation_results': validation_results,
                'exported_files': exported_files,
                'error_count': len(self.error_logs)
            }
            
            self.logger.info("Truth extraction workflow completed successfully")
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
        'basin_sample_file': 'basin_sample.csv',
        'buffer_tolerance': 500,
        'min_area_ratio': 0.1,
        'max_area_ratio': 10.0,
        'target_crs': 'EPSG:5070',
        'output_crs': 'EPSG:4326',
        'files': {
            'nhd_catchments': 'nhd_hr_catchments.shp',
            'nhd_flowlines': 'nhd_flowlines.shp'
        },
        'quality_checks': {
            'topology_validation': True,
            'area_validation': True,
            'completeness_check': True,
            'drainage_check': False
        },
        'export': {
            'gpkg': True,
            'csv': True,
            'summary': True,
            'failed_extractions': True
        }
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
        default='truth_polygons',
        help='Output file prefix (default: truth_polygons)'
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
        # Initialize extractor
        extractor = TruthExtractor(config_path=args.config, data_dir=args.data_dir)
        
        # Run complete workflow
        results = extractor.run_complete_workflow(args.output)
        
        if results['success']:
            print(f"\n✅ Truth extraction completed successfully!")
            print(f"📊 Extracted {results['extracted_count']} truth polygons")
            print(f"📁 Exported {len(results['exported_files'])} files")
            if results['error_count'] > 0:
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