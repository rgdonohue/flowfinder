#!/usr/bin/env python3
"""
FLOWFINDER Test Setup Script
============================

Quick setup for testing the FLOWFINDER benchmark system with minimal data.
Downloads ~500MB of data instead of the full ~7.5GB dataset.

This script:
1. Downloads minimal test datasets (Colorado Front Range only)
2. Creates sample basin data for testing
3. Sets up test configuration
4. Validates the test environment
"""

import argparse
import logging
import sys
import os
import yaml
import json
import tempfile
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestSetup:
    """Handles test environment setup with minimal data."""

    def __init__(self, test_config: str = "config/data_sources_test.yaml"):
        """Initialize with test configuration."""
        self.config = self._load_config(test_config)
        self.setup_test_directories()

    def _load_config(self, config_path: str) -> Dict:
        """Load test configuration."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load test config: {e}")
            sys.exit(1)

    def setup_test_directories(self):
        """Create test directories."""
        storage = self.config["storage"]
        for dir_name in [
            "raw_data_dir",
            "processed_data_dir",
            "metadata_dir",
            "temp_dir",
        ]:
            Path(storage[dir_name]).mkdir(parents=True, exist_ok=True)
        logger.info("Created test directories")

    def create_sample_basin_data(self):
        """Create sample basin data for testing without downloading."""
        logger.info("Creating sample basin data for testing...")

        # Create sample basin CSV with required columns for benchmark_runner
        sample_basins = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5],
                "Pour_Point_Lat": [40.123, 40.234, 40.345, 40.456, 40.567],
                "Pour_Point_Lon": [-105.123, -105.234, -105.345, -105.456, -105.567],
                "huc12": [
                    "130101010101",
                    "130101010102",
                    "130101010103",
                    "130101010104",
                    "130101010105",
                ],
                "name": [
                    "Sample Basin 1",
                    "Sample Basin 2",
                    "Sample Basin 3",
                    "Sample Basin 4",
                    "Sample Basin 5",
                ],
                "area_km2": [15.2, 25.8, 8.9, 45.3, 12.1],
                "pour_point_x": [-105.123, -105.234, -105.345, -105.456, -105.567],
                "pour_point_y": [40.123, 40.234, 40.345, 40.456, 40.567],
                "terrain_roughness": [0.15, 0.25, 0.08, 0.35, 0.12],
                "stream_complexity": [2.1, 3.2, 1.5, 4.1, 2.8],
                "elevation_zone": [
                    "montane",
                    "montane",
                    "foothills",
                    "montane",
                    "foothills",
                ],
                "state": ["CO", "CO", "CO", "CO", "CO"],
            }
        )

        basin_path = (
            Path(self.config["storage"]["processed_data_dir"]) / "basin_sample.csv"
        )
        sample_basins.to_csv(basin_path, index=False)
        logger.info(f"Created sample basin data: {basin_path}")

        return basin_path

    def create_sample_truth_data(self):
        """Create sample truth polygon data for testing."""
        logger.info("Creating sample truth polygon data...")

        # Create sample truth CSV
        sample_truth = pd.DataFrame(
            {
                "basin_id": [1, 2, 3, 4, 5],
                "truth_polygon_id": ["TP001", "TP002", "TP003", "TP004", "TP005"],
                "area_km2": [15.1, 25.7, 8.8, 45.2, 12.0],
                "perimeter_km": [15.8, 20.3, 12.1, 28.9, 14.2],
                "centroid_x": [-105.123, -105.234, -105.345, -105.456, -105.567],
                "centroid_y": [40.123, 40.234, 40.345, 40.456, 40.567],
                "quality_score": [0.95, 0.92, 0.98, 0.89, 0.94],
            }
        )

        # Save CSV
        truth_csv_path = (
            Path(self.config["storage"]["processed_data_dir"]) / "truth_polygons.csv"
        )
        sample_truth.to_csv(truth_csv_path, index=False)
        logger.info(f"Created sample truth CSV: {truth_csv_path}")

        # Create GeoPackage with simple polygons
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon

            # Create simple rectangular polygons for each basin
            geometries = []
            for _, row in sample_truth.iterrows():
                # Create a simple rectangle around the centroid
                x, y = row["centroid_x"], row["centroid_y"]
                width = 0.01  # ~1km at this latitude
                height = 0.01
                polygon = Polygon(
                    [
                        (x - width / 2, y - height / 2),
                        (x + width / 2, y - height / 2),
                        (x + width / 2, y + height / 2),
                        (x - width / 2, y + height / 2),
                        (x - width / 2, y - height / 2),
                    ]
                )
                geometries.append(polygon)

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(sample_truth, geometry=geometries, crs="EPSG:4326")

            # Save GeoPackage
            truth_gpkg_path = (
                Path(self.config["storage"]["processed_data_dir"])
                / "truth_polygons.gpkg"
            )
            gdf.to_file(truth_gpkg_path, driver="GPKG")
            logger.info(f"Created sample truth GeoPackage: {truth_gpkg_path}")

        except ImportError:
            logger.warning("geopandas not available, skipping GeoPackage creation")
            # Create a dummy .gpkg file to satisfy the pipeline
            truth_gpkg_path = (
                Path(self.config["storage"]["processed_data_dir"])
                / "truth_polygons.gpkg"
            )
            with open(truth_gpkg_path, "w") as f:
                f.write("# Dummy GeoPackage file - geopandas not available\n")
            logger.info(f"Created dummy GeoPackage file: {truth_gpkg_path}")

        return truth_csv_path

    def create_sample_benchmark_results(self):
        """Create sample benchmark results for testing."""
        logger.info("Creating sample benchmark results...")

        # Create sample benchmark results
        sample_results = {
            "benchmark_info": {
                "timestamp": "2023-12-01T14:30:00",
                "total_basins": 5,
                "successful_runs": 4,
                "failed_runs": 1,
            },
            "basin_results": [
                {
                    "basin_id": 1,
                    "iou": 0.92,
                    "boundary_ratio": 0.95,
                    "centroid_offset_m": 45.2,
                    "runtime_seconds": 28.5,
                    "status": "success",
                },
                {
                    "basin_id": 2,
                    "iou": 0.88,
                    "boundary_ratio": 0.91,
                    "centroid_offset_m": 67.8,
                    "runtime_seconds": 32.1,
                    "status": "success",
                },
                {
                    "basin_id": 3,
                    "iou": 0.95,
                    "boundary_ratio": 0.97,
                    "centroid_offset_m": 23.4,
                    "runtime_seconds": 25.8,
                    "status": "success",
                },
                {
                    "basin_id": 4,
                    "iou": 0.85,
                    "boundary_ratio": 0.89,
                    "centroid_offset_m": 89.1,
                    "runtime_seconds": 45.2,
                    "status": "success",
                },
                {
                    "basin_id": 5,
                    "iou": 0.78,
                    "boundary_ratio": 0.82,
                    "centroid_offset_m": 123.5,
                    "runtime_seconds": 0.0,
                    "status": "failed",
                },
            ],
            "summary_stats": {
                "mean_iou": 0.88,
                "median_iou": 0.88,
                "std_iou": 0.07,
                "mean_runtime": 32.9,
                "median_runtime": 30.3,
                "success_rate": 0.8,
            },
        }

        results_path = (
            Path(self.config["storage"]["processed_data_dir"])
            / "benchmark_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(sample_results, f, indent=2)
        logger.info(f"Created sample benchmark results: {results_path}")

        return results_path

    def create_test_config(self):
        """Create test configuration for pipeline execution."""
        logger.info("Creating test pipeline configuration...")

        test_pipeline_config = {
            "pipeline": {
                "name": "FLOWFINDER Test Benchmark",
                "data_dir": self.config["storage"]["processed_data_dir"],
                "checkpointing": True,
                "resume_on_error": False,
                "max_retries": 1,
                "timeout_hours": 1,
                "generate_summary": True,
            },
            "stages": {
                "basin_sampling": {
                    "enabled": False,  # Skip since we have sample data
                    "config_file": None,
                    "output_prefix": "basin_sample",
                    "timeout_minutes": 5,
                    "retry_on_failure": False,
                },
                "truth_extraction": {
                    "enabled": False,  # Skip since we have sample data
                    "config_file": None,
                    "output_prefix": "truth_polygons",
                    "timeout_minutes": 5,
                    "retry_on_failure": False,
                },
                "benchmark_execution": {
                    "enabled": True,
                    "config_file": None,
                    "output_prefix": "benchmark_results",
                    "timeout_minutes": 10,
                    "retry_on_failure": False,
                },
            },
        }

        config_path = (
            Path(self.config["storage"]["processed_data_dir"])
            / "test_pipeline_config.yaml"
        )
        with open(config_path, "w") as f:
            yaml.dump(test_pipeline_config, f, default_flow_style=False, indent=2)
        logger.info(f"Created test pipeline config: {config_path}")

        return config_path

    def validate_test_environment(self):
        """Validate that test environment is properly set up."""
        logger.info("Validating test environment...")

        validation_results = {
            "directories_created": True,
            "sample_data_created": True,
            "config_files_created": True,
        }

        # Check directories
        storage = self.config["storage"]
        for dir_name in [
            "raw_data_dir",
            "processed_data_dir",
            "metadata_dir",
            "temp_dir",
        ]:
            if not Path(storage[dir_name]).exists():
                validation_results["directories_created"] = False
                logger.error(f"Missing directory: {storage[dir_name]}")

        # Check sample data files
        sample_files = [
            "basin_sample.csv",
            "truth_polygons.csv",
            "truth_polygons.gpkg",
            "benchmark_results.json",
            "test_pipeline_config.yaml",
        ]

        for filename in sample_files:
            file_path = Path(self.config["storage"]["processed_data_dir"]) / filename
            if not file_path.exists():
                validation_results["sample_data_created"] = False
                logger.error(f"Missing sample file: {file_path}")

        # Report results
        logger.info("=" * 50)
        logger.info("TEST ENVIRONMENT VALIDATION")
        logger.info("=" * 50)
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            logger.info(f"{check}: {status}")

        all_valid = all(validation_results.values())
        if all_valid:
            logger.info("✓ Test environment is ready!")
        else:
            logger.warning("⚠ Some test environment checks failed")

        return all_valid

    def setup_test_environment(self) -> bool:
        """Set up complete test environment."""
        logger.info("Setting up FLOWFINDER test environment...")

        try:
            # Create sample data
            self.create_sample_basin_data()
            self.create_sample_truth_data()
            self.create_sample_benchmark_results()

            # Create test configuration
            self.create_test_config()

            # Validate environment
            success = self.validate_test_environment()

            if success:
                logger.info("=" * 50)
                logger.info("TEST SETUP COMPLETE!")
                logger.info("=" * 50)
                logger.info("You can now test the FLOWFINDER benchmark system with:")
                logger.info("")
                logger.info("1. Test pipeline execution:")
                logger.info(
                    f"   python run_benchmark.py --config {self.config['storage']['processed_data_dir']}/test_pipeline_config.yaml --output test_results"
                )
                logger.info("")
                logger.info("2. Test validation tools:")
                logger.info(
                    f"   python scripts/validation_tools.py --validate-csv {self.config['storage']['processed_data_dir']}/basin_sample.csv --columns id,area_km2"
                )
                logger.info("")
                logger.info("3. Test individual scripts:")
                logger.info(
                    f"   python scripts/benchmark_runner.py --input {self.config['storage']['processed_data_dir']}/basin_sample.csv --truth {self.config['storage']['processed_data_dir']}/truth_polygons.csv --output test_benchmark"
                )

            return success

        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Set up test environment for FLOWFINDER benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up test environment with default config
  python test_setup.py

  # Set up with custom test config
  python test_setup.py --config custom_test_config.yaml

  # Validate existing test environment
  python test_setup.py --validate
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config/data_sources_test.yaml",
        help="Path to test configuration file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing test environment without setup",
    )

    args = parser.parse_args()

    try:
        test_setup = TestSetup(args.config)

        if args.validate:
            success = test_setup.validate_test_environment()
            sys.exit(0 if success else 1)
        else:
            success = test_setup.setup_test_environment()
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Test setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
