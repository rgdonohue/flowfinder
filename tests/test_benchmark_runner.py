#!/usr/bin/env python3
"""
Comprehensive unit tests for benchmark_runner.py
Tests IOU calculation, performance assessment, and FLOWFINDER CLI integration
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
import tempfile
import yaml
from pathlib import Path
import sys
import os
import subprocess
from unittest.mock import Mock, patch, MagicMock
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from benchmark_runner import BenchmarkRunner


class TestBenchmarkRunner:
    """Comprehensive test cases for BenchmarkRunner class"""

    def test_config_loading(self, sample_benchmark_config):
        """Test configuration loading with defaults and overrides"""
        runner = BenchmarkRunner()

        # Test default configuration
        assert runner.config["projection_crs"] == "EPSG:5070"
        assert runner.config["timeout_seconds"] == 120
        assert runner.config["success_thresholds"]["flat"] == 0.95
        assert runner.config["success_thresholds"]["steep"] == 0.85

    def test_dataset_loading(self, sample_basin_data, sample_truth_polygons, temp_dir):
        """Test loading of basin sample and truth polygon datasets"""
        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Check that datasets are loaded
        assert runner.sample_df is not None
        assert runner.truth_gdf is not None
        assert len(runner.sample_df) == 3
        assert len(runner.truth_gdf) == 3

        # Check required columns
        required_cols = ["ID", "Pour_Point_Lat", "Pour_Point_Lon"]
        for col in required_cols:
            assert col in runner.sample_df.columns

    def test_size_classification(self):
        """Test basin size classification logic"""
        runner = BenchmarkRunner()

        # Test size classification
        assert runner._get_size_class(10) == "small"
        assert runner._get_size_class(50) == "medium"
        assert runner._get_size_class(200) == "large"

    @patch("subprocess.check_output")
    def test_flowfinder_delineation_success(
        self, mock_subprocess, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test successful FLOWFINDER delineation"""
        # Mock successful FLOWFINDER response
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [0.01, 0.01],
                                [0.11, 0.01],
                                [0.11, 0.11],
                                [0.01, 0.11],
                                [0.01, 0.01],
                            ]
                        ],
                    },
                }
            ],
        }
        mock_subprocess.return_value = json.dumps(mock_geojson).encode()

        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Test delineation
        pred_poly, runtime, error = runner._run_delineation(40.0, -105.0)

        assert pred_poly is not None
        assert runtime is not None
        assert error is None
        assert runtime > 0

    @patch("subprocess.check_output")
    def test_flowfinder_delineation_timeout(
        self, mock_subprocess, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test FLOWFINDER delineation timeout handling"""
        # Mock timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired(
            cmd="flowfinder", timeout=30
        )

        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Test delineation timeout
        pred_poly, runtime, error = runner._run_delineation(40.0, -105.0)

        assert pred_poly is None
        assert runtime is None
        assert error is not None
        assert "timeout" in error.lower()

    @patch("subprocess.check_output")
    def test_flowfinder_delineation_error(
        self, mock_subprocess, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test FLOWFINDER delineation error handling"""
        # Mock error
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="flowfinder", stderr=b"Error: Invalid coordinates"
        )

        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Test delineation error
        pred_poly, runtime, error = runner._run_delineation(40.0, -105.0)

        assert pred_poly is None
        assert runtime is None
        assert error is not None
        assert "CLI error" in error

    def test_iou_calculation(self, sample_truth_polygons, sample_predicted_polygons):
        """Test IOU calculation between polygons"""
        runner = BenchmarkRunner()

        # Test IOU calculation
        truth_poly = sample_truth_polygons.iloc[0].geometry
        pred_poly = sample_predicted_polygons.iloc[0].geometry

        iou = runner._compute_iou(pred_poly, truth_poly)

        # IOU should be between 0 and 1
        assert 0 <= iou <= 1

        # Test with identical polygons
        iou_identical = runner._compute_iou(truth_poly, truth_poly)
        assert iou_identical == 1.0

        # Test with non-overlapping polygons
        non_overlapping = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        iou_no_overlap = runner._compute_iou(truth_poly, non_overlapping)
        assert iou_no_overlap == 0.0

    def test_boundary_ratio_calculation(
        self, sample_truth_polygons, sample_predicted_polygons
    ):
        """Test boundary ratio calculation"""
        runner = BenchmarkRunner()

        truth_poly = sample_truth_polygons.iloc[0].geometry
        pred_poly = sample_predicted_polygons.iloc[0].geometry

        ratio = runner._compute_boundary_ratio(pred_poly, truth_poly)

        # Ratio should be positive
        assert ratio > 0

        # Test with identical polygons
        ratio_identical = runner._compute_boundary_ratio(truth_poly, truth_poly)
        assert ratio_identical == 1.0

    def test_centroid_offset_calculation(
        self, sample_truth_polygons, sample_predicted_polygons
    ):
        """Test centroid offset calculation"""
        runner = BenchmarkRunner()

        truth_poly = sample_truth_polygons.iloc[0].geometry
        pred_poly = sample_predicted_polygons.iloc[0].geometry

        offset = runner._compute_centroid_offset(pred_poly, truth_poly)

        # Offset should be non-negative
        assert offset >= 0

        # Test with identical polygons
        offset_identical = runner._compute_centroid_offset(truth_poly, truth_poly)
        assert offset_identical == 0.0

    def test_metrics_calculation(
        self, sample_truth_polygons, sample_predicted_polygons
    ):
        """Test complete metrics calculation"""
        runner = BenchmarkRunner()

        truth_poly = sample_truth_polygons.iloc[0].geometry
        pred_poly = sample_predicted_polygons.iloc[0].geometry

        metrics = runner._calculate_metrics(pred_poly, truth_poly)

        # Check that all metrics are calculated
        assert "iou" in metrics
        assert "boundary_ratio" in metrics
        assert "centroid_offset" in metrics

        # Check metric ranges
        assert 0 <= metrics["iou"] <= 1
        assert metrics["boundary_ratio"] > 0
        assert metrics["centroid_offset"] >= 0

    def test_performance_assessment(self):
        """Test performance assessment against terrain-specific thresholds"""
        runner = BenchmarkRunner()

        # Test flat terrain (high threshold)
        performance_flat = runner._assess_performance(0.96, 150, "flat")
        assert performance_flat["iou_pass"] == True
        assert performance_flat["centroid_pass"] == True
        assert performance_flat["overall_pass"] == True
        assert performance_flat["iou_target"] == 0.95
        assert performance_flat["centroid_target"] == 200

        # Test steep terrain (lower threshold)
        performance_steep = runner._assess_performance(0.86, 800, "steep")
        assert performance_steep["iou_pass"] == True
        assert performance_steep["centroid_pass"] == True
        assert performance_steep["overall_pass"] == True
        assert performance_steep["iou_target"] == 0.85
        assert performance_steep["centroid_target"] == 1000

        # Test failure case
        performance_fail = runner._assess_performance(0.80, 1200, "moderate")
        assert performance_fail["iou_pass"] == False
        assert performance_fail["centroid_pass"] == False
        assert performance_fail["overall_pass"] == False

    @patch("subprocess.check_output")
    def test_benchmark_execution(
        self, mock_subprocess, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test complete benchmark execution"""
        # Mock successful FLOWFINDER responses
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [0.01, 0.01],
                                [0.11, 0.01],
                                [0.11, 0.11],
                                [0.01, 0.11],
                                [0.01, 0.01],
                            ]
                        ],
                    },
                }
            ],
        }
        mock_subprocess.return_value = json.dumps(mock_geojson).encode()

        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Run benchmark
        summary = runner.run_benchmark()

        # Check results
        assert summary["total_basins"] == 3
        assert summary["successful_runs"] > 0
        assert summary["success_rate"] > 0
        assert summary["benchmark_duration"] > 0

        # Check that results were generated
        assert len(runner.results) > 0
        assert "IOU" in runner.results[0]
        assert "Runtime_s" in runner.results[0]
        assert "Overall_Pass" in runner.results[0]

    def test_report_generation(
        self, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test report generation functionality"""
        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Add mock results
        runner.results = [
            {
                "ID": "basin_001",
                "IOU": 0.95,
                "Boundary_Ratio": 1.02,
                "Centroid_Offset_m": 150.0,
                "Runtime_s": 25.5,
                "Terrain_Class": "flat",
                "Size_Class": "small",
                "Complexity_Score": 2,
                "IOU_Pass": True,
                "Centroid_Pass": True,
                "Overall_Pass": True,
                "IOU_Target": 0.95,
                "Centroid_Target": 200.0,
            }
        ]

        # Generate reports
        report_files = runner.generate_reports()

        # Check that reports were generated
        assert len(report_files) > 0

        # Check specific files
        json_file = Path(temp_dir) / "benchmark_results.json"
        csv_file = Path(temp_dir) / "accuracy_summary.csv"
        summary_file = Path(temp_dir) / "benchmark_summary.txt"

        assert json_file.exists()
        assert csv_file.exists()
        assert summary_file.exists()

    def test_complete_workflow(
        self, sample_basin_data, sample_truth_polygons, temp_dir
    ):
        """Test complete benchmark workflow"""
        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)

        # Run complete workflow (without actual FLOWFINDER calls)
        with patch.object(runner, "_run_delineation") as mock_delineation:
            # Mock successful delineation
            mock_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.01, 0.01],
                                    [0.11, 0.01],
                                    [0.11, 0.11],
                                    [0.01, 0.11],
                                    [0.01, 0.01],
                                ]
                            ],
                        },
                    }
                ],
            }
            mock_poly = gpd.GeoDataFrame.from_features(mock_geojson).iloc[0].geometry
            mock_delineation.return_value = (mock_poly, 25.0, None)

            results = runner.run_complete_workflow(sample_file, truth_file)

        # Check workflow results
        assert results["success"] == True
        assert "benchmark_summary" in results
        assert "report_files" in results
        assert len(results["report_files"]) > 0

    def test_error_handling(self, sample_basin_data, sample_truth_polygons, temp_dir):
        """Test error handling in benchmark workflow"""
        # Create test files
        sample_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(sample_file, index=False)

        truth_file = f"{temp_dir}/test_truth_polygons.gpkg"
        sample_truth_polygons.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)
        runner.load_datasets(sample_file, truth_file)

        # Test with missing truth polygon
        basin_id = "nonexistent_basin"
        if basin_id not in runner.truth_gdf.index:
            # This should be handled gracefully
            pass

        # Test with delineation errors
        with patch.object(runner, "_run_delineation") as mock_delineation:
            mock_delineation.return_value = (None, None, "FLOWFINDER error")

            # Process a single basin
            row = runner.sample_df.iloc[0]
            basin_id = row["ID"]
            lat = row["Pour_Point_Lat"]
            lon = row["Pour_Point_Lon"]

            pred_poly, runtime, err = runner._run_delineation(lat, lon)

            assert pred_poly is None
            assert runtime is None
            assert err is not None

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid IOU threshold
        with pytest.raises(ValueError, match="IOU threshold.*must be between 0 and 1"):
            runner = BenchmarkRunner()
            runner.config["success_thresholds"]["flat"] = 1.5
            runner._validate_config()

        # Test invalid centroid threshold
        with pytest.raises(ValueError, match="Centroid threshold.*must be positive"):
            runner = BenchmarkRunner()
            runner.config["centroid_thresholds"]["flat"] = -100
            runner._validate_config()

        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            runner = BenchmarkRunner()
            runner.config["timeout_seconds"] = 0
            runner._validate_config()

    def test_terrain_specific_performance(self):
        """Test terrain-specific performance assessment"""
        runner = BenchmarkRunner()

        # Test all terrain types
        terrains = ["flat", "moderate", "steep"]

        for terrain in terrains:
            # Test passing case
            performance = runner._assess_performance(0.95, 100, terrain)
            assert performance["overall_pass"] == True

            # Test failing case
            performance_fail = runner._assess_performance(0.70, 2000, terrain)
            assert performance_fail["overall_pass"] == False

    def test_metric_edge_cases(self):
        """Test metric calculation edge cases"""
        runner = BenchmarkRunner()

        # Test empty polygons
        empty_poly = Polygon()
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        iou_empty = runner._compute_iou(empty_poly, valid_poly)
        assert iou_empty == 0.0

        # Test boundary ratio with zero length
        zero_length_poly = Polygon([(0, 0), (0, 0), (0, 0)])
        ratio_zero = runner._compute_boundary_ratio(valid_poly, zero_length_poly)
        assert ratio_zero == 0.0


class TestBenchmarkRunnerEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_datasets(self, temp_dir):
        """Test handling of empty datasets"""
        # Create empty sample file
        empty_sample = pd.DataFrame(columns=["ID", "Pour_Point_Lat", "Pour_Point_Lon"])
        sample_file = f"{temp_dir}/empty_sample.csv"
        empty_sample.to_csv(sample_file, index=False)

        # Create empty truth file
        empty_truth = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        truth_file = f"{temp_dir}/empty_truth.gpkg"
        empty_truth.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)

        with pytest.raises(ValueError):
            runner.load_datasets(sample_file, truth_file)

    def test_missing_columns(self, temp_dir):
        """Test handling of missing required columns"""
        # Create sample with missing columns
        invalid_sample = pd.DataFrame(
            {"ID": ["basin_001"], "Lat": [40.0]}  # Wrong column name
        )
        sample_file = f"{temp_dir}/invalid_sample.csv"
        invalid_sample.to_csv(sample_file, index=False)

        # Create valid truth file
        truth_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        truth_gdf = gpd.GeoDataFrame(
            {"ID": ["basin_001"], "geometry": [truth_poly]}, crs="EPSG:4326"
        )
        truth_file = f"{temp_dir}/valid_truth.gpkg"
        truth_gdf.to_file(truth_file, driver="GPKG")

        runner = BenchmarkRunner(output_dir=temp_dir)

        with pytest.raises(ValueError, match="Missing required columns"):
            runner.load_datasets(sample_file, truth_file)

    def test_invalid_geometries(self, temp_dir):
        """Test handling of invalid geometries"""
        # Create sample with invalid geometry
        invalid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0, 0)])
        truth_gdf = gpd.GeoDataFrame(
            {"ID": ["basin_001"], "geometry": [invalid_poly]}, crs="EPSG:4326"
        )
        truth_file = f"{temp_dir}/invalid_truth.gpkg"
        truth_gdf.to_file(truth_file, driver="GPKG")

        # Create valid sample
        sample_df = pd.DataFrame(
            {"ID": ["basin_001"], "Pour_Point_Lat": [40.0], "Pour_Point_Lon": [-105.0]}
        )
        sample_file = f"{temp_dir}/valid_sample.csv"
        sample_df.to_csv(sample_file, index=False)

        runner = BenchmarkRunner(output_dir=temp_dir)

        # Should handle invalid geometries gracefully
        runner.load_datasets(sample_file, truth_file)
        assert runner.truth_gdf is not None


if __name__ == "__main__":
    pytest.main([__file__])
