#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for FLOWFINDER benchmark tests
"""

import warnings
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import tempfile
import yaml
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch
import rasterio
from rasterio.transform import from_bounds

# Suppress common deprecation warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyogrio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyproj")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*shapely.geos.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*CRS.*unsafe.*"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*ndim.*scalar.*"
)

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_basins_gdf(temp_dir):
    """Create sample basin geometries for testing"""
    # Create diverse basin geometries representing Mountain West terrain with realistic sizes
    # Using coordinates around Denver, CO area for realistic location
    # Made larger to ensure areas are >5 km² after coordinate transformation
    basins = [
        # Small alpine basin (steep, complex) - ~10 km²
        Polygon([(-105.0, 39.7), (-105.05, 39.7), (-105.05, 39.75), (-105.0, 39.75)]),
        # Medium foothill basin (moderate, medium complexity) - ~50 km²
        Polygon([(-105.1, 39.8), (-105.2, 39.8), (-105.2, 39.9), (-105.1, 39.9)]),
        # Large valley basin (flat, simple) - ~150 km²
        Polygon([(-105.3, 40.0), (-105.5, 40.0), (-105.5, 40.2), (-105.3, 40.2)]),
        # Complex plateau basin (moderate, high complexity) - ~75 km²
        Polygon([(-105.6, 40.3), (-105.75, 40.3), (-105.75, 40.45), (-105.6, 40.45)]),
        # Desert wash basin (flat, low complexity) - ~25 km²
        Polygon([(-105.8, 40.5), (-105.9, 40.5), (-105.9, 40.6), (-105.8, 40.6)]),
    ]

    basin_data = {
        "HUC12": ["1201", "1202", "1203", "1204", "1205"],
        "STATES": ["CO", "UT", "NM", "WY", "AZ"],
        "geometry": basins,
    }

    gdf = gpd.GeoDataFrame(basin_data, crs="EPSG:4326")
    gdf.to_file(f"{temp_dir}/test_huc12.shp")
    return gdf


@pytest.fixture(scope="session")
def sample_flowlines_gdf(temp_dir):
    """Create sample flowline geometries for testing"""
    # Create flowlines that intersect with basins in Colorado coordinates
    flowlines = [
        LineString([(-105.01, 39.71), (-105.011, 39.711)]),  # Alpine stream
        LineString([(-105.125, 39.825), (-105.13, 39.83)]),  # Foothill stream
        LineString([(-105.25, 39.95), (-105.26, 39.96)]),  # Valley stream
        LineString([(-105.435, 40.135), (-105.44, 40.14)]),  # Plateau stream
        LineString([(-105.52, 40.22), (-105.525, 40.225)]),  # Desert wash
    ]

    flowline_data = {
        "geometry": flowlines,
        "GNIS_NAME": [
            "Alpine Creek",
            "Foothill Stream",
            "Valley River",
            "Plateau Creek",
            "Desert Wash",
        ],
    }

    gdf = gpd.GeoDataFrame(flowline_data, crs="EPSG:4326")
    gdf.to_file(f"{temp_dir}/test_flowlines.shp")
    return gdf


@pytest.fixture(scope="session")
def sample_catchments_gdf(temp_dir):
    """Create sample catchment polygons for truth extraction testing"""
    # Create catchments that correspond to basins in Colorado coordinates
    catchments = [
        Polygon(
            [
                (-105.005, 39.695),
                (-105.025, 39.695),
                (-105.025, 39.725),
                (-105.005, 39.725),
            ]
        ),  # Alpine catchment
        Polygon(
            [(-105.08, 39.78), (-105.17, 39.78), (-105.17, 39.87), (-105.08, 39.87)]
        ),  # Foothill catchment
        Polygon(
            [(-105.18, 39.88), (-105.32, 39.88), (-105.32, 40.02), (-105.18, 40.02)]
        ),  # Valley catchment
        Polygon(
            [(-105.38, 40.08), (-105.49, 40.08), (-105.49, 40.19), (-105.38, 40.19)]
        ),  # Plateau catchment
        Polygon(
            [(-105.48, 40.18), (-105.56, 40.18), (-105.56, 40.26), (-105.48, 40.26)]
        ),  # Desert catchment
    ]

    catchment_data = {
        "FEATUREID": ["1001", "1002", "1003", "1004", "1005"],
        "AREA": [14.4, 576.0, 1156.0, 1156.0, 576.0],  # km²
        "geometry": catchments,
    }

    gdf = gpd.GeoDataFrame(catchment_data, crs="EPSG:4326")
    gdf.to_file(f"{temp_dir}/test_catchments.shp")
    return gdf


@pytest.fixture(scope="session")
def sample_dem_raster(temp_dir):
    """Create a sample DEM raster for testing"""
    # Create a simple DEM with elevation data
    height, width = 100, 100
    elevation_data = np.random.randint(1000, 4000, (height, width)).astype(np.float32)

    # Add some terrain features
    elevation_data[20:30, 20:30] += 1000  # Mountain peak
    elevation_data[60:80, 60:80] -= 500  # Valley

    # Create raster file
    dem_path = f"{temp_dir}/test_dem.tif"

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=elevation_data.dtype,
        crs="EPSG:4326",
        transform=from_bounds(-105.6, 39.6, -104.9, 40.3, width, height),
    ) as dst:
        dst.write(elevation_data, 1)

    return dem_path


@pytest.fixture(scope="session")
def sample_basin_data():
    """Create sample basin data for testing"""
    return pd.DataFrame(
        {
            "ID": ["basin_001", "basin_002", "basin_003"],
            "HUC12": ["1201", "1202", "1203"],
            "Pour_Point_Lat": [40.0, 41.0, 42.0],
            "Pour_Point_Lon": [-105.0, -106.0, -107.0],
            "Area_km2": [15.5, 45.2, 125.8],
            "Terrain_Class": ["steep", "moderate", "flat"],
            "Size_Class": ["small", "medium", "large"],
            "Complexity_Score": [3, 2, 1],
            "Slope_Std": [25.5, 12.3, 3.2],
            "Stream_Density": [0.15, 0.08, 0.03],
        }
    )


@pytest.fixture(scope="session")
def sample_truth_polygons():
    """Create sample truth polygons for testing"""
    truth_polygons = [
        Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]),
        Polygon([(0.5, 0.5), (0.8, 0.5), (0.8, 0.8), (0.5, 0.8)]),
    ]

    truth_data = {
        "ID": ["basin_001", "basin_002", "basin_003"],
        "geometry": truth_polygons,
        "Area_km2": [15.2, 44.8, 126.1],
    }

    return gpd.GeoDataFrame(truth_data, crs="EPSG:4326")


@pytest.fixture(scope="session")
def sample_predicted_polygons():
    """Create sample predicted polygons for testing"""
    # Slightly different from truth to test IOU calculation
    predicted_polygons = [
        Polygon([(0.01, 0.01), (0.11, 0.01), (0.11, 0.11), (0.01, 0.11)]),
        Polygon([(0.21, 0.21), (0.41, 0.21), (0.41, 0.41), (0.21, 0.41)]),
        Polygon([(0.51, 0.51), (0.81, 0.51), (0.81, 0.81), (0.51, 0.81)]),
    ]

    predicted_data = {
        "ID": ["basin_001", "basin_002", "basin_003"],
        "geometry": predicted_polygons,
        "Area_km2": [15.8, 45.5, 125.2],
    }

    return gpd.GeoDataFrame(predicted_data, crs="EPSG:4326")


@pytest.fixture
def mock_flowfinder_cli():
    """Mock FLOWFINDER CLI for testing"""
    with patch("subprocess.run") as mock_run:
        # Mock successful FLOWFINDER execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout=b"FLOWFINDER delineation completed successfully",
            stderr=b"",
        )
        yield mock_run


@pytest.fixture
def mock_flowfinder_output():
    """Mock FLOWFINDER output GeoJSON"""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"basin_id": "basin_001"},
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


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "data_dir": "test_data",
        "area_range": [5, 500],
        "snap_tolerance": 150,
        "n_per_stratum": 1,
        "target_crs": "EPSG:5070",
        "output_crs": "EPSG:4326",
        "mountain_west_states": ["CO", "UT", "NM", "WY", "MT", "ID", "AZ"],
        "files": {
            "huc12": "test_huc12.shp",
            "catchments": "test_catchments.shp",
            "flowlines": "test_flowlines.shp",
            "dem": "test_dem.tif",
            "slope": None,
        },
        "export": {"csv": True, "gpkg": True, "summary": True},
    }


@pytest.fixture
def sample_benchmark_config():
    """Sample benchmark configuration for testing"""
    return {
        "projection_crs": "EPSG:5070",
        "timeout_seconds": 120,
        "success_thresholds": {
            "flat": 0.95,
            "moderate": 0.92,
            "steep": 0.85,
            "default": 0.90,
        },
        "centroid_thresholds": {
            "flat": 200,
            "moderate": 500,
            "steep": 1000,
            "default": 500,
        },
        "flowfinder_cli": {
            "command": "flowfinder",
            "subcommand": "delineate",
            "output_format": "geojson",
            "additional_args": [],
            "env_vars": {},
        },
        "metrics": {
            "iou": True,
            "boundary_ratio": True,
            "centroid_offset": True,
            "runtime": True,
        },
        "output_formats": ["json", "csv", "summary", "errors"],
    }


@pytest.fixture(scope="function")
def setup_test_data_files(
    temp_dir,
    sample_basins_gdf,
    sample_flowlines_gdf,
    sample_catchments_gdf,
    sample_dem_raster,
):
    """Set up test data files with expected names in the temp directory"""
    import shutil

    # Copy files with expected names
    shutil.copy(f"{temp_dir}/test_huc12.shp", f"{temp_dir}/huc12.shp")
    shutil.copy(f"{temp_dir}/test_huc12.shx", f"{temp_dir}/huc12.shx")
    shutil.copy(f"{temp_dir}/test_huc12.dbf", f"{temp_dir}/huc12.dbf")
    shutil.copy(f"{temp_dir}/test_huc12.prj", f"{temp_dir}/huc12.prj")
    shutil.copy(f"{temp_dir}/test_huc12.cpg", f"{temp_dir}/huc12.cpg")

    shutil.copy(f"{temp_dir}/test_flowlines.shp", f"{temp_dir}/nhd_flowlines.shp")
    shutil.copy(f"{temp_dir}/test_flowlines.shx", f"{temp_dir}/nhd_flowlines.shx")
    shutil.copy(f"{temp_dir}/test_flowlines.dbf", f"{temp_dir}/nhd_flowlines.dbf")
    shutil.copy(f"{temp_dir}/test_flowlines.prj", f"{temp_dir}/nhd_flowlines.prj")
    shutil.copy(f"{temp_dir}/test_flowlines.cpg", f"{temp_dir}/nhd_flowlines.cpg")

    shutil.copy(f"{temp_dir}/test_catchments.shp", f"{temp_dir}/nhd_hr_catchments.shp")
    shutil.copy(f"{temp_dir}/test_catchments.shx", f"{temp_dir}/nhd_hr_catchments.shx")
    shutil.copy(f"{temp_dir}/test_catchments.dbf", f"{temp_dir}/nhd_hr_catchments.dbf")
    shutil.copy(f"{temp_dir}/test_catchments.prj", f"{temp_dir}/nhd_hr_catchments.prj")
    shutil.copy(f"{temp_dir}/test_catchments.cpg", f"{temp_dir}/nhd_hr_catchments.cpg")

    shutil.copy(sample_dem_raster, f"{temp_dir}/dem_10m.tif")

    return temp_dir
