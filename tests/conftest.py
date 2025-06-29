#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for FLOWFINDER benchmark tests
"""

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

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


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
    # Create diverse basin geometries representing Mountain West terrain
    basins = [
        # Small alpine basin (steep, complex)
        Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        # Medium foothill basin (moderate, medium complexity)
        Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]),
        # Large valley basin (flat, simple)
        Polygon([(0.5, 0.5), (0.8, 0.5), (0.8, 0.8), (0.5, 0.8)]),
        # Complex plateau basin (moderate, high complexity)
        Polygon([(1.0, 1.0), (1.3, 1.0), (1.3, 1.3), (1.0, 1.3)]),
        # Desert wash basin (flat, low complexity)
        Polygon([(1.5, 1.5), (1.7, 1.5), (1.7, 1.7), (1.5, 1.7)])
    ]
    
    basin_data = {
        'HUC12': ['1201', '1202', '1203', '1204', '1205'],
        'STATES': ['CO', 'UT', 'NM', 'WY', 'AZ'],
        'geometry': basins
    }
    
    gdf = gpd.GeoDataFrame(basin_data, crs='EPSG:4326')
    gdf.to_file(f"{temp_dir}/test_huc12.shp")
    return gdf


@pytest.fixture(scope="session")
def sample_flowlines_gdf(temp_dir):
    """Create sample flowline geometries for testing"""
    # Create flowlines that intersect with basins
    flowlines = [
        LineString([(0.05, 0.05), (0.08, 0.08)]),  # Alpine stream
        LineString([(0.3, 0.3), (0.35, 0.35)]),    # Foothill stream
        LineString([(0.65, 0.65), (0.75, 0.75)]),  # Valley stream
        LineString([(1.15, 1.15), (1.25, 1.25)]),  # Plateau stream
        LineString([(1.6, 1.6), (1.65, 1.65)])     # Desert wash
    ]
    
    flowline_data = {
        'geometry': flowlines,
        'GNIS_NAME': ['Alpine Creek', 'Foothill Stream', 'Valley River', 'Plateau Creek', 'Desert Wash']
    }
    
    gdf = gpd.GeoDataFrame(flowline_data, crs='EPSG:4326')
    gdf.to_file(f"{temp_dir}/test_flowlines.shp")
    return gdf


@pytest.fixture(scope="session")
def sample_catchments_gdf(temp_dir):
    """Create sample catchment polygons for truth extraction testing"""
    # Create catchments that correspond to basins
    catchments = [
        Polygon([(0, 0), (0.12, 0), (0.12, 0.12), (0, 0.12)]),  # Alpine catchment
        Polygon([(0.18, 0.18), (0.42, 0.18), (0.42, 0.42), (0.18, 0.42)]),  # Foothill catchment
        Polygon([(0.48, 0.48), (0.82, 0.48), (0.82, 0.82), (0.48, 0.82)]),  # Valley catchment
        Polygon([(0.98, 0.98), (1.32, 0.98), (1.32, 1.32), (0.98, 1.32)]),  # Plateau catchment
        Polygon([(1.48, 1.48), (1.72, 1.48), (1.72, 1.72), (1.48, 1.72)])   # Desert catchment
    ]
    
    catchment_data = {
        'FEATUREID': ['1001', '1002', '1003', '1004', '1005'],
        'AREA': [14.4, 576.0, 1156.0, 1156.0, 576.0],  # km²
        'geometry': catchments
    }
    
    gdf = gpd.GeoDataFrame(catchment_data, crs='EPSG:4326')
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
    elevation_data[60:80, 60:80] -= 500   # Valley
    
    # Create raster file
    dem_path = f"{temp_dir}/test_dem.tif"
    
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elevation_data.dtype,
        crs='EPSG:4326',
        transform=from_bounds(0, 0, 2, 2, width, height)
    ) as dst:
        dst.write(elevation_data, 1)
    
    return dem_path


@pytest.fixture(scope="session")
def sample_basin_data():
    """Create sample basin data for testing"""
    return pd.DataFrame({
        'ID': ['basin_001', 'basin_002', 'basin_003'],
        'HUC12': ['1201', '1202', '1203'],
        'Pour_Point_Lat': [40.0, 41.0, 42.0],
        'Pour_Point_Lon': [-105.0, -106.0, -107.0],
        'Area_km2': [15.5, 45.2, 125.8],
        'Terrain_Class': ['steep', 'moderate', 'flat'],
        'Size_Class': ['small', 'medium', 'large'],
        'Complexity_Score': [3, 2, 1],
        'Slope_Std': [25.5, 12.3, 3.2],
        'Stream_Density': [0.15, 0.08, 0.03]
    })


@pytest.fixture(scope="session")
def sample_truth_polygons():
    """Create sample truth polygons for testing"""
    truth_polygons = [
        Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]),
        Polygon([(0.5, 0.5), (0.8, 0.5), (0.8, 0.8), (0.5, 0.8)])
    ]
    
    truth_data = {
        'ID': ['basin_001', 'basin_002', 'basin_003'],
        'geometry': truth_polygons,
        'Area_km2': [15.2, 44.8, 126.1]
    }
    
    return gpd.GeoDataFrame(truth_data, crs='EPSG:4326')


@pytest.fixture(scope="session")
def sample_predicted_polygons():
    """Create sample predicted polygons for testing"""
    # Slightly different from truth to test IOU calculation
    predicted_polygons = [
        Polygon([(0.01, 0.01), (0.11, 0.01), (0.11, 0.11), (0.01, 0.11)]),
        Polygon([(0.21, 0.21), (0.41, 0.21), (0.41, 0.41), (0.21, 0.41)]),
        Polygon([(0.51, 0.51), (0.81, 0.51), (0.81, 0.81), (0.51, 0.81)])
    ]
    
    predicted_data = {
        'ID': ['basin_001', 'basin_002', 'basin_003'],
        'geometry': predicted_polygons,
        'Area_km2': [15.8, 45.5, 125.2]
    }
    
    return gpd.GeoDataFrame(predicted_data, crs='EPSG:4326')


@pytest.fixture
def mock_flowfinder_cli():
    """Mock FLOWFINDER CLI for testing"""
    with patch('subprocess.run') as mock_run:
        # Mock successful FLOWFINDER execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout=b"FLOWFINDER delineation completed successfully",
            stderr=b""
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
                    "coordinates": [[[0.01, 0.01], [0.11, 0.01], [0.11, 0.11], [0.01, 0.11], [0.01, 0.01]]]
                }
            }
        ]
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'data_dir': 'test_data',
        'area_range': [5, 500],
        'snap_tolerance': 150,
        'n_per_stratum': 1,
        'target_crs': 'EPSG:5070',
        'output_crs': 'EPSG:4326',
        'mountain_west_states': ['CO', 'UT', 'NM', 'WY', 'MT', 'ID', 'AZ'],
        'files': {
            'huc12': 'test_huc12.shp',
            'catchments': 'test_catchments.shp',
            'flowlines': 'test_flowlines.shp',
            'dem': 'test_dem.tif',
            'slope': None
        },
        'export': {
            'csv': True,
            'gpkg': True,
            'summary': True
        }
    }


@pytest.fixture
def sample_benchmark_config():
    """Sample benchmark configuration for testing"""
    return {
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
        }
    } 