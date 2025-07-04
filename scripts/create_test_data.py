#!/usr/bin/env python3
"""
Create essential test data files to resolve test dependency issues.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point

    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: GeoPandas not available, creating minimal CSV files only")


def create_huc12_shapefile(data_dir: Path):
    """Create a minimal HUC12 shapefile for testing"""
    if not GEOSPATIAL_AVAILABLE:
        print("Skipping HUC12 shapefile creation - GeoPandas not available")
        return

    # Create simple test polygons
    huc12_data = {
        "HUC12": [
            "130101010101",
            "130101010102",
            "130101010103",
            "130101010104",
            "130101010105",
        ],
        "Name": ["Test HUC 1", "Test HUC 2", "Test HUC 3", "Test HUC 4", "Test HUC 5"],
        "AreaSqKm": [25.5, 35.2, 18.9, 52.1, 28.3],
        "geometry": [
            Polygon([(-105.2, 40.0), (-105.0, 40.0), (-105.0, 40.2), (-105.2, 40.2)]),
            Polygon([(-105.4, 40.1), (-105.2, 40.1), (-105.2, 40.3), (-105.4, 40.3)]),
            Polygon([(-105.6, 40.2), (-105.4, 40.2), (-105.4, 40.4), (-105.6, 40.4)]),
            Polygon([(-105.8, 40.3), (-105.6, 40.3), (-105.6, 40.5), (-105.8, 40.5)]),
            Polygon([(-106.0, 40.4), (-105.8, 40.4), (-105.8, 40.6), (-106.0, 40.6)]),
        ],
    }

    gdf = gpd.GeoDataFrame(huc12_data, crs="EPSG:4326")

    # Save to multiple locations tests expect
    huc12_file = data_dir / "huc12.shp"
    gdf.to_file(huc12_file)
    print(f"✅ Created HUC12 shapefile: {huc12_file}")


def create_nhd_catchments(data_dir: Path):
    """Create NHD+ HR catchments shapefile for testing"""
    if not GEOSPATIAL_AVAILABLE:
        print("Skipping NHD catchments creation - GeoPandas not available")
        return

    # Create test catchments that overlap with basin sample points
    catchments_data = {
        "FEATUREID": [1001, 1002, 1003, 1004, 1005],
        "GRIDCODE": [101, 102, 103, 104, 105],
        "AREASQKM": [15.2, 25.8, 8.9, 45.3, 12.1],
        "geometry": [
            Polygon([(-105.2, 40.0), (-105.0, 40.0), (-105.0, 40.2), (-105.2, 40.2)]),
            Polygon([(-105.3, 40.1), (-105.1, 40.1), (-105.1, 40.3), (-105.3, 40.3)]),
            Polygon([(-105.4, 40.2), (-105.2, 40.2), (-105.2, 40.4), (-105.4, 40.4)]),
            Polygon([(-105.5, 40.3), (-105.3, 40.3), (-105.3, 40.5), (-105.5, 40.5)]),
            Polygon([(-105.6, 40.4), (-105.4, 40.4), (-105.4, 40.6), (-105.6, 40.6)]),
        ],
    }

    gdf = gpd.GeoDataFrame(catchments_data, crs="EPSG:4326")

    # Save to test directory
    catchments_file = data_dir / "nhd_hr_catchments.shp"
    gdf.to_file(catchments_file)
    print(f"✅ Created NHD catchments: {catchments_file}")


def create_nhd_flowlines(data_dir: Path):
    """Create NHD+ flowlines for testing"""
    if not GEOSPATIAL_AVAILABLE:
        print("Skipping NHD flowlines creation - GeoPandas not available")
        return

    from shapely.geometry import LineString

    # Create test flowlines
    flowlines_data = {
        "COMID": [2001, 2002, 2003, 2004, 2005],
        "LENGTHKM": [5.2, 8.1, 3.9, 12.5, 6.8],
        "geometry": [
            LineString([(-105.15, 40.05), (-105.10, 40.15)]),
            LineString([(-105.25, 40.15), (-105.15, 40.25)]),
            LineString([(-105.35, 40.25), (-105.25, 40.35)]),
            LineString([(-105.45, 40.35), (-105.35, 40.45)]),
            LineString([(-105.55, 40.45), (-105.45, 40.55)]),
        ],
    }

    gdf = gpd.GeoDataFrame(flowlines_data, crs="EPSG:4326")

    flowlines_file = data_dir / "nhd_flowlines.shp"
    gdf.to_file(flowlines_file)
    print(f"✅ Created NHD flowlines: {flowlines_file}")


def create_enhanced_basin_sample(data_dir: Path):
    """Create enhanced basin sample with all required columns"""
    basin_data = {
        "ID": ["basin_001", "basin_002", "basin_003", "basin_004", "basin_005"],
        "Pour_Point_Lat": [40.123, 40.234, 40.345, 40.456, 40.567],
        "Pour_Point_Lon": [-105.123, -105.234, -105.345, -105.456, -105.567],
        "HUC12": [
            "130101010101",
            "130101010102",
            "130101010103",
            "130101010104",
            "130101010105",
        ],
        "Area_km2": [15.2, 25.8, 8.9, 45.3, 12.1],
        "Terrain_Class": ["montane", "montane", "foothills", "montane", "foothills"],
        "Elevation_m": [2100, 2450, 1890, 2650, 1950],
        "Slope_Mean": [15.2, 22.5, 8.9, 28.1, 12.3],
        "Slope_Std": [25.5, 32.1, 15.8, 38.9, 18.7],
        "Stream_Density": [0.15, 0.23, 0.08, 0.31, 0.12],
    }

    df = pd.DataFrame(basin_data)
    basin_file = data_dir / "basin_sample.csv"
    df.to_csv(basin_file, index=False)
    print(f"✅ Enhanced basin sample: {basin_file}")


def main():
    """Create all necessary test data files"""
    print("🏗️  Creating Test Data Files...")
    print("=" * 40)

    # Define data directories
    base_data_dir = Path(__file__).parent.parent / "data"
    test_data_dir = base_data_dir / "test" / "processed"

    # Ensure directories exist
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create main data directory files (for basin sampler tests)
    print("\n📁 Creating main data directory files...")
    create_huc12_shapefile(base_data_dir)
    create_nhd_catchments(base_data_dir)
    create_nhd_flowlines(base_data_dir)

    # Create test-specific files
    print("\n📁 Creating test-specific files...")
    create_enhanced_basin_sample(test_data_dir)
    create_nhd_catchments(test_data_dir)
    create_nhd_flowlines(test_data_dir)

    print("\n🎉 Test data creation completed!")
    print("\nCreated files:")

    # List created files
    for data_dir in [base_data_dir, test_data_dir]:
        for pattern in ["*.shp", "*.csv"]:
            for file in data_dir.glob(pattern):
                print(f"  ✅ {file}")


if __name__ == "__main__":
    main()
