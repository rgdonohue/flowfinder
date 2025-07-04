#!/usr/bin/env python3
"""
Simple test runner for GeometryDiagnostics without pytest dependency.
Tests critical functionality to ensure the implementation works.
"""

import sys
import os
import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from geometry_utils import GeometryDiagnostics


def test_basic_initialization():
    """Test basic initialization."""
    print("Testing basic initialization...")
    diag = GeometryDiagnostics()
    assert diag.logger is not None
    assert diag.config is not None
    assert "geometry_repair" in diag.config
    print("✅ Basic initialization test passed")


def test_valid_geometry_diagnosis():
    """Test diagnosis of valid geometry."""
    print("Testing valid geometry diagnosis...")
    diag = GeometryDiagnostics()
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    diagnosis = diag._diagnose_single_geometry(polygon, 0)

    assert diagnosis["is_valid"] is True
    assert diagnosis["is_empty"] is False
    assert diagnosis["geometry_type"] == "Polygon"
    assert diagnosis["repair_strategy"] == "none"
    print("✅ Valid geometry diagnosis test passed")


def test_invalid_geometry_diagnosis():
    """Test diagnosis of self-intersecting geometry."""
    print("Testing invalid geometry diagnosis...")
    diag = GeometryDiagnostics()
    # Create bow-tie polygon (self-intersecting)
    coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
    invalid_polygon = Polygon(coords)

    diagnosis = diag._diagnose_single_geometry(invalid_polygon, 0)

    assert diagnosis["is_valid"] is False
    assert (
        "self_intersection" in diagnosis["issues"]
        or "ring_self_intersection" in diagnosis["issues"]
    )
    assert diagnosis["repair_strategy"] == "buffer_fix"
    print("✅ Invalid geometry diagnosis test passed")


def test_none_geometry_handling():
    """Test handling of None geometry."""
    print("Testing None geometry handling...")
    diag = GeometryDiagnostics()
    diagnosis = diag._diagnose_single_geometry(None, 0)

    assert diagnosis["is_valid"] is False
    assert diagnosis["is_critical"] is True
    assert diagnosis["geometry_type"] == "None"
    assert "null_geometry" in diagnosis["issues"]
    assert diagnosis["repair_strategy"] == "remove"
    print("✅ None geometry handling test passed")


def test_repair_strategies():
    """Test basic repair strategies."""
    print("Testing repair strategies...")
    diag = GeometryDiagnostics()

    # Test make_valid strategy
    coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
    invalid_polygon = Polygon(coords)

    repaired = diag._apply_repair_strategy(invalid_polygon, "make_valid", 0)
    assert repaired is not None
    assert repaired.is_valid

    # Test convex_hull strategy
    repaired_hull = diag._apply_repair_strategy(invalid_polygon, "convex_hull", 0)
    assert repaired_hull is not None
    assert repaired_hull.is_valid

    # Test remove strategy
    removed = diag._apply_repair_strategy(invalid_polygon, "remove", 0)
    assert removed is None

    print("✅ Repair strategies test passed")


def test_geodataframe_processing():
    """Test complete GeoDataFrame processing."""
    print("Testing GeoDataFrame processing...")
    diag = GeometryDiagnostics()

    # Create mix of valid and invalid geometries
    valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    invalid_polygon = Polygon(
        [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
    )  # Self-intersecting

    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": [valid_polygon, invalid_polygon]})

    result = diag.diagnose_and_repair_geometries(gdf, "test data")

    # Should have both geometries (invalid one repaired)
    assert len(result) == 2
    # All geometries should be valid after repair
    assert all(geom.is_valid for geom in result.geometry)

    print("✅ GeoDataFrame processing test passed")


def test_empty_geodataframe():
    """Test handling of empty GeoDataFrame."""
    print("Testing empty GeoDataFrame handling...")
    diag = GeometryDiagnostics()

    empty_gdf = gpd.GeoDataFrame({"geometry": []})
    result = diag.diagnose_and_repair_geometries(empty_gdf, "empty test")

    assert len(result) == 0
    assert result.empty
    print("✅ Empty GeoDataFrame test passed")


def test_memory_efficient_processing():
    """Test processing of moderately large dataset."""
    print("Testing memory efficient processing...")
    diag = GeometryDiagnostics()

    # Create dataset with 100 geometries
    num_geoms = 100
    geometries = []

    for i in range(num_geoms):
        if i % 10 == 0:
            # Add some invalid geometries
            coords = [(i, i), (i + 2, i + 2), (i + 2, i), (i, i + 2), (i, i)]
            geometries.append(Polygon(coords))
        else:
            # Add valid geometries
            geom = Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i + 1), (i, i)])
            geometries.append(geom)

    gdf = gpd.GeoDataFrame({"id": range(num_geoms), "geometry": geometries})

    result = diag.diagnose_and_repair_geometries(gdf, "large dataset")

    # Should process all geometries
    assert len(result) == num_geoms
    # All remaining should be valid
    assert all(geom.is_valid for geom in result.geometry)

    print("✅ Memory efficient processing test passed")


def test_analysis_stats():
    """Test geometry analysis statistics."""
    print("Testing analysis statistics...")
    diag = GeometryDiagnostics()

    # Create geometries with known issues
    valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    invalid_geom = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    empty_geom = Polygon()

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "geometry": [valid_geom, invalid_geom, empty_geom]}
    )

    stats = diag.analyze_geometry_issues(gdf, "stats test")

    assert stats["total_features"] == 3
    assert stats["total_valid"] == 2  # valid_geom and empty_geom are both valid
    assert stats["total_invalid"] >= 1  # At least the self-intersecting one
    assert stats["total_empty"] == 1
    assert "Polygon" in stats["geometry_types"]

    print("✅ Analysis statistics test passed")


def run_all_tests():
    """Run all tests."""
    print("Running comprehensive GeometryDiagnostics tests...")
    print("=" * 60)

    try:
        test_basic_initialization()
        test_valid_geometry_diagnosis()
        test_invalid_geometry_diagnosis()
        test_none_geometry_handling()
        test_repair_strategies()
        test_geodataframe_processing()
        test_empty_geodataframe()
        test_memory_efficient_processing()
        test_analysis_stats()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("GeometryDiagnostics class is working correctly.")
        print("Critical spatial operations are functioning as expected.")

    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
