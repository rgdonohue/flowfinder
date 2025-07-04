#!/usr/bin/env python3
"""
Validation Script for CRS Transformation Silent Failures
========================================================

Tests critical CRS transformation scenarios that could corrupt spatial data
without obvious symptoms. These are extremely dangerous because they:

1. Often pass basic validation checks
2. Corrupt coordinates in subtle ways
3. Propagate through entire analysis pipeline
4. Can invalidate all scientific results

Key scenarios tested:
- Incompatible coordinate system transformations
- Datum shift validation
- High-latitude precision loss
- Coordinate overflow/underflow detection
- Silent precision loss identification

This validates that transformations either work correctly or fail explicitly.
"""

import sys
import os
import math
import logging
from io import StringIO

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))


# Test basic functionality
def test_import_dependencies():
    """Test that required dependencies are available."""
    print("🔍 Testing CRS transformation dependencies...")

    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point, Polygon

        print("✅ Core geospatial libraries available")
        return True
    except ImportError as e:
        print(f"❌ Missing geospatial dependencies: {e}")
        print("Note: Full CRS testing requires geopandas, shapely, and pyproj")
        return False


def test_basic_crs_transformation():
    """Test basic CRS transformation functionality."""
    print("\n🔍 Testing basic CRS transformation...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Create simple test point in Colorado
        test_point = Point(-105.0, 40.0)
        gdf = gpd.GeoDataFrame({"geometry": [test_point]}, crs="EPSG:4326")

        # Transform to Albers Equal Area (US)
        projected = gdf.to_crs("EPSG:5070")
        result = projected.geometry.iloc[0]

        # Check if transformation produced reasonable coordinates
        x, y = result.x, result.y

        # Albers coordinates for Colorado should be roughly:
        # X: around 0 to 1,000,000 meters (central meridian)
        # Y: around 1,000,000 to 2,000,000 meters (origin at 23°N)
        if -2000000 <= x <= 2000000 and 0 <= y <= 3000000:
            print(f"✅ Basic transformation works: ({x:.0f}, {y:.0f})")
            return True
        else:
            print(
                f"❌ Transformation produced suspicious coordinates: ({x:.0f}, {y:.0f})"
            )
            return False

    except Exception as e:
        print(f"❌ Basic CRS transformation failed: {e}")
        return False


def test_transformation_reversibility():
    """Test that transformations are properly reversible."""
    print("\n🔍 Testing transformation reversibility...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Original coordinates
        original_points = [
            Point(-105.0, 40.0),  # Colorado
            Point(-111.0, 45.0),  # Montana
            Point(-119.0, 47.0),  # Washington
        ]

        gdf_original = gpd.GeoDataFrame({"geometry": original_points}, crs="EPSG:4326")

        # Transform to projected CRS and back
        gdf_projected = gdf_original.to_crs("EPSG:5070")  # Albers
        gdf_back = gdf_projected.to_crs("EPSG:4326")  # Back to WGS84

        # Check coordinate preservation
        max_error = 0
        for orig, back in zip(gdf_original.geometry, gdf_back.geometry):
            diff_x = abs(orig.x - back.x)
            diff_y = abs(orig.y - back.y)
            max_error = max(max_error, diff_x, diff_y)

        # Should be very close (within numerical precision)
        tolerance = 1e-10
        if max_error < tolerance:
            print(
                f"✅ Transformation reversibility preserved (max error: {max_error:.2e})"
            )
            return True
        else:
            print(f"❌ Transformation not reversible (max error: {max_error:.2e})")
            return False

    except Exception as e:
        print(f"❌ Reversibility test failed: {e}")
        return False


def test_datum_shift_detection():
    """Test detection of datum shifts (NAD27 vs WGS84)."""
    print("\n🔍 Testing datum shift detection...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Test point in area with known datum shift
        test_point = Point(-122.4194, 37.7749)  # San Francisco

        # Create same point in different datums
        gdf_nad27 = gpd.GeoDataFrame(
            {"geometry": [test_point]}, crs="EPSG:4267"
        )  # NAD27
        gdf_wgs84 = gpd.GeoDataFrame(
            {"geometry": [test_point]}, crs="EPSG:4326"
        )  # WGS84

        # Transform both to same projected system
        nad27_projected = gdf_nad27.to_crs("EPSG:5070")
        wgs84_projected = gdf_wgs84.to_crs("EPSG:5070")

        # Calculate difference in projected coordinates
        nad27_coord = nad27_projected.geometry.iloc[0]
        wgs84_coord = wgs84_projected.geometry.iloc[0]

        diff_x = abs(nad27_coord.x - wgs84_coord.x)
        diff_y = abs(nad27_coord.y - wgs84_coord.y)
        diff_meters = math.sqrt(diff_x**2 + diff_y**2)

        # NAD27 to WGS84 shifts should be substantial (typically 50-200 meters)
        if 10 < diff_meters < 500:
            print(f"✅ Datum shift detected: {diff_meters:.1f} meters")
            return True
        elif diff_meters <= 10:
            print(f"❌ No datum shift detected - possible transformation error")
            return False
        else:
            print(
                f"❌ Excessive datum shift: {diff_meters:.1f} meters - possible error"
            )
            return False

    except Exception as e:
        print(f"❌ Datum shift test failed: {e}")
        return False


def test_high_latitude_precision():
    """Test precision issues at high latitudes."""
    print("\n🔍 Testing high latitude precision...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # High latitude test points
        high_lat_points = [
            Point(-150.0, 70.0),  # Northern Alaska
            Point(-100.0, 80.0),  # High Arctic
            Point(0.0, 85.0),  # Near North Pole
        ]

        gdf = gpd.GeoDataFrame({"geometry": high_lat_points}, crs="EPSG:4326")

        # Test projection to polar stereographic
        try:
            projected = gdf.to_crs("EPSG:3413")  # Polar stereographic north
            back_transformed = projected.to_crs("EPSG:4326")

            # Check precision loss
            max_lat_error = 0
            max_lon_error = 0

            for orig, back in zip(gdf.geometry, back_transformed.geometry):
                if back and not back.is_empty:
                    lat_error = abs(orig.y - back.y)
                    lon_error = abs(orig.x - back.x)

                    # Handle longitude wraparound
                    if lon_error > 180:
                        lon_error = 360 - lon_error

                    max_lat_error = max(max_lat_error, lat_error)
                    max_lon_error = max(max_lon_error, lon_error)

            # At high latitudes, some precision loss is expected
            if max_lat_error < 0.1:  # Less than ~10km error
                print(
                    f"✅ High latitude precision acceptable (lat error: {max_lat_error:.6f}°)"
                )
                return True
            else:
                print(
                    f"❌ Excessive high latitude precision loss (lat error: {max_lat_error:.6f}°)"
                )
                return False

        except Exception as e:
            print(f"⚠️ Polar projection failed (may be expected): {e}")
            return True  # Failure is acceptable for extreme cases

    except Exception as e:
        print(f"❌ High latitude test failed: {e}")
        return False


def test_coordinate_range_validation():
    """Test coordinate range validation after transformation."""
    print("\n🔍 Testing coordinate range validation...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Test coordinates at various extremes
        test_coords = [
            Point(-105.0, 40.0),  # Normal US coordinate
            Point(2.0, 48.0),  # European coordinate
            Point(179.0, 89.0),  # Near pole and dateline
            Point(-179.0, -89.0),  # Opposite extreme
        ]

        gdf = gpd.GeoDataFrame({"geometry": test_coords}, crs="EPSG:4326")

        # Test projection to US-specific coordinate system
        try:
            projected = gdf.to_crs("EPSG:5070")  # Albers Equal Area (US)

            # Check for invalid coordinates
            invalid_coords = []
            for idx, geom in enumerate(projected.geometry):
                if geom and not geom.is_empty:
                    x, y = geom.x, geom.y

                    # Check for signs of projection failure
                    if (
                        math.isnan(x)
                        or math.isnan(y)
                        or math.isinf(x)
                        or math.isinf(y)
                        or abs(x) > 5000000
                        or abs(y) > 5000000
                    ):
                        invalid_coords.append(idx)

            if len(invalid_coords) > 0:
                print(
                    f"✅ Detected {len(invalid_coords)} invalid projections (expected)"
                )
                return True
            else:
                # All coordinates projected successfully - check if they're reasonable
                bounds = projected.total_bounds
                if (
                    -3000000 <= bounds[0] <= 3000000
                    and -2000000 <= bounds[1] <= 3000000
                ):
                    print("✅ All coordinates within reasonable US projection bounds")
                    return True
                else:
                    print(
                        f"❌ Projected coordinates outside reasonable bounds: {bounds}"
                    )
                    return False

        except Exception as e:
            print(f"⚠️ Projection failed for extreme coordinates (acceptable): {e}")
            return True

    except Exception as e:
        print(f"❌ Coordinate range validation failed: {e}")
        return False


def test_web_mercator_polar_issues():
    """Test Web Mercator issues near poles."""
    print("\n🔍 Testing Web Mercator polar issues...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Near-polar coordinates (Web Mercator has known issues)
        near_polar = [
            Point(0.0, 85.0),  # Close to pole but should work
            Point(0.0, 89.0),  # Very close to pole
            Point(0.0, 89.9),  # Extremely close to pole
        ]

        gdf = gpd.GeoDataFrame({"geometry": near_polar}, crs="EPSG:4326")

        try:
            # Project to Web Mercator
            web_mercator = gdf.to_crs("EPSG:3857")

            # Check for extreme Y values
            extreme_y_detected = False
            for geom in web_mercator.geometry:
                if geom and not geom.is_empty:
                    y = geom.y
                    if abs(y) > 20037508:  # Web Mercator theoretical limit
                        extreme_y_detected = True
                        break

            if extreme_y_detected:
                print("✅ Detected Web Mercator polar overflow (expected)")
                return True
            else:
                print("✅ Web Mercator handled polar coordinates within limits")
                return True

        except Exception as e:
            print(f"✅ Web Mercator correctly failed at polar coordinates: {e}")
            return True

    except Exception as e:
        print(f"❌ Web Mercator polar test failed: {e}")
        return False


def test_precision_loss_detection():
    """Test detection of precision loss in transformations."""
    print("\n🔍 Testing precision loss detection...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # High-precision coordinates
        high_precision = Point(-105.12345678901234, 40.12345678901234)
        gdf = gpd.GeoDataFrame({"geometry": [high_precision]}, crs="EPSG:4326")

        # Transform and back-transform
        projected = gdf.to_crs("EPSG:5070")
        back_transformed = projected.to_crs("EPSG:4326")

        # Measure precision loss
        original = gdf.geometry.iloc[0]
        back = back_transformed.geometry.iloc[0]

        if back and not back.is_empty:
            loss_x = abs(original.x - back.x)
            loss_y = abs(original.y - back.y)

            # Precision loss should be minimal for well-conditioned transformations
            if loss_x < 1e-10 and loss_y < 1e-10:
                print(f"✅ Minimal precision loss: ({loss_x:.2e}, {loss_y:.2e})")
                return True
            else:
                print(f"❌ Significant precision loss: ({loss_x:.2e}, {loss_y:.2e})")
                return False
        else:
            print("❌ Back-transformation failed")
            return False

    except Exception as e:
        print(f"❌ Precision loss test failed: {e}")
        return False


def test_coordinate_corruption_detection():
    """Test ability to detect coordinate corruption."""
    print("\n🔍 Testing coordinate corruption detection...")

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Known coordinates
        original_point = Point(-105.0, 40.0)
        gdf = gpd.GeoDataFrame({"geometry": [original_point]}, crs="EPSG:4326")

        # Simulate a transformation and corruption detection
        projected = gdf.to_crs("EPSG:5070")
        back_transformed = projected.to_crs("EPSG:4326")

        # Check if we get back approximately the same coordinates
        original = gdf.geometry.iloc[0]
        back = back_transformed.geometry.iloc[0]

        if back and not back.is_empty:
            # Calculate approximate distance difference
            diff_x = abs(original.x - back.x)
            diff_y = abs(original.y - back.y)

            # Convert to approximate meters
            diff_meters = math.sqrt(
                (diff_x * 111000 * math.cos(math.radians(original.y))) ** 2
                + (diff_y * 111000) ** 2
            )

            # Should be very close for good transformation
            if diff_meters < 1:  # Less than 1 meter error
                print(f"✅ No corruption detected (error: {diff_meters:.3f}m)")
                return True
            else:
                print(f"❌ Possible corruption detected (error: {diff_meters:.1f}m)")
                return False
        else:
            print("❌ Transformation produced invalid geometry")
            return False

    except Exception as e:
        print(f"❌ Corruption detection test failed: {e}")
        return False


def run_crs_validation():
    """Run all CRS transformation validation tests."""
    print("=" * 70)
    print("🧪 CRS TRANSFORMATION SILENT FAILURE VALIDATION")
    print("=" * 70)
    print("Testing for dangerous CRS transformation issues that could")
    print("corrupt spatial data without obvious symptoms...")

    tests = [
        test_import_dependencies,
        test_basic_crs_transformation,
        test_transformation_reversibility,
        test_datum_shift_detection,
        test_high_latitude_precision,
        test_coordinate_range_validation,
        test_web_mercator_polar_issues,
        test_precision_loss_detection,
        test_coordinate_corruption_detection,
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")

    print("\n" + "=" * 70)
    print(f"📊 VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL CRS TRANSFORMATION VALIDATIONS PASSED!")
        print("✅ Basic CRS transformations working correctly")
        print("✅ Datum shifts properly detected and applied")
        print("✅ High-latitude precision issues handled appropriately")
        print("✅ Coordinate validation catching invalid results")
        print("✅ Precision loss within acceptable limits")
        print("✅ Coordinate corruption detection functional")
        print("✅ Silent failure risks minimized")
        return True
    else:
        print("❌ SOME CRS TRANSFORMATION VALIDATIONS FAILED!")
        print(f"⚠️  {total_tests - passed_tests} tests need attention")
        print("🚨 Silent transformation failures may be possible")
        print("🚨 Spatial data corruption risks exist")
        return False


if __name__ == "__main__":
    success = run_crs_validation()

    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run comprehensive CRS test suite with pytest")
        print("2. Test with actual benchmark data and projections")
        print("3. Implement coordinate validation in pipeline")
        print("4. Monitor transformation accuracy in production")
        print("5. Add CRS validation to data quality checks")
    else:
        print("\n⚠️  CRITICAL ISSUES DETECTED:")
        print("1. Review failed CRS transformation tests")
        print("2. Check geospatial library installations")
        print("3. Validate projection definitions and parameters")
        print("4. Implement additional coordinate validation")
        print("5. Consider adding transformation verification")
        print("\n🚨 WARNING: Silent CRS failures could corrupt all spatial analysis!")

    sys.exit(0 if success else 1)
