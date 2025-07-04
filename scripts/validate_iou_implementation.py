#!/usr/bin/env python3
"""
Validation Script for Robust IOU Implementation
===============================================

Tests the recently fixed IOU calculation in benchmark_runner.py to ensure
the robust implementation actually works under edge case conditions.

Tests critical scenarios:
1. Degenerate geometries (Point, LineString inputs)
2. Invalid return value handling (-1.0 for failures)
3. Error detection and logging
4. Geometry validation and repair

This validates that silent failures have been eliminated.
"""

import sys
import os
import logging
from io import StringIO
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

# Test imports
try:
    from benchmark_runner import BenchmarkRunner

    print("✅ Successfully imported BenchmarkRunner")
except ImportError as e:
    print(f"❌ Failed to import BenchmarkRunner: {e}")
    print("Note: This validation requires geopandas/shapely dependencies")
    sys.exit(1)

try:
    import pandas as pd
    from shapely.geometry import Point, Polygon, LineString, MultiPolygon

    print("✅ Successfully imported required dependencies")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


def create_test_runner():
    """Create a test BenchmarkRunner instance."""
    config = {
        "projection_crs": "EPSG:5070",
        "timeout_seconds": 120,
        "success_thresholds": {"default": 0.90},
        "centroid_thresholds": {"default": 500},
    }
    return BenchmarkRunner(sample_df=pd.DataFrame(), truth_path="dummy", config=config)


def test_iou_method_exists():
    """Test that the robust IOU methods exist."""
    print("\n🔍 Testing IOU method availability...")

    try:
        runner = create_test_runner()

        # Check that the enhanced IOU methods exist
        required_methods = [
            "_compute_iou",
            "_validate_geometries_for_iou",
            "_validate_repaired_geometry",
            "_is_valid_area_geometry",
        ]

        for method_name in required_methods:
            assert hasattr(runner, method_name), f"Missing method: {method_name}"
            method = getattr(runner, method_name)
            assert callable(method), f"Method {method_name} is not callable"

        print("✅ All robust IOU methods are available")
        return True

    except Exception as e:
        print(f"❌ IOU method availability test failed: {e}")
        return False


def test_degenerate_geometry_handling():
    """Test handling of degenerate geometries."""
    print("\n🔍 Testing degenerate geometry handling...")

    try:
        runner = create_test_runner()

        # Test Point vs Polygon (should return -1.0)
        point = Point(0.5, 0.5)
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        iou = runner._compute_iou(point, polygon)
        assert iou == -1.0, f"Point vs Polygon should return -1.0, got {iou}"

        # Test LineString vs Polygon (should return -1.0)
        line = LineString([(0, 0), (1, 1)])
        iou = runner._compute_iou(line, polygon)
        assert iou == -1.0, f"LineString vs Polygon should return -1.0, got {iou}"

        # Test Point vs Point (should return -1.0)
        point2 = Point(0.6, 0.6)
        iou = runner._compute_iou(point, point2)
        assert iou == -1.0, f"Point vs Point should return -1.0, got {iou}"

        print("✅ Degenerate geometry handling works correctly")
        return True

    except Exception as e:
        print(f"❌ Degenerate geometry test failed: {e}")
        return False


def test_none_geometry_handling():
    """Test handling of None geometries."""
    print("\n🔍 Testing None geometry handling...")

    try:
        runner = create_test_runner()
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        # Test None as first parameter
        iou = runner._compute_iou(None, polygon)
        assert iou == -1.0, f"None geometry should return -1.0, got {iou}"

        # Test None as second parameter
        iou = runner._compute_iou(polygon, None)
        assert iou == -1.0, f"None geometry should return -1.0, got {iou}"

        # Test both None
        iou = runner._compute_iou(None, None)
        assert iou == -1.0, f"Both None should return -1.0, got {iou}"

        print("✅ None geometry handling works correctly")
        return True

    except Exception as e:
        print(f"❌ None geometry test failed: {e}")
        return False


def test_empty_geometry_handling():
    """Test handling of empty geometries."""
    print("\n🔍 Testing empty geometry handling...")

    try:
        runner = create_test_runner()

        valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        empty_polygon = Polygon()

        # Test empty as first parameter (should return 0.0, not -1.0)
        iou = runner._compute_iou(empty_polygon, valid_polygon)
        assert iou == 0.0, f"Empty geometry should return 0.0, got {iou}"

        # Test empty as second parameter
        iou = runner._compute_iou(valid_polygon, empty_polygon)
        assert iou == 0.0, f"Empty geometry should return 0.0, got {iou}"

        # Test both empty
        iou = runner._compute_iou(empty_polygon, empty_polygon)
        assert iou == 0.0, f"Both empty should return 0.0, got {iou}"

        print("✅ Empty geometry handling works correctly")
        return True

    except Exception as e:
        print(f"❌ Empty geometry test failed: {e}")
        return False


def test_valid_iou_calculation():
    """Test valid IOU calculation for known overlapping polygons."""
    print("\n🔍 Testing valid IOU calculation...")

    try:
        runner = create_test_runner()

        # Create two overlapping squares with known IOU
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])  # Area = 4
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])  # Area = 4

        # Intersection should be 1x1 square = area 1
        # Union should be area 4 + 4 - 1 = 7
        # IOU should be 1/7 ≈ 0.142857

        iou = runner._compute_iou(poly1, poly2)

        # Should return valid IOU between 0 and 1
        assert 0 <= iou <= 1, f"IOU should be between 0 and 1, got {iou}"

        # Check if it's approximately correct
        expected_iou = 1.0 / 7.0  # ≈ 0.142857
        tolerance = 0.01
        assert (
            abs(iou - expected_iou) < tolerance
        ), f"IOU should be ~{expected_iou:.3f}, got {iou:.3f}"

        print(f"✅ Valid IOU calculation works correctly (IOU = {iou:.3f})")
        return True

    except Exception as e:
        print(f"❌ Valid IOU calculation test failed: {e}")
        return False


def test_self_intersecting_polygon_handling():
    """Test handling of self-intersecting polygons."""
    print("\n🔍 Testing self-intersecting polygon handling...")

    try:
        runner = create_test_runner()

        # Create bow-tie (self-intersecting) polygon
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        valid_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

        # The robust implementation should either:
        # 1. Successfully repair and calculate IOU (return 0.0-1.0)
        # 2. Detect failure and return -1.0
        iou = runner._compute_iou(bowtie, valid_poly)

        # Should not crash and should return valid result or -1.0
        assert (
            iou >= 0.0 or iou == -1.0
        ), f"Self-intersecting polygon should return valid IOU or -1.0, got {iou}"

        if iou >= 0.0:
            assert iou <= 1.0, f"Valid IOU should be <= 1.0, got {iou}"
            print(
                f"✅ Self-intersecting polygon successfully repaired (IOU = {iou:.3f})"
            )
        else:
            print("✅ Self-intersecting polygon correctly flagged as invalid (-1.0)")

        return True

    except Exception as e:
        print(f"❌ Self-intersecting polygon test failed: {e}")
        return False


def test_error_logging():
    """Test that errors are properly logged."""
    print("\n🔍 Testing error logging...")

    try:
        # Set up logging capture
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("test_iou_logger")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        runner = create_test_runner()
        runner.logger = logger

        # Test operation that should generate error logs
        iou = runner._compute_iou(
            None, Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        )
        assert iou == -1.0, "Should return -1.0 for None geometry"

        # Check that error was logged
        log_output = log_stream.getvalue()
        assert len(log_output) > 0, "Should have logged error message"
        assert (
            "None" in log_output or "cannot compute" in log_output
        ), "Should log specific error"

        logger.removeHandler(handler)

        print("✅ Error logging works correctly")
        return True

    except Exception as e:
        print(f"❌ Error logging test failed: {e}")
        return False


def test_multipolygon_handling():
    """Test handling of MultiPolygon geometries."""
    print("\n🔍 Testing MultiPolygon handling...")

    try:
        runner = create_test_runner()

        # Create MultiPolygon
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        multi_poly = MultiPolygon([poly1, poly2])

        # Test against single polygon
        single_poly = Polygon(
            [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]
        )

        iou = runner._compute_iou(multi_poly, single_poly)

        # Should handle gracefully
        assert (
            iou >= 0.0 or iou == -1.0
        ), f"MultiPolygon should return valid IOU or -1.0, got {iou}"

        if iou >= 0.0:
            assert iou <= 1.0, f"Valid IOU should be <= 1.0, got {iou}"
            print(f"✅ MultiPolygon handling works correctly (IOU = {iou:.3f})")
        else:
            print("✅ MultiPolygon correctly handled as invalid case")

        return True

    except Exception as e:
        print(f"❌ MultiPolygon handling test failed: {e}")
        return False


def test_validation_helper_methods():
    """Test the new validation helper methods."""
    print("\n🔍 Testing validation helper methods...")

    try:
        runner = create_test_runner()

        # Test _validate_geometries_for_iou
        point = Point(0.5, 0.5)
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        # Should detect Point as invalid for IOU
        result = runner._validate_geometries_for_iou(point, polygon)
        assert result == -1.0, f"Should detect Point as invalid, got {result}"

        # Should accept valid polygons
        result = runner._validate_geometries_for_iou(polygon, polygon)
        assert result is None, f"Should accept valid polygons, got {result}"

        # Test _is_valid_area_geometry
        assert (
            runner._is_valid_area_geometry(polygon) == True
        ), "Should accept valid polygon"
        assert (
            runner._is_valid_area_geometry(point) == False
        ), "Should reject point for area calc"
        assert runner._is_valid_area_geometry(None) == False, "Should reject None"

        print("✅ Validation helper methods work correctly")
        return True

    except Exception as e:
        print(f"❌ Validation helper methods test failed: {e}")
        return False


def run_validation():
    """Run all IOU validation tests."""
    print("=" * 70)
    print("🧪 IOU ROBUST IMPLEMENTATION VALIDATION")
    print("=" * 70)
    print("Testing the recently fixed IOU calculation for edge case handling...")

    tests = [
        test_iou_method_exists,
        test_degenerate_geometry_handling,
        test_none_geometry_handling,
        test_empty_geometry_handling,
        test_valid_iou_calculation,
        test_self_intersecting_polygon_handling,
        test_error_logging,
        test_multipolygon_handling,
        test_validation_helper_methods,
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
        print("🎉 ALL IOU VALIDATIONS PASSED!")
        print("✅ Robust IOU implementation is working correctly")
        print("✅ Silent failures have been eliminated")
        print("✅ Error handling is comprehensive")
        print("✅ Edge cases are properly handled")
        print("✅ Invalid calculations return -1.0 as designed")
        print("✅ Scientific integrity is protected")
        return True
    else:
        print("❌ SOME IOU VALIDATIONS FAILED!")
        print(f"⚠️  {total_tests - passed_tests} tests need attention")
        print("🚨 Silent failures may still be possible")
        return False


if __name__ == "__main__":
    success = run_validation()

    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run comprehensive edge case test suite with pytest")
        print("2. Test with real FLOWFINDER benchmark data")
        print("3. Validate under memory pressure and performance stress")
        print("4. Monitor for any new edge cases in production")
    else:
        print("\n⚠️  ISSUES DETECTED:")
        print("1. Review failed tests above")
        print("2. Check IOU implementation in benchmark_runner.py")
        print("3. Ensure all edge cases are properly handled")

    sys.exit(0 if success else 1)
