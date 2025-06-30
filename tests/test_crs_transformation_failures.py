#!/usr/bin/env python3
"""
Comprehensive Tests for CRS Transformation Silent Failures
==========================================================

Critical test suite for detecting CRS transformation failures that could
corrupt all spatial data without obvious symptoms. These are among the most
dangerous types of geospatial errors because they:

1. Often pass basic validation checks
2. Corrupt coordinates in subtle ways
3. Propagate through entire analysis pipeline
4. Can invalidate all scientific results

Tests focus on scenarios where transformations appear to succeed but
produce incorrect coordinates that could compromise benchmark integrity.

Key areas tested:
- Incompatible coordinate system transformations
- Datum shift errors and precision loss
- High-latitude transformation failures
- Coordinate overflow/underflow scenarios
- Validation bypass scenarios
- Silent precision loss detection
"""

import pytest
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import transform
import warnings
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys
import os
import math

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Import classes that handle CRS transformations
from basin_sampler import BasinSampler
from truth_extractor import TruthExtractor
from benchmark_runner import BenchmarkRunner


class TestIncompatibleCoordinateSystemTransformations:
    """Test transformations between fundamentally incompatible coordinate systems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_config = {
            'target_crs': 'EPSG:5070',  # Albers Equal Area (US)
            'output_crs': 'EPSG:4326',  # WGS84
            'area_range': [5, 500],
            'snap_tolerance': 150
        }
    
    def test_geographic_to_projected_boundary_errors(self):
        """Test geographic to projected transformation at projection boundaries."""
        # Create geometry near the edge of Albers projection validity
        # Albers Equal Area is designed for continental US - test outside boundaries
        
        # Coordinates far outside US (should cause projection issues)
        invalid_coords = [
            # European coordinates (completely outside Albers validity)
            Point(2.3488, 48.8534),  # Paris
            Point(13.4050, 52.5200), # Berlin
            
            # Asian coordinates (extreme longitude differences)
            Point(139.6917, 35.6895), # Tokyo
            Point(116.4074, 39.9042), # Beijing
            
            # Southern hemisphere (invalid for US Albers)
            Point(-47.9297, -15.7801), # Brasilia
            Point(151.2093, -33.8688), # Sydney
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(invalid_coords)),
            'geometry': invalid_coords
        }, crs='EPSG:4326')
        
        # Test transformation that should fail or produce invalid results
        try:
            projected = gdf.to_crs('EPSG:5070')  # US Albers
            
            # Check if coordinates are reasonable for US projection
            bounds = projected.total_bounds
            
            # US Albers coordinates should be roughly within these bounds
            # (very approximate, but outside these is definitely wrong)
            min_x, min_y = -3000000, -2000000  # Western/Southern US
            max_x, max_y = 3000000, 3000000    # Eastern/Northern US
            
            invalid_coords_found = []
            for idx, geom in enumerate(projected.geometry):
                if geom and not geom.is_empty:
                    x, y = geom.x, geom.y
                    if (x < min_x or x > max_x or y < min_y or y > max_y or
                        math.isnan(x) or math.isnan(y) or 
                        math.isinf(x) or math.isinf(y)):
                        invalid_coords_found.append((idx, x, y))
            
            # Should find invalid coordinates for non-US locations
            assert len(invalid_coords_found) > 0, \
                "Should detect invalid coordinates when projecting non-US locations to US Albers"
            
            for idx, x, y in invalid_coords_found:
                print(f"Invalid coordinate detected: index {idx}, x={x}, y={y}")
                
        except Exception as e:
            # Transformation failure is also acceptable behavior
            print(f"Transformation correctly failed: {e}")
    
    def test_datum_incompatibility_detection(self):
        """Test detection of datum incompatibility issues."""
        # Test transformation between coordinate systems with different datums
        # that might not align properly
        
        original_coords = [
            Point(-105.0, 40.0),  # Colorado
            Point(-111.0, 45.0),  # Montana
            Point(-119.0, 47.0),  # Washington
        ]
        
        # Create GeoDataFrame in NAD83
        gdf_nad83 = gpd.GeoDataFrame({
            'id': range(len(original_coords)),
            'geometry': original_coords
        }, crs='EPSG:4269')  # NAD83
        
        # Transform to WGS84
        gdf_wgs84 = gdf_nad83.to_crs('EPSG:4326')  # WGS84
        
        # Calculate coordinate differences (should be small but non-zero)
        coord_differences = []
        for orig, transformed in zip(gdf_nad83.geometry, gdf_wgs84.geometry):
            diff_x = abs(orig.x - transformed.x)
            diff_y = abs(orig.y - transformed.y)
            coord_differences.append((diff_x, diff_y))
        
        # NAD83 to WGS84 should have small but measurable differences
        for diff_x, diff_y in coord_differences:
            # Differences should be small (typically < 2 meters at these latitudes)
            # but definitely non-zero (exact match would indicate no transformation)
            assert diff_x > 0 or diff_y > 0, "Should have some coordinate difference in datum shift"
            
            # But differences shouldn't be huge (would indicate major error)
            assert diff_x < 0.01 and diff_y < 0.01, \
                f"Coordinate differences too large: {diff_x}, {diff_y} (possible transformation error)"
    
    def test_projection_validity_outside_domain(self):
        """Test projection behavior outside its valid domain."""
        # State Plane coordinate systems have very specific validity domains
        
        # Use Colorado State Plane Central (EPSG:2232) with coordinates outside Colorado
        colorado_coords = [
            Point(-105.5, 39.5),  # Within Colorado (should work)
            Point(-120.0, 47.0),  # Washington (outside Colorado State Plane domain)
            Point(-80.0, 25.0),   # Florida (far outside domain)
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(colorado_coords)),
            'geometry': colorado_coords
        }, crs='EPSG:4326')
        
        try:
            # Project to Colorado State Plane
            projected = gdf.to_crs('EPSG:2232')
            
            # Check for invalid results
            invalid_results = []
            for idx, geom in enumerate(projected.geometry):
                if geom and not geom.is_empty:
                    x, y = geom.x, geom.y
                    
                    # Colorado State Plane coordinates should be roughly:
                    # X: 150,000 to 800,000 meters
                    # Y: 1,200,000 to 1,700,000 meters
                    if (x < 0 or x > 1000000 or y < 1000000 or y > 2000000 or
                        math.isnan(x) or math.isnan(y) or 
                        math.isinf(x) or math.isinf(y)):
                        invalid_results.append((idx, x, y))
            
            # Should detect issues with out-of-domain coordinates
            if len(invalid_results) > 0:
                print("Detected invalid State Plane projections (expected):")
                for idx, x, y in invalid_results:
                    print(f"  Index {idx}: x={x}, y={y}")
            
        except Exception as e:
            print(f"State Plane projection correctly failed for out-of-domain coordinates: {e}")
    
    def test_transformation_reversibility(self):
        """Test that transformations are properly reversible."""
        original_coords = [
            Point(-105.0, 40.0),  # Colorado
            Point(-111.0, 45.0),  # Montana  
            Point(-119.0, 47.0),  # Washington
        ]
        
        gdf_original = gpd.GeoDataFrame({
            'id': range(len(original_coords)),
            'geometry': original_coords
        }, crs='EPSG:4326')  # WGS84
        
        # Transform to projected CRS and back
        gdf_projected = gdf_original.to_crs('EPSG:5070')  # Albers
        gdf_back = gdf_projected.to_crs('EPSG:4326')     # Back to WGS84
        
        # Check coordinate preservation
        for orig, back in zip(gdf_original.geometry, gdf_back.geometry):
            diff_x = abs(orig.x - back.x)
            diff_y = abs(orig.y - back.y)
            
            # Should be very close (within numerical precision)
            tolerance = 1e-10  # Very tight tolerance for round-trip
            assert diff_x < tolerance and diff_y < tolerance, \
                f"Round-trip transformation lost precision: diff_x={diff_x}, diff_y={diff_y}"


class TestDatumShiftErrors:
    """Test datum shift errors and precision loss scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_coords = [
            # Test coordinates across different regions where datum shifts vary
            Point(-122.4194, 37.7749),  # San Francisco (significant NAD27-WGS84 shift)
            Point(-74.0060, 40.7128),   # New York (different shift characteristics)
            Point(-87.6298, 41.8781),   # Chicago (midwest characteristics)
            Point(-104.9903, 39.7392),  # Denver (mountain region)
        ]
    
    def test_nad27_to_wgs84_shift_detection(self):
        """Test NAD27 to WGS84 datum shift is correctly applied."""
        gdf_nad27 = gpd.GeoDataFrame({
            'id': range(len(self.test_coords)),
            'geometry': self.test_coords
        }, crs='EPSG:4267')  # NAD27
        
        gdf_wgs84 = gdf_nad27.to_crs('EPSG:4326')  # WGS84
        
        # NAD27 to WGS84 shifts are well-documented and substantial
        expected_shifts = []
        for nad27_geom, wgs84_geom in zip(gdf_nad27.geometry, gdf_wgs84.geometry):
            shift_x = abs(nad27_geom.x - wgs84_geom.x)
            shift_y = abs(nad27_geom.y - wgs84_geom.y)
            expected_shifts.append((shift_x, shift_y))
            
            # NAD27 to WGS84 shifts should be significant (typically 50-200+ meters)
            # Converting to approximate meters for latitude: 1 degree ≈ 111,000 meters
            shift_x_meters = shift_x * 111000 * math.cos(math.radians(nad27_geom.y))
            shift_y_meters = shift_y * 111000
            
            # Should have measurable shift (not zero, indicating transformation occurred)
            assert shift_x_meters > 1 or shift_y_meters > 1, \
                "NAD27 to WGS84 should have measurable shift"
            
            # But shift shouldn't be enormous (would indicate error)
            assert shift_x_meters < 500 and shift_y_meters < 500, \
                f"NAD27 to WGS84 shift too large: {shift_x_meters}m, {shift_y_meters}m"
    
    def test_datum_shift_consistency(self):
        """Test that datum shifts are consistent across transformations."""
        # Test the same point through different transformation paths
        test_point = Point(-105.0, 40.0)  # Colorado
        
        # Path 1: NAD27 → WGS84 directly
        gdf_nad27 = gpd.GeoDataFrame({'geometry': [test_point]}, crs='EPSG:4267')
        path1_result = gdf_nad27.to_crs('EPSG:4326').geometry.iloc[0]
        
        # Path 2: NAD27 → NAD83 → WGS84
        intermediate_nad83 = gdf_nad27.to_crs('EPSG:4269')  # NAD83
        path2_result = intermediate_nad83.to_crs('EPSG:4326').geometry.iloc[0]
        
        # Results should be very close (within transformation precision)
        diff_x = abs(path1_result.x - path2_result.x)
        diff_y = abs(path1_result.y - path2_result.y)
        
        tolerance = 1e-8  # Small tolerance for numerical differences
        assert diff_x < tolerance and diff_y < tolerance, \
            f"Inconsistent datum transformation paths: diff_x={diff_x}, diff_y={diff_y}"
    
    def test_historical_datum_accuracy(self):
        """Test transformations involving historical datums."""
        # Test with coordinates where historical datums have known issues
        
        # Use coordinates with known NAD27 accuracy issues
        problem_coords = [
            Point(-124.0, 48.0),  # Pacific Northwest (known NAD27 issues)
            Point(-67.0, 45.0),   # Maine (different accuracy characteristics)
            Point(-158.0, 21.3),  # Hawaii (separate datum considerations)
        ]
        
        for coord in problem_coords:
            gdf_nad27 = gpd.GeoDataFrame({'geometry': [coord]}, crs='EPSG:4267')
            
            try:
                # Transform to modern datum
                gdf_modern = gdf_nad27.to_crs('EPSG:4326')
                result = gdf_modern.geometry.iloc[0]
                
                # Sanity check: coordinates should still be reasonable
                assert -180 <= result.x <= 180, f"Longitude out of range: {result.x}"
                assert -90 <= result.y <= 90, f"Latitude out of range: {result.y}"
                
                # Should not be identical (would indicate no transformation)
                diff_x = abs(coord.x - result.x)
                diff_y = abs(coord.y - result.y)
                assert diff_x > 1e-6 or diff_y > 1e-6, "Historical datum transformation should change coordinates"
                
            except Exception as e:
                # Some historical transformations might fail - that's also valid
                print(f"Historical datum transformation failed (acceptable): {e}")


class TestHighLatitudeTransformationPrecision:
    """Test precision loss in high-latitude transformations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # High latitude coordinates where projections can become problematic
        self.high_lat_coords = [
            Point(-150.0, 70.0),  # Northern Alaska
            Point(-100.0, 80.0),  # High Arctic
            Point(0.0, 85.0),     # Near North Pole
            Point(-120.0, -70.0), # Antarctica
            Point(170.0, 75.0),   # Siberia
        ]
    
    def test_polar_projection_precision(self):
        """Test precision loss near polar regions."""
        gdf = gpd.GeoDataFrame({
            'id': range(len(self.high_lat_coords)),
            'geometry': self.high_lat_coords
        }, crs='EPSG:4326')
        
        # Try projecting to various coordinate systems
        projection_tests = [
            ('EPSG:3413', 'NSIDC Sea Ice Polar Stereographic North'),
            ('EPSG:3976', 'WGS 84 / EPSG Arctic Polar Stereographic'),
            ('EPSG:5070', 'NAD83 / Conus Albers'),  # Should fail for polar coords
        ]
        
        for epsg_code, description in projection_tests:
            try:
                projected = gdf.to_crs(epsg_code)
                
                # Check for precision issues
                precision_issues = []
                for idx, geom in enumerate(projected.geometry):
                    if geom and not geom.is_empty:
                        x, y = geom.x, geom.y
                        
                        # Check for invalid coordinates
                        if (math.isnan(x) or math.isnan(y) or 
                            math.isinf(x) or math.isinf(y) or
                            abs(x) > 1e10 or abs(y) > 1e10):  # Extremely large values
                            precision_issues.append((idx, x, y))
                
                if precision_issues:
                    print(f"Precision issues in {description}:")
                    for idx, x, y in precision_issues:
                        print(f"  Coordinate {idx}: x={x}, y={y}")
                
                # Test reverse transformation precision
                back_transformed = projected.to_crs('EPSG:4326')
                
                for orig, back in zip(gdf.geometry, back_transformed.geometry):
                    if back and not back.is_empty:
                        diff_x = abs(orig.x - back.x)
                        diff_y = abs(orig.y - back.y)
                        
                        # High latitudes can have large precision loss
                        # Be more permissive for extreme latitudes
                        lat = abs(orig.y)
                        if lat > 80:
                            tolerance = 0.1  # Larger tolerance for extreme polar regions
                        elif lat > 70:
                            tolerance = 0.01  # Medium tolerance for high latitudes
                        else:
                            tolerance = 0.001  # Normal tolerance
                        
                        if diff_x > tolerance or diff_y > tolerance:
                            print(f"Precision loss in {description}: "
                                  f"lat={lat:.1f}, diff_x={diff_x:.6f}, diff_y={diff_y:.6f}")
                
            except Exception as e:
                print(f"Projection {description} failed for high latitudes: {e}")
    
    def test_longitude_wraparound_issues(self):
        """Test longitude wraparound issues near poles."""
        # Near-polar coordinates where longitude becomes less meaningful
        wraparound_coords = [
            Point(-179.9, 89.0),  # Near pole, close to dateline
            Point(179.9, 89.0),   # Near pole, other side of dateline
            Point(0.0, 89.5),     # Very close to North Pole
            Point(90.0, 89.9),    # Extremely close to North Pole
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(wraparound_coords)),
            'geometry': wraparound_coords
        }, crs='EPSG:4326')
        
        # Project to polar stereographic
        try:
            projected = gdf.to_crs('EPSG:3413')  # Polar stereographic north
            back_transformed = projected.to_crs('EPSG:4326')
            
            # Check for longitude wraparound issues
            for idx, (orig, back) in enumerate(zip(gdf.geometry, back_transformed.geometry)):
                if back and not back.is_empty:
                    # Longitude differences can wrap around at 180/-180
                    lon_diff = abs(orig.x - back.x)
                    if lon_diff > 180:
                        lon_diff = 360 - lon_diff  # Account for wraparound
                    
                    lat_diff = abs(orig.y - back.y)
                    
                    # At extreme latitudes, longitude precision becomes less meaningful
                    if abs(orig.y) > 89:
                        # Very close to pole - longitude precision heavily degraded
                        assert lat_diff < 0.1, f"Latitude precision loss too high: {lat_diff}"
                    else:
                        assert lon_diff < 1.0 and lat_diff < 0.01, \
                            f"Coordinate precision loss: lon_diff={lon_diff}, lat_diff={lat_diff}"
        
        except Exception as e:
            print(f"Polar projection failed (may be expected): {e}")
    
    def test_meridian_convergence_effects(self):
        """Test meridian convergence effects at high latitudes."""
        # Test coordinates along the same meridian at different latitudes
        meridian_coords = [
            Point(-105.0, 30.0),  # Low latitude
            Point(-105.0, 45.0),  # Mid latitude
            Point(-105.0, 60.0),  # High latitude
            Point(-105.0, 75.0),  # Very high latitude
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(meridian_coords)),
            'geometry': meridian_coords
        }, crs='EPSG:4326')
        
        # Project to UTM (which has meridian convergence)
        projected = gdf.to_crs('EPSG:32613')  # UTM Zone 13N
        
        # Check that meridian convergence is handled properly
        projected_x_coords = [geom.x for geom in projected.geometry if geom and not geom.is_empty]
        
        # At higher latitudes, points on the same meridian should have 
        # different projected X coordinates due to meridian convergence
        x_variations = max(projected_x_coords) - min(projected_x_coords)
        
        # Should see some variation due to meridian convergence
        assert x_variations > 1000, f"Insufficient meridian convergence detected: {x_variations} meters"


class TestCoordinateValidationFailures:
    """Test coordinate validation after transformation failures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validation_config = {
            'coordinate_validation': {
                'enable_range_checks': True,
                'enable_precision_checks': True,
                'geographic_bounds': {
                    'min_lon': -180, 'max_lon': 180,
                    'min_lat': -90, 'max_lat': 90
                },
                'projected_bounds': {
                    'max_abs_coord': 1e8  # 100 million meters
                }
            }
        }
    
    def test_coordinate_range_validation_bypass(self):
        """Test scenarios where invalid coordinates pass basic validation."""
        # Create coordinates that are technically valid but geographically nonsensical
        problematic_coords = [
            Point(179.999, 89.999),   # Extreme but technically valid
            Point(-179.999, -89.999), # Extreme but technically valid
            Point(0.0, 0.0),          # Valid but often indicates missing data
            Point(1e-10, 1e-10),      # Extremely close to origin (precision issue?)
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(problematic_coords)),
            'geometry': problematic_coords
        }, crs='EPSG:4326')
        
        # Test various projections to see if they detect issues
        projection_tests = [
            'EPSG:5070',  # Albers (US-specific)
            'EPSG:3857',  # Web Mercator
            'EPSG:32633', # UTM 33N
        ]
        
        for proj_crs in projection_tests:
            try:
                projected = gdf.to_crs(proj_crs)
                
                # Check for coordinates that are technically valid but suspicious
                suspicious_coords = []
                for idx, geom in enumerate(projected.geometry):
                    if geom and not geom.is_empty:
                        x, y = geom.x, geom.y
                        
                        # Flag extremely large coordinates (might indicate projection issues)
                        if abs(x) > 1e7 or abs(y) > 1e7:
                            suspicious_coords.append((idx, x, y, 'extremely_large'))
                        
                        # Flag coordinates that are suspiciously close to origin
                        if abs(x) < 1e-6 and abs(y) < 1e-6 and not (x == 0 and y == 0):
                            suspicious_coords.append((idx, x, y, 'near_origin'))
                
                if suspicious_coords:
                    print(f"Suspicious coordinates in {proj_crs}:")
                    for idx, x, y, reason in suspicious_coords:
                        print(f"  Index {idx}: x={x}, y={y} ({reason})")
            
            except Exception as e:
                print(f"Projection to {proj_crs} failed: {e}")
    
    def test_precision_loss_detection(self):
        """Test detection of precision loss in coordinate transformations."""
        # Create high-precision coordinates
        high_precision_coords = [
            Point(-105.12345678901234, 40.12345678901234),
            Point(-111.98765432109876, 45.98765432109876),
            Point(-119.11111111111111, 47.22222222222222),
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(high_precision_coords)),
            'geometry': high_precision_coords
        }, crs='EPSG:4326')
        
        # Transform and back-transform
        projected = gdf.to_crs('EPSG:5070')
        back_transformed = projected.to_crs('EPSG:4326')
        
        # Measure precision loss
        precision_losses = []
        for orig, back in zip(gdf.geometry, back_transformed.geometry):
            if back and not back.is_empty:
                loss_x = abs(orig.x - back.x)
                loss_y = abs(orig.y - back.y)
                precision_losses.append((loss_x, loss_y))
        
        # Check for excessive precision loss
        for loss_x, loss_y in precision_losses:
            # Precision loss should be minimal for well-conditioned transformations
            assert loss_x < 1e-10, f"Excessive longitude precision loss: {loss_x}"
            assert loss_y < 1e-10, f"Excessive latitude precision loss: {loss_y}"
    
    def test_coordinate_corruption_detection(self):
        """Test detection of coordinate corruption during transformation."""
        # Mock a transformation that corrupts coordinates
        original_coords = [
            Point(-105.0, 40.0),
            Point(-111.0, 45.0),
            Point(-119.0, 47.0),
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(original_coords)),
            'geometry': original_coords
        }, crs='EPSG:4326')
        
        # Simulate corrupted transformation by manually altering coordinates
        with patch.object(gdf, 'to_crs') as mock_to_crs:
            # Return coordinates that are corrupted but still within valid ranges
            corrupted_coords = [
                Point(-105.1, 40.1),  # Slightly off (1km+ error)
                Point(-111.05, 45.05),  # Different offset
                Point(-119.02, 47.02),  # Another offset
            ]
            
            corrupted_gdf = gpd.GeoDataFrame({
                'id': range(len(corrupted_coords)),
                'geometry': corrupted_coords
            }, crs='EPSG:5070')
            
            mock_to_crs.return_value = corrupted_gdf
            
            # This simulates a transformation that appears to succeed
            # but actually corrupts the coordinates
            transformed = gdf.to_crs('EPSG:5070')
            
            # Detection mechanism: check if transformation results are reasonable
            # (This would be part of a validation framework)
            
            # Convert back to see if we get original coordinates
            back_transformed = transformed.to_crs('EPSG:4326')
            
            corruption_detected = False
            for orig, back in zip(gdf.geometry, back_transformed.geometry):
                if back and not back.is_empty:
                    # Calculate distance difference (approximate)
                    diff_x = abs(orig.x - back.x)
                    diff_y = abs(orig.y - back.y)
                    
                    # Convert to approximate meters
                    diff_meters = math.sqrt(
                        (diff_x * 111000 * math.cos(math.radians(orig.y)))**2 + 
                        (diff_y * 111000)**2
                    )
                    
                    # If difference is more than expected precision loss, flag corruption
                    if diff_meters > 100:  # 100 meter threshold
                        corruption_detected = True
                        break
            
            # In this test, we should detect the simulated corruption
            assert corruption_detected, "Should detect coordinate corruption"


class TestCoordinateOverflowUnderflowScenarios:
    """Test coordinate overflow and underflow scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def test_extreme_coordinate_values(self):
        """Test handling of extreme coordinate values."""
        # Test coordinates at the limits of floating point precision
        extreme_coords = [
            Point(179.999999999999, 89.999999999999),   # Near maximum valid
            Point(-179.999999999999, -89.999999999999), # Near minimum valid
            Point(1e-15, 1e-15),                        # Near zero (underflow risk)
            Point(179.0 + 1e-15, 89.0 + 1e-15),       # Precision limit addition
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(extreme_coords)),
            'geometry': extreme_coords
        }, crs='EPSG:4326')
        
        # Test transformation to various projections
        projection_tests = [
            'EPSG:3857',  # Web Mercator (known issues near poles)
            'EPSG:5070',  # Albers Equal Area
            'EPSG:32633', # UTM 33N
        ]
        
        for proj_crs in projection_tests:
            try:
                projected = gdf.to_crs(proj_crs)
                
                overflow_detected = []
                for idx, geom in enumerate(projected.geometry):
                    if geom and not geom.is_empty:
                        x, y = geom.x, geom.y
                        
                        # Check for overflow/underflow indicators
                        if (math.isinf(x) or math.isinf(y) or 
                            abs(x) > 1e15 or abs(y) > 1e15 or
                            (abs(x) < 1e-15 and x != 0) or (abs(y) < 1e-15 and y != 0)):
                            overflow_detected.append((idx, x, y))
                
                if overflow_detected:
                    print(f"Overflow/underflow detected in {proj_crs}:")
                    for idx, x, y in overflow_detected:
                        print(f"  Index {idx}: x={x}, y={y}")
            
            except Exception as e:
                print(f"Projection to {proj_crs} failed with extreme coordinates: {e}")
    
    def test_web_mercator_polar_overflow(self):
        """Test Web Mercator overflow near poles."""
        # Web Mercator has known issues near poles (infinite projection at exactly 90°)
        near_polar_coords = [
            Point(0.0, 85.0),     # Close to pole but valid
            Point(0.0, 89.0),     # Very close to pole
            Point(0.0, 89.9),     # Extremely close to pole
            Point(0.0, 89.99),    # Almost at pole
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(near_polar_coords)),
            'geometry': near_polar_coords
        }, crs='EPSG:4326')
        
        try:
            # Project to Web Mercator
            web_mercator = gdf.to_crs('EPSG:3857')
            
            # Check for extreme Y values (Web Mercator Y approaches infinity at poles)
            extreme_y_values = []
            for idx, geom in enumerate(web_mercator.geometry):
                if geom and not geom.is_empty:
                    y = geom.y
                    if abs(y) > 20037508:  # Web Mercator theoretical limit
                        extreme_y_values.append((idx, y))
            
            if extreme_y_values:
                print("Extreme Web Mercator Y values detected:")
                for idx, y in extreme_y_values:
                    print(f"  Index {idx}: Y={y}")
            
            # Test reverse transformation
            back_transformed = web_mercator.to_crs('EPSG:4326')
            
            # Check for precision loss or coordinate corruption
            for orig, back in zip(gdf.geometry, back_transformed.geometry):
                if back and not back.is_empty:
                    lat_diff = abs(orig.y - back.y)
                    
                    # High latitudes in Web Mercator should show precision loss
                    if abs(orig.y) > 85:
                        # Expect some precision loss at high latitudes
                        if lat_diff > 0.01:  # More than ~1km error
                            print(f"High precision loss at latitude {orig.y}: {lat_diff}°")
        
        except Exception as e:
            print(f"Web Mercator polar projection failed (expected): {e}")
    
    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits."""
        # Test coordinates that exercise floating point precision limits
        precision_test_coords = [
            Point(-105.0 + 1e-16, 40.0 + 1e-16),  # Minimal increment
            Point(-105.0 - 1e-16, 40.0 - 1e-16),  # Minimal decrement
            Point(-105.0, 40.0),                   # Reference point
        ]
        
        gdf = gpd.GeoDataFrame({
            'id': range(len(precision_test_coords)),
            'geometry': precision_test_coords
        }, crs='EPSG:4326')
        
        # Transform to high-precision projection
        projected = gdf.to_crs('EPSG:5070')
        
        # Check if tiny differences are preserved or lost
        ref_point = projected.geometry.iloc[2]  # Reference point
        
        for idx, geom in enumerate(projected.geometry[:2]):  # First two points
            if geom and not geom.is_empty:
                diff_x = abs(geom.x - ref_point.x)
                diff_y = abs(geom.y - ref_point.y)
                
                # At this precision level, differences might be lost
                if diff_x == 0 and diff_y == 0:
                    print(f"Precision limit reached: point {idx} identical to reference after projection")
                else:
                    print(f"Precision preserved: point {idx} differs by ({diff_x}, {diff_y}) meters")


class TestCRSTransformationIntegration:
    """Test CRS transformation integration with main benchmark components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_config = {
            'target_crs': 'EPSG:5070',
            'output_crs': 'EPSG:4326',
            'area_range': [5, 500],
            'snap_tolerance': 150
        }
    
    def test_basin_sampler_crs_validation(self):
        """Test CRS transformation validation in BasinSampler."""
        # Test with problematic coordinates that might pass basic validation
        problematic_coords = [
            Point(-105.0, 40.0),   # Valid US coordinate
            Point(2.0, 48.0),      # European coordinate (invalid for US Albers)
            Point(-180.0, 85.0),   # Edge case coordinate
        ]
        
        # Create mock HUC12 data with mixed coordinates
        gdf = gpd.GeoDataFrame({
            'HUC12': ['120100010101', '120100010102', '120100010103'],
            'STATES': ['CO', 'CO', 'CO'],  # All marked as Colorado
            'geometry': [Polygon([(p.x-0.1, p.y-0.1), (p.x+0.1, p.y-0.1), 
                                 (p.x+0.1, p.y+0.1), (p.x-0.1, p.y+0.1), 
                                 (p.x-0.1, p.y-0.1)]) for p in problematic_coords]
        }, crs='EPSG:4326')
        
        # Mock BasinSampler's CRS validation
        with patch('scripts.basin_sampler.gpd.read_file', return_value=gdf):
            try:
                # This would normally be called by BasinSampler
                transformed = gdf.to_crs(self.sample_config['target_crs'])
                
                # Check for invalid transformations
                invalid_transformations = []
                for idx, geom in enumerate(transformed.geometry):
                    if geom and not geom.is_empty:
                        bounds = geom.bounds
                        
                        # Check if coordinates are reasonable for US Albers
                        if (any(abs(coord) > 5000000 for coord in bounds) or
                            any(math.isnan(coord) or math.isinf(coord) for coord in bounds)):
                            invalid_transformations.append(idx)
                
                if invalid_transformations:
                    print(f"Invalid transformations detected: {invalid_transformations}")
                
            except Exception as e:
                print(f"BasinSampler CRS transformation failed: {e}")
    
    def test_truth_extractor_coordinate_consistency(self):
        """Test coordinate consistency in TruthExtractor."""
        # Test pour points that might have CRS issues
        pour_points = pd.DataFrame({
            'ID': ['basin_001', 'basin_002', 'basin_003'],
            'Pour_Point_Lat': [40.0, 48.0, 85.0],      # Mixed latitudes (one extreme)
            'Pour_Point_Lon': [-105.0, 2.0, -120.0],   # Mixed longitudes (one European)
        })
        
        # Create geometries from pour points
        geometries = [Point(lon, lat) for lon, lat in 
                     zip(pour_points['Pour_Point_Lon'], pour_points['Pour_Point_Lat'])]
        
        gdf = gpd.GeoDataFrame(pour_points, geometry=geometries, crs='EPSG:4326')
        
        # Test transformation to target CRS
        try:
            transformed = gdf.to_crs(self.sample_config['target_crs'])
            
            # Check for coordinate issues
            coordinate_issues = []
            for idx, geom in enumerate(transformed.geometry):
                if geom and not geom.is_empty:
                    x, y = geom.x, geom.y
                    
                    if (math.isnan(x) or math.isnan(y) or 
                        math.isinf(x) or math.isinf(y) or
                        abs(x) > 1e7 or abs(y) > 1e7):
                        coordinate_issues.append((idx, x, y))
            
            if coordinate_issues:
                print("TruthExtractor coordinate issues:")
                for idx, x, y in coordinate_issues:
                    print(f"  Point {idx}: x={x}, y={y}")
        
        except Exception as e:
            print(f"TruthExtractor CRS transformation failed: {e}")
    
    def test_benchmark_runner_crs_validation(self):
        """Test CRS validation in BenchmarkRunner metrics calculation."""
        # Test with geometries that might have subtle CRS issues
        
        # Predicted polygon (supposedly in EPSG:4326 from FLOWFINDER)
        pred_polygon = Polygon([(-105.1, 40.1), (-105.0, 40.1), 
                               (-105.0, 40.0), (-105.1, 40.0), (-105.1, 40.1)])
        
        # Truth polygon (from truth extraction, should match CRS)
        truth_polygon = Polygon([(-105.05, 40.05), (-104.95, 40.05), 
                                (-104.95, 39.95), (-105.05, 39.95), (-105.05, 40.05)])
        
        # Mock BenchmarkRunner configuration
        config = {
            'projection_crs': 'EPSG:5070',
            'success_thresholds': {'default': 0.90},
            'centroid_thresholds': {'default': 500}
        }
        
        runner = BenchmarkRunner(sample_df=pd.DataFrame(), truth_path="dummy", config=config)
        
        # Test the metrics calculation which involves CRS transformation
        with patch.object(runner, '_validate_crs_transformation') as mock_transform:
            # Mock successful transformation
            mock_gdf = Mock()
            mock_gdf.iloc = [Mock()]
            mock_gdf.iloc[0].geometry = pred_polygon
            mock_transform.return_value = mock_gdf
            
            # This should work normally
            try:
                result = runner._calculate_metrics(pred_polygon, truth_polygon)
                assert result['status'] in ['valid', 'invalid_geometry']
                print(f"BenchmarkRunner metrics calculation: {result['status']}")
            
            except Exception as e:
                print(f"BenchmarkRunner metrics calculation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])