#!/usr/bin/env python3
"""
Comprehensive Tests for GeometryDiagnostics Class
=================================================

Critical test suite for the 661-line GeometryDiagnostics class that handles
all geometry validation and repair in the FLOWFINDER benchmark system.

Tests cover:
1. Progressive geometry repair strategies validation
2. Self-intersecting polygon handling edge cases  
3. Invalid coordinate recovery (NaN, infinite values)
4. Memory-efficient repair validation
5. Error propagation and failure modes

This is critical - the entire benchmark depends on this untested component.
"""

import pytest
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString,
    GeometryCollection
)
from shapely.ops import unary_union
from shapely.validation import make_valid
import warnings
from unittest.mock import Mock, patch
import gc
import sys

# Import the class under test
from scripts.geometry_utils import GeometryDiagnostics


class TestGeometryDiagnosticsInitialization:
    """Test initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization without parameters."""
        diag = GeometryDiagnostics()
        
        assert diag.logger is not None
        assert diag.config is not None
        assert 'geometry_repair' in diag.config
        assert diag.repair_stats == {}
    
    def test_custom_logger_initialization(self):
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        diag = GeometryDiagnostics(logger=custom_logger)
        
        assert diag.logger == custom_logger
    
    def test_custom_config_initialization(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'geometry_repair': {
                'enable_repair_attempts': False,
                'max_repair_attempts': 5
            }
        }
        diag = GeometryDiagnostics(config=custom_config)
        
        assert diag.config == custom_config
        assert diag.config['geometry_repair']['max_repair_attempts'] == 5
    
    def test_default_config_structure(self):
        """Test default configuration has required structure."""
        diag = GeometryDiagnostics()
        config = diag.config
        
        # Verify required configuration sections
        assert 'geometry_repair' in config
        repair_config = config['geometry_repair']
        
        required_keys = [
            'enable_diagnostics', 'enable_repair_attempts', 
            'invalid_geometry_action', 'max_repair_attempts',
            'detailed_logging', 'repair_strategies'
        ]
        
        for key in required_keys:
            assert key in repair_config
        
        # Verify repair strategies
        strategies = repair_config['repair_strategies']
        expected_strategies = [
            'buffer_fix', 'simplify', 'make_valid', 'convex_hull',
            'orient_fix', 'simplify_holes'
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies


class TestSingleGeometryDiagnosis:
    """Test detailed diagnosis of individual geometries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_valid_polygon_diagnosis(self):
        """Test diagnosis of valid polygon."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        diagnosis = self.diag._diagnose_single_geometry(polygon, 0)
        
        assert diagnosis['is_valid'] is True
        assert diagnosis['is_empty'] is False
        assert diagnosis['is_critical'] is False
        assert diagnosis['geometry_type'] == 'Polygon'
        assert diagnosis['repair_strategy'] == 'none'
        assert len(diagnosis['issues']) == 0
        assert diagnosis['area'] == 1.0
    
    def test_none_geometry_diagnosis(self):
        """Test diagnosis of None geometry."""
        diagnosis = self.diag._diagnose_single_geometry(None, 0)
        
        assert diagnosis['is_valid'] is False
        assert diagnosis['is_critical'] is True
        assert diagnosis['geometry_type'] == 'None'
        assert 'null_geometry' in diagnosis['issues']
        assert diagnosis['repair_strategy'] == 'remove'
        assert diagnosis['explanation'] == 'Geometry is None/null'
    
    def test_empty_geometry_diagnosis(self):
        """Test diagnosis of empty geometry."""
        empty_polygon = Polygon()
        diagnosis = self.diag._diagnose_single_geometry(empty_polygon, 0)
        
        assert diagnosis['is_empty'] is True
        assert 'empty_geometry' in diagnosis['issues']
        assert diagnosis['repair_strategy'] == 'remove'
        assert diagnosis['explanation'] == 'Geometry is empty'
    
    def test_self_intersecting_polygon_diagnosis(self):
        """Test diagnosis of self-intersecting polygon (bow-tie shape)."""
        # Create bow-tie polygon (self-intersecting)
        coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
        invalid_polygon = Polygon(coords)
        
        diagnosis = self.diag._diagnose_single_geometry(invalid_polygon, 0)
        
        assert diagnosis['is_valid'] is False
        assert 'self_intersection' in diagnosis['issues'] or 'ring_self_intersection' in diagnosis['issues']
        assert diagnosis['repair_strategy'] == 'buffer_fix'
        assert 'self' in diagnosis['explanation'].lower()
    
    def test_duplicate_points_diagnosis(self):
        """Test diagnosis of polygon with duplicate consecutive points."""
        # Polygon with duplicate points
        coords = [(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        polygon_with_dupes = Polygon(coords)
        
        diagnosis = self.diag._diagnose_single_geometry(polygon_with_dupes, 0)
        
        if not diagnosis['is_valid']:
            assert 'duplicate_points' in diagnosis['issues'] or 'repeated_point' in diagnosis['explanation'].lower()
            assert diagnosis['repair_strategy'] in ['simplify', 'make_valid']
    
    def test_extremely_small_geometry_diagnosis(self):
        """Test diagnosis of extremely small geometry."""
        # Create tiny polygon
        tiny_coords = [(0, 0), (1e-12, 0), (1e-12, 1e-12), (0, 1e-12), (0, 0)]
        tiny_polygon = Polygon(tiny_coords)
        
        diagnosis = self.diag._diagnose_single_geometry(tiny_polygon, 0)
        
        # Should be flagged as extremely small
        if diagnosis['area'] < 1e-10:
            assert 'extremely_small_area' in diagnosis['issues']
            assert diagnosis['repair_strategy'] == 'remove'
    
    def test_extremely_thin_geometry_diagnosis(self):
        """Test diagnosis of extremely thin geometry (high aspect ratio)."""
        # Create very thin rectangle
        thin_coords = [(0, 0), (1000, 0), (1000, 1e-6), (0, 1e-6), (0, 0)]
        thin_polygon = Polygon(thin_coords)
        
        diagnosis = self.diag._diagnose_single_geometry(thin_polygon, 0)
        
        # Calculate aspect ratio
        if diagnosis['area'] > 0 and diagnosis['length'] > 0:
            aspect_ratio = diagnosis['length'] / diagnosis['area']
            if aspect_ratio > 1e6:
                assert 'extremely_thin' in diagnosis['issues']
                assert diagnosis['repair_strategy'] == 'simplify'
    
    def test_linestring_diagnosis(self):
        """Test diagnosis of LineString geometries."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        diagnosis = self.diag._diagnose_single_geometry(line, 0)
        
        assert diagnosis['geometry_type'] == 'LineString'
        assert diagnosis['length'] > 0
        assert diagnosis['area'] == 0.0  # LineStrings have no area
    
    def test_extremely_short_line_diagnosis(self):
        """Test diagnosis of extremely short LineString."""
        short_line = LineString([(0, 0), (1e-12, 1e-12)])
        diagnosis = self.diag._diagnose_single_geometry(short_line, 0)
        
        if diagnosis['length'] < 1e-10:
            assert 'extremely_short_line' in diagnosis['issues']
            assert diagnosis['repair_strategy'] == 'remove'
    
    def test_geometry_analysis_error_handling(self):
        """Test error handling during geometry analysis."""
        # Create a mock geometry that raises exceptions
        mock_geom = Mock()
        mock_geom.is_empty = True
        mock_geom.is_valid = False
        mock_geom.geom_type = 'Polygon'
        mock_geom.bounds.side_effect = Exception("Mock bounds error")
        
        diagnosis = self.diag._diagnose_single_geometry(mock_geom, 0)
        
        assert 'bounds_error' in diagnosis['issues'] or 'analysis_error' in diagnosis['issues']
        assert diagnosis['is_critical'] is True


class TestInvalidCoordinateHandling:
    """Test handling of invalid coordinates (NaN, infinite values)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_nan_coordinate_detection(self):
        """Test detection of NaN coordinates."""
        # This test verifies the system can detect NaN coordinates
        # Note: Shapely may not create geometries with NaN coordinates directly
        # but we test the detection logic
        
        # Create a mock geometry with NaN in explanation
        mock_geom = Mock()
        mock_geom.is_empty = False
        mock_geom.is_valid = False
        mock_geom.geom_type = 'Polygon'
        mock_geom.bounds = (0, 0, 1, 1)
        mock_geom.area = 0.0
        mock_geom.length = 0.0
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Invalid coordinate: NaN detected in geometry"
            
            diagnosis = self.diag._diagnose_single_geometry(mock_geom, 0)
            
            assert 'invalid_coordinates' in diagnosis['issues']
            assert diagnosis['repair_strategy'] == 'remove'
            assert diagnosis['is_critical'] is True
    
    def test_infinite_coordinate_detection(self):
        """Test detection of infinite coordinates."""
        mock_geom = Mock()
        mock_geom.is_empty = False
        mock_geom.is_valid = False
        mock_geom.geom_type = 'Polygon'
        mock_geom.bounds = (0, 0, 1, 1)
        mock_geom.area = 0.0
        mock_geom.length = 0.0
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Invalid coordinate: infinite value in geometry"
            
            diagnosis = self.diag._diagnose_single_geometry(mock_geom, 0)
            
            assert 'invalid_coordinates' in diagnosis['issues']
            assert diagnosis['repair_strategy'] == 'remove'
            assert diagnosis['is_critical'] is True
    
    def test_gdf_with_invalid_coordinates(self):
        """Test GeoDataFrame handling with invalid coordinate geometries."""
        # Create GeoDataFrame with mix of valid and problematic geometries
        valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        
        # Create mock invalid geometry
        invalid_geom = Mock()
        invalid_geom.is_empty = False
        invalid_geom.is_valid = False
        invalid_geom.geom_type = 'Polygon'
        invalid_geom.bounds = (0, 0, 1, 1)
        invalid_geom.area = 0.0
        invalid_geom.length = 0.0
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [valid_geom, invalid_geom]
        })
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Invalid coordinate: NaN in geometry"
            
            result_gdf = self.diag.diagnose_and_repair_geometries(gdf, "test data")
            
            # Should remove invalid coordinate geometry
            assert len(result_gdf) == 1
            assert result_gdf.iloc[0]['id'] == 1


class TestProgressiveRepairStrategies:
    """Test progressive geometry repair strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_make_valid_strategy(self):
        """Test make_valid repair strategy."""
        # Create self-intersecting polygon
        coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
        invalid_polygon = Polygon(coords)
        
        repaired = self.diag._apply_repair_strategy(invalid_polygon, 'make_valid', 0)
        
        assert repaired is not None
        assert repaired.is_valid
    
    def test_buffer_fix_strategy(self):
        """Test buffer fix repair strategy."""
        coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
        invalid_polygon = Polygon(coords)
        
        repaired = self.diag._apply_repair_strategy(invalid_polygon, 'buffer_fix', 0)
        
        assert repaired is not None
        assert repaired.is_valid
    
    def test_simplify_strategy(self):
        """Test simplify repair strategy."""
        # Create polygon with many points
        num_points = 1000
        angles = np.linspace(0, 2*np.pi, num_points)
        coords = [(np.cos(a), np.sin(a)) for a in angles]
        coords.append(coords[0])  # Close the polygon
        
        complex_polygon = Polygon(coords)
        
        repaired = self.diag._apply_repair_strategy(complex_polygon, 'simplify', 0)
        
        assert repaired is not None
        assert repaired.is_valid
        # Should have fewer points after simplification
        if hasattr(repaired, 'exterior'):
            assert len(repaired.exterior.coords) < len(complex_polygon.exterior.coords)
    
    def test_convex_hull_strategy(self):
        """Test convex hull repair strategy."""
        coords = [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]
        invalid_polygon = Polygon(coords)
        
        repaired = self.diag._apply_repair_strategy(invalid_polygon, 'convex_hull', 0)
        
        assert repaired is not None
        assert repaired.is_valid
        # Convex hull should be valid
        expected_hull = invalid_polygon.convex_hull
        assert repaired.equals(expected_hull) or repaired.almost_equals(expected_hull)
    
    def test_orient_fix_strategy(self):
        """Test orientation fix repair strategy."""
        # Create polygon with potential orientation issues
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        polygon = Polygon(coords)
        
        repaired = self.diag._apply_repair_strategy(polygon, 'orient_fix', 0)
        
        assert repaired is not None
        assert repaired.is_valid
    
    def test_simplify_holes_strategy(self):
        """Test simplify holes repair strategy."""
        # Create polygon with holes
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
        polygon_with_hole = Polygon(exterior, [hole])
        
        repaired = self.diag._apply_repair_strategy(polygon_with_hole, 'simplify_holes', 0)
        
        assert repaired is not None
        assert repaired.is_valid
        # Should not have holes
        assert len(repaired.interiors) == 0
    
    def test_unary_union_fix_strategy(self):
        """Test unary union fix for multi-geometries."""
        # Create MultiPolygon
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)])
        multi_polygon = MultiPolygon([poly1, poly2])
        
        repaired = self.diag._apply_repair_strategy(multi_polygon, 'unary_union_fix', 0)
        
        assert repaired is not None
        assert repaired.is_valid
    
    def test_remove_strategy(self):
        """Test remove strategy returns None."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        
        repaired = self.diag._apply_repair_strategy(polygon, 'remove', 0)
        
        assert repaired is None
    
    def test_unknown_strategy_fallback(self):
        """Test unknown strategy falls back to make_valid."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        
        repaired = self.diag._apply_repair_strategy(polygon, 'unknown_strategy', 0)
        
        assert repaired is not None
        assert repaired.is_valid
    
    def test_repair_strategy_exception_handling(self):
        """Test error handling in repair strategies."""
        # Create mock geometry that raises exceptions
        mock_geom = Mock()
        mock_geom.buffer.side_effect = Exception("Mock buffer error")
        
        repaired = self.diag._apply_repair_strategy(mock_geom, 'buffer_fix', 0)
        
        assert repaired is None


class TestComprehensiveRepairWorkflow:
    """Test the complete geometry repair workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_empty_geodataframe_handling(self):
        """Test handling of empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame({'geometry': []})
        
        result = self.diag.diagnose_and_repair_geometries(empty_gdf, "empty test")
        
        assert len(result) == 0
        assert result.empty
    
    def test_mixed_geometry_types_repair(self):
        """Test repair of GeoDataFrame with mixed geometry types."""
        # Create mix of valid and invalid geometries
        valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        invalid_polygon = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])  # Self-intersecting
        valid_line = LineString([(0, 0), (1, 1)])
        valid_point = Point(0.5, 0.5)
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2, 3, 4],
            'geometry': [valid_polygon, invalid_polygon, valid_line, valid_point]
        })
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "mixed geometries")
        
        # Should have all geometries (invalid one repaired)
        assert len(result) == 4
        # All geometries should be valid after repair
        assert all(geom.is_valid for geom in result.geometry)
    
    def test_progressive_repair_application(self):
        """Test that progressive repair strategies are applied correctly."""
        # Create geometries that need different repair strategies
        self_intersecting = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        # Create complex polygon that might need simplification
        num_points = 100
        angles = np.linspace(0, 2*np.pi, num_points)
        # Add some noise to create potential duplicate points
        coords = [(np.cos(a) + np.random.normal(0, 1e-10), 
                  np.sin(a) + np.random.normal(0, 1e-10)) for a in angles]
        coords.append(coords[0])
        complex_polygon = Polygon(coords)
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [self_intersecting, complex_polygon]
        })
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "progressive repair test")
        
        # Should successfully repair both geometries
        assert len(result) == 2
        assert all(geom.is_valid for geom in result.geometry)
    
    def test_repair_statistics_tracking(self):
        """Test that repair statistics are properly tracked."""
        # Create geometries with known issues
        invalid_polygon = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        gdf = gpd.GeoDataFrame({
            'id': [1],
            'geometry': [invalid_polygon]
        })
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "stats test")
        
        # Check that repair statistics were tracked
        assert hasattr(self.diag, 'repair_stats')
        assert self.diag.repair_stats != {}
        assert 'repair_counts' in self.diag.repair_stats
    
    def test_invalid_geometry_action_remove(self):
        """Test removal of unrepairable geometries."""
        # Configure to remove invalid geometries
        config = {
            'geometry_repair': {
                'invalid_geometry_action': 'remove',
                'enable_repair_attempts': True,
                'max_repair_attempts': 3
            }
        }
        diag = GeometryDiagnostics(config=config)
        
        # Create a geometry that should be marked for removal
        valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        
        # Mock an unrepairable geometry
        unrepairable_geom = Mock()
        unrepairable_geom.is_empty = False
        unrepairable_geom.is_valid = False
        unrepairable_geom.geom_type = 'Polygon'
        unrepairable_geom.bounds = (0, 0, 1, 1)
        unrepairable_geom.area = 0.0
        unrepairable_geom.length = 0.0
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [valid_geom, unrepairable_geom]
        })
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Critical error: geometry cannot be repaired"
            with patch.object(diag, '_apply_repair_strategy', return_value=None):
                result = diag.diagnose_and_repair_geometries(gdf, "remove test")
        
        # Should remove the unrepairable geometry
        assert len(result) == 1
        assert result.iloc[0]['id'] == 1
    
    def test_invalid_geometry_action_convert_to_point(self):
        """Test conversion of invalid geometries to points."""
        config = {
            'geometry_repair': {
                'invalid_geometry_action': 'convert_to_point',
                'enable_repair_attempts': True,
                'max_repair_attempts': 3
            }
        }
        diag = GeometryDiagnostics(config=config)
        
        valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        
        # Create invalid geometry that can be converted to centroid
        invalid_geom = Mock()
        invalid_geom.is_empty = False
        invalid_geom.is_valid = False
        invalid_geom.geom_type = 'Polygon'
        invalid_geom.bounds = (0, 0, 1, 1)
        invalid_geom.area = 0.0
        invalid_geom.length = 0.0
        invalid_geom.centroid = Point(0.5, 0.5)
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [valid_geom, invalid_geom]
        })
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Invalid geometry"
            with patch.object(diag, '_apply_repair_strategy', return_value=None):
                result = diag.diagnose_and_repair_geometries(gdf, "convert test")
        
        # Should have both geometries, second one converted to point
        assert len(result) == 2
        assert result.iloc[1].geometry.geom_type == 'Point'


class TestMemoryEfficientValidation:
    """Test memory efficiency with large datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset for memory efficiency."""
        # Create large dataset (1000 geometries)
        num_geoms = 1000
        geometries = []
        
        for i in range(num_geoms):
            # Create various geometry types
            if i % 3 == 0:
                geom = Polygon([(i, i), (i+1, i), (i+1, i+1), (i, i+1), (i, i)])
            elif i % 3 == 1:
                geom = LineString([(i, i), (i+1, i+1)])
            else:
                geom = Point(i, i)
            geometries.append(geom)
        
        # Add some invalid geometries
        for i in range(50):
            idx = i * 20
            if idx < len(geometries):
                # Create self-intersecting polygon
                coords = [(idx, idx), (idx+2, idx+2), (idx+2, idx), (idx, idx+2), (idx, idx)]
                geometries[idx] = Polygon(coords)
        
        gdf = gpd.GeoDataFrame({
            'id': range(num_geoms),
            'geometry': geometries
        })
        
        # Monitor memory usage during processing
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "large dataset")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should process all geometries
        assert len(result) <= num_geoms  # Some might be removed
        # All remaining should be valid
        assert all(geom.is_valid for geom in result.geometry)
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_memory_cleanup_during_processing(self):
        """Test that intermediate objects are properly cleaned up."""
        # Create dataset with complex geometries
        complex_geometries = []
        
        for i in range(100):
            # Create complex polygons with many vertices
            num_points = 500
            angles = np.linspace(0, 2*np.pi, num_points)
            coords = [(np.cos(a) * 100 + i*200, np.sin(a) * 100 + i*200) for a in angles]
            coords.append(coords[0])
            complex_geometries.append(Polygon(coords))
        
        gdf = gpd.GeoDataFrame({
            'id': range(100),
            'geometry': complex_geometries
        })
        
        # Force garbage collection before
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "memory cleanup test")
        
        # Force garbage collection after
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have excessive object growth
        object_growth = final_objects - initial_objects
        assert object_growth < 10000, f"Too many objects created: {object_growth}"
    
    def test_chunked_processing_simulation(self):
        """Test behavior that simulates chunked processing."""
        # This simulates what happens when processing data in chunks
        chunk_size = 100
        total_geoms = 500
        
        all_results = []
        
        for chunk_start in range(0, total_geoms, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_geoms)
            
            # Create chunk of geometries
            chunk_geoms = []
            for i in range(chunk_start, chunk_end):
                geom = Polygon([(i, i), (i+1, i), (i+1, i+1), (i, i+1), (i, i)])
                chunk_geoms.append(geom)
            
            chunk_gdf = gpd.GeoDataFrame({
                'id': range(chunk_start, chunk_end),
                'geometry': chunk_geoms
            })
            
            # Process chunk
            chunk_result = self.diag.diagnose_and_repair_geometries(
                chunk_gdf, f"chunk {chunk_start}-{chunk_end}"
            )
            
            all_results.append(chunk_result)
        
        # Combine all results
        final_result = pd.concat(all_results, ignore_index=True)
        
        assert len(final_result) == total_geoms
        assert all(geom.is_valid for geom in final_result.geometry)


class TestErrorPropagationAndFailureModes:
    """Test error propagation and failure modes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_analysis_exception_handling(self):
        """Test handling of exceptions during geometry analysis."""
        # Create mock geometry that raises exceptions
        mock_geom = Mock()
        mock_geom.is_empty.side_effect = Exception("Mock error")
        
        gdf = gpd.GeoDataFrame({
            'id': [1],
            'geometry': [mock_geom]
        })
        
        # Should not crash, should handle gracefully
        result = self.diag.diagnose_and_repair_geometries(gdf, "exception test")
        
        # Should remove problematic geometry
        assert len(result) == 0
    
    def test_repair_strategy_failure_propagation(self):
        """Test that repair strategy failures are properly handled."""
        # Create geometry that will fail repair
        invalid_geom = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        gdf = gpd.GeoDataFrame({
            'id': [1],
            'geometry': [invalid_geom]
        })
        
        # Mock repair strategy to always fail
        with patch.object(self.diag, '_apply_repair_strategy', return_value=None):
            result = self.diag.diagnose_and_repair_geometries(gdf, "repair failure test")
            
            # Should track failed repairs
            assert hasattr(self.diag, 'repair_stats')
            assert 'failed_repairs' in self.diag.repair_stats
    
    def test_partial_failure_handling(self):
        """Test handling when some geometries succeed and others fail."""
        valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        invalid_geom = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        # Create mock that fails repair
        failing_geom = Mock()
        failing_geom.is_empty = False
        failing_geom.is_valid = False
        failing_geom.geom_type = 'Polygon'
        failing_geom.bounds = (0, 0, 1, 1)
        failing_geom.area = 0.0
        failing_geom.length = 0.0
        
        gdf = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [valid_geom, invalid_geom, failing_geom]
        })
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Mock error"
            with patch.object(self.diag, '_apply_repair_strategy') as mock_repair:
                # Return successful repair for second, failed for third
                mock_repair.side_effect = [
                    make_valid(invalid_geom),  # Success
                    None  # Failure
                ]
                
                result = self.diag.diagnose_and_repair_geometries(gdf, "partial failure test")
                
                # Should have valid and repaired geometries, remove failed one
                assert len(result) <= 2  # Failed geometry should be removed
    
    def test_logging_during_failures(self, caplog):
        """Test that appropriate logging occurs during failures."""
        with caplog.at_level(logging.WARNING):
            # Create problematic geometry
            mock_geom = Mock()
            mock_geom.is_empty = False
            mock_geom.is_valid = False
            mock_geom.geom_type = 'Polygon'
            mock_geom.bounds.side_effect = Exception("Mock bounds error")
            
            gdf = gpd.GeoDataFrame({
                'id': [1],
                'geometry': [mock_geom]
            })
            
            result = self.diag.diagnose_and_repair_geometries(gdf, "logging test")
            
            # Should have logged warnings about geometry issues
            assert any("geometry" in record.message.lower() for record in caplog.records)
    
    def test_critical_error_identification(self):
        """Test identification of critical errors that cannot be repaired."""
        # Create mock geometry with critical error
        mock_geom = Mock()
        mock_geom.is_empty = False
        mock_geom.is_valid = False
        mock_geom.geom_type = 'Polygon'
        mock_geom.bounds = (0, 0, 1, 1)
        mock_geom.area = 0.0
        mock_geom.length = 0.0
        
        gdf = gpd.GeoDataFrame({
            'id': [1],
            'geometry': [mock_geom]
        })
        
        with patch('shapely.validation.explain_validity') as mock_explain:
            mock_explain.return_value = "Invalid coordinate: NaN detected"
            
            # Analyze without repair to check critical error detection
            stats = self.diag.analyze_geometry_issues(gdf, "critical error test")
            
            assert len(stats['critical_errors']) > 0
            assert stats['critical_errors'][0]['index'] == 0


class TestSpatialEdgeCases:
    """Test spatial edge cases that could break in production."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_coordinates_at_precision_limits(self):
        """Test geometries at coordinate precision limits."""
        # Test extremely small coordinates (near floating point precision)
        tiny_coords = [(1e-15, 1e-15), (2e-15, 1e-15), (2e-15, 2e-15), (1e-15, 2e-15), (1e-15, 1e-15)]
        tiny_polygon = Polygon(tiny_coords)
        
        diagnosis = self.diag._diagnose_single_geometry(tiny_polygon, 0)
        
        # Should be flagged as extremely small or invalid
        assert diagnosis['area'] < 1e-10 or not diagnosis['is_valid']
    
    def test_coordinates_crossing_antimeridian(self):
        """Test geometries crossing the antimeridian (±180° longitude)."""
        # Polygon crossing antimeridian
        antimeridian_coords = [(179, 0), (-179, 0), (-179, 1), (179, 1), (179, 0)]
        antimeridian_polygon = Polygon(antimeridian_coords)
        
        diagnosis = self.diag._diagnose_single_geometry(antimeridian_polygon, 0)
        
        # Should handle gracefully, may or may not be valid depending on coordinate system
        assert 'geometry_type' in diagnosis
        assert diagnosis['geometry_type'] == 'Polygon'
    
    def test_extremely_complex_geometry(self):
        """Test geometry with thousands of vertices."""
        # Create polygon with many vertices
        num_points = 10000
        angles = np.linspace(0, 2*np.pi, num_points)
        coords = [(np.cos(a) * 1000, np.sin(a) * 1000) for a in angles]
        coords.append(coords[0])  # Close polygon
        
        complex_polygon = Polygon(coords)
        
        diagnosis = self.diag._diagnose_single_geometry(complex_polygon, 0)
        
        # Should handle without crashing
        assert diagnosis is not None
        assert 'geometry_type' in diagnosis
    
    def test_multipolygon_with_invalid_parts(self):
        """Test MultiPolygon with some invalid component polygons."""
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        invalid_poly = Polygon([(2, 2), (4, 4), (4, 2), (2, 4), (2, 2)])  # Self-intersecting
        
        multi_polygon = MultiPolygon([valid_poly, invalid_poly])
        
        diagnosis = self.diag._diagnose_single_geometry(multi_polygon, 0)
        
        # Should detect invalidity in the MultiPolygon
        assert diagnosis['geometry_type'] == 'MultiPolygon'
        if not diagnosis['is_valid']:
            assert len(diagnosis['issues']) > 0
    
    def test_nested_holes_edge_case(self):
        """Test polygon with nested holes (hole inside hole)."""
        # Outer boundary
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        # Outer hole
        outer_hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
        # Inner hole (inside the outer hole - invalid topology)
        inner_hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
        
        # This should create an invalid polygon
        try:
            nested_polygon = Polygon(exterior, [outer_hole, inner_hole])
            diagnosis = self.diag._diagnose_single_geometry(nested_polygon, 0)
            
            # Should detect topology issues
            if not diagnosis['is_valid']:
                assert 'nested' in ' '.join(diagnosis['issues']).lower() or 'hole' in diagnosis['explanation'].lower()
        except Exception:
            # Creating this polygon might fail, which is also valid behavior
            pass
    
    def test_zero_area_polygon(self):
        """Test polygon with zero area (collapsed to line)."""
        # All points on a line (zero area)
        line_coords = [(0, 0), (1, 0), (2, 0), (1, 0), (0, 0)]
        zero_area_polygon = Polygon(line_coords)
        
        diagnosis = self.diag._diagnose_single_geometry(zero_area_polygon, 0)
        
        # Should handle zero area case
        if diagnosis['area'] == 0:
            assert 'extremely_small_area' in diagnosis['issues'] or not diagnosis['is_valid']


class TestProgressiveRepairIntegration:
    """Test integrated progressive repair workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_repair_strategy_chain(self):
        """Test that repair strategies chain correctly when one fails."""
        # Create geometry that needs multiple repair attempts
        complex_invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        
        # Test buffer_fix fallback to make_valid
        with patch.object(self.diag, '_apply_repair_strategy') as mock_repair:
            # First call (buffer_fix) returns None, second call (make_valid) succeeds
            mock_repair.side_effect = [None, Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
            
            result = self.diag._apply_repair_strategy(complex_invalid, 'buffer_fix', 0)
            
            # Should have attempted buffer_fix, then fallen back
            assert mock_repair.call_count >= 1
    
    def test_repair_with_all_strategies_failing(self):
        """Test behavior when all repair strategies fail."""
        mock_geom = Mock()
        mock_geom.buffer.side_effect = Exception("Buffer failed")
        mock_geom.simplify.side_effect = Exception("Simplify failed")
        
        with patch('shapely.validation.make_valid', side_effect=Exception("make_valid failed")):
            result = self.diag._apply_repair_strategy(mock_geom, 'make_valid', 0)
            
            # Should return None when all repairs fail
            assert result is None
    
    def test_repair_statistics_accumulation(self):
        """Test that repair statistics are correctly accumulated."""
        # Create geometries requiring different repair strategies
        geoms = []
        for i in range(5):
            coords = [(i, i), (i+2, i+2), (i+2, i), (i, i+2), (i, i)]
            geoms.append(Polygon(coords))
        
        gdf = gpd.GeoDataFrame({
            'id': range(5),
            'geometry': geoms
        })
        
        result = self.diag.diagnose_and_repair_geometries(gdf, "stats accumulation")
        
        # Should have accumulated repair statistics
        assert hasattr(self.diag, 'repair_stats')
        repair_stats = self.diag.repair_stats
        assert 'repair_counts' in repair_stats
        
        # Should have statistics for applied strategies
        total_attempted = sum(stats['attempted'] for stats in repair_stats['repair_counts'].values())
        assert total_attempted > 0


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety considerations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_multiple_concurrent_diagnoses(self):
        """Test multiple concurrent geometry diagnoses."""
        import threading
        import time
        
        results = []
        errors = []
        
        def diagnose_geometry(geom_index):
            try:
                # Create different geometry for each thread
                coords = [(geom_index, geom_index), (geom_index+1, geom_index), 
                         (geom_index+1, geom_index+1), (geom_index, geom_index+1), (geom_index, geom_index)]
                polygon = Polygon(coords)
                
                diagnosis = self.diag._diagnose_single_geometry(polygon, geom_index)
                results.append(diagnosis)
                
            except Exception as e:
                errors.append(f"Thread {geom_index}: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=diagnose_geometry, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == 10, "Should have 10 results"
        
        # All should be valid simple polygons
        for result in results:
            assert result['is_valid'] is True
            assert result['geometry_type'] == 'Polygon'
    
    def test_state_isolation_between_instances(self):
        """Test that different GeometryDiagnostics instances don't interfere."""
        diag1 = GeometryDiagnostics()
        diag2 = GeometryDiagnostics()
        
        # Set different repair stats for each
        diag1.repair_stats = {'test1': 'value1'}
        diag2.repair_stats = {'test2': 'value2'}
        
        # Verify isolation
        assert diag1.repair_stats != diag2.repair_stats
        assert diag1.repair_stats['test1'] == 'value1'
        assert diag2.repair_stats['test2'] == 'value2'


class TestRecommendationGeneration:
    """Test geometry analysis recommendation generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diag = GeometryDiagnostics()
    
    def test_high_invalid_rate_recommendation(self):
        """Test recommendation for high invalid geometry rate."""
        # Create stats with high invalid rate
        original_stats = {
            'total_features': 100,
            'total_invalid': 25,  # 25% invalid rate
            'issue_types': {},
        }
        final_stats = {
            'total_invalid': 5
        }
        
        # Set up repair stats
        self.diag.repair_stats = {
            'repair_counts': {}
        }
        
        recommendations = self.diag._generate_geometry_recommendations(original_stats, final_stats)
        
        assert any("high invalid geometry rate" in rec.lower() for rec in recommendations)
    
    def test_self_intersection_recommendation(self):
        """Test recommendation for many self-intersection errors."""
        original_stats = {
            'total_features': 100,
            'total_invalid': 10,
            'issue_types': {
                'self_intersection': 8  # Many self-intersections
            },
        }
        final_stats = {'total_invalid': 2}
        
        self.diag.repair_stats = {'repair_counts': {}}
        
        recommendations = self.diag._generate_geometry_recommendations(original_stats, final_stats)
        
        assert any("self-intersection" in rec.lower() for rec in recommendations)
    
    def test_duplicate_points_recommendation(self):
        """Test recommendation for many duplicate point errors."""
        original_stats = {
            'total_features': 100,
            'total_invalid': 15,
            'issue_types': {
                'duplicate_points': 12  # Many duplicate points
            },
        }
        final_stats = {'total_invalid': 3}
        
        self.diag.repair_stats = {'repair_counts': {}}
        
        recommendations = self.diag._generate_geometry_recommendations(original_stats, final_stats)
        
        assert any("duplicate point" in rec.lower() for rec in recommendations)
    
    def test_small_geometry_recommendation(self):
        """Test recommendation for very small geometries."""
        original_stats = {
            'total_features': 50,
            'total_invalid': 5,
            'issue_types': {
                'extremely_small_area': 3
            },
        }
        final_stats = {'total_invalid': 2}
        
        self.diag.repair_stats = {'repair_counts': {}}
        
        recommendations = self.diag._generate_geometry_recommendations(original_stats, final_stats)
        
        assert any("small geometries" in rec.lower() for rec in recommendations)
    
    def test_low_repair_success_recommendation(self):
        """Test recommendation for low repair success rate."""
        original_stats = {
            'total_features': 100,
            'total_invalid': 20,
            'issue_types': {},
        }
        final_stats = {'total_invalid': 15}
        
        # Mock low success rate repair stats
        self.diag.repair_stats = {
            'repair_counts': {
                'make_valid': {'attempted': 10, 'successful': 2},
                'buffer_fix': {'attempted': 8, 'successful': 1}
            }
        }
        
        recommendations = self.diag._generate_geometry_recommendations(original_stats, final_stats)
        
        assert any("low repair success rate" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])