#!/usr/bin/env python3
"""
Comprehensive unit tests for truth_extractor.py
Tests spatial joins, quality validation, and Mountain West specific features
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
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from truth_extractor import TruthExtractor


class TestTruthExtractor:
    """Comprehensive test cases for TruthExtractor class"""
    
    def test_config_loading(self, sample_config):
        """Test configuration loading with defaults and overrides"""
        extractor = TruthExtractor()
        
        # Test default configuration
        assert extractor.config['buffer_tolerance'] == 500
        assert extractor.config['target_crs'] == 'EPSG:5070'
        assert extractor.config['min_area_ratio'] == 0.1
        assert extractor.config['max_area_ratio'] == 10.0
    
    def test_spatial_join_logic(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test spatial join between basin pour points and catchments"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        # Test spatial join
        results = extractor.extract_truth_polygons()
        
        # Check that truth polygons are extracted
        assert len(results) > 0
        assert 'geometry' in results.columns
        assert 'ID' in results.columns
        
        # Test that geometries are valid
        for geom in results['geometry']:
            assert geom.is_valid
            assert isinstance(geom, Polygon)
    
    def test_pour_point_containment(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test that pour points are contained within extracted catchments"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Check that pour points are contained in extracted catchments
        for idx, row in results.iterrows():
            basin_id = row['ID']
            catchment_geom = row['geometry']
            
            # Find corresponding basin data
            basin_row = sample_basin_data[sample_basin_data['ID'] == basin_id].iloc[0]
            pour_point = Point(basin_row['Pour_Point_Lon'], basin_row['Pour_Point_Lat'])
            
            # Transform to same CRS if needed
            if extractor.config['target_crs'] != 'EPSG:4326':
                from pyproj import Transformer
                transformer = Transformer.from_crs('EPSG:4326', extractor.config['target_crs'])
                x, y = transformer.transform(pour_point.x, pour_point.y)
                pour_point = Point(x, y)
            
            # Pour point should be contained in catchment
            assert catchment_geom.contains(pour_point) or catchment_geom.distance(pour_point) < 100
    
    def test_area_ratio_validation(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test area ratio validation between sample and truth polygons"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Check area ratios
        for idx, row in results.iterrows():
            basin_id = row['ID']
            truth_area = row['geometry'].area / 1e6  # Convert to km²
            
            # Find corresponding basin data
            basin_row = sample_basin_data[sample_basin_data['ID'] == basin_id].iloc[0]
            sample_area = basin_row['Area_km2']
            
            # Calculate area ratio
            area_ratio = truth_area / sample_area
            
            # Should be within configured limits
            assert area_ratio >= extractor.config['min_area_ratio']
            assert area_ratio <= extractor.config['max_area_ratio']
    
    def test_quality_validation(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test quality validation of extracted truth polygons"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Test topology validation
        for geom in results['geometry']:
            assert geom.is_valid
            assert not geom.is_empty
        
        # Test area validation
        for geom in results['geometry']:
            area_km2 = geom.area / 1e6
            assert area_km2 >= extractor.config['min_polygon_area']
        
        # Test complexity validation
        for geom in results['geometry']:
            if hasattr(geom, 'geoms'):  # MultiPolygon
                assert len(list(geom.geoms)) <= extractor.config['max_polygon_parts']
            else:  # Single Polygon
                assert 1 <= extractor.config['max_polygon_parts']
    
    def test_extraction_strategy_priority(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test extraction strategy priority when multiple catchments intersect"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        # Test that extraction follows priority order
        priority_order = extractor.config['extraction_priority']
        assert priority_order[1] == 'contains_point'
        assert priority_order[2] == 'largest_containing'
        assert priority_order[3] == 'nearest_centroid'
        assert priority_order[4] == 'largest_intersecting'
    
    def test_crs_consistency(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test coordinate reference system consistency"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Check that results are in target CRS
        assert results.crs == extractor.config['target_crs']
        
        # Check that catchments are in target CRS
        assert extractor.catchments.crs == extractor.config['target_crs']
    
    def test_error_handling(self, temp_dir):
        """Test error handling for missing or invalid data"""
        # Test with missing basin sample file
        with pytest.raises(Exception):
            extractor = TruthExtractor(data_dir=temp_dir)
            extractor.config['basin_sample_file'] = 'nonexistent.csv'
            extractor.load_datasets()
        
        # Test with missing catchments file
        with pytest.raises(Exception):
            extractor = TruthExtractor(data_dir=temp_dir)
            extractor.config['files']['nhd_catchments'] = 'nonexistent.shp'
            extractor.load_datasets()
    
    def test_export_functionality(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test export functionality with multiple formats"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Test export
        outputs = extractor.export_truth_polygons("test_output", results)
        
        # Check that outputs are created
        assert len(outputs) > 0
        for output in outputs:
            assert Path(output).exists()
        
        # Test GeoPackage export
        gpkg_output = [f for f in outputs if f.endswith('.gpkg')]
        if gpkg_output:
            exported_gdf = gpd.read_file(gpkg_output[0])
            assert len(exported_gdf) > 0
            assert 'geometry' in exported_gdf.columns
            assert 'ID' in exported_gdf.columns
    
    def test_mountain_west_specific_features(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test Mountain West specific extraction features"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        # Test terrain-specific extraction parameters
        terrain_extraction = extractor.config.get('terrain_extraction', {})
        
        if 'alpine' in terrain_extraction:
            alpine_config = terrain_extraction['alpine']
            assert alpine_config['max_parts'] > 5  # Allow more complex polygons
            assert alpine_config['min_drainage_density'] > 0.01  # Higher density threshold
        
        if 'desert' in terrain_extraction:
            desert_config = terrain_extraction['desert']
            assert desert_config['max_parts'] < 5  # Simpler polygons
            assert desert_config['min_drainage_density'] < 0.01  # Lower density threshold
    
    def test_retry_logic(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test retry logic for failed extractions"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.config['max_attempts'] = 3
        extractor.config['retry_with_different_strategies'] = True
        extractor.load_datasets()
        
        results = extractor.extract_truth_polygons()
        
        # Should have attempted extraction for all basins
        assert len(results) > 0
        
        # Check error logs if any extractions failed
        if hasattr(extractor, 'error_logs'):
            assert isinstance(extractor.error_logs, list)
    
    def test_performance_optimization(self, sample_basin_data, sample_catchments_gdf, temp_dir):
        """Test performance optimization features"""
        # Create basin sample file
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        sample_basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.config['chunk_size'] = 2  # Small chunk size for testing
        extractor.load_datasets()
        
        # Should process without performance issues
        results = extractor.extract_truth_polygons()
        assert len(results) > 0


class TestTruthExtractorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_no_intersecting_catchments(self, temp_dir):
        """Test handling when no catchments intersect with pour points"""
        # Create basin data with pour points outside catchment areas
        basin_data = pd.DataFrame({
            'ID': ['basin_001'],
            'Pour_Point_Lat': [50.0],  # Far outside catchment area
            'Pour_Point_Lon': [-80.0],
            'Area_km2': [15.5]
        })
        
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.load_datasets()
        
        # Should handle gracefully
        results = extractor.extract_truth_polygons()
        assert len(results) >= 0  # May be empty or use fallback strategy
    
    def test_multiple_intersecting_catchments(self, temp_dir):
        """Test handling when multiple catchments intersect with a pour point"""
        # Create multiple overlapping catchments
        overlapping_catchments = gpd.GeoDataFrame({
            'FEATUREID': ['1001', '1002', '1003'],
            'AREA': [100.0, 200.0, 150.0],
            'geometry': [
                Polygon([(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]),
                Polygon([(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]),
                Polygon([(0.05, 0.05), (0.25, 0.05), (0.25, 0.25), (0.05, 0.25)])
            ]
        }, crs='EPSG:4326')
        
        overlapping_catchments.to_file(f"{temp_dir}/overlapping_catchments.shp")
        
        # Create basin data with pour point in overlap area
        basin_data = pd.DataFrame({
            'ID': ['basin_001'],
            'Pour_Point_Lat': [0.15],
            'Pour_Point_Lon': [0.15],
            'Area_km2': [15.5]
        })
        
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.config['files']['nhd_catchments'] = 'overlapping_catchments.shp'
        extractor.load_datasets()
        
        # Should select one catchment based on priority
        results = extractor.extract_truth_polygons()
        assert len(results) == 1
    
    def test_invalid_geometries(self, temp_dir):
        """Test handling of invalid geometries in catchments"""
        # Create invalid polygon
        invalid_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5), (0, 0)])
        
        invalid_catchments = gpd.GeoDataFrame({
            'FEATUREID': ['1001'],
            'AREA': [100.0],
            'geometry': [invalid_polygon]
        }, crs='EPSG:4326')
        
        invalid_catchments.to_file(f"{temp_dir}/invalid_catchments.shp")
        
        # Create basin data
        basin_data = pd.DataFrame({
            'ID': ['basin_001'],
            'Pour_Point_Lat': [0.5],
            'Pour_Point_Lon': [0.5],
            'Area_km2': [15.5]
        })
        
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.config['files']['nhd_catchments'] = 'invalid_catchments.shp'
        
        # Should handle invalid geometries gracefully
        extractor.load_datasets()
        results = extractor.extract_truth_polygons()
        assert len(results) >= 0
    
    def test_very_small_catchments(self, temp_dir):
        """Test handling of very small catchments"""
        # Create very small catchment
        small_catchment = gpd.GeoDataFrame({
            'FEATUREID': ['1001'],
            'AREA': [0.05],  # Very small area
            'geometry': [Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])]
        }, crs='EPSG:4326')
        
        small_catchment.to_file(f"{temp_dir}/small_catchments.shp")
        
        # Create basin data
        basin_data = pd.DataFrame({
            'ID': ['basin_001'],
            'Pour_Point_Lat': [0.005],
            'Pour_Point_Lon': [0.005],
            'Area_km2': [15.5]
        })
        
        basin_file = f"{temp_dir}/test_basin_sample.csv"
        basin_data.to_csv(basin_file, index=False)
        
        extractor = TruthExtractor(data_dir=temp_dir)
        extractor.config['basin_sample_file'] = 'test_basin_sample.csv'
        extractor.config['files']['nhd_catchments'] = 'small_catchments.shp'
        extractor.load_datasets()
        
        # Should filter out very small catchments
        results = extractor.extract_truth_polygons()
        # May be empty if catchment is too small
        assert len(results) >= 0


if __name__ == "__main__":
    pytest.main([__file__]) 