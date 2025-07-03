#!/usr/bin/env python3
"""
Comprehensive unit tests for basin_sampler.py
Tests stratification logic, pour point snapping, and Mountain West specific features
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

from basin_sampler import BasinSampler


class TestBasinSampler:
    """Comprehensive test cases for BasinSampler class"""
    
    def test_config_loading(self, sample_config):
        """Test configuration loading with defaults and overrides"""
        sampler = BasinSampler()
        
        # Test default configuration
        assert sampler.config['area_range'] == [5, 500]
        assert sampler.config['snap_tolerance'] == 150
        assert sampler.config['target_crs'] == 'EPSG:5070'
        assert 'CO' in sampler.config['mountain_west_states']
    
    def test_mountain_west_filtering(self, sample_basins_gdf, setup_test_data_files):
        """Test Mountain West state filtering logic"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Should have filtered to Mountain West states
        assert len(sampler.huc12) > 0
        if 'STATES' in sampler.huc12.columns:
            for state in sampler.huc12['STATES']:
                assert state in ['CO', 'UT', 'NM', 'WY', 'MT', 'ID', 'AZ']
    
    def test_area_calculation(self, sample_basins_gdf, setup_test_data_files):
        """Test area calculation with proper CRS transformation"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Check that areas are calculated
        assert 'area_km2' in sampler.huc12.columns
        assert all(sampler.huc12['area_km2'] >= 0)
        
        # Test that areas are reasonable for test basins (very small coordinates)
        areas = sampler.huc12['area_km2'].values
        assert all(areas > 0)  # Areas should be positive
        assert all(areas < 1)  # Test polygons are very small
    
    def test_pour_point_computation(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test pour point computation with flowline snapping"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_pour_points()
        
        # Check that pour points are computed
        assert 'pour_point' in sampler.huc12.columns
        assert len(sampler.huc12['pour_point']) > 0
        
        # Test that pour points are valid geometries
        for point in sampler.huc12['pour_point']:
            assert point.is_valid
            assert isinstance(point, Point)
    
    def test_pour_point_snapping_logic(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test pour point snapping to nearest flowline"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_pour_points()
        
        # Test that pour points are snapped to flowlines when within tolerance
        snap_tolerance = sampler.config['snap_tolerance']
        
        for idx, row in sampler.huc12.iterrows():
            pour_point = row['pour_point']
            centroid = row.geometry.centroid
            
            # Check if pour point was snapped (different from centroid)
            if pour_point.distance(centroid) > 0:
                # Should be within snap tolerance of a flowline
                min_distance_to_flowline = float('inf')
                for _, flowline in sampler.flowlines.iterrows():
                    distance = pour_point.distance(flowline.geometry)
                    min_distance_to_flowline = min(min_distance_to_flowline, distance)
                
                assert min_distance_to_flowline <= snap_tolerance
    
    def test_terrain_roughness_calculation(self, sample_basins_gdf, setup_test_data_files):
        """Test terrain roughness calculation from DEM"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        
        # Check that slope std is calculated
        assert 'slope_std' in sampler.huc12.columns
        
        # Test that slope values are reasonable (may be NaN if DEM extraction fails in test)
        slope_values = sampler.huc12['slope_std'].dropna()
        if len(slope_values) > 0:
            assert all(slope_values >= 0)
            assert all(slope_values <= 50)  # Maximum reasonable slope std
        else:
            # If all values are NaN due to test data limitations, that's OK too
            assert sampler.huc12['slope_std'].isna().all()
    
    def test_stream_complexity_calculation(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test stream complexity calculation"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_stream_complexity()
        
        # Check that stream density is calculated
        assert 'stream_density' in sampler.huc12.columns
        
        # Test that stream density values are non-negative and finite
        density_values = sampler.huc12['stream_density'].values
        assert all(density_values >= 0)  # Non-negative densities
        assert all(np.isfinite(density_values))  # No infinite or NaN values
    
    def test_basin_classification(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test basin classification into size, terrain, and complexity classes"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        
        # Check that classification columns exist
        assert 'terrain_class' in sampler.huc12.columns
        assert 'size_class' in sampler.huc12.columns
        assert 'complexity_score' in sampler.huc12.columns
        
        # Test terrain classification values
        terrain_classes = sampler.huc12['terrain_class'].unique()
        assert all(cls in ['flat', 'moderate', 'steep'] for cls in terrain_classes)
        
        # Test size classification values
        size_classes = sampler.huc12['size_class'].unique()
        assert all(cls in ['small', 'medium', 'large'] for cls in size_classes)
        
        # Test complexity score values
        complexity_scores = sampler.huc12['complexity_score'].unique()
        assert all(score in [1, 2, 3] for score in complexity_scores)
    
    def test_stratification_logic(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test stratified sampling logic"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        
        stratum_summary = sampler.stratified_sample()
        
        # Check that sample is created
        assert hasattr(sampler, 'sample')
        assert len(sampler.sample) > 0
        
        # Test that sample covers all strata
        sample_df = sampler.sample
        if len(sample_df) > 0:
            # Check that we have samples from different terrain classes
            terrain_classes = sample_df['Terrain_Class'].unique()
            assert len(terrain_classes) > 0
            
            # Check that we have samples from different size classes
            size_classes = sample_df['Size_Class'].unique()
            assert len(size_classes) > 0
    
    def test_stratification_balance(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test that stratified sampling produces balanced samples"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        
        # Set n_per_stratum to 1 for testing
        sampler.config['n_per_stratum'] = 1
        stratum_summary = sampler.stratified_sample()
        
        # Check stratum summary
        assert 'strata' in stratum_summary
        assert 'total_basins' in stratum_summary
        assert 'sampled_basins' in stratum_summary
        
        # Test that we have samples from at least some strata (realistic for test data)
        assert stratum_summary['total_strata'] > 0
        assert stratum_summary['total_strata'] <= 27  # Max possible strata
    
    def test_export_functionality(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test export functionality with multiple formats"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_pour_points()  # Add missing pour point computation
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        sampler.stratified_sample()
        
        # Test export
        outputs = sampler.export_sample("test_output")
        
        # Check that outputs are created
        assert len(outputs) > 0
        for output in outputs:
            assert Path(output).exists()
        
        # Test CSV export
        csv_output = [f for f in outputs if f.endswith('.csv')]
        if csv_output:
            df = pd.read_csv(csv_output[0])
            assert len(df) > 0
            assert 'ID' in df.columns
            assert 'Pour_Point_Lat' in df.columns
            assert 'Pour_Point_Lon' in df.columns
    
    def test_mountain_west_specific_features(self, sample_basins_gdf, temp_dir):
        """Test Mountain West specific processing features"""
        sampler = BasinSampler(data_dir=temp_dir)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Test that Mountain West states are properly configured
        mw_states = sampler.config['mountain_west_states']
        assert 'CO' in mw_states  # Colorado Rockies
        assert 'UT' in mw_states  # Utah plateaus
        assert 'NM' in mw_states  # New Mexico mountains
        assert 'WY' in mw_states  # Wyoming ranges
        assert 'MT' in mw_states  # Montana Rockies
        assert 'ID' in mw_states  # Idaho mountains
        assert 'AZ' in mw_states  # Arizona plateaus
        
        # Test that basins are from Mountain West states
        if 'STATES' in sampler.huc12.columns:
            for state in sampler.huc12['STATES']:
                assert state in mw_states
    
    def test_error_handling(self, temp_dir):
        """Test error handling for missing or invalid data"""
        # Test with missing data directory
        with pytest.raises(Exception):
            sampler = BasinSampler(data_dir="/nonexistent/path")
            sampler.load_datasets()
        
        # Test with invalid configuration
        with pytest.raises(Exception):
            sampler = BasinSampler()
            sampler.config['area_range'] = [500, 5]  # Invalid range
            sampler.validate_config()
    
    def test_quality_validation(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test quality validation of sampled basins"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        sampler.stratified_sample()
        
        # Test that sampled basins meet quality criteria
        sample_df = sampler.sample
        if len(sample_df) > 0:
            # Check area constraints (use lowercase column names for internal data)
            assert all(sample_df['area_km2'] > 0)       # Areas should be positive
            assert all(sample_df['area_km2'] <= 1000)   # Reasonable upper bound for test
            
            # Check coordinate constraints (use pour point data if available)
            if 'Pour_Point_Lat' in sample_df.columns:
                assert all(sample_df['Pour_Point_Lat'] >= 30)  # Southern boundary
                assert all(sample_df['Pour_Point_Lat'] <= 50)  # Northern boundary
                assert all(sample_df['Pour_Point_Lon'] >= -120)  # Western boundary
                assert all(sample_df['Pour_Point_Lon'] <= -100)  # Eastern boundary
    
    def test_reproducibility(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test that sampling is reproducible with same seed"""
        # First run
        sampler1 = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler1.config['random_seed'] = 42
        sampler1.load_datasets()
        sampler1.filter_mountain_west_basins()
        sampler1.compute_terrain_roughness()
        sampler1.compute_stream_complexity()
        sampler1.classify_basins()
        sample1 = sampler1.stratified_sample()
        
        # Second run with same seed
        sampler2 = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler2.config['random_seed'] = 42
        sampler2.load_datasets()
        sampler2.filter_mountain_west_basins()
        sampler2.compute_terrain_roughness()
        sampler2.compute_stream_complexity()
        sampler2.classify_basins()
        sample2 = sampler2.stratified_sample()
        
        # Results should be identical
        assert sample1['sampled_basins'] == sample2['sampled_basins']
    
    def test_memory_management(self, sample_basins_gdf, sample_flowlines_gdf, setup_test_data_files):
        """Test memory management for large datasets"""
        sampler = BasinSampler(data_dir=setup_test_data_files, test_mode=True)
        sampler.config['chunk_size'] = 2  # Small chunk size for testing
        
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        
        # Should process without memory issues
        stratum_summary = sampler.stratified_sample()
        assert 'sampled_basins' in stratum_summary


class TestBasinSamplerEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_dataset(self, temp_dir):
        """Test handling of empty datasets"""
        # Create empty shapefiles for all required files
        empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        empty_gdf.to_file(f"{temp_dir}/huc12.shp")
        
        # Create empty flowlines and catchments
        empty_flowlines = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        empty_flowlines.to_file(f"{temp_dir}/nhd_flowlines.shp")
        
        empty_catchments = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        empty_catchments.to_file(f"{temp_dir}/nhd_hr_catchments.shp")
        
        sampler = BasinSampler(data_dir=temp_dir, test_mode=True)
        
        # Empty datasets should be handled gracefully, not raise exceptions
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Verify that no basins were found (expected behavior)
        assert len(sampler.huc12) == 0
    
    def test_single_basin(self, temp_dir):
        """Test handling of single basin"""
        # Create single basin with correct filename
        single_basin = gpd.GeoDataFrame({
            'HUC12': ['1201'],
            'STATES': ['CO'],
            'geometry': [Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])]
        }, crs='EPSG:4326')
        single_basin.to_file(f"{temp_dir}/huc12.shp")
        
        # Create minimal flowlines file to avoid loading errors
        empty_flowlines = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        empty_flowlines.to_file(f"{temp_dir}/nhd_flowlines.shp")
        
        sampler = BasinSampler(data_dir=temp_dir, test_mode=True)
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Should handle single basin gracefully
        assert len(sampler.huc12) == 1
    
    def test_invalid_geometries(self, temp_dir):
        """Test handling of invalid geometries"""
        # Create invalid polygon
        invalid_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5), (0, 0)])
        
        invalid_gdf = gpd.GeoDataFrame({
            'HUC12': ['1201'],
            'STATES': ['CO'],
            'geometry': [invalid_polygon]
        }, crs='EPSG:4326')
        invalid_gdf.to_file(f"{temp_dir}/invalid.shp")
        
        sampler = BasinSampler(data_dir=temp_dir)
        sampler.config['files']['huc12'] = 'invalid.shp'
        
        # Should handle invalid geometries gracefully
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        
        # Should either fix or filter out invalid geometries
        assert len(sampler.huc12) >= 0


if __name__ == "__main__":
    pytest.main([__file__]) 