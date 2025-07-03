#!/usr/bin/env python3
"""
Unit tests for pipeline orchestrator (run_benchmark.py)
Tests pipeline orchestration, checkpointing, and error handling
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
import sys
import os
import subprocess
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from run_benchmark import BenchmarkPipeline


class TestBenchmarkPipeline:
    """Test cases for BenchmarkPipeline class"""
    
    def test_config_loading(self, temp_dir):
        """Test configuration loading with defaults and overrides"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test default configuration
        assert pipeline.config['pipeline']['name'] == 'FLOWFINDER Accuracy Benchmark'
        assert pipeline.config['pipeline']['checkpointing'] == True
        assert pipeline.config['pipeline']['max_retries'] == 3
        assert pipeline.config['stages']['basin_sampling']['enabled'] == True
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test valid configuration
        pipeline._validate_config()
        
        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout_hours must be positive"):
            pipeline.config['pipeline']['timeout_hours'] = 0
            pipeline._validate_config()
    
    def test_checkpoint_save_load(self, temp_dir):
        """Test checkpoint saving and loading"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Save checkpoint
        pipeline.pipeline_status['current_stage'] = 'basin_sampling'
        pipeline._save_checkpoint()
        
        # Verify checkpoint file exists
        assert pipeline.checkpoint_file.exists()
        
        # Load checkpoint
        loaded = pipeline._load_checkpoint()
        assert loaded == True
        assert pipeline.pipeline_status['current_stage'] == 'basin_sampling'
    
    def test_dependency_checking(self, temp_dir):
        """Test dependency checking for stages"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Create mock data files
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()
        
        # Create required files
        (data_dir / "huc12.shp").touch()
        (data_dir / "nhd_flowlines.shp").touch()
        (data_dir / "nhd_hr_catchments.shp").touch()
        
        # Update data directory in config
        pipeline.config['pipeline']['data_dir'] = str(data_dir)
        
        # Test dependency checking
        assert pipeline._check_dependencies('basin_sampling') == True
        assert pipeline._check_dependencies('truth_extraction') == False  # Missing basin_sample.csv
    
    def test_output_checking(self, temp_dir):
        """Test output file checking"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Create mock output files
        (Path(temp_dir) / "basin_sample.csv").touch()
        (Path(temp_dir) / "basin_sample.gpkg").touch()
        
        # Test output checking
        assert pipeline._check_outputs('basin_sampling') == True
        
        # Remove one file
        (Path(temp_dir) / "basin_sample.gpkg").unlink()
        assert pipeline._check_outputs('basin_sampling') == False
    
    @patch('subprocess.run')
    def test_stage_execution_success(self, mock_run, temp_dir):
        """Test successful stage execution"""
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Create mock output files
        (Path(temp_dir) / "basin_sample.csv").touch()
        (Path(temp_dir) / "basin_sample.gpkg").touch()
        
        # Test stage execution
        result = pipeline._run_stage('basin_sampling')
        
        assert result['status'] == 'success'
        assert result['return_code'] == 0
        assert 'execution_time' in result
    
    @patch('subprocess.run')
    def test_stage_execution_failure(self, mock_run, temp_dir):
        """Test failed stage execution"""
        # Mock failed subprocess run
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="python", stderr="Error"
        )
        
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test stage execution
        result = pipeline._run_stage('basin_sampling')
        
        assert result['status'] == 'error'
        assert 'error' in result
    
    @patch('subprocess.run')
    def test_stage_execution_timeout(self, mock_run, temp_dir):
        """Test stage execution timeout"""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=30)
        
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test stage execution
        result = pipeline._run_stage('basin_sampling')
        
        assert result['status'] == 'timeout'
        assert 'execution_time' in result
    
    def test_retry_logic(self, temp_dir):
        """Test stage retry logic"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Mock failed stage result
        pipeline.pipeline_status['stage_results']['basin_sampling'] = {
            'status': 'error',
            'error': 'Test error'
        }
        pipeline.pipeline_status['failed_stages'].append('basin_sampling')
        
        # Test retry with mock successful execution
        with patch.object(pipeline, '_run_stage') as mock_run_stage:
            mock_run_stage.return_value = {
                'status': 'success',
                'execution_time': 30.0
            }
            
            result = pipeline._retry_stage('basin_sampling')
            
            assert result['status'] == 'success'
            assert 'basin_sampling' not in pipeline.pipeline_status['failed_stages']
    
    def test_summary_report_generation(self, temp_dir):
        """Test summary report generation"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Add some mock results
        pipeline.pipeline_status['completed_stages'] = ['basin_sampling']
        pipeline.pipeline_status['stage_results']['basin_sampling'] = {
            'status': 'success',
            'execution_time': 45.2
        }
        
        # Create mock output files
        (Path(temp_dir) / "basin_sample.csv").touch()
        (Path(temp_dir) / "test_file.txt").touch()
        
        # Generate summary
        summary_file = pipeline._generate_summary_report()
        
        # Check summary file exists
        assert Path(summary_file).exists()
        
        # Check content
        with open(summary_file, 'r') as f:
            content = f.read()
            assert 'Pipeline Summary' in content
            assert 'Basin Sampling' in content  # Display name, not internal key
            assert '45.2s' in content
    
    def test_duration_calculation(self, temp_dir):
        """Test duration calculation and formatting"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test short duration
        pipeline.pipeline_status['start_time'] = '2023-01-01T10:00:00'
        with patch('run_benchmark.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 30)
            mock_datetime.fromisoformat.return_value = datetime(2023, 1, 1, 10, 0, 0)
            
            duration = pipeline._calculate_duration()
            assert '30s' in duration
    
    def test_file_size_formatting(self, temp_dir):
        """Test file size formatting"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Test different sizes
        assert pipeline._format_file_size(1024) == "1.0 KB"
        assert pipeline._format_file_size(1024 * 1024) == "1.0 MB"
        assert pipeline._format_file_size(500) == "500.0 B"
    
    def test_pipeline_initialization(self, temp_dir):
        """Test complete pipeline initialization"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Check initialization
        assert pipeline.results_dir == Path(temp_dir)
        assert pipeline.checkpoint_file == Path(temp_dir) / "pipeline_checkpoint.json"
        assert pipeline.pipeline_status['overall_status'] == 'initialized'
        assert 'pipeline_id' in pipeline.pipeline_status
    
    def test_stage_disabling(self, temp_dir):
        """Test stage disabling functionality"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Disable a stage
        pipeline.config['stages']['benchmark_execution']['enabled'] = False
        
        # Check that stage is disabled
        assert pipeline.config['stages']['benchmark_execution']['enabled'] == False
    
    def test_signal_handlers(self, temp_dir):
        """Test signal handler setup"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Check that signal handlers are set up
        # This is mostly a smoke test since we can't easily test signal handling
        assert pipeline._setup_signal_handlers is not None


class TestPipelineIntegration:
    """Integration tests for pipeline components"""
    
    def test_pipeline_with_mock_data(self, temp_dir):
        """Test pipeline with mock data and dependencies"""
        # Create mock data structure
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create required data files
        (data_dir / "huc12.shp").touch()
        (data_dir / "nhd_flowlines.shp").touch()
        (data_dir / "nhd_hr_catchments.shp").touch()
        
        # Create configuration
        config = {
            'pipeline': {
                'data_dir': str(data_dir),
                'checkpointing': True,
                'resume_on_error': False,
                'max_retries': 1,
                'timeout_hours': 1
            },
            'stages': {
                'basin_sampling': {
                    'enabled': True,
                    'timeout_minutes': 5,
                    'retry_on_failure': False
                },
                'truth_extraction': {
                    'enabled': False,  # Disable for testing
                    'timeout_minutes': 5,
                    'retry_on_failure': False
                },
                'benchmark_execution': {
                    'enabled': False,  # Disable for testing
                    'timeout_minutes': 5,
                    'retry_on_failure': False
                }
            }
        }
        
        config_file = Path(temp_dir) / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Test pipeline initialization with config
        pipeline = BenchmarkPipeline(config_path=str(config_file), results_dir=temp_dir)
        
        assert pipeline.config['pipeline']['data_dir'] == str(data_dir)
        assert pipeline.config['stages']['truth_extraction']['enabled'] == False
    
    def test_pipeline_status_tracking(self, temp_dir):
        """Test pipeline status tracking throughout execution"""
        pipeline = BenchmarkPipeline(results_dir=temp_dir)
        
        # Simulate stage completion
        pipeline.pipeline_status['completed_stages'].append('basin_sampling')
        pipeline.pipeline_status['stage_results']['basin_sampling'] = {
            'status': 'success',
            'execution_time': 30.0
        }
        
        # Check status
        assert 'basin_sampling' in pipeline.pipeline_status['completed_stages']
        assert pipeline.pipeline_status['stage_results']['basin_sampling']['status'] == 'success'
        
        # Simulate stage failure
        pipeline.pipeline_status['failed_stages'].append('truth_extraction')
        pipeline.pipeline_status['error_count'] += 1
        
        # Check error tracking
        assert 'truth_extraction' in pipeline.pipeline_status['failed_stages']
        assert pipeline.pipeline_status['error_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 