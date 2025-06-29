#!/usr/bin/env python3
"""
FLOWFINDER Accuracy Benchmark Pipeline Orchestrator
===================================================

This script orchestrates the complete FLOWFINDER accuracy benchmark pipeline,
running all three components in sequence with comprehensive error handling,
progress tracking, checkpointing, and unified reporting.

Pipeline Components:
1. Basin Sampler - Stratified sampling of Mountain West basins
2. Truth Extractor - Ground truth polygon extraction from NHD+ HR
3. Benchmark Runner - FLOWFINDER accuracy and performance testing

Features:
- Sequential execution with dependency management
- Comprehensive error handling and recovery
- Progress tracking and status updates
- Checkpointing for resuming interrupted runs
- Unified logging and reporting
- Interactive and automated execution modes
- Configuration validation and data integrity checks

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml
import shutil
import subprocess
import signal
import atexit

import pandas as pd

# Optional imports for geospatial functionality
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas not available - some geospatial features may be limited")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class BenchmarkPipeline:
    """
    FLOWFINDER accuracy benchmark pipeline orchestrator.
    
    This class manages the complete benchmark workflow, including data dependencies,
    sequential execution, error handling, checkpointing, and unified reporting.
    
    Attributes:
        config (Dict[str, Any]): Pipeline configuration
        logger (logging.Logger): Logger instance for the pipeline
        checkpoint_file (Path): Checkpoint file for resuming interrupted runs
        results_dir (Path): Results directory for all outputs
        pipeline_status (Dict[str, Any]): Current pipeline status
        start_time (datetime): Pipeline start time
    """
    
    # Pipeline stages
    STAGES = {
        'basin_sampling': {
            'script': 'scripts/basin_sampler.py',
            'outputs': ['basin_sample.csv', 'basin_sample.gpkg'],
            'dependencies': ['data/huc12.shp', 'data/nhd_flowlines.shp'],
            'description': 'Stratified basin sampling'
        },
        'truth_extraction': {
            'script': 'scripts/truth_extractor.py',
            'outputs': ['truth_polygons.gpkg', 'truth_polygons.csv'],
            'dependencies': ['basin_sample.csv', 'data/nhd_hr_catchments.shp'],
            'description': 'Ground truth polygon extraction'
        },
        'benchmark_execution': {
            'script': 'scripts/benchmark_runner.py',
            'outputs': ['benchmark_results.json', 'accuracy_summary.csv'],
            'dependencies': ['basin_sample.csv', 'truth_polygons.gpkg'],
            'description': 'FLOWFINDER accuracy benchmarking'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, results_dir: str = "results") -> None:
        """
        Initialize the benchmark pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            results_dir: Directory for all pipeline outputs
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        self.config = self._load_config(config_path)
        self.results_dir = Path(results_dir)
        self.checkpoint_file = self.results_dir / "pipeline_checkpoint.json"
        self.pipeline_status = self._initialize_status()
        self.start_time = datetime.now()
        
        self._setup_logging()
        self._validate_config()
        self._setup_signal_handlers()
        
        self.logger.info("BenchmarkPipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load pipeline configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'pipeline': {
                'name': 'FLOWFINDER Accuracy Benchmark',
                'version': '1.0.0',
                'description': 'Complete accuracy benchmark for FLOWFINDER watershed delineation',
                'data_dir': 'data',
                'checkpointing': True,
                'resume_on_error': True,
                'max_retries': 3,
                'timeout_hours': 24,
                'cleanup_on_success': False,
                'generate_summary': True
            },
            'stages': {
                'basin_sampling': {
                    'enabled': True,
                    'config_file': None,
                    'output_prefix': 'basin_sample',
                    'timeout_minutes': 60,
                    'retry_on_failure': True
                },
                'truth_extraction': {
                    'enabled': True,
                    'config_file': None,
                    'output_prefix': 'truth_polygons',
                    'timeout_minutes': 120,
                    'retry_on_failure': True
                },
                'benchmark_execution': {
                    'enabled': True,
                    'config_file': None,
                    'output_prefix': 'benchmark_results',
                    'timeout_minutes': 480,  # 8 hours
                    'retry_on_failure': False  # Don't retry benchmark due to cost
                }
            },
            'reporting': {
                'generate_html': True,
                'generate_pdf': False,
                'include_plots': True,
                'email_notifications': False,
                'slack_notifications': False
            },
            'notifications': {
                'email': {
                    'enabled': False,
                    'recipients': [],
                    'smtp_server': 'localhost',
                    'smtp_port': 587
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': None,
                    'channel': '#benchmarks'
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Deep merge configuration
                    self._deep_merge(default_config, user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
        else:
            self.logger.info("Using default configuration")
            
        return default_config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline session."""
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.results_dir / f"pipeline_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline logging initialized - log file: {log_file}")
    
    def _validate_config(self) -> None:
        """
        Validate pipeline configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pipeline_config = self.config['pipeline']
        
        # Validate timeouts
        if pipeline_config['timeout_hours'] <= 0:
            raise ValueError("timeout_hours must be positive")
        
        if pipeline_config['max_retries'] < 0:
            raise ValueError("max_retries must be non-negative")
        
        # Validate data directory
        data_dir = Path(pipeline_config['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        # Validate stage configurations
        for stage_name, stage_config in self.config['stages'].items():
            if stage_config['timeout_minutes'] <= 0:
                raise ValueError(f"timeout_minutes for {stage_name} must be positive")
        
        self.logger.info("Configuration validation passed")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._save_checkpoint()
            self._cleanup()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup function
        atexit.register(self._cleanup)
    
    def _initialize_status(self) -> Dict[str, Any]:
        """Initialize pipeline status."""
        return {
            'pipeline_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_results': {},
            'overall_status': 'initialized',
            'error_count': 0,
            'warning_count': 0
        }
    
    def _save_checkpoint(self) -> None:
        """Save current pipeline status to checkpoint file."""
        if not self.config['pipeline']['checkpointing']:
            return
        
        try:
            checkpoint_data = {
                'pipeline_status': self.pipeline_status,
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            self.logger.debug("Checkpoint saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> bool:
        """Load pipeline status from checkpoint file."""
        if not self.config['pipeline']['checkpointing'] or not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.pipeline_status = checkpoint_data['pipeline_status']
            self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            self.logger.info(f"Resuming from stage: {self.pipeline_status.get('current_stage', 'unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    def _check_dependencies(self, stage_name: str) -> bool:
        """
        Check if all dependencies for a stage are satisfied.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        stage_info = self.STAGES[stage_name]
        dependencies = stage_info['dependencies']
        
        self.logger.info(f"Checking dependencies for {stage_name}...")
        
        for dep in dependencies:
            dep_path = Path(dep)
            
            # Handle relative paths
            if not dep_path.is_absolute():
                if dep.startswith('data/'):
                    dep_path = Path(self.config['pipeline']['data_dir']) / dep_path.name
                else:
                    dep_path = self.results_dir / dep_path.name
            
            if not dep_path.exists():
                self.logger.error(f"Missing dependency: {dep_path}")
                return False
            
            self.logger.debug(f"✓ Found dependency: {dep_path}")
        
        self.logger.info(f"All dependencies satisfied for {stage_name}")
        return True
    
    def _check_outputs(self, stage_name: str) -> bool:
        """
        Check if stage outputs already exist.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            True if all outputs exist, False otherwise
        """
        stage_info = self.STAGES[stage_name]
        outputs = stage_info['outputs']
        
        existing_outputs = []
        missing_outputs = []
        
        for output in outputs:
            output_path = self.results_dir / output
            if output_path.exists():
                existing_outputs.append(output)
            else:
                missing_outputs.append(output)
        
        if existing_outputs:
            self.logger.info(f"Found existing outputs for {stage_name}: {existing_outputs}")
        
        if missing_outputs:
            self.logger.info(f"Missing outputs for {stage_name}: {missing_outputs}")
        
        return len(missing_outputs) == 0
    
    def _run_stage(self, stage_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline stage.
        
        Args:
            stage_name: Name of the stage to run
            
        Returns:
            Dictionary containing stage execution results
        """
        stage_info = self.STAGES[stage_name]
        stage_config = self.config['stages'][stage_name]
        script_path = stage_info['script']
        
        self.logger.info(f"Starting {stage_name}: {stage_info['description']}")
        self.logger.info(f"Script: {script_path}")
        
        # Update pipeline status
        self.pipeline_status['current_stage'] = stage_name
        self._save_checkpoint()
        
        start_time = time.time()
        timeout_seconds = stage_config['timeout_minutes'] * 60
        
        try:
            # Build command
            cmd = ['python', script_path]
            
            # Add configuration file if specified
            if stage_config.get('config_file'):
                cmd.extend(['--config', stage_config['config_file']])
            
            # Add output prefix
            cmd.extend(['--output', stage_config['output_prefix']])
            
            # Add data directory
            cmd.extend(['--data-dir', self.config['pipeline']['data_dir']])
            
            # Add verbose flag
            cmd.append('--verbose')
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute stage
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                timeout=timeout_seconds,
                capture_output=True,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✓ {stage_name} completed successfully in {execution_time:.1f}s")
                
                # Check outputs
                if self._check_outputs(stage_name):
                    stage_result = {
                        'status': 'success',
                        'execution_time': execution_time,
                        'return_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                    
                    self.pipeline_status['completed_stages'].append(stage_name)
                    self.pipeline_status['stage_results'][stage_name] = stage_result
                    
                    return stage_result
                else:
                    raise RuntimeError(f"Stage completed but outputs are missing")
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Stage {stage_name} timed out after {execution_time:.1f}s"
            self.logger.error(error_msg)
            
            stage_result = {
                'status': 'timeout',
                'execution_time': execution_time,
                'error': error_msg
            }
            
            self.pipeline_status['failed_stages'].append(stage_name)
            self.pipeline_status['stage_results'][stage_name] = stage_result
            self.pipeline_status['error_count'] += 1
            
            return stage_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Stage {stage_name} failed: {e}"
            self.logger.error(error_msg)
            
            stage_result = {
                'status': 'error',
                'execution_time': execution_time,
                'error': str(e)
            }
            
            self.pipeline_status['failed_stages'].append(stage_name)
            self.pipeline_status['stage_results'][stage_name] = stage_result
            self.pipeline_status['error_count'] += 1
            
            return stage_result
    
    def _retry_stage(self, stage_name: str) -> Dict[str, Any]:
        """
        Retry a failed stage with exponential backoff.
        
        Args:
            stage_name: Name of the stage to retry
            
        Returns:
            Dictionary containing retry results
        """
        stage_config = self.config['stages'][stage_name]
        max_retries = self.config['pipeline']['max_retries']
        
        if not stage_config.get('retry_on_failure', True):
            self.logger.info(f"Retry disabled for {stage_name}")
            return self.pipeline_status['stage_results'][stage_name]
        
        for attempt in range(1, max_retries + 1):
            self.logger.info(f"Retrying {stage_name} (attempt {attempt}/{max_retries})")
            
            # Exponential backoff
            if attempt > 1:
                backoff_time = 2 ** (attempt - 1) * 30  # 30s, 60s, 120s, ...
                self.logger.info(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)
            
            # Remove failed stage from lists
            if stage_name in self.pipeline_status['failed_stages']:
                self.pipeline_status['failed_stages'].remove(stage_name)
            
            # Run stage again
            result = self._run_stage(stage_name)
            
            if result['status'] == 'success':
                self.logger.info(f"✓ {stage_name} succeeded on retry attempt {attempt}")
                return result
        
        self.logger.error(f"✗ {stage_name} failed after {max_retries} retry attempts")
        return self.pipeline_status['stage_results'][stage_name]
    
    def _generate_summary_report(self) -> str:
        """
        Generate comprehensive pipeline summary report.
        
        Returns:
            Path to the generated summary report
        """
        self.logger.info("Generating pipeline summary report...")
        
        summary_file = self.results_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("FLOWFINDER Accuracy Benchmark Pipeline Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Pipeline metadata
            f.write(f"Pipeline ID: {self.pipeline_status['pipeline_id']}\n")
            f.write(f"Start Time: {self.pipeline_status['start_time']}\n")
            f.write(f"End Time: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {self._calculate_duration()}\n\n")
            
            # Overall status
            f.write(f"Overall Status: {self.pipeline_status['overall_status'].upper()}\n")
            f.write(f"Completed Stages: {len(self.pipeline_status['completed_stages'])}\n")
            f.write(f"Failed Stages: {len(self.pipeline_status['failed_stages'])}\n")
            f.write(f"Total Errors: {self.pipeline_status['error_count']}\n")
            f.write(f"Total Warnings: {self.pipeline_status['warning_count']}\n\n")
            
            # Stage details
            f.write("Stage Results:\n")
            f.write("-" * 20 + "\n")
            
            for stage_name in self.STAGES.keys():
                if stage_name in self.pipeline_status['stage_results']:
                    result = self.pipeline_status['stage_results'][stage_name]
                    status = result['status'].upper()
                    duration = f"{result.get('execution_time', 0):.1f}s"
                    
                    f.write(f"{stage_name.replace('_', ' ').title()}:\n")
                    f.write(f"  Status: {status}\n")
                    f.write(f"  Duration: {duration}\n")
                    
                    if 'error' in result:
                        f.write(f"  Error: {result['error']}\n")
                    
                    f.write("\n")
            
            # File outputs
            f.write("Generated Files:\n")
            f.write("-" * 15 + "\n")
            
            for file_path in self.results_dir.glob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    f.write(f"{file_path.name} ({self._format_file_size(size)})\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")
            
            if self.pipeline_status['failed_stages']:
                f.write("• Review failed stages and check error logs\n")
                f.write("• Verify input data integrity and dependencies\n")
                f.write("• Consider adjusting timeout values if stages timed out\n")
            else:
                f.write("• All stages completed successfully\n")
                f.write("• Review benchmark results for accuracy insights\n")
                f.write("• Consider running additional analysis on results\n")
        
        self.logger.info(f"Summary report generated: {summary_file}")
        return str(summary_file)
    
    def _calculate_duration(self) -> str:
        """Calculate and format pipeline duration."""
        start_time = datetime.fromisoformat(self.pipeline_status['start_time'])
        end_time = datetime.now()
        duration = end_time - start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _cleanup(self) -> None:
        """Perform cleanup operations."""
        self.logger.info("Performing cleanup...")
        
        # Save final checkpoint
        self._save_checkpoint()
        
        # Cleanup on success if configured
        if (self.config['pipeline']['cleanup_on_success'] and 
            self.pipeline_status['overall_status'] == 'completed'):
            
            # Remove checkpoint file
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.info("Removed checkpoint file")
    
    def run_pipeline(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the complete benchmark pipeline.
        
        Args:
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary containing pipeline execution results
        """
        self.logger.info("Starting FLOWFINDER Accuracy Benchmark Pipeline")
        self.logger.info(f"Results directory: {self.results_dir}")
        
        # Load checkpoint if resuming
        if resume and self._load_checkpoint():
            self.logger.info("Resuming pipeline from checkpoint")
        else:
            self.logger.info("Starting fresh pipeline run")
        
        # Check overall timeout
        pipeline_timeout = timedelta(hours=self.config['pipeline']['timeout_hours'])
        if datetime.now() - self.start_time > pipeline_timeout:
            raise TimeoutError(f"Pipeline timeout exceeded ({pipeline_timeout})")
        
        # Run stages in sequence
        for stage_name in self.STAGES.keys():
            stage_config = self.config['stages'][stage_name]
            
            # Skip disabled stages
            if not stage_config.get('enabled', True):
                self.logger.info(f"Skipping disabled stage: {stage_name}")
                continue
            
            # Skip completed stages when resuming
            if resume and stage_name in self.pipeline_status['completed_stages']:
                self.logger.info(f"Skipping completed stage: {stage_name}")
                continue
            
            # Check dependencies
            if not self._check_dependencies(stage_name):
                error_msg = f"Dependencies not satisfied for {stage_name}"
                self.logger.error(error_msg)
                self.pipeline_status['overall_status'] = 'failed'
                self.pipeline_status['error_count'] += 1
                raise RuntimeError(error_msg)
            
            # Check if outputs already exist
            if self._check_outputs(stage_name):
                self.logger.info(f"Outputs already exist for {stage_name}, skipping")
                self.pipeline_status['completed_stages'].append(stage_name)
                continue
            
            # Run stage
            result = self._run_stage(stage_name)
            
            # Handle stage failure
            if result['status'] != 'success':
                if self.config['pipeline']['resume_on_error']:
                    self.logger.warning(f"Stage {stage_name} failed, attempting retry...")
                    result = self._retry_stage(stage_name)
                
                if result['status'] != 'success':
                    self.logger.error(f"Stage {stage_name} failed permanently")
                    self.pipeline_status['overall_status'] = 'failed'
                    
                    if not self.config['pipeline']['resume_on_error']:
                        break
            
            # Save checkpoint after each stage
            self._save_checkpoint()
        
        # Determine final status
        if not self.pipeline_status['failed_stages']:
            self.pipeline_status['overall_status'] = 'completed'
            self.logger.info("✓ Pipeline completed successfully")
        else:
            self.logger.warning(f"Pipeline completed with {len(self.pipeline_status['failed_stages'])} failed stages")
        
        # Generate summary report
        if self.config['pipeline']['generate_summary']:
            summary_file = self._generate_summary_report()
        else:
            summary_file = None
        
        # Final cleanup
        self._cleanup()
        
        # Prepare results
        pipeline_results = {
            'success': self.pipeline_status['overall_status'] == 'completed',
            'pipeline_status': self.pipeline_status,
            'summary_file': summary_file,
            'results_directory': str(self.results_dir),
            'duration': self._calculate_duration()
        }
        
        self.logger.info(f"Pipeline execution completed: {pipeline_results['success']}")
        return pipeline_results


def create_sample_config() -> str:
    """Create a sample pipeline configuration file."""
    sample_config = {
        'pipeline': {
            'name': 'FLOWFINDER Accuracy Benchmark',
            'version': '1.0.0',
            'description': 'Complete accuracy benchmark for FLOWFINDER watershed delineation',
            'data_dir': 'data',
            'checkpointing': True,
            'resume_on_error': True,
            'max_retries': 3,
            'timeout_hours': 24,
            'cleanup_on_success': False,
            'generate_summary': True
        },
        'stages': {
            'basin_sampling': {
                'enabled': True,
                'config_file': None,
                'output_prefix': 'basin_sample',
                'timeout_minutes': 60,
                'retry_on_failure': True
            },
            'truth_extraction': {
                'enabled': True,
                'config_file': None,
                'output_prefix': 'truth_polygons',
                'timeout_minutes': 120,
                'retry_on_failure': True
            },
            'benchmark_execution': {
                'enabled': True,
                'config_file': None,
                'output_prefix': 'benchmark_results',
                'timeout_minutes': 480,
                'retry_on_failure': False
            }
        },
        'reporting': {
            'generate_html': True,
            'generate_pdf': False,
            'include_plots': True
        }
    }
    
    config_content = yaml.dump(sample_config, default_flow_style=False, indent=2)
    return config_content


def main() -> None:
    """Main CLI entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Accuracy Benchmark Pipeline - Complete workflow orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default configuration
  python run_benchmark.py --output results
  
  # Run with custom configuration
  python run_benchmark.py --config pipeline_config.yaml --output custom_results
  
  # Resume interrupted pipeline
  python run_benchmark.py --resume --output results
  
  # Run specific stages only
  python run_benchmark.py --stages basin_sampling,truth_extraction --output results
  
  # Create sample configuration
  python run_benchmark.py --create-config > pipeline_config.yaml
  
  # Run in automated mode (no interactive prompts)
  python run_benchmark.py --automated --output results
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML pipeline configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for all pipeline results (default: results)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume pipeline from checkpoint'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        help='Comma-separated list of stages to run (default: all stages)'
    )
    
    parser.add_argument(
        '--automated',
        action='store_true',
        help='Run in automated mode without interactive prompts'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create sample configuration file and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and dependencies without execution'
    )
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        print(create_sample_config())
        return
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = BenchmarkPipeline(config_path=args.config, results_dir=args.output)
        
        # Handle stage filtering
        if args.stages:
            stage_list = [s.strip() for s in args.stages.split(',')]
            for stage_name in pipeline.STAGES.keys():
                if stage_name not in stage_list:
                    pipeline.config['stages'][stage_name]['enabled'] = False
                    pipeline.logger.info(f"Disabled stage: {stage_name}")
        
        # Dry run mode
        if args.dry_run:
            pipeline.logger.info("DRY RUN MODE - Validating configuration and dependencies...")
            
            # Check all dependencies
            all_deps_ok = True
            for stage_name in pipeline.STAGES.keys():
                if pipeline.config['stages'][stage_name].get('enabled', True):
                    if not pipeline._check_dependencies(stage_name):
                        all_deps_ok = False
            
            if all_deps_ok:
                pipeline.logger.info("✓ All dependencies satisfied")
                print("\n✅ Dry run completed successfully - pipeline is ready to run")
            else:
                pipeline.logger.error("✗ Some dependencies are missing")
                print("\n❌ Dry run failed - check missing dependencies")
                sys.exit(1)
            
            return
        
        # Interactive mode confirmation
        if not args.automated:
            print("\n🚀 FLOWFINDER Accuracy Benchmark Pipeline")
            print("=" * 50)
            print(f"Output directory: {args.output}")
            print(f"Configuration: {args.config or 'default'}")
            print(f"Resume mode: {args.resume}")
            
            if args.stages:
                print(f"Stages to run: {args.stages}")
            else:
                print("Stages to run: all")
            
            print("\nThis will execute the complete benchmark pipeline.")
            print("Estimated runtime: 2-8 hours depending on data size")
            
            response = input("\nProceed? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Pipeline execution cancelled")
                return
        
        # Run pipeline
        results = pipeline.run_pipeline(resume=args.resume)
        
        # Display results
        if results['success']:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📊 Duration: {results['duration']}")
            print(f"📁 Results: {results['results_directory']}")
            if results['summary_file']:
                print(f"📋 Summary: {results['summary_file']}")
        else:
            print(f"\n⚠️  Pipeline completed with issues")
            print(f"📊 Duration: {results['duration']}")
            print(f"📁 Results: {results['results_directory']}")
            print(f"❌ Failed stages: {len(results['pipeline_status']['failed_stages'])}")
            if results['summary_file']:
                print(f"📋 Summary: {results['summary_file']}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 