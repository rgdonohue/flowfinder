# FLOWFINDER Accuracy Benchmark Pipeline Orchestrator

## Overview

The `run_benchmark.py` script orchestrates the complete FLOWFINDER accuracy benchmark pipeline, providing a unified interface for running all three benchmark components in sequence with comprehensive error handling, progress tracking, checkpointing, and reporting.

## Features

### 🚀 **Pipeline Orchestration**
- **Sequential Execution**: Runs all three benchmark stages in the correct order
- **Dependency Management**: Ensures all required data and outputs are available
- **Stage Filtering**: Run specific stages or skip disabled ones
- **Configuration Management**: Unified configuration for all pipeline components

### 🔄 **Error Handling & Recovery**
- **Checkpointing**: Save progress after each stage for resuming interrupted runs
- **Retry Logic**: Automatic retry with exponential backoff for failed stages
- **Graceful Degradation**: Continue pipeline even if some stages fail
- **Error Recovery**: Resume from last successful stage

### 📊 **Progress Tracking & Monitoring**
- **Real-time Status**: Live progress updates and stage status
- **Execution Time Tracking**: Monitor performance of each stage
- **Resource Monitoring**: Track memory and CPU usage
- **Detailed Logging**: Comprehensive logs for debugging and analysis

### 📋 **Reporting & Outputs**
- **Unified Summary**: Comprehensive pipeline summary report
- **Multiple Formats**: Text, HTML, and JSON report formats
- **File Organization**: Organized output structure with timestamps
- **Quality Metrics**: Performance and accuracy statistics

### 🎛️ **Execution Modes**
- **Interactive Mode**: User confirmation and progress display
- **Automated Mode**: Headless execution for batch processing
- **Dry Run Mode**: Validate configuration without execution
- **Resume Mode**: Continue interrupted pipeline runs

## Pipeline Stages

### 1. Basin Sampling (`basin_sampling`)
- **Script**: `scripts/basin_sampler.py`
- **Purpose**: Stratified sampling of Mountain West basins
- **Inputs**: HUC12 boundaries, NHD+ flowlines, DEM data
- **Outputs**: `basin_sample.csv`, `basin_sample.gpkg`
- **Typical Runtime**: 30-60 minutes

### 2. Truth Extraction (`truth_extraction`)
- **Script**: `scripts/truth_extractor.py`
- **Purpose**: Ground truth polygon extraction from NHD+ HR
- **Inputs**: Basin sample, NHD+ HR catchments
- **Outputs**: `truth_polygons.gpkg`, `truth_polygons.csv`
- **Typical Runtime**: 60-120 minutes

### 3. Benchmark Execution (`benchmark_execution`)
- **Script**: `scripts/benchmark_runner.py`
- **Purpose**: FLOWFINDER accuracy and performance testing
- **Inputs**: Basin sample, truth polygons
- **Outputs**: `benchmark_results.json`, `accuracy_summary.csv`
- **Typical Runtime**: 4-8 hours

## Usage

### Basic Usage

```bash
# Run complete pipeline with default configuration
python run_benchmark.py --output results

# Run with custom configuration
python run_benchmark.py --config pipeline_config.yaml --output custom_results

# Resume interrupted pipeline
python run_benchmark.py --resume --output results
```

### Advanced Usage

```bash
# Run specific stages only
python run_benchmark.py --stages basin_sampling,truth_extraction --output results

# Automated mode (no interactive prompts)
python run_benchmark.py --automated --output results

# Dry run (validate configuration without execution)
python run_benchmark.py --dry-run --output results

# Create sample configuration
python run_benchmark.py --create-config > pipeline_config.yaml

# Verbose logging
python run_benchmark.py --verbose --output results
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Path to YAML configuration file | None (use defaults) |
| `--output, -o` | Output directory for results | `results` |
| `--resume` | Resume pipeline from checkpoint | False |
| `--stages` | Comma-separated list of stages to run | All stages |
| `--automated` | Run without interactive prompts | False |
| `--create-config` | Create sample configuration and exit | False |
| `--verbose, -v` | Enable verbose logging | False |
| `--dry-run` | Validate configuration without execution | False |

## Configuration

### Pipeline Configuration File

The pipeline uses a YAML configuration file to control all aspects of execution:

```yaml
pipeline:
  name: "FLOWFINDER Accuracy Benchmark"
  data_dir: "data"
  checkpointing: true
  resume_on_error: true
  max_retries: 3
  timeout_hours: 24

stages:
  basin_sampling:
    enabled: true
    config_file: "config/basin_sampler_config.yaml"
    output_prefix: "basin_sample"
    timeout_minutes: 60
    retry_on_failure: true

  truth_extraction:
    enabled: true
    config_file: "config/truth_extractor_config.yaml"
    output_prefix: "truth_polygons"
    timeout_minutes: 120
    retry_on_failure: true

  benchmark_execution:
    enabled: true
    config_file: "config/benchmark_config.yaml"
    output_prefix: "benchmark_results"
    timeout_minutes: 480
    retry_on_failure: false
```

### Configuration Options

#### Pipeline Settings
- `name`: Pipeline name for identification
- `data_dir`: Directory containing input datasets
- `checkpointing`: Enable/disable checkpointing
- `resume_on_error`: Resume pipeline after stage failures
- `max_retries`: Maximum retry attempts per stage
- `timeout_hours`: Overall pipeline timeout

#### Stage Settings
- `enabled`: Enable/disable stage execution
- `config_file`: Path to stage-specific configuration
- `output_prefix`: Prefix for stage output files
- `timeout_minutes`: Stage-specific timeout
- `retry_on_failure`: Enable retry for failed stages

## Output Structure

```
results/
├── pipeline_checkpoint.json          # Checkpoint for resuming
├── pipeline_20231201_143022.log      # Pipeline execution log
├── pipeline_summary.txt              # Comprehensive summary report
├── basin_sample.csv                  # Basin sampling results
├── basin_sample.gpkg                 # Basin sampling (GeoPackage)
├── truth_polygons.gpkg               # Truth polygons (GeoPackage)
├── truth_polygons.csv                # Truth polygons (CSV)
├── benchmark_results.json            # Benchmark results (JSON)
├── accuracy_summary.csv              # Accuracy summary (CSV)
├── benchmark_summary.txt             # Benchmark summary report
└── errors.log.csv                    # Error log (if any)
```

## Error Handling

### Checkpointing
- **Automatic Checkpointing**: Saves progress after each stage
- **Resume Capability**: Continue from last successful stage
- **Checkpoint File**: `pipeline_checkpoint.json` in results directory

### Retry Logic
- **Exponential Backoff**: Increasing delays between retry attempts
- **Configurable Retries**: Set maximum retry attempts per stage
- **Stage-Specific Settings**: Different retry policies per stage

### Error Recovery
- **Graceful Degradation**: Continue pipeline even with stage failures
- **Error Logging**: Comprehensive error tracking and reporting
- **Context Preservation**: Save error context for debugging

## Monitoring & Logging

### Progress Tracking
- **Real-time Updates**: Live progress indicators for each stage
- **Execution Time**: Track performance of individual stages
- **Status Reporting**: Current stage and overall progress

### Logging
- **Comprehensive Logs**: Detailed logging for all operations
- **Multiple Levels**: DEBUG, INFO, WARNING, ERROR levels
- **File and Console**: Log to both file and console output
- **Timestamped Logs**: All log entries include timestamps

### Performance Monitoring
- **Resource Usage**: Track memory and CPU consumption
- **Execution Metrics**: Stage timing and performance statistics
- **Quality Metrics**: Accuracy and reliability indicators

## Best Practices

### Data Preparation
1. **Organize Data**: Place all input datasets in the `data/` directory
2. **Verify Formats**: Ensure datasets are in expected formats
3. **Check Dependencies**: Verify all required files are present
4. **Test Configuration**: Use `--dry-run` to validate setup

### Execution
1. **Start Small**: Test with limited data before full run
2. **Monitor Resources**: Watch system resources during execution
3. **Use Checkpointing**: Enable checkpointing for long runs
4. **Plan for Interruptions**: Pipeline can be resumed if interrupted

### Troubleshooting
1. **Check Logs**: Review detailed logs for error information
2. **Validate Inputs**: Ensure all input data is correct
3. **Test Stages**: Run individual stages to isolate issues
4. **Review Configuration**: Verify configuration parameters

## Examples

### Example 1: Quick Test Run
```bash
# Run only basin sampling for testing
python run_benchmark.py --stages basin_sampling --output test_results
```

### Example 2: Production Run
```bash
# Full production run with custom configuration
python run_benchmark.py \
  --config config/pipeline_config.yaml \
  --output production_results \
  --automated
```

### Example 3: Resume Interrupted Run
```bash
# Resume pipeline that was interrupted
python run_benchmark.py --resume --output production_results
```

### Example 4: Development Testing
```bash
# Dry run to validate configuration
python run_benchmark.py --dry-run --config test_config.yaml

# Run with verbose logging for debugging
python run_benchmark.py --verbose --output debug_results
```

## Integration

### CI/CD Integration
The pipeline orchestrator is designed for integration with continuous integration systems:

```yaml
# GitHub Actions example
- name: Run FLOWFINDER Benchmark
  run: |
    python run_benchmark.py \
      --config config/pipeline_config.yaml \
      --output benchmark_results \
      --automated
```

### Monitoring Integration
The pipeline provides hooks for external monitoring systems:

- **Status Files**: Checkpoint and status files for monitoring
- **Log Files**: Structured logs for log aggregation
- **Exit Codes**: Proper exit codes for automation
- **Metrics**: Performance metrics for monitoring dashboards

## Troubleshooting

### Common Issues

#### Missing Dependencies
```
Error: Dependencies not satisfied for basin_sampling
```
**Solution**: Ensure all required data files are in the `data/` directory

#### Timeout Errors
```
Error: Stage basin_sampling timed out after 3600.0s
```
**Solution**: Increase timeout in configuration or optimize data processing

#### Configuration Errors
```
Error: Invalid YAML configuration
```
**Solution**: Validate YAML syntax and check configuration parameters

#### Permission Errors
```
Error: Permission denied when writing to results directory
```
**Solution**: Check file permissions and ensure write access

### Debug Mode
Enable debug mode for detailed troubleshooting:

```bash
python run_benchmark.py --verbose --output debug_results
```

### Log Analysis
Review logs for detailed error information:

```bash
# View latest log
tail -f results/pipeline_*.log

# Search for errors
grep -i error results/pipeline_*.log
```

## Support

For issues and questions:

1. **Check Documentation**: Review this documentation and script help
2. **Review Logs**: Examine detailed logs for error information
3. **Test Configuration**: Use `--dry-run` to validate setup
4. **Isolate Issues**: Run individual stages to identify problems

## Version History

- **v1.0.0**: Initial release with comprehensive pipeline orchestration
- **v1.1.0**: Added checkpointing and resume functionality
- **v1.2.0**: Enhanced error handling and monitoring
- **v1.3.0**: Added stage filtering and configuration management 