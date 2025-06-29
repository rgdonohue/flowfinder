# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLOWFINDER Accuracy Benchmark System - A scientific watershed delineation benchmarking tool for the Mountain West region. This is a research tool, not enterprise software, focusing on validating FLOWFINDER's spatial accuracy (95% IOU target) and performance (30s runtime target) using 10m DEM data.

## Architecture & Data Flow

The system follows a 3-script pipeline architecture:

1. **Basin Sampler** (`scripts/basin_sampler.py`) → Stratified sampling of Mountain West basins
2. **Truth Extractor** (`scripts/truth_extractor.py`) → Ground truth polygon extraction from NHD+ HR  
3. **Benchmark Runner** (`scripts/benchmark_runner.py`) → FLOWFINDER accuracy and performance testing

**Pipeline Orchestrator** (`run_benchmark.py`) - Comprehensive orchestrator that runs all three scripts sequentially with checkpointing, error handling, and unified reporting.

### Key Data Dependencies
- Input: HUC12 boundaries, NHD+ HR catchments/flowlines, 10m DEM data
- Intermediate: `basin_sample.csv`, `truth_polygons.gpkg`
- Output: `benchmark_results.json`, `accuracy_summary.csv`

## Essential Commands

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template  
cp env.example .env
# Edit .env with your data paths
```

### Running the Complete Pipeline
```bash
# Complete orchestrated pipeline
python run_benchmark.py --output results

# Resume interrupted run
python run_benchmark.py --resume --output results

# Individual scripts (for development/debugging)
python scripts/basin_sampler.py --config config/basin_sampler_config.yaml
python scripts/truth_extractor.py --config config/truth_extractor_config.yaml  
python scripts/benchmark_runner.py --sample basin_sample.csv --truth truth_polygons.gpkg
```

### Testing & Validation
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html

# Validate data quality
python scripts/validation_tools.py --check-inputs data/
```

### Code Quality
```bash
# Format code
black scripts/ tests/

# Lint code
flake8 scripts/ tests/

# Type checking
mypy scripts/
```

### Analysis
```bash
# Start Jupyter for analysis
jupyter lab notebooks/
# Open benchmark_analysis.ipynb
```

## Core Components

### BasinSampler Class
- Stratified sampling across size, terrain, complexity dimensions
- Mountain West state filtering (CO, UT, NM, WY, MT, ID, AZ)
- Pour point computation with flowline snapping
- Terrain roughness calculation from DEM data

### TruthExtractor Class  
- Spatial joins between pour points and NHD+ catchments
- Quality validation (topology, area ratios, completeness)
- Multiple extraction strategies with priority ordering

### BenchmarkRunner Class
- FLOWFINDER CLI integration with timeout handling
- Spatial accuracy metrics (IOU, boundary ratio, centroid offset)
- Performance analysis and comprehensive reporting

### BenchmarkPipeline Class (Orchestrator)
- Sequential execution with dependency management
- Checkpointing for resuming interrupted runs
- Comprehensive error handling and recovery
- Progress tracking and unified reporting

## Configuration System

All components use YAML configuration files in `config/`:
- `basin_sampler_config.yaml` - Basin sampling parameters, stratification settings
- `truth_extractor_config.yaml` - Truth extraction settings, quality thresholds  
- `benchmark_config.yaml` - FLOWFINDER CLI settings, accuracy thresholds
- `pipeline_config.yaml` - Pipeline orchestration settings

## Development Guidelines

### Scientific Tool Focus
- Maintain KISS principles - simple > complex
- This is research software, not enterprise/web application
- Avoid over-engineering or unnecessary frameworks
- Focus on data quality, reproducibility, and clear results

### Geospatial Libraries
Key dependencies are geopandas, rasterio, shapely for geospatial processing. Always check imports and handle optional dependencies gracefully.

### Error Handling
- Comprehensive error logging with structured error tracking
- Graceful degradation for missing optional dependencies
- Progress bars for long-running operations (tqdm)

### Testing Philosophy
Focus on testing critical geospatial operations, data quality validation, and configuration handling. Less emphasis on unit testing every utility function.

## Success Metrics & Targets

| Metric | Target |
|--------|--------|
| IOU (mean) | ≥ 0.90 |
| IOU (90th percentile) | ≥ 0.95 |
| Runtime (mean) | ≤ 30s |
| Error-free basin coverage | ≥ 90% |

## File Organization

- `scripts/` - Core pipeline components
- `config/` - YAML configuration files
- `data/` - Input datasets (gitignored)
- `results/` - Output directory (gitignored)
- `tests/` - Comprehensive test suite
- `notebooks/` - Jupyter analysis notebooks
- `docs/` - Documentation and PRD

DO NOT suggest:
- Switching to async/await unless solving specific I/O bottlenecks
- Adding ORM/database layers for simple file-based workflows
- Complex design patterns (Factory, Observer, etc.) for straightforward scripts
- Microservices splitting for this monolithic tool
- Advanced configuration management beyond YAML files
- Logging frameworks beyond Python's built-in logging
- API frameworks unless specifically needed

DO focus on:
- Spatial operation correctness and edge cases
- File handling robustness (permissions, disk space, corruption)
- Memory efficiency with large geospatial datasets
- CLI interface usability and error messages
- Configuration validation and helpful error messages
- Test coverage for critical spatial operations
- Documentation clarity for scientific users
- Performance with realistic dataset sizes