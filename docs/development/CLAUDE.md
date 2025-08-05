# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLOWFINDER Accuracy Benchmark System - A **production-exemplary** scientific watershed delineation benchmarking tool for the Mountain West region. This is a research tool, not enterprise software, focusing on validating FLOWFINDER's spatial accuracy (95% IOU target) and performance (30s runtime target) using 10m DEM data.

**Current Status**: The project has undergone dramatic recent improvement and now represents **best practices** for scientific software development with comprehensive CI/CD infrastructure.

## Architecture & Data Flow

The system follows a 3-script pipeline architecture:

1. **Basin Sampler** (`scripts/basin_sampler.py`) → Stratified sampling of Mountain West basins
2. **Truth Extractor** (`scripts/truth_extractor.py`) → Ground truth polygon extraction from NHD+ HR
3. **Benchmark Runner** (`scripts/benchmark_runner.py`) → FLOWFINDER accuracy and performance testing

**Pipeline Orchestrator** (`run_benchmark.py`) - Comprehensive orchestrator that runs all three scripts sequentially with checkpointing, error handling, and unified reporting.

**Multi-Tool Framework** (`scripts/watershed_experiment_runner.py`) - Advanced framework supporting FLOWFINDER, TauDEM, GRASS, and WhiteboxTools with standardized result comparison.

### Key Data Dependencies
- Input: HUC12 boundaries, NHD+ HR catchments/flowlines, 10m DEM data
- Intermediate: `basin_sample.csv`, `truth_polygons.gpkg`
- Output: `benchmark_results.json`, `accuracy_summary.csv`

## Essential Commands

### Installation & Setup
```bash
# Install dependencies
pip install -e .[dev]

# Copy environment template
cp env.example .env
# Edit .env with your data paths
```

### Running the Complete Pipeline

REMEMBER TO TEST WITHIN THE VENV

```bash
# Complete orchestrated pipeline
python run_benchmark.py --output results

# Resume interrupted run
python run_benchmark.py --resume --output results

# Individual scripts (for development/debugging)
python scripts/basin_sampler.py --config config/basin_sampler_config.yaml
python scripts/truth_extractor.py --config config/truth_extractor_config.yaml
python scripts/benchmark_runner.py --sample basin_sample.csv --truth truth_polygons.gpkg

# Multi-tool watershed analysis
python scripts/watershed_experiment_runner.py --single --lat 40.0 --lon -105.5 --name "test_run"
```

### Testing & Validation

REMEMBER TO TEST WITHIN THE VENV

✅ **Current Status**: **206/206 tests passing** - Excellent test coverage!

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_benchmark_runner.py -v
python -m pytest tests/test_basin_sampler.py -v

# Run with coverage
python -m pytest tests/ --cov=scripts --cov=flowfinder --cov-report=html

# Core functionality tests
python flowfinder/test_basic.py  # Should work correctly
python test_configuration_system.py  # Configuration validation test
python test_standardized_results.py  # Standardized format test
```

### Code Quality & CI/CD

The project has **comprehensive CI/CD infrastructure** via GitHub Actions:

```bash
# Format code (matches CI pipeline)
black scripts/ tests/ flowfinder/ --exclude "scripts/benchmark_runner_backup.py"

# Lint code (matches CI pipeline)
flake8 scripts/ tests/ flowfinder/ --exclude "scripts/benchmark_runner_backup.py"

# Type checking (matches CI pipeline)
mypy scripts/ flowfinder/ --ignore-missing-imports

# Security scanning (matches CI pipeline)
safety check
bandit -r scripts/ flowfinder/
```

### Analysis
```bash
# Start Jupyter for analysis
jupyter lab notebooks/
# Open benchmark_analysis.ipynb
```

## Current Technical Status

### ✅ **Major Strengths (Production-Ready)**
- **Test Suite**: 206/206 tests passing (100% success rate)
- **Core Functionality**: Basic watershed delineation working correctly
- **CI/CD Pipeline**: Multi-Python testing (3.8-3.11), security scans, package validation
- **Code Quality**: Black, Flake8, MyPy integration with automated checks
- **Package Structure**: Properly installable via `pip install -e .`
- **Documentation**: Comprehensive README, PRD, technical documentation
- **License**: MIT license properly implemented

### 🔧 **Outstanding Technical Issues (Final Polish)**

1. **Configuration System Validation Bug** (Priority: HIGH)
   - **Issue**: Tool adapters failing with `['json'] is not of type 'object'` error
   - **Command**: `python test_configuration_system.py` shows the error
   - **Location**: `config/tools/*/` configurations need schema alignment
   - **Impact**: Prevents full multi-tool framework functionality

2. **Pre-commit Hooks Missing** (Priority: MEDIUM)
   - **Issue**: No `.pre-commit-config.yaml` file exists
   - **Impact**: Manual code quality checks instead of automated
   - **CI Integration**: Should align with existing GitHub Actions workflow

3. **Documentation Alignment** (Priority: MEDIUM)
   - **Issue**: README.md references non-existent paths (`config/experiments/`, `config/schemas/`)
   - **Impact**: New user confusion during setup
   - **Solution**: Update README to match actual repository structure

4. **Code Quality Polish** (Priority: LOW)
   - **Issue**: Flake8 issues exist (marked as non-blocking in CI)
   - **Impact**: Professional code standards
   - **Approach**: Focus on unused imports, line length, type annotations

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

### WatershedExperimentRunner Class
- Multi-tool watershed delineation framework
- Standardized result comparison across tools
- Mock mode for tools without executables installed

### Configuration System
Hierarchical configuration with inheritance:
- **Base configurations**: `config/base.yaml`
- **Environment-specific**: `config/environments/{development,testing,production}.yaml`
- **Tool-specific**: `config/tools/{flowfinder,taudem,grass,whitebox}.yaml`

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
Focus on testing critical geospatial operations, data quality validation, and configuration handling. **Current test suite is excellent** with 206/206 passing tests.

### CI/CD Integration
The project has **comprehensive GitHub Actions workflow** in `.github/workflows/ci.yml`:
- Multi-Python version testing (3.8-3.11)
- Automated linting and type checking
- Security scanning (Safety, Bandit)
- Package validation and coverage reporting
- Non-blocking quality checks (allows CI to pass while flagging issues)

## Success Metrics & Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| IOU (mean) | ≥ 0.90 | ✅ Framework ready |
| IOU (90th percentile) | ≥ 0.95 | ✅ Framework ready |
| Runtime (mean) | ≤ 30s | ✅ Framework ready |
| Error-free basin coverage | ≥ 90% | ✅ Framework ready |
| Test Coverage | 100% | ✅ 206/206 passing |

## File Organization

- `scripts/` - Core pipeline components
- `config/` - Hierarchical YAML configuration system
- `data/` - Input datasets (gitignored)
- `results/` - Output directory (gitignored)
- `tests/` - Comprehensive test suite (206 tests)
- `notebooks/` - Jupyter analysis notebooks
- `docs/` - Documentation and PRD
- `flowfinder/` - Core watershed delineation algorithms
- `.github/workflows/` - CI/CD pipeline configuration

## Immediate Development Priorities

### 1. Fix Configuration System (Est: 2-3 hours)
```bash
# Debug the validation error
python test_configuration_system.py
# Expected error: Configuration validation failed for flowfinder: ['json'] is not of type 'object'
```

### 2. Add Pre-commit Hooks (Est: 1-2 hours)
```bash
# Create .pre-commit-config.yaml
# Install pre-commit hooks
pre-commit install
```

### 3. Align Documentation (Est: 1-2 hours)
```bash
# Update README.md paths to match actual structure
# Verify all command examples work
```

### 4. Polish Code Quality (Est: 2-3 hours)
```bash
# Address flake8 issues systematically
flake8 --statistics
# Focus on unused imports, line length, type annotations
```

## What NOT to Change

DO NOT suggest:
- Switching to async/await unless solving specific I/O bottlenecks
- Adding ORM/database layers for simple file-based workflows
- Complex design patterns (Factory, Observer, etc.) for straightforward scripts
- Microservices splitting for this monolithic tool
- Advanced configuration management beyond YAML files
- Logging frameworks beyond Python's built-in logging
- API frameworks unless specifically needed
- **Breaking existing functionality** - all 206 tests must continue passing

DO focus on:
- **Configuration system debugging** (highest priority)
- **Pre-commit hooks implementation** (development workflow)
- **Documentation accuracy** (user experience)
- **Code quality polish** (professional standards)
- Spatial operation correctness and edge cases
- File handling robustness (permissions, disk space, corruption)
- Memory efficiency with large geospatial datasets
- CLI interface usability and error messages
- Configuration validation and helpful error messages
- Test coverage for critical spatial operations
- Documentation clarity for scientific users
- Performance with realistic dataset sizes

## Testing Before/After Changes

**Before making changes:**
```bash
python -m pytest tests/ -v  # Should show 206/206 passing
python flowfinder/test_basic.py  # Should work correctly
python test_standardized_results.py  # Should pass
```

**After making changes:**
```bash
python -m pytest tests/ -v  # Must still show 206/206 passing
python test_configuration_system.py  # Should pass after config fixes
```

**Final validation:**
```bash
python scripts/watershed_experiment_runner.py --single --lat 40.0 --lon -105.5 --name "validation_test"
```

This project is **production-exemplary** and represents best practices for scientific software development. Focus on targeted fixes for the remaining technical issues while preserving the excellent infrastructure already in place.
