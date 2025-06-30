# FLOWFINDER Test Setup Guide

## Quick Start for Testing

This guide helps you get the FLOWFINDER benchmark system running quickly with minimal data requirements.

## 🚀 **5-Minute Test Setup**

### Step 1: Set up test environment
```bash
# Create test environment with sample data (no downloads needed)
python scripts/test_setup.py
```

This creates:
- Sample basin data (5 test basins)
- Sample truth polygon data
- Sample benchmark results
- Test configuration files
- All required directories

### Step 2: Test the pipeline
```bash
# Run the benchmark pipeline with test data
python run_benchmark.py \
  --config data/test/processed/test_pipeline_config.yaml \
  --output test_results \
  --automated
```

### Step 3: Test individual components
```bash
# Test validation tools
python scripts/validation_tools.py --validate-csv data/test/processed/basin_sample.csv --columns id,area_km2

# Test benchmark runner with sample data
python scripts/benchmark_runner.py \
  --input data/test/processed/basin_sample.csv \
  --truth data/test/processed/truth_polygons.csv \
  --output test_benchmark
```

## 📊 **What You Get**

### Test Data (Generated, no downloads)
- **5 sample basins** in Colorado Front Range
- **Sample truth polygons** with quality scores
- **Mock benchmark results** showing typical performance
- **Test configuration** optimized for quick execution

### Test Results Expected
- **IOU scores**: 0.78-0.95 (realistic range)
- **Runtime**: 25-45 seconds per basin
- **Success rate**: 80% (4/5 basins successful)
- **Processing time**: <1 minute total

## 🔧 **Test Configuration**

The test setup uses:
- **Colorado Front Range only** (small geographic area)
- **5 basins** instead of 50+ for full benchmark
- **Sample data** instead of real downloads
- **Quick timeouts** (5-10 minutes per stage)
- **Minimal storage** (~1MB instead of 16GB)

## 📁 **Test Directory Structure**

```
data/test/
├── raw/           # Empty (no downloads needed)
├── processed/     # Sample data files
│   ├── basin_sample.csv
│   ├── truth_polygons.csv
│   ├── benchmark_results.json
│   └── test_pipeline_config.yaml
├── metadata/      # Test metadata
└── temp/          # Temporary files
```

## 🎯 **What This Tests**

### ✅ **Pipeline Orchestration**
- Configuration loading and validation
- Stage execution and dependency management
- Error handling and checkpointing
- Progress tracking and reporting

### ✅ **Individual Scripts**
- CLI argument parsing
- Configuration validation
- Data loading and processing
- Output generation and formatting

### ✅ **Validation Tools**
- Data quality checks
- Format verification
- Performance regression detection
- Configuration schema validation

### ✅ **Integration**
- End-to-end workflow execution
- Data flow between components
- Error recovery and logging
- Results generation and reporting

## 🚀 **Next Steps After Testing**

Once you've verified everything works with test data:

### Option 1: Download Real Data (Recommended)
```bash
# Download minimal real data (~500MB)
python scripts/download_data.py --config config/data_sources_test.yaml --all

# Run with real data
python run_benchmark.py --config config/data_sources_test.yaml --output real_results
```

### Option 2: Full Production Run
```bash
# Download complete dataset (~7.5GB)
python scripts/download_data.py --all

# Run full benchmark
python run_benchmark.py --config config/pipeline_config.yaml --output production_results
```

## 🔍 **Troubleshooting**

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
# or
pip install geopandas rasterio fiona shapely pyyaml jsonschema
```

**"Permission denied" errors:**
```bash
chmod +x scripts/*.py
```

**"Configuration not found" errors:**
```bash
# Make sure you're in the project root directory
cd /path/to/flowfinder
python scripts/test_setup.py
```

### Validation Commands

```bash
# Check test environment
python scripts/test_setup.py --validate

# Check individual components
python scripts/validation_tools.py --help
python run_benchmark.py --help
```

## 📈 **Expected Test Output**

```
2023-12-01 14:30:00 - INFO - TEST SETUP COMPLETE!
==================================================
You can now test the FLOWFINDER benchmark system with:

1. Test pipeline execution:
   python run_benchmark.py --config data/test/processed/test_pipeline_config.yaml --output test_results

2. Test validation tools:
   python scripts/validation_tools.py --validate-csv data/test/processed/basin_sample.csv --columns id,area_km2

3. Test individual scripts:
   python scripts/benchmark_runner.py --input data/test/processed/basin_sample.csv --truth data/test/processed/truth_polygons.csv --output test_benchmark
```

## 🎉 **Success Criteria**

Your test is successful if:
- ✅ Test setup completes without errors
- ✅ Pipeline runs and generates results
- ✅ Validation tools work with sample data
- ✅ All scripts respond to --help commands
- ✅ Results are generated in expected format

Once you've achieved this, you're ready to scale up to real data! 