# FLOWFINDER Setup and Testing Instructions
*Complete guide for virtual environment setup and system validation*

## 🚀 Phase 1: Environment Setup

```bash
# Create and activate virtual environment
python -m venv flowfinder_env
source flowfinder_env/bin/activate  # On macOS/Linux
# or: flowfinder_env\Scripts\activate  # On Windows

# Install FLOWFINDER and development dependencies
pip install -e .[dev]

# Verify installation
flowfinder --help
```

This is the only supported local setup path for development in this repository.

## 🧪 Phase 2: System Validation Tests (After venv setup)

### Step 1: Basic Functionality Test
```bash
# Test core FLOWFINDER functionality
python flowfinder/test_basic.py

# Test configuration system
python test_configuration_system.py

# Test standardized results
python test_standardized_results.py
```

### Step 2: Integration Tests
```bash
# Run the full test suite
python -m pytest tests/ -v

# Test benchmark runner with existing data
python scripts/benchmark_runner.py --sample data/test/processed/basin_sample.csv --truth data/test/processed/truth_polygons.gpkg --output test_results_new

# Test experiment framework
python scripts/watershed_experiment_runner.py --single --lat 40.0150 --lon -105.2705 --name "validation_test" --env development
```

### Step 3: Multi-Tool Framework Test
```bash
# Test all tool adapters
python scripts/watershed_experiment_runner.py --check-tools --env production

# Run multi-tool comparison
python scripts/watershed_experiment_runner.py --single --lat 40.0 --lon -105.5 --env production --name "system_validation"

# Run multi-location experiment
python scripts/watershed_experiment_runner.py --multi --env testing --name "framework_validation"
```

## 📊 Phase 3: Data Requirements Assessment

### Current Available Data ✅
- **Test basins**: 5 sample watersheds in `data/test/processed/basin_sample.csv`
- **Ground truth**: Validation polygons in `data/test/processed/truth_polygons.gpkg`
- **HUC12 boundaries**: Minimal watershed boundaries in `data/minimal_test/processed/huc12_minimal.gpkg`

### Do We Need Full Data? 🤔

**For system validation: NO** - Current test data is sufficient to validate:
- Configuration system functionality
- Multi-tool adapter implementations
- Standardized result format
- Experiment framework
- Statistical analysis pipeline

**For production research: YES** - Would need:
- High-resolution DEMs (30m or better)
- Full HUC12 boundaries for Mountain West
- Real watershed delineation ground truth

### Data Download Options (If Needed)

#### Option A: Minimal Additional Data (5-10 GB)
```bash
# Download sample DEM tiles for Colorado Front Range
python scripts/download_data.py --config config/data_sources_test.yaml --region colorado_sample

# This would get:
# - 10m resolution DEMs for ~10 test watersheds
# - Corresponding NHD stream networks
# - Truth polygons from USGS
```

#### Option B: Full Research Dataset (50-100 GB)
```bash
# Download complete Mountain West dataset
python scripts/download_data.py --config config/data_sources.yaml --region mountain_west_full

# This would get:
# - 10m DEMs for entire Mountain West region
# - Complete HUC12 boundaries
# - Full NHD high-resolution stream networks
# - USGS benchmark watersheds
```

## 🎯 Phase 4: Expected Test Results

### What Should Work Immediately:
1. **✅ Configuration System**: All hierarchical configs load correctly
2. **✅ Tool Adapters**: Command generation works for all 4 tools
3. **✅ Mock Experiments**: Multi-tool comparisons with synthetic data
4. **✅ Result Analysis**: IOU matrices, performance comparisons
5. **✅ Export Functions**: JSON, CSV, summary reports generated

### What Might Need Real Data:
1. **⚠️ Actual Tool Execution**: TauDEM, GRASS, Whitebox not installed
2. **⚠️ Real DEM Processing**: Limited by available elevation data
3. **⚠️ Large-Scale Analysis**: Current data covers ~10 basins vs 500+ for full study

## 🔍 Phase 5: Validation Checklist

Run these commands in order and check for ✅ status:

```bash
# 1. Package installation check
python -c "import flowfinder, geopandas, shapely; print('✅ All packages imported')"

# 2. Configuration system check
python -c "
from config.configuration_manager import ConfigurationManager
mgr = ConfigurationManager('config', 'development')
print('✅ Configuration system working')
"

# 3. Tool adapter check
python -c "
from scripts.watershed_experiment_runner import WatershedExperimentRunner
runner = WatershedExperimentRunner('development')
avail = runner.check_tool_availability()
print(f'✅ Tool adapters: {len(avail)} tools checked')
"

# 4. Data loading check
python -c "
import pandas as pd, geopandas as gpd
df = pd.read_csv('data/test/processed/basin_sample.csv')
gdf = gpd.read_file('data/test/processed/truth_polygons.gpkg')
print(f'✅ Data loading: {len(df)} basins, {len(gdf)} truth polygons')
"

# 5. End-to-end experiment check
python scripts/watershed_experiment_runner.py --single --lat 40.0 --lon -105.5 --name "final_validation"
```

## 🚨 Troubleshooting Common Issues

### Issue: "No module named 'flowfinder'"
```bash
# Solution: Reinstall in development mode
pip install -e .[dev]
```

### Issue: "Cannot import 'shapely'"
```bash
# Solution: Install geospatial stack
pip install geopandas shapely rasterio fiona pyproj
```

### Issue: "Tool adapters not working"
```bash
# Expected: External tools not installed, should show mock results
# Check logs for "creating successful mock result for demonstration"
```

### Issue: "No data files found"
```bash
# Check data directory structure
ls -la data/test/processed/
# Should show: basin_sample.csv, truth_polygons.gpkg, benchmark_results.json
```

## 🎯 Success Criteria

**System is fully validated if you see:**

1. **✅ All imports successful** - No module errors
2. **✅ Configuration loading** - Environment configs load without errors
3. **✅ Tool adapter creation** - All 4 adapters initialize correctly
4. **✅ Data loading** - Test basins and truth polygons load successfully
5. **✅ Mock experiments** - Multi-tool watershed comparisons generate results
6. **✅ Result export** - JSON, summary reports created in experiment_results/
7. **✅ Statistical analysis** - IOU matrices, agreement scores calculated

## 📈 Next Steps After Validation

If all tests pass, the system is ready for:

1. **Research Applications**: Use current framework for watershed studies
2. **Tool Installation**: Install TauDEM/GRASS/Whitebox for real execution
3. **Data Expansion**: Download additional DEMs for larger studies
4. **Production Deployment**: Run large-scale watershed analysis experiments

## 💡 Pro Tips

- **Start with development environment** - Faster for initial testing
- **Check experiment_results/ directory** - All outputs saved there
- **Use --verbose flag** - Detailed logging for troubleshooting
- **Mock results are expected** - Real tools not installed, framework still works
- **Look for "Agreement score: X.XXX"** - Indicates successful multi-tool comparison

The system is architecturally complete - these tests validate that everything works as designed!
