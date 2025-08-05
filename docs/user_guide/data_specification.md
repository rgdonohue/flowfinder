# FLOWFINDER Benchmark Data Specification

## Overview

This document specifies all data sources, access methods, preprocessing requirements, and quality control criteria for the FLOWFINDER accuracy benchmark system.

## Data Sources

### 1. Watershed Boundaries (HUC12)

**Source**: USGS Watershed Boundary Dataset (WBD)
**URL**: https://www.usgs.gov/national-hydrography/watershed-boundary-dataset
**Format**: Shapefile
**Coverage**: National
**License**: Public domain

**Access Method**:
```bash
# Download WBD HUC12 for Mountain West states
wget https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip
```

**Required Fields**:
- `HUC12`: 12-digit hydrologic unit code
- `NAME`: Watershed name
- `AREASQKM`: Area in square kilometers
- `STATES`: State codes (for filtering)

### 2. NHD+ High Resolution Catchments

**Source**: USGS National Hydrography Dataset Plus High Resolution
**URL**: https://www.usgs.gov/national-hydrography/national-hydrography-dataset
**Format**: File Geodatabase
**Coverage**: National (by HUC4)
**License**: Public domain

**Access Method**:
```bash
# Download NHD+ HR for Mountain West HUC4 regions
# Example for Colorado: https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPlus_H_1301_HU4_GDB.zip
```

**Required Layers**:
- `CatchmentSP`: Catchment polygons
- `Flowline`: Stream network
- `NHDFlowline`: Detailed flowlines

### 3. Digital Elevation Model (10m)

**Source**: USGS 3D Elevation Program (3DEP)
**URL**: https://www.usgs.gov/3d-elevation-program
**Format**: GeoTIFF
**Coverage**: National (1-degree tiles)
**License**: Public domain

**Access Method**:
```bash
# Download 10m DEM tiles for Mountain West
# Example: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/ArcGrid/USGS_13_n40w106_20221115.tif
```

## Data Organization

### Directory Structure
```
data/
├── raw/
│   ├── huc12/
│   │   └── WBD_National_GDB.zip
│   ├── nhd_hr/
│   │   ├── NHDPlus_H_1301_HU4_GDB.zip  # Colorado
│   │   ├── NHDPlus_H_1601_HU4_GDB.zip  # Utah
│   │   └── ...
│   └── dem/
│       ├── USGS_13_n40w106_20221115.tif
│       ├── USGS_13_n40w105_20221115.tif
│       └── ...
├── processed/
│   ├── mountain_west_huc12.shp
│   ├── mountain_west_catchments.gpkg
│   ├── mountain_west_flowlines.gpkg
│   └── mountain_west_dem.tif
└── metadata/
    ├── data_sources.json
    ├── processing_log.txt
    └── quality_checks.json
```

## Preprocessing Requirements

### 1. HUC12 Processing
```python
# Filter Mountain West states
mountain_west_states = ['CO', 'UT', 'NM', 'WY', 'MT', 'ID', 'AZ']
filtered_huc12 = huc12[huc12['STATES'].str.contains('|'.join(mountain_west_states))]

# Validate geometry and CRS
filtered_huc12 = filtered_huc12[filtered_huc12.geometry.is_valid]
filtered_huc12 = filtered_huc12.to_crs('EPSG:4326')
```

### 2. NHD+ HR Processing
```python
# Extract catchments and flowlines
catchments = gpd.read_file(nhd_gdb, layer='CatchmentSP')
flowlines = gpd.read_file(nhd_gdb, layer='NHDFlowline')

# Filter to Mountain West region
mountain_west_catchments = catchments[catchments.intersects(mountain_west_boundary)]
mountain_west_flowlines = flowlines[flowlines.intersects(mountain_west_boundary)]
```

### 3. DEM Processing
```python
# Mosaic multiple tiles
dem_mosaic = rasterio.merge([tile1, tile2, tile3])

# Validate resolution (should be 10m)
assert dem_mosaic.res[0] == 10.0, "DEM resolution must be 10m"

# Check for nodata values
assert not np.isnan(dem_mosaic.read(1)).any(), "DEM contains NaN values"
```

## Quality Control Criteria

### 1. HUC12 Quality Checks
- **Geometry Validity**: All polygons must be valid
- **Area Range**: 5-500 km² for benchmark basins
- **CRS Consistency**: All data in EPSG:4326
- **Attribute Completeness**: No missing HUC12 codes or names

### 2. NHD+ HR Quality Checks
- **Catchment Completeness**: All catchments must have associated flowlines
- **Geometry Validity**: No invalid geometries
- **Spatial Coverage**: Complete coverage of Mountain West region
- **Attribute Completeness**: Required fields present and non-null

### 3. DEM Quality Checks
- **Resolution**: Exactly 10m resolution
- **NoData Handling**: Proper nodata values
- **Value Range**: Elevation values within expected range (500-4500m)
- **CRS Consistency**: Consistent coordinate system

## Data Download Scripts

### Automated Download Script
```python
#!/usr/bin/env python3
"""
Data download script for FLOWFINDER benchmark
"""

import requests
import zipfile
from pathlib import Path

def download_huc12():
    """Download HUC12 watershed boundaries"""
    url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
    # Implementation...

def download_nhd_hr():
    """Download NHD+ HR for Mountain West states"""
    huc4_regions = ['1301', '1601', '1302', '1002', '1001', '1701', '1501']
    # Implementation...

def download_dem():
    """Download 10m DEM tiles for Mountain West"""
    # Implementation...
```

## Validation Scripts

### Data Validation
```python
def validate_huc12_data(huc12_path):
    """Validate HUC12 data quality"""
    gdf = gpd.read_file(huc12_path)

    # Check geometry validity
    invalid_count = len(gdf[~gdf.geometry.is_valid])
    assert invalid_count == 0, f"{invalid_count} invalid geometries found"

    # Check area range
    areas = gdf.geometry.area * 111 * 111  # Convert to km²
    valid_areas = areas[(areas >= 5) & (areas <= 500)]
    assert len(valid_areas) > 0, "No basins in target size range"

    return True
```

## Usage Examples

### 1. Download All Data
```bash
python scripts/download_data.py --all
```

### 2. Validate Data Quality
```bash
python scripts/validation_tools.py --check-shapefile data/processed/mountain_west_huc12.shp
python scripts/validation_tools.py --check-raster data/processed/mountain_west_dem.tif
```

### 3. Run Preprocessing
```bash
python scripts/preprocess_data.py --input data/raw --output data/processed
```

## Data Requirements Summary

| Dataset | Size Estimate | Download Time | Storage |
|---------|---------------|---------------|---------|
| HUC12 National | ~500MB | 10 minutes | 1GB |
| NHD+ HR (Mountain West) | ~2GB | 30 minutes | 5GB |
| 10m DEM (Mountain West) | ~5GB | 2 hours | 10GB |
| **Total** | **~7.5GB** | **~3 hours** | **~16GB** |

## Next Steps

1. **Create download scripts** for automated data acquisition
2. **Implement preprocessing pipeline** for data preparation
3. **Add quality control checks** to validation tools
4. **Test with sample data** before full download
5. **Document any data access issues** or preprocessing challenges

This specification provides the foundation for acquiring and preparing all required data for the FLOWFINDER accuracy benchmark system.
