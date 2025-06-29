import os
import tempfile
import json
import yaml
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scripts import validation_tools as vt

@pytest.fixture
def temp_csv(tmp_path):
    path = tmp_path / "test.csv"
    df = pd.DataFrame({"id": [1, 2], "area_km2": [10.5, 20.1]})
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def bad_csv(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w") as f:
        f.write("bad,data\n1,2,3\n")
    return str(path)

@pytest.fixture
def temp_json(tmp_path):
    path = tmp_path / "test.json"
    with open(path, "w") as f:
        json.dump({"iou": 0.95, "runtime": 25.0}, f)
    return str(path)

@pytest.fixture
def temp_gpkg(tmp_path):
    # Only create if geopandas is available
    if not vt.GEOPANDAS_AVAILABLE:
        pytest.skip("geopandas not available")
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    path = tmp_path / "test.gpkg"
    gdf.to_file(path, layer="basins", driver="GPKG")
    return str(path)

@pytest.fixture
def temp_raster(tmp_path):
    import rasterio
    path = tmp_path / "test.tif"
    arr = np.random.rand(10, 10).astype(np.float32)
    with rasterio.open(
        path, 'w', driver='GTiff', height=10, width=10, count=1, dtype='float32', crs='+proj=latlong',
        transform=rasterio.transform.from_origin(0, 10, 1, 1), nodata=None
    ) as dst:
        dst.write(arr, 1)
    return str(path)

@pytest.fixture
def temp_yaml(tmp_path):
    path = tmp_path / "test.yaml"
    config = {"pipeline": {"name": "Test", "data_dir": "data", "checkpointing": True, "resume_on_error": True, "max_retries": 1, "timeout_hours": 1},
              "stages": {"basin_sampling": {"enabled": True, "output_prefix": "a", "timeout_minutes": 1, "retry_on_failure": True},
                         "truth_extraction": {"enabled": True, "output_prefix": "b", "timeout_minutes": 1, "retry_on_failure": True},
                         "benchmark_execution": {"enabled": True, "output_prefix": "c", "timeout_minutes": 1, "retry_on_failure": True}}}
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)

@pytest.fixture
def temp_schema(tmp_path):
    path = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {"pipeline": {"type": "object"}},
        "required": ["pipeline"]
    }
    with open(path, "w") as f:
        json.dump(schema, f)
    return str(path)


def test_check_shapefile_quality_missing():
    result = vt.check_shapefile_quality("/no/such/file.shp")
    assert not result['exists']
    assert 'File does not exist' in result['issues']

def test_check_raster_quality_missing():
    result = vt.check_raster_quality("/no/such/file.tif")
    assert not result['exists']
    assert 'File does not exist' in result['issues']

def test_validate_intermediate_csv(temp_csv):
    result = vt.validate_intermediate_csv(temp_csv, ["id", "area_km2"])
    assert result['exists']
    assert result['row_count'] == 2
    assert not result['missing_columns']
    assert not result['issues']

def test_validate_intermediate_csv_missing_column(temp_csv):
    result = vt.validate_intermediate_csv(temp_csv, ["id", "missing_col"])
    assert "missing_col" in result['missing_columns']
    assert result['issues']

def test_validate_intermediate_csv_bad(bad_csv):
    result = vt.validate_intermediate_csv(bad_csv, ["id"])
    assert 'Error reading CSV' in result['issues'][0]

def test_verify_output_format_csv(temp_csv):
    result = vt.verify_output_format(temp_csv, "csv")
    assert result['exists']
    assert result['format_ok']

def test_verify_output_format_json(temp_json):
    result = vt.verify_output_format(temp_json, "json")
    assert result['exists']
    assert result['format_ok']

def test_verify_output_format_bad():
    result = vt.verify_output_format("/no/such/file.csv", "csv")
    assert not result['exists']
    assert 'File does not exist' in result['issues']

def test_detect_performance_regression(tmp_path, temp_json):
    baseline = {"iou": 0.95, "runtime": 25.0}
    current = {"iou": 0.90, "runtime": 30.0}
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f)
    result = vt.detect_performance_regression(current, str(baseline_path), {"iou": 0.05, "runtime": 0.1})
    assert 'regressions' in result
    assert 'iou' in result['regressions'] or 'runtime' in result['regressions']

def test_validate_config_schema_valid(temp_yaml, temp_schema):
    result = vt.validate_config_schema(temp_yaml, temp_schema)
    assert result['valid']
    assert not result['issues']

def test_validate_config_schema_invalid(temp_yaml, tmp_path):
    # Use a schema that requires a missing field
    schema = {"type": "object", "required": ["not_present"]}
    schema_path = tmp_path / "bad_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    result = vt.validate_config_schema(temp_yaml, str(schema_path))
    assert not result['valid']
    assert result['issues']

@pytest.mark.skipif(not vt.GEOPANDAS_AVAILABLE, reason="geopandas not available")
def test_validate_geopackage(temp_gpkg):
    result = vt.validate_geopackage(temp_gpkg, ["basins"])
    assert result['exists']
    assert "basins" in result['layers']
    assert not result['missing_layers']
    assert not result['empty_layers'] 