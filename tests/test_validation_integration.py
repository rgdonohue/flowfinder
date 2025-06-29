import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scripts import validation_tools as vt

def test_full_pipeline_validation(tmp_path):
    # Simulate input shapefile/DEM (skip actual file creation for speed)
    # Instead, test missing file detection
    assert not vt.check_shapefile_quality(str(tmp_path / "missing.shp"))['exists']
    assert not vt.check_raster_quality(str(tmp_path / "missing.tif"))['exists']

    # Create and validate intermediate CSV
    csv_path = tmp_path / "basin_sample.csv"
    df = pd.DataFrame({"id": [1, 2], "area_km2": [10, 20], "pour_point_x": [100, 200]})
    df.to_csv(csv_path, index=False)
    result = vt.validate_intermediate_csv(str(csv_path), ["id", "area_km2", "pour_point_x"])
    assert result['exists']
    assert not result['issues']

    # Simulate missing column
    result = vt.validate_intermediate_csv(str(csv_path), ["id", "missing_col"])
    assert result['issues']

    # Simulate output format verification
    json_path = tmp_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({"iou": 0.91, "runtime": 28.0}, f)
    result = vt.verify_output_format(str(json_path), "json")
    assert result['format_ok']

    # Simulate performance regression detection
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"iou": 0.95, "runtime": 25.0}, f)
    current_metrics = {"iou": 0.91, "runtime": 28.0}
    thresholds = {"iou": 0.05, "runtime": 0.2}
    result = vt.detect_performance_regression(current_metrics, str(baseline_path), thresholds)
    assert 'regressions' in result
    # Should detect IOU regression
    assert 'iou' in result['regressions']

    # Simulate config schema validation
    config = {"pipeline": {"name": "Test", "data_dir": "data", "checkpointing": True, "resume_on_error": True, "max_retries": 1, "timeout_hours": 1},
              "stages": {"basin_sampling": {"enabled": True, "output_prefix": "a", "timeout_minutes": 1, "retry_on_failure": True},
                         "truth_extraction": {"enabled": True, "output_prefix": "b", "timeout_minutes": 1, "retry_on_failure": True},
                         "benchmark_execution": {"enabled": True, "output_prefix": "c", "timeout_minutes": 1, "retry_on_failure": True}}}
    config_path = tmp_path / "test.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config, f)
    schema = {"type": "object", "required": ["pipeline"]}
    schema_path = tmp_path / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    result = vt.validate_config_schema(str(config_path), str(schema_path))
    assert result['valid'] 