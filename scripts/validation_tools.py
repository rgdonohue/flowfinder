#!/usr/bin/env python3
"""
FLOWFINDER Validation Tools
==========================

Comprehensive validation utilities for the FLOWFINDER pipeline:
- Data quality checks for input shapefiles/rasters
- Intermediate result validation between pipeline stages
- Output format verification
- Performance regression detection
- Configuration schema validation

Can be used as a module or standalone CLI.
"""

import argparse
import logging
import sys
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from shapely.geometry import shape
from shapely.validation import explain_validity
from jsonschema import (
    validate as jsonschema_validate,
    ValidationError as JSONSchemaValidationError,
)

# Optional imports for geospatial functionality
try:
    import fiona

    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validation_tools")


# ----------------------
# Data Quality Checks
# ----------------------
def check_shapefile_quality(shapefile_path: str) -> Dict[str, Any]:
    """
    Validate a shapefile or GeoPackage for geometry, CRS, and attribute quality.
    Returns a dict with summary and issues found.
    """
    result = {
        "file": shapefile_path,
        "exists": False,
        "crs": None,
        "geometry_type": None,
        "feature_count": 0,
        "invalid_geometries": 0,
        "missing_attributes": [],
        "issues": [],
    }
    path = Path(shapefile_path)
    if not path.exists():
        result["issues"].append("File does not exist")
        return result
    result["exists"] = True

    if not FIONA_AVAILABLE:
        result["issues"].append("fiona not available - cannot validate shapefile")
        return result

    try:
        if GEOPANDAS_AVAILABLE:
            gdf = gpd.read_file(shapefile_path)
            result["feature_count"] = len(gdf)
            result["crs"] = str(gdf.crs)
            result["geometry_type"] = gdf.geom_type.unique().tolist()
            # Geometry validity
            invalid = gdf[~gdf.is_valid]
            result["invalid_geometries"] = len(invalid)
            if len(invalid) > 0:
                result["issues"].append(
                    f"{len(invalid)} invalid geometries: "
                    + ", ".join(
                        [explain_validity(geom) for geom in invalid.geometry[:5]]
                    )
                )
            # Attribute missingness
            for col in gdf.columns:
                if gdf[col].isnull().any():
                    result["missing_attributes"].append(col)
            if result["missing_attributes"]:
                result["issues"].append(
                    f"Missing values in: {result['missing_attributes']}"
                )
        else:
            # Fallback: use Fiona
            with fiona.open(shapefile_path) as src:
                result["feature_count"] = len(src)
                result["crs"] = str(src.crs)
                result["geometry_type"] = list(set(f["geometry"]["type"] for f in src))
                # Geometry validity (basic)
                invalid_count = 0
                for f in src:
                    try:
                        geom = shape(f["geometry"])
                        if not geom.is_valid:
                            invalid_count += 1
                    except Exception:
                        invalid_count += 1
                result["invalid_geometries"] = invalid_count
                if invalid_count > 0:
                    result["issues"].append(
                        f"{invalid_count} invalid geometries (Fiona)"
                    )
    except Exception as e:
        result["issues"].append(f"Error reading file: {e}")
    return result


def check_raster_quality(raster_path: str) -> Dict[str, Any]:
    """
    Validate a raster (DEM) for nodata, CRS, resolution, and value range.
    Returns a dict with summary and issues found.
    """
    result = {
        "file": raster_path,
        "exists": False,
        "crs": None,
        "resolution": None,
        "nodata": None,
        "min": None,
        "max": None,
        "issues": [],
    }
    path = Path(raster_path)
    if not path.exists():
        result["issues"].append("File does not exist")
        return result
    result["exists"] = True

    if not RASTERIO_AVAILABLE:
        result["issues"].append("rasterio not available - cannot validate raster")
        return result

    try:
        with rasterio.open(raster_path) as src:
            result["crs"] = str(src.crs)
            result["resolution"] = src.res
            result["nodata"] = src.nodata
            arr = src.read(1, masked=True)
            result["min"] = float(np.nanmin(arr))
            result["max"] = float(np.nanmax(arr))
            if np.isnan(arr).any():
                result["issues"].append("Raster contains NaN values")
            if src.nodata is not None and np.any(arr == src.nodata):
                result["issues"].append("Raster contains nodata values")
    except Exception as e:
        result["issues"].append(f"Error reading raster: {e}")
    return result


# ----------------------
# Intermediate Result Validation
# ----------------------
def validate_intermediate_csv(
    csv_path: str, required_columns: List[str]
) -> Dict[str, Any]:
    """
    Validate a CSV file for required columns and non-empty rows.
    """
    result = {
        "file": csv_path,
        "exists": False,
        "row_count": 0,
        "missing_columns": [],
        "issues": [],
    }
    path = Path(csv_path)
    if not path.exists():
        result["issues"].append("File does not exist")
        return result
    result["exists"] = True
    try:
        df = pd.read_csv(csv_path)
        result["row_count"] = len(df)
        for col in required_columns:
            if col not in df.columns:
                result["missing_columns"].append(col)
        if result["missing_columns"]:
            result["issues"].append(f"Missing columns: {result['missing_columns']}")
        if result["row_count"] == 0:
            result["issues"].append("CSV is empty")
    except Exception as e:
        result["issues"].append(f"Error reading CSV: {e}")
    return result


def validate_geopackage(
    gpkg_path: str, required_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate a GeoPackage for required layers and non-empty features.
    """
    result = {
        "file": gpkg_path,
        "exists": False,
        "layers": [],
        "empty_layers": [],
        "missing_layers": [],
        "issues": [],
    }
    path = Path(gpkg_path)
    if not path.exists():
        result["issues"].append("File does not exist")
        return result
    result["exists"] = True
    try:
        if GEOPANDAS_AVAILABLE:
            layers = fiona.listlayers(gpkg_path)
            result["layers"] = layers
            if required_layers:
                for lyr in required_layers:
                    if lyr not in layers:
                        result["missing_layers"].append(lyr)
            for lyr in layers:
                gdf = gpd.read_file(gpkg_path, layer=lyr)
                if len(gdf) == 0:
                    result["empty_layers"].append(lyr)
            if result["missing_layers"]:
                result["issues"].append(f"Missing layers: {result['missing_layers']}")
            if result["empty_layers"]:
                result["issues"].append(f"Empty layers: {result['empty_layers']}")
        else:
            result["issues"].append("geopandas required for GeoPackage validation")
    except Exception as e:
        result["issues"].append(f"Error reading GeoPackage: {e}")
    return result


# ----------------------
# Output Format Verification
# ----------------------
def verify_output_format(file_path: str, expected_format: str) -> Dict[str, Any]:
    """
    Verify that an output file exists and matches the expected format.
    Supported formats: csv, json, geojson, gpkg
    """
    result = {"file": file_path, "exists": False, "format_ok": False, "issues": []}
    path = Path(file_path)
    if not path.exists():
        result["issues"].append("File does not exist")
        return result
    result["exists"] = True
    try:
        if expected_format == "csv":
            pd.read_csv(file_path)
            result["format_ok"] = True
        elif expected_format == "json":
            with open(file_path) as f:
                json.load(f)
            result["format_ok"] = True
        elif expected_format == "geojson":
            if GEOPANDAS_AVAILABLE:
                gpd.read_file(file_path)
                result["format_ok"] = True
            else:
                with open(file_path) as f:
                    data = json.load(f)
                    if "features" in data:
                        result["format_ok"] = True
        elif expected_format == "gpkg":
            if GEOPANDAS_AVAILABLE:
                gpd.read_file(file_path)
                result["format_ok"] = True
            else:
                result["issues"].append("geopandas required for GPKG format check")
        else:
            result["issues"].append(f"Unknown format: {expected_format}")
    except Exception as e:
        result["issues"].append(f"Format check failed: {e}")
    return result


# ----------------------
# Performance Regression Detection
# ----------------------
def detect_performance_regression(
    current_metrics: Dict[str, Any],
    baseline_metrics_path: str,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare current run metrics to baseline and detect regressions.
    thresholds: dict of {metric: max_allowed_degradation_fraction}
    """
    result = {"regressions": [], "details": {}, "issues": []}
    try:
        with open(baseline_metrics_path) as f:
            baseline = json.load(f)
        for metric, threshold in thresholds.items():
            if metric in current_metrics and metric in baseline:
                base = baseline[metric]
                curr = current_metrics[metric]
                if base == 0:
                    continue
                change = (curr - base) / base
                result["details"][metric] = {
                    "baseline": base,
                    "current": curr,
                    "change": change,
                }
                if abs(change) > threshold:
                    result["regressions"].append(metric)
        if result["regressions"]:
            result["issues"].append(f"Regressions detected: {result['regressions']}")
    except Exception as e:
        result["issues"].append(f"Error reading baseline: {e}")
    return result


# ----------------------
# Configuration Schema Validation
# ----------------------
def validate_config_schema(config_path: str, schema_path: str) -> Dict[str, Any]:
    """
    Validate a YAML config file against a JSON schema.
    """
    result = {
        "config_file": config_path,
        "schema_file": schema_path,
        "valid": False,
        "issues": [],
    }
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(schema_path) as f:
            schema = json.load(f)
        jsonschema_validate(instance=config, schema=schema)
        result["valid"] = True
    except JSONSchemaValidationError as e:
        result["issues"].append(f"Schema validation error: {e.message}")
    except Exception as e:
        result["issues"].append(f"Error loading config or schema: {e}")
    return result


# ----------------------
# CLI Entrypoint
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Validation Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check shapefile quality
  python validation_tools.py --check-shapefile data/huc12.shp

  # Check raster quality
  python validation_tools.py --check-raster data/dem.tif

  # Validate intermediate CSV
  python validation_tools.py --validate-csv results/basin_sample.csv --columns id,area_km2

  # Validate GeoPackage
  python validation_tools.py --validate-gpkg results/sample.gpkg --layers basins,streams

  # Verify output format
  python validation_tools.py --verify-format results/accuracy_summary.csv --format csv

  # Detect performance regression
  python validation_tools.py --detect-regression --current-metrics results/current_metrics.json --baseline results/baseline_metrics.json --thresholds iou=0.05,runtime=0.2

  # Validate config schema
  python validation_tools.py --validate-config config/pipeline_config.yaml --schema config/pipeline_schema.json
        """,
    )
    parser.add_argument(
        "--check-shapefile", type=str, help="Path to shapefile or GeoPackage to check"
    )
    parser.add_argument(
        "--check-raster", type=str, help="Path to raster (DEM) to check"
    )
    parser.add_argument("--validate-csv", type=str, help="Path to CSV to validate")
    parser.add_argument(
        "--columns", type=str, help="Comma-separated list of required columns for CSV"
    )
    parser.add_argument(
        "--validate-gpkg", type=str, help="Path to GeoPackage to validate"
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated list of required layers for GeoPackage",
    )
    parser.add_argument(
        "--verify-format", type=str, help="Path to output file to verify format"
    )
    parser.add_argument(
        "--format", type=str, help="Expected format: csv, json, geojson, gpkg"
    )
    parser.add_argument(
        "--detect-regression", action="store_true", help="Detect performance regression"
    )
    parser.add_argument(
        "--current-metrics", type=str, help="Path to current metrics JSON"
    )
    parser.add_argument("--baseline", type=str, help="Path to baseline metrics JSON")
    parser.add_argument(
        "--thresholds",
        type=str,
        help="Comma-separated metric=threshold pairs (e.g., iou=0.05,runtime=0.2)",
    )
    parser.add_argument(
        "--validate-config", type=str, help="Path to YAML config to validate"
    )
    parser.add_argument("--schema", type=str, help="Path to JSON schema for config")
    args = parser.parse_args()

    if args.check_shapefile:
        result = check_shapefile_quality(args.check_shapefile)
        print(json.dumps(result, indent=2))
    elif args.check_raster:
        result = check_raster_quality(args.check_raster)
        print(json.dumps(result, indent=2))
    elif args.validate_csv:
        columns = args.columns.split(",") if args.columns else []
        result = validate_intermediate_csv(args.validate_csv, columns)
        print(json.dumps(result, indent=2))
    elif args.validate_gpkg:
        layers = args.layers.split(",") if args.layers else None
        result = validate_geopackage(args.validate_gpkg, layers)
        print(json.dumps(result, indent=2))
    elif args.verify_format:
        result = verify_output_format(args.verify_format, args.format)
        print(json.dumps(result, indent=2))
    elif args.detect_regression:
        if not (args.current_metrics and args.baseline and args.thresholds):
            print(
                "--current-metrics, --baseline, and --thresholds are required for regression detection"
            )
            sys.exit(1)
        thresholds = {
            k: float(v)
            for k, v in (item.split("=") for item in args.thresholds.split(","))
        }
        with open(args.current_metrics) as f:
            current_metrics = json.load(f)
        result = detect_performance_regression(
            current_metrics, args.baseline, thresholds
        )
        print(json.dumps(result, indent=2))
    elif args.validate_config:
        if not args.schema:
            print("--schema is required for config validation")
            sys.exit(1)
        result = validate_config_schema(args.validate_config, args.schema)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
