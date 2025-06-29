# PRD: FLOWFINDER Accuracy Benchmark System

## 1. Overview

This project establishes a rigorous accuracy benchmark for the FLOWFINDER watershed delineation tool, focusing on basins within the Mountain West region of the U.S. The purpose is to validate FLOWFINDER's core claims: fast (<30s) delineation and high spatial accuracy (targeting 95% Intersection over Union, or IOU) using 10 m DEM data. This benchmark will define the tool's performance envelope, guide optimization, and serve as a public competitive differentiator.

---

## 2. Objectives

* Quantify delineation accuracy across terrain and basin complexity.
* Test whether the 30s runtime and 95% IOU targets are realistically achievable.
* Identify terrain/flowline conditions under which FLOWFINDER underperforms.
* Provide a reproducible benchmarking framework and publish transparent results.

---

## 3. Core Features

### 3.1 Stratified Basin Sampling Script

**Module**: `basin_sampler.py`

* Input: HUC12 boundaries, NHD+ HR catchments/flowlines, 10 m DEM
* Filters Mountain West basins (CO, UT, NM, WY, MT, ID, AZ)
* Calculates:

  * Area (km²)
  * Terrain roughness (slope stddev)
  * Stream complexity (stream length/km²)
  * Snap-to-flowline pour point
* Stratifies basins across a 3D matrix:

  * Size: small (5–20), medium (20–100), large (100–500 km²)
  * Terrain: flat, moderate, steep
  * Complexity: 1–3 (based on stream density tertiles)
* Outputs:

  * `basin_sample.csv`
  * Optional: `sampled_basins.gpkg` (GeoPackage export)
  * Error logs for diagnostics

### 3.2 Truth Dataset Generator

**Module**: `truth_extractor.py`

* Input: `basin_sample.csv`, NHD+ HR watershed polygons (e.g., catchments.shp)
* For each basin:

  * Use pour point coordinates to find matching NHD catchment polygon
  * If needed, aggregate polygons upstream using flow direction data
  * Output: `truth_polygons.gpkg`

### 3.3 Benchmark Harness

**Module**: `benchmark_runner.py`

* Input: `truth_polygons.gpkg`, FLOWFINDER delineation output (GeoJSON)
* Metrics per basin:

  * Intersection over Union (IOU)
  * Boundary length ratio
  * Centroid distance offset
  * Processing time
* Outputs:

  * `benchmark_results.json`
  * `accuracy_summary.csv`
  * Optional: `accuracy_report.html` (Bokeh visualization)

### 3.4 Benchmark Config

**Config**: `benchmark_config.yaml`

* Parameters:

  * DEM path
  * Stream burn-in toggle
  * Snapping threshold
  * Max runtime threshold (seconds)
  * Output formats

---

## 4. Success Metrics

| Metric                    | Target                           |
| ------------------------- | -------------------------------- |
| IOU (mean)                | ≥ 0.90                           |
| IOU (90th percentile)     | ≥ 0.95                           |
| Runtime (mean)            | ≤ 30 s                           |
| Benchmark reproducibility | 100% (via GitHub repo + scripts) |
| Error-free basin coverage | ≥ 90% of selected sample         |

---

## 5. Timeline

| Day | Task                                                           |
| --- | -------------------------------------------------------------- |
| 1   | Build and run `basin_sampler.py` to generate 50 diverse basins |
| 2   | Extract ground truth polygons with `truth_extractor.py`        |
| 3   | Build `benchmark_runner.py`, implement IOU/metrics logic       |
| 4   | Run full benchmark, generate results JSON, analyze summary     |
| 5   | Visualize results, draft findings report, publish on GitHub    |

---

## 6. Architecture Overview

| Layer        | Component                     | Tech                           |
| ------------ | ----------------------------- | ------------------------------ |
| Data Input   | HUC12, NHD+ HR, DEM           | Shapefiles, GeoTIFF            |
| Processing   | Sampling, Masking, Slope Calc | GeoPandas, Rasterio, Shapely   |
| Benchmarking | IOU, performance stats        | NumPy, GeoPandas, custom logic |
| Output       | Reports, CSV, JSON            | Pandas, Jinja2, Bokeh (opt)    |

---

## 7. Risks & Mitigation

| Risk                           | Likelihood | Impact | Mitigation                       |
| ------------------------------ | ---------- | ------ | -------------------------------- |
| DEM/flowline misalignment      | Medium     | Medium | Snap threshold, burn-in          |
| Poor IOU on complex basins     | High       | High   | Document as known limitation     |
| Runtime above 30s with burn-in | Medium     | Medium | Flag during benchmark, optimize  |
| Missing "truth" polygons       | Medium     | Low    | Manual fallback for small subset |

---

## 8. Open Questions

* Should the benchmark use burned-in DEMs from the FLOWFINDER pipeline or raw 3DEP?
* What threshold is acceptable for centroid offset (meters)?
* How should urban/engineered basins be flagged or excluded?

---

## 9. Deliverables

* `basin_sample.csv`
* `truth_polygons.gpkg`
* `benchmark_results.json`
* `accuracy_summary.csv`
* Final report: accuracy distribution, bottlenecks, success envelope

---

## 10. License / Data Ethics

* All scripts licensed under MIT
* Attribution provided for all base datasets (NHD+ HR, 3DEP)
* No user data collected or processed

> *"Measure twice, delineate once. Truth earns trust."* 