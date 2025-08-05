#!/usr/bin/env python3
"""
FLOWFINDER Accuracy Benchmark - Basin Selection Script
Stratified sampling of 50 diverse watershed basins across Mountain West

Production-grade tool with structured logging, optimized spatial operations,
and configurable parameters for reproducible benchmarking.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
from shapely.strtree import STRtree
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class BasinSampler:
    def __init__(self, config_path: str = None, data_dir: str = None):
        """Initialize with configuration file or data directory"""
        self.config = self.load_config(config_path, data_dir)
        self.logs = []  # Structured error tracking
        self.mountain_west_states = self.config.get(
            "mountain_west_states", ["CO", "UT", "NM", "WY", "MT", "ID", "AZ"]
        )

        # Set up logging
        self.setup_logging()

    def load_config(self, config_path: str = None, data_dir: str = None) -> dict:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            "data_dir": data_dir or "data",
            "area_range": [5, 500],  # km²
            "snap_tolerance": 150,  # meters
            "n_per_stratum": 2,
            "target_crs": "EPSG:5070",  # Albers Equal Area CONUS
            "output_crs": "EPSG:4326",  # WGS84 for lat/lon export
            "files": {
                "huc12": "huc12_mountain_west.shp",
                "catchments": "nhd_hr_catchments.shp",
                "flowlines": "nhd_flowlines.shp",
                "dem": "dem_10m.tif",
                "slope": None,  # Optional pre-computed slope raster
            },
            "export": {"csv": True, "gpkg": True, "summary": True},
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)

        return default_config

    def setup_logging(self):
        """Configure logging for the session"""
        log_file = f"basin_sampler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def log_error(self, basin_id: str, error_type: str, message: str):
        """Log structured error for later analysis"""
        error_record = {
            "basin_id": basin_id,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.logs.append(error_record)
        self.logger.warning(f"Basin {basin_id} - {error_type}: {message}")

    def load_datasets(self):
        """Load all required spatial datasets with proper CRS handling"""
        data_dir = self.config["data_dir"]
        files = self.config["files"]
        target_crs = self.config["target_crs"]

        self.logger.info("Loading spatial datasets...")

        try:
            self.logger.info("Loading HUC12 boundaries...")
            self.huc12 = gpd.read_file(f"{data_dir}/{files['huc12']}")

            self.logger.info("Loading NHD+ HR catchments...")
            self.catchments = gpd.read_file(f"{data_dir}/{files['catchments']}")

            self.logger.info("Loading NHD flowlines...")
            self.flowlines = gpd.read_file(f"{data_dir}/{files['flowlines']}")

            self.logger.info(f"Loading DEM: {files['dem']}")
            self.dem = rasterio.open(f"{data_dir}/{files['dem']}")

            # Load optional pre-computed slope raster
            if files.get("slope"):
                self.logger.info(f"Loading pre-computed slope: {files['slope']}")
                self.slope_raster = rasterio.open(f"{data_dir}/{files['slope']}")
            else:
                self.slope_raster = None

            # Transform to target CRS (Albers Equal Area for accurate area calculations)
            self.logger.info(f"Transforming to {target_crs}...")
            self.huc12 = self.huc12.to_crs(target_crs)
            self.catchments = self.catchments.to_crs(target_crs)
            self.flowlines = self.flowlines.to_crs(target_crs)

            self.logger.info("✅ All datasets loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise

    def filter_mountain_west_basins(self):
        """Filter to Mountain West region and size constraints"""
        # Filter HUC12s by state codes (assuming STATES field exists)
        if "STATES" in self.huc12.columns:
            mw_mask = self.huc12["STATES"].str.contains(
                "|".join(self.mountain_west_states), na=False
            )
            self.huc12 = self.huc12[mw_mask]

        # Calculate areas in km² (accurate with Albers Equal Area projection)
        self.huc12["area_km2"] = self.huc12.geometry.area / 1e6

        # Filter to target size range
        min_area, max_area = self.config["area_range"]
        size_mask = (self.huc12["area_km2"] >= min_area) & (
            self.huc12["area_km2"] <= max_area
        )
        self.huc12 = self.huc12[size_mask].copy()

        self.logger.info(
            f"Filtered to {len(self.huc12)} Mountain West basins ({min_area}-{max_area} km²)"
        )
        self.logger.info(
            f"Area calculation CRS: {self.config['target_crs']} (Albers Equal Area CONUS)"
        )

    def compute_pour_points(self):
        """Snap basin centroids to nearest flowline points using optimized spatial index"""
        self.logger.info("Computing pour points with STRtree optimization...")
        snap_tolerance = self.config["snap_tolerance"]

        pour_points = []

        for idx, basin in self.huc12.iterrows():
            basin_id = str(basin.get("HUC12", idx))

            try:
                centroid = basin.geometry.centroid

                # Find flowlines intersecting this basin
                basin_flowlines = self.flowlines[
                    self.flowlines.intersects(basin.geometry)
                ]

                if len(basin_flowlines) > 0:
                    # Extract all coordinate points from flowlines
                    all_points = []
                    for _, line in basin_flowlines.iterrows():
                        if hasattr(line.geometry, "coords"):
                            coords = list(line.geometry.coords)
                        else:  # MultiLineString
                            coords = []
                            for geom in line.geometry.geoms:
                                coords.extend(list(geom.coords))

                        all_points.extend([Point(coord) for coord in coords])

                    if all_points:
                        # Use STRtree for fast nearest neighbor search
                        tree = STRtree(all_points)
                        nearest_point = tree.nearest(centroid)

                        # Check distance tolerance
                        distance = centroid.distance(nearest_point)
                        if distance <= snap_tolerance:
                            pour_point = nearest_point
                        else:
                            pour_point = centroid
                            self.log_error(
                                basin_id,
                                "snap_tolerance",
                                f"Nearest flowline {distance:.1f}m > {snap_tolerance}m tolerance",
                            )
                    else:
                        pour_point = centroid
                        self.log_error(
                            basin_id,
                            "no_coordinates",
                            "No valid flowline coordinates found",
                        )
                else:
                    pour_point = centroid
                    self.log_error(
                        basin_id, "no_flowlines", "No intersecting flowlines found"
                    )

                pour_points.append(pour_point)

            except Exception as e:
                pour_points.append(basin.geometry.centroid)
                self.log_error(basin_id, "processing_error", str(e))

        self.huc12["pour_point"] = pour_points
        self.logger.info(f"Computed {len(pour_points)} pour points")

    def compute_terrain_roughness(self):
        """Calculate slope standard deviation using DEM or pre-computed slope raster"""
        if self.slope_raster:
            self.logger.info(
                "Computing terrain roughness from pre-computed slope raster..."
            )
            source_raster = self.slope_raster
            compute_slope = False
        else:
            self.logger.info(
                "Computing terrain roughness from DEM (calculating slope)..."
            )
            source_raster = self.dem
            compute_slope = True

        roughness_scores = []

        for idx, basin in self.huc12.iterrows():
            basin_id = str(basin.get("HUC12", idx))

            try:
                # Mask raster to basin boundary
                basin_geom = [basin.geometry.__geo_interface__]
                masked_data, _ = mask(
                    source_raster, basin_geom, crop=True, nodata=np.nan
                )

                if masked_data.size > 0 and not np.all(np.isnan(masked_data)):
                    raster_data = masked_data[0]  # First band
                    valid_mask = ~np.isnan(raster_data)

                    if np.sum(valid_mask) > 10:  # Need minimum valid pixels
                        if compute_slope:
                            # Calculate slope from DEM using numpy gradient
                            dy, dx = np.gradient(raster_data)
                            slope_data = np.sqrt(dx**2 + dy**2)
                        else:
                            # Use pre-computed slope directly
                            slope_data = raster_data

                        # Standard deviation of slope (terrain roughness)
                        slope_std = np.nanstd(slope_data[valid_mask])
                        roughness_scores.append(
                            slope_std if not np.isnan(slope_std) else 0
                        )
                    else:
                        roughness_scores.append(0)
                        self.log_error(
                            basin_id, "insufficient_pixels", "< 10 valid DEM pixels"
                        )
                else:
                    roughness_scores.append(0)
                    self.log_error(
                        basin_id, "no_raster_data", "No valid raster data in basin"
                    )

            except Exception as e:
                roughness_scores.append(0)
                self.log_error(basin_id, "terrain_processing_error", str(e))

        self.huc12["slope_std"] = roughness_scores
        self.logger.info(
            f"Computed terrain roughness for {len(roughness_scores)} basins"
        )

    def compute_stream_complexity(self):
        """Calculate stream complexity: total stream length / basin area"""
        self.logger.info("Computing stream complexity...")

        complexity_scores = []

        for idx, basin in self.huc12.iterrows():
            basin_id = str(basin.get("HUC12", idx))

            try:
                # Find flowlines within basin
                basin_streams = self.flowlines[
                    self.flowlines.intersects(basin.geometry)
                ]

                if len(basin_streams) > 0:
                    # Calculate total stream length in basin
                    total_length = 0
                    for _, stream in basin_streams.iterrows():
                        # Clip stream to basin boundary
                        clipped = stream.geometry.intersection(basin.geometry)
                        if hasattr(clipped, "length"):
                            total_length += clipped.length

                    # Stream density: km of stream per km² of basin
                    stream_density = (total_length / 1000) / basin["area_km2"]
                    complexity_scores.append(stream_density)
                else:
                    complexity_scores.append(0)
                    self.log_error(
                        basin_id, "no_streams", "No intersecting streams found"
                    )

            except Exception as e:
                complexity_scores.append(0)
                self.log_error(basin_id, "complexity_error", str(e))

        self.huc12["stream_density"] = complexity_scores
        self.logger.info(
            f"Computed stream complexity for {len(complexity_scores)} basins"
        )

    def classify_basins(self):
        """Classify basins into terrain and complexity categories"""
        self.logger.info("Classifying basins into stratification categories...")

        # Terrain classification (slope std tertiles)
        slope_33 = self.huc12["slope_std"].quantile(0.33)
        slope_67 = self.huc12["slope_std"].quantile(0.67)

        def terrain_class(slope_std):
            if slope_std <= slope_33:
                return "flat"
            elif slope_std <= slope_67:
                return "moderate"
            else:
                return "steep"

        self.huc12["terrain_class"] = self.huc12["slope_std"].apply(terrain_class)

        # Size classification
        def size_class(area):
            if area < 20:
                return "small"
            elif area < 100:
                return "medium"
            else:
                return "large"

        self.huc12["size_class"] = self.huc12["area_km2"].apply(size_class)

        # Complexity classification (stream density tertiles)
        dens_33 = self.huc12["stream_density"].quantile(0.33)
        dens_67 = self.huc12["stream_density"].quantile(0.67)

        def complexity_score(density):
            if density <= dens_33:
                return 1
            elif density <= dens_67:
                return 2
            else:
                return 3

        self.huc12["complexity_score"] = self.huc12["stream_density"].apply(
            complexity_score
        )

        # Log classification thresholds for reproducibility
        self.logger.info(
            f"Terrain thresholds - Flat: ≤{slope_33:.3f}, Moderate: ≤{slope_67:.3f}, Steep: >{slope_67:.3f}"
        )
        self.logger.info(
            f"Complexity thresholds - Low: ≤{dens_33:.3f}, Med: ≤{dens_67:.3f}, High: >{dens_67:.3f}"
        )

    def stratified_sample(self):
        """Perform stratified sampling across size×terrain×complexity cube"""
        n_per_stratum = self.config["n_per_stratum"]
        self.logger.info(
            f"Performing stratified sampling ({n_per_stratum} per stratum)..."
        )

        # Group by all three dimensions
        grouped = self.huc12.groupby(
            ["size_class", "terrain_class", "complexity_score"]
        )

        sampled_basins = []
        stratum_summary = []

        for name, group in grouped:
            if len(group) >= n_per_stratum:
                # Random sample
                sample = group.sample(n=n_per_stratum, random_state=42)
            else:
                # Take all if fewer than target
                sample = group

            sampled_basins.append(sample)
            size_cls, terrain_cls, complexity = name
            stratum_info = f"{size_cls}/{terrain_cls}/complexity-{complexity}: {len(sample)} basins"
            stratum_summary.append(stratum_info)
            self.logger.info(f"  {stratum_info}")

        self.sample = pd.concat(sampled_basins, ignore_index=True)
        self.logger.info(f"Total sample: {len(self.sample)} basins")

        return stratum_summary

    def export_sample(self, output_prefix="basin_sample"):
        """Export final sample to CSV and optionally GeoPackage"""
        export_config = self.config["export"]
        output_crs = self.config["output_crs"]

        # Extract lat/lon from pour points (convert to WGS84)
        pour_points_wgs84 = gpd.GeoSeries(
            self.sample["pour_point"], crs=self.config["target_crs"]
        ).to_crs(output_crs)

        export_df = pd.DataFrame(
            {
                "ID": self.sample.index,
                "HUC12": self.sample.get("HUC12", self.sample.index),
                "Pour_Point_Lat": pour_points_wgs84.y,
                "Pour_Point_Lon": pour_points_wgs84.x,
                "Area_km2": self.sample["area_km2"].round(2),
                "Terrain_Class": self.sample["terrain_class"],
                "Complexity_Score": self.sample["complexity_score"],
                "Slope_Std": self.sample["slope_std"].round(3),
                "Stream_Density": self.sample["stream_density"].round(3),
                "Notes": "",
            }
        )

        outputs = []

        # Export CSV
        if export_config.get("csv", True):
            csv_path = f"{output_prefix}.csv"
            export_df.to_csv(csv_path, index=False)
            outputs.append(csv_path)
            self.logger.info(f"✅ Exported CSV: {csv_path}")

        # Export GeoPackage for spatial analysis
        if export_config.get("gpkg", True):
            gpkg_path = f"{output_prefix}.gpkg"
            sample_gdf = self.sample.copy()
            sample_gdf = sample_gdf.to_crs(output_crs)
            sample_gdf.to_file(gpkg_path, driver="GPKG")
            outputs.append(gpkg_path)
            self.logger.info(f"✅ Exported GeoPackage: {gpkg_path}")

        # Export error log
        if self.logs:
            errors_path = f"{output_prefix}_errors.csv"
            errors_df = pd.DataFrame(self.logs)
            errors_df.to_csv(errors_path, index=False)
            outputs.append(errors_path)
            self.logger.warning(
                f"⚠️  Exported {len(self.logs)} errors to: {errors_path}"
            )

        # Export summary
        if export_config.get("summary", True):
            summary_path = f"{output_prefix}_summary.txt"
            self._write_summary(summary_path, export_df)
            outputs.append(summary_path)
            self.logger.info(f"📊 Exported summary: {summary_path}")

        return outputs

    def _write_summary(self, summary_path: str, export_df: pd.DataFrame):
        """Write detailed summary of sampling results"""
        with open(summary_path, "w") as f:
            f.write("FLOWFINDER Basin Selection Summary\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config.get('config_file', 'default')}\n")
            f.write(f"Area calculation CRS: {self.config['target_crs']}\n")
            f.write(f"Export CRS: {self.config['output_crs']}\n\n")

            f.write(f"Total basins selected: {len(export_df)}\n")
            f.write(f"Processing errors: {len(self.logs)}\n\n")

            f.write("Area Distribution (km²):\n")
            f.write(str(export_df["Area_km2"].describe().round(1)) + "\n\n")

            f.write("Terrain Classes:\n")
            f.write(str(export_df["Terrain_Class"].value_counts()) + "\n\n")

            f.write("Complexity Scores:\n")
            f.write(str(export_df["Complexity_Score"].value_counts()) + "\n\n")

            if self.logs:
                f.write("Error Summary:\n")
                error_types = pd.DataFrame(self.logs)["error_type"].value_counts()
                f.write(str(error_types) + "\n")

        self.logger.info(f"Summary written to {summary_path}")


def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "data_dir": "data",
        "area_range": [5, 500],
        "snap_tolerance": 150,
        "n_per_stratum": 2,
        "target_crs": "EPSG:5070",
        "output_crs": "EPSG:4326",
        "mountain_west_states": ["CO", "UT", "NM", "WY", "MT", "ID", "AZ"],
        "files": {
            "huc12": "huc12_mountain_west.shp",
            "catchments": "nhd_hr_catchments.shp",
            "flowlines": "nhd_flowlines.shp",
            "dem": "dem_10m.tif",
            "slope": None,  # Optional: 'slope_10m.tif'
        },
        "export": {"csv": True, "gpkg": True, "summary": True},
    }

    with open("basin_sampler_config.yaml", "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)

    print("✅ Created sample configuration: basin_sampler_config.yaml")


def main():
    """Main execution pipeline with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Basin Selection Tool - Stratified sampling for watershed benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with data directory
  python basin_sampler.py --data-dir data/

  # Use configuration file
  python basin_sampler.py --config basin_config.yaml

  # Create sample configuration
  python basin_sampler.py --create-config

  # Custom output and settings
  python basin_sampler.py --data-dir data/ --output basin_selection --n-per-stratum 3
        """,
    )

    parser.add_argument(
        "--data-dir", "-d", type=str, help="Directory containing input spatial datasets"
    )
    parser.add_argument("--config", "-c", type=str, help="YAML configuration file path")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="basin_sample",
        help="Output file prefix (default: basin_sample)",
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        help="Number of basins per stratum (overrides config)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create sample configuration file and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create sample config and exit
    if args.create_config:
        create_sample_config()
        return

    # Validate inputs
    if not args.config and not args.data_dir:
        parser.error("Either --config or --data-dir must be specified")

    try:
        # Initialize sampler
        sampler = BasinSampler(config_path=args.config, data_dir=args.data_dir)

        # Override config with CLI arguments
        if args.n_per_stratum:
            sampler.config["n_per_stratum"] = args.n_per_stratum

        # Set verbose logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Execute pipeline
        sampler.load_datasets()
        sampler.filter_mountain_west_basins()
        sampler.compute_pour_points()
        sampler.compute_terrain_roughness()
        sampler.compute_stream_complexity()
        sampler.classify_basins()
        sampler.stratified_sample()
        outputs = sampler.export_sample(args.output)

        # Final summary
        sampler.logger.info("🎯 Basin selection complete!")
        sampler.logger.info(f"Outputs: {', '.join(outputs)}")

        if sampler.logs:
            sampler.logger.warning(
                f"⚠️  {len(sampler.logs)} processing errors occurred - see error log"
            )
        else:
            sampler.logger.info("✨ All basins processed successfully")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
