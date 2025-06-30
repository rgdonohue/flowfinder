#!/usr/bin/env python3
"""
FLOWFINDER Command Line Interface
================================

Command-line interface for FLOWFINDER watershed delineation tool.
Provides easy access to watershed delineation functionality.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .core import FlowFinder
from .exceptions import FlowFinderError, DEMError, WatershedError, ValidationError


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def save_geojson(polygon, output_path: str) -> None:
    """Save watershed polygon as GeoJSON."""
    try:
        # Convert polygon to GeoJSON format
        if polygon.is_empty:
            geojson = {
                "type": "FeatureCollection",
                "features": []
            }
        else:
            # Extract coordinates from polygon
            if hasattr(polygon, 'exterior'):
                coords = list(polygon.exterior.coords)
            else:
                # Handle MultiPolygon
                coords = list(polygon.geoms[0].exterior.coords)
            
            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "source": "FLOWFINDER",
                    "version": "1.0.0"
                }
            }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Watershed saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving GeoJSON: {e}", file=sys.stderr)
        sys.exit(1)


def save_shapefile(polygon, output_path: str) -> None:
    """Save watershed polygon as Shapefile."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Create GeoDataFrame
        if polygon.is_empty:
            gdf = gpd.GeoDataFrame(geometry=[])
        else:
            gdf = gpd.GeoDataFrame(geometry=[polygon])
            gdf.crs = "EPSG:4326"  # WGS84
        
        # Save to shapefile
        gdf.to_file(output_path)
        print(f"Watershed saved to: {output_path}")
        
    except ImportError:
        print("geopandas not available, cannot save shapefile", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error saving shapefile: {e}", file=sys.stderr)
        sys.exit(1)


def delineate_command(args) -> None:
    """Execute watershed delineation command."""
    try:
        # Validate input
        if not Path(args.dem).exists():
            print(f"DEM file not found: {args.dem}", file=sys.stderr)
            sys.exit(1)
        
        # Setup configuration
        config = {}
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Override config with command line arguments
        if args.timeout:
            config['timeout_seconds'] = args.timeout
        if args.method:
            config['flow_direction_method'] = args.method
        
        print(f"Loading DEM: {args.dem}")
        
        # Initialize FLOWFINDER
        with FlowFinder(args.dem, config) as flowfinder:
            print(f"Delineating watershed for point: ({args.lat}, {args.lon})")
            
            # Perform watershed delineation
            watershed = flowfinder.delineate_watershed(
                lat=args.lat, 
                lon=args.lon,
                timeout=args.timeout
            )
            
            # Calculate watershed properties
            area_km2 = watershed.area * 111 * 111  # Rough conversion to km²
            print(f"Watershed area: {area_km2:.2f} km²")
            
            # Save output
            if args.output:
                output_path = Path(args.output)
                
                if output_path.suffix.lower() == '.geojson':
                    save_geojson(watershed, args.output)
                elif output_path.suffix.lower() in ['.shp', '.shapefile']:
                    save_shapefile(watershed, args.output)
                else:
                    print(f"Unsupported output format: {output_path.suffix}", file=sys.stderr)
                    sys.exit(1)
            
            # Print summary
            print("\nWatershed delineation completed successfully!")
            print(f"Area: {area_km2:.2f} km²")
            print(f"Perimeter: {watershed.length * 111:.2f} km")
            
    except (DEMError, WatershedError, ValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def validate_command(args) -> None:
    """Execute DEM validation command."""
    try:
        print(f"Validating DEM: {args.dem}")
        
        # Initialize FLOWFINDER (this will validate the DEM)
        with FlowFinder(args.dem) as flowfinder:
            print("✓ DEM validation passed")
            print(f"  Resolution: {flowfinder.dem_data.res[0]}m")
            print(f"  Size: {flowfinder.dem_data.width}x{flowfinder.dem_data.height} pixels")
            print(f"  CRS: {flowfinder.dem_data.crs}")
            
    except (DEMError, ValidationError) as e:
        print(f"✗ DEM validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def info_command(args) -> None:
    """Execute DEM information command."""
    try:
        import rasterio
        
        with rasterio.open(args.dem) as dataset:
            print(f"DEM Information: {args.dem}")
            print(f"  Size: {dataset.width}x{dataset.height} pixels")
            print(f"  Resolution: {dataset.res[0]}m x {dataset.res[1]}m")
            print(f"  CRS: {dataset.crs}")
            print(f"  Bounds: {dataset.bounds}")
            print(f"  Data type: {dataset.dtypes[0]}")
            print(f"  No-data value: {dataset.nodata}")
            
            # Read a sample of data for statistics
            data = dataset.read(1)
            valid_data = data[data != dataset.nodata]
            
            if len(valid_data) > 0:
                print(f"  Elevation range: {valid_data.min():.1f}m - {valid_data.max():.1f}m")
                print(f"  Mean elevation: {valid_data.mean():.1f}m")
                print(f"  Valid pixels: {len(valid_data)}/{data.size} ({len(valid_data)/data.size*100:.1f}%)")
            
    except Exception as e:
        print(f"Error reading DEM: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER - Watershed Delineation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delineate watershed for a point
  flowfinder delineate --dem dem.tif --lat 40.0 --lon -105.0 --output watershed.geojson
  
  # Validate DEM file
  flowfinder validate --dem dem.tif
  
  # Get DEM information
  flowfinder info --dem dem.tif
  
  # Use custom configuration
  flowfinder delineate --dem dem.tif --lat 40.0 --lon -105.0 --config config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Delineate command
    delineate_parser = subparsers.add_parser(
        'delineate', 
        help='Delineate watershed for a pour point'
    )
    delineate_parser.add_argument('--dem', required=True, help='Path to DEM file')
    delineate_parser.add_argument('--lat', type=float, required=True, help='Latitude of pour point')
    delineate_parser.add_argument('--lon', type=float, required=True, help='Longitude of pour point')
    delineate_parser.add_argument('--output', help='Output file path (.geojson or .shp)')
    delineate_parser.add_argument('--config', help='Configuration file (YAML)')
    delineate_parser.add_argument('--timeout', type=float, help='Timeout in seconds')
    delineate_parser.add_argument('--method', choices=['d8', 'dinf', 'mfd'], help='Flow direction method')
    delineate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate', 
        help='Validate DEM file'
    )
    validate_parser.add_argument('--dem', required=True, help='Path to DEM file')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Info command
    info_parser = subparsers.add_parser(
        'info', 
        help='Display DEM information'
    )
    info_parser.add_argument('--dem', required=True, help='Path to DEM file')
    
    # Global options
    parser.add_argument('--version', action='version', version='FLOWFINDER 1.0.0')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)
    
    # Execute command
    if args.command == 'delineate':
        delineate_command(args)
    elif args.command == 'validate':
        validate_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 