#!/usr/bin/env python3
"""
FLOWFINDER Data Download Script
==============================

Downloads all required datasets for the FLOWFINDER accuracy benchmark:
- HUC12 watershed boundaries
- NHD+ High Resolution catchments and flowlines
- 3DEP 10m Digital Elevation Model tiles

Uses configuration from config/data_sources.yaml
"""

import argparse
import logging
import sys
import os
import yaml
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Handles downloading and organizing all required datasets."""
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.setup_directories()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load data sources configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def setup_directories(self):
        """Create required directories."""
        storage = self.config['storage']
        for dir_name in ['raw_data_dir', 'processed_data_dir', 'metadata_dir', 'temp_dir']:
            Path(storage[dir_name]).mkdir(parents=True, exist_ok=True)
        logger.info("Created data directories")
    
    def download_file(self, url: str, output_path: Path, description: str) -> bool:
        """Download a single file with progress tracking."""
        try:
            logger.info(f"Downloading {description} from {url}")
            
            # Check if file already exists
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                return True
            
            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            response = requests.get(url, stream=True, timeout=self.config['download']['timeout_seconds'])
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Downloaded {description}: {percent:.1f}%")
            
            logger.info(f"Successfully downloaded {description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {description}: {e}")
            return False
    
    def download_huc12(self) -> bool:
        """Download HUC12 watershed boundaries."""
        huc12_config = self.config['data_sources']['huc12']
        url = huc12_config['url']
        output_path = Path(self.config['storage']['raw_data_dir']) / "huc12" / "WBD_National_GDB.zip"
        
        success = self.download_file(url, output_path, "HUC12 watershed boundaries")
        
        if success:
            # Extract zip file
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    extract_dir = output_path.parent
                    zip_ref.extractall(extract_dir)
                logger.info("Extracted HUC12 data")
                return True
            except Exception as e:
                logger.error(f"Failed to extract HUC12 data: {e}")
                return False
        
        return False
    
    def download_nhd_hr(self) -> bool:
        """Download NHD+ High Resolution data for Mountain West."""
        nhd_config = self.config['data_sources']['nhd_hr_catchments']
        base_url = nhd_config['base_url']
        huc4_regions = nhd_config['huc4_regions']
        
        download_config = self.config['download']
        max_workers = download_config['max_concurrent_downloads']
        
        def download_huc4_region(huc4):
            url = f"{base_url}NHDPlus_H_{huc4}_HU4_GDB.zip"
            output_path = Path(self.config['storage']['raw_data_dir']) / "nhd_hr" / f"NHDPlus_H_{huc4}_HU4_GDB.zip"
            return self.download_file(url, output_path, f"NHD+ HR for HUC4 {huc4}")
        
        logger.info(f"Downloading NHD+ HR data for {len(huc4_regions)} HUC4 regions")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_huc4 = {executor.submit(download_huc4_region, huc4): huc4 for huc4 in huc4_regions}
            
            for future in as_completed(future_to_huc4):
                huc4 = future_to_huc4[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        # Extract zip file
                        output_path = Path(self.config['storage']['raw_data_dir']) / "nhd_hr" / f"NHDPlus_H_{huc4}_HU4_GDB.zip"
                        if output_path.exists():
                            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                                extract_dir = output_path.parent
                                zip_ref.extractall(extract_dir)
                            logger.info(f"Extracted NHD+ HR data for HUC4 {huc4}")
                except Exception as e:
                    logger.error(f"Error downloading HUC4 {huc4}: {e}")
        
        logger.info(f"Downloaded {success_count}/{len(huc4_regions)} NHD+ HR regions")
        return success_count == len(huc4_regions)
    
    def download_dem(self) -> bool:
        """Download 3DEP 10m DEM tiles for Mountain West."""
        dem_config = self.config['data_sources']['dem_10m']
        base_url = dem_config['base_url']
        required_tiles = dem_config['required_tiles']
        
        download_config = self.config['download']
        max_workers = download_config['max_concurrent_downloads']
        
        def download_dem_tile(tile_path):
            url = f"{base_url}{tile_path}/USGS_{tile_path.replace('/', '_')}_20221115.tif"
            output_path = Path(self.config['storage']['raw_data_dir']) / "dem" / f"USGS_{tile_path.replace('/', '_')}_20221115.tif"
            return self.download_file(url, output_path, f"DEM tile {tile_path}")
        
        logger.info(f"Downloading 10m DEM data for {len(required_tiles)} tiles")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {executor.submit(download_dem_tile, tile): tile for tile in required_tiles}
            
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error downloading DEM tile {tile}: {e}")
        
        logger.info(f"Downloaded {success_count}/{len(required_tiles)} DEM tiles")
        return success_count == len(required_tiles)
    
    def validate_downloads(self) -> Dict[str, bool]:
        """Validate that all required files were downloaded."""
        validation_results = {}
        
        # Check HUC12
        huc12_path = Path(self.config['storage']['raw_data_dir']) / "huc12"
        validation_results['huc12'] = huc12_path.exists() and any(huc12_path.glob("*.shp"))
        
        # Check NHD+ HR
        nhd_path = Path(self.config['storage']['raw_data_dir']) / "nhd_hr"
        validation_results['nhd_hr'] = nhd_path.exists() and any(nhd_path.glob("*.gdb"))
        
        # Check DEM
        dem_path = Path(self.config['storage']['raw_data_dir']) / "dem"
        validation_results['dem'] = dem_path.exists() and any(dem_path.glob("*.tif"))
        
        return validation_results
    
    def generate_metadata(self):
        """Generate metadata about downloaded datasets."""
        metadata = {
            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_sources': self.config['data_sources'],
            'validation_results': self.validate_downloads(),
            'storage_info': {}
        }
        
        # Calculate storage usage
        for dir_name, dir_path in self.config['storage'].items():
            if dir_name.endswith('_dir'):
                path = Path(dir_path)
                if path.exists():
                    total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    metadata['storage_info'][dir_name] = {
                        'path': str(path),
                        'size_bytes': total_size,
                        'size_mb': total_size / (1024 * 1024)
                    }
        
        # Save metadata
        metadata_path = Path(self.config['storage']['metadata_dir']) / "download_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated metadata: {metadata_path}")
    
    def download_all(self) -> bool:
        """Download all required datasets."""
        logger.info("Starting download of all FLOWFINDER benchmark datasets")
        
        start_time = time.time()
        
        # Download each dataset
        huc12_success = self.download_huc12()
        nhd_success = self.download_nhd_hr()
        dem_success = self.download_dem()
        
        # Validate downloads
        validation_results = self.validate_downloads()
        
        # Generate metadata
        self.generate_metadata()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Report results
        logger.info("=" * 50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"HUC12 watershed boundaries: {'✓' if huc12_success else '✗'}")
        logger.info(f"NHD+ High Resolution data: {'✓' if nhd_success else '✗'}")
        logger.info(f"3DEP 10m DEM tiles: {'✓' if dem_success else '✗'}")
        logger.info(f"Total download time: {duration:.1f} seconds")
        logger.info(f"Validation results: {validation_results}")
        
        all_success = huc12_success and nhd_success and dem_success
        if all_success:
            logger.info("✓ All datasets downloaded successfully!")
        else:
            logger.warning("⚠ Some datasets failed to download")
        
        return all_success


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download all required datasets for FLOWFINDER benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_data.py --all
  
  # Download specific dataset
  python download_data.py --dataset huc12
  
  # Use custom config
  python download_data.py --config custom_data_sources.yaml --all
        """
    )
    
    parser.add_argument('--config', '-c', default='config/data_sources.yaml',
                       help='Path to data sources configuration file')
    parser.add_argument('--all', action='store_true',
                       help='Download all required datasets')
    parser.add_argument('--dataset', choices=['huc12', 'nhd_hr', 'dem'],
                       help='Download specific dataset only')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing downloads without downloading')
    
    args = parser.parse_args()
    
    try:
        downloader = DataDownloader(args.config)
        
        if args.validate:
            validation_results = downloader.validate_downloads()
            logger.info("Validation results:")
            for dataset, valid in validation_results.items():
                logger.info(f"  {dataset}: {'✓' if valid else '✗'}")
            return
        
        if args.all:
            success = downloader.download_all()
            sys.exit(0 if success else 1)
        elif args.dataset:
            if args.dataset == 'huc12':
                success = downloader.download_huc12()
            elif args.dataset == 'nhd_hr':
                success = downloader.download_nhd_hr()
            elif args.dataset == 'dem':
                success = downloader.download_dem()
            
            sys.exit(0 if success else 1)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 