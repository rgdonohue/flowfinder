#!/usr/bin/env python3
"""
Basic FLOWFINDER Test
====================

Simple test to verify FLOWFINDER can be imported and initialized.
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_dem():
    """Create a simple test DEM for testing."""
    import rasterio
    from rasterio.transform import from_origin
    
    # Create a simple 10x10 DEM with a valley
    dem_data = np.array([
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [100, 95,  90,  85,  80,  80,  85,  90,  95,  100],
        [100, 90,  80,  70,  60,  60,  70,  80,  90,  100],
        [100, 85,  70,  55,  40,  40,  55,  70,  85,  100],
        [100, 80,  60,  40,  20,  20,  40,  60,  80,  100],
        [100, 80,  60,  40,  20,  20,  40,  60,  80,  100],
        [100, 85,  70,  55,  40,  40,  55,  70,  85,  100],
        [100, 90,  80,  70,  60,  60,  70,  80,  90,  100],
        [100, 95,  90,  85,  80,  80,  85,  90,  95,  100],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    ], dtype=np.float32)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        dem_path = tmp.name
    
    # Write DEM to file
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=10,
        width=10,
        count=1,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=from_origin(-105.0, 40.0, 0.01, 0.01),
        nodata=-9999
    ) as dst:
        dst.write(dem_data, 1)
    
    return dem_path

def test_import():
    """Test that FLOWFINDER can be imported."""
    print("Testing FLOWFINDER import...")
    
    try:
        from flowfinder import FlowFinder
        print("✓ FLOWFINDER imported successfully")
        return True
    except ImportError as e:
        print(f"✗ FLOWFINDER import failed: {e}")
        return False

def test_initialization():
    """Test that FLOWFINDER can be initialized."""
    print("Testing FLOWFINDER initialization...")
    
    try:
        from flowfinder import FlowFinder
        
        # Create test DEM
        dem_path = create_test_dem()
        
        # Initialize FLOWFINDER
        with FlowFinder(dem_path) as flowfinder:
            print("✓ FLOWFINDER initialized successfully")
            print(f"  DEM size: {flowfinder.dem_data.width}x{flowfinder.dem_data.height}")
            print(f"  Resolution: {flowfinder.dem_data.res[0]}m")
        
        # Clean up
        Path(dem_path).unlink()
        return True
        
    except Exception as e:
        print(f"✗ FLOWFINDER initialization failed: {e}")
        return False

def test_basic_delineation():
    """Test basic watershed delineation."""
    print("Testing basic watershed delineation...")
    
    try:
        from flowfinder import FlowFinder
        
        # Create test DEM
        dem_path = create_test_dem()
        
        # Initialize FLOWFINDER
        with FlowFinder(dem_path) as flowfinder:
            # Test delineation at center point (should be a small watershed)
            watershed = flowfinder.delineate_watershed(lat=39.995, lon=-104.995)
            
            if watershed.is_empty:
                print("✗ Watershed delineation returned empty polygon")
                return False
            
            print("✓ Watershed delineation completed")
            print(f"  Watershed area: {watershed.area:.6f} degrees²")
            print(f"  Watershed perimeter: {watershed.length:.6f} degrees")
        
        # Clean up
        Path(dem_path).unlink()
        return True
        
    except Exception as e:
        print(f"✗ Watershed delineation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("FLOWFINDER Basic Tests")
    print("=" * 50)
    
    tests = [
        test_import,
        test_initialization,
        test_basic_delineation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! FLOWFINDER is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 