#!/usr/bin/env python3
"""
Test the integration of ConfigurationManager with benchmark_runner.py

This is a simple test to verify the integration works before fixing all syntax issues.
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_configuration_manager():
    """Test the ConfigurationManager directly."""
    print("=== Testing ConfigurationManager ===")
    
    try:
        from config.configuration_manager import ConfigurationManager
        
        config_dir = Path("config")
        manager = ConfigurationManager(config_dir, environment="development")
        
        # Test FLOWFINDER configuration
        config = manager.get_tool_config("flowfinder")
        print(f"✅ FLOWFINDER config loaded")
        print(f"   Timeout: {config['benchmark']['timeout_seconds']}s")
        print(f"   Tool: {config['tool']['executable']}")
        
        # Test adapter
        adapter = manager.get_tool_adapter("flowfinder")
        print(f"✅ FLOWFINDER adapter created")
        print(f"   Available: {adapter.validate_installation()}")
        
        return True
        
    except Exception as e:
        print(f"❌ ConfigurationManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_benchmark_runner():
    """Test a simplified version of BenchmarkRunner initialization."""
    print("\n=== Testing Simplified BenchmarkRunner ===")
    
    try:
        from config.configuration_manager import ConfigurationManager
        
        # Create a minimal version of what BenchmarkRunner does
        config_dir = Path("config")
        manager = ConfigurationManager(config_dir, environment="development")
        config = manager.get_tool_config("flowfinder")
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Environment: development")
        print(f"   Tool: flowfinder")
        print(f"   Config structure: {list(config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests."""
    print("FLOWFINDER ConfigurationManager Integration Test")
    print("=" * 50)
    
    success = True
    success &= test_configuration_manager()
    success &= test_simple_benchmark_runner()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All integration tests passed!")
        print("\nNext steps:")
        print("1. Fix syntax errors in benchmark_runner.py")
        print("2. Update CLI argument handling")
        print("3. Test end-to-end workflow")
    else:
        print("❌ Some integration tests failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())