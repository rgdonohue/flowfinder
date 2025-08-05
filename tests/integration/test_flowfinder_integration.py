#!/usr/bin/env python3
"""
Test FLOWFINDER CLI Integration
===============================

Simple test to verify FLOWFINDER command availability and benchmark runner integration.
"""

import subprocess
import sys
import os
from pathlib import Path


def test_flowfinder_cli():
    """Test if FLOWFINDER CLI is available."""
    print("Testing FLOWFINDER CLI availability...")

    try:
        result = subprocess.run(
            ["flowfinder", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✓ FLOWFINDER CLI is working!")
            print("Output preview:")
            print(
                result.stdout[:200] + "..."
                if len(result.stdout) > 200
                else result.stdout
            )
            return True
        else:
            print(f"✗ FLOWFINDER CLI failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ FLOWFINDER command not found")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FLOWFINDER command timed out")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_python_module_access():
    """Test if FLOWFINDER can be imported as Python module."""
    print("\nTesting FLOWFINDER Python module access...")

    # Add flowfinder directory to path
    flowfinder_dir = Path(__file__).parent / "flowfinder"
    sys.path.insert(0, str(flowfinder_dir))

    try:
        # Try importing the CLI module directly
        import cli

        print("✓ FLOWFINDER CLI module can be imported")
        return True
    except ImportError as e:
        print(f"✗ Cannot import FLOWFINDER CLI module: {e}")
        return False


def test_benchmark_integration():
    """Test benchmark runner's FLOWFINDER integration logic."""
    print("\nTesting benchmark runner FLOWFINDER integration...")

    # Test command construction
    lat, lon = 40.0, -105.0
    cli_config = {
        "command": "flowfinder",
        "subcommand": "delineate",
        "output_format": "geojson",
        "additional_args": [],
    }

    cmd = [
        cli_config["command"],
        cli_config["subcommand"],
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--output-format",
        cli_config["output_format"],
    ]

    print(f"Would execute command: {' '.join(cmd)}")

    # Test actual execution
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ FLOWFINDER execution succeeded!")
            return True
        else:
            print(f"✗ FLOWFINDER execution failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ FLOWFINDER command not found - will use mock results")
        return False
    except Exception as e:
        print(f"✗ FLOWFINDER execution error: {e}")
        return False


if __name__ == "__main__":
    print("FLOWFINDER Integration Test")
    print("=" * 40)

    # Run tests
    cli_works = test_flowfinder_cli()
    module_works = test_python_module_access()
    integration_works = test_benchmark_integration()

    print("\nSummary:")
    print("=" * 40)
    print(f"FLOWFINDER CLI available: {'✓' if cli_works else '✗'}")
    print(f"Python module import: {'✓' if module_works else '✗'}")
    print(f"Benchmark integration: {'✓' if integration_works else '✗'}")

    if not cli_works:
        print(
            "\nCONCLUSION: FLOWFINDER CLI not available - benchmark will use mock results"
        )
        print("This is expected behavior for testing the benchmark infrastructure.")
    else:
        print("\nCONCLUSION: FLOWFINDER CLI is ready for real watershed delineation!")
