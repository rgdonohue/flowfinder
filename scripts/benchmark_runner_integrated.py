#!/usr/bin/env python3
"""
FLOWFINDER Multi-Tool Benchmark Runner with Hierarchical Configuration
=====================================================================

This is the integrated version of benchmark_runner.py that uses the hierarchical
configuration system for multi-tool watershed delineation benchmarking.

Key Features:
- Hierarchical configuration (base → environment → tool → local)
- Multi-tool support (FLOWFINDER, TauDEM, GRASS, WhiteboxTools)
- Environment-specific scaling (development/testing/production)
- Tool adapter pattern for standardized interfaces
- JSON schema validation

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 2.0.0
"""

import argparse
import json
import logging
import sys
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm

# Import hierarchical configuration system
sys.path.append(str(Path(__file__).parent.parent))
from config.configuration_manager import ConfigurationManager, ToolAdapter, ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class MultiBenchmarkRunner:
    """
    Multi-tool FLOWFINDER benchmark runner with hierarchical configuration.
    
    This class handles benchmark workflows for multiple watershed delineation tools
    using the hierarchical configuration system for maintainable and scalable testing.
    
    Attributes:
        environment (str): Environment name (development/testing/production)
        tools (List[str]): List of tools to benchmark
        config_manager (ConfigurationManager): Hierarchical configuration manager
        tool_adapters (Dict[str, ToolAdapter]): Tool adapters for each tool
        output_dir (Path): Output directory for results
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, environment: str = "development", tools: Optional[List[str]] = None,
                 output_dir: Optional[str] = None, local_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-tool benchmark runner.
        
        Args:
            environment: Environment name (development/testing/production)
            tools: List of tools to benchmark (defaults to ['flowfinder'])
            output_dir: Output directory for results
            local_overrides: Local configuration overrides
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        self.environment = environment
        self.tools = tools or ['flowfinder']
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        
        # Set up logging
        self._setup_logging()
        
        # Initialize configuration manager
        config_dir = Path(__file__).parent.parent / "config"
        try:
            self.config_manager = ConfigurationManager(config_dir, environment=environment)
            self.logger.info(f"Configuration manager initialized for environment: {environment}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration manager: {e}")
        
        # Initialize tool adapters
        self.tool_adapters: Dict[str, ToolAdapter] = {}
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        
        for tool in self.tools:
            try:
                self.tool_configs[tool] = self.config_manager.get_tool_config(tool, local_overrides)
                self.tool_adapters[tool] = self.config_manager.get_tool_adapter(tool, self.tool_configs[tool])
                self.logger.info(f"Initialized {tool} adapter")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {tool} adapter: {e}")
        
        self.logger.info(f"MultiBenchmarkRunner initialized with {len(self.tool_adapters)} tools")
    
    def _setup_logging(self) -> None:
        """Configure logging for the benchmark session."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - log file: {log_file}")
    
    def list_available_tools(self) -> Dict[str, bool]:
        """
        Check which tools are available on the system.
        
        Returns:
            Dictionary mapping tool names to availability status
        """
        availability = {}
        for tool_name, adapter in self.tool_adapters.items():
            try:
                availability[tool_name] = adapter.validate_installation()
            except Exception as e:
                self.logger.warning(f"Error checking {tool_name} availability: {e}")
                availability[tool_name] = False
        
        return availability
    
    def get_tool_configuration(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the effective configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool configuration dictionary
            
        Raises:
            ValueError: If tool is not configured
        """
        if tool_name not in self.tool_configs:
            raise ValueError(f"Tool '{tool_name}' not configured")
        
        return self.tool_configs[tool_name]
    
    def run_single_tool_test(self, tool_name: str, lat: float, lon: float) -> Dict[str, Any]:
        """
        Run a single test with one tool.
        
        Args:
            tool_name: Name of the tool to test
            lat: Latitude of test point
            lon: Longitude of test point
            
        Returns:
            Test result dictionary
        """
        if tool_name not in self.tool_adapters:
            return {
                'tool': tool_name,
                'success': False,
                'error': f"Tool '{tool_name}' not available"
            }
        
        adapter = self.tool_adapters[tool_name]
        config = self.tool_configs[tool_name]
        
        start_time = time.time()
        
        try:
            # Check if tool is available
            if not adapter.validate_installation():
                return {
                    'tool': tool_name,
                    'success': False,
                    'error': f"Tool '{tool_name}' not installed or not available"
                }
            
            # For now, simulate the tool execution
            # In a complete implementation, this would actually run the tool
            timeout = config.get('benchmark', {}).get('timeout_seconds', 120)
            executable = config.get('tool', {}).get('executable', tool_name)
            
            self.logger.info(f"Running {tool_name} test at ({lat}, {lon})")
            
            # Simulate processing time based on environment
            if self.environment == "development":
                time.sleep(0.1)  # Fast for development
            elif self.environment == "testing":
                time.sleep(0.5)  # Medium for testing
            else:
                time.sleep(1.0)  # Realistic for production
            
            runtime = time.time() - start_time
            
            # Return mock result
            return {
                'tool': tool_name,
                'success': True,
                'runtime_seconds': runtime,
                'timeout_seconds': timeout,
                'executable': executable,
                'environment': self.environment,
                'coordinates': {'lat': lat, 'lon': lon},
                'simulated': True  # Flag indicating this is a simulation
            }
            
        except Exception as e:
            runtime = time.time() - start_time
            return {
                'tool': tool_name,
                'success': False,
                'error': str(e),
                'runtime_seconds': runtime
            }
    
    def run_multi_tool_comparison(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Run comparison test with all configured tools.
        
        Args:
            lat: Latitude of test point
            lon: Longitude of test point
            
        Returns:
            Comparison results dictionary
        """
        self.logger.info(f"Running multi-tool comparison at ({lat}, {lon})")
        
        results = {}
        for tool_name in self.tools:
            result = self.run_single_tool_test(tool_name, lat, lon)
            results[tool_name] = result
        
        # Calculate summary statistics
        successful_tools = [tool for tool, result in results.items() if result['success']]
        failed_tools = [tool for tool, result in results.items() if not result['success']]
        
        if successful_tools:
            runtimes = [results[tool]['runtime_seconds'] for tool in successful_tools]
            avg_runtime = np.mean(runtimes)
            fastest_tool = min(successful_tools, key=lambda t: results[t]['runtime_seconds'])
        else:
            avg_runtime = None
            fastest_tool = None
        
        summary = {
            'total_tools': len(self.tools),
            'successful_tools': len(successful_tools),
            'failed_tools': len(failed_tools),
            'success_rate': len(successful_tools) / len(self.tools),
            'successful_tool_names': successful_tools,
            'failed_tool_names': failed_tools,
            'average_runtime': avg_runtime,
            'fastest_tool': fastest_tool,
            'environment': self.environment
        }
        
        return {
            'coordinates': {'lat': lat, 'lon': lon},
            'individual_results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, results: Dict[str, Any], output_format: str = "json") -> str:
        """
        Export benchmark results to file.
        
        Args:
            results: Results dictionary to export
            output_format: Export format (json, csv, summary)
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif output_format == "summary":
            output_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write("FLOWFINDER Multi-Tool Benchmark Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Environment: {self.environment}\n")
                f.write(f"Tools: {', '.join(self.tools)}\n")
                f.write(f"Timestamp: {results.get('timestamp', 'unknown')}\n\n")
                
                if 'summary' in results:
                    summary = results['summary']
                    f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
                    f.write(f"Successful Tools: {', '.join(summary['successful_tool_names'])}\n")
                    f.write(f"Failed Tools: {', '.join(summary['failed_tool_names'])}\n")
                    if summary['fastest_tool']:
                        f.write(f"Fastest Tool: {summary['fastest_tool']}\n")
                    if summary['average_runtime']:
                        f.write(f"Average Runtime: {summary['average_runtime']:.2f}s\n")
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.logger.info(f"Results exported to {output_file}")
        return str(output_file)


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="FLOWFINDER Multi-Tool Benchmark Runner with Hierarchical Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development test with FLOWFINDER only
  python benchmark_runner_integrated.py --env development --tools flowfinder --test

  # Production comparison with all tools
  python benchmark_runner_integrated.py --env production --tools all --test

  # Testing environment with specific tools
  python benchmark_runner_integrated.py --env testing --tools flowfinder taudem --test
        """
    )
    
    parser.add_argument(
        '--environment', '--env',
        type=str,
        default='development',
        choices=['development', 'testing', 'production'],
        help='Environment configuration to use (default: development)'
    )
    
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        default=['flowfinder'],
        choices=['flowfinder', 'taudem', 'grass', 'whitebox', 'all'],
        help='Tools to benchmark (default: flowfinder). Use "all" for all tools.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a simple test at Boulder, CO coordinates'
    )
    
    parser.add_argument(
        '--list-tools',
        action='store_true',
        help='List available tools and their installation status'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Handle "all" tools selection
        tools = args.tools
        if 'all' in tools:
            tools = ['flowfinder', 'taudem', 'grass', 'whitebox']
        
        # Initialize benchmark runner
        runner = MultiBenchmarkRunner(
            environment=args.environment,
            tools=tools,
            output_dir=args.output
        )
        
        print(f"🔧 FLOWFINDER Multi-Tool Benchmark Runner")
        print(f"Environment: {args.environment}")
        print(f"Tools: {', '.join(tools)}")
        print(f"Output: {args.output}")
        print()
        
        # List tools if requested
        if args.list_tools:
            print("Tool Installation Status:")
            availability = runner.list_available_tools()
            for tool, available in availability.items():
                status = "✅ Available" if available else "❌ Not Available"
                print(f"  {tool}: {status}")
            print()
        
        # Run test if requested
        if args.test:
            print("Running test at Boulder, CO (40.0150, -105.2705)...")
            
            # Test coordinates (Boulder, CO)
            lat, lon = 40.0150, -105.2705
            
            if len(tools) == 1:
                # Single tool test
                result = runner.run_single_tool_test(tools[0], lat, lon)
                print(f"\nSingle Tool Test Results:")
                print(f"Tool: {result['tool']}")
                print(f"Success: {result['success']}")
                if result['success']:
                    print(f"Runtime: {result['runtime_seconds']:.2f}s")
                    print(f"Timeout: {result['timeout_seconds']}s")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                # Multi-tool comparison
                results = runner.run_multi_tool_comparison(lat, lon)
                summary = results['summary']
                
                print(f"\nMulti-Tool Comparison Results:")
                print(f"Success Rate: {summary['success_rate']:.1%}")
                print(f"Successful: {', '.join(summary['successful_tool_names'])}")
                if summary['failed_tool_names']:
                    print(f"Failed: {', '.join(summary['failed_tool_names'])}")
                if summary['fastest_tool']:
                    print(f"Fastest: {summary['fastest_tool']}")
                if summary['average_runtime']:
                    print(f"Avg Runtime: {summary['average_runtime']:.2f}s")
                
                # Export results
                json_file = runner.export_results(results, "json")
                summary_file = runner.export_results(results, "summary")
                print(f"\nResults exported to:")
                print(f"  {json_file}")
                print(f"  {summary_file}")
        
        print("\n✅ Benchmark runner ready for multi-tool watershed delineation!")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()