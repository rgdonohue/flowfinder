# Configuration Architecture Examples

*Practical examples of the proposed hierarchical configuration system*

## Configuration Inheritance Examples

### Base Configuration Files

#### `config/base/regions.yaml`
```yaml
# Geographic regions and boundaries
regions:
  mountain_west_full:
    name: "Mountain West - Full Region"
    states: ["CO", "UT", "NM", "WY", "MT", "ID", "AZ"]
    bounding_box:
      west: -116.0
      east: -104.0
      south: 31.0
      north: 49.0
    basin_count_estimate: 2500
    
  mountain_west_minimal:
    name: "Mountain West - Development Subset"
    states: ["CO"]
    bounding_box:
      west: -105.7
      east: -105.0
      south: 39.9
      north: 40.3
    basin_count_estimate: 10
    
  colorado_front_range:
    name: "Colorado Front Range"
    states: ["CO"]
    bounding_box:
      west: -105.8
      east: -104.5
      south: 39.5
      north: 40.5
    basin_count_estimate: 150
```

#### `config/base/quality_standards.yaml`
```yaml
# Accuracy thresholds and metrics
accuracy_thresholds:
  research_grade:
    iou_thresholds:
      flat: 0.95
      moderate: 0.92
      steep: 0.85
    centroid_thresholds:
      flat: 200
      moderate: 500
      steep: 1000
      
  development_grade:
    iou_thresholds:
      flat: 0.80
      moderate: 0.75
      steep: 0.70
    centroid_thresholds:
      flat: 500
      moderate: 1000
      steep: 2000

metrics:
  primary:
    - "iou"
    - "centroid_offset"
    - "runtime"
  secondary:
    - "boundary_ratio"
    - "area_ratio"
    - "shape_complexity"
```

### Environment-Specific Configurations

#### `config/environments/development.yaml`
```yaml
# Development environment - fast, minimal data
environment: "development"
data_scale: "minimal"
performance_priority: "speed"

data_sources:
  scale_factor: 0.01  # 1% of full dataset
  max_basins: 10
  timeout_seconds: 30
  
storage:
  max_disk_usage_gb: 1
  cleanup_temp_files: true
  
quality_standards: "development_grade"
```

#### `config/environments/production.yaml`
```yaml
# Production environment - comprehensive, full datasets
environment: "production"
data_scale: "full"
performance_priority: "accuracy"

data_sources:
  scale_factor: 1.0  # Full dataset
  max_basins: null  # No limit
  timeout_seconds: 300
  
storage:
  max_disk_usage_gb: 100
  cleanup_temp_files: false
  
quality_standards: "research_grade"
```

### Tool-Specific Configurations

#### `config/tools/flowfinder/base.yaml`
```yaml
# FLOWFINDER tool configuration
tool:
  name: "FLOWFINDER"
  version: "1.0.0"
  type: "python_package"
  
command:
  executable: "flowfinder"
  subcommand: "delineate"
  output_format: "geojson"
  
algorithms:
  flow_direction: "d8"  # d8, dinf, mfd
  flow_accumulation: "standard"
  depression_filling: "priority_flood"
  
parameters:
  min_area: 1000  # minimum watershed area (pixels)
  snap_threshold: 150  # meters
  
performance:
  memory_limit_gb: 4
  cpu_cores: 1
  timeout_seconds: 120
```

#### `config/tools/taudem/base.yaml`
```yaml
# TauDEM tool configuration
tool:
  name: "TauDEM"
  version: "5.3.7"
  type: "executable"
  
command:
  executable: "mpiexec"
  base_args: ["-n", "4"]  # 4 MPI processes
  
modules:
  pitfill: "pitremove"
  d8flowdir: "d8flowdir"
  aread8: "aread8"
  threshold: "threshold"
  streamnet: "streamnet"
  
algorithms:
  flow_direction: "d8"
  depression_filling: "pitfill"
  
parameters:
  threshold_area: 1000
  stream_threshold: 100
  
performance:
  mpi_processes: 4
  memory_per_process_gb: 2
  timeout_seconds: 300
```

### Experiment Configurations

#### `config/experiments/accuracy_comparison.yaml`
```yaml
# Multi-tool accuracy comparison experiment
experiment:
  name: "Multi-Tool Accuracy Comparison"
  type: "accuracy_benchmark"
  
tools:
  - "flowfinder"
  - "taudem"
  - "grass"
  - "whitebox"
  
metrics:
  - "iou"
  - "centroid_offset"
  - "boundary_ratio"
  - "runtime"
  
sampling:
  strategy: "stratified"
  strata:
    - "size"
    - "terrain"
    - "complexity"
  n_per_stratum: 5
  
analysis:
  statistical_tests:
    - "anova"
    - "tukey_hsd"
    - "kruskal_wallis"
  significance_level: 0.05
  
output:
  generate_plots: true
  include_raw_data: true
  summary_statistics: true
```

## Configuration Composition Examples

### Example 1: Development FLOWFINDER Test

**Command**: `python run_benchmark.py --env development --tool flowfinder --experiment accuracy_comparison`

**Effective Configuration**:
```yaml
# Composed from:
# - base/regions.yaml#mountain_west_minimal
# - base/quality_standards.yaml#development_grade
# - environments/development.yaml
# - tools/flowfinder/base.yaml
# - experiments/accuracy_comparison.yaml

region:
  name: "Mountain West - Development Subset"
  states: ["CO"]
  bounding_box:
    west: -105.7
    east: -105.0
    south: 39.9
    north: 40.3
  basin_count_estimate: 10

quality_standards:
  iou_thresholds:
    flat: 0.80
    moderate: 0.75
    steep: 0.70

tool:
  name: "FLOWFINDER"
  command:
    executable: "flowfinder"
    subcommand: "delineate"
    timeout_seconds: 30  # Overridden by environment

data_sources:
  max_basins: 10
  scale_factor: 0.01

experiment:
  tools: ["flowfinder"]  # Filtered to requested tool
  n_per_stratum: 1  # Reduced for development
```

### Example 2: Production Multi-Tool Comparison

**Command**: `python run_benchmark.py --env production --tools all --experiment accuracy_comparison`

**Effective Configuration**:
```yaml
# Composed from:
# - base/regions.yaml#mountain_west_full
# - base/quality_standards.yaml#research_grade
# - environments/production.yaml
# - tools/*/base.yaml (all tools)
# - experiments/accuracy_comparison.yaml

region:
  name: "Mountain West - Full Region"
  states: ["CO", "UT", "NM", "WY", "MT", "ID", "AZ"]
  basin_count_estimate: 2500

quality_standards:
  iou_thresholds:
    flat: 0.95
    moderate: 0.92
    steep: 0.85

tools:
  flowfinder:
    executable: "flowfinder"
    timeout_seconds: 300
  taudem:
    executable: "mpiexec"
    mpi_processes: 4
    timeout_seconds: 300
  grass:
    executable: "grass"
    timeout_seconds: 300
  whitebox:
    executable: "whitebox_tools"
    timeout_seconds: 300

data_sources:
  scale_factor: 1.0
  max_basins: null

experiment:
  n_per_stratum: 5
  tools: ["flowfinder", "taudem", "grass", "whitebox"]
```

## Configuration Loader Implementation

```python
# config_manager.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from jsonschema import validate, ValidationError

class ConfigurationManager:
    """Hierarchical configuration loader with validation."""
    
    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.schemas = self._load_schemas()
        
    def get_effective_config(self, 
                           environment: str,
                           tool: str = None,
                           experiment: str = None,
                           region: str = None) -> Dict[str, Any]:
        """Get final composed configuration."""
        
        config = {}
        
        # 1. Load base configurations
        config.update(self._load_base_configs(region))
        
        # 2. Apply environment settings
        env_config = self._load_file(f"environments/{environment}.yaml")
        config = self._merge_configs(config, env_config)
        
        # 3. Apply tool-specific settings
        if tool:
            tool_config = self._load_file(f"tools/{tool}/base.yaml")
            config = self._merge_configs(config, tool_config)
            
        # 4. Apply experiment settings
        if experiment:
            exp_config = self._load_file(f"experiments/{experiment}.yaml")
            config = self._merge_configs(config, exp_config)
            
        # 5. Validate final configuration
        self._validate_config(config)
        
        return config
    
    def _load_base_configs(self, region: str = None) -> Dict[str, Any]:
        """Load base configuration files."""
        config = {}
        
        # Load all base configs
        base_files = [
            "base/regions.yaml",
            "base/data_sources.yaml", 
            "base/quality_standards.yaml",
            "base/crs_definitions.yaml"
        ]
        
        for file_path in base_files:
            if self._file_exists(file_path):
                file_config = self._load_file(file_path)
                config = self._merge_configs(config, file_config)
                
        # Select specific region if requested
        if region and "regions" in config:
            if region in config["regions"]:
                config["region"] = config["regions"][region]
                
        return config
    
    def _load_file(self, relative_path: str) -> Dict[str, Any]:
        """Load a single configuration file."""
        file_path = self.config_root / relative_path
        
        if not file_path.exists():
            return {}
            
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schemas."""
        # Validate against appropriate schema
        if "tool" in config:
            validate(config, self.schemas["tool_config"])
        if "experiment" in config:
            validate(config, self.schemas["experiment_config"])
            
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for validation."""
        schemas = {}
        schema_dir = self.config_root / "schemas"
        
        for schema_file in schema_dir.glob("*.json"):
            schema_name = schema_file.stem
            with open(schema_file, 'r') as f:
                schemas[schema_name] = json.load(f)
                
        return schemas
```

## Tool Adapter Examples

```python
# tool_adapters.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from shapely.geometry import Point, Polygon
import subprocess
import tempfile
import json

class ToolAdapter(ABC):
    """Abstract base class for watershed delineation tools."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_metrics = {}
        
    @abstractmethod
    def delineate_watershed(self, pour_point: Point, dem_path: str) -> Tuple[Polygon, Dict[str, Any]]:
        """Delineate watershed from pour point."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if tool is available on system."""
        pass

class FlowfinderAdapter(ToolAdapter):
    """FLOWFINDER tool adapter."""
    
    def delineate_watershed(self, pour_point: Point, dem_path: str) -> Tuple[Polygon, Dict[str, Any]]:
        """Delineate watershed using FLOWFINDER."""
        import time
        start_time = time.time()
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            output_path = tmp.name
            
        # Build command
        cmd = [
            self.config['tool']['command']['executable'],
            self.config['tool']['command']['subcommand'],
            '--dem', dem_path,
            '--pour-point', f"{pour_point.x},{pour_point.y}",
            '--output', output_path,
            '--format', self.config['tool']['command']['output_format']
        ]
        
        # Add algorithm parameters
        if 'algorithms' in self.config['tool']:
            alg = self.config['tool']['algorithms']
            cmd.extend(['--flow-direction', alg.get('flow_direction', 'd8')])
            cmd.extend(['--depression-filling', alg.get('depression_filling', 'priority_flood')])
            
        # Execute command
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=self.config['tool']['performance']['timeout_seconds'])
            
            if result.returncode != 0:
                raise RuntimeError(f"FLOWFINDER failed: {result.stderr}")
                
            # Load result
            with open(output_path, 'r') as f:
                geojson_data = json.load(f)
                
            # Convert to Polygon (simplified)
            coords = geojson_data['features'][0]['geometry']['coordinates'][0]
            polygon = Polygon(coords)
            
            # Record performance metrics
            runtime = time.time() - start_time
            metrics = {
                'runtime_seconds': runtime,
                'memory_usage_mb': 0,  # Would need psutil for real measurement
                'tool': 'flowfinder',
                'algorithm': alg.get('flow_direction', 'd8')
            }
            
            return polygon, metrics
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FLOWFINDER timed out after {self.config['tool']['performance']['timeout_seconds']} seconds")
    
    def is_available(self) -> bool:
        """Check if FLOWFINDER is available."""
        try:
            result = subprocess.run([self.config['tool']['command']['executable'], '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

class TaudemAdapter(ToolAdapter):
    """TauDEM tool adapter."""
    
    def delineate_watershed(self, pour_point: Point, dem_path: str) -> Tuple[Polygon, Dict[str, Any]]:
        """Delineate watershed using TauDEM."""
        import time
        start_time = time.time()
        
        # TauDEM requires multiple steps
        # 1. Fill pits
        # 2. Calculate flow directions
        # 3. Calculate flow accumulation
        # 4. Define streams
        # 5. Segment watershed
        
        # This would be a complex implementation
        # For now, return placeholder
        
        runtime = time.time() - start_time
        metrics = {
            'runtime_seconds': runtime,
            'memory_usage_mb': 0,
            'tool': 'taudem',
            'mpi_processes': self.config['tool']['performance']['mpi_processes']
        }
        
        # Placeholder polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        return polygon, metrics
    
    def is_available(self) -> bool:
        """Check if TauDEM is available."""
        try:
            result = subprocess.run(['mpiexec', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
```

## Usage Examples

```python
# Example usage of the configuration system
from config_manager import ConfigurationManager
from tool_adapters import FlowfinderAdapter, TaudemAdapter

# Initialize configuration manager
config_mgr = ConfigurationManager("config/")

# Load development configuration for FLOWFINDER
config = config_mgr.get_effective_config(
    environment="development",
    tool="flowfinder", 
    experiment="accuracy_comparison",
    region="mountain_west_minimal"
)

# Create tool adapter
adapter = FlowfinderAdapter(config)

# Check if tool is available
if adapter.is_available():
    # Delineate watershed
    pour_point = Point(-105.5, 40.0)
    polygon, metrics = adapter.delineate_watershed(pour_point, "dem.tif")
    
    print(f"Watershed area: {polygon.area} square degrees")
    print(f"Runtime: {metrics['runtime_seconds']} seconds")
else:
    print("FLOWFINDER not available")
```

This architecture provides a clean, maintainable, and extensible configuration system that can grow with the project's multi-tool ambitions while keeping the complexity manageable. 