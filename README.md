# FLOWFINDER: Python Watershed Delineation Tool

A Python implementation of watershed delineation algorithms with benchmarking infrastructure. FLOWFINDER provides reliable watershed boundary extraction from Digital Elevation Models (DEMs) using standard hydrological algorithms, along with performance monitoring and validation tools.

**Current Status:** Core watershed delineation functionality is complete and tested. Multi-tool benchmarking and research framework components are in development.

![flow finder](images/flowfinder.png)

## 🎯 Project Goals

**What does FLOWFINDER currently provide?**

1. **Reliable Watershed Delineation**: Fast, accurate watershed boundary extraction from DEM data
2. **Performance Monitoring**: Built-in timing, memory usage, and accuracy tracking
3. **Validation Tools**: Topology checking and quality assessment for generated watersheds
4. **Python & CLI Access**: Both programmatic API and command-line interface

**Future Development Goals:**
- Multi-tool comparison framework (TauDEM, GRASS, WhiteboxTools integration)
- Systematic benchmarking across different terrain types  
- Research-grade validation studies
- Geographic specialization optimization

## 🔬 Technical Background

### Core Implementation
FLOWFINDER implements standard hydrological algorithms using modern Python scientific libraries:

- **Flow Direction**: D8 algorithm with priority-flood depression filling
- **Flow Accumulation**: Topological sorting (Kahn's algorithm) for O(n) performance  
- **Watershed Extraction**: Upstream tracing from pour points
- **Polygon Creation**: Morphological operations with boundary tracing

### Validation & Quality Assurance
- Real-time performance monitoring (runtime, memory usage)
- Topology validation (geometry validity, containment checks)
- Accuracy assessment tools (when ground truth available)
- Comprehensive error handling and logging

## 📋 Prerequisites

### Required
- Python 3.8+
- DEM data in GeoTIFF format
- 4GB+ RAM recommended for processing large datasets

### Optional (for development/benchmarking)
- Additional watershed tools for comparison:
  - TauDEM (requires Docker)
  - GRASS GIS
  - WhiteboxTools
- Ground truth watershed boundaries for validation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd flowfinder

# Install dependencies
pip install -e .[dev]

# Install FLOWFINDER
pip install flowfinder

# Copy environment template (if it exists)
cp .env.example .env || echo "Create .env file with your data paths"
# Edit .env with your data paths and configuration
```

### 2. Configuration Setup

The system uses a hierarchical configuration architecture to manage complexity:

```bash
# Configuration structure is already set up:

# Environment-specific configurations
config/environments/development.yaml    # Local dev (10 basins)
config/environments/testing.yaml        # CI/testing (50 basins)
config/environments/production.yaml     # Full-scale (500+ basins)

# Tool-specific configurations
config/tools/flowfinder.yaml            # FLOWFINDER settings
config/tools/taudem.yaml               # TauDEM MPI settings
config/tools/grass.yaml                # GRASS r.watershed settings
config/tools/whitebox.yaml             # WhiteboxTools settings
```

### 3. Data Preparation

Place your input datasets in the `data/` directory:

```
data/
├── huc12_mountain_west.shp    # HUC12 boundaries for Mountain West
├── nhd_hr_catchments.shp      # NHD+ HR catchment polygons
├── nhd_flowlines.shp          # NHD+ HR flowlines
└── dem_10m.tif               # 10m DEM mosaic or tiles
```

### 4. Basic Usage

#### Python API
```python
from flowfinder import FlowFinder

# Initialize with DEM
with FlowFinder("path/to/dem.tif") as ff:
    # Delineate watershed from a pour point
    watershed, metrics = ff.delineate_watershed(lat=40.0, lon=-105.0)
    print(f"Watershed area: {watershed.area:.6f} degrees²")
```

#### Command Line Interface
```bash
# Delineate watershed
python -m flowfinder.cli delineate --dem dem.tif --lat 40.0 --lon -105.0 --output watershed.geojson

# Validate DEM
python -m flowfinder.cli validate --dem dem.tif

# Get DEM info
python -m flowfinder.cli info --dem dem.tif
```

### 5. Benchmarking (Experimental)

⚠️ **Note**: Multi-tool comparison features are currently in development. Some components may use mock data for testing infrastructure.

```bash
# Run basic benchmark with FLOWFINDER
python scripts/benchmark_runner.py \
    --environment development \
    --tools flowfinder \
    --outdir results/
```

## 📁 Project Structure

```
├── README.md                    # Project overview + setup
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Modern Python project config
├── .env.example                # Environment template
├── .gitignore                  # Standard Python gitignore
│
├── config/                     # Hierarchical configuration system
│   ├── base.yaml              # Foundation configurations
│   ├── configuration_manager.py # Configuration inheritance system
│   ├── schema.json            # JSON Schema validation
│   ├── environments/           # Environment-specific settings
│   │   ├── development.yaml   # Local development (10 basins)
│   │   ├── testing.yaml       # CI/testing (50 basins)
│   │   └── production.yaml    # Full-scale (500+ basins)
│   └── tools/                  # Tool-specific configurations
│       ├── flowfinder.yaml    # FLOWFINDER settings
│       ├── taudem.yaml        # TauDEM MPI settings
│       ├── grass.yaml         # GRASS r.watershed settings
│       └── whitebox.yaml      # WhiteboxTools settings
│
├── scripts/                    # Core benchmark scripts
│   ├── basin_sampler.py       # Stratified basin sampling
│   ├── truth_extractor.py     # Truth polygon extraction
│   ├── benchmark_runner.py    # FLOWFINDER accuracy testing
│   ├── watershed_experiment_runner.py # Multi-tool comparison
│   ├── validation_tools.py    # Validation utilities
│   └── backup/                # Deprecated scripts and files
│
├── data/                       # Input datasets (gitignored)
│   └── test_outputs/          # Test result files
├── results/                    # Output directory (gitignored)
├── tests/                      # Unit and integration tests
│   └── integration/           # Integration test suite
├── docs/                       # Research and technical documentation
│   ├── README.md              # Documentation index
│   ├── PIPELINE.md            # Pipeline orchestrator guide
│   ├── user_guide/            # User documentation and guides
│   ├── architecture/          # System architecture documentation
│   ├── test_coverage/         # Test coverage documentation
│   └── development/           # Development notes and status reports
│
└── notebooks/                  # Jupyter exploration
    └── benchmark_analysis.ipynb
```

## 🔧 Configuration Architecture

The system uses a **hierarchical configuration architecture** to manage complexity across different tools and environments:

### Configuration Hierarchy
```
Base Configurations → Environment → Tool → Local Overrides
```

### Example Configuration Composition
```yaml
# Development FLOWFINDER experiment
inherits:
  - "base/regions.yaml#mountain_west_minimal"
  - "base/quality_standards.yaml#development_grade"
  - "environments/development.yaml"
  - "tools/flowfinder/base.yaml"
  - "experiments/accuracy_comparison.yaml"

overrides:
  basin_sampling:
    n_per_stratum: 1  # Minimal for dev
  benchmark:
    timeout_seconds: 30  # Quick timeout
```

### Tool Adapter Interface
```python
class ToolAdapter(ABC):
    @abstractmethod
    def delineate_watershed(self, pour_point: Point, dem_path: str) -> Tuple[Polygon, Dict]:
        """Delineate watershed and return polygon + performance metrics"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if tool is available on system"""
        pass
```

## 📊 Outputs

### FLOWFINDER Results
- **Watershed Polygons**: GeoJSON or Shapefile format
- **Performance Metrics**: Runtime, memory usage, processing rate
- **Quality Assessment**: Topology validation, accuracy scores
- **Detailed Logs**: Complete processing history and diagnostics

### Benchmarking Outputs (When Available)
- `benchmark_results.json`: Detailed per-watershed metrics
- `accuracy_summary.csv`: Tabular results for analysis
- `performance_comparison.csv`: Runtime and memory comparisons

⚠️ **Note**: Multi-tool comparison outputs are generated using mock data when external tools are not available.

## 🎯 Current Performance

### Core FLOWFINDER Capabilities
| Feature                   | Status                           | Notes |
| ------------------------- | --------------------------------- | ------ |
| Basic watershed delineation | ✅ Implemented                 | Tested and working |
| D8 flow direction         | ✅ Implemented                   | With depression filling |
| Flow accumulation         | ✅ Implemented                   | O(n) topological sorting |
| Performance monitoring    | ✅ Implemented                   | Runtime, memory tracking |
| Python API               | ✅ Implemented                   | Full functionality |
| CLI interface            | ✅ Implemented                   | Basic commands available |
| Topology validation      | ✅ Implemented                   | Geometry checking |

### Development Roadmap
| Feature                   | Timeline                         | Status |
| ------------------------- | --------------------------------- | ------ |
| Multi-tool integration    | Future release                   | 📋 Planned |
| Batch processing         | Next minor version               | 🔄 In Development |
| Advanced algorithms      | Future release                   | 📋 Planned |
| Performance optimization | Ongoing                          | 🔄 Continuous |
| Documentation improvements | Next patch                     | 🔄 In Progress |

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Test configuration system
python tests/integration/test_configuration_system.py

# Test multi-tool integration
python tests/integration/test_integration.py

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html
```

## 📈 Analysis

Use the Jupyter notebook for detailed analysis:

```bash
# Start Jupyter
jupyter lab notebooks/

# Open benchmark_analysis.ipynb for interactive exploration
```

## 🎯 Development Roadmap

### Current Release (v0.1.0) - COMPLETED
- ✅ **Core Algorithm Implementation**: D8 flow direction, flow accumulation, watershed extraction
- ✅ **Python API**: Complete programmatic interface
- ✅ **Basic CLI**: Command-line watershed delineation
- ✅ **Performance Monitoring**: Runtime and memory tracking
- ✅ **Validation Tools**: Topology checking and quality assessment

### Next Minor Release (v0.2.0) - PLANNED
- 📋 **Batch Processing**: Process multiple pour points efficiently  
- 📋 **Output Formats**: Additional export options (KML, WKT)
- 📋 **Configuration Improvements**: Better parameter management
- 📋 **Documentation**: Comprehensive user guide and API reference

### Future Development - UNDER CONSIDERATION
- 📋 **Multi-tool Integration**: TauDEM, GRASS, WhiteboxTools comparison (requires external tool installation)
- 📋 **Advanced Algorithms**: D-infinity, multiple flow direction methods
- 📋 **Performance Optimization**: Parallel processing, memory optimization
- 📋 **Research Framework**: Systematic validation studies (academic collaboration needed)

## 📚 Documentation

📖 **[Complete Documentation](docs/README.md)** - Full documentation index

### Quick Links
- **[Setup Guide](docs/user_guide/setup.md)** - Installation and environment setup
- **[Pipeline Guide](docs/PIPELINE.md)** - Running benchmarks and workflows
- **[Data Specification](docs/user_guide/data_specification.md)** - Data sources and requirements
- **[Configuration Examples](docs/user_guide/configuration_examples.md)** - Configuration system usage

### Technical Documentation
- **[Architecture Overview](docs/architecture/)** - System design and architecture
- **[Configuration Architecture](docs/architecture/configuration_architecture.md)** - Hierarchical configuration system
- **[Multi-Tool Framework](docs/architecture/multi_tool_benchmark_architecture.md)** - Multi-tool comparison framework
- **[Test Coverage](docs/test_coverage/)** - Comprehensive testing documentation

## 🤝 Contributing

We welcome contributions from the research community:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/research-improvement`)
3. Commit your changes (`git commit -m 'Add research improvement'`)
4. Push to the branch (`git push origin feature/research-improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- USGS for NHD+ HR and 3DEP data
- FLOWFINDER development team
- Open source geospatial community
- Academic research community for feedback and validation

## 📞 Support

For research questions and technical issues:
- Check the [documentation](docs/)
- Review the [Research Roadmap](docs/strategic_analysis_implementation_roadmap_v2.md)
- See the [Multi-Tool Integration Strategy](docs/multi_tool_integration_strategy.md)
- Open an issue on GitHub

---

**FLOWFINDER: Reliable Python watershed delineation with performance monitoring and validation tools.**
