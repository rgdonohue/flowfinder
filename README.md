# FLOWFINDER: Watershed Delineation Research & Benchmark Framework

A research project exploring watershed delineation accuracy and developing systematic comparison methods for hydrological analysis tools. This work addresses key challenges in watershed delineation: reliability validation, systematic benchmarking, and geographic specialization for complex terrain.

![flow finder](images/flowfinder.png)

## 🎯 Research Questions

**What problems are we trying to solve?**

1. **Reliability Gap**: How can we systematically validate watershed delineation tools across diverse terrain types?
2. **Benchmarking Gap**: Why is there no standardized framework for comparing watershed delineation tools?
3. **Geographic Bias**: How do existing tools perform in Mountain West terrain compared to other regions?
4. **Reproducibility Crisis**: How can we ensure watershed delineation results are reproducible and comparable?

**Our approach**: Develop FLOWFINDER as both a research tool and benchmark framework to systematically investigate these questions.

## 🔬 Research Context

### Current State of Watershed Delineation
- **Tool proliferation**: Multiple tools (TauDEM, GRASS, WhiteboxTools) with different algorithms
- **Validation challenges**: Limited systematic comparison of accuracy and performance
- **Geographic bias**: Most studies focus on eastern US or international basins
- **Reproducibility issues**: Ad-hoc validation methods make results hard to compare

### Research Gaps We're Addressing
- **Systematic benchmarking**: No standardized framework for multi-tool comparison
- **Mountain West terrain**: Limited research on complex terrain performance
- **Reliability metrics**: Need for consistent validation across tools
- **Open methodology**: Reproducible research practices for watershed analysis

## 📋 Prerequisites

- Python 3.8+
- FLOWFINDER CLI tool installed and accessible
- Access to USGS NHD+ HR data and 3DEP 10m DEM data
- 8GB+ RAM recommended for processing large datasets
- Docker (for TauDEM integration)
- GRASS GIS (for r.watershed comparison)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd flowfinder

# Install dependencies
pip install -r requirements.txt

# Install FLOWFINDER
pip install flowfinder

# Copy environment template
cp .env.example .env
# Edit .env with your data paths and configuration
```

### 2. Configuration Setup

The system uses a hierarchical configuration architecture to manage complexity:

```bash
# Create configuration structure
mkdir -p config/{base,environments,tools,experiments,schemas}

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

### 4. Run Single-Tool Benchmark

```bash
# Step 1: Generate stratified basin sample
python scripts/basin_sampler.py --config config/basin_sampler_config.yaml

# Step 2: Extract truth polygons
python scripts/truth_extractor.py --config config/truth_extractor_config.yaml

# Step 3: Run FLOWFINDER benchmark
python scripts/benchmark_runner.py \
    --sample basin_sample.csv \
    --truth truth_polygons.gpkg \
    --config config/benchmark_config.yaml \
    --outdir results/
```

### 5. Run Multi-Tool Comparison (Experimental)

```bash
# Using the new configuration system
python scripts/benchmark_runner_integrated.py \
    --environment development \
    --tools flowfinder,taudem \
    --experiment accuracy_comparison \
    --outdir results/multi_tool/
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
│   ├── environments/           # Environment-specific settings
│   │   ├── development.yaml   # Local development (10 basins)
│   │   ├── testing.yaml       # CI/testing (50 basins)
│   │   └── production.yaml    # Full-scale (500+ basins)
│   ├── tools/                  # Tool-specific configurations
│   │   ├── flowfinder.yaml    # FLOWFINDER settings
│   │   ├── taudem.yaml        # TauDEM MPI settings
│   │   ├── grass.yaml         # GRASS r.watershed settings
│   │   └── whitebox.yaml      # WhiteboxTools settings
│   └── schemas/               # JSON Schema validation
│
├── scripts/                    # Core benchmark scripts
│   ├── basin_sampler.py       # Stratified basin sampling
│   ├── truth_extractor.py     # Truth polygon extraction
│   ├── benchmark_runner.py    # FLOWFINDER accuracy testing
│   ├── benchmark_runner_integrated.py # Multi-tool comparison
│   └── tool_adapters/         # Tool adapter implementations
│
├── data/                       # Input datasets (gitignored)
├── results/                    # Output directory (gitignored)
├── tests/                      # Unit tests
├── docs/                       # Research and technical documentation
│   ├── strategic_analysis_implementation_roadmap_v2.md # Research roadmap
│   ├── multi_tool_integration_strategy.md # Integration approach
│   ├── strategic_analysis_assessment.md # Research evaluation
│   ├── immediate_next_steps.md # Implementation priorities
│   ├── configuration_architecture.md # Configuration system design
│   ├── multi_tool_benchmark_architecture.md # Framework design
│   └── test_coverage/          # Test coverage documentation
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

## 📊 Research Outputs

### Single-Tool Benchmark
- `benchmark_results.json`: Detailed per-basin metrics
- `accuracy_summary.csv`: Tabular results for analysis
- `benchmark_summary.txt`: Performance analysis and key findings

### Multi-Tool Comparison (Experimental)
- `multi_tool_results.json`: Comparative analysis across tools
- `performance_comparison.csv`: Runtime and memory comparisons
- `statistical_analysis.csv`: ANOVA, Tukey HSD, Kruskal-Wallis results
- `publication_figures/`: Research-ready charts and graphs

## 🎯 Research Metrics

### Technical Validation
| Metric                    | Current Target                    | Status |
| ------------------------- | --------------------------------- | ------ |
| FLOWFINDER IOU (mean)     | ≥ 0.90                           | 🔄 In Progress |
| FLOWFINDER IOU (90th percentile) | ≥ 0.95                       | 🔄 In Progress |
| Runtime (mean)            | ≤ 30 s                           | 🔄 In Progress |
| Configuration redundancy  | 90% reduction                    | ✅ Achieved |
| Tool integration success  | 4 major tools integrated         | 🔄 In Progress |

### Research Impact Goals
| Metric                    | Target                           | Status |
| ------------------------- | --------------------------------- | ------ |
| Peer-reviewed publications | 2+ papers submitted              | 🔄 In Progress |
| Conference presentations  | 5+ presentations                 | 🔄 In Progress |
| Citations (2 years)       | 100+ citations                   | 🔄 In Progress |
| Framework adoption        | 3+ external research groups      | 🔄 In Progress |

### Community Engagement Goals
| Metric                    | Target                           | Status |
| ------------------------- | --------------------------------- | ------ |
| GitHub stars              | 500+ stars                       | 🔄 In Progress |
| FLOWFINDER downloads      | 1000+ downloads                  | 🔄 In Progress |
| External contributors     | 10+ contributors                 | 🔄 In Progress |
| Institutional adoptions   | 5+ adoptions                     | 🔄 In Progress |

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test configuration system
python test_configuration_system.py

# Test multi-tool integration
python test_integration.py

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

## 🎯 Research Roadmap

### Phase 1: Foundation - IN PROGRESS
- ✅ **Configuration Architecture**: Hierarchical system implemented
- ✅ **FLOWFINDER Development**: Core tool with validation framework
- 🔄 **Benchmark Framework MVP**: Multi-tool comparison development
- 🔄 **Literature Review**: Research gap analysis and methodology development

### Phase 2: Tool Integration - PLANNED
- 🔄 **WhiteboxTools Integration**: Rust-based performance comparison
- 🔄 **TauDEM Integration**: Academic gold standard validation
- 🔄 **GRASS GIS Integration**: Comprehensive hydrological suite
- 🔄 **SAGA GIS Integration**: European academic adoption

## 📚 Documentation

### Research Documents
- **[Research Roadmap](docs/strategic_analysis_implementation_roadmap_v2.md)**: Implementation plan with research milestones
- **[Multi-Tool Integration Strategy](docs/multi_tool_integration_strategy.md)**: Research-based tool integration approach
- **[Research Assessment](docs/strategic_analysis_assessment.md)**: Comprehensive research evaluation
- **[Next Steps](docs/immediate_next_steps.md)**: Implementation priorities

### Technical Documents
- **[Configuration Architecture](docs/configuration_architecture.md)**: Hierarchical configuration system design
- **[Multi-Tool Benchmark Architecture](docs/multi_tool_benchmark_architecture.md)**: Framework design and implementation
- **[Test Coverage](docs/test_coverage/)**: Comprehensive testing documentation

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

*"Research is formalized curiosity. It is poking and prying with a purpose."*

**FLOWFINDER: Exploring watershed delineation accuracy and developing systematic comparison methods for hydrological research.** 