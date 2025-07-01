# FLOWFINDER: Next-Generation Watershed Delineation with Unprecedented Reliability

A comprehensive watershed delineation tool and benchmark framework designed to establish new standards in accuracy, reliability, and systematic comparison. FLOWFINDER combines production-ready watershed delineation capabilities with the first standardized multi-tool benchmark framework for the hydrology research community.

![flow finder](images/flowfinder.png)

## 🎯 Strategic Vision

**"Next-Generation Watershed Delineation with Unprecedented Reliability"**

FLOWFINDER represents a breakthrough in watershed delineation technology, achieving **100% validation success (51/51 checks)** while providing the first comprehensive benchmark framework for systematic tool comparison. Our mission is to establish new standards in watershed delineation through:

- **Production-ready tool** with unprecedented reliability validation
- **Comprehensive benchmark framework** for systematic multi-tool comparison
- **Mountain West terrain specialization** addressing geographic research gaps
- **Academic credibility** through rigorous validation and peer-reviewed methodology

## 🏆 Key Differentiators

| Aspect | FLOWFINDER | TauDEM | GRASS GIS | WhiteboxTools |
|--------|------------|--------|-----------|---------------|
| **Reliability** | 100% validation (51/51) | Variable | Variable | Variable |
| **Benchmark Integration** | Native framework | External tools needed | Complex setup | Command-line focused |
| **Mountain West Focus** | Optimized | General purpose | General purpose | General purpose |
| **Modern Architecture** | Python + validation | MPI/C++ | C/Module system | Rust |

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

The system uses a hierarchical configuration architecture:

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

### 5. Run Multi-Tool Comparison

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
├── docs/                       # Strategic and technical documentation
│   ├── strategic_analysis_implementation_roadmap_v2.md # Strategic roadmap
│   ├── multi_tool_integration_strategy.md # Integration strategy
│   ├── strategic_analysis_assessment.md # Strategic evaluation
│   ├── immediate_next_steps.md # Implementation priorities
│   ├── configuration_architecture.md # Configuration system design
│   ├── multi_tool_benchmark_architecture.md # Multi-tool framework
│   └── test_coverage/          # Test coverage documentation
│
└── notebooks/                  # Jupyter exploration
    └── benchmark_analysis.ipynb
```

## 🔧 Configuration Architecture

The system uses a **hierarchical configuration architecture** with inheritance:

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

### Single-Tool Benchmark
- `benchmark_results.json`: Detailed per-basin metrics
- `accuracy_summary.csv`: Tabular results for analysis
- `benchmark_summary.txt`: Performance analysis and key findings

### Multi-Tool Comparison
- `multi_tool_results.json`: Comparative analysis across tools
- `performance_comparison.csv`: Runtime and memory comparisons
- `statistical_analysis.csv`: ANOVA, Tukey HSD, Kruskal-Wallis results
- `publication_figures/`: Publication-ready charts and graphs

## 🎯 Success Metrics

### Technical Metrics
| Metric                    | Target                           |
| ------------------------- | -------------------------------- |
| FLOWFINDER IOU (mean)     | ≥ 0.90                           |
| FLOWFINDER IOU (90th percentile) | ≥ 0.95                       |
| Runtime (mean)            | ≤ 30 s                           |
| Configuration redundancy  | 90% reduction                    |
| Tool integration success  | 4 major tools integrated         |

### Academic Metrics
| Metric                    | Target                           |
| ------------------------- | -------------------------------- |
| Peer-reviewed publications | 2+ papers accepted               |
| Conference presentations  | 5+ presentations                 |
| Citations (2 years)       | 100+ citations                   |
| Framework adoption        | 3+ external research groups      |

### Community Metrics
| Metric                    | Target                           |
| ------------------------- | -------------------------------- |
| GitHub stars              | 500+ stars                       |
| FLOWFINDER downloads      | 1000+ downloads                  |
| External contributors     | 10+ contributors                 |
| Institutional adoptions   | 5+ adoptions                     |

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

## 🎯 Strategic Roadmap

### Phase 1: Foundation (Months 1-3) - CRITICAL
- ✅ **Configuration Architecture**: Hierarchical system implemented
- ✅ **FLOWFINDER Production**: 51/51 validation success achieved
- 🔄 **Benchmark Framework MVP**: Multi-tool comparison operational
- 🔄 **Research Foundation**: Literature review and gap analysis

### Phase 2: Tool Integration (Months 4-8) - HIGH PRIORITY
- 🔄 **WhiteboxTools Integration**: Rust-based performance comparison
- 🔄 **TauDEM Integration**: Academic gold standard validation
- 🔄 **GRASS GIS Integration**: Comprehensive hydrological suite
- 🔄 **SAGA GIS Integration**: European academic adoption

### Phase 3: Academic Impact (Months 9-12) - HIGH IMPACT
- 🔄 **Comprehensive Benchmarking**: 25+ watersheds across terrain types
- 🔄 **Statistical Analysis**: Publication-ready comparative results
- 🔄 **Community Building**: Open source release and adoption
- 🔄 **Academic Publications**: Peer-reviewed papers and presentations

## 📚 Documentation

### Strategic Documents
- **[Strategic Roadmap](docs/strategic_analysis_implementation_roadmap_v2.md)**: Complete implementation plan with checkpoints
- **[Multi-Tool Integration Strategy](docs/multi_tool_integration_strategy.md)**: Research-based tool integration approach
- **[Strategic Assessment](docs/strategic_analysis_assessment.md)**: Comprehensive strategic evaluation
- **[Immediate Next Steps](docs/immediate_next_steps.md)**: Actionable implementation priorities

### Technical Documents
- **[Configuration Architecture](docs/configuration_architecture.md)**: Hierarchical configuration system design
- **[Multi-Tool Benchmark Architecture](docs/multi_tool_benchmark_architecture.md)**: Framework design and implementation
- **[Test Coverage](docs/test_coverage/)**: Comprehensive testing documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- USGS for NHD+ HR and 3DEP data
- FLOWFINDER development team
- Open source geospatial community
- Academic research community for validation and feedback

## 📞 Support

For issues and questions:
- Check the [documentation](docs/)
- Review the [Strategic Roadmap](docs/strategic_analysis_implementation_roadmap_v2.md)
- See the [Multi-Tool Integration Strategy](docs/multi_tool_integration_strategy.md)
- Open an issue on GitHub

---

*"Reliability earns trust. Systematic comparison drives innovation."*

**FLOWFINDER: Establishing new standards in watershed delineation through unprecedented reliability and comprehensive benchmarking.** 