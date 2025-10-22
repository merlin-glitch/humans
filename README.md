# Human Society Simulation

An agent-based simulation modeling human societies with resource competition, trust dynamics, and evolutionary mechanics. This project explores how social structures, cooperation, and resource distribution patterns emerge from individual agent behaviors.

## Overview

The simulation features two competing families (Blue and Red houses) that must forage for resources, build trust relationships, and reproduce to survive. Agents develop social networks through food sharing, territorial behavior, and leadership dynamics, creating emergent patterns of cooperation and competition.

### Key Features

- **Trust System**: Dynamic relationship tracking with numpy-based O(1) lookups
- **Resource Management**: Zone-based food spawning with adaptive respawn dynamics
- **Social Mechanics**: Food sharing, leadership competition, and family loyalty
- **Reproductive Dynamics**: Mating based on trust thresholds and energy costs
- **Territorial Behavior**: Zone-based resource exploitation patterns
- **Comprehensive Analytics**: 20+ visualization types for population, trust, and resource flow

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd humans
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Full installation with all dependencies
   pip install -r requirements.txt
   
   # Or minimal installation (core dependencies only)
   pip install -r requirements-minimal.txt
   
   # Or manual installation
   pip install numpy pandas matplotlib pygame pygame-menu opencv-python tqdm pillow
   ```

## Quick Start

### Interactive UI Simulation

Run the visual simulation with real-time controls:

```bash
python ui_simulation.py
```

Features:
- Real-time population visualization
- Adjustable simulation speed
- Pause/resume controls (Press 'P')
- Parameter sliders for human count and speed
- Export trust matrix functionality

### Headless Batch Simulations

Run multiple simulations for statistical analysis:

```bash
python batch_simulation.py
```

This generates CSV data suitable for analysis and plotting.

### Generate Analysis Plots

Create comprehensive visualizations from batch results:

```bash
python batch_plot.py --csv batch_results/all_combined.csv --out plots/
```

## Project Structure

```
humans/
├── README.md                 # This file
├── config.py                 # Configuration constants and parameters
├── human.py                  # Human and House agent classes
├── trust_system.py          # Trust system implementation
├── resource_manager.py      # Resource management and zone detection
├── social_mechanics.py      # Social interaction and trust mechanics
├── simulation_utils.py      # World building and shared utilities
├── ui_simulation.py         # Interactive UI simulation with pygame
├── headless_simulation.py   # Optimized headless simulation
├── batch_simulation.py      # Batch simulation runner
├── batch_plot.py            # Comprehensive plotting suite
├── visualization.py         # Spatial heatmap visualization
├── ui_components.py         # UI components and controls
├── plot_utils.py            # Simple plotting utilities
├── validation_utils.py      # Testing and validation
├── images/                  # Map images and assets
├── batch_results/           # Simulation output data
├── previews/                # Generated visualization frames
├── docs/                    # Comprehensive documentation
└── venv/                    # Virtual environment
```

## Configuration

Key parameters in `config.py`:

### Map Settings
- `MAP_WIDTH`, `MAP_HEIGHT`: Simulation world dimensions
- `CELL_SIZE`: Pixel size for visualization
- `NEW_PALETTE`: Color mapping for terrain zones

### Population
- `Nbre_HUMANS`: Initial population size
- `ENERGY_COST`: Energy required for mating
- `MATING_COOLDOWN`: Minimum time between mating attempts

### Resources
- `FOOD_LIFETIME`: How long food remains available
- `FOOD_STACK`: Maximum food units per cell
- `INITIAL_FOOD_COUNT`: Starting food amount
- `FOOD_SPAWN_COUNT`: Food units spawned per event

### Time
- `DAY_LENGTH`: Simulation ticks per day
- Day/night cycle: 70% day, 30% night

## Simulation Mechanics

### Agent Behavior

**Humans** are autonomous agents that:
- Move around the map seeking resources
- Build trust relationships through food sharing
- Follow trusted leaders during resource gathering
- Compete for leadership within families
- Mate when trust thresholds and energy requirements are met
- Die from energy depletion

**Houses** serve as:
- Safe zones for family members
- Resource storage locations
- Centers for social interaction

### Trust System

The trust system tracks relationships between all agents:
- Trust scores range from 0.0 to 1.0
- Updated through successful food sharing
- Influences cooperation and leadership selection
- Cached for efficient lookups during simulation

### Resource Dynamics

- Food spawns in designated zones based on consumption patterns
- Adaptive respawn rates prevent over-exploitation
- Zone-based territorial behavior emerges naturally
- Resource decay simulates natural food spoilage

### Social Structure

- **Blue vs Red Houses**: Two competing family groups
- **Leadership Competition**: Red house members compete for leadership at dawn
- **Trust-Based Cooperation**: Agents share resources based on trust levels
- **Territorial Behavior**: Families develop preferences for specific zones

## Visualization

The simulation provides extensive visualization capabilities:

### Population Dynamics
- Population counts over time by family
- Birth and death rates
- Survival curves

### Trust Evolution
- Within-family vs between-family trust levels
- Trust distribution across the population
- Trust matrix heatmaps

### Resource Flow
- Zone consumption patterns by family
- Spawn vs consumption ratios
- Per-capita resource utilization

### Spatial Analysis
- Heatmaps of resource exploitation
- Zone dominance patterns
- Territorial behavior visualization

## Research Applications

This simulation is suitable for studying:

### Game Theory
- Tragedy of the commons scenarios
- Cooperation vs competition dynamics
- Resource sharing strategies

### Social Science
- Emergence of social hierarchies
- Trust formation and maintenance
- Group loyalty and territorial behavior

### Evolutionary Biology
- Altruism and kin selection
- Population dynamics under resource constraints
- Behavioral adaptation and learning

### Economics
- Resource allocation efficiency
- Market-like behavior in resource distribution
- Inequality emergence in resource access

## Advanced Usage

### Custom Maps

Create custom simulation environments by providing PNG images with specific color mappings defined in `NEW_PALETTE`.

### Parameter Studies

Modify `config.py` to explore different scenarios:
- Vary population sizes and resource abundance
- Adjust trust thresholds and energy costs
- Change map layouts and zone distributions

### Batch Analysis

Run multiple simulations with different seeds to study:
- Statistical significance of emergent behaviors
- Robustness of social patterns
- Sensitivity to initial conditions

### Export and Integration

The simulation exports data in CSV format compatible with:
- Statistical analysis tools (R, Python pandas)
- Machine learning frameworks
- Custom analysis pipelines

## Troubleshooting

### Common Issues

1. **Pygame display errors**: Ensure you have a display available or set `SDL_VIDEODRIVER=dummy` for headless mode
2. **Import errors**: Verify all dependencies are installed in the virtual environment
3. **Memory issues**: Reduce `Nbre_HUMANS` or `MAP_WIDTH`/`MAP_HEIGHT` for large simulations
4. **Slow performance**: Use `headless_simulation.py` for batch runs instead of the UI version

### Performance Tips

- Use `headless_simulation.py` for batch simulations
- Use `ui_simulation.py` for interactive exploration
- Reduce visualization frequency for large populations
- Enable numpy optimizations in config
- Consider smaller map sizes for faster iteration

## Contributing

This project welcomes contributions in:
- Additional visualization types
- New agent behaviors and social mechanics
- Performance optimizations
- Documentation improvements
- Research applications and case studies

## License

[Add your license information here]

## Citation

If you use this simulation in research, please cite:

```
[Add citation format here]
```

## Contact

[Add contact information here]

---

*This simulation represents a complex multi-agent system modeling human social behavior. The emergent patterns observed reflect both the programmed mechanics and the stochastic nature of agent interactions, providing insights into how individual behaviors can lead to collective social structures.*