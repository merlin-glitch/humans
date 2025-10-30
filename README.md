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

**Step-by-step launch procedure:**

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Configure simulation** (optional):
   - Edit `config.py` to adjust parameters (population, energy costs, food dynamics, etc.)
   - Key settings: `Nbre_HUMANS`, `ENERGY_COST`, `MATING_COOLDOWN`, `FOOD_LIFETIME`

3. **Prepare map image**:
   - Place your map PNG in the `images/` directory
   - Update `MAP_IMAGE_PATH` in `ui_simulation.py` (default: `images/desert_oasis_well.png`)
   - On first run, define color categories interactively:
     - Wall colors (impassable terrain)
     - House colors (spawn points)
     - Food_1 colors (regenerative resources)
     - Food_2 colors (finite resources)
   - A JSON file is saved for reuse on subsequent runs

4. **Launch the UI simulation**:
   ```bash
   python ui_simulation.py
   ```

5. **Interactive controls**:
   - **Human Slider (H)**: Set initial population (before start) or adjust during pause (cheat mode)
   - **Trust Slider (T)**: Set leadership trust threshold (0.0-1.0)
   - **Speed Slider**: Control simulation time speed
   - **[P] key**: Pause/unpause
   - **[R] key**: Toggle food respawn ON/OFF
   - **[T] key**: Toggle trust system ON/OFF
   - **Export button**: Save trust matrix and house movement data
   - **Reset button**: Restart simulation

6. **Output files** (on export or quit):
   - `trust_matrix.csv`: Trust relationships between all agents
   - `house_movements.csv`: House relocation log
   - `house_movement_plots/`: Analysis graphs and trajectory maps
   - `blue_population.csv`, `red_population.csv`: Population time series

---

### Headless Simulation

**Step-by-step launch procedure:**

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Configure simulation**:
   - Edit `config.py` for global settings
   - Edit `batch_simulation.py` for batch-specific settings:
     - `DAYS`: Number of simulation days per run
     - `N_RUNS`: Number of independent runs
     - `SEED_BASE`: Random seed base for reproducibility

3. **Prepare map**:
   - Set map path in `headless_simulation.py` or `batch_simulation.py`
   - **Recommended:** Run UI simulation once with your map to generate `*_color_analysis.json`
   - Headless will automatically load the JSON for consistent color mapping
   - **Alternative:** Define colors in `config.py` NEW_PALETTE (less flexible)

4. **Launch headless simulation**:
   ```bash
   # Single run
   python headless_simulation.py
   
   # Batch runs (multiple seeds)
   python batch_simulation.py
   ```

5. **Output files**:
   - `batch_results/all_*_combined_sim_function.csv`: Combined results from all runs
   - Individual run CSVs with time series data
   - Per-run metrics (births, deaths, trust, resource consumption)

6. **Generate analysis plots**:
   ```bash
   python batch_plot.py
   ```
   - Reads from `batch_results/all_*_combined_sim_function.csv`
   - Outputs 6 essential plots to `batch_results/plots/` directory
   - Includes:
     - Population overview (with confidence bands)
     - Trust evolution (within/between houses)
     - Resource flow (spawn vs consumption balance)
     - Zone exploitation (dominance by house)
     - Survival analysis (extinction timing)
     - Per-capita consumption metrics

## Project Structure

```
humans/
├── README.md                 # This file
├── config.py                 # Configuration constants and parameters
├── human.py                  # Human and House agent classes
├── trust_system.py          # Trust system implementation
├── resource_manager.py      # Resource management and zone detection
├── social_mechanics.py      # Social interaction and trust mechanics
├── simulation_utils.py      # World building and adaptive house relocation
├── ui_simulation.py         # Interactive UI simulation with pygame
├── headless_simulation.py   # Optimized headless simulation
├── batch_simulation.py      # Batch simulation runner
├── batch_plot.py            # Streamlined plotting suite (6 essential plots)
├── ui_components.py         # UI widgets (sliders, buttons)
├── map_color_tools.py       # Map color analysis and JSON generation
├── tests.py                 # Testing, validation, and performance benchmarks
├── images/                  # Map images and color mapping JSONs
├── batch_results/           # Simulation output data and plots
├── house_movement_plots/    # House relocation analysis visualizations
├── docs/
│   └── DOCUMENTATION.md     # Complete technical documentation
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

## Testing and Validation

Run the comprehensive test suite:

```bash
# Run all tests (validation + performance)
python tests.py --all

# Run only validation tests
python tests.py --validate

# Run only performance benchmarks
python tests.py --performance
```

Tests include:
- Trust system consistency checks
- House storage cap validation
- Configuration parameter validation
- Normalization factor verification
- Performance benchmarking across different scales

## Documentation

For complete technical documentation, see:
- **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** - Complete API reference, architecture, parameters, and extension points

## Troubleshooting

### Common Issues

1. **Pygame display errors**: Ensure you have a display available or set `SDL_VIDEODRIVER=dummy` for headless mode
2. **Import errors**: Verify all dependencies are installed in the virtual environment
3. **Memory issues**: Reduce `Nbre_HUMANS` or `MAP_WIDTH`/`MAP_HEIGHT` for large simulations
4. **Slow performance**: Use `headless_simulation.py` for batch runs instead of the UI version
5. **Map not loading**: Ensure corresponding `*_color_analysis.json` file exists in `images/` directory
6. **Houses not moving**: Check normalization parameters in `config.py` (MAX_HOUSE_STORAGE, etc.)

### Performance Tips

- Use `headless_simulation.py` for batch simulations
- Use `ui_simulation.py` for interactive exploration
- Run `python tests.py --performance` to benchmark your system
- Reduce visualization frequency for large populations
- Consider smaller map sizes for faster iteration



*This simulation represents a complex multi-agent system modeling human social behavior. The emergent patterns observed reflect both the programmed mechanics and the stochastic nature of agent interactions, providing insights into how individual behaviors can lead to collective social structures.*