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

- **Python 3.10 or higher** (tested with Python 3.12)
- **Virtual environment** (strongly recommended)
- A display for UI mode (or set `SDL_VIDEODRIVER=dummy` for headless)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd humans
   ```

2. **Create and activate virtual environment** (⚠️ REQUIRED):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
   **Important**: You must activate the virtual environment every time you open a new terminal before running any scripts.

3. **Install dependencies**:
   ```bash
   # Full installation with all dependencies (recommended)
   pip install -r requirements.txt
   
   # Or minimal installation (core dependencies only)
   pip install -r requirements-minimal.txt
   
   # Or development installation (includes testing tools)
   pip install -r requirements-dev.txt
   
   # Or manual installation
   pip install numpy pandas matplotlib pygame pygame-menu opencv-python tqdm pillow seaborn scipy networkx
   ```

## Quick Start

### Absolute Beginner - Quick Test Run

**If you just want to see it working immediately:**

1. Make sure you're in the project directory with virtual environment activated:
   ```bash
   source venv/bin/activate
   ```

2. Run the UI simulation with the default map:
   ```bash
   python ui_simulation.py
   ```

3. If this is your first time with the default map, you'll be prompted to define color categories interactively. Simply follow the prompts to categorize colors as:
   - **Wall** (impassable terrain - black borders)
   - **House** (spawn points - blue and red areas)
   - **Food_1** (regenerative resources - green/brown areas)
   - **Food_2** (finite resources - optional)

4. Once the menu appears, click **"Start"** and watch the simulation run!

5. Use **[P]** to pause/unpause, adjust sliders to experiment, and explore the controls.

---

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

---

## Additional Utilities

### Performance Optimization Tool

**`optimize_simulation.py`** - Interactive performance configuration helper

```bash
python optimize_simulation.py
```

This tool provides an interactive menu to optimize simulation performance for different use cases:
- **Speed Mode**: Fastest execution with reduced population (5+5 humans, 10 days)
- **Balanced Mode**: Good performance with reasonable detail (20+20 humans, 50 days)
- **Quality Mode**: Full detail for research-grade simulations (50+50 humans, 100+ days)

The script creates `config_performance.py` with optimized settings that you can import or copy into your main `config.py`.

**Use cases:**
- Quick testing during development
- Finding optimal parameters for your hardware
- Benchmarking different configurations

---

### Spatial Visualization

**`visualization_fixed.py`** - Generate spatial heatmaps of resource exploitation

```bash
python visualization_fixed.py
```

Creates overlay heatmaps showing:
- Blue vs Red family resource consumption patterns
- Zone-specific exploitation intensity
- Spatial distribution of foraging activity
- Consumption patterns overlaid on the actual map

**Requirements:**
- Batch simulation results in `batch_results/`
- Zone coordinates file (generated automatically)
- Map image from simulation

**Output:** High-resolution spatial heatmaps in `batch_results/` or specified directory.

---

### Zone Coordinate Extraction

**`generate_coords.py`** - Extract zone coordinates from map images

```bash
# Generate coordinates from a map
python generate_coords.py
```

This utility analyzes your map image and extracts the exact coordinates of each food zone, creating a `resources_coords.csv` file used by visualization tools.

**Use cases:**
- Setting up visualizations for custom maps
- Verifying zone detection accuracy
- Debugging map configuration issues

**Functions:**
- `generate_coords_from_simulation(map_path)`: Extract coordinates from map
- `generate_coords_from_batch_data(data_path)`: Infer zones from batch results

---

### Custom Map Setup Guide

**First-Time Map Setup Process:**

1. Place your PNG map in the `images/` directory
2. Edit the map path in `ui_simulation.py` or `headless_simulation.py`:
   ```python
   MAP_IMAGE_PATH = "images/your_map_name.png"
   ```

3. Run the UI simulation for the first time:
   ```bash
   python ui_simulation.py
   ```

4. **Interactive color definition** (one-time setup):
   - The system will display all unique colors found in your map
   - For each color, you'll be asked to categorize it as:
     - **Wall**: Impassable terrain (borders, obstacles)
     - **House**: Spawn points for Blue/Red families
     - **Food_1**: Regenerative food zones (decay and respawn)
     - **Food_2**: Finite resources (no respawn)
     - **Background**: Empty walkable space

5. Your choices are saved to `images/your_map_name.png_color_analysis.json`

6. All future runs (UI or headless) automatically load this JSON file - no re-prompting!

**Tips:**
- Use distinct colors for different terrain types
- Walls should form closed borders to keep agents in bounds
- Houses should be small, distinct colored areas
- Food zones work best as connected regions (not scattered pixels)
- Keep maps to reasonable sizes (100×60 to 200×120 cells) for performance

## Project Structure

```
humans/
├── README.md                 # This file
├── docs/
│   └── DOCUMENTATION.md     # Complete technical documentation
│
├── Core Simulation Files
│   ├── config.py                 # Configuration constants and parameters
│   ├── human.py                  # Human and House agent classes
│   ├── trust_system.py          # Trust system implementation
│   ├── resource_manager.py      # Resource management and zone detection
│   ├── social_mechanics.py      # Social interaction and trust mechanics
│   └── simulation_utils.py      # World building and adaptive house relocation
│
├── Simulation Runners
│   ├── ui_simulation.py         # Interactive UI simulation with pygame
│   ├── headless_simulation.py   # Optimized headless simulation
│   └── batch_simulation.py      # Batch simulation runner (multiple seeds)
│
├── Analysis & Visualization
│   ├── batch_plot.py            # Streamlined plotting suite (6 essential plots)
│   ├── visualization_fixed.py   # Spatial heatmaps of resource exploitation
│   ├── generate_coords.py       # Zone coordinate extraction from maps
│   └── plot_utils.py            # Additional plotting utilities
│
├── Utilities & Tools
│   ├── ui_components.py         # UI widgets (sliders, buttons)
│   ├── map_color_tools.py       # Map color analysis and JSON generation
│   ├── optimize_simulation.py   # Performance optimization helper
│   └── tests.py                 # Testing, validation, and performance benchmarks
│
├── Configuration Files
│   ├── requirements.txt         # Full dependencies
│   ├── requirements-minimal.txt # Core dependencies only
│   └── requirements-dev.txt     # Development and testing tools
│
├── Data & Output Directories
│   ├── images/                  # Map images and color mapping JSONs
│   ├── batch_results/           # Simulation output data and plots
│   │   └── plots/              # Generated visualization plots
│   └── house_movement_plots/    # House relocation analysis visualizations
│
└── venv/                        # Virtual environment (created during setup)
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

1. **"ModuleNotFoundError" or Import errors**: 
   - Make sure you activated the virtual environment: `source venv/bin/activate`
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check you're in the correct directory (project root)

2. **Pygame display errors**: 
   - Ensure you have a display available
   - For true headless mode (no display needed), set: `export SDL_VIDEODRIVER=dummy` before running
   - On remote servers, use `headless_simulation.py` instead of `ui_simulation.py`

3. **"cv2.imread returns None" or map loading errors**:
   - Verify the map file exists in the `images/` directory
   - Check the file path is correct (use absolute paths if needed)
   - Ensure the file is a valid PNG image
   - Verify file permissions (readable)

4. **Color mapping issues** (first run with new map):
   - If prompted repeatedly for colors, check your responses are valid
   - Delete the `*_color_analysis.json` file to restart the color definition process
   - Ensure your map has distinct colors (not gradients or anti-aliasing)

5. **Memory issues or crashes**: 
   - Reduce `Nbre_HUMANS` in `config.py` (try starting with 10-20)
   - Reduce `MAP_WIDTH`/`MAP_HEIGHT` for large simulations
   - Use `optimize_simulation.py` to create a performance-optimized configuration
   - Close other memory-intensive applications

6. **Slow performance**: 
   - Use `headless_simulation.py` for batch runs instead of the UI version
   - Reduce population size and map dimensions
   - Disable detailed metrics collection (`COLLECT_METRICS = False` in `simulation_utils.py`)
   - Run `python tests.py --performance` to benchmark your system

7. **Map not loading in headless mode**: 
   - Ensure corresponding `*_color_analysis.json` file exists in `images/` directory
   - Run UI simulation once first to generate the JSON file
   - Or manually define colors in `config.py` NEW_PALETTE

8. **Houses not moving**: 
   - Check normalization parameters in `config.py` (`MAX_HOUSE_STORAGE`, `HOUSE_MAX_TRAVEL_PER_DAY`, etc.)
   - Verify house relocation weights (`HOUSE_RELOC_ALPHA1`, `ALPHA2`, `ALPHA3`, `BETA`)
   - Enable relocation debugging by checking `house_movements.csv` output
   - Increase pressure by adjusting alpha/beta parameters

9. **No plots generated**:
   - Verify `batch_results/` directory exists and contains CSV data
   - Check that `matplotlib` and `seaborn` are installed
   - Ensure the CSV file format matches expected columns
   - Run with `python batch_plot.py --verbose` for detailed error messages

10. **Virtual environment issues**:
    - If commands fail, always verify venv is activated: `which python` should point to `./venv/bin/python`
    - Recreate venv if corrupted: `rm -rf venv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

### Performance Tips

- Use `headless_simulation.py` for batch simulations
- Use `ui_simulation.py` for interactive exploration
- Use `optimize_simulation.py` to auto-configure for your needs
- Run `python tests.py --performance` to benchmark your system
- Reduce visualization frequency for large populations
- Consider smaller map sizes for faster iteration

---

## Typical Workflow

### For First-Time Users

1. **Setup**: Install dependencies and activate virtual environment
2. **Quick test**: Run `python ui_simulation.py` with default map
3. **Define colors**: Interactively categorize map colors (one-time setup)
4. **Explore**: Use UI controls to experiment with parameters
5. **Export data**: Use Export button to save results

### For Research & Analysis

1. **Configure**: Edit `config.py` or use `optimize_simulation.py` for optimal settings
2. **Setup map**: Place custom map in `images/` and run UI once to define colors
3. **Batch runs**: Execute `python batch_simulation.py` for multiple seeds
4. **Analyze**: Run `python batch_plot.py` to generate comprehensive visualizations
5. **Spatial analysis**: Use `python visualization_fixed.py` for heatmaps
6. **Custom analysis**: Export CSV data for external analysis tools

### For Development & Testing

1. **Test setup**: Run `python tests.py --all` to verify installation
2. **Benchmark**: Run `python tests.py --performance` to check system capabilities
3. **Iterate**: Make changes to config or code
4. **Validate**: Test with UI simulation first, then batch runs
5. **Analyze**: Generate plots and review metrics

---

## Quick Reference

### Essential Commands

```bash
# Activate environment (always required first)
source venv/bin/activate

# Interactive simulation with UI
python ui_simulation.py

# Single headless run
python headless_simulation.py

# Batch runs (multiple seeds)
python batch_simulation.py

# Generate analysis plots
python batch_plot.py

# Performance optimization helper
python optimize_simulation.py

# Spatial heatmap visualization
python visualization_fixed.py

# Run all tests
python tests.py --all
```

### Key Files to Know

- **config.py**: All simulation parameters (edit this to customize behavior)
- **images/*.png**: Your map files
- **images/*.json**: Color mappings (auto-generated, one per map)
- **batch_results/*.csv**: Simulation output data
- **batch_results/plots/**: Generated visualizations
- **trust_matrix.csv**: Exported trust relationships
- **house_movements.csv**: House relocation log

---

*This simulation represents a complex multi-agent system modeling human social behavior. The emergent patterns observed reflect both the programmed mechanics and the stochastic nature of agent interactions, providing insights into how individual behaviors can lead to collective social structures.*