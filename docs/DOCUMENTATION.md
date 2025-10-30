# Human Society Simulation - Complete Documentation

**Version:** 2.0  
**Last Updated:** October 2025

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [API Reference](#api-reference)
3. [Configuration Parameters](#configuration-parameters)
4. [Module Descriptions](#module-descriptions)
5. [Data Flow](#data-flow)
6. [Performance Optimization](#performance-optimization)
7. [Extension Points](#extension-points)

---

## System Architecture

### Overview

The Human Society Simulation is a modular agent-based system with clear separation between core simulation logic, visualization, and analysis components. The architecture supports both real-time interactive simulation and high-performance batch processing.

### Core Components

#### 1. Agent System (`human.py`)

**Human Class**
- Autonomous agents with energy, position, and social relationships
- Behavior includes movement, foraging, social interaction, and reproduction
- State management for alive/dead status and resource carrying capacity
- Bag system for resource carrying (default capacity: 10 units)
- Energy management with decay, consumption, and thresholds

**House Class**
- Safe zones serving as resource storage and family centers
- Color-coded for competing families (Blue vs Red)
- Storage tracking with 10,000 unit maximum capacity
- Adaptive relocation system based on environmental factors
- Inertia mechanism to prevent excessive movement

**TrustSystem Class (`trust_system.py`)**
- Numpy-based trust relationship tracking with O(1) lookups
- Cached trusted/untrusted sets for efficient queries
- Batch update support with dirty tracking for performance
- Self-confidence tracking per agent
- Trust decay mechanism (configurable interval and amount)

#### 2. Resource Management (`resource_manager.py`)

**Resource Grid**
- 3D numpy array: `resources[y, x, 0]` = lifetime, `resources[y, x, 1]` = amount
- Zone-based spawning with adaptive respawn dynamics
- Two food types:
  - **Food_1**: Regenerative, decaying, adaptive respawn (higher consumption → slower respawn)
  - **Food_2**: Finite, non-decaying, no respawn
- Food decay simulation with configurable lifetime

**Zone Detection**
- Image processing pipeline for map-based zone identification
- Connected component analysis for zone merging
- Color palette mapping stored in JSON files per map
- Persistent color categorization to avoid re-prompting

**Adaptive Spawning**
- Consumption-based respawn rate adjustment
- Cooldown periods for depleted zones
- Per-zone parameter customization
- Overexploitation awareness for Food_1 zones

#### 3. Simulation Engine

**UI Simulation (`ui_simulation.py`)**
- Interactive pygame-based simulation with real-time visualization
- Menu-driven interface with parameter controls
- Runtime toggles for food respawn and trust mechanisms
- Population slider with "cheat mode" during pause
- Trust threshold slider for leadership emergence
- Export functionality for trust matrices, population data, and house movement logs
- House movement visualization and analytics

**Headless Simulation (`headless_simulation.py`)**
- Optimized for batch processing without UI overhead
- Occupancy map optimization for neighbor finding
- Per-zone resource tracking and analytics
- Standardized agent behavior across all simulation modes
- Supports single and batch runs with different seeds

**Performance Optimizations**
- Precomputed zone mappings for O(1) lookups
- Batch trust updates with deferred refresh
- Efficient dead agent cleanup
- Occupancy maps for spatial queries

#### 4. Social Mechanics (`social_mechanics.py`)

**Trust Dynamics**
- Food sharing increases trust between agents
- House contributions boost family trust (+0.001 per contribution)
- Trust decay every N days (default: -0.05 every 20 days)
- Leadership competition based on trust scores

**Reproductive System**
- Mutual trust requirements for mating (threshold: 0.7)
- Energy cost constraints (default: 8.0 per parent)
- Cooldown periods to prevent rapid reproduction (default: 2000 ticks)
- Requires same house membership

**Competition Resolution**
- Leadership selection at dawn (both houses support multiple leaders)
- Trust-based follower assignment
- Resource location memory sharing
- Leaders validated by spot correctness (±2 cells)
- Trust/self-confidence adjustments based on outcome
- Leaders receive 10% share bonus

#### 5. Adaptive House Relocation (`simulation_utils.py`)

**Decision System**
- Evaluates three factors daily:
  - **S (Storage)**: Normalized food reserves (0-1)
  - **D (Travel)**: Normalized average daily travel distance (0-1)
  - **F (Food)**: Normalized local food availability (0-1)

**Decision Equation**
```
Pressure = α₁(1-S) + α₂D + α₃(1-F) - β*E_move
P_move = sigmoid(Pressure) * (1 - Inertia)
```

**Parameters**
- `α₁, α₂, α₃`: Weights for S, D, F importance (default: 1.0, 0.8, 1.2)
- `β`: Movement cost sensitivity (default: 6.0)
- `E_move`: Base movement cost (default: 0.1)
- `Inertia`: Stability memory (increases when stationary, resets on move)
- `max_move_distance`: Maximum cells per night (default: 2)

**Location Selection**
- Finds best spot within move radius
- Maximizes local food density
- Avoids resource tiles and walls
- Gradual movement (2 cells per night)

---

## API Reference

### Core Classes

#### TrustSystem

**Constructor**
```python
TrustSystem()
```

**Methods**

##### `init_human(h_id: int) -> None`
Initialize data structures for a new human agent. Idempotent - safe to call multiple times.

##### `trust_score(h_id: int, other_id: int) -> float`
Get trust score between two agents.
- Returns: 0.0-1.0, or 0.5 if no prior interaction
- Raises: ValueError if agent attempts to query self-trust

##### `increase_trust(trustor_id: int, trustee_id: int, increment: float = 0.01, refresh: bool = True) -> None`
Increase trust between two agents with optional immediate cache refresh.

##### `flush() -> None`
Refresh trust lists for all agents marked as dirty since last flush.

#### Human

**Constructor**
```python
Human(human_id: int, sex: str, x: int, y: int, home: House, codes,
      initial_energy: float = 10.0, exploration_factor: int = 2,
      bag_capacity: int = 10)
```

**Parameters:**
- `human_id`: Unique identifier
- `sex`: 'M' or 'F'
- `x, y`: Starting position
- `home`: House object
- `codes`: Terrain map array
- `initial_energy`: Starting energy (default: 10.0)
- `exploration_factor`: Movement randomness (default: 2)
- `bag_capacity`: Max resource carrying capacity (default: 10)

**Key Methods**

##### `step(...) -> Tuple[Optional[Tuple[int, int]], bool]`
Execute one simulation step. Returns (picked_location, shared_food).

##### `deposit_food() -> None`
Deposit all bag contents into house storage.

##### `store_in_bag(spot: Tuple[int, int]) -> None`
Store food from current location if energy ≥ 9 and bag not full.

#### House

**Constructor**
```python
House(x: int, y: int, color: Tuple[int, int, int])
```

**Attributes:**
- `storage`: Current food storage (max: 10,000)
- `color`: RGB tuple for visualization
- `inertia`: Movement stability memory (0.0-1.0)

**Methods**

##### `deposit(amount: float) -> None`
Add food to storage, capped at 10,000 units.

### Core Functions

#### `simulate_headless(num_days, seed, map_path, min_size, tol, precomputed, preview_every, preview_dir) -> Optional[Tuple]`
Run optimized headless simulation.

**Returns:**
```python
(days, blue_pop, red_pop, within_trust, between_trust,
 within_blue_trust, within_red_trust, dead_blue_cum, dead_red_cum,
 born_blue_daily, born_red_daily, zone_spawned, zone_consumed, ...)
```

#### `build_world(map_path, seed, color_mapping) -> Tuple`
Initialize simulation world from map image.

**Returns:**
```python
(zone_map, resources, humans, houses, food_cells_type1,
 food_cells_type2, food_ids, trust_system)
```

#### `should_move_house(house, humans, resources) -> float`
Calculate house relocation probability (0.0-1.0).

#### `find_best_house_location(house, humans, resources, max_move_distance=2) -> Optional[Tuple[int, int]]`
Find optimal new house location within movement radius.

#### `boost_house_trust(trust_system, contributor, humans, increment=0.001) -> None`
Increase trust in contributor among housemates.

#### `run_competition(family, trust_system, threshold=0.55) -> None`
Run leadership competition within a family. Supports multiple leaders.

#### `to_mate(h1, h2, trust_system, humans, codes, next_id, threshold=0.7, energy_cost=8.0) -> Tuple[int, int]`
Attempt mating between two humans. Returns (num_children, updated_next_id).

#### `export_trust_matrix(trust_system, humans, filename="trust_matrix.csv") -> None`
Export trust matrix to CSV. Diagonal shows self-confidence values.

#### `export_house_movement_log(movement_log, filename="house_movements.csv") -> None`
Export house movement events with decision factors.

#### `plot_house_movement_analysis(movement_log) -> None`
Generate comprehensive house movement analysis plots (3x2 layout).

---

## Configuration Parameters

### Map Settings

**`MAP_WIDTH`, `MAP_HEIGHT`**
- Default: 100 × 60 cells
- Effect: Simulation world dimensions
- Performance: O(W×H) operations

**`CELL_SIZE`**
- Default: 8 pixels
- Effect: Visualization pixel size per grid cell

**`MAP_IMAGE_PATH`**
- Location: Set in `ui_simulation.py` or `headless_simulation.py`
- Requires: Corresponding `*_color_analysis.json` file

### Population Parameters

**`Nbre_HUMANS`**
- Default: 20 (adjustable via UI slider)
- Recommended range: 10-500
- Performance impact: O(N²) for interactions

**`ENERGY_COST`**
- Default: 8.0
- Effect: Energy required for mating
- Range: 4.0-20.0 (lower = faster growth)

**`MATING_COOLDOWN`**
- Default: 2000 ticks (10 days)
- Effect: Minimum time between mating attempts for same pair

### Resource Parameters

**`INITIAL_FOOD_COUNT`**
- Default: 50
- Recommended: 2-3 per human
- Effect: Starting food abundance

**`FOOD_LIFETIME`**
- Default: 50 ticks
- Effect: Food_1 decay time (Food_2 never decays)

**`FOOD_STACK`**
- Default: 20
- Effect: Maximum food units per cell

**`FOOD_SPAWN_COUNT`**
- Default: 20
- Effect: Units spawned per event

### Time Parameters

**`DAY_LENGTH`**
- Default: 200 ticks
- Day/night split: 70% day, 30% night
- Recommended: 100-600 ticks

### Trust Parameters

**`TRUST_INCREMENT`**
- Default: 0.01
- Effect: Trust increase per positive interaction

**`TRUST_DECAY_AMOUNT`**
- Default: 0.05
- Effect: Trust decrease per decay event

**`TRUST_DECAY_INTERVAL`**
- Default: 20 days
- Effect: How often trust decays

**Trust Thresholds**
- Mating: 0.7
- Leadership: 0.55 (adjustable via UI slider)

### Adaptive House Migration Parameters

**`HOUSE_RELOC_ALPHA1, ALPHA2, ALPHA3`**
- Defaults: 1.0, 0.8, 1.2
- Effect: Weights for S, D, F factors

**`HOUSE_RELOC_BETA`**
- Default: 6.0
- Effect: Movement cost sensitivity

**`HOUSE_RELOC_E_MOVE`**
- Default: 0.1
- Effect: Base energy cost of moving

**`MAX_HOUSE_STORAGE`**
- Default: 10,000.0
- Effect: Maximum storage and normalization factor

**`HOUSE_MAX_TRAVEL_PER_DAY`**
- Default: 40.0
- Effect: Normalization for travel distance

**`HOUSE_LOCAL_RADIUS`**
- Default: 10 cells
- Effect: Radius for local food availability calculation

**`MAX_FOOD_PER_CELL`**
- Default: 5.0
- Effect: Normalization for food density

**`HOUSE_INERTIA_STEP, INERTIA_MAX`**
- Defaults: 0.04, 1.0
- Effect: Inertia accumulation per stationary day

### Feature Flags

**`ENABLE_FOOD_RESPAWN`**
- Default: True
- Runtime toggle: 'R' key in UI

**`ENABLE_TRUST`**
- Default: True
- Runtime toggle: 'T' key in UI

**`ENABLE_MATING`**
- Default: True
- Effect: Enable/disable reproduction

**`COLLECT_METRICS`**
- Default: False
- Effect: Enable detailed per-tick metrics collection

### Adaptive Spawning Parameters

**`I_max, I_min, k, cooldown_days`**
- Used by `resource_spawn_interval_inverse()`
- Control respawn rate adaptation based on consumption

---

## Module Descriptions

### `config.py`
Centralized configuration constants. Includes validation to ensure parameters are within valid ranges.

### `human.py`
Defines `Human` and `House` agent classes with all behavior logic.

### `trust_system.py`
Manages trust relationships using numpy arrays for efficiency.

### `resource_manager.py`
Handles resource spawning, decay, zone detection, and adaptive respawn logic.

### `social_mechanics.py`
Implements trust dynamics, mating, competition, and cooperation mechanics.

### `simulation_utils.py`
World building, house relocation logic, and shared utilities.

### `ui_simulation.py`
Interactive pygame simulation with sliders, buttons, and real-time visualization.

### `headless_simulation.py`
Optimized non-visual simulation for batch processing.

### `batch_simulation.py`
Runs multiple headless simulations with different seeds for statistical analysis.

### `batch_plot.py`
Generates 30+ comprehensive visualization plots from batch results.

### `ui_components.py`
Reusable UI widgets (sliders, buttons, etc.).

### `map_color_tools.py`
Tools for analyzing map colors and creating persistent color mappings.

### `validation_utils.py`
Testing and validation utilities for simulation correctness.

### `performance_test.py`
Benchmarking script for performance analysis.

---

## Data Flow

### Simulation Cycle

```
1. Initialization
   ├── Load map and analyze colors (or load JSON)
   ├── Build world (zones, resources, humans, houses, trust)
   └── Initialize UI or headless mode

2. Daily Loop (each day = DAY_LENGTH ticks)
   ├── Dawn Phase (tick 0 of day)
   │   ├── Trust decay (every TRUST_DECAY_INTERVAL days)
   │   ├── House relocation evaluation and movement
   │   ├── Leadership competition (both houses)
   │   └── Mating attempts (if ENABLE_MATING and trust_enabled)
   │
   ├── Day/Night Cycle (ticks 1-DAY_LENGTH)
   │   ├── Agent movement and foraging (day: 0-70%)
   │   ├── Return home and deposit food (night: 70-100%)
   │   ├── Resource consumption and sharing
   │   ├── Trust relationship updates
   │   └── Energy decay and death checks
   │
   ├── Resource Management (continuous)
   │   ├── Food decay (Food_1 only)
   │   ├── Adaptive spawning (based on consumption)
   │   └── Zone consumption tracking
   │
   └── End of Day
       ├── Metrics collection
       ├── Population cleanup
       └── Trust cache refresh

3. Export/Analysis
   ├── Trust matrix CSV
   ├── House movement log and plots
   ├── Population time series
   └── Batch analysis plots (if batch mode)
```

### Trust System Flow

```
Agent Interaction → Trust Update → Cache Invalidation → Batch Refresh
     ↓                    ↓              ↓                ↓
Food Sharing        Score Change    Dirty Tracking   Trust Lists
House Contribution  Increment       Mark Agents      Flush on Dawn
```

### House Relocation Flow

```
End of Day → Calculate S,D,F → Compute Pressure → Sigmoid + Inertia
     ↓              ↓               ↓                    ↓
 Members      Normalize        Decision Eq.        Probabilistic
 Activity     (0-1 scale)      α₁(1-S)+α₂D+α₃(1-F)  P_move
     ↓                              ↓                    ↓
Daily Travel              Compare to E_move      Sample Decision
Local Food                                            ↓
Storage Level                               Find Best Location
                                            (max food, avoid walls)
                                                     ↓
                                              Move House (2 cells)
                                              Update Members
                                              Reset Inertia
```

---

## Performance Optimization

### Memory Management
- Numpy arrays for efficient numerical operations
- Pre-allocated arrays with managed growth
- Regular dead agent cleanup
- Batch processing for large-scale runs

### Computational Optimizations
- O(1) zone lookups via precomputed mappings
- Batch trust updates with deferred refresh
- Vectorized operations where possible
- Occupancy maps for spatial neighbor queries

### Scalability Considerations
- Configurable population sizes (10-500)
- Adjustable map dimensions (50×30 to 200×120)
- Modular component design for easy extension
- Headless mode for batch efficiency

### Best Practices
- Use headless mode for batch simulations
- Use UI mode for interactive exploration
- Reduce visualization frequency for large populations
- Monitor memory usage for 300+ agents
- Precompute zone mappings and save to JSON

---

## Extension Points

### Custom Agent Behaviors
- Inherit from `Human` class
- Override `step()` method for custom logic
- Add new interaction types
- Implement learning or adaptation

### Additional Social Mechanics
- Extend `TrustSystem` for new relationship types
- Add new competition mechanisms
- Implement coalition formation
- Create hierarchical social structures

### New Resource Types
- Extend resource grid dimensions (e.g., add water, tools)
- Add resource-specific behaviors
- Implement resource transformations (crafting)

### Custom Visualizations
- Add new plot types to `batch_plot.py`
- Create specialized heatmap visualizations
- Implement real-time metric displays in UI
- Export data for external analysis tools (R, Tableau)

### Map and Environment Extensions
- Support multiple map layers
- Add dynamic environmental events
- Implement seasonal resource variations
- Create procedurally generated maps

---

## Testing and Validation

### Consistency Checks
- Trust system symmetry validation
- Resource conservation laws
- Agent behavior verification
- Population dynamics validation

### Performance Benchmarking
- Use `performance_test.py` for profiling
- Monitor memory usage with `psutil`
- Track simulation speed (ticks/second)
- Identify bottlenecks with cProfile

### Validation Tools
- Trust matrix export for manual verification
- Population tracking consistency checks
- House movement log validation
- Zone consumption balance verification

---

## Troubleshooting

### Common Issues

**Population extinction**
- Increase `INITIAL_FOOD_COUNT` or reduce `ENERGY_COST`
- Lower mating thresholds or increase food respawn rates

**No social structure emerging**
- Lower trust thresholds for mating and leadership
- Increase trust increment values
- Ensure trust system is enabled

**Excessive competition**
- Increase resource abundance (`FOOD_STACK`, `FOOD_SPAWN_COUNT`)
- Distribute resources more evenly
- Adjust house relocation parameters

**Poor performance**
- Reduce population size or map dimensions
- Use headless mode for batch runs
- Disable detailed metrics collection

**Houses not moving**
- Check normalization parameters (`MAX_HOUSE_STORAGE`, etc.)
- Verify S, D, F calculations are meaningful
- Adjust α, β weights to increase pressure
- Reduce inertia accumulation

---

**For detailed usage instructions, see README.md**  
**For map setup and color configuration, see map_color_tools.py**  
**For batch analysis plotting, see batch_plot.py --help**

---

*This simulation models emergent social behavior through individual agent interactions. Patterns observed reflect both programmed mechanics and stochastic variation.*

