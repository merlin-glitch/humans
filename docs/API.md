# API Reference

## Core Classes

### TrustSystem

The central trust management system for tracking relationships between agents.

#### Constructor
```python
TrustSystem()
```
Initialize an empty trust system.

#### Methods

##### `init_human(h_id: int) -> None`
Initialize data structures for a new human agent.

**Parameters:**
- `h_id`: Unique identifier for the human agent

**Note:** This method is idempotent - safe to call multiple times for the same human ID.

##### `trust_score(h_id: int, other_id: int) -> float`
Get the trust score between two agents.

**Parameters:**
- `h_id`: The trustor (agent whose trust we're measuring)
- `other_id`: The trustee (agent being trusted)

**Returns:**
- Trust score between 0.0 and 1.0, or 0.5 if no prior interaction

##### `increase_trust(trustor_id: int, trustee_id: int, increment: float = 0.01, refresh: bool = True) -> None`
Increase trust between two agents.

**Parameters:**
- `trustor_id`: The agent whose trust is being updated
- `trustee_id`: The agent being trusted more
- `increment`: Amount to increase trust by
- `refresh`: Whether to immediately refresh cached trust lists

##### `update_on_meeting(h1: Human, h2: Human, resources: list) -> None`
Update trust based on a meeting between two agents.

**Parameters:**
- `h1`: First agent in the meeting
- `h2`: Second agent in the meeting
- `resources`: List of current resources for validation

##### `flush() -> None`
Refresh trust lists for all agents marked as dirty since last flush.

### Human

Individual agent class representing a human in the simulation.

#### Constructor
```python
Human(human_id: int, sex: str, x: int, y: int, home: House, codes, 
      initial_energy: float = 10.0, exploration_factor: int = 2, 
      bag_capacity: int = 10)
```

**Parameters:**
- `human_id`: Unique identifier for this human
- `sex`: Gender ('homme' or 'femme')
- `x`, `y`: Starting position coordinates
- `home`: House object this human belongs to
- `codes`: Terrain map array
- `initial_energy`: Starting energy level
- `exploration_factor`: Movement randomness factor
- `bag_capacity`: Maximum resources this human can carry

#### Methods

##### `step(resources, houses, humans, trust_system, is_day: bool, action_cost: float = 0.05, food_gain: float = 1.0, decay_rate: float = 0.001) -> Tuple[Optional[Tuple[int, int]], bool]`
Execute one simulation step for this human.

**Parameters:**
- `resources`: 3D resource grid array
- `houses`: List of house objects
- `humans`: List of all humans
- `trust_system`: Trust system for social interactions
- `is_day`: Whether it's currently day time
- `action_cost`: Energy cost per action
- `food_gain`: Energy gained from food
- `decay_rate`: Energy decay rate

**Returns:**
- Tuple of (picked_location, shared_food) where picked_location is (x,y) if food was picked, None otherwise

### House

Represents a safe zone and resource storage for a family group.

#### Constructor
```python
House(x: int, y: int, color: Tuple[int, int, int])
```

**Parameters:**
- `x`, `y`: Position coordinates
- `color`: RGB color tuple for visualization

#### Attributes
- `storage`: Current food storage amount
- `color`: RGB color for visualization

#### Methods

##### `store_food(amount: int) -> None`
Add food to the house storage.

**Parameters:**
- `amount`: Number of food units to add

## Core Functions

### Simulation Functions

#### `simulate_headless(num_days: int, seed: Optional[int] = None, map_path: str, min_size: int, tol: int, precomputed=None, preview_every: Optional[int] = None, preview_dir: str = "previews") -> Tuple`
Run a complete simulation for the specified number of days.

**Parameters:**
- `num_days`: Number of simulation days to run
- `seed`: Random seed for reproducible results (optional)
- `return_zone_series`: Include per-zone data in return (default False)
- `return_final_state`: Include final human/trust state (default False)
- `progress`: Show progress bar during simulation (default True)

**Returns:**
Tuple containing simulation results. Basic format:
```python
(days, blue_pop, red_pop, within_trust, between_trust,
 within_blue_trust, within_red_trust, dead_blue_cum, dead_red_cum,
 born_blue_daily, born_red_daily)
```

#### `simulate_headless(num_days: int, seed: Optional[int], map_path: str, min_size: int, tol: int, precomputed=None, preview_every: Optional[int] = None, preview_dir: str = "previews") -> Optional[Tuple]`
Optimized headless simulation runner.

**Parameters:**
- `num_days`: Number of days to simulate
- `seed`: Random seed
- `map_path`: Path to map image file
- `min_size`: Minimum zone size for detection
- `tol`: Color tolerance for zone identification
- `precomputed`: Precomputed zone data (optional)
- `preview_every`: Generate preview frames every N ticks (optional)
- `preview_dir`: Directory to save preview frames

### Social Functions

#### `boost_house_trust(trust_system: TrustSystem, contributor: Human, humans: List[Human], increment: float = 0.001) -> None`
Increase trust in a contributor among their housemates.

**Parameters:**
- `trust_system`: The trust system managing relationships
- `contributor`: The human who made a contribution
- `humans`: List of all humans in the simulation
- `increment`: Amount to increase trust by

#### `run_competition(family: List[Human], trust_system: TrustSystem, threshold: float = 0.55) -> None`
Run leadership competition within a family group.

**Parameters:**
- `family`: List of humans in the same house
- `trust_system`: Trust system for evaluating relationships
- `threshold`: Minimum trust score to become a leader

#### `to_mate(h1: Human, h2: Human, trust_system: TrustSystem, humans: List[Human], codes, next_id: int, threshold: float = 0.7, energy_cost: float = 5.0) -> Tuple[int, int]`
Attempt mating between two humans if conditions are met.

**Parameters:**
- `h1`, `h2`: Potential parents
- `trust_system`: Trust system for checking mutual trust
- `humans`: List of all humans (children will be added here)
- `codes`: Terrain map array for child placement
- `next_id`: Next available human ID for children
- `threshold`: Minimum mutual trust required
- `energy_cost`: Energy cost for each parent

**Returns:**
- Tuple of (num_children_created, updated_next_id)

### Resource Functions

#### `identify_zones(map_image_path: str, min_size: int = 30, tol: int = 30) -> Tuple[np.ndarray, Dict[str, int]]`
Identify and label zones from a map image.

**Parameters:**
- `map_image_path`: Path to the PNG map image
- `min_size`: Minimum zone size to keep as separate
- `tol`: Color tolerance for zone identification

**Returns:**
- Tuple of (zone_map, food_zone_ids) where zone_map is a 2D numpy array with zone IDs

#### `resource_spawn_interval_inverse(f_avg: float, zone_id: int = 0, zone_map=None, I_max: int = 200, k: float = 0.6, I_min: int = 10, stock_today=None, spawn_baseline: int = FOOD_SPAWN_COUNT, reset: bool = False, cooldown_days: int = 10) -> Optional[int]`
Compute adaptive respawn interval based on consumption patterns.

**Parameters:**
- `f_avg`: Average consumption rate for this zone
- `zone_id`: Unique identifier for the zone
- `zone_map`: Map array for stock calculation (optional)
- `I_max`: Maximum respawn interval
- `k`: Responsiveness factor
- `I_min`: Minimum respawn interval
- `stock_today`: Current food stock in zone (optional)
- `spawn_baseline`: Baseline spawn amount for threshold calculation
- `reset`: Reset cooldown state
- `cooldown_days`: Cooldown period when depleted

**Returns:**
- Respawn interval in ticks, or None if zone is in cooldown

### Utility Functions

#### `_avg_pairwise_trust(ids: List[int], trust_system: TrustSystem) -> float`
Calculate average trust over all ordered pairs in a list of agent IDs.

**Parameters:**
- `ids`: List of human IDs
- `trust_system`: Trust system for querying scores

**Returns:**
- Average trust score, or 0.0 if fewer than 2 IDs

#### `export_trust_matrix(trust_system: TrustSystem, humans: List[Human], filename: str = "trust_matrix.csv") -> None`
Export trust matrix to CSV file.

**Parameters:**
- `trust_system`: Trust system to export
- `humans`: List of humans (determines matrix dimensions)
- `filename`: Output CSV filename

## Configuration Constants

### Map Settings
- `MAP_WIDTH`: Width of simulation world in grid cells (default: 100)
- `MAP_HEIGHT`: Height of simulation world in grid cells (default: 60)
- `CELL_SIZE`: Pixel size of each grid cell for visualization (default: 8)

### Population Settings
- `Nbre_HUMANS`: Initial number of humans in the simulation (default: 20)
- `ENERGY_COST`: Energy required for mating (default: 8.0)
- `MATING_COOLDOWN`: Minimum time between mating attempts (default: 10 days)

### Resource Settings
- `INITIAL_FOOD_COUNT`: Starting food units (default: 50)
- `FOOD_LIFETIME`: Ticks before food disappears (default: 9000)
- `FOOD_STACK`: Maximum food units per cell (default: 5)
- `FOOD_SPAWN_COUNT`: Food units spawned per event (default: 20)

### Time Settings
- `DAY_LENGTH`: Simulation ticks per day (default: 200)
- Day/night cycle: 70% day, 30% night

### Zone Color Palette
```python
NEW_PALETTE = {
    (0, 0, 0): 0,           # border/walls
    (94, 185, 30): 1,       # grass background
    (0, 0, 128): 2,         # blue house
    (255, 0, 0): 3,         # red house
    (54, 109, 70): 4,       # food zones (dark green)
}
```

## Error Handling

### Common Exceptions
- `ValueError`: Invalid parameter values (e.g., negative population)
- `FileNotFoundError`: Missing map image files
- `IndexError`: Out-of-bounds array access
- `KeyError`: Missing trust relationship data

### Best Practices
- Always initialize humans in the trust system before interactions
- Check agent alive status before operations
- Validate map dimensions before simulation start
- Use try-catch blocks for file operations

## Performance Considerations

### Memory Usage
- Trust system uses numpy arrays for efficiency
- Pre-allocate arrays where possible
- Clean up dead agents regularly

### Computational Complexity
- Trust lookups: O(1) average case
- Zone detection: O(W×H) where W,H are map dimensions
- Agent step: O(N) where N is population size

### Optimization Tips
- Use headless mode for batch simulations
- Batch trust updates when possible
- Cache frequently accessed data
- Monitor memory usage for large populations
