"""
config.py - Configuration constants for the human society simulation

Part of the Human Society Simulation project.

Defines map dimensions, resource parameters (lifetime, spawn rates),
population settings, energy costs, and the color palette for zone mapping.
"""

# Population settings
Nbre_HUMANS = 100  # Initial number of humans in the simulation

# ── Map dimensions (in cells) ──────────────────────────────────────────────
MAP_WIDTH  = 100  # Width of simulation world in grid cells
MAP_HEIGHT = 60   # Height of simulation world in grid cells

# ── Cell size (in pixels) ──────────────────────────────────────────────────
CELL_SIZE = 8  # Pixel size of each grid cell for visualization

# ── House dimensions ────────────────────────────────────────────────────────
HOUSE_SIZE = 1  # Size of houses in grid cells (1x1)

# ── Zone color palette: RGB → integer mapping ─────────────────────────────
# Legend: 0=border/walls, 1=grass background, 2=blue house, 3=red house, 4=food zones
NEW_PALETTE = {
    (0, 0, 0): 0,           # border/walls
    (94, 185, 30): 1,         # grass background
    (0, 0, 128): 2,         # blue house
    (255, 0, 0): 3,         # red house
    (54, 109, 70): 4,       # food zones (dark green)
}
# ── Resource parameters ────────────────────────────────────────────────────
INITIAL_FOOD_COUNT = 500  # Starting food units (about 25 per human - very generous)
SPAWN_INTERVAL     = 500  # Ticks between spawn events (legacy, now adaptive)
FOOD_SPAWN_COUNT   = 100  # Food units spawned per event (increased from 50)
FOOD_LIFETIME      = 9000  # Ticks before food disappears (45 days)
FOOD_STACK        = 1000  # Maximum food units per cell (increased from 100)

# ── Energy and reproduction ────────────────────────────────────────────────
ENERGY_COST       = 8.0   # Energy required for mating (prevents rapid reproduction)
DAY_LENGTH        = 200   # Simulation ticks per day (70% day, 30% night)
MATING_COOLDOWN   = 10 * DAY_LENGTH  # 10 days between mating attempts

# ── Standardized agent behavior parameters ────────────────────────────────
ACTION_COST       = 0.001 # Energy cost per action (further reduced from 0.005)
FOOD_GAIN         = 5.0   # Energy gained from consuming food (increased from 2.0)
ENERGY_DECAY_RATE = 0.001 # Energy decay rate per tick (further reduced from 0.002)
TRUST_INCREMENT   = 0.001 # Trust increase when sharing resources

# ── Feature flags ──────────────────────────────────────────────────────────────
ENABLE_MATING = True      # Enable human reproduction
COLLECT_METRICS = True    # Enable detailed metrics collection
PER_ZONE_RESPAWN = True   # Enable per-zone resource spawning

# ── Configuration validation ──────────────────────────────────────────────────

def validate_config() -> None:
    """
    Validate all configuration parameters for consistency and correctness.
    
    Raises:
        ValueError: If any configuration parameter is invalid
        AssertionError: If configuration violates simulation constraints
    """
    # Map dimensions validation
    if MAP_WIDTH <= 0 or MAP_HEIGHT <= 0:
        raise ValueError(f"Map dimensions must be positive, got {MAP_WIDTH}x{MAP_HEIGHT}")
    
    if MAP_WIDTH > 1000 or MAP_HEIGHT > 1000:
        raise ValueError(f"Map dimensions too large (max 1000x1000), got {MAP_WIDTH}x{MAP_HEIGHT}")
    
    # Population validation
    if Nbre_HUMANS <= 0:
        raise ValueError(f"Number of humans must be positive, got {Nbre_HUMANS}")
    
    if Nbre_HUMANS > 1000:
        raise ValueError(f"Too many humans (max 1000), got {Nbre_HUMANS}")
    
    # Resource parameters validation
    if INITIAL_FOOD_COUNT <= 0:
        raise ValueError(f"Initial food count must be positive, got {INITIAL_FOOD_COUNT}")
    
    if FOOD_LIFETIME <= 0:
        raise ValueError(f"Food lifetime must be positive, got {FOOD_LIFETIME}")
    
    if FOOD_STACK <= 0:
        raise ValueError(f"Food stack size must be positive, got {FOOD_STACK}")
    
    if FOOD_SPAWN_COUNT <= 0:
        raise ValueError(f"Food spawn count must be positive, got {FOOD_SPAWN_COUNT}")
    
    # Energy parameters validation
    if ENERGY_COST <= 0:
        raise ValueError(f"Energy cost must be positive, got {ENERGY_COST}")
    
    if FOOD_GAIN <= 0:
        raise ValueError(f"Food gain must be positive, got {FOOD_GAIN}")
    
    if ENERGY_DECAY_RATE < 0:
        raise ValueError(f"Energy decay rate cannot be negative, got {ENERGY_DECAY_RATE}")
    
    # Time parameters validation
    if DAY_LENGTH <= 0:
        raise ValueError(f"Day length must be positive, got {DAY_LENGTH}")
    
    if MATING_COOLDOWN <= 0:
        raise ValueError(f"Mating cooldown must be positive, got {MATING_COOLDOWN}")
    
    # Trust parameters validation
    if not (0 <= TRUST_INCREMENT <= 1):
        raise ValueError(f"Trust increment must be between 0 and 1, got {TRUST_INCREMENT}")
    
    # House size validation
    if HOUSE_SIZE <= 0:
        raise ValueError(f"House size must be positive, got {HOUSE_SIZE}")
    
    # Cell size validation
    if CELL_SIZE <= 0:
        raise ValueError(f"Cell size must be positive, got {CELL_SIZE}")
    
    # Palette validation
    if not NEW_PALETTE:
        raise ValueError("Color palette cannot be empty")
    
    for color, zone_id in NEW_PALETTE.items():
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"Color must be 3-element tuple/list, got {color}")
        
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError(f"Color values must be integers 0-255, got {color}")
        
        if not isinstance(zone_id, int):
            raise ValueError(f"Zone ID must be integer, got {zone_id} for color {color}")
    
    # Ensure required zones exist
    required_zones = {0, 1, 2, 3}  # walls, grass, blue house, red house
    zone_ids = set(NEW_PALETTE.values())
    missing_zones = required_zones - zone_ids
    if missing_zones:
        raise ValueError(f"Missing required zones: {missing_zones}")
    
    print("✅ Configuration validation passed")

# Auto-validate on import
validate_config()