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
INITIAL_FOOD_COUNT = 50  # Starting food units (5 per human - generous start)
FOOD_SPAWN_COUNT   = 10  # Food units spawned per respawn event
FOOD_LIFETIME      = 900 # Ticks before food_1 type disappears (45 days)
FOOD_STACK         = 50 # Maximum food units stackable per cell

# ── Energy and reproduction ────────────────────────────────────────────────
ENERGY_COST       = 6.0   # Energy required for mating (prevents rapid reproduction)
DAY_LENGTH        = 200   # Simulation ticks per day (70% day, 30% night)
MATING_COOLDOWN   = 1 * DAY_LENGTH  # 10 days between mating attempts

# ── Standardized agent behavior parameters ────────────────────────────────
ACTION_COST       = 0.001 # Energy cost per action (further reduced from 0.005)
FOOD_GAIN         = 5.0   # Energy gained from consuming food (increased from 2.0)
ENERGY_DECAY_RATE = 0.001 # Energy decay rate per tick (further reduced from 0.002)
TRUST_INCREMENT   = 0.01  # Trust increase when sharing resources (increased from 0.001 for faster trust building)
TRUST_DECAY_AMOUNT = 0.05  # Trust decrease due to forgetting
TRUST_DECAY_INTERVAL = 20  # Days between trust decay events

# ── Feature flags ──────────────────────────────────────────────────────────────
ENABLE_MATING = True      # Enable human reproduction
COLLECT_METRICS = True
PER_ZONE_RESPAWN = True
ENABLE_FOOD_RESPAWN = True  # Enable food respawning (can be toggled in UI)
ENABLE_TRUST = True  # Enable trust system and cooperation (can be toggled in UI)

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

# ── Adaptive Migration Parameters ─────────────────────────────────────────────
# Weight for low storage
HOUSE_RELOC_ALPHA1 = 1.5  # S importance (increased: storage matters more)
# Weight for mean distance
HOUSE_RELOC_ALPHA2 = 1.0  # D importance (increased: travel distance more important)
# Weight for local food
HOUSE_RELOC_ALPHA3 = 2.0  # F importance (increased: food availability critical)
# Movement cost coefficient
HOUSE_RELOC_BETA   = 10.0  # Increased to reduce excessive movement (was 6.0)
# Energy cost to move (relative scale)
HOUSE_RELOC_E_MOVE = 0.3  # Increased movement cost (was 0.1)
# Normalization: house storage (maximum possible value for S)
MAX_HOUSE_STORAGE = 5000.0  # Match house.deposit() cap
# Normalization: maximum expected daily travel for D (cells)
HOUSE_MAX_TRAVEL_PER_DAY = 100.0  # Increased: humans travel farther (was 40.0)
# Normalization: local food radius and max food per cell
HOUSE_LOCAL_RADIUS = 10
MAX_FOOD_PER_CELL = 1000.0  # FIXED: must match FOOD_STACK! (was 5.0)
# Inertia (memory of stability)
HOUSE_INERTIA_STEP = 0.1  # Faster inertia buildup (was 0.04)
HOUSE_INERTIA_MAX = 0.8    # Higher max inertia for more stability (was 0.7)
