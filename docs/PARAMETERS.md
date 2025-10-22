# Parameter Reference

This document provides detailed explanations of all configurable parameters in the Human Society Simulation, their effects on simulation behavior, and recommended values for different research scenarios.

## Map and World Parameters

### `MAP_WIDTH` and `MAP_HEIGHT`
**Default:** 100 × 60 cells  
**Type:** Integer  
**Effect:** Determines the size of the simulation world in grid cells.

**Impact:**
- Larger maps provide more space for agent movement and resource distribution
- Smaller maps create more intense competition and faster interactions
- Affects computational performance (O(W×H) operations)

**Recommended Values:**
- Small experiments: 50 × 30
- Standard research: 100 × 60
- Large-scale studies: 200 × 120

### `CELL_SIZE`
**Default:** 8 pixels  
**Type:** Integer  
**Effect:** Pixel size of each grid cell for visualization.

**Impact:**
- Only affects visualization quality, not simulation logic
- Larger values create higher-resolution displays
- Smaller values allow more map to fit on screen

## Population Parameters

### `Nbre_HUMANS`
**Default:** 20  
**Type:** Integer  
**Effect:** Initial number of humans in the simulation.

**Impact:**
- Higher populations create more complex social dynamics
- Lower populations may lead to extinction events
- Affects computational performance (O(N²) for interactions)

**Recommended Values:**
- Minimal viable population: 8-10
- Standard research: 20-50
- Large-scale studies: 100-500

### `ENERGY_COST`
**Default:** 8.0  
**Type:** Float  
**Effect:** Energy required for mating attempts.

**Impact:**
- Higher values prevent rapid population growth
- Lower values allow faster reproduction
- Balances population growth with resource constraints

**Recommended Values:**
- Rapid growth scenarios: 4.0-6.0
- Balanced scenarios: 8.0-12.0
- Slow growth scenarios: 15.0-20.0

### `MATING_COOLDOWN`
**Default:** 2000 ticks (10 days)  
**Type:** Integer  
**Effect:** Minimum time between mating attempts for the same pair.

**Impact:**
- Prevents unrealistic rapid reproduction
- Creates natural population growth patterns
- Longer cooldowns lead to more stable populations

## Resource Parameters

### `INITIAL_FOOD_COUNT`
**Default:** 50  
**Type:** Integer  
**Effect:** Starting number of food units in the simulation.

**Impact:**
- Higher values provide initial abundance
- Lower values create immediate competition
- Should scale with population size (recommended: 2-3 per human)

**Recommended Values:**
- Scarcity scenarios: 1-2 per human
- Abundance scenarios: 3-5 per human
- Standard scenarios: 2-3 per human

### `FOOD_LIFETIME`
**Default:** 9000 ticks (45 days)  
**Type:** Integer  
**Effect:** How long food remains available before disappearing.

**Impact:**
- Longer lifetime reduces urgency of resource gathering
- Shorter lifetime creates time pressure and competition
- Affects the balance between exploration and exploitation

**Recommended Values:**
- High pressure scenarios: 3000-5000 ticks (15-25 days)
- Standard scenarios: 7000-12000 ticks (35-60 days)
- Low pressure scenarios: 15000+ ticks (75+ days)

### `FOOD_STACK`
**Default:** 5  
**Type:** Integer  
**Effect:** Maximum food units that can be stored in a single cell.

**Impact:**
- Higher values create resource concentration
- Lower values distribute resources more evenly
- Affects territorial behavior and competition

**Recommended Values:**
- Distributed resources: 1-3 units
- Concentrated resources: 5-10 units
- Mixed scenarios: 3-7 units

### `FOOD_SPAWN_COUNT`
**Default:** 20  
**Type:** Integer  
**Effect:** Number of food units spawned per spawning event.

**Impact:**
- Higher values provide more resources per event
- Lower values create more frequent, smaller events
- Works in conjunction with adaptive spawning system

## Time and Cycle Parameters

### `DAY_LENGTH`
**Default:** 200 ticks  
**Type:** Integer  
**Effect:** Number of simulation ticks per day.

**Impact:**
- Longer days allow more actions per day
- Shorter days create faster time progression
- Affects day/night cycle and agent behavior patterns

**Day/Night Cycle:**
- Day: 0-70% of DAY_LENGTH (140 ticks at default)
- Night: 70-100% of DAY_LENGTH (60 ticks at default)

**Recommended Values:**
- Fast progression: 100-150 ticks
- Standard scenarios: 200-300 ticks
- Detailed analysis: 400-600 ticks

## Trust and Social Parameters

### Trust Thresholds

#### Mating Trust Threshold
**Default:** 0.7  
**Type:** Float (0.0-1.0)  
**Effect:** Minimum mutual trust required for mating.

**Impact:**
- Higher values require stronger relationships for reproduction
- Lower values allow reproduction with weaker social bonds
- Affects population growth patterns and social structure

#### Leadership Trust Threshold
**Default:** 0.55  
**Type:** Float (0.0-1.0)  
**Effect:** Minimum trust required to become a leader.

**Impact:**
- Higher values create more exclusive leadership
- Lower values allow more agents to become leaders
- Affects social hierarchy formation

### Trust Increment Values

#### House Contribution Increment
**Default:** 0.001  
**Type:** Float  
**Effect:** Trust increase when contributing food to house storage.

**Impact:**
- Higher values reward contributions more strongly
- Lower values create more gradual trust building
- Affects cooperation incentives

#### Food Sharing Increment
**Default:** Variable (based on interaction success)  
**Type:** Float  
**Effect:** Trust increase when successfully sharing food.

**Impact:**
- Higher values strengthen social bonds faster
- Lower values require more interactions for trust building
- Affects social network formation

## Zone and Territory Parameters

### Zone Detection Parameters

#### `min_size`
**Default:** 30  
**Type:** Integer  
**Effect:** Minimum zone size to keep as separate during zone merging.

**Impact:**
- Higher values merge more small zones together
- Lower values preserve smaller, isolated zones
- Affects resource distribution patterns

#### `tol` (Tolerance)
**Default:** 30  
**Type:** Integer  
**Effect:** Color tolerance for zone identification from map images.

**Impact:**
- Higher values are more forgiving of color variations
- Lower values require more precise color matching
- Affects zone boundary detection accuracy

### Adaptive Spawning Parameters

#### `I_max` (Maximum Interval)
**Default:** 200  
**Type:** Integer  
**Effect:** Maximum respawn interval when consumption is very low.

**Impact:**
- Higher values create longer periods between spawns
- Lower values maintain more frequent spawning
- Affects resource availability patterns

#### `I_min` (Minimum Interval)
**Default:** 10  
**Type:** Integer  
**Effect:** Minimum respawn interval when consumption is very high.

**Impact:**
- Higher values prevent extremely rapid spawning
- Lower values allow very responsive spawning
- Affects system responsiveness

#### `k` (Responsiveness Factor)
**Default:** 0.6  
**Type:** Float  
**Effect:** How quickly the system responds to consumption changes.

**Impact:**
- Higher values create more responsive spawning
- Lower values create more gradual adjustments
- Affects system stability vs. responsiveness

#### `cooldown_days`
**Default:** 10  
**Type:** Integer  
**Effect:** Days of no spawning when a zone becomes severely depleted.

**Impact:**
- Higher values create longer recovery periods
- Lower values allow faster zone recovery
- Affects resource scarcity patterns

## Agent Behavior Parameters

### Movement Parameters

#### `exploration_factor`
**Default:** 2  
**Type:** Integer  
**Effect:** Randomness factor in agent movement decisions.

**Impact:**
- Higher values create more random, exploratory behavior
- Lower values create more directed, purposeful movement
- Affects territorial behavior and resource discovery

### Energy Parameters

#### `action_cost`
**Default:** 0.05  
**Type:** Float  
**Effect:** Energy cost per agent action.

**Impact:**
- Higher values create energy pressure and strategic decisions
- Lower values allow more actions per day
- Affects agent survival and behavior patterns

#### `food_gain`
**Default:** 1.0  
**Type:** Float  
**Effect:** Energy gained from consuming one food unit.

**Impact:**
- Higher values make food more valuable
- Lower values require more food for survival
- Affects resource competition intensity

#### `decay_rate`
**Default:** 0.001  
**Type:** Float  
**Effect:** Energy decay rate per tick.

**Impact:**
- Higher values create more urgent survival pressure
- Lower values allow longer survival without food
- Affects population dynamics and extinction risk

## Research Scenario Configurations

### Tragedy of the Commons
```python
Nbre_HUMANS = 30
INITIAL_FOOD_COUNT = 60  # 2 per human
FOOD_LIFETIME = 5000     # Moderate pressure
FOOD_STACK = 8           # Concentrated resources
ENERGY_COST = 6.0        # Moderate reproduction cost
```

### Cooperation Evolution
```python
Nbre_HUMANS = 25
INITIAL_FOOD_COUNT = 75  # Abundance
FOOD_LIFETIME = 12000    # Low pressure
ENERGY_COST = 10.0       # High reproduction cost
# High trust thresholds for strong social bonds
```

### Resource Scarcity
```python
Nbre_HUMANS = 40
INITIAL_FOOD_COUNT = 40  # 1 per human
FOOD_LIFETIME = 3000     # High pressure
FOOD_STACK = 3           # Distributed resources
ENERGY_COST = 12.0       # High reproduction cost
```

### Population Dynamics
```python
Nbre_HUMANS = 15
INITIAL_FOOD_COUNT = 45  # 3 per human
FOOD_LIFETIME = 8000     # Moderate pressure
ENERGY_COST = 8.0        # Standard cost
MATING_COOLDOWN = 1500   # Shorter cooldown
```

### Social Hierarchy
```python
Nbre_HUMANS = 35
INITIAL_FOOD_COUNT = 70  # Moderate abundance
FOOD_LIFETIME = 10000    # Moderate pressure
# High leadership trust threshold
# High house contribution rewards
```

## Parameter Interaction Effects

### Population vs. Resources
- **High population + low resources** = Intense competition, potential extinction
- **Low population + high resources** = Cooperation, stable growth
- **Balanced** = Dynamic equilibrium with emergent patterns

### Trust vs. Reproduction
- **High trust thresholds + high energy costs** = Slow, stable population growth
- **Low trust thresholds + low energy costs** = Rapid population growth
- **Mixed** = Complex social dynamics with leadership emergence

### Resource Distribution vs. Territorial Behavior
- **Concentrated resources (high FOOD_STACK)** = Territorial competition
- **Distributed resources (low FOOD_STACK)** = Exploration and sharing
- **Adaptive spawning** = Dynamic territorial patterns

## Monitoring and Adjustment

### Key Metrics to Monitor
1. **Population survival rate** - Should be >80% for stable scenarios
2. **Trust network density** - Indicates social cohesion
3. **Resource consumption efficiency** - Should match spawning rates
4. **Territorial behavior emergence** - Zone preference patterns
5. **Leadership stability** - Trust-based hierarchy formation

### Parameter Adjustment Guidelines
1. **Start with default values** and observe baseline behavior
2. **Adjust one parameter at a time** to understand individual effects
3. **Monitor multiple runs** to account for stochastic variation
4. **Use batch simulations** to validate parameter effects statistically
5. **Document parameter sets** that produce interesting emergent behaviors

### Common Issues and Solutions
- **Population extinction**: Increase resources or reduce energy costs
- **No social structure**: Lower trust thresholds or increase sharing rewards
- **Excessive competition**: Increase resource abundance or distribution
- **Unrealistic behavior**: Adjust time scales or energy parameters
- **Poor performance**: Reduce population size or map dimensions
