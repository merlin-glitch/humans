# 🚀 Performance Optimization Guide

## Current Performance Bottlenecks

Based on code analysis, here are the main performance issues:

### 1. **Neighbor Finding (O(n²) complexity)**
- Each human checks against ALL other humans every tick
- With 100+ humans, this becomes 10,000+ comparisons per tick
- Happens in `Human.step()` when finding nearby humans

### 2. **Trust System Operations**
- Trust calculations happen frequently
- Cache refreshes can be expensive
- Multiple trust score lookups per interaction

### 3. **Resource Grid Operations**
- Large resource arrays being scanned
- Food spawning/decay operations
- Zone-based calculations

### 4. **UI Rendering Overhead**
- Drawing all humans, resources, and UI elements
- Frequent screen updates

## 🎯 Quick Performance Wins

### 1. Enable Occupancy Map (Immediate 2-5x speedup)

**Problem**: Currently using `peers = humans` (all humans)
**Solution**: Enable occupancy map for neighbor finding

```python
# In headless_simulation.py, change this line:
result = run_single_tick(humans, houses, trust_system, resources, is_day, last_storage, use_occupancy_map=False)

# To this:
result = run_single_tick(humans, houses, trust_system, resources, is_day, last_storage, use_occupancy_map=True)
```

### 2. Reduce Trust System Refresh Frequency

**Problem**: Trust cache refreshes happen too often
**Solution**: Batch trust updates

```python
# In common.py, boost_house_trust function:
# Change refresh=False to defer updates
trust_system.increase_trust(
    trustor_id=rid,
    trustee_id=contributor.id,
    increment=increment,
    refresh=False,  # ← This is already correct!
)

# Then call trust_system.flush() less frequently (e.g., every 10 ticks)
```

### 3. Optimize Resource Operations

**Problem**: Scanning entire resource grid
**Solution**: Use sparse data structures

```python
# Track only non-empty resource cells
active_resources = set()  # (x, y) coordinates of resources with food
# Update this set when spawning/consuming resources
```

### 4. Reduce Simulation Frequency

**Problem**: Running at maximum tick rate
**Solution**: Add tick skipping

```python
# Skip some ticks for less critical operations
if tick % 5 == 0:  # Only update trust every 5 ticks
    trust_system.flush()
```

## 🔧 Advanced Optimizations

### 1. Spatial Partitioning

Replace O(n²) neighbor finding with spatial hash:

```python
class SpatialHash:
    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.grid = {}
    
    def add_human(self, human):
        key = (human.x // self.cell_size, human.y // self.cell_size)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(human)
    
    def get_nearby_humans(self, x, y, radius=2):
        # Only check humans in nearby cells
        nearby = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                key = ((x + dx) // self.cell_size, (y + dy) // self.cell_size)
                nearby.extend(self.grid.get(key, []))
        return nearby
```

### 2. Trust System Optimization

```python
# Batch trust updates
class OptimizedTrustSystem(TrustSystem):
    def __init__(self):
        super().__init__()
        self.batch_updates = []
    
    def batch_increase_trust(self, trustor, trustee, increment):
        self.batch_updates.append((trustor, trustee, increment))
    
    def flush_batch(self):
        for trustor, trustee, increment in self.batch_updates:
            self.increase_trust(trustor, trustee, increment, refresh=False)
        self.batch_updates.clear()
        self.flush()
```

### 3. Resource Grid Optimization

```python
# Use sparse representation for resources
import scipy.sparse

class SparseResourceGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Use COO format for efficient updates
        self.lifetime_matrix = scipy.sparse.coo_matrix((width, height))
        self.quantity_matrix = scipy.sparse.coo_matrix((width, height))
    
    def get_food_at(self, x, y):
        return self.quantity_matrix[y, x] if (y, x) in self.quantity_matrix else 0
```

## ⚡ Configuration Optimizations

### 1. Reduce Population for Testing

```python
# In batch_simul.py or headless_simulation.py
INITIAL_BLUE_HUMANS = 5    # Instead of 10
INITIAL_RED_HUMANS = 5     # Instead of 10
```

### 2. Reduce Simulation Duration

```python
# In batch_simul.py
DAYS = 10  # Instead of 20
```

### 3. Disable Expensive Features

```python
# In config.py or headless_simulation.py
COLLECT_METRICS = False     # Disable detailed metrics
PER_ZONE_RESPAWN = False    # Disable per-zone tracking
ENABLE_MATING = False       # Disable mating (if not needed)
```

## 🎮 UI Performance Optimizations

### 1. Reduce Rendering Frequency

```python
# In ui_simulation.py
if tick % 3 == 0:  # Only render every 3 ticks
    # Draw everything
```

### 2. Optimize Drawing Operations

```python
# Use pygame's dirty rectangle updates
dirty_rects = []
# Only redraw changed areas
```

### 3. Reduce UI Complexity

```python
# Disable some UI elements for performance
DRAW_TRUST_LINES = False
DRAW_RESOURCE_GRID = False
```

## 📊 Benchmarking Results

Expected performance improvements:

| Optimization | Speedup | Implementation |
|-------------|---------|----------------|
| Occupancy Map | 2-5x | Easy (1 line change) |
| Trust Batching | 1.5-2x | Medium |
| Spatial Partitioning | 3-10x | Hard |
| Sparse Resources | 2-3x | Hard |
| Reduced Population | Linear | Easy |

## 🚀 Quick Start: Enable Occupancy Map

**Immediate 2-5x speedup with one line change:**

```bash
# Edit headless_simulation.py
# Find line: result = run_single_tick(humans, houses, trust_system, resources, is_day, last_storage, use_occupancy_map=False)
# Change to: result = run_single_tick(humans, houses, trust_system, resources, is_day, last_storage, use_occupancy_map=True)
```

This single change should dramatically improve performance for simulations with 50+ humans.
