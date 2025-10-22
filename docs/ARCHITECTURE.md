# System Architecture

## Overview

The Human Society Simulation is built as a modular agent-based system with clear separation between core simulation logic, visualization, and analysis components. The architecture supports both real-time interactive simulation and high-performance batch processing.

## Core Components

### 1. Agent System (`human.py`, `caracteristics.py`)

**Human Class**
- Autonomous agents with energy, position, and social relationships
- Behavior includes movement, foraging, social interaction, and reproduction
- State management for alive/dead status and resource carrying capacity

**House Class**
- Safe zones that serve as resource storage and family centers
- Color-coded to distinguish between competing families (Blue vs Red)
- Storage tracking for resource contributions

**TrustSystem Class**
- Numpy-based trust relationship tracking with O(1) lookups
- Cached trusted/untrusted sets for efficient queries
- Batch update support with dirty tracking for performance

### 2. Resource Management (`new_ressource.py`)

**Resource Grid**
- 3D numpy array: `resources[y, x, 0]` = lifetime, `resources[y, x, 1]` = amount
- Zone-based spawning with adaptive respawn dynamics
- Food decay simulation with configurable lifetime

**Zone Detection**
- Image processing pipeline for map-based zone identification
- Connected component analysis for zone merging
- Color palette mapping for terrain types

**Adaptive Spawning**
- Consumption-based respawn rate adjustment
- Cooldown periods for depleted zones
- Per-zone parameter customization

### 3. Simulation Engine (`ui_simulation.py`, `headless_simulation.py`)

**UI Simulation**
- Interactive pygame-based simulation with real-time visualization
- Menu-driven interface with parameter controls
- Programmatic UI mode for scripting and automation
- Export functionality for trust matrices and population data

**Headless Simulation**
- Optimized for batch processing without UI overhead
- Occupancy map optimization for neighbor finding
- Per-zone resource tracking and analytics
- Preview frame generation for visualization
- Standardized agent behavior across all simulation modes

**Performance Optimizations**
- Precomputed zone mappings for O(1) lookups
- Batch trust updates with deferred refresh
- Efficient dead agent cleanup

**Analytics Integration**
- Real-time metrics collection
- Zone-based consumption tracking
- Export capabilities for external analysis

### 4. Social Mechanics (`common.py`)

**Trust Dynamics**
- Food sharing increases trust between agents
- House contributions boost family trust
- Leadership competition based on trust scores

**Reproductive System**
- Mutual trust requirements for mating
- Energy cost constraints
- Cooldown periods to prevent rapid reproduction

**Competition Resolution**
- Red house leadership selection at dawn
- Trust-based follower assignment
- Resource location memory sharing

## Data Flow

### Simulation Cycle

```
1. Dawn Phase
   ├── Red house leadership competition
   └── Trust-based follower assignment

2. Day/Night Cycle
   ├── Agent movement and foraging
   ├── Resource consumption and sharing
   ├── Trust relationship updates
   └── Population state changes

3. Resource Management
   ├── Food decay processing
   ├── Adaptive spawning calculation
   └── Zone consumption tracking

4. Social Interactions
   ├── Trust score updates
   ├── House contribution bonuses
   └── Mating attempts

5. End of Day
   ├── Metrics collection
   ├── Population cleanup
   └── Cache refresh
```

### Trust System Flow

```
Agent Interaction → Trust Update → Cache Invalidation → Batch Refresh
     ↓                    ↓              ↓                ↓
Food Sharing        Score Change    Dirty Tracking   Trust Lists
```

## Performance Architecture

### Memory Management
- Numpy arrays for efficient numerical operations
- Slotted classes to reduce memory overhead
- Pre-allocated arrays with dynamic growth

### Computational Optimizations
- O(1) zone lookups via precomputed mappings
- Batch trust updates with deferred refresh
- Vectorized operations where possible

### Scalability Considerations
- Configurable population sizes
- Adjustable map dimensions
- Modular component design for easy extension

## Visualization Architecture

### Real-time UI (`ui_simulation.py`, `menu.py`)
- Pygame-based interactive visualization
- Real-time parameter adjustment
- Export functionality for trust matrices

### Batch Analysis (`batch_plot.py`)
- Matplotlib-based statistical visualization
- 20+ chart types for comprehensive analysis
- Configurable smoothing and aggregation

### Spatial Visualization (`heat_map_draw.py`)
- Overlay heatmaps on map images
- Zone-based consumption pattern display
- Color-coded family exploitation patterns

## Configuration System

### Centralized Parameters (`config.py`)
- Map dimensions and visualization settings
- Resource parameters and spawn rates
- Population and energy constraints
- Time cycle definitions

### Runtime Configuration
- Seed-based reproducibility
- Parameter override capabilities
- Environment-specific settings

## Extension Points

### Custom Agent Behaviors
- Inherit from Human class
- Override step() method for custom logic
- Add new interaction types

### Additional Social Mechanics
- Extend TrustSystem for new relationship types
- Add new competition mechanisms
- Implement additional cooperation patterns

### New Resource Types
- Extend resource grid dimensions
- Add resource-specific behaviors
- Implement resource interactions

### Custom Visualizations
- Add new plot types to batch_plot.py
- Create specialized heatmap visualizations
- Implement real-time metric displays

## Dependencies

### Core Dependencies
- **numpy**: Numerical computations and array operations
- **pandas**: Data analysis and CSV export
- **matplotlib**: Statistical visualization
- **opencv-python**: Image processing for zone detection

### UI Dependencies
- **pygame**: Interactive visualization and controls
- **pygame-menu**: Menu system and UI components

### Analysis Dependencies
- **tqdm**: Progress bars for batch operations
- **pillow**: Image processing for map overlays

## Testing and Validation

### Unit Testing
- Trust system consistency checks
- Resource management validation
- Agent behavior verification

### Integration Testing
- End-to-end simulation runs
- Performance benchmarking
- Memory usage monitoring

### Validation Tools
- Trust matrix export for manual verification
- Population tracking consistency checks
- Resource conservation validation

## Future Architecture Considerations

### Distributed Computing
- Multi-process batch simulation support
- Shared memory for large populations
- Cloud-based batch processing

### Machine Learning Integration
- Agent behavior learning
- Adaptive parameter optimization
- Pattern recognition in emergent behaviors

### Database Integration
- Persistent simulation state
- Historical data analysis
- Multi-user collaboration support
