# Optimized configuration for better performance
# Add these lines to config.py for performance tuning

# Performance optimizations
PERFORMANCE_MODE = True
REDUCED_POPULATION = False  # Set to True for faster testing
SKIP_UI_RENDERING = False   # Set to True to disable some UI elements
TRUST_BATCH_SIZE = 10       # Batch trust updates every N ticks

# Population settings (reduce for faster testing)
INITIAL_BLUE_HUMANS = 5     # Default: 10
INITIAL_RED_HUMANS = 5      # Default: 10

# Simulation settings
FAST_MODE = False           # Skip some calculations
REDUCED_METRICS = False     # Disable detailed metrics collection
