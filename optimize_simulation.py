#!/usr/bin/env python3
"""
Quick optimization script for Human Society Simulation.

Provides easy ways to configure simulation for different performance needs.
"""

import os
import sys

def create_optimized_config():
    """Create an optimized configuration file for better performance."""
    
    config_content = '''# Optimized configuration for better performance
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
'''
    
    with open("config_performance.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created config_performance.py with optimization settings")

def show_optimization_menu():
    """Show interactive optimization menu."""
    
    print("🚀 Human Society Simulation - Performance Optimizer")
    print("=" * 50)
    print()
    print("Choose optimization level:")
    print()
    print("1. 🏃‍♂️ Speed Mode (Fastest)")
    print("   • Reduced population (5+5 humans)")
    print("   • Shorter simulations (10 days)")
    print("   • Minimal metrics")
    print()
    print("2. ⚖️ Balanced Mode (Recommended)")
    print("   • Normal population (10+10 humans)")
    print("   • Medium simulations (20 days)")
    print("   • Standard metrics")
    print()
    print("3. 🔬 Research Mode (Slowest)")
    print("   • Full population (20+20 humans)")
    print("   • Long simulations (100+ days)")
    print("   • Detailed metrics")
    print()
    print("4. 🛠️ Custom Settings")
    print("   • Configure individual parameters")
    print()
    print("5. 📊 Run Performance Test")
    print("   • Benchmark current settings")
    print()
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "1":
                apply_speed_mode()
                break
            elif choice == "2":
                apply_balanced_mode()
                break
            elif choice == "3":
                apply_research_mode()
                break
            elif choice == "4":
                apply_custom_settings()
                break
            elif choice == "5":
                run_performance_test()
                break
            elif choice == "0":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def apply_speed_mode():
    """Apply speed mode optimizations."""
    print("\n🏃‍♂️ Applying Speed Mode optimizations...")
    
    # Modify batch_simul.py
    modify_batch_config(DAYS=10, N_RUNS=2)
    
    # Create optimized config
    config = {
        'INITIAL_BLUE_HUMANS': 5,
        'INITIAL_RED_HUMANS': 5,
        'COLLECT_METRICS': False,
        'PER_ZONE_RESPAWN': False,
        'FAST_MODE': True
    }
    
    apply_config_changes(config)
    print("✅ Speed mode applied!")
    print("   • Simulations will run ~3x faster")
    print("   • Reduced population for quick testing")
    print("   • Minimal data collection")

def apply_balanced_mode():
    """Apply balanced mode optimizations."""
    print("\n⚖️ Applying Balanced Mode optimizations...")
    
    modify_batch_config(DAYS=20, N_RUNS=5)
    
    config = {
        'INITIAL_BLUE_HUMANS': 10,
        'INITIAL_RED_HUMANS': 10,
        'COLLECT_METRICS': True,
        'PER_ZONE_RESPAWN': True,
        'FAST_MODE': False
    }
    
    apply_config_changes(config)
    print("✅ Balanced mode applied!")
    print("   • Good balance of speed and detail")
    print("   • Recommended for most users")

def apply_research_mode():
    """Apply research mode optimizations."""
    print("\n🔬 Applying Research Mode optimizations...")
    
    modify_batch_config(DAYS=100, N_RUNS=10)
    
    config = {
        'INITIAL_BLUE_HUMANS': 20,
        'INITIAL_RED_HUMANS': 20,
        'COLLECT_METRICS': True,
        'PER_ZONE_RESPAWN': True,
        'FAST_MODE': False,
        'DETAILED_LOGGING': True
    }
    
    apply_config_changes(config)
    print("✅ Research mode applied!")
    print("   • Maximum detail and accuracy")
    print("   • Suitable for research analysis")
    print("   • May be slower but more comprehensive")

def apply_custom_settings():
    """Apply custom settings."""
    print("\n🛠️ Custom Settings")
    print("=" * 30)
    
    try:
        days = int(input("Simulation days (default 20): ") or "20")
        runs = int(input("Number of runs (default 5): ") or "5")
        blue_pop = int(input("Initial blue population (default 10): ") or "10")
        red_pop = int(input("Initial red population (default 10): ") or "10")
        
        modify_batch_config(DAYS=days, N_RUNS=runs)
        
        config = {
            'INITIAL_BLUE_HUMANS': blue_pop,
            'INITIAL_RED_HUMANS': red_pop,
            'COLLECT_METRICS': True,
            'PER_ZONE_RESPAWN': True
        }
        
        apply_config_changes(config)
        print(f"✅ Custom settings applied!")
        print(f"   • {days} days per simulation")
        print(f"   • {runs} simulation runs")
        print(f"   • {blue_pop}+{red_pop} initial population")
        
    except ValueError:
        print("❌ Invalid input. Using default values.")

def modify_batch_config(DAYS=None, N_RUNS=None):
    """Modify batch_simul.py configuration."""
    try:
        with open("batch_simul.py", "r") as f:
            content = f.read()
        
        if DAYS is not None:
            content = content.replace("DAYS   = 20", f"DAYS   = {DAYS}")
        
        if N_RUNS is not None:
            content = content.replace("N_RUNS = 2", f"N_RUNS = {N_RUNS}")
        
        with open("batch_simul.py", "w") as f:
            f.write(content)
            
    except FileNotFoundError:
        print("⚠️ batch_simul.py not found, skipping batch configuration")

def apply_config_changes(config):
    """Apply configuration changes to config.py."""
    try:
        with open("config.py", "r") as f:
            content = f.read()
        
        for key, value in config.items():
            # Add or modify configuration values
            if f"{key} =" in content:
                # Replace existing value
                import re
                pattern = rf"{key}\s*=\s*.*"
                replacement = f"{key} = {value}"
                content = re.sub(pattern, replacement, content)
            else:
                # Add new value at the end
                content += f"\n# Performance optimization\n{key} = {value}\n"
        
        with open("config.py", "w") as f:
            f.write(content)
            
    except FileNotFoundError:
        print("⚠️ config.py not found, creating backup configuration")
        with open("config_backup.py", "w") as f:
            for key, value in config.items():
                f.write(f"{key} = {value}\n")

def run_performance_test():
    """Run the performance test."""
    print("\n📊 Running Performance Test...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "performance_test.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"❌ Error running performance test: {e}")

if __name__ == "__main__":
    create_optimized_config()
    show_optimization_menu()
