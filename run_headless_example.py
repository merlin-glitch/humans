#!/usr/bin/env python3
"""
Example script showing how to run headless simulations.

This demonstrates different ways to run simulations programmatically.
"""

import os
from headless_simulation import simulate_headless

def run_single_simulation():
    """Run a single headless simulation."""
    print("🚀 Running single headless simulation...")
    
    result = simulate_headless(
        num_days=10,           # Run for 10 days
        seed=42,              # Fixed seed for reproducibility
        map_path="images/3_spots.png",  # Map file
        min_size=1,           # Minimum zone size
        tol=20                # Zone tolerance
    )
    
    if result:
        days, blue_pop, red_pop, within_blue_trust, within_red_trust, between_trust = result[:6]
        print(f"✅ Simulation completed!")
        print(f"   Final Blue population: {blue_pop[-1]}")
        print(f"   Final Red population: {red_pop[-1]}")
        print(f"   Final within trust: {within_blue_trust[-1]:.3f}")
    else:
        print("❌ Simulation failed")

def run_multiple_simulations():
    """Run multiple simulations with different seeds."""
    print("🚀 Running multiple headless simulations...")
    
    seeds = [100, 200, 300]
    results = []
    
    for i, seed in enumerate(seeds):
        print(f"   Running simulation {i+1}/{len(seeds)} (seed={seed})...")
        
        result = simulate_headless(
            num_days=5,
            seed=seed,
            map_path="images/3_spots.png",
            min_size=1,
            tol=20
        )
        
        if result:
            results.append(result)
            print(f"   ✅ Completed (final pop: {result[1][-1]} blue, {result[2][-1]} red)")
        else:
            print(f"   ❌ Failed")
    
    print(f"\n✅ Completed {len(results)}/{len(seeds)} simulations")

if __name__ == "__main__":
    print("🎯 Headless Simulation Examples")
    print("=" * 40)
    
    # Check if map exists
    if not os.path.exists("images/3_spots.png"):
        print("❌ Map file not found: images/3_spots.png")
        print("   Make sure you're running from the project root directory")
        exit(1)
    
    # Run examples
    run_single_simulation()
    print()
    run_multiple_simulations()
    
    print("\n🎉 All examples completed!")
