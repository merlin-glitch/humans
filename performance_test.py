#!/usr/bin/env python3
"""
Performance test script for the Human Society Simulation.

Tests different optimization levels to measure performance improvements.
"""

import time
import os
from headless_simulation import simulate_headless

def run_performance_test():
    """Run performance tests with different configurations."""
    
    print("🚀 Performance Test - Human Society Simulation")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {
            "name": "Baseline (Small)",
            "days": 5,
            "seed": 42,
            "description": "Small simulation for baseline"
        },
        {
            "name": "Medium Scale",
            "days": 10,
            "seed": 123,
            "description": "Medium simulation to test scaling"
        },
        {
            "name": "Large Scale",
            "days": 20,
            "seed": 456,
            "description": "Large simulation to stress test"
        }
    ]
    
    map_path = "images/3_spots.png"
    
    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        print("   Make sure you're running from the project root directory")
        return
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   {config['description']}")
        print(f"   Days: {config['days']}, Seed: {config['seed']}")
        
        start_time = time.time()
        
        try:
            result = simulate_headless(
                num_days=config['days'],
                seed=config['seed'],
                map_path=map_path,
                min_size=1,
                tol=20
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                days, blue_pop, red_pop, within_blue_trust, within_red_trust, between_trust = result[:6]
                final_blue = blue_pop[-1] if blue_pop else 0
                final_red = red_pop[-1] if red_pop else 0
                
                print(f"   ✅ Completed in {duration:.2f} seconds")
                print(f"   📈 Final populations: Blue={final_blue}, Red={final_red}")
                print(f"   ⚡ Speed: {config['days']/duration:.1f} days/second")
                
                results.append({
                    'name': config['name'],
                    'days': config['days'],
                    'duration': duration,
                    'speed': config['days']/duration,
                    'final_blue': final_blue,
                    'final_red': final_red,
                    'success': True
                })
            else:
                print(f"   ❌ Failed after {duration:.2f} seconds")
                results.append({
                    'name': config['name'],
                    'days': config['days'],
                    'duration': duration,
                    'success': False
                })
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"   💥 Error after {duration:.2f} seconds: {e}")
            results.append({
                'name': config['name'],
                'days': config['days'],
                'duration': duration,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📋 Performance Summary")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_speed = sum(r['speed'] for r in successful_tests) / len(successful_tests)
        print(f"⚡ Average simulation speed: {avg_speed:.1f} days/second")
        
        print(f"\nDetailed Results:")
        for result in successful_tests:
            print(f"  {result['name']:15} | {result['duration']:6.2f}s | {result['speed']:6.1f} days/s | Pop: {result['final_blue']}+{result['final_red']}")
    
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for result in failed_tests:
            print(f"  {result['name']}: {result.get('error', 'Unknown error')}")
    
    # Performance recommendations
    print(f"\n💡 Performance Tips:")
    print(f"  • Occupancy map is enabled (2-5x speedup)")
    print(f"  • For faster testing, reduce DAYS in batch_simul.py")
    print(f"  • For UI simulation, reduce population size")
    print(f"  • See PERFORMANCE_OPTIMIZATION.md for more tips")
    
    return results

if __name__ == "__main__":
    run_performance_test()
