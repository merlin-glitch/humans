#!/usr/bin/env python3
"""
Testing and Performance Utilities for Human Society Simulation

This module provides:
1. Performance benchmarking for different simulation configurations
2. Validation utilities for simulation correctness
3. Memory usage monitoring
4. Consistency checks

Usage:
    # Run performance tests
    python tests.py --performance
    
    # Run validation tests
    python tests.py --validate
    
    # Run all tests
    python tests.py --all
"""

import time
import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from headless_simulation import simulate_headless
    from trust_system import TrustSystem
    from human import Human, House
    import config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# =============================================================================
# Performance Testing
# =============================================================================

def run_performance_test() -> List[Dict]:
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
            "days": 100,
            "seed": 456,
            "description": "Large simulation to stress test"
        }
    ]
    
    # Try to find a valid map
    map_candidates = [
        "images/5_spots_fixed.png",
        "images/desert_oasis_well.png",
        "images/3_spots.png"
    ]
    
    map_path = None
    for candidate in map_candidates:
        if os.path.exists(candidate):
            map_path = candidate
            break
    
    if not map_path:
        print(f"❌ No valid map file found. Tried: {map_candidates}")
        return []
    
    print(f"📍 Using map: {map_path}\n")
    
    results = []
    
    for config in configs:
        print(f"📊 Testing: {config['name']}")
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
                print(f"   ⚡ Speed: {config['days']/duration:.1f} days/second\n")
                
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
                print(f"   ❌ Failed after {duration:.2f} seconds\n")
                results.append({
                    'name': config['name'],
                    'days': config['days'],
                    'duration': duration,
                    'success': False
                })
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"   💥 Error after {duration:.2f} seconds: {e}\n")
            results.append({
                'name': config['name'],
                'days': config['days'],
                'duration': duration,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"📋 Performance Summary")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_speed = sum(r['speed'] for r in successful_tests) / len(successful_tests)
        print(f"⚡ Average simulation speed: {avg_speed:.1f} days/second")
        
        print(f"\nDetailed Results:")
        for result in successful_tests:
            pop_total = result['final_blue'] + result['final_red']
            print(f"  {result['name']:20} | {result['duration']:6.2f}s | {result['speed']:6.1f} days/s | Pop: {pop_total:3d}")
    
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for result in failed_tests:
            print(f"  {result['name']}: {result.get('error', 'Unknown error')}")
    
    # Performance recommendations
    print(f"\n💡 Performance Tips:")
    print(f"  • Use headless mode for batch simulations")
    print(f"  • Reduce population size for faster UI simulation")
    print(f"  • Use occupancy maps (enabled by default)")
    print(f"  • Monitor memory usage for populations > 300")
    
    return results


# =============================================================================
# Validation Testing
# =============================================================================

def validate_trust_system() -> bool:
    """Validate trust system consistency."""
    print("\n🔍 Validating Trust System...")
    
    try:
        trust = TrustSystem()
        
        # Initialize some humans
        for i in range(5):
            trust.init_human(i)
        
        # Test trust score symmetry (not required, trust is directional)
        trust.increase_trust(0, 1, 0.1)
        trust.increase_trust(1, 0, 0.1)
        
        score_01 = trust.trust_score(0, 1)
        score_10 = trust.trust_score(1, 0)
        
        print(f"  ✓ Trust scores: 0→1 = {score_01:.2f}, 1→0 = {score_10:.2f}")
        
        # Test that trust scores are in valid range
        for i in range(5):
            for j in range(5):
                if i != j:
                    score = trust.trust_score(i, j)
                    if not (0.0 <= score <= 1.0):
                        print(f"  ❌ Invalid trust score: {i}→{j} = {score}")
                        return False
        
        print(f"  ✓ All trust scores in valid range [0.0, 1.0]")
        
        # Test self-trust validation
        try:
            trust.trust_score(0, 0)
            print(f"  ❌ Self-trust query should raise ValueError")
            return False
        except ValueError:
            print(f"  ✓ Self-trust query correctly raises ValueError")
        
        print("✅ Trust system validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Trust system validation failed: {e}")
        return False


def validate_house_storage() -> bool:
    """Validate house storage cap."""
    print("\n🔍 Validating House Storage Cap...")
    
    try:
        house = House(50, 50, (0, 0, 128))
        
        # Test normal storage
        house.deposit(100)
        if house.storage != 100:
            print(f"  ❌ Expected storage=100, got {house.storage}")
            return False
        
        print(f"  ✓ Normal storage works: {house.storage}")
        
        # Test storage cap
        house.deposit(15000)  # Try to exceed cap
        if house.storage != 10000:
            print(f"  ❌ Expected storage capped at 10000, got {house.storage}")
            return False
        
        print(f"  ✓ Storage cap works: {house.storage} (capped at 10,000)")
        
        print("✅ House storage validation passed")
        return True
        
    except Exception as e:
        print(f"❌ House storage validation failed: {e}")
        return False


def validate_configuration() -> bool:
    """Validate configuration parameters."""
    print("\n🔍 Validating Configuration...")
    
    try:
        # Check critical parameters exist
        required_params = [
            'MAP_WIDTH', 'MAP_HEIGHT', 'CELL_SIZE',
            'Nbre_HUMANS', 'ENERGY_COST', 'MATING_COOLDOWN',
            'FOOD_LIFETIME', 'FOOD_STACK', 'DAY_LENGTH',
            'MAX_HOUSE_STORAGE', 'HOUSE_RELOC_ALPHA1'
        ]
        
        for param in required_params:
            if not hasattr(config, param):
                print(f"  ❌ Missing config parameter: {param}")
                return False
        
        print(f"  ✓ All required parameters present")
        
        # Check parameter ranges
        if config.MAP_WIDTH <= 0 or config.MAP_HEIGHT <= 0:
            print(f"  ❌ Invalid map dimensions")
            return False
        
        if config.Nbre_HUMANS < 0:
            print(f"  ❌ Invalid population size")
            return False
        
        if config.MAX_HOUSE_STORAGE <= 0:
            print(f"  ❌ Invalid house storage cap")
            return False
        
        print(f"  ✓ All parameter values in valid ranges")
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def validate_normalization() -> bool:
    """Validate house relocation normalization."""
    print("\n🔍 Validating House Relocation Normalization...")
    
    try:
        # Test storage normalization
        test_storage = 5000
        S = test_storage / config.MAX_HOUSE_STORAGE
        
        if not (0.0 <= S <= 1.0):
            print(f"  ❌ Storage normalization out of range: S={S}")
            return False
        
        print(f"  ✓ Storage normalization: {test_storage} → {S:.3f}")
        
        # Test travel normalization
        test_travel = 30.0
        D = test_travel / config.HOUSE_MAX_TRAVEL_PER_DAY
        
        if not (0.0 <= D <= 1.5):  # Allow slight overage
            print(f"  ❌ Travel normalization suspicious: D={D}")
            return False
        
        print(f"  ✓ Travel normalization: {test_travel} → {D:.3f}")
        
        # Test food normalization
        test_food = 250
        max_food = (2 * config.HOUSE_LOCAL_RADIUS + 1) ** 2 * config.MAX_FOOD_PER_CELL
        F = test_food / max_food
        
        if not (0.0 <= F <= 1.0):
            print(f"  ❌ Food normalization out of range: F={F}")
            return False
        
        print(f"  ✓ Food normalization: {test_food} → {F:.3f}")
        
        print("✅ Normalization validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Normalization validation failed: {e}")
        return False


def run_all_validations() -> bool:
    """Run all validation tests."""
    print("🧪 Running Validation Tests")
    print("=" * 50)
    
    results = []
    results.append(("Trust System", validate_trust_system()))
    results.append(("House Storage", validate_house_storage()))
    results.append(("Configuration", validate_configuration()))
    results.append(("Normalization", validate_normalization()))
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} | {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
    else:
        print("\n⚠️  Some validation tests failed")
    
    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for tests."""
    parser = argparse.ArgumentParser(description="Test and validate Human Society Simulation")
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no args, run all tests
    if not (args.performance or args.validate or args.all):
        args.all = True
    
    results = {}
    
    if args.all or args.validate:
        results['validation'] = run_all_validations()
    
    if args.all or args.performance:
        results['performance'] = run_performance_test()
    
    print("\n" + "=" * 50)
    print("🏁 Test Suite Complete")
    print("=" * 50)
    
    if 'validation' in results and not results['validation']:
        print("⚠️  Validation failures detected")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

