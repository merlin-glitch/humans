# Usage Examples

This document provides practical examples of how to use the Human Society Simulation for various research scenarios and analysis tasks.

## Basic Usage

### Running a Simple Simulation

```python
from headless_simulation import simulate_headless

# Run a 100-day simulation with default parameters
results = simulate_headless(num_days=100, seed=42, map_path="images/big_map.png", 
                           min_size=1, tol=20)

# Extract population data
days, blue_pop, red_pop, within_trust, between_trust = results[:5]

print(f"Final populations: Blue={blue_pop[-1]}, Red={red_pop[-1]}")
print(f"Final trust levels: Within={within_trust[-1]:.3f}, Between={between_trust[-1]:.3f}")
```

### Interactive Visualization

```python
# Run the interactive UI
python main.py

# Features available in UI:
# - Real-time population visualization
# - Adjustable speed control
# - Pause/resume (Press 'P')
# - Export trust matrix
# - Parameter sliders
```

### Batch Simulation for Statistical Analysis

```python
from batch_simul import run_batch_simulation

# Run 50 simulations with different seeds
results = run_batch_simulation(
    n_runs=50,
    num_days=200,
    seed_base=1000
)

# Results saved to batch_results/all_combined.csv
```

## Research Scenarios

### Tragedy of the Commons Study

```python
from sim_headless import simulate_headless as run_simulation
from config import *

# Configure for resource competition scenario
original_humans = Nbre_HUMANS
original_food = INITIAL_FOOD_COUNT

# Override parameters for scarcity
Nbre_HUMANS = 30
INITIAL_FOOD_COUNT = 40  # 1.33 per human - scarcity
FOOD_LIFETIME = 4000     # High pressure
FOOD_STACK = 6           # Concentrated resources

try:
    results = run_simulation(num_days=150, seed=42, return_zone_series=True)
    
    # Analyze resource exploitation patterns
    zone_data = results[7:12]  # zone series data
    analyze_resource_competition(results)
    
finally:
    # Restore original parameters
    Nbre_HUMANS = original_humans
    INITIAL_FOOD_COUNT = original_food
```

### Trust Network Analysis

```python
from sim_headless import simulate_headless as run_simulation
from common import export_trust_matrix
import networkx as nx
import matplotlib.pyplot as plt

# Run simulation with final state return
results = run_simulation(
    num_days=100, 
    seed=42, 
    return_final_state=True
)

# Extract final state
humans, trust_system = results[-2:]

# Export trust matrix for analysis
export_trust_matrix(trust_system, humans, "trust_analysis.csv")

# Create trust network visualization
def create_trust_network(trust_system, humans, threshold=0.5):
    G = nx.Graph()
    
    # Add nodes
    for human in humans:
        if human.alive:
            G.add_node(human.id, house=human.home.color)
    
    # Add edges for trusted relationships
    for human in humans:
        if not human.alive:
            continue
        for other in humans:
            if other.alive and human.id != other.id:
                trust = trust_system.trust_score(human.id, other.id)
                if trust > threshold:
                    G.add_edge(human.id, other.id, weight=trust)
    
    return G

# Visualize trust network
G = create_trust_network(trust_system, humans, threshold=0.6)
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=1, iterations=50)

# Color nodes by house
blue_nodes = [n for n, d in G.nodes(data=True) if d['house'] == (0, 0, 128)]
red_nodes = [n for n, d in G.nodes(data=True) if d['house'] == (255, 0, 0)]

nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='blue', alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='red', alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("Trust Network (threshold > 0.6)")
plt.axis('off')
plt.show()
```

### Zone Exploitation Analysis

```python
from sim_headless import simulate_headless as run_simulation
import numpy as np
import matplotlib.pyplot as plt

# Run simulation with zone data
results = run_simulation(
    num_days=200, 
    seed=42, 
    return_zone_series=True
)

# Extract zone consumption data
days = results[0]
zone_consumed_daily = results[9]  # per-zone consumption
zone_consumed_by_house = results[10]  # per-house consumption

# Analyze zone preference patterns
def analyze_zone_exploitation(zone_consumed_daily):
    """Analyze which zones are most exploited over time"""
    total_consumption = np.sum(zone_consumed_daily, axis=0)
    zone_preferences = total_consumption / np.sum(total_consumption)
    
    return zone_preferences

# Calculate zone preferences
preferences = analyze_zone_exploitation(zone_consumed_daily)

# Visualize zone exploitation over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Time series of zone consumption
for i, zone_consumption in enumerate(zone_consumed_daily):
    ax1.plot(days, zone_consumption, label=f'Zone {i}', linewidth=2)

ax1.set_xlabel('Day')
ax1.set_ylabel('Daily Consumption')
ax1.set_title('Zone Consumption Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zone preference pie chart
zone_labels = [f'Zone {i}' for i in range(len(preferences))]
ax2.pie(preferences, labels=zone_labels, autopct='%1.1f%%', startangle=90)
ax2.set_title('Overall Zone Exploitation Preferences')

plt.tight_layout()
plt.show()
```

### Parameter Sensitivity Analysis

```python
import pandas as pd
import numpy as np
from sim_headless import simulate_headless as run_simulation

def parameter_sensitivity_analysis():
    """Analyze how different parameters affect simulation outcomes"""
    
    # Parameter ranges to test
    population_sizes = [10, 20, 30, 40]
    food_amounts = [30, 50, 70, 90]
    
    results = []
    
    for pop_size in population_sizes:
        for food_amount in food_amounts:
            print(f"Testing: Population={pop_size}, Food={food_amount}")
            
            # Temporarily modify global parameters
            global Nbre_HUMANS, INITIAL_FOOD_COUNT
            original_pop = Nbre_HUMANS
            original_food = INITIAL_FOOD_COUNT
            
            Nbre_HUMANS = pop_size
            INITIAL_FOOD_COUNT = food_amount
            
            try:
                # Run simulation
                sim_results = run_simulation(
                    num_days=100, 
                    seed=42 + pop_size + food_amount
                )
                
                days, blue_pop, red_pop, within_trust, between_trust = sim_results[:5]
                
                # Record key metrics
                results.append({
                    'population_size': pop_size,
                    'food_amount': food_amount,
                    'final_blue_pop': blue_pop[-1],
                    'final_red_pop': red_pop[-1],
                    'final_within_trust': within_trust[-1],
                    'final_between_trust': between_trust[-1],
                    'survival_rate': (blue_pop[-1] + red_pop[-1]) / (2 * pop_size)
                })
                
            finally:
                # Restore original parameters
                Nbre_HUMANS = original_pop
                INITIAL_FOOD_COUNT = original_food
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Analyze results
    print("\nParameter Sensitivity Results:")
    print(df.groupby(['population_size', 'food_amount']).mean())
    
    return df

# Run sensitivity analysis
sensitivity_results = parameter_sensitivity_analysis()
```

### Evolutionary Dynamics Study

```python
from sim_headless import simulate_headless as run_simulation
import matplotlib.pyplot as plt

def study_evolutionary_dynamics():
    """Study how population dynamics evolve over multiple generations"""
    
    # Run multiple simulations to track evolutionary patterns
    generations = 10
    simulation_length = 200
    
    all_results = []
    
    for gen in range(generations):
        print(f"Running generation {gen + 1}/{generations}")
        
        # Use different seeds for each generation
        results = run_simulation(
            num_days=simulation_length,
            seed=1000 + gen,
            return_zone_series=True
        )
        
        all_results.append(results)
    
    # Analyze evolutionary trends
    def analyze_evolutionary_trends(all_results):
        """Analyze how population characteristics change across generations"""
        
        metrics = {
            'final_population': [],
            'trust_levels': [],
            'resource_efficiency': [],
            'extinction_events': []
        }
        
        for results in all_results:
            days, blue_pop, red_pop, within_trust, between_trust = results[:5]
            
            # Calculate metrics
            final_pop = blue_pop[-1] + red_pop[-1]
            avg_trust = (within_trust[-1] + between_trust[-1]) / 2
            
            # Resource efficiency (consumption vs spawning)
            if len(results) > 7:
                total_spawned = np.sum(results[7])  # total_spawned
                total_consumed = np.sum(results[8])  # total_picked
                efficiency = total_consumed / total_spawned if total_spawned > 0 else 0
            else:
                efficiency = 0
            
            # Check for extinction
            extinction = (blue_pop[-1] == 0) or (red_pop[-1] == 0)
            
            metrics['final_population'].append(final_pop)
            metrics['trust_levels'].append(avg_trust)
            metrics['resource_efficiency'].append(efficiency)
            metrics['extinction_events'].append(extinction)
        
        return metrics
    
    # Analyze trends
    trends = analyze_evolutionary_trends(all_results)
    
    # Visualize evolutionary trends
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Population trends
    axes[0, 0].plot(range(1, generations + 1), trends['final_population'], 'bo-')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Final Population')
    axes[0, 0].set_title('Population Size Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Trust evolution
    axes[0, 1].plot(range(1, generations + 1), trends['trust_levels'], 'ro-')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Average Trust Level')
    axes[0, 1].set_title('Trust Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Resource efficiency
    axes[1, 0].plot(range(1, generations + 1), trends['resource_efficiency'], 'go-')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Resource Efficiency')
    axes[1, 0].set_title('Resource Efficiency Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Extinction events
    axes[1, 1].bar(range(1, generations + 1), trends['extinction_events'], color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Extinction Event (1=Yes, 0=No)')
    axes[1, 1].set_title('Extinction Events')
    
    plt.tight_layout()
    plt.show()
    
    return trends

# Run evolutionary dynamics study
evolution_results = study_evolutionary_dynamics()
```

### Custom Visualization Examples

```python
from batch_plot import *
import pandas as pd

# Load batch simulation results
df = pd.read_csv('batch_results/all_combined.csv')

# Create custom visualizations
def create_custom_analysis(df):
    """Create custom analysis plots"""
    
    # Population survival curves
    plt.figure(figsize=(12, 8))
    
    # Plot survival curves for each run
    for run_id in df['run'].unique()[:10]:  # Show first 10 runs
        run_data = df[df['run'] == run_id]
        plt.plot(run_data['day'], run_data['blue_pop'], 'b-', alpha=0.3)
        plt.plot(run_data['day'], run_data['red_pop'], 'r-', alpha=0.3)
    
    # Plot average survival curves
    avg_data = df.groupby('day').agg({
        'blue_pop': 'mean',
        'red_pop': 'mean'
    }).reset_index()
    
    plt.plot(avg_data['day'], avg_data['blue_pop'], 'b-', linewidth=3, label='Blue Average')
    plt.plot(avg_data['day'], avg_data['red_pop'], 'r-', linewidth=3, label='Red Average')
    
    plt.xlabel('Day')
    plt.ylabel('Population')
    plt.title('Population Survival Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Create custom analysis
create_custom_analysis(df)
```

### Export and Integration Examples

```python
# Export data for external analysis tools
import pandas as pd
import json

def export_for_r_analysis(results, filename_prefix='simulation'):
    """Export simulation results for R analysis"""
    
    # Convert to long format for R
    days, blue_pop, red_pop, within_trust, between_trust = results[:5]
    
    data = []
    for i, day in enumerate(days):
        data.append({
            'day': day,
            'blue_population': blue_pop[i],
            'red_population': red_pop[i],
            'within_trust': within_trust[i],
            'between_trust': between_trust[i]
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(f'{filename_prefix}_data.csv', index=False)
    
    # Save metadata as JSON
    metadata = {
        'simulation_parameters': {
            'population_size': Nbre_HUMANS,
            'initial_food': INITIAL_FOOD_COUNT,
            'food_lifetime': FOOD_LIFETIME,
            'energy_cost': ENERGY_COST
        },
        'simulation_info': {
            'total_days': len(days),
            'final_population': blue_pop[-1] + red_pop[-1],
            'final_trust': (within_trust[-1] + between_trust[-1]) / 2
        }
    }
    
    with open(f'{filename_prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

# Export simulation results
results = run_simulation(num_days=100, seed=42)
export_for_r_analysis(results, 'tragedy_of_commons')
```

## Performance Optimization Examples

### Memory-Efficient Batch Processing

```python
import gc
import psutil
import os

def memory_efficient_batch(n_runs=100, num_days=200):
    """Run batch simulations with memory management"""
    
    results_dir = "batch_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process in chunks to manage memory
    chunk_size = 10
    chunks = [range(i, min(i + chunk_size, n_runs)) for i in range(0, n_runs, chunk_size)]
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
        
        chunk_results = []
        for run_idx in chunk:
            # Run simulation
            results = run_simulation(
                num_days=num_days,
                seed=1000 + run_idx,
                return_zone_series=True,
                progress=False  # Disable progress bar for batch
            )
            chunk_results.append(results)
            
            # Monitor memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if memory_usage > 1000:  # 1GB threshold
                print(f"High memory usage: {memory_usage:.1f}MB")
                gc.collect()  # Force garbage collection
        
        # Save chunk results
        chunk_file = os.path.join(results_dir, f'chunk_{chunk_idx:03d}.csv')
        save_chunk_results(chunk_results, chunk_file)
        
        # Clear memory
        del chunk_results
        gc.collect()

def save_chunk_results(results_list, filename):
    """Save chunk results to CSV"""
    # Implementation depends on your data format
    # This is a placeholder for the actual saving logic
    pass
```

These examples demonstrate the flexibility and power of the Human Society Simulation for various research applications. Each example can be modified and extended to suit specific research questions and analytical needs.
