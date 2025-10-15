# Simulation Workflow

## Overview

The simulator is split into two separate scripts for efficiency:
1. **`run_simulation.py`** - Runs the simulation and saves results
2. **`plot_results.py`** - Generates plots from saved results

This allows you to run expensive simulations once, then iterate on visualization without rerunning the simulation.

---

## Basic Usage

### Step 1: Run Simulation

```bash
python run_simulation.py
```

This will:
- Run the earthquake simulation based on parameters in `config.py`
- Save results to `results/simulation_results.pkl`
- Print instructions for generating plots

**Output:**
```
Results saved to: results/simulation_results.pkl

To generate plots, run:
  python plot_results.py results/simulation_results.pkl
```

### Step 2: Generate Plots

```bash
python plot_results.py
```

Or specify a custom results file:
```bash
python plot_results.py path/to/results.pkl
```

This generates all diagnostic plots in the `results/` directory:
- `magnitude_time_series.png`
- `moment_snapshots.png`
- `cumulative_slip.png`
- `moment_history.png`
- `moment_budget.png`
- `event_rate_evolution.png`

---

## Advanced Usage

### Run simulation with immediate plotting

If you want plots generated right after simulation (old behavior):

```bash
python run_simulation.py --plot
```

### Iterate on visualization

After running a long simulation, you can modify visualization code in `visualize.py` and regenerate plots instantly:

```bash
# Run simulation once (may take hours)
python run_simulation.py

# Iterate on plots (takes seconds)
python plot_results.py
# ... edit visualize.py ...
python plot_results.py
# ... edit visualize.py ...
python plot_results.py
```

### Multiple parameter studies

Save results with different names to compare:

```bash
# Run with b=1.0
python run_simulation.py
mv results/simulation_results.pkl results/b1.0.pkl

# Edit config.py: change b_value to 1.2
# Run with b=1.2
python run_simulation.py
mv results/simulation_results.pkl results/b1.2.pkl

# Compare visualizations
python plot_results.py results/b1.0.pkl
python plot_results.py results/b1.2.pkl
```

---

## Configuration

Edit `config.py` to change simulation parameters:
- Geometry (fault size, element size)
- Loading rates (background + Gaussian pulses)
- Magnitude range (M_min, M_max, b-value)
- Omori parameters (aftershock sequences)
- Adaptive correction (coupling control)
- Duration and time step

All critical parameters are now centralized in `config.py` - no need to edit other files.

---

## File Structure

```
simple_simulator/
├── config.py               # All simulation parameters (EDIT THIS)
├── run_simulation.py       # Main simulation script
├── plot_results.py         # Standalone visualization script
├── simulator.py            # Core simulation loop
├── visualize.py            # Plotting functions
├── moment.py               # Moment accumulation/release
├── temporal_prob.py        # Earthquake rate calculation
├── spatial_prob.py         # Hypocenter selection
├── slip_generator.py       # Slip distribution generation
├── geometry.py             # Fault mesh creation
└── results/
    ├── simulation_results.pkl   # Saved simulation data
    └── *.png                    # Generated plots
```
