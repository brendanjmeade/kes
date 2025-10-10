# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a strike-slip fault earthquake simulator built on statistical mechanics principles. The simulator combines:
- **MaxEnt spatial distribution**: Magnitude-dependent nucleation probability based on accumulated moment
- **Exponential temporal rate**: Event rate driven by loading, Omori aftershocks, and moment depletion
- **Moment balance**: All parameters derived from tectonic loading rate and Gutenberg-Richter statistics

## Running the Simulation

```bash
python run_simulation.py
```

This runs the complete pipeline: simulation → save results → generate diagnostic plots.

Results are saved to `results/` directory:
- `simulation_results.pkl`: Pickled results dictionary
- Various PNG diagnostic plots (moment history, magnitude time series, slip maps, etc.)

## Architecture

### Core Simulation Flow

1. **Initialization** (`simulator.py:run_simulation`)
   - Compute temporal rate parameters from moment balance (`temporal_prob.py:compute_exponential_rate_parameters`)
   - Create fault mesh geometry (`geometry.py:create_fault_mesh`)
   - Initialize moment distribution with background + Gaussian pulses (`moment.py:initialize_moment`)

2. **Time-stepping loop** (`simulator.py:run_simulation`)
   - Accumulate moment from tectonic loading (`moment.py:accumulate_moment`)
   - Attempt event generation (`event_generator.py:generate_event`)
   - If event occurs: sample magnitude → compute spatial probability → generate slip → release moment
   - Store snapshots at regular intervals

3. **Visualization** (`visualize.py:plot_all_diagnostics`)
   - Moment history and budget analysis
   - Event rate evolution
   - Magnitude time series
   - Cumulative slip maps

### Key Components

**Moment Framework** (`moment.py`)
- Geometric moment: `m = slip × area` (units: m³)
- Seismic moment: `M₀ = μ × slip × area = μ × m` (units: N·m)
- Moment accumulates continuously from tectonic loading
- Released instantaneously during earthquakes

**Temporal Probability** (`temporal_prob.py`)
- Event rate: `λ(t) = λ₀ × exp(β × r_total)`
- Components: `r_total = r_accumulation + r_omori + r_depletion`
- All parameters (`λ₀`, `β`, `C_a`, `C_r`) computed from moment balance requirement
- Target: average event rate must balance loading rate divided by average event size

**Spatial Probability** (`spatial_prob.py`)
- Nucleation probability: `p(i|M) ∝ m_i^γ(M)`
- Selectivity: `γ(M) = γ_max - (γ_max - γ_min) × exp(-α × (M - M_min))`
- Small events (M~5): γ ≈ 0.5 (weakly selective, SOC-like)
- Large events (M~8): γ ≈ 1.5 (highly selective, nucleate where moment is high)

**Slip Generation** (`slip_generator.py`)
- Rupture area from magnitude-area scaling (Allen & Hayes 2017)
- Select ruptured elements by growing from hypocenter
- Heterogeneous slip: exponential taper from hypocenter with stochastic perturbations
- Scale to match target seismic moment from G-R distribution

### Configuration

All parameters in `config.py:Config` class:
- **Geometry**: Fault dimensions, element size
- **Loading**: Background slip rate, Gaussian moment pulses
- **Spatial**: γ_min, γ_max, α (selectivity parameters)
- **Temporal**: Omori parameters, depletion exponent ψ
- **G-R**: b-value, magnitude range
- **Simulation**: Duration, time step

## Critical Implementation Details

1. **Moment Balance Closure**: The parameters `λ₀`, `β`, `C_a`, `C_r` are NOT free parameters. They are computed in `compute_exponential_rate_parameters` to ensure:
   - Long-term average event rate × average event size = tectonic loading rate
   - Typical accumulated moment produces O(1) dimensionless loading term
   - This ensures the simulator is in quasi-steady state

2. **Time Step Selection**: Currently `time_step_days = 1.0`. Event probability per step is `p = λ(t) × dt_years`, capped at 0.1. If events are too frequent or rare, adjust time step.

3. **Geometric vs Seismic Moment**:
   - Code tracks geometric moment `m` (m³) throughout
   - Convert to seismic moment `M₀ = μ × m` for magnitudes
   - Spatial probability uses geometric moment
   - Depletion term uses seismic moment (cumulative M₀ from past events)

4. **Debug Output**: Several debug print statements exist in `temporal_prob.py` and `event_generator.py` for tracking moment budget and event rates. These can be toggled or removed.

## Modifying the Simulator

- **Change fault geometry**: Edit `Config.fault_length_km`, `Config.fault_depth_km`, `Config.element_size_km`
- **Adjust loading pattern**: Modify `Config.moment_pulses` (can add multiple Gaussian pulses)
- **Tune event rate**: Don't manually adjust `λ₀` or `C_a`. Instead, change physical inputs like `background_slip_rate_mm_yr` or `b_value`, then let `compute_exponential_rate_parameters` recompute
- **Longer simulations**: Increase `Config.duration_years`
- **Add new visualizations**: Add functions to `visualize.py` and call from `plot_all_diagnostics`
