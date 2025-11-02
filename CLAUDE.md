# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a strike-slip fault earthquake simulator built on statistical mechanics principles. The simulator combines:
- **MaxEnt spatial distribution**: Magnitude-dependent nucleation probability based on accumulated moment
- **Adaptive temporal rate**: Event rate driven by moment deficit with continuous adaptive correction to maintain moment balance
- **Afterslip physics**: MaxEnt-based aseismic creep following large events
- **Omori aftershocks**: Spatially-localized aftershock sequences triggered by mainshocks

## Running the Simulation

**Basic workflow** (separate simulation from visualization):
```bash
# Run simulation (saves to HDF5)
python run_simulation.py

# Generate plots from saved results
python plot_results.py
```

**Immediate plotting** (old behavior):
```bash
python run_simulation.py --plot
```

Results are saved to `results/` directory:
- `simulation_results.h5`: HDF5 file with full simulation state (snapshots, events, config)
- Various PNG diagnostic plots (moment history, magnitude time series, slip maps, etc.)

**Iterating on visualization**: After running a long simulation once, you can modify `visualize.py` and regenerate plots in seconds using `python plot_results.py`

## Architecture

### Core Simulation Flow

1. **Initialization** (`simulator.py:run_simulation`)
   - Compute rate coefficient from moment balance (`temporal_prob.py:compute_rate_parameters`)
   - Create fault mesh geometry (`geometry.py:create_fault_mesh`)
   - Initialize moment distribution with spin-up fraction (`moment.py:initialize_moment`)
   - Create HDF5 output file with buffered writer (`hdf5_io.py`)

2. **Time-stepping loop** (`simulator.py:run_simulation`)
   - Update active afterslip sequences (reduces moment deficit aseismically)
   - Accumulate moment from tectonic loading (`moment.py:accumulate_moment`)
   - Compute earthquake rate λ(t) from moment deficit (`temporal_prob.py:earthquake_rate`)
   - Use deterministic accumulator: accumulate fractional events, generate integer events when debt ≥ 1
   - For each event: sample magnitude → compute spatial probability (with aftershock weighting) → generate slip → release moment
   - Trigger afterslip for M ≥ M_min (`afterslip.py:initialize_afterslip_sequence`)
   - Update adaptive rate correction every timestep (`temporal_prob.py:update_rate_correction`)
   - Buffered writes of snapshots to HDF5

3. **Visualization** (`plot_results.py` → `visualize.py:plot_all`)
   - Lazy-load HDF5 results (`hdf5_io.py:load_lazy_results`)
   - Moment history and budget analysis
   - Event rate evolution
   - Magnitude time series and cumulative counts
   - Spatial maps: moment snapshots, cumulative slip, deficit

### Key Components

**Moment Framework** (`moment.py`)
- Geometric moment: `m = slip × area` (units: m³)
- Seismic moment: `M₀ = μ × slip × area = μ × m` (units: N·m)
- Moment accumulates continuously from tectonic loading
- Released instantaneously during earthquakes

**Temporal Probability** (`temporal_prob.py`)
- Event rate: `λ(t) = λ_background + C × correction_factor(t) × moment_deficit(t) + λ_aftershock(t) + λ_perturbation(t)`
- Rate coefficient `C` computed analytically from G-R distribution and moment balance
- Adaptive correction factor continuously adjusts to maintain coupling ≈ 1.0 (applies only to moment-dependent term)
- Background rate: constant external forcing (independent of moment deficit)
- Perturbations: white noise or Ornstein-Uhlenbeck process for stochastic forcing
- Deterministic accumulator prevents event starvation at low rates

**Spatial Probability** (`spatial_prob.py`)
- Nucleation probability: `p(i|M) ∝ m_i^γ(M) × w_aftershock(i)`
- Selectivity: `γ(M) = γ_max - (γ_max - γ_min) × exp(-α × (M - M_min))`
- Small events (M~5): γ ≈ 0.5 (weakly selective, SOC-like)
- Large events (M~8): γ ≈ 1.5 (highly selective, nucleate where moment is high)
- Aftershock weighting: combines Omori temporal decay with spatial activation kernels from recent mainshocks

**Slip Generation** (`slip_generator.py`)
- Rupture area from magnitude-area scaling (Allen & Hayes 2017)
- Select ruptured elements by growing from hypocenter
- Heterogeneous slip: exponential taper from hypocenter with stochastic perturbations
- Scale to match target seismic moment from G-R distribution

**Afterslip** (`afterslip.py`)
- Triggered for events M ≥ M_min (default 6.0)
- Spatial activation kernel: `Φ(x,y)` decays exponentially from rupture zone with anisotropic correlation lengths
- Initial velocity: `v₀ = v_ref × (M/M_ref)^β × Φ × m_residual`
- Temporal decay: `v(t) = v₀ × exp(-decay_rate × t)` where decay rate ensures moment conservation
- Same spatial kernel `Φ` used for aftershock localization
- Afterslip reduces moment deficit aseismically, competing with seismic release

### Configuration

All parameters in `config.py:Config` class:
- **Geometry**: Fault dimensions, element size
- **Loading**: Background slip rate, Gaussian moment pulses
- **Spatial**: γ_min, γ_max, α (selectivity parameters)
- **Adaptive Rate Correction**: Enable/disable, gain, min/max bounds
- **Background Rate**: λ_background (constant rate independent of moment deficit)
- **Random Perturbations**: Type (none/white_noise/ornstein_uhlenbeck), sigma, mean, theta
- **Omori Aftershocks**: Enable/disable, p, c (in years), K_ref, M_ref, alpha, duration
- **Afterslip**: Enable/disable, v_ref, correlation lengths, kernel type, M_min threshold, spatial threshold
- **Moment Initialization**: spinup_fraction (initialize with fraction of mid-cycle moment)
- **G-R**: b-value, magnitude range
- **Simulation**: Duration, time step (in years)
- **Output**: HDF5 compression level, snapshot interval (in years)

## Critical Implementation Details

1. **Adaptive Rate Correction**: The rate coefficient `C` is computed analytically from G-R distribution and moment balance (`compute_rate_parameters`). If `adaptive_correction_enabled = True`, a correction factor is continuously updated every timestep to maintain geometric coupling ≈ 1.0:
   - `correction_factor += gain × (1.0 - coupling) × dt`
   - This ensures long-term moment balance without manual parameter tuning
   - The correction factor is bounded by `correction_factor_min` and `correction_factor_max`

2. **Deterministic Accumulator**: Instead of Poisson sampling, uses fractional event accumulation:
   - `event_debt += λ(t) × dt`
   - Generate `floor(event_debt)` events when debt ≥ 1
   - Prevents event starvation at low rates while maintaining correct long-term statistics

3. **Geometric vs Seismic Moment**:
   - Code tracks geometric moment `m` (m³) throughout for moment deficit and spatial probability
   - Convert to seismic moment `M₀ = μ × m` for magnitudes and physical output
   - Cumulative loading/release tracking uses geometric moment for consistency
   - Event records store both `M0` (seismic) and `geom_moment` (geometric)

4. **HDF5 Storage and Memory Efficiency**:
   - Snapshots written to HDF5 with buffered I/O (`BufferedHDF5Writer`, default 5000 snapshots)
   - Visualization uses lazy loading (`load_lazy_results`) to avoid loading full arrays into memory
   - Compression can be adjusted: `hdf5_compression = 0` (fastest) to `9` (smallest files)
   - Snapshot interval configurable via `snapshot_interval_days`

5. **Afterslip-Aftershock Coupling**:
   - Both use same spatial activation kernel `Φ` from `calculate_spatial_activation_kernel`
   - Afterslip modifies moment deficit before each timestep, affecting both background rate and spatial probability
   - Aftershock spatial weighting stored in event records via `spatial_activation` field
   - Active sequences tracked in parallel lists, updated each timestep

## Modifying the Simulator

- **Change fault geometry**: Edit `Config.fault_length_km`, `Config.fault_depth_km`, `Config.element_size_km`
- **Adjust loading pattern**: Modify `Config.moment_pulses` (can add multiple Gaussian pulses) or change `background_slip_rate_mm_yr`
- **Tune event rate**: Rate is automatically computed from moment balance. Adjust physical inputs (`background_slip_rate_mm_yr`, `b_value`, `M_min`, `M_max`). If adaptive correction is enabled, rate self-corrects to achieve coupling ≈ 1.0
- **Enable/disable afterslip or aftershocks**: Set `Config.afterslip_enabled` or `Config.omori_enabled` to `True/False`
- **Tune afterslip behavior**: Adjust `afterslip_v_ref_m_yr` (initial velocity), `afterslip_M_min` (minimum triggering magnitude), correlation lengths, or spatial threshold
- **Faster simulations**: Increase `time_step_years`, decrease `duration_years`, or use coarser `element_size_km`
- **Save disk space**: Increase `snapshot_interval_years` or enable `hdf5_compression` (1-9)
- **Add new visualizations**: Add functions to `visualize.py` and call from `plot_all` function

## Testing and Development

- **Quick test runs**: Set `duration_years = 10.0` and `M_min = 6.0` for faster iteration
- **Test single afterslip sequence**: Use `test_single_afterslip.py` to visualize afterslip evolution for a single event
- **Parameter studies**: Run with different configs, save to different output files, then compare using `plot_results.py`
- **Inspect HDF5 files**: Use `h5py` or HDF5 viewers to examine raw simulation data without loading into memory
