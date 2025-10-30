# Parameter Estimation and Data Requirements

A comprehensive analysis of all model parameters, categorizing them by observability and identifying data requirements for applying this simulator to real fault systems.

---

## Parameter Categories

Parameters are grouped into four categories based on how they can be constrained from observational data:

1. **Directly Observable** - Can be measured/estimated from existing data
2. **Weakly Constrained** - Observable in principle but highly uncertain
3. **Free Parameters** - Require tuning/calibration against simulation output
4. **Technical Choices** - Numerical/computational settings

---

## 1. DIRECTLY OBSERVABLE FROM DATA

These parameters can be estimated from geophysical observations with reasonable confidence.

### 1.1 Fault Geometry

**Parameters:**
```python
fault_length_km = 200.0        # Along-strike length
fault_depth_km = 25.0          # Down-dip depth (seismogenic)
element_size_km = 1.0          # Discretization
```

**Data sources:**
- **Geologic mapping:** Surface traces, fault scarps
- **Seismicity:** Earthquake hypocenter distributions define seismogenic zone
- **Geodesy:** Locking depth from GPS/InSAR inversions
- **Seismic reflection:** Subsurface fault geometry

**Typical uncertainties:**
- Length: ±10-20% (depends on fault complexity, segmentation)
- Depth: ±3-5 km (transition from brittle to ductile)
- Element size: Modeling choice (1-5 km typical)

**Example for San Andreas (Parkfield):**
- Length: ~30 km (segment scale)
- Depth: 12-15 km (shallow transition)
- Well-constrained from decades of monitoring

---

### 1.2 Slip Rate (Loading)

**Parameters:**
```python
background_slip_rate_mm_yr = 10.0     # Background rate
moment_pulses = [...]                  # Spatial variations
```

**Data sources:**
- **GPS:** Interseismic velocity fields
- **Geologic:** Offset markers (longer timescales)
- **InSAR:** Spatial variations in slip rate
- **Paleoseismology:** Long-term slip rates

**Typical uncertainties:**
- GPS (short-term): ±1-2 mm/yr (regional average)
- Geologic (long-term): ±20-50% (dating uncertainties)
- Spatial variations: Poorly constrained (smooth vs patchy?)

**Critical question:** Should `moment_pulses` represent:
- Real spatial heterogeneity in plate motion?
- Locked vs creeping patches?
- Transient slow slip?

**Example for San Andreas:**
- Central California: ~34 mm/yr (geologic, GPS agree)
- Southern California: ~24 mm/yr
- Spatial variations: Creeping sections vs locked sections

---

### 1.3 Physical Properties

**Parameters:**
```python
shear_modulus_Pa = 3e10  # 30 GPa
```

**Data sources:**
- **Seismic velocities:** V_s determines μ
- **Laboratory experiments:** Rock properties
- **Seismic moment inversions:** Back-calculate from M_W

**Typical uncertainties:**
- Well-constrained: 25-35 GPa for crustal rocks
- Depth-dependent (pressure, temperature)
- ±10% uncertainty reasonable

**Standard values:**
- Upper crust: 30 GPa (this simulator)
- Lower crust: 35-40 GPa
- Sediments: 5-15 GPa (if modeled)

---

### 1.4 Gutenberg-Richter Statistics

**Parameters:**
```python
b_value = 1.0
M_min = 5.0
M_max = 8.0
```

**Data sources:**
- **Seismicity catalogs:** Fit G-R to observed earthquakes
- **Historical records:** Maximum observed magnitude
- **Paleoseismology:** Prehistoric large events
- **Fault dimensions:** Geometric M_max from area

**Typical uncertainties:**
- **b-value:** 0.8-1.2 (regional variations)
  - California: ~1.0 (global average)
  - Some faults: 0.8-0.9 (more large events)
  - Volcanic: 1.2-1.5 (fewer large events)
- **M_min:** Catalog completeness (typically M 2-5)
- **M_max:** Uncertain! Depends on:
  - Maximum fault dimension
  - Multi-segment rupture potential
  - Historical record length

**Critical issue:** b-value may vary spatially/temporally!
- Aftershocks: higher b
- Background: lower b
- Need to decide: use global or local b?

**Example for SAF Parkfield:**
- b ≈ 1.0 for regional seismicity
- M_max = 6.0 (characteristic Parkfield events)
- Or M_max = 7.9 (if connected to larger system)

---

### 1.5 Omori Aftershock Parameters

**Parameters:**
```python
omori_p = 1.0              # Decay exponent
omori_c_days = 1.0         # Time offset
omori_K_ref = 0.1          # Productivity at M_ref
omori_M_ref = 6.0          # Reference magnitude
omori_alpha = 0.8          # Magnitude scaling
omori_duration_years = 10.0
```

**Data sources:**
- **Aftershock catalogs:** Fit Omori law to sequences
- **Reasenberg & Jones (1989):** Standard parameters
- **Regional studies:** Vary by tectonic setting

**Typical values from literature:**
- **p:** 0.9-1.2 (typically ~1.0)
- **c:** 0.001-0.1 days (often ~0.01 days)
- **α:** 0.8-1.0 (Båth's law: α ≈ 0.8)
- **K_ref:** Highly variable (depends on M_ref choice)

**Uncertainties:**
- p: ±0.1 (robust from data)
- c: Factor of 10 (early-time effects, catalog incompleteness)
- α: ±0.1 (well-established empirically)
- K_ref: Factor of 2-3 (normalization choice)

**Regional variations:**
- California: p ≈ 1.0, α ≈ 0.8 (well-studied)
- Japan: p ≈ 1.1 (slightly faster decay)
- Italy: Similar to California

**Example for Landers 1992 (M 7.3):**
- p = 1.07 ± 0.02
- c = 0.05 days
- Hundreds of M>5 aftershocks over years

---

## 2. WEAKLY CONSTRAINED (HIGH UNCERTAINTY)

These parameters are theoretically observable but difficult to constrain in practice.

### 2.1 Spatial Probability (MaxEnt/SOC)

**Parameters:**
```python
gamma_min = 0.5      # Small events (random/SOC)
gamma_max = 1.5      # Large events (systematic)
alpha_spatial = 0.35 # Transition rate
```

**Physical meaning:**
- **γ = 0.5:** Small events nucleate randomly (weak preference for high moment)
- **γ = 1.5:** Large events nucleate where moment is concentrated
- **α:** How quickly selectivity increases with magnitude

**Why weakly constrained:**
- Requires **spatial** catalog of nucleation points
- Need magnitude-dependent patterns
- Confounded by:
  - Catalog location uncertainties
  - Moment distribution evolution
  - Finite observation window

**Possible data constraints:**
- **Aftershock locations:** Do they cluster on high-slip patches? (suggests γ > 1)
- **Repeating earthquakes:** Nucleate at same spots (suggests γ >> 1)
- **Small event scatter:** Random locations? (suggests γ ≈ 0.5)

**Literature estimates:**
- Not well-established! This is a **model assumption**
- Could test by comparing:
  - Synthetic catalogs with different γ(M)
  - Observed spatial clustering statistics

**Recommendation:**
- **Sensitivity analysis:** Vary γ_min, γ_max, α
- Compare to spatial statistics of real seismicity
- May need to calibrate to match observed clustering

---

### 2.2 Slip Distribution Heterogeneity

**Parameters:**
```python
slip_decay_rate = 2.0        # Exponential decay from hypocenter
slip_heterogeneity = 0.3     # Random perturbation (±30%)
```

**Data sources:**
- **Finite-fault inversions:** Slip maps from strong motion/GPS
- **Statistical studies:** Variability across many events

**Why weakly constrained:**
- Only available for large events (M > 6)
- Strong dependence on inversion method
- Different events show different patterns

**Empirical observations:**
- Slip typically **concentrated** near hypocenter (decay with distance)
- Large events: **patchy** slip (multiple asperities)
- Heterogeneity: 30-50% typical (log-normal distribution)

**Scaling relations (Mai & Beroza 2002):**
- Small events: Smoother (less heterogeneity)
- Large events: Rougher (more heterogeneity)
- **This model:** Fixed heterogeneity (could be magnitude-dependent)

**Recommendation:**
- Current values (decay=2.0, heterogeneity=0.3) are **reasonable defaults**
- Could be refined from:
  - Local finite-fault studies
  - Statistical slip distributions
- Sensitivity: Affects individual event details, not long-term statistics

---

### 2.3 Moment Initialization

**Parameters:**
```python
spinup_fraction = 0.25  # Start 1/4 way into earthquake cycle
```

**Why uncertain:**
- Real faults: Unknown initial state!
- Could be:
  - Just after large event (low moment)
  - Mid-cycle (moderate moment)
  - Just before large event (high moment)

**Effects:**
- Early simulation transients
- Time to reach statistical equilibrium
- Should not affect long-term statistics (if duration >> recurrence)

**Recommendation:**
- Run **spin-up period** (first 10-20% of simulation)
- Discard early events for statistics
- Verify results independent of initial condition

---

## 3. FREE PARAMETERS (REQUIRE CALIBRATION)

These parameters cannot be directly observed and must be tuned to match desired behavior.

### 3.1 Adaptive Rate Correction

**Parameters:**
```python
adaptive_correction_enabled = True/False
adaptive_correction_gain = 5.0
correction_factor_min = 0.1
correction_factor_max = 10.0
```

**Purpose:** Drive seismic coupling → 1.0

**Not observable because:**
- Real-world: We don't know if coupling = 1.0!
- Aseismic slip, slow slip events, postseismic relaxation
- Coupling may be < 1.0 or > 1.0 regionally

**Calibration strategy:**

**If adaptive correction ENABLED:**
- `gain`: Tune for convergence speed
  - Too low: Slow to reach coupling=1.0
  - Too high: Oscillations
  - Recommendation: 2.0 - 10.0
- `min/max`: Bounds for numerical stability (0.1 - 10.0 reasonable)

**If adaptive correction DISABLED:**
- Accept natural coupling from C_analytical
- Compare to observed coupling:
  - **California SAF:** ~0.5-0.8 (some aseismic slip)
  - **Cascadia:** ~0.0-0.3 (mostly aseismic)
  - **Japan trench:** ~0.3-0.5

**Recommendation:**
- For **physics exploration:** Disable (see what emerges)
- For **catalog generation:** Enable (ensure moment balance)

---

## 4. TECHNICAL/NUMERICAL CHOICES

These are modeling decisions, not physical parameters.

### 4.1 Discretization

**Parameters:**
```python
element_size_km = 1.0
time_step_days = 1.0
```

**Trade-offs:**

**Element size:**
- Smaller → Better spatial resolution, slower computation
- Larger → Faster, but may miss details
- Must be << rupture dimensions
- Recommendation: 1-2 km for M 5-8 events

**Time step:**
- Smaller → Better temporal resolution
- Larger → Faster, but may miss rate changes
- Must resolve aftershock decay (c ~ 1 day → need dt ~ 1 day)
- Recommendation: 0.1 - 1 day

**Convergence testing:**
- Halve element_size and time_step
- Verify results don't change significantly

---

### 4.2 Simulation Duration

**Parameters:**
```python
duration_years = 1000.0
```

**Requirements:**
- Should be >> recurrence time of largest events
- Need enough events for statistics (100s to 1000s)
- Longer = better statistics but more computation

**Recommendation:**
- For M_max = 8 fault: Recurrence ~ 100-1000 years
- Run 1000-10,000 years
- Or multiple shorter runs with different seeds

---

### 4.3 Output Control

**Parameters:**
```python
snapshot_interval_days = 1.0
random_seed = 42
```

**snapshot_interval:**
- Every timestep: Full resolution (large files)
- Every 10-100 days: Reasonable for visualization
- Trade-off: Detail vs disk space

**random_seed:**
- Fixed: Reproducible (for debugging)
- Random: Different realizations (for ensemble)

---

## 5. SENSITIVITY ANALYSIS PRIORITIES

Which parameters matter most for simulation outcomes?

### HIGH SENSITIVITY (Must Constrain Well)

1. **Slip rate** (background_slip_rate_mm_yr)
   - Directly sets moment accumulation rate
   - Errors propagate linearly to seismicity rate
   - **±20% error → ±20% event rate**

2. **b-value** (b_value)
   - Controls magnitude distribution
   - Affects expected moment per event
   - **b=0.9 vs b=1.1 → 2x difference in large event frequency**

3. **M_max** (M_max)
   - Determines largest possible event
   - If too low: Underpredicts moment release
   - If too high: Rare events dominate

4. **Fault dimensions** (length, depth)
   - Set total system size
   - Affect available moment
   - **±20% area → ±20% total moment rate**

### MODERATE SENSITIVITY

5. **Spatial probability** (gamma_min, gamma_max, alpha_spatial)
   - Affects earthquake clustering patterns
   - Less impact on total seismicity rate
   - **Important for spatial statistics**

6. **Slip heterogeneity** (slip_decay_rate, slip_heterogeneity)
   - Changes individual event details
   - Minimal impact on long-term statistics
   - **Important for realistic catalogs**

7. **Omori parameters** (p, c, K_ref, alpha)
   - Control aftershock productivity
   - Important for short-term forecasting
   - **Less impact on decadal statistics**

### LOW SENSITIVITY (Defaults OK)

8. **Adaptive correction** (if enabled)
   - Just ensures coupling → 1.0
   - Gain affects convergence speed
   - **Outcome independent if converged**

9. **Initialization** (spinup_fraction)
   - Only affects early transients
   - Negligible for long simulations

10. **Numerical** (element_size, time_step)
    - As long as "small enough"
    - Check convergence

---

## 6. DATA REQUIREMENTS FOR REAL FAULT APPLICATION

To apply this simulator to a specific fault system, you would need:

### ESSENTIAL DATA (Cannot proceed without)

1. **Fault geometry:**
   - Length, depth, orientation
   - **Source:** Geologic maps, seismicity

2. **Slip rate:**
   - Long-term average
   - **Source:** GPS, geologic offsets

3. **Seismicity catalog:**
   - For b-value, M_max
   - **Source:** USGS/regional networks

### HIGHLY DESIRABLE (Significantly improves model)

4. **Spatial slip rate variations:**
   - GPS velocity field
   - Creeping vs locked sections
   - **Source:** Dense GPS, InSAR

5. **Aftershock statistics:**
   - Omori parameters for region
   - **Source:** Regional catalogs

6. **Historical large events:**
   - Paleoseismology
   - Constrains M_max, recurrence
   - **Source:** Trenching studies

### NICE TO HAVE (Refinements)

7. **Finite-fault slip models:**
   - Slip heterogeneity statistics
   - **Source:** Kinematic inversions

8. **Spatial nucleation patterns:**
   - For gamma calibration
   - **Source:** Relocated catalogs

9. **Coupling estimates:**
   - Seismic vs geodetic moment
   - **Source:** Long-term comparisons

---

## 7. CALIBRATION WORKFLOW

**Recommended approach for new fault:**

### Phase 1: Observable Parameters
```python
# Set from data
fault_length_km = [from geology]
background_slip_rate_mm_yr = [from GPS]
b_value = [from seismicity catalog]
M_max = [from fault dimensions or paleoseis]
```

### Phase 2: Literature Defaults
```python
# Use standard values
shear_modulus_Pa = 3e10
omori_p = 1.0
omori_alpha = 0.8
slip_heterogeneity = 0.3
```

### Phase 3: Tuning
```python
# Vary to match observations
gamma_min, gamma_max, alpha_spatial = [fit to spatial clustering]
adaptive_correction_gain = [tune for convergence]
```

### Phase 4: Validation
- Run ensemble (100s of realizations)
- Compare to observed:
  - Magnitude-frequency distribution
  - Spatial clustering (nearest-neighbor, correlation)
  - Temporal clustering (Omori aftershocks)
  - Seismic coupling (if known)

### Phase 5: Sensitivity
- Vary uncertain parameters within error bars
- Quantify output uncertainty
- Identify which parameters need better constraints

---

## 8. EXAMPLE: San Andreas Fault (Parkfield)

**Well-constrained:**
```python
# GEOMETRY
fault_length_km = 30.0         # Parkfield segment
fault_depth_km = 12.0          # Shallow transition

# LOADING
background_slip_rate_mm_yr = 34.0  # GPS + geologic (very well known)

# SEISMICITY
b_value = 1.0                  # Regional California
M_min = 2.0                    # Catalog completeness
M_max = 6.0                    # Characteristic Parkfield events

# PHYSICAL
shear_modulus_Pa = 3e10        # Standard crustal
```

**Weakly constrained:**
```python
# SPATIAL PROBABILITY
gamma_min = 0.5                # Assumption (could test)
gamma_max = 1.5                # Assumption
alpha_spatial = 0.35           # Assumption

# SLIP HETEROGENEITY
slip_decay_rate = 2.0          # Reasonable default
slip_heterogeneity = 0.3       # From inversions (varied)
```

**Expected outcomes:**
- M 6 recurrence: ~25 years (observed: ~24 years historically)
- Total seismicity rate: ~1-2 M>2 events per year
- Aftershock productivity: Well-studied for Parkfield events

**Validation:**
- Rich observational dataset (50+ years of monitoring)
- Multiple M 6 events (1922, 1934, 1966, 2004)
- Can directly compare synthetic vs observed catalogs

---

## 9. UNCERTAINTY PROPAGATION

**Key insight:** Parameter uncertainties compound!

**Example error budget:**
```
Slip rate: ±10%
b-value: ±0.1 (factor of 1.26 in small event rate)
M_max: ±0.3 (factor of 2 in largest event recurrence)
Spatial parameters: Unknown (factor of 2-3?)

→ Total uncertainty in seismicity rate: Factor of 2-5
```

**Recommendation:**
- Run **ensemble forecasts** varying parameters within uncertainties
- Report ranges, not single values
- Identify dominant uncertainties (often b-value and M_max)

---

## 10. SUMMARY TABLE

| Parameter | Category | Data Source | Typical Uncertainty | Sensitivity |
|-----------|----------|-------------|---------------------|-------------|
| `fault_length_km` | Observable | Geology, seismicity | ±20% | High |
| `fault_depth_km` | Observable | Seismicity, geodesy | ±3 km | High |
| `background_slip_rate_mm_yr` | Observable | GPS, geology | ±10-20% | **Very High** |
| `shear_modulus_Pa` | Observable | Seismology | ±10% | Moderate |
| `b_value` | Observable | Catalogs | ±0.1-0.2 | **Very High** |
| `M_min` | Observable | Catalog completeness | ±0.5 | Moderate |
| `M_max` | Observable | Geometry, paleoseis | ±0.3-0.5 | **Very High** |
| `omori_p` | Observable | Aftershocks | ±0.1 | Moderate |
| `omori_c` | Observable | Aftershocks | Factor of 10 | Low |
| `omori_alpha` | Observable | Aftershocks | ±0.1 | Moderate |
| `gamma_min` | Weak | Spatial patterns | Unknown | Moderate |
| `gamma_max` | Weak | Spatial patterns | Unknown | Moderate |
| `alpha_spatial` | Weak | Spatial patterns | Unknown | Low |
| `slip_decay_rate` | Weak | Finite-fault | Factor of 2 | Low |
| `slip_heterogeneity` | Weak | Finite-fault | ±50% | Low |
| `spinup_fraction` | Weak | None (initial state) | N/A | Very Low |
| `adaptive_correction_gain` | Free | Tuning | N/A | Low (if enabled) |
| `element_size_km` | Technical | Convergence | N/A | Low (if small) |
| `time_step_days` | Technical | Convergence | N/A | Low (if small) |

---

## 11. RECOMMENDATIONS

**For applying this model to real faults:**

1. **Start simple:** Use well-constrained parameters only
2. **Literature values:** For weakly-constrained parameters
3. **Ensemble approach:** Vary uncertain parameters
4. **Validate:** Compare to observations before forecasting
5. **Sensitivity analysis:** Identify what matters most
6. **Document assumptions:** Especially spatial probability model

**Most critical to get right:**
- Slip rate (drives everything)
- b-value (controls size distribution)
- M_max (affects largest events)

**Can use defaults:**
- Spatial probability (unless studying clustering)
- Slip heterogeneity (unless studying individual events)
- Numerical parameters (after convergence check)

**The simulator is most sensitive to parameters that control moment budget:**
- Loading rate
- Event size distribution
- Moment balance (coupling)

This aligns with the physical insight that **moment balance is fundamental**!
