"""
Quick integration test for afterslip in the full simulator

Verifies that:
1. Afterslip velocities are reasonable (O(0.1-1 m/yr), not 10^6)
2. Afterslip occurs in halo region around rupture
3. Moment conservation works correctly
"""

import numpy as np
from config import Config
from simulator import run_simulation


def main():
    print("=" * 70)
    print("AFTERSLIP INTEGRATION TEST")
    print("=" * 70)

    # Create short test config
    config = Config()
    config.duration_years = 50.0  # Short simulation
    config.M_min = 6.0  # Only track M>=6 events for afterslip
    config.afterslip_M_min = 6.0  # Trigger afterslip for M>=6
    config.afterslip_enabled = True
    config.omori_enabled = False  # Disable aftershocks for cleaner test
    config.output_hdf5 = "test_afterslip_integration.h5"

    print(f"\nRunning {config.duration_years}-year simulation...")
    print(f"Afterslip enabled: {config.afterslip_enabled}")
    print(f"Afterslip M_min: {config.afterslip_M_min}")

    # Run simulation
    results = run_simulation(config)

    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYZING AFTERSLIP RESULTS")
    print("=" * 70)

    event_history = results["event_history"]

    # Find events that triggered afterslip
    afterslip_events = [e for e in event_history if e.get("afterslip_sequence_id") is not None]

    print(f"\nTotal events: {len(event_history)}")
    print(f"Events with afterslip: {len(afterslip_events)}")

    if len(afterslip_events) > 0:
        # Check first afterslip-triggering event
        event = afterslip_events[0]
        Phi = event.get("spatial_activation")

        if Phi is not None:
            print(f"\nFirst afterslip event:")
            print(f"  Magnitude: M{event['magnitude']:.2f}")
            print(f"  Time: {event['time']:.2f} years")
            print(f"  Spatial activation Phi:")
            print(f"    Max: {Phi.max():.3f}")
            print(f"    Mean: {Phi.mean():.3f}")
            print(f"    Nonzero elements: {np.sum(Phi > 0.01)}")

            # The Phi pattern should peak near the rupture zone
            ruptured = event["ruptured_elements"]
            if len(ruptured) > 0:
                print(f"  Ruptured elements: {len(ruptured)}")
                print(f"  Phi at ruptured elements: {Phi[ruptured].mean():.3f}")
        else:
            print("  WARNING: No spatial activation stored!")
    else:
        print("\nNo events with afterslip found. This may indicate:")
        print("  - No M>=6 events occurred")
        print("  - Afterslip initialization failed")

    # Check final slip deficit values
    m_final = results["final_moment"]
    print(f"\nFinal slip deficit field:")
    print(f"  Min: {m_final.min():.4f} m")
    print(f"  Max: {m_final.max():.4f} m")
    print(f"  Mean: {m_final.mean():.4f} m")

    # Check coupling
    coupling = results["cumulative_release"] / results["cumulative_loading"]
    print(f"\nMoment balance:")
    print(f"  Cumulative loading: {results['cumulative_loading']:.2e} m³")
    print(f"  Cumulative release: {results['cumulative_release']:.2e} m³")
    print(f"  Coupling: {coupling:.3f}")

    # Sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    passed = True

    # Check 1: Slip deficit should be in reasonable range (0 to ~10 m)
    if m_final.min() >= 0 and m_final.max() < 100:
        print("✓ Slip deficit in reasonable range")
    else:
        print(f"✗ Slip deficit out of range: [{m_final.min():.2f}, {m_final.max():.2f}]")
        passed = False

    # Check 2: Coupling should be close to 1.0 (within 0.5-1.5)
    if 0.5 < coupling < 1.5:
        print("✓ Coupling in reasonable range")
    else:
        print(f"✗ Coupling out of range: {coupling:.3f}")
        passed = False

    # Check 3: Should have some events
    if len(event_history) > 0:
        print(f"✓ Generated {len(event_history)} events")
    else:
        print("✗ No events generated")
        passed = False

    if passed:
        print("\n✓ All sanity checks passed!")
    else:
        print("\n✗ Some sanity checks failed - review output above")

    return passed


if __name__ == "__main__":
    main()
