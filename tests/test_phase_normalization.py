import numpy as np

from src.pipeline import data_analysis


def test_phase_normalize_cycles_interpolates_cycle_local_positions():
    t_rel = np.arange(0.0, 3.0, 0.1, dtype=float)
    signal = t_rel.copy()
    phase_grid = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=float)

    cycles = data_analysis.phase_normalize_cycles(
        signal=signal,
        t_rel=t_rel,
        task_freq=1.0,
        trial_dur_sec=3.0,
        phase_grid=phase_grid,
        min_samples_per_cycle=2,
        min_cycle_coverage_ratio=0.0,
        use_interpolation=True,
    )

    assert len(cycles) == 3

    # For signal=t and 1 Hz cycles, phase 180 deg equals t0 + 0.5 sec.
    expected_midpoints = np.array([0.5, 1.5, 2.5], dtype=float)
    observed_midpoints = np.array([cycle[2] for cycle in cycles], dtype=float)
    assert np.allclose(observed_midpoints, expected_midpoints, atol=1e-6)


def test_phase_normalize_cycles_closes_wrapped_phase_grid_profiles():
    t_rel = np.arange(0.0, 3.0, 0.1, dtype=float)
    signal = np.array([2.0 * t + 3.0 for t in t_rel], dtype=float)
    phase_grid = np.array([0.0, 120.0, 240.0, 360.0], dtype=float)

    cycles = data_analysis.phase_normalize_cycles(
        signal=signal,
        t_rel=t_rel,
        task_freq=1.0,
        trial_dur_sec=3.0,
        phase_grid=phase_grid,
        min_samples_per_cycle=2,
        min_cycle_coverage_ratio=0.0,
        use_interpolation=True,
    )

    assert len(cycles) == 3
    for cycle in cycles:
        assert cycle[0] == cycle[-1]


def test_phase_normalize_cycles_does_not_invent_first_bin_for_partial_cycle():
    # Cycle 0 has broad coverage, cycle 1 starts late and does not observe phase 0.
    t_rel = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8], dtype=float)
    signal = np.sin(2.0 * np.pi * t_rel)
    phase_grid = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)

    cycles = data_analysis.phase_normalize_cycles(
        signal=signal,
        t_rel=t_rel,
        task_freq=1.0,
        trial_dur_sec=2.0,
        phase_grid=phase_grid,
        min_samples_per_cycle=2,
        min_cycle_coverage_ratio=0.0,
        use_interpolation=True,
    )

    assert len(cycles) == 2
    assert np.isfinite(cycles[0][0])
    assert np.isnan(cycles[1][0])

    avg_profile = np.nanmean(np.stack(cycles, axis=0), axis=0)
    assert np.isclose(avg_profile[0], cycles[0][0], atol=1e-9)


