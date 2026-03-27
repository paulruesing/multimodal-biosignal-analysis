import numpy as np

from src.pipeline import data_analysis, data_integration


def test_build_accuracy_relative_time_axis_respects_offset_window():
    t_rel = data_integration.build_accuracy_relative_time_axis(
        n_samples=8,
        trial_dur_sec=20.0,
        start_offset_sec=5.0,
        endpoint=False,
    )

    assert t_rel.shape == (8,)
    assert t_rel[0] == 5.0
    assert np.all(t_rel >= 5.0)
    assert np.all(t_rel < 20.0)


def test_build_accuracy_relative_time_axis_can_include_trial_end():
    t_rel = data_integration.build_accuracy_relative_time_axis(
        n_samples=8,
        trial_dur_sec=20.0,
        start_offset_sec=5.0,
        endpoint=True,
    )

    assert t_rel[0] == 5.0
    assert t_rel[-1] == 20.0


def test_phase_normalize_cycles_ignores_pre_offset_interpolation_source():
    # Samples before the offset carry a very different value and must not leak
    # into interpolation at phase 0 of the first retained cycle.
    t_rel = np.array([0.0, 1.0, 2.0, 5.2, 5.4, 5.6, 6.2, 6.4, 6.6], dtype=float)
    signal = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    cycles = data_analysis.phase_normalize_cycles(
        signal=signal,
        t_rel=t_rel,
        task_freq=1.0,
        trial_dur_sec=7.0,
        phase_grid=np.array([0.0, 180.0], dtype=float),
        min_samples_per_cycle=1,
        start_offset_sec=5.0,
        min_cycle_coverage_ratio=0.0,
        use_interpolation=True,
    )

    assert len(cycles) == 2
    assert np.isnan(cycles[0][0])
    assert cycles[0][1] == 1.0

