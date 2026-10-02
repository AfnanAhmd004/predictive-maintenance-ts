import numpy as np

from pdm import SENSORS, PCAResidual, ZScore, debounce, false_alarms_per_week, lead_times, pump_run


def test_data_has_degradation_before_failures():
    df = pump_run(seed=1)
    f = np.flatnonzero(df["failure"].values)
    assert len(f) == 2
    assert df["vibration_rms"].iloc[f[0] - 5 : f[0]].mean() > df["vibration_rms"].iloc[:500].mean() + 0.5


def test_debounce():
    a = np.array([1, 1, 0, 1, 1, 1, 1, 0], bool)
    assert debounce(a, 3).tolist() == [0, 0, 0, 0, 0, 1, 1, 0]


def test_lead_time_and_false_alarms():
    alarms = np.zeros(1000, bool)
    alarms[900:] = True
    alarms[100:103] = True
    assert lead_times(alarms, [950]) == [50.0]
    assert false_alarms_per_week(alarms, [950]) > 0


def test_pca_residual_warns_before_failure_with_few_false_alarms():
    df = pump_run(seed=2)
    X = df[SENSORS].values
    det = PCAResidual().fit(X[:1000])
    alarms = debounce(det.alarms(X), 3)
    failures = np.flatnonzero(df["failure"].values)
    lt = lead_times(alarms, failures)
    assert all(v >= 24 for v in lt)  # at least a day of warning
    assert false_alarms_per_week(alarms, failures) < 0.5


def test_threshold_from_healthy_data():
    X = np.random.default_rng(0).standard_normal((2000, 3))
    det = ZScore().fit(X)
    assert det.alarms(X).mean() < 0.01
