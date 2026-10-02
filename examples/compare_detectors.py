"""Train on the first 1000 healthy hours; compare warning lead time and false alarms on the rest."""
import numpy as np

from pdm import SENSORS, IForest, PCAResidual, WindowAutoencoder, ZScore, debounce, false_alarms_per_week, lead_times, pump_run

df = pump_run(seed=0)
X = df[SENSORS].values
train = X[:1000]  # healthy period only
failures = list(np.flatnonzero(df["failure"].values))

print(f"{'detector':<20}{'lead time per failure (h)':>28}{'false alarms / week':>22}")
for name, det in [("z-score", ZScore()), ("PCA residual (SPE)", PCAResidual()),
                  ("isolation forest", IForest()), ("window autoencoder", WindowAutoencoder())]:
    det.fit(train)
    alarms = debounce(det.alarms(X), k=3)
    lt = lead_times(alarms, failures)
    print(f"{name:<20}{str([None if np.isnan(v) else int(v) for v in lt]):>28}{false_alarms_per_week(alarms, failures):>22.2f}")
