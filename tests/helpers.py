import numpy as np
import pandas as pd


def make_records(n_series=3, n=150, freq="W-MON", anomaly_at=None, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    ds = pd.date_range(start, periods=n, freq=freq)
    rec = []
    for i in range(n_series):
        y = 100 + 0.3 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + rng.normal(0, 2, n)
        if anomaly_at is not None and i == n_series - 1:
            y[anomaly_at] += 60
        rec += [{"unique_id": f"s{i}", "ds": str(d.date()), "y": float(v)} for d, v in zip(ds, y)]
    return rec
