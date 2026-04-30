"""
Synthetic PMSM inverter dataset generator.

Fault classes F0–F8:
  F0 — Healthy
  F1 — Switch S1 open-circuit
  F2 — Switch S2 open-circuit
  F3 — Switch S3 open-circuit
  F4 — Switch S4 open-circuit
  F5 — Switch S5 open-circuit
  F6 — Switch S6 open-circuit
  F7 — Two-switch open-circuit
  F8 — Short-circuit (DC bus)
"""
import numpy as np
import pandas as pd
from pathlib import Path

"""Synthetic PMSM dataset — used when no real CSV is present."""
import numpy as np
import pandas as pd

FEATURES = ["Ia", "Ib", "VDC", "IDC", "T1", "T2", "T3", "VD"]

_PARAMS = {
    "F0": {"Ia":(1.00,0.05),"Ib":(1.00,0.05),"VDC":(560,5), "IDC":(10.0,0.5),
           "T1":(40,1),"T2":(40,1),"T3":(40,1),"VD":(0.70,0.02)},
    "F1": {"Ia":(0.50,0.10),"Ib":(1.00,0.05),"VDC":(555,8), "IDC":(9.5,0.5),
           "T1":(42,2),"T2":(41,1),"T3":(40,1),"VD":(0.68,0.03)},
    "F2": {"Ia":(1.00,0.05),"Ib":(0.50,0.10),"VDC":(555,8), "IDC":(9.5,0.5),
           "T1":(40,1),"T2":(42,2),"T3":(41,1),"VD":(0.68,0.03)},
    "F3": {"Ia":(0.60,0.10),"Ib":(0.90,0.05),"VDC":(552,10),"IDC":(9.2,0.6),
           "T1":(43,2),"T2":(41,1),"T3":(40,1),"VD":(0.66,0.03)},
    "F4": {"Ia":(0.90,0.05),"Ib":(0.60,0.10),"VDC":(552,10),"IDC":(9.2,0.6),
           "T1":(40,1),"T2":(43,2),"T3":(41,1),"VD":(0.66,0.03)},
    "F5": {"Ia":(0.55,0.10),"Ib":(0.95,0.05),"VDC":(550,12),"IDC":(9.0,0.7),
           "T1":(44,2),"T2":(41,1),"T3":(42,2),"VD":(0.65,0.04)},
    "F6": {"Ia":(0.95,0.05),"Ib":(0.55,0.10),"VDC":(550,12),"IDC":(9.0,0.7),
           "T1":(40,1),"T2":(44,2),"T3":(42,2),"VD":(0.65,0.04)},
    "F7": {"Ia":(0.40,0.15),"Ib":(0.40,0.15),"VDC":(545,15),"IDC":(8.5,0.8),
           "T1":(46,3),"T2":(45,3),"T3":(44,2),"VD":(0.62,0.05)},
    "F8": {"Ia":(1.80,0.20),"Ib":(1.80,0.20),"VDC":(520,20),"IDC":(14.0,1.5),
           "T1":(55,5),"T2":(54,4),"T3":(53,4),"VD":(0.58,0.06)},
}

def generate(n_per_class: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, params in _PARAMS.items():
        for _ in range(n_per_class):
            row = {f: float(rng.normal(params[f][0], params[f][1])) for f in FEATURES}
            row["FDD"] = label
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    df = generate(1000)
    df.to_csv("data/converted_dataset.csv", index=False)
    print(f"Generated {len(df)} rows")
    print(df["FDD"].value_counts().sort_index())