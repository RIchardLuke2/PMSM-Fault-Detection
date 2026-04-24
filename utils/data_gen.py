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


FAULT_PROFILES = {
    "F0": dict(Ia_amp=1.0, Ib_amp=1.0, phase_offset=0,   VDC_base=560, IDC_base=10, T_base=[40,40,40], VD_base=0.7,  noise=0.02),
    "F1": dict(Ia_amp=0.5, Ib_amp=1.1, phase_offset=0.1,  VDC_base=555, IDC_base=10.5, T_base=[45,40,40], VD_base=0.75, noise=0.04),
    "F2": dict(Ia_amp=1.1, Ib_amp=0.5, phase_offset=-0.1, VDC_base=555, IDC_base=10.5, T_base=[40,45,40], VD_base=0.75, noise=0.04),
    "F3": dict(Ia_amp=0.6, Ib_amp=1.0, phase_offset=0.2,  VDC_base=550, IDC_base=11,   T_base=[40,40,45], VD_base=0.78, noise=0.05),
    "F4": dict(Ia_amp=1.0, Ib_amp=0.6, phase_offset=-0.2, VDC_base=550, IDC_base=11,   T_base=[46,40,40], VD_base=0.78, noise=0.05),
    "F5": dict(Ia_amp=0.4, Ib_amp=1.2, phase_offset=0.3,  VDC_base=545, IDC_base=11.5, T_base=[40,46,40], VD_base=0.80, noise=0.06),
    "F6": dict(Ia_amp=1.2, Ib_amp=0.4, phase_offset=-0.3, VDC_base=545, IDC_base=11.5, T_base=[40,40,46], VD_base=0.80, noise=0.06),
    "F7": dict(Ia_amp=0.3, Ib_amp=0.3, phase_offset=0.5,  VDC_base=535, IDC_base=13,   T_base=[50,48,44], VD_base=0.85, noise=0.08),
    "F8": dict(Ia_amp=2.0, Ib_amp=2.0, phase_offset=0.0,  VDC_base=520, IDC_base=18,   T_base=[60,60,58], VD_base=1.10, noise=0.10),
}

N_SAMPLES_PER_CLASS = 3000
FS = 10000  # Hz
T_PERIOD = 0.02  # 50 Hz fundamental


def _gen_class(fault_id: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = FAULT_PROFILES[fault_id]
    t = np.linspace(0, n * T_PERIOD / FS, n)
    omega = 2 * np.pi * 50

    Ia  = p["Ia_amp"] * np.sin(omega * t + p["phase_offset"]) + rng.normal(0, p["noise"], n)
    Ib  = p["Ib_amp"] * np.sin(omega * t + p["phase_offset"] + 2 * np.pi / 3) + rng.normal(0, p["noise"], n)
    VDC = p["VDC_base"] + 5 * np.sin(0.1 * omega * t) + rng.normal(0, 2, n)
    IDC = p["IDC_base"] + 0.5 * np.abs(np.sin(omega * t)) + rng.normal(0, 0.2, n)
    T1  = p["T_base"][0] + 3 * np.sin(0.05 * omega * t) + rng.normal(0, 0.5, n)
    T2  = p["T_base"][1] + 2 * np.sin(0.05 * omega * t + 1) + rng.normal(0, 0.5, n)
    T3  = p["T_base"][2] + 2.5 * np.sin(0.05 * omega * t + 2) + rng.normal(0, 0.5, n)
    VD  = p["VD_base"] + 0.05 * np.sin(omega * t) + rng.normal(0, 0.01, n)

    # Inject ~2% NaN for missing-data simulation
    for arr in [Ia, Ib, VDC, IDC, T1, T2, T3, VD]:
        idx = rng.choice(n, size=int(0.02 * n), replace=False)
        arr[idx] = np.nan

    return pd.DataFrame({"Ia": Ia, "Ib": Ib, "VDC": VDC, "IDC": IDC,
                          "T1": T1, "T2": T2, "T3": T3, "VD": VD, "FDD": fault_id})


def generate_dataset(out_path: str = "data/pmsm_data.csv", seed: int = 42):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dfs = [_gen_class(fid, N_SAMPLES_PER_CLASS, rng) for fid in FAULT_PROFILES]
    df  = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed)
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Saved {len(df):,} rows → {out_path}")
    print(df["FDD"].value_counts().to_string())
    return df


if __name__ == "__main__":
    generate_dataset()
