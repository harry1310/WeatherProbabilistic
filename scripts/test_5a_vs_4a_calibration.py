"""Fast calibration test: does 5a's narrow CI carry information that 4a's
wider CI doesn't?

Caveat: 5a's test slice (2025-06 to 2025-10) is disjoint from 4a's bundle
test slices (2025-12 to 2026-05). Can't do a strict row-for-row join.
Instead compute each model's calibration on its OWN test set.  If 5a's CI
is structurally narrow (which is the hypothesis), it'll show up irrespective
of the test window.

Four metrics:
  1. Brier per model               -> point-prediction quality
  2. Mean CI80/90 width per model  -> sharpness
  3. Reliability per model         -> calibration of point predictions
  4. CI-width-conditioned Brier    -> does the CI predict where the model is right?

For (4), 5a only since 4a bundle predictions lack CI.

5a source: reports/phase5a_artefacts/predictions/test_predictions_with_ci.parquet
4a source: data/models/precipitation/<station>/v*_phase4a/test_predictions.parquet
4a CI:     data/predictions/precipitation/<station>/model_version=*phase4a*/date=*/predictions.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WB_DATA = Path("C:/Projects/Weather/WeatherBlend/data")
OUT = WB_DATA / "reports" / "5a_vs_4a_calibration_2026-05-25.md"


def brier(p, y):
    p = np.asarray(p, dtype="float64"); y = np.asarray(y, dtype="float64")
    return float(np.mean((p - y) ** 2))


def reliability(p, y, n_bins=10):
    """Returns (bin_centers, bin_mean_pred, bin_mean_obs, bin_n)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        rows.append({
            "bin_lo": bins[b], "bin_hi": bins[b + 1],
            "mean_pred": float(p[m].mean()),
            "mean_obs":  float(y[m].mean()),
            "n":         int(m.sum()),
        })
    return pd.DataFrame(rows)


def main():
    print("== 5a (Bayesian INLA) ==")
    df5 = pd.read_parquet(ROOT / "reports" / "phase5a_artefacts" / "predictions" / "test_predictions_with_ci.parquet")
    df5["valid_time"] = pd.to_datetime(df5["valid_time"])
    n5 = len(df5)
    p5 = df5["p_wet_mean"].to_numpy()
    y5 = df5["observed_wet"].to_numpy()
    b5 = brier(p5, y5)
    w80_5 = float(df5["ci80_width"].mean())
    w90_5 = float(df5["ci90_width"].mean())
    w80_5_med = float(df5["ci80_width"].median())
    w80_5_p95 = float(df5["ci80_width"].quantile(0.95))
    print(f"  rows: {n5:,}")
    print(f"  Brier: {b5:.4f}")
    print(f"  CI80 width: mean={w80_5:.4f}  median={w80_5_med:.4f}  p95={w80_5_p95:.4f}")
    print(f"  CI90 width: mean={w90_5:.4f}")

    print("\n== 4a (BART) - bundle test_predictions across 3 stations ==")
    paths = sorted(glob.glob(
        str(WB_DATA / "models" / "precipitation" / "*" / "v*_phase4a" / "test_predictions.parquet")))
    # Pick the latest bundle per station
    by_station: dict[str, str] = {}
    for p in paths:
        bundle_dir = Path(p).parent.name
        station = Path(p).parent.parent.name
        # Latest bundle wins (sorted ascending so last assignment is latest)
        by_station[station] = p
    print(f"  4a bundles per station (latest): {list(by_station.keys())}")
    frames = []
    for st, path in by_station.items():
        d = pd.read_parquet(path)
        d["valid_time"] = pd.to_datetime(d["valid_time"])
        frames.append(d)
    df4 = pd.concat(frames, ignore_index=True)
    n4 = len(df4)
    p4 = df4["p_wet"].to_numpy()
    y4 = df4["observed_wet"].to_numpy()
    b4 = brier(p4, y4)
    print(f"  rows: {n4:,}")
    print(f"  Brier: {b4:.4f}")
    print("  CI not in bundle; reading from live-prediction tree separately...")

    # 4a CI from live predictions (any available)
    live_paths = sorted(glob.glob(str(
        WB_DATA / "predictions" / "precipitation" / "ea_*" / "model_version=*phase4a*" / "date=*" / "predictions.parquet")))
    print(f"  live 4a prediction parquets found: {len(live_paths)}")
    live_frames = []
    for p in live_paths:
        live_frames.append(pd.read_parquet(p))
    if live_frames:
        df4_live = pd.concat(live_frames, ignore_index=True)
        w80_4 = float(df4_live["Ci80Width"].mean())
        w90_4 = float(df4_live["Ci90Width"].mean())
        w80_4_med = float(df4_live["Ci80Width"].median())
        w80_4_p95 = float(df4_live["Ci80Width"].quantile(0.95))
        print(f"  4a CI80 width: mean={w80_4:.4f}  median={w80_4_med:.4f}  p95={w80_4_p95:.4f}  (n={len(df4_live):,})")
        print(f"  4a CI90 width: mean={w90_4:.4f}")
    else:
        w80_4 = w90_4 = w80_4_med = w80_4_p95 = float("nan")
        print("  no 4a live predictions found locally")

    print("\n== Reliability curve (bin pred, observed freq per bin) ==")
    print("\n5a reliability:")
    rel5 = reliability(p5, y5, n_bins=10)
    print(rel5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n4a reliability:")
    rel4 = reliability(p4, y4, n_bins=10)
    print(rel4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ECE = Expected Calibration Error (weighted mean |mean_pred - mean_obs|)
    def ece(rel):
        tot = rel["n"].sum()
        return float((np.abs(rel["mean_pred"] - rel["mean_obs"]) * rel["n"]).sum() / tot)
    ece5 = ece(rel5); ece4 = ece(rel4)
    print(f"\n  5a ECE: {ece5:.4f}")
    print(f"  4a ECE: {ece4:.4f}")

    print("\n== CI-width-conditioned Brier (5a only - does narrow CI predict better point fit?) ==")
    ci_bins = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0]
    rows = []
    for lo, hi in zip(ci_bins[:-1], ci_bins[1:]):
        m = (df5["ci80_width"] >= lo) & (df5["ci80_width"] < hi)
        if m.sum() < 50:
            continue
        rows.append({
            "ci80_lo": lo, "ci80_hi": hi,
            "n": int(m.sum()),
            "mean_p": float(p5[m].mean()),
            "wet_rate": float(y5[m].mean()),
            "brier": brier(p5[m], y5[m]),
            "abs_err": float(np.abs(p5[m] - y5[m]).mean()),
        })
    cwb = pd.DataFrame(rows)
    print(cwb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Write markdown report
    lines = ["# 5a vs 4a calibration test (2026-05-25, fast version)", "",
             "Caveat: test slices are disjoint (5a: 2025-06-07 to 2025-10-09; "
             "4a: 2025-12-29 to 2026-05-17). Comparison is each model's calibration "
             "on its own test set, NOT a row-for-row join.", "",
             "## Headline", "",
             "| Metric | 5a | 4a | gap |", "|---|---:|---:|---:|",
             f"| Brier (lower better) | {b5:.4f} | {b4:.4f} | {(b5-b4)/b4*100:+.1f}% |",
             f"| ECE (lower better) | {ece5:.4f} | {ece4:.4f} | {(ece5-ece4)/ece4*100:+.1f}% |",
             f"| CI80 width mean | {w80_5:.4f} | {w80_4:.4f} | 5a is {w80_4/w80_5:.1f}x narrower |"
             if not np.isnan(w80_4) else f"| CI80 width mean | {w80_5:.4f} | (n/a) |",
             f"| CI90 width mean | {w90_5:.4f} | {w90_4:.4f} | 5a is {w90_4/w90_5:.1f}x narrower |"
             if not np.isnan(w90_4) else "",
             "", "## Reliability — 5a", "",
             "| bin_lo | bin_hi | mean_pred | mean_obs | n |",
             "|---:|---:|---:|---:|---:|"]
    for _, r in rel5.iterrows():
        lines.append(f"| {r['bin_lo']:.2f} | {r['bin_hi']:.2f} | {r['mean_pred']:.4f} | {r['mean_obs']:.4f} | {int(r['n'])} |")
    lines.append("")
    lines.append("## Reliability — 4a")
    lines.append("")
    lines.append("| bin_lo | bin_hi | mean_pred | mean_obs | n |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in rel4.iterrows():
        lines.append(f"| {r['bin_lo']:.2f} | {r['bin_hi']:.2f} | {r['mean_pred']:.4f} | {r['mean_obs']:.4f} | {int(r['n'])} |")
    lines.append("")
    lines.append("## 5a CI-width-conditioned Brier (does narrow CI = better point fit?)")
    lines.append("")
    lines.append("| CI80 lo | CI80 hi | n | mean_p | wet_rate | Brier | abs_err |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in cwb.iterrows():
        lines.append(f"| {r['ci80_lo']:.3f} | {r['ci80_hi']:.3f} | {int(r['n'])} | {r['mean_p']:.3f} | {r['wet_rate']:.3f} | {r['brier']:.4f} | {r['abs_err']:.4f} |")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
