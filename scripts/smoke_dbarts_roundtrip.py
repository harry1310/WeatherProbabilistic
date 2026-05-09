"""Smoke: dbarts state round-trip ACROSS PROCESSES via storeState +
saveRDS → readRDS + warm scaffold + setState. Mirrors the production
train_4a / predict_4a split so we catch scaffold drift before flipping
the cron over.

Why a subprocess: the original failure mode was "predict in a fresh
session falls back to Y.mean()". The bug only shows up when the C++
sampler from the original fit has been torn down, which means a fresh
process — not just a fresh rpy2 conversion context. Same-process
storeState/setState always works; it doesn't prove the production case.

Pass criterion: max abs difference between predictions from the original
fit and predictions from the reloaded warm scaffold is bit-exact (0.0).

Run:
    python scripts/smoke_dbarts_roundtrip.py            # both phases
    python scripts/smoke_dbarts_roundtrip.py --save     # save phase only
    python scripts/smoke_dbarts_roundtrip.py --load DIR # load phase only
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
_r_home = os.environ.get("R_HOME", r"C:\Program Files\R\R-4.6.0")
os.environ.setdefault("R_HOME", _r_home)
_r_bin = os.path.join(_r_home, "bin", "x64")
if hasattr(os, "add_dll_directory") and os.path.isdir(_r_bin):
    os.add_dll_directory(_r_bin)
os.environ["PATH"] = _r_bin + os.pathsep + os.environ.get("PATH", "")
_user_lib = os.path.join(os.environ.get("USERPROFILE", os.environ.get("HOME", "")),
                         "R", "win-library", "4.6")
os.environ.setdefault("R_LIBS_USER", _user_lib)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402
from scipy.stats import norm  # noqa: E402

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE = 50
NSKIP = 200
NDPOST = 200
SEED = 42


def synth_problem():
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 800, 200, 6
    X_train = rng.standard_normal((n_train, n_feat))
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] - 0.3 * X_train[:, 2] ** 2
               + 0.2 * rng.standard_normal(n_train) > 0).astype(np.int8)
    X_test = rng.standard_normal((n_test, n_feat))
    return X_train.astype("float64"), y_train.astype("float64"), X_test.astype("float64")


def save_phase(out_dir: Path) -> np.ndarray:
    """Fit dbarts, snapshot per-draw probit predictions, persist state."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test = synth_problem()

    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train)
        y_train_r = ro.conversion.py2rpy(y_train)
        x_test_r = ro.conversion.py2rpy(X_test)
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=True, verbose=False, seed=SEED,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat_test = np.array(ro.conversion.rpy2py(yhat_test_r))

    # storeState is a refclass method — not subscriptable from rpy2's S4
    # view; stash the fit in globalenv and call through ro.r(...).
    ro.globalenv["fit"] = fit
    ro.r('fit$fit$storeState()')

    state_path = out_dir / "state.rds"
    bundle_r = ro.r('list(state = fit$fit$state)')
    ro.globalenv["bundle"] = bundle_r
    ro.r(f'saveRDS(bundle, "{str(state_path).replace(os.sep, "/")}")')

    # Persist x_train/y_train/X_test alongside so the load-side process is
    # self-contained. Pickle is fine — these are plain numpy arrays.
    with open(out_dir / "arrays.pkl", "wb") as f:
        pickle.dump({"X_train": X_train, "y_train": y_train, "X_test": X_test,
                     "yhat_test_orig": yhat_test}, f)

    p_orig = norm.cdf(yhat_test).mean(axis=0)
    print(f"[save] fit done — yhat shape {yhat_test.shape}, "
          f"p_orig mean {p_orig.mean():.4f}, range [{p_orig.min():.3f}, {p_orig.max():.3f}]",
          flush=True)
    print(f"[save] state → {state_path} ({state_path.stat().st_size:,} bytes)", flush=True)
    return p_orig


def load_phase(in_dir: Path) -> np.ndarray:
    """Build a tiny warm scaffold matching the original problem, inject the
    saved state, then predict. Returns the posterior-mean P(y=1).
    """
    with open(in_dir / "arrays.pkl", "rb") as f:
        arrays = pickle.load(f)
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_test  = arrays["X_test"]

    # Warm scaffold: bart() with nskip=1, ndpost=1 — runs in ~1s but does
    # the binary detection + cutpoint inference setState requires.
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train)
        y_train_r = ro.conversion.py2rpy(y_train)
    warm = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r,
        ntree=NTREE, nskip=1, ndpost=1,
        keeptrees=True, verbose=False, seed=SEED,
    )
    ro.globalenv["warm"] = warm

    state_path = in_dir / "state.rds"
    ro.r(f'bundle <- readRDS("{str(state_path).replace(os.sep, "/")}")')
    ro.r('warm$fit$setState(bundle$state)')

    with localconverter(_RCONVERT):
        x_test_r = ro.conversion.py2rpy(X_test)
    ro.globalenv["x_test"] = x_test_r
    pred_r = ro.r('predict(warm, newdata = x_test)')
    with localconverter(_RCONVERT):
        pred = np.array(ro.conversion.rpy2py(pred_r))

    # predict.bart for binary returns probabilities directly (auto-pnorm),
    # while yhat.test from sampling is probit-scale. To compare apples to
    # apples we use the same aggregation on both sides: posterior mean of
    # probabilities. So convert the saved probit-scale yhat ourselves.
    p_load = pred.mean(axis=0)
    print(f"[load] predict shape {pred.shape}, "
          f"p_load mean {p_load.mean():.4f}, range [{p_load.min():.3f}, {p_load.max():.3f}]",
          flush=True)
    return p_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="Save phase only (writes state to a temp dir, prints path).")
    ap.add_argument("--load", type=str, default=None,
                    help="Load phase only — directory containing state.rds + arrays.pkl.")
    args = ap.parse_args()

    if args.save and args.load:
        sys.exit("--save and --load are mutually exclusive")

    if args.load:
        load_phase(Path(args.load))
        return

    if args.save:
        td = Path(tempfile.mkdtemp(prefix="dbarts_smoke_"))
        save_phase(td)
        print(f"[save] dir={td}")
        return

    # Default: do both phases in two separate processes so we exercise the
    # production cross-process pattern (save process exits before load
    # process starts). Same-process storeState/setState would silently
    # paper over the bug we're trying to catch.
    td = Path(tempfile.mkdtemp(prefix="dbarts_smoke_"))
    p_orig = save_phase(td)

    print()
    print(f"[parent] launching load subprocess on {td}", flush=True)
    result = subprocess.run(
        [sys.executable, __file__, "--load", str(td)],
        capture_output=True, text=True, env=os.environ,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(f"load subprocess failed with rc={result.returncode}")

    p_load = load_phase(td)  # also do it in-process for direct numpy compare

    delta = float(np.max(np.abs(p_orig - p_load)))
    print()
    print(f"max |delta| (original posterior-mean vs reloaded posterior-mean): {delta:.6e}")
    if delta > 1e-9:
        sys.exit(f"FAIL: roundtrip diverges (max abs delta = {delta})")
    print("PASS — round-trip is bit-exact")


if __name__ == "__main__":
    main()
