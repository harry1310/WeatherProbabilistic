"""Smoke: dbarts via rpy2 — round-trip a tiny BART problem against the
pymc-bart baseline so we can confirm the R path is wired correctly,
then time both on the same problem to see how big the speedup is.

Goal: prove dbarts gives sensible predictions before considering it
for the real Phase 6 setup. Not a calibration test — just shape +
finite + rough numerical agreement.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
# rpy2 needs R_HOME set on Windows. Also: R's bin\x64 directory must be on
# the Windows DLL search path BEFORE any rpy2 import — Python 3.8+ requires
# os.add_dll_directory() (mutating PATH alone is not enough). Without this,
# stats.dll fails LoadLibrary because its dependent R.dll / Rblas.dll can't
# be located.
_r_home = r"C:\Program Files\R\R-4.6.0"
os.environ.setdefault("R_HOME", _r_home)
_r_bin = os.path.join(_r_home, "bin", "x64")
if hasattr(os, "add_dll_directory") and os.path.isdir(_r_bin):
    os.add_dll_directory(_r_bin)
os.environ["PATH"] = _r_bin + os.pathsep + os.environ.get("PATH", "")

# Per-user R library — we installed dbarts there to avoid Program Files perms.
_user_lib = os.path.join(os.environ.get("USERPROFILE", ""), "R", "win-library", "4.6")
os.environ.setdefault("R_LIBS_USER", _user_lib)

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

# Modern rpy2 (>= 3.5) uses scoped conversion contexts instead of the
# old activate()/deactivate() globals. Build one we can re-enter inside
# every R-call site.
_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter

# Make sure the per-user lib is on R's libPaths so importr can find dbarts.
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")


def fit_dbarts_binary(X_train, y_train, X_test, *, n_trees=50, n_burn=200,
                      n_samples=1000, seed=42):
    """Probit-link binary BART via dbarts::bart. Returns posterior-mean
    P(y=1) per row of X_test. dbarts auto-selects probit when y is 0/1.
    """
    # Convert numpy → R only for the call's INPUTS. The returned `fit`
    # object should stay as a raw R list so we can use .rx2() to extract
    # `yhat.test` by name. Under localconverter the result gets eagerly
    # converted to a Python NamedList that lacks .rx2().
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train)
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_test_r = ro.conversion.py2rpy(X_test)
    fit = dbarts.bart(
        x_train=x_train_r,
        y_train=y_train_r,
        x_test=x_test_r,
        ntree=n_trees,
        nskip=n_burn,
        ndpost=n_samples,
        keeptrees=True,
        verbose=False,
        seed=seed,
    )
    # bart$yhat.test is (n_samples, n_test) on the latent (probit) scale
    # for binary y.
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat_test = np.array(ro.conversion.rpy2py(yhat_test_r))
    # In the probit binary case dbarts stores the LATENT mean per draw;
    # apply pnorm to get probabilities. (For continuous y this would be
    # the response mean directly. For binary, we hit pnorm.)
    from scipy.stats import norm
    p_post = norm.cdf(yhat_test)  # (draws, n_test)
    return p_post.mean(axis=0)


def main() -> None:
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 1500, 500, 6
    X_train = rng.standard_normal((n_train, n_feat))
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] - 0.3 * X_train[:, 2] ** 2
               + 0.2 * rng.standard_normal(n_train) > 0).astype(np.int8)
    X_test = rng.standard_normal((n_test, n_feat))
    y_test = (X_test[:, 0] + 0.5 * X_test[:, 1] - 0.3 * X_test[:, 2] ** 2
              + 0.2 * rng.standard_normal(n_test) > 0).astype(np.int8)

    print(f"smoke: {n_train} train, {n_test} test, {n_feat} features")
    print()
    print("[dbarts] fitting…", flush=True)
    t0 = time.time()
    p_dbarts = fit_dbarts_binary(X_train, y_train, X_test,
                                 n_trees=50, n_burn=200, n_samples=1000, seed=42)
    t_dbarts = time.time() - t0
    brier_db = float(np.mean((p_dbarts - y_test) ** 2))
    print(f"[dbarts] done in {t_dbarts:.1f}s")
    print(f"[dbarts] p_test mean={p_dbarts.mean():.3f}, "
          f"range=[{p_dbarts.min():.3f}, {p_dbarts.max():.3f}], "
          f"Brier={brier_db:.4f}")
    assert np.all(np.isfinite(p_dbarts)), "dbarts produced non-finite probs"
    assert p_dbarts.min() >= 0 and p_dbarts.max() <= 1, "dbarts probs out of [0,1]"

    # pymc-bart comparison (smaller config for speed)
    print()
    print("[pymc-bart] fitting (200 burn / 200 draws / 1 chain, 50 trees)…", flush=True)
    import pymc as pm
    import pymc_bart as pmb
    t0 = time.time()
    coords = {"feature": [f"f{i}" for i in range(n_feat)]}
    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X_train, dims=("row", "feature"))
        y_data = pm.Data("y", y_train, dims="row")
        mu = pmb.BART("mu", X=X_data, Y=y_train.astype("float64"), m=50)
        p = pm.Deterministic("p", pm.math.invlogit(mu), dims="row")
        pm.Bernoulli("obs", p=p, observed=y_data, dims="row")
        idata = pm.sample(draws=200, tune=200, chains=1, random_seed=42,
                          return_inferencedata=True, progressbar=False)
    with model:
        pm.set_data({"X": X_test, "y": np.zeros(n_test, dtype="int8")})
        ppc = pm.sample_posterior_predictive(idata, var_names=["p"],
                                              predictions=True, random_seed=42,
                                              progressbar=False)
    p_pmb = ppc.predictions["p"].values.mean(axis=(0, 1))
    t_pmb = time.time() - t0
    brier_pmb = float(np.mean((p_pmb - y_test) ** 2))
    print(f"[pymc-bart] done in {t_pmb:.1f}s")
    print(f"[pymc-bart] p_test mean={p_pmb.mean():.3f}, "
          f"range=[{p_pmb.min():.3f}, {p_pmb.max():.3f}], "
          f"Brier={brier_pmb:.4f}")

    print()
    print(f"speedup dbarts/pymc-bart: {t_pmb / t_dbarts:.1f}×")
    print(f"Brier delta (dbarts − pymc-bart): {brier_db - brier_pmb:+.4f}")
    print("OK")


if __name__ == "__main__":
    main()
