"""Smoke: fit a tiny BART, pickle trees, rebuild model, reload trees,
predict on a different-size X, and verify it returns finite values that
match the live-model predictions to within sampling noise."""
from __future__ import annotations

import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import pickle
import sys
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pymc_bart as pmb

sys.path.insert(0, str(Path(__file__).parent))
from run_phase6_bart_bakeoff import get_bart_trees, set_bart_trees, predict_bart  # type: ignore


def main() -> None:
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 200, 80, 4
    X_train = rng.standard_normal((n_train, n_feat))
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(np.int8)
    X_test = rng.standard_normal((n_test, n_feat))
    feature_names = [f"f{i}" for i in range(n_feat)]

    coords = {"feature": feature_names}
    with pm.Model(coords=coords) as model_a:
        X_data = pm.Data("X", X_train, dims=("row", "feature"))
        y_data = pm.Data("y", y_train, dims="row")
        mu = pmb.BART("mu", X=X_data, Y=y_train.astype("float64"), m=4)
        p = pm.Deterministic("p", pm.math.invlogit(mu), dims="row")
        pm.Bernoulli("obs", p=p, observed=y_data, dims="row")
        idata = pm.sample(draws=20, tune=20, chains=1, random_seed=0,
                          return_inferencedata=True, progressbar=False)

    p_test_a = predict_bart(model_a, idata, X_test, seed=0)
    assert p_test_a.shape == (n_test,), f"bad shape from live model: {p_test_a.shape}"
    assert np.all(np.isfinite(p_test_a)), "non-finite from live model"

    # Plain mkdtemp (no auto-cleanup) — Windows holds a lock on the .nc
    # via netcdf4/HDF5 longer than the with-block, breaking auto-rmtree.
    td = tempfile.mkdtemp()
    if True:
        nc = Path(td) / "post.nc"
        pkl = Path(td) / "trees.pkl"
        idata.to_netcdf(str(nc))
        with open(pkl, "wb") as f:
            pickle.dump(get_bart_trees(model_a), f)

        idata_b = az.from_netcdf(str(nc))
        with pm.Model(coords=coords) as model_b:
            X_data = pm.Data("X", X_train, dims=("row", "feature"))
            y_data = pm.Data("y", y_train, dims="row")
            mu = pmb.BART("mu", X=X_data, Y=y_train.astype("float64"), m=4)
            p = pm.Deterministic("p", pm.math.invlogit(mu), dims="row")
            pm.Bernoulli("obs", p=p, observed=y_data, dims="row")
        with open(pkl, "rb") as f:
            set_bart_trees(model_b, pickle.load(f))

        p_test_b = predict_bart(model_b, idata_b, X_test, seed=0)

    assert p_test_b.shape == (n_test,), f"bad shape from rebuild: {p_test_b.shape}"
    assert np.all(np.isfinite(p_test_b)), "non-finite from rebuild"
    delta = np.max(np.abs(p_test_a - p_test_b))
    print(f"live    p_test mean={p_test_a.mean():.4f}  range=[{p_test_a.min():.3f}, {p_test_a.max():.3f}]")
    print(f"rebuild p_test mean={p_test_b.mean():.4f}  range=[{p_test_b.min():.3f}, {p_test_b.max():.3f}]")
    print(f"max |delta| live vs rebuild: {delta:.4f}")
    if delta > 0.1:
        raise SystemExit(f"FAIL: rebuild diverges from live (max abs delta = {delta:.3f})")
    print("OK")


if __name__ == "__main__":
    main()
