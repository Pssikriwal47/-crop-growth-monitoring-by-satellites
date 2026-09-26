"""
UMRCGM-S7 — training + export for the web demo
=================================================
Reproduces the synthetic calibration data and the UMRCGM feature set from
`UMRCGM_Bihta_Patna.ipynb`, trains a compact neural network per wheat growth
stage, and exports the trained weights + scaler as a single JSON file that
the static frontend (public/app.js) loads and runs directly in the browser
(no server, no TensorFlow needed at inference time).

Why not call Google Earth Engine here?
Step 2 of the original notebook (`ee.Authenticate()`) requires an
interactive browser login tied to your personal Google account, so it can't
run unattended in a build pipeline or on Vercel. The notebook itself
already falls back to `synthetic_data()` whenever GEE access fails — this
script trains on exactly that same fallback data. To train on your real
Sentinel-1/-3 extracts instead, replace `synthetic_data()` below with your
exported CSV and re-run this script.

Usage:
    pip install -r requirements.txt
    python train_and_export.py
"""

import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

GROWTH_STAGES = {
    "green_up": {"label": "Green-up", "window": "Dec 15 – Jan 15"},
    "jointing": {"label": "Jointing", "window": "Jan 15 – Feb 15"},
    "heading_filling": {"label": "Heading-filling", "window": "Feb 15 – Mar 15"},
    "milk_maturity": {"label": "Milk maturity", "window": "Mar 15 – Apr 5"},
}

FEATURES = ["sigma_VV", "sigma_VH", "LIA", "C11", "C12rel", "C12img", "C22"]
TARGETS = ["LAI", "CCC"]


def synthetic_data(stage, n=1200):
    """Same physically-calibrated generator as the notebook's fallback path."""
    rng = np.random.RandomState(hash(stage) % (2**32 - 1))
    params = {
        "green_up": {"lm": 1.5, "ls": 0.8, "cm": 120, "cs": 40},
        "jointing": {"lm": 2.8, "ls": 0.9, "cm": 200, "cs": 50},
        "heading_filling": {"lm": 3.5, "ls": 0.8, "cm": 180, "cs": 55},
        "milk_maturity": {"lm": 2.0, "ls": 0.7, "cm": 80, "cs": 35},
    }
    p = params[stage]
    lai = np.abs(rng.normal(p["lm"], p["ls"], n)).clip(0, 7)
    ccc = np.abs(rng.normal(p["cm"], p["cs"], n)).clip(0, 320)

    vv_db = -12 + 2 * lai + rng.normal(0, 1.5, n)
    vh_db = -18 + 2.5 * lai + rng.normal(0, 1.8, n)
    svv = 10 ** (vv_db / 10)
    svh = 10 ** (vh_db / 10)
    C11 = svv * (1 + rng.normal(0, 0.05, n))
    C22 = svh * (1 + rng.normal(0, 0.05, n))
    C12rel = np.sqrt(C11 * C22) * rng.uniform(0.3, 0.7, n)
    C12img = C12rel * rng.uniform(-0.3, 0.3, n)
    lia = rng.uniform(30, 45, n)

    return {
        "sigma_VV": svv, "sigma_VH": svh, "LIA": lia,
        "C11": C11, "C22": C22, "C12rel": C12rel, "C12img": C12img,
        "LAI": lai, "CCC": ccc,
    }


def mlp_to_json(model: MLPRegressor):
    layers = []
    n_layers = len(model.coefs_)
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        activation = "relu" if i < n_layers - 1 else "identity"
        layers.append({"W": W.tolist(), "b": b.tolist(), "activation": activation})
    return layers


def main():
    export = {"feature_order": FEATURES, "target_order": TARGETS, "stages": {}}

    print("Training UMRCGM-S7 reproduction per growth stage")
    print("=" * 60)
    for stage, info in GROWTH_STAGES.items():
        data = synthetic_data(stage)
        X = np.column_stack([data[f] for f in FEATURES]).astype(np.float32)
        y = np.column_stack([data[t] for t in TARGETS]).astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
        )
        model.fit(X_tr_s, y_tr)

        y_pred = model.predict(X_te_s)
        r2_lai = r2_score(y_te[:, 0], y_pred[:, 0])
        rmse_lai = float(np.sqrt(mean_squared_error(y_te[:, 0], y_pred[:, 0])))
        r2_ccc = r2_score(y_te[:, 1], y_pred[:, 1])
        rmse_ccc = float(np.sqrt(mean_squared_error(y_te[:, 1], y_pred[:, 1])))

        print(f"{info['label']:<18} LAI R2={r2_lai:.3f} RMSE={rmse_lai:.3f} | "
              f"CCC R2={r2_ccc:.3f} RMSE={rmse_ccc:.2f}")

        export["stages"][stage] = {
            "label": info["label"],
            "window": info["window"],
            "scaler": {
                "scale": scaler.scale_.tolist(),
                "min": scaler.min_.tolist(),
            },
            "layers": mlp_to_json(model),
            "metrics": {
                "r2_lai": round(float(r2_lai), 3),
                "rmse_lai": round(rmse_lai, 3),
                "r2_ccc": round(float(r2_ccc), 3),
                "rmse_ccc": round(rmse_ccc, 2),
            },
            "feature_ranges": {
                f: [float(data[f].min()), float(data[f].max())] for f in FEATURES
            },
        }

    out_path = "../public/model/umrcgm_weights.json"
    with open(out_path, "w") as f:
        json.dump(export, f)
    print("=" * 60)
    print(f"Exported weights -> {out_path}")


if __name__ == "__main__":
    main()
