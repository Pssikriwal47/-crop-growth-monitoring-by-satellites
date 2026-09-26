# UMRCGM Wheat Growth Monitoring — Bihta, Patna

A browser-based demo of the UMRCGM model (Han et al., 2022, IEEE JSTARS),
which combines **two satellites**, across four Rabi growth stages for
Bihta, Patna, Bihar:

- **Sentinel-1 (SAR)** supplies the 7 input features this demo's sliders
  control — σVV, σVH, local incidence angle, and 4 polarimetric covariance
  terms (C11, C22, C12rel, C12img).
- **Sentinel-3 (OLCI, optical)** is where **Leaf Area Index (LAI)** and
  **Canopy Chlorophyll Content (CCC)** come from in the first place — the
  notebook's biophysical processor derives them from Sentinel-3 reflectance,
  and those values are the *training labels* the model learns to predict
  from Sentinel-1 alone.

That's the paper's core idea: once trained, the model estimates what
Sentinel-3 would have measured, using only Sentinel-1 radar — which still
works through cloud cover and at night, unlike optical imagery. So in this
demo, Sentinel-3 isn't a slider you set; it's the satellite whose
measurements LAI and CCC (the outputs) are standing in for.

- `training/` — Python script that trains the model and exports its weights.
- `public/` — the static site (HTML/CSS/JS) that runs the trained model
  directly in the browser. No backend, no server, nothing to keep running.

**Why it's not calling Google Earth Engine live:** the notebook's Step 2
(`ee.Authenticate()`) needs an interactive login to *your* Google account, so
it can't run unattended on Vercel or in any CI pipeline. The notebook itself
already handles this — whenever GEE isn't available it falls back to
`synthetic_data()`, a physically-calibrated data generator. This project
trains on exactly that fallback data. If you have real Sentinel-1/-3 extracts
(CSV of the 7 features + LAI/CCC), swap them into `training/train_and_export.py`
and re-run it — see "Training on your own data" below.

---

## 1. Project structure

```
umrcgm-app/
├── training/
│   ├── train_and_export.py  
│   └── requirements.txt
├── public/
│   ├── index.html
│   ├── style.css
│   ├── app.js                 
│   └── model/
│       └── umrcgm_weights.json
├── vercel.json
└── README.md
```


## 2. Run it locally (optional, before deploying)

```bash
cd umrcgm-app/public
python3 -m http.server 8000
```

Open http://localhost:8000 — pick a growth stage, adjust the SAR feature
sliders (or click "Use a sample field"), and it predicts LAI and CCC
instantly, all client-side.


## 6. Reference

Han et al. (2022). *Combining Sentinel-1 and -3 Imagery for Retrievals of
Regional Multitemporal Biophysical Parameters Under a Deep Learning
Framework.* IEEE JSTARS, 15, 6985–6998. DOI: 10.1109/JSTARS.2022.3200735
