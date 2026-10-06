# rwfsas 
Utilizing public data to provide sitautional awareness to the general public. 

rwfsas is a split from fireintel
the purpose of rwfsas is to make advancements into the region of remote sensing as a real-time source of reliable information.

rwfsas will be released when >3x growth factor steps reach >85% acreage and 0.65 IoU. 
Current issuses
severe perimeter overestimations in small, explosive fires
FRP variance per fire type per conditions
missing data 
2km FRP blur


Over 100gb of training data!

## Code in this repo

Perimeter-estimation code, copied from the fireintel working tree with its local dependencies. Paths are relative to the repo no changes are necessary.

- `fpm/` — Bayesian burn-belief fusion (`fire_fusion.py`, `fusion_v3.py`), the interactive `perimeter_studio.py`, the local `console.py` app (port 8095), and the aircraft, camera and source-health inputs.
- `fire-spread-lab/` — `models/learned_field_v3.py` (learned perimeter-field model) and `data/abi_fire_area.py` (GOES ABI fire-area reader).
- `fire-spread-lab-claude/` — the live product: `deploy_train_field.py`, `deploy_live_field.py`, `field_loop.py`, `field_supervisor.py`, and `scripts/firms_fixed.py` (VIIRS).
- `sim/` — `data_ingest.py` (wind and fuel-moisture fusion), `rothermel.py`, `config.json`.

## Not included

- Trained weights and models: YOLO `.pt`/`.onnx` files and `deployable_field_v3.pkl`. The live field runner needs the pickle, which `deploy_train_field.py` rebuilds from training data.
- Training data and caches. DEM, fuel, GOES and VIIRS caches are fetched on demand.
- Aircraft and airport lookup tables, which are downloaded on first use.

## Configuration

- `FIRMS_MAP_KEY` — NASA FIRMS key (`fpm/fire_analysis.py`).
- `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET` — optional OpenSky access.
- `synoptic_api_key` in `sim/config.json` — optional.

Install with `pip install -r requirements.txt`.

## Results

Testing results are located in v0.1 documentation.pdf
