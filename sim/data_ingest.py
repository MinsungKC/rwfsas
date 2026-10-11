#!/usr/bin/env python3
"""
Shared data ingestion + caching for the fire simulation pipeline
(spread_engine, pyrocb_model, danger_map, trainers, job_runner).

Rules honored everywhere:
  - urllib.request only (no requests/httpx/aiohttp)
  - never CONUS/state-wide raster pulls — always bbox-clipped (+ buffer)
  - every fetched artifact cached to sim_cache/ with per-category TTLs
    (minutes, from config cache_ttl_minutes; 0 = never expires)
  - all files written atomically (.tmp then os.replace) so the servers
    never see a half-written file
  - netCDF4 for NOMADS OPeNDAP (RTMA + HRRR). HRRR uses the NOMADS dods
    endpoint rather than filter_hrrr_2d.pl: the filter CGI returns GRIB2,
    which would force a cfgrib/eccodes dependency; dods serves the same
    5 surface fields (UGRD/VGRD/RH/TMP/GUST) bbox-sliced via netCDF4.
  - Synoptic RAWS is optional: placeholder key in config => skipped
    gracefully, never an error.

Every getter degrades gracefully — a dead endpoint returns a fallback (or
None) with a logged warning instead of raising.
"""
import csv
import io
import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import numpy as np

log = logging.getLogger("sim.ingest")

SIM_DIR = Path(__file__).parent
FDV_DIR = SIM_DIR.parent
FPM_DIR = FDV_DIR / "fpm"              # moved into FDV 2026-07-04 (junction at
                                       # Documents/fpm keeps old callers alive)
CONFIG  = json.loads((SIM_DIR / "config.json").read_text(encoding="utf-8"))

CACHE_DIR = (SIM_DIR / CONFIG["paths"]["cache_dir"]).resolve()
DEM_DIR   = (SIM_DIR / CONFIG["paths"]["dem_dir"]).resolve()
IGN_DIR   = CACHE_DIR / "ignitions"
for d in (CACHE_DIR, DEM_DIR, IGN_DIR):
    d.mkdir(parents=True, exist_ok=True)

_UA = {"User-Agent": "FDV-fire-sim/2.0 (research; youngimyoo@yahoo.com)"}


# ── Atomic writes ─────────────────────────────────────────────────────────────

def _replace_retry(tmp, path, tries=8):
    """os.replace, retrying on Windows sharing violations. status.json and
    the danger geojson are polled by the 8080 server every ~1s; on Windows
    os.replace raises PermissionError(WinError 5/32) if the destination is
    momentarily open for reading, which would otherwise crash a sim
    mid-run. Retry briefly, then fall back to a plain write."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                break
            time.sleep(0.05 * (i + 1))
    # last resort: in-place overwrite (loses atomicity but never crashes)
    try:
        with open(tmp, "rb") as s, open(path, "wb") as d:
            d.write(s.read())
        os.remove(tmp)
    except Exception as e:
        log.warning(f"atomic replace fallback failed for {path}: {e}")


def _tmp_path(path):
    """Unique tmp name per process+write. A fixed '<file>.tmp' races when
    two processes write the same file (server + sim both write status.json):
    each deletes the other's tmp mid-rename -> FileNotFoundError(WinError 2)
    crashes a sim mid-run."""
    return path.with_name(f"{path.name}.{os.getpid()}-{os.urandom(3).hex()}.tmp")


def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = _tmp_path(path)
    tmp.write_bytes(data)
    _replace_retry(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    tmp = _tmp_path(path)
    tmp.write_text(text, encoding="utf-8")
    _replace_retry(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


# ── HTTP + cache ──────────────────────────────────────────────────────────────

def http_get(url, timeout=60, retries=3, headers=None):
    hdrs = dict(_UA)
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"GET failed after {retries} tries: {url[:120]} — {e}")
                return None
            time.sleep(1.5 * 2 ** attempt)


def _cache_path(key, ext="json"):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return CACHE_DIR / f"{safe}.{ext}"


def cache_get(key, category="weather", ext="json"):
    p = _cache_path(key, ext)
    if not p.exists():
        return None
    ttl_min = CONFIG["cache_ttl_minutes"].get(category, 60)
    if ttl_min > 0 and (time.time() - p.stat().st_mtime) > ttl_min * 60:
        return None
    if ext == "json":
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return p.read_bytes()


def cache_put(key, payload, ext="json"):
    p = _cache_path(key, ext)
    if ext == "json":
        atomic_write_json(p, payload)
    else:
        atomic_write_bytes(p, payload)
    return p


def get_osm_roads(bbox):
    """Major road centerlines (motorway/trunk/primary) within bbox from the
    public OSM Overpass API -- fuel-break geometry for the simulator (a
    real fire stops at a wide/cleared road corridor unless it's intense
    enough to throw embers across, so these become an "unburnable unless
    jumped by spotting" layer, same idea as water). Cached forever (roads
    rarely change); degrades to [] on any failure rather than raising, per
    this module's contract.
    Returns [{"kind": "motorway"|"trunk"|"primary", "points": [[lat,lon],...]}, ...].
    """
    key = "osm_roads_" + "_".join(f"{c:.4f}" for c in bbox)
    cached = cache_get(key, category="static", ext="json")
    if cached is not None:
        return cached
    w, s, e, n = bbox
    query = ("[out:json][timeout:60];"
            f'way["highway"~"^(motorway|trunk|primary)$"]({s},{w},{n},{e});'
            "out geom;")
    url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query)
    raw = http_get(url, timeout=90, retries=2)
    ways = []
    if raw is not None:
        try:
            data = json.loads(raw)
            for el in data.get("elements", []):
                if el.get("type") != "way" or "geometry" not in el:
                    continue
                kind = el.get("tags", {}).get("highway", "road")
                pts = [[g["lat"], g["lon"]] for g in el["geometry"]]
                if len(pts) >= 2:
                    ways.append({"kind": kind, "points": pts})
        except Exception as e:
            log.warning(f"OSM Overpass roads parse failed: {e}")
    cache_put(key, ways, ext="json")
    return ways


def buffer_bbox(bbox, km=None):
    """Expand bbox by config sim_bbox_buffer_km (or km) on all sides."""
    km = CONFIG["sim_bbox_buffer_km"] if km is None else km
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians((bbox[1] + bbox[3]) / 2)))
    return (bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat)


# ── Grid helpers ──────────────────────────────────────────────────────────────

def make_grid(bbox, res_m):
    """Regular lat/lon grid; row 0 = north edge (raster order)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_step = res_m / 111320.0
    lon_step = res_m / (111320.0 * math.cos(math.radians((min_lat + max_lat) / 2)))
    lats = np.arange(max_lat, min_lat, -lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    return lats, lons


def bilinear_sample(coarse, c_lats, c_lons, f_lats, f_lons):
    from scipy.interpolate import RegularGridInterpolator
    c_lats = np.asarray(c_lats, dtype=float)
    c_lons = np.asarray(c_lons, dtype=float)
    arr = np.asarray(coarse, dtype=float)
    if c_lats[0] > c_lats[-1]:
        c_lats = c_lats[::-1]
        arr = arr[::-1, :]
    interp = RegularGridInterpolator((c_lats, c_lons), arr,
                                     bounds_error=False, fill_value=None)
    gy, gx = np.meshgrid(f_lats, f_lons, indexing="ij")
    return interp(np.stack([gy.ravel(), gx.ravel()], axis=1)).reshape(gy.shape)


def raster_rowcol(src, xs, ys):
    """Vectorized world→pixel (rasterio's src.index rejects arrays)."""
    inv = ~src.transform
    cols, rows = inv * (np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64))
    return np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)


# ── DEM / terrain (USGS 3DEP, 1×1° tiles, cached permanently) ────────────────

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
_TNM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second"


def _dem_tile_paths(bbox):
    q = urllib.parse.urlencode({
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": _TNM_DATASET, "prodFormats": "GeoTIFF",
    })
    raw = http_get(f"{TNM_API}?{q}", timeout=45)
    if not raw:
        return [p for p in DEM_DIR.glob("*.tif")]      # offline: use whatever we have
    best = {}
    for it in json.loads(raw).get("items", []):
        url = it.get("downloadURL")
        if not url:
            continue
        fname = url.rsplit("/", 1)[-1]
        tkey = fname.rsplit("_", 1)[0]
        if tkey not in best or fname > best[tkey].rsplit("/", 1)[-1]:
            best[tkey] = url
    paths = []
    for url in best.values():
        p = DEM_DIR / url.rsplit("/", 1)[-1]
        if not p.exists():
            log.info(f"Downloading DEM tile {p.name} …")
            data = http_get(url, timeout=600, retries=2)
            if data is None:
                continue
            atomic_write_bytes(p, data)
        paths.append(p)
    return paths


def get_dem(bbox, lats, lons):
    """Elevation (m) mosaicked from 3DEP tiles onto the grid; zeros if none."""
    out = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio missing — flat terrain")
        return np.zeros_like(out)
    for p in _dem_tile_paths(bbox):
        try:
            with rasterio.open(p) as src:
                gy, gx = np.meshgrid(lats, lons, indexing="ij")
                rows, cols = raster_rowcol(src, gx.ravel(), gy.ravel())
                ok = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
                if not ok.any():
                    continue
                band = src.read(1)
                vals = np.full(rows.shape, np.nan, dtype=np.float32)
                vals[ok] = band[rows[ok], cols[ok]]
                vals[vals < -1000] = np.nan
                flat = out.ravel()
                take = ~np.isnan(vals)
                flat[take] = vals[take]
                out = flat.reshape(out.shape)
        except Exception as e:
            log.warning(f"DEM read failed for {p.name}: {e}")
    if np.isnan(out).all():
        log.warning("No DEM coverage — flat terrain")
        return np.zeros_like(out)
    out[np.isnan(out)] = np.nanmean(out)
    return out


def get_dem_coarse(bbox, lats, lons):
    """
    Coarse elevation for big grids (danger map @1km). Local tiles are used
    when present; missing tiles are read REMOTELY via /vsicurl windowed,
    decimated reads against the public 3DEP COGs (internally tiled +
    overview pyramids) — a few MB of HTTP range requests per tile instead
    of a ~450MB download. Never pulls a full state-wide raster.
    """
    out = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        return np.zeros_like(out)
    q = urllib.parse.urlencode({
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": _TNM_DATASET, "prodFormats": "GeoTIFF",
    })
    raw = http_get(f"{TNM_API}?{q}", timeout=60)
    urls = {}
    if raw:
        for it in json.loads(raw).get("items", []):
            u = it.get("downloadURL")
            if not u:
                continue
            fname = u.rsplit("/", 1)[-1]
            tkey = fname.rsplit("_", 1)[0]
            if tkey not in urls or fname > urls[tkey].rsplit("/", 1)[-1]:
                urls[tkey] = u
    res_deg = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.01
    for tkey, u in urls.items():
        local = DEM_DIR / u.rsplit("/", 1)[-1]
        src_path = str(local) if local.exists() else f"/vsicurl/{u}"
        try:
            with rasterio.open(src_path) as src:
                b = src.bounds
                ix0, iy0 = max(bbox[0], b.left), max(bbox[1], b.bottom)
                ix1, iy1 = min(bbox[2], b.right), min(bbox[3], b.top)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                win = from_bounds(ix0, iy0, ix1, iy1, src.transform)
                oh = max(int((iy1 - iy0) / res_deg), 2)
                ow = max(int((ix1 - ix0) / res_deg), 2)
                arr = src.read(1, window=win, out_shape=(oh, ow)).astype(np.float32)
                arr[arr < -1000] = np.nan
                t_lats = np.linspace(iy1, iy0, oh)
                t_lons = np.linspace(ix0, ix1, ow)
                rsel = np.where((lats <= iy1) & (lats >= iy0))[0]
                csel = np.where((lons >= ix0) & (lons <= ix1))[0]
                if len(rsel) == 0 or len(csel) == 0:
                    continue
                ri = np.clip(((iy1 - lats[rsel]) / max(iy1 - iy0, 1e-9)
                              * (oh - 1)).astype(int), 0, oh - 1)
                ci = np.clip(((lons[csel] - ix0) / max(ix1 - ix0, 1e-9)
                              * (ow - 1)).astype(int), 0, ow - 1)
                vals = arr[np.ix_(ri, ci)]
                blk = out[np.ix_(rsel, csel)]
                take = ~np.isnan(vals)
                blk[take] = vals[take]
                out[np.ix_(rsel, csel)] = blk
        except Exception as e:
            log.warning(f"coarse DEM {tkey}: {e}")
    if np.isnan(out).all():
        log.warning("coarse DEM empty — flat terrain")
        return np.zeros_like(out)
    out[np.isnan(out)] = np.nanmean(out)
    return out


def get_dem_best(bbox, lats, lons):
    """
    Best-quality elevation: GDAL area-weighted read at the EXACT target grid
    resolution — every ~10m 3DEP native pixel under each target cell
    contributes to that cell's value (rasterio/GDAL Resampling.average),
    instead of get_dem_coarse's single-nearest-pixel decimated read. Slower
    (touches every native pixel instead of one per cell) — opt-in via the
    "Best Quality" toggle.
    """
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    ny, nx = len(lats), len(lons)
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    try:
        import rasterio
    except ImportError:
        return np.zeros_like(out)
    q = urllib.parse.urlencode({
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": _TNM_DATASET, "prodFormats": "GeoTIFF",
    })
    raw = http_get(f"{TNM_API}?{q}", timeout=60)
    urls = {}
    if raw:
        for it in json.loads(raw).get("items", []):
            u = it.get("downloadURL")
            if not u:
                continue
            fname = u.rsplit("/", 1)[-1]
            tkey = fname.rsplit("_", 1)[0]
            if tkey not in urls or fname > urls[tkey].rsplit("/", 1)[-1]:
                urls[tkey] = u
    lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.002
    lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.002
    for tkey, u in urls.items():
        local = DEM_DIR / u.rsplit("/", 1)[-1]
        src_path = str(local) if local.exists() else f"/vsicurl/{u}"
        try:
            with rasterio.open(src_path) as src:
                b = src.bounds
                rsel = np.where((lats <= b.top) & (lats >= b.bottom))[0]
                csel = np.where((lons >= b.left) & (lons <= b.right))[0]
                if len(rsel) == 0 or len(csel) == 0:
                    continue
                # window spans cell EDGES (not centers) so the average is honest
                west, east = lons[csel[0]] - lon_step / 2, lons[csel[-1]] + lon_step / 2
                north, south = lats[rsel[0]] + lat_step / 2, lats[rsel[-1]] - lat_step / 2
                win = from_bounds(west, south, east, north, src.transform)
                nodata = src.nodata if src.nodata is not None else -9999.0
                arr = src.read(1, window=win, out_shape=(len(rsel), len(csel)),
                              resampling=Resampling.average,
                              boundless=True, fill_value=nodata).astype(np.float32)
                arr[arr <= nodata + 1] = np.nan
                arr[arr < -1000] = np.nan
                blk = out[np.ix_(rsel, csel)]
                take = ~np.isnan(arr) & np.isnan(blk)
                blk[take] = arr[take]
                out[np.ix_(rsel, csel)] = blk
        except Exception as e:
            log.warning(f"best-quality DEM {tkey}: {e}")
    if np.isnan(out).all():
        log.warning("best-quality DEM empty — flat terrain")
        return np.zeros_like(out)
    out[np.isnan(out)] = np.nanmean(out)
    return out


def slope_aspect(elev, lats, lons):
    res_y = abs(lats[1] - lats[0]) * 111320.0 if len(lats) > 1 else 1.0
    res_x = abs(lons[1] - lons[0]) * 111320.0 * math.cos(math.radians(float(np.mean(lats))))
    dz_dy, dz_dx = np.gradient(elev.astype(np.float64), res_y, res_x)
    # row 0 = north (make_grid), so +row is southward: negate to get the
    # NORTHWARD derivative. dz_dx is already the EASTWARD derivative.
    dz_dy = -dz_dy
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    # Aspect = DOWNSLOPE compass azimuth (standard GIS convention, and what
    # every consumer here assumes: rothermel.py/sim_worker.js derive upslope
    # as aspect+180, the downhill-deceleration term treats cos(bearing-aspect)
    # =+1 as straight downhill, and the hillshade lights from it).
    # The steepest-ASCENT vector is (east=dz_dx, north=dz_dy), so descent is
    # its negation -> atan2(-dz_dx, -dz_dy). Negating only the east component
    # (the previous form) mirrored the azimuth about the E-W axis: correct for
    # east/west-facing slopes but 180deg WRONG for north/south-facing ones,
    # which made fire run downhill and stall uphill on N/S slopes.
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


def compute_tpi(elev, lats, lons, radius_m=1000):
    """Topographic Position Index: elevation minus a local neighborhood
    mean (radius_m real-world radius, converted to a cell count from grid
    spacing). Raw meters, unclipped/unnormalized — callers apply their own
    scaling. Shared by influence_maps.py and sim_data.py."""
    from scipy.ndimage import uniform_filter
    res_y = abs(lats[1] - lats[0]) * 111320.0 if len(lats) > 1 else 250.0
    radius_cells = max(1, round(radius_m / max(res_y, 1.0)))
    size = 2 * radius_cells + 1
    mean_elev = uniform_filter(elev.astype(np.float64), size=size, mode="nearest")
    return (elev.astype(np.float64) - mean_elev).astype(np.float32)


def compute_curvature(elev, lats, lons):
    """Finite-difference Laplacian of elevation (1/m): d2z/dx2 + d2z/dy2,
    using the same real-world cell spacing as slope_aspect. Positive =
    convex (ridge/knob), negative = concave (valley/bowl)."""
    e = elev.astype(np.float64)
    res_y = abs(lats[1] - lats[0]) * 111320.0 if len(lats) > 1 else 1.0
    res_x = abs(lons[1] - lons[0]) * 111320.0 * math.cos(math.radians(float(np.mean(lats))))
    dzdy, dzdx = np.gradient(e, res_y, res_x)
    d2zdy2, _ = np.gradient(dzdy, res_y, res_x)
    _, d2zdx2 = np.gradient(dzdx, res_y, res_x)
    return (d2zdx2 + d2zdy2).astype(np.float32)


# ── LANDFIRE rasters via LFPS job API (fuel model + canopy cover) ─────────────

def _lfps_fetch_tif(bbox, layer, tif_path):
    import zipfile
    lf = CONFIG["landfire"]
    aoi = f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}"
    q = urllib.parse.urlencode({
        "Layer_List": layer, "Area_of_Interest": aoi,
        "Output_Projection": "4326", "Email": lf["email"],
    })
    raw = http_get(f'{lf["lfps_base"]}/job/submit?{q}', timeout=60)
    if not raw:
        return False
    job = json.loads(raw)
    job_id = job.get("jobId")
    if not job_id:
        log.warning(f"LFPS submit rejected: {job}")
        return False
    log.info(f"LFPS job {job_id} ({layer}, AOI {aoi}) — polling…")
    deadline = time.time() + lf["poll_timeout_min"] * 60
    out_url = None
    while time.time() < deadline:
        time.sleep(8)
        raw = http_get(f'{lf["lfps_base"]}/job/status?JobId={job_id}', timeout=30)
        if not raw:
            continue
        st = json.loads(raw)
        if st.get("status") == "Succeeded":
            out_url = st.get("outputFile")
            break
        if st.get("status") == "Failed":
            log.warning("LFPS failed: " + "; ".join(
                m["description"] for m in st.get("messages", [])
                if m.get("type", "").endswith("Error")))
            return False
    if not out_url:
        log.warning("LFPS job timed out")
        return False
    data = http_get(out_url, timeout=600, retries=2)
    if not data:
        return False
    zpath = tif_path.with_suffix(".zip")
    atomic_write_bytes(zpath, data)
    with zipfile.ZipFile(zpath) as zf:
        tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        if not tifs:
            return False
        atomic_write_bytes(tif_path, zf.read(tifs[0]))
    zpath.unlink(missing_ok=True)
    log.info(f"{layer} raster cached ({tif_path.stat().st_size/1e6:.1f} MB)")
    return True


def _sample_lfps_raster(layer, key_prefix, bbox, lats, lons, dtype, fallback,
                        resample_mode="nearest"):
    """
    resample_mode:
      "nearest"  — one native LFPS pixel (30m) per target cell (fast, default).
      "average"  — GDAL area-weighted mean over every native pixel under each
                   target cell (continuous fields: canopy %, canopy bulk density).
      "majority" — GDAL majority-vote (mode) over every native pixel under
                   each target cell (categorical: fuel model codes — never
                   average a fuel code).
    """
    key = f"{key_prefix}_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    tif_path = _cache_path(key, "tif")
    if not tif_path.exists():
        try:
            _lfps_fetch_tif(bbox, layer, tif_path)
        except Exception as e:
            log.warning(f"LFPS fetch failed ({layer}): {e}")
    if tif_path.exists():
        try:
            import rasterio
            with rasterio.open(tif_path) as src:
                if resample_mode == "nearest":
                    gy, gx = np.meshgrid(lats, lons, indexing="ij")
                    rows, cols = raster_rowcol(src, gx.ravel(), gy.ravel())
                    rows = np.clip(rows, 0, src.height - 1)
                    cols = np.clip(cols, 0, src.width - 1)
                    band = src.read(1)
                    return band[rows, cols].reshape(gy.shape).astype(dtype)
                from rasterio.enums import Resampling
                from rasterio.windows import from_bounds
                lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.002
                lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.002
                west, east = lons[0] - lon_step / 2, lons[-1] + lon_step / 2
                north, south = lats[0] + lat_step / 2, lats[-1] - lat_step / 2
                win = from_bounds(west, south, east, north, src.transform)
                resampling = (Resampling.mode if resample_mode == "majority"
                             else Resampling.average)
                nodata = src.nodata if src.nodata is not None else 0
                arr = src.read(1, window=win, out_shape=(len(lats), len(lons)),
                              resampling=resampling, boundless=True,
                              fill_value=nodata)
                return arr.astype(dtype)
        except Exception as e:
            log.warning(f"{layer} raster read failed: {e}")
    log.warning(f"{layer} unavailable — uniform fallback {fallback}")
    return np.full((len(lats), len(lons)), fallback, dtype=dtype)


def get_fbfm40(bbox, lats, lons, best_quality=False):
    """Scott & Burgan fuel-model codes (int16) from LANDFIRE via LFPS.
    best_quality: majority-vote over every native 30m pixel per target cell
    instead of one nearest pixel (fuel codes are categorical — never average
    them)."""
    fb = int(CONFIG["landfire"]["fallback_uniform_model"])
    return _sample_lfps_raster(CONFIG["landfire"]["layer"], "fbfm40",
                               bbox, lats, lons, np.int16, fb,
                               resample_mode="majority" if best_quality else "nearest")


def get_canopy_cover(bbox, lats, lons, best_quality=False):
    """LANDFIRE canopy cover % (drives the wind adjustment factor).
    best_quality: area-weighted average over every native 30m pixel per
    target cell instead of one nearest pixel."""
    cc = _sample_lfps_raster(CONFIG["landfire"]["canopy_layer"], "canopy",
                             bbox, lats, lons, np.float32, 0.0,
                             resample_mode="average" if best_quality else "nearest")
    return np.clip(cc, 0, 100)


def get_canopy_bulk_density(bbox, lats, lons, best_quality=False):
    """LANDFIRE canopy bulk density (kg/m^3 * 100, per LF convention) —
    used only for the influence-map spotting-potential layer."""
    layer = CONFIG["landfire"].get("cbd_layer", "LF2024_CBD")
    cbd = _sample_lfps_raster(layer, "cbd", bbox, lats, lons, np.float32, 5.0,
                              resample_mode="average" if best_quality else "nearest")
    return np.clip(cbd, 0, 50)


# ── Open-Meteo (multi-point, forecast + historical) ───────────────────────────

_OM_FORE = "https://api.open-meteo.com/v1/forecast"
_OM_HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_OM_ERA5 = "https://archive-api.open-meteo.com/v1/archive"


def om_multi(base, pts, params):
    """Batched (≤50/call) multi-point Open-Meteo query."""
    results = [None] * len(pts)
    for i0 in range(0, len(pts), 50):
        chunk = pts[i0:i0 + 50]
        q = dict(params)
        q["latitude"]  = ",".join(f"{la:.4f}" for la, lo in chunk)
        q["longitude"] = ",".join(f"{lo:.4f}" for la, lo in chunk)
        raw = http_get(f"{base}?{urllib.parse.urlencode(q)}", timeout=90)
        if not raw:
            continue
        d = json.loads(raw)
        d = d if isinstance(d, list) else [d]
        for j, item in enumerate(d):
            if i0 + j < len(results):
                results[i0 + j] = item
        time.sleep(0.15)
    return results


def get_weather_grid(bbox, n=5, hours_ahead=48, start_date=None, end_date=None,
                     hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,"
                            "wind_direction_10m,wind_gusts_10m,precipitation"):
    """n×n hourly Open-Meteo grid (forecast, or historical when dates given)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.linspace(max_lat, min_lat, n)
    lons = np.linspace(min_lon, max_lon, n)
    pts = [(la, lo) for la in lats for lo in lons]
    mode = "hist" if start_date else "fore"
    key = (f"wxgrid_{mode}_{n}_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_"
           + (f"{start_date}_{end_date}" if start_date else f"h{hours_ahead}"))
    cached = cache_get(key, "weather")
    if cached:
        return cached
    params = {"hourly": hourly, "timezone": "UTC", "wind_speed_unit": "mph"}
    if start_date:
        params["start_date"] = str(start_date)
        params["end_date"] = str(end_date or start_date)
        base = _OM_HIST if start_date >= date(2016, 1, 1) else _OM_ERA5
    else:
        params["forecast_days"] = max(1, math.ceil(hours_ahead / 24))
        params["past_hours"] = 24
        base = _OM_FORE
    res = om_multi(base, pts, params)
    points = []
    for item in res:
        if item and item.get("hourly", {}).get("time"):
            h = item["hourly"]
            points.append({k: h.get(k) for k in ["time"] + hourly.split(",")})
        else:
            points.append(None)
    if not any(points):
        log.warning("Open-Meteo grid fetch failed entirely")
        return None
    out = {"n": n, "lats": lats.tolist(), "lons": lons.tolist(),
           "points": points, "source": "open-meteo"}
    cache_put(key, out)
    return out


def get_atmos_profile(lat, lon, start_date=None, end_date=None, hours_ahead=48):
    """PyroCb atmosphere: CAPE, CIN(li), mid RH, lapse-rate inputs, shear."""
    hourly = ("cape,lifted_index,relative_humidity_500hPa,relative_humidity_700hPa,"
              "wind_speed_500hPa,wind_speed_700hPa,wind_speed_10m,"
              "temperature_2m,dew_point_2m,temperature_500hPa,temperature_700hPa,"
              "geopotential_height_500hPa,geopotential_height_700hPa")
    key = f"atmos_{lat:.3f}_{lon:.3f}_" + (f"{start_date}_{end_date}"
                                           if start_date else f"h{hours_ahead}")
    cached = cache_get(key, "weather")
    if cached:
        return cached
    params = {"hourly": hourly, "timezone": "UTC", "wind_speed_unit": "ms"}
    if start_date:
        params["start_date"] = str(start_date)
        params["end_date"] = str(end_date or start_date)
        base = _OM_HIST
    else:
        params["forecast_days"] = max(1, math.ceil(hours_ahead / 24))
        base = _OM_FORE
    res = om_multi(base, [(lat, lon)], params)
    if not res or not res[0] or not res[0].get("hourly", {}).get("time"):
        log.warning("Atmos profile fetch failed")
        return None
    out = res[0]["hourly"]
    cache_put(key, out)
    return out


# ── IEM Mesonet: real ASOS/RAWS station obs at an arbitrary historical hour ──
# Free, no API key, same archive the main map's live station overlay uses
# (goes19_replay_v3.py's _fetch_iem_asos/_fetch_iem_raws) — generalized here
# to any past hour, not just "now". This is real ground truth for a past
# fire-weather event, vs Open-Meteo's smoothed reanalysis model, which can
# miss hyper-local canyon/foehn effects during extreme wind events.

_IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_IEM_CONUS_STATES = ["AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA",
                     "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
                     "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",
                     "TN","TX","UT","VT","VA","WA","WV","WI","WY"]
_IEM_RAWS_STATES = ["CA","OR","WA","NV","AZ","NM","CO","UT","WY","MT","ID","TX","OK"]


def _iem_parse_station_csv(text, target_epoch):
    """Parse IEM asos.py 'onlycomma' CSV into station dicts, best (closest
    to target_epoch, within 1h) observation per station.
    Returns [{id, lat, lon, temp_f, rh, u, v, delta}] — u/v are wind
    components in mph (the direction the wind blows TOWARD, standard
    vector convention, so they can be IDW-interpolated directly without
    the 0/360 wraparound problem a raw compass direction would have)."""
    best = {}
    header = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("station,"):
            header = line.split(",")
            continue
        if header is None:
            continue
        row = line.split(",")
        if len(row) < len(header):
            continue

        def _col(name):
            try:
                v = row[header.index(name)].strip()
                return None if v in ("M", "T", "") else v
            except (ValueError, IndexError):
                return None

        sid = row[0].strip()
        valid_s = row[1].strip() if len(row) > 1 else ""
        lon_s = _col("lon"); lat_s = _col("lat")
        tmpf_s = _col("tmpf"); dwpf_s = _col("dwpf")
        sknt_s = _col("sknt"); drct_s = _col("drct")
        if not all([sid, valid_s, lon_s, lat_s, tmpf_s, sknt_s, drct_s]):
            continue
        try:
            ot = int(datetime.strptime(valid_s, "%Y-%m-%d %H:%M")
                     .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
        delta = abs(ot - target_epoch)
        if delta > 3600:
            continue
        if sid in best and delta >= best[sid]["delta"]:
            continue
        try:
            spd_mph = float(sknt_s) * 1.15078
            rad = math.radians(float(drct_s))
            u_v = -spd_mph * math.sin(rad)
            v_v = -spd_mph * math.cos(rad)
            rh = None
            if dwpf_s:
                t_c = (float(tmpf_s) - 32) * 5 / 9
                d_c = (float(dwpf_s) - 32) * 5 / 9
                rh = 100 * math.exp(17.625 * d_c / (243.04 + d_c)) / \
                          math.exp(17.625 * t_c / (243.04 + t_c))
        except (ValueError, TypeError):
            continue
        best[sid] = dict(id=sid, lat=float(lat_s), lon=float(lon_s),
                         temp_f=float(tmpf_s), rh=rh, u=u_v, v=v_v, delta=delta)
    return list(best.values())


def get_station_obs_at(bbox, dt, buffer_km=150):
    """
    Real ASOS+RAWS station observations within buffer_km of bbox's center
    at a specific historical hour. Returns None if no station reported
    near bbox in that ±65 min window, so callers can fall back to a model
    source. dt should be in the past — there's obviously no future obs.
    """
    key = (f"iemobs_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_"
           f"{dt.strftime('%Y%m%dT%H')}")
    cached = cache_get(key, "weather")
    if cached is not None:
        return cached or None

    t1 = dt - timedelta(minutes=65)
    t2 = dt + timedelta(minutes=65)
    target_epoch = int(dt.timestamp())
    time_q = (f"year1={t1.year}&month1={t1.month:02d}&day1={t1.day:02d}"
             f"&hour1={t1.hour:02d}&minute1={t1.minute:02d}"
             f"&year2={t2.year}&month2={t2.month:02d}&day2={t2.day:02d}"
             f"&hour2={t2.hour:02d}&minute2={t2.minute:02d}")

    def _fetch(url):
        try:
            raw = http_get(url, timeout=25)
            return _iem_parse_station_csv(raw.decode("utf-8", "replace"), target_epoch) if raw else []
        except Exception:
            return []

    asos_url = (f"{_IEM_ASOS_URL}?data=tmpf,dwpf,sknt,drct&tz=UTC&format=onlycomma"
               f"&latlon=yes&missing=M&trace=T&direct=no&report_type=3,4&{time_q}&"
               + "&".join(f"state={s}" for s in _IEM_CONUS_STATES))
    raws_urls = [(f"{_IEM_ASOS_URL}?network={s}_RAWS&data=tmpf,dwpf,sknt,drct&tz=UTC"
                 f"&format=onlycomma&latlon=yes&missing=M&trace=T&direct=no&{time_q}")
                for s in _IEM_RAWS_STATES]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_fetch, [asos_url] + raws_urls))
    seen = {}
    for batch in results:
        for s in batch:
            if s["id"] not in seen or s["delta"] < seen[s["id"]]["delta"]:
                seen[s["id"]] = s

    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    nearby = [s for s in seen.values()
             if math.hypot((s["lat"] - cy) * 111.0,
                          (s["lon"] - cx) * 111.0 * math.cos(math.radians(cy))) <= buffer_km]
    cache_put(key, nearby)
    return nearby or None


def station_obs_to_grid(stations, lats, lons, radius_deg=1.5, power=2.0):
    """IDW-interpolate station obs onto an arbitrary lat/lon grid.
    Returns (wind_mph, wind_dir_from_deg, temp_f, rh) 2D arrays, or None if
    no station has any usable data within radius_deg of any grid cell."""
    s_lats = np.array([s["lat"] for s in stations])
    s_lons = np.array([s["lon"] for s in stations])
    s_u = np.array([s["u"] for s in stations])
    s_v = np.array([s["v"] for s in stations])
    s_t = np.array([s["temp_f"] for s in stations])
    s_rh = np.array([s["rh"] if s["rh"] is not None else np.nan for s in stations])

    g_lats, g_lons = np.meshgrid(np.asarray(lats), np.asarray(lons), indexing="ij")
    dlat = g_lats[..., None] - s_lats[None, None, :]
    dlon = g_lons[..., None] - s_lons[None, None, :]
    d = np.maximum(np.sqrt(dlat**2 + dlon**2), 0.001)
    w = np.where(d < radius_deg, 1.0 / d**power, 0.0)
    wsum = w.sum(axis=-1)
    has_data = wsum > 0
    if not has_data.any():
        return None
    wsum_safe = np.where(has_data, wsum, 1.0)
    w_n = w / wsum_safe[..., None]
    u_out = (w_n * s_u).sum(axis=-1)
    v_out = (w_n * s_v).sum(axis=-1)
    t_out = (w_n * s_t).sum(axis=-1)

    rh_valid = ~np.isnan(s_rh)
    w_rh = np.where(rh_valid[None, None, :], w, 0.0)
    ws_rh = w_rh.sum(axis=-1)
    ws_rh_safe = np.where(ws_rh > 0, ws_rh, 1.0)
    rh_num = (w_rh * np.where(rh_valid, s_rh, 0.0)[None, None, :]).sum(axis=-1)
    rh_out = np.where(ws_rh > 0, rh_num / ws_rh_safe, np.nan)

    fallback_t = float(np.mean(t_out[has_data])) if has_data.any() else 60.0
    t_out = np.where(has_data, t_out, fallback_t)
    u_out = np.where(has_data, u_out, 0.0)
    v_out = np.where(has_data, v_out, 0.0)
    rh_out = np.where(np.isnan(rh_out), 40.0, rh_out)

    speed_mph = np.hypot(u_out, v_out)
    to_bearing_deg = np.degrees(np.arctan2(u_out, v_out))
    from_deg = (to_bearing_deg + 180.0) % 360.0
    return speed_mph, from_deg, t_out, rh_out


# ── NOMADS OPeNDAP: RTMA (current analysis) + HRRR (forecast) ─────────────────

def _nomads_slice(ds, var, yi, xi, tidx=0, stride=1):
    sl = np.s_[tidx, yi[0]:yi[-1] + 1:stride, xi[0]:xi[-1] + 1:stride]
    return np.asarray(ds.variables[var][sl], dtype=np.float64)


def get_rtma_current(bbox):
    """Latest RTMA 2.5km wind/T/RH/gust over bbox. None on failure."""
    key = f"rtma_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    cached = cache_get(key, "rtma")
    if cached:
        return cached
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None
    now = datetime.now(timezone.utc)
    for lag_h in range(1, 5):
        t = now - timedelta(hours=lag_h)
        url = (f"https://nomads.ncep.noaa.gov/dods/rtma2p5/"
               f"rtma2p5{t:%Y%m%d}/rtma2p5_anl_{t:%H}z")
        try:
            ds = Dataset(url)
        except Exception:
            continue
        try:
            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
            lon = np.where(lon > 180, lon - 360, lon)
            yi = np.where((lat >= bbox[1]) & (lat <= bbox[3]))[0]
            xi = np.where((lon >= bbox[0]) & (lon <= bbox[2]))[0]
            if len(yi) == 0 or len(xi) == 0:
                ds.close(); continue
            stride = max(1, len(yi) // 80, len(xi) // 80)
            u = _nomads_slice(ds, "ugrd10m", yi, xi, 0, stride)
            v = _nomads_slice(ds, "vgrd10m", yi, xi, 0, stride)
            tmp = _nomads_slice(ds, "tmp2m", yi, xi, 0, stride)
            try:
                gust = _nomads_slice(ds, "gustsfc", yi, xi, 0, stride)
            except Exception:
                gust = np.hypot(u, v) * 1.4
            try:
                dpt = _nomads_slice(ds, "dpt2m", yi, xi, 0, stride)
            except Exception:
                dpt = tmp - 10.0
            ds.close()
        except Exception as e:
            try: ds.close()
            except Exception: pass
            log.warning(f"RTMA read failed: {e}")
            continue
        ws = np.hypot(u, v) * 2.23694
        wd = (np.degrees(np.arctan2(-u, -v))) % 360.0
        es = 6.112 * np.exp(17.67 * (tmp - 273.15) / (tmp - 273.15 + 243.5))
        e  = 6.112 * np.exp(17.67 * (dpt - 273.15) / (dpt - 273.15 + 243.5))
        rh = np.clip(100.0 * e / es, 1, 100)
        out = {"lats": lat[yi[0]:yi[-1] + 1:stride].tolist(),
               "lons": lon[xi[0]:xi[-1] + 1:stride].tolist(),
               "wind_mph": ws.tolist(), "wind_dir": wd.tolist(),
               "gust_mph": (gust * 2.23694).tolist(),
               "temp_f": ((tmp - 273.15) * 9 / 5 + 32).tolist(),
               "rh": rh.tolist(),
               "valid": t.strftime("%Y-%m-%dT%H:00Z"), "source": "rtma"}
        cache_put(key, out)
        return out
    return None


def get_hrrr_forecast(bbox, hours=24):
    """
    HRRR surface forecast over bbox via NOMADS dods (netCDF4), hourly.
    Only UGRD/VGRD/TMP/RH-equivalent/GUST are read, bbox-sliced — never
    full files. Returns {"times": [...], "lats", "lons", per-hour fields}
    or None.
    """
    key = f"hrrr_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_h{hours}"
    cached = cache_get(key, "hrrr")
    if cached:
        return cached
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None
    now = datetime.now(timezone.utc)
    for lag_h in range(1, 7):
        t = now - timedelta(hours=lag_h)
        url = (f"https://nomads.ncep.noaa.gov/dods/hrrr/hrrr{t:%Y%m%d}/"
               f"hrrr_sfc.t{t:%H}z")
        try:
            ds = Dataset(url)
        except Exception:
            continue
        try:
            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
            lon = np.where(lon > 180, lon - 360, lon)
            yi = np.where((lat >= bbox[1]) & (lat <= bbox[3]))[0]
            xi = np.where((lon >= bbox[0]) & (lon <= bbox[2]))[0]
            if len(yi) == 0 or len(xi) == 0:
                ds.close(); continue
            stride = max(1, len(yi) // 60, len(xi) // 60)
            n_t = min(hours + 1, ds.variables["time"].shape[0])
            frames = []
            for k in range(n_t):
                u = _nomads_slice(ds, "ugrd10m", yi, xi, k, stride)
                v = _nomads_slice(ds, "vgrd10m", yi, xi, k, stride)
                tmp = _nomads_slice(ds, "tmp2m", yi, xi, k, stride)
                try:
                    rh = _nomads_slice(ds, "rh2m", yi, xi, k, stride)
                except Exception:
                    rh = np.full_like(tmp, 40.0)
                try:
                    gust = _nomads_slice(ds, "gustsfc", yi, xi, k, stride)
                except Exception:
                    gust = np.hypot(u, v) * 1.4
                frames.append({
                    "wind_mph": (np.hypot(u, v) * 2.23694).tolist(),
                    "wind_dir": ((np.degrees(np.arctan2(-u, -v))) % 360.0).tolist(),
                    "gust_mph": (gust * 2.23694).tolist(),
                    "temp_f": ((tmp - 273.15) * 9 / 5 + 32).tolist(),
                    "rh": np.clip(rh, 1, 100).tolist(),
                })
            ds.close()
            out = {"lats": lat[yi[0]:yi[-1] + 1:stride].tolist(),
                   "lons": lon[xi[0]:xi[-1] + 1:stride].tolist(),
                   "run": t.strftime("%Y-%m-%dT%H:00Z"),
                   "times": [(t + timedelta(hours=k)).strftime("%Y-%m-%dT%H:00Z")
                             for k in range(n_t)],
                   "frames": frames, "source": "hrrr"}
            cache_put(key, out)
            return out
        except Exception as e:
            try: ds.close()
            except Exception: pass
            log.warning(f"HRRR read failed: {e}")
    return None


# ── Synoptic RAWS (optional — placeholder key ⇒ graceful skip) ────────────────

def get_raws_obs(bbox):
    """Latest RAWS obs in bbox from Synoptic free tier. [] if no key/failure."""
    tok = CONFIG.get("synoptic_api_key", "")
    if not tok or tok.startswith("YOUR_"):
        log.info("Synoptic key not configured — RAWS source skipped")
        return []
    key = f"raws_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    cached = cache_get(key, "weather")
    if cached is not None:
        return cached
    q = urllib.parse.urlencode({
        "token": tok, "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "network": "2", "recent": "120",
        "vars": "air_temp,relative_humidity,wind_speed,wind_direction,fuel_moisture",
        "units": "english",
    })
    raw = http_get(f"https://api.synopticdata.com/v2/stations/timeseries?{q}",
                   timeout=45)
    if not raw:
        return []
    try:
        stations = json.loads(raw).get("STATION", [])
    except Exception:
        return []
    out = []
    for s in stations:
        obs = s.get("OBSERVATIONS", {})
        def last(varname):
            for k, v in obs.items():
                if k.startswith(varname) and isinstance(v, list) and v:
                    vals = [x for x in v if x is not None]
                    return vals[-1] if vals else None
            return None
        out.append({
            "id": s.get("STID"), "name": s.get("NAME"),
            "lat": float(s.get("LATITUDE", 0)), "lon": float(s.get("LONGITUDE", 0)),
            "temp_f": last("air_temp"), "rh": last("relative_humidity"),
            "wind_mph": last("wind_speed"), "wind_dir": last("wind_direction"),
            "fuel_moisture_pct": last("fuel_moisture"),
        })
    cache_put(key, out)
    return out


# ── FIRMS / WFIGS / MTBS / alerts ─────────────────────────────────────────────

_FIRMS_24H = [
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/"
    "USA_contiguous_and_Hawaii/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/"
    "USA_contiguous_and_Hawaii/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv",
]


def get_firms_24h(bbox=None):
    key = "firms_24h"
    rows = cache_get(key, "satellite")
    if rows is None:
        rows = []
        for url in _FIRMS_24H:
            raw = http_get(url, timeout=60)
            if not raw:
                continue
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))):
                try:
                    rows.append({
                        "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                        "frp": float(r.get("frp") or 0),
                        "datetime": f'{r["acq_date"]}T{int(r["acq_time"]):04d}'[:15],
                        "confidence": r.get("confidence", "n"),
                    })
                except (KeyError, ValueError):
                    continue
        cache_put(key, rows)
    if bbox:
        rows = [h for h in rows
                if bbox[0] <= h["lon"] <= bbox[2] and bbox[1] <= h["lat"] <= bbox[3]]
    return rows


def get_firms_archive(bbox, start_d, end_d):
    import sys
    if str(FPM_DIR) not in sys.path:
        sys.path.insert(0, str(FPM_DIR))
    try:
        from fire_analysis import fetch_hotspots, FIRMS_KEY
        return fetch_hotspots(FIRMS_KEY, bbox, start_d, end_d)
    except Exception as e:
        log.warning(f"FIRMS archive fetch failed: {e}")
        return []


_WFIGS_CURRENT = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/"
                  "services/WFIGS_Interagency_Perimeters_Current/"
                  "FeatureServer/0/query")

# CAL FIRE's own active-incidents map service (the one powering
# fire.ca.gov/incidents) — a combined layer of same-day FIRIS infrared
# aircraft-mapped perimeters plus whatever NIFC/WFIGS/USFS/county perimeters
# CAL FIRE folds in. Discovered from that map's network requests, not
# documented publicly, but unauthenticated and public. This is routinely
# MORE current than WFIGS Current: a state-managed initial-attack fire gets
# a FIRIS overflight same-day, while its WFIGS sync can lag hours to a day+.
_FIRIS_COMBO = ("https://bz1uwwpkuinzbk94.svcs5.arcgis.com/bz1uwWPKUInZBK94/"
               "arcgis/rest/services/CA_Perimeters_NIFC_FIRIS_public_view/"
               "FeatureServer/0/query")


def _geom_centroid(geometry):
    def flatten(c):
        if not c:
            return
        if isinstance(c[0], (int, float)):
            yield c
        else:
            for cc in c:
                yield from flatten(cc)
    pts = list(flatten((geometry or {}).get("coordinates")))
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _centroid_dist_km(a, b):
    return math.hypot((a[1] - b[1]) * 111.0,
                      (a[0] - b[0]) * 111.0 * math.cos(math.radians(a[1])))


def _mission_fire_name(mission):
    """CAL FIRE FIRIS mission codes embed the fire name as a middle segment,
    e.g. 'CA-PMQ-PIPELINE-N57B' -> 'Pipeline'. Best-effort label for rows
    with no incident_name yet (raw FIRIS overflights aren't always synced
    to an official incident record right away)."""
    if not mission:
        return None
    parts = mission.split("-")
    return (parts[2] if len(parts) >= 3 else mission).strip().title()


def get_firis_perimeters(bbox):
    """
    Same-day CAL FIRE FIRIS/NIFC/WFIGS/USFS/county perimeters from CAL
    FIRE's own active-incidents map service (see _FIRIS_COMBO). Multiple
    rows can describe the same incident — one per source agency or
    overflight pass — so rows within ~2km of each other are collapsed to
    one, keeping whichever has the newest poly_DateCurrent.
    Returns [{name, acres, geometry(GeoJSON dict)}], newest first.
    """
    key = f"firis_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    cached = cache_get(key, "firis")
    if cached is not None:
        return cached
    q = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope", "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "incident_name,area_acres,NIFC_GISAcres,mission,poly_DateCurrent",
        "outSR": "4326", "returnGeometry": "true", "f": "geojson",
        "resultRecordCount": 200,
    })
    raw = http_get(f"{_FIRIS_COMBO}?{q}", timeout=45)
    rows = []
    if raw:
        try:
            for f in json.loads(raw).get("features", []):
                p = f.get("properties") or {}
                if not f.get("geometry"):
                    continue
                acres = p.get("NIFC_GISAcres")
                if acres is None:
                    acres = p.get("area_acres")
                rows.append({
                    "name": (p.get("incident_name") or _mission_fire_name(p.get("mission"))
                            or "Unnamed").strip(),
                    "acres": float(acres or 0),
                    "updated": p.get("poly_DateCurrent") or 0,
                    "geometry": f["geometry"],
                    "centroid": _geom_centroid(f["geometry"]),
                })
        except Exception as e:
            log.warning(f"FIRIS combo perimeters parse failed: {e}")

    # collapse duplicate rows (same incident, different source/overflight)
    # by proximity — keep the most recently updated one per cluster
    groups = []
    for r in sorted(rows, key=lambda r: r["updated"], reverse=True):
        placed = False
        if r["centroid"]:
            for g in groups:
                if g["centroid"] and _centroid_dist_km(r["centroid"], g["centroid"]) < 2.0:
                    placed = True
                    break
        if not placed:
            groups.append(r)
    out = [{"name": r["name"], "acres": r["acres"], "geometry": r["geometry"]}
          for r in groups]
    cache_put(key, out)
    return out


def get_active_perimeters(bbox):
    """
    Currently-active fire perimeters intersecting bbox, FIRIS-first: CAL
    FIRE's same-day infrared-mapped perimeters (get_firis_perimeters) are
    prioritized over the WFIGS *Current* layer, since WFIGS routinely lags
    a fresh state-managed initial-attack fire by hours to a day+. A WFIGS
    Current perimeter is only added if FIRIS doesn't already cover that
    location (matched by centroid proximity) — e.g. federal-land incidents
    CAL FIRE's own map doesn't track.
    Returns [{name, acres, irwin_id, geometry(GeoJSON dict)}], biggest first.
    """
    firis = get_firis_perimeters(bbox)
    out = [{"name": r["name"], "acres": r["acres"], "irwin_id": None,
           "geometry": r["geometry"]} for r in firis]
    firis_centroids = [_geom_centroid(r["geometry"]) for r in firis]

    key = f"wfigs_cur_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    cached = cache_get(key, "perimeters")
    if cached is None:
        q = urllib.parse.urlencode({
            "where": "1=1",
            "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "attr_IncidentName,poly_GISAcres,attr_IrwinID",
            "outSR": "4326", "returnGeometry": "true", "f": "geojson",
            "resultRecordCount": 100, "orderByFields": "poly_GISAcres DESC",
        })
        raw = http_get(f"{_WFIGS_CURRENT}?{q}", timeout=60)
        cached = []
        if raw:
            try:
                for f in json.loads(raw).get("features", []):
                    p = f.get("properties") or {}
                    if not f.get("geometry"):
                        continue
                    cached.append({
                        "name": (p.get("attr_IncidentName") or "Unnamed").strip(),
                        "acres": float(p.get("poly_GISAcres") or 0),
                        "irwin_id": p.get("attr_IrwinID"),
                        "geometry": f["geometry"],
                    })
            except Exception as e:
                log.warning(f"WFIGS current perimeters parse failed: {e}")
        cache_put(key, cached)

    for w in cached:
        c = _geom_centroid(w["geometry"])
        if c and any(cc and _centroid_dist_km(c, cc) < 3.0 for cc in firis_centroids):
            continue   # FIRIS combo already has a fresher perimeter here
        out.append(w)
    out.sort(key=lambda r: r["acres"], reverse=True)
    return out


_NGFS_BASE = "https://fire.data.nesdis.noaa.gov/api/ogc/detections/collections"
_NGFS_COLLECTIONS = ["ngfs_schema.ngfs_detections_scene_east_conus",
                    "ngfs_schema.ngfs_detections_scene_west_conus"]


def get_goes19_hotspots(bbox, hours=12, start_dt=None, end_dt=None):
    """
    GOES-19 NGFS fire-detection points over bbox in the last `hours` hours —
    the same NESDIS OGC 'detections' service the main map's live NGFS layer
    renders as vector tiles, queried here via its OGC API Features /items
    endpoint (bbox + datetime range) instead of the tile pyramid. GOES is
    geostationary (~5 min revisit) so a single active flank accumulates many
    detections fast — this is a temporal-density signal ("how continuously
    hot has this spot been"), not a spatial-resolution one (each pixel is
    ~2km, far coarser than FIRMS/VIIRS's ~375m).

    Pass explicit `start_dt`/`end_dt` (datetime, any tzinfo) for an arbitrary
    historical window instead of "last `hours` hours from now" — e.g. a past
    fire's full lifetime. `hours` is ignored when both are given.

    Returns [{lat, lon, frp, acq_date_time, confidence, incident_name}],
    GOES-19 (not GOES-18/west) only, deduplicated across the two collections.
    """
    if start_dt is not None and end_dt is not None:
        start = start_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        key = f"ngfs19_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{start}_{end}"
    else:
        start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        key = f"ngfs19_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{hours}"
    # its own short TTL, not the 6h "satellite" one FIRMS uses — GOES-19 is
    # geostationary with a ~5min revisit and present-mode now treats it as
    # the trusted confirm-activity source, so a stale cache here directly
    # undermines that. A past historical window is immutable though, so cache
    # those forever (category="static") rather than expiring them.
    cached = cache_get(key, "static" if (start_dt and end_dt) else "goes19")
    if cached is not None:
        return cached
    historical = start_dt is not None and end_dt is not None
    # numberMatched can run into the tens of thousands for a month-long,
    # 500k-acre fire (single-page limit=2000 silently truncates otherwise) —
    # only worth paginating in historical mode; the live rolling window
    # ("present mode") stays a single fast request.
    max_pages = 25 if historical else 1
    seen, out = set(), []
    for coll in _NGFS_COLLECTIONS:
        url = f"{_NGFS_BASE}/{coll}/items?" + urllib.parse.urlencode({
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "datetime": f"{start}/{end}", "limit": 2000, "f": "json",
        })
        for _page in range(max_pages):
            raw = http_get(url, timeout=45)
            if not raw:
                break
            try:
                doc = json.loads(raw)
                for f in doc.get("features", []):
                    p = f.get("properties") or {}
                    if p.get("satellite") != "GOES-19":
                        continue
                    fid = p.get("id") or f.get("id")
                    if fid in seen:
                        continue
                    seen.add(fid)
                    out.append({
                        "lat": p.get("latitude"), "lon": p.get("longitude"),
                        "frp": p.get("frp"), "acq_date_time": p.get("acq_date_time"),
                        "confidence": p.get("confidence"),
                        "incident_name": p.get("known_incident_name"),
                    })
            except Exception as e:
                log.warning(f"GOES-19 NGFS parse failed ({coll}): {e}")
                break
            next_url = next((l["href"] for l in doc.get("links", []) if l.get("rel") == "next"), None)
            if not next_url or len(doc.get("features", [])) < 2000:
                break
            url = next_url
    cache_put(key, out)
    return out


def get_wfigs_perimeters(fire_name, year=None, state=None):
    """Real WFIGS/GeoMAC perimeter snapshots for a historical fire. Cached
    forever (category="static") -- unlike every other caller of this data,
    a contained historical fire's perimeter history never changes, but the
    underlying fetch_perimeters() call has no cache of its own and can take
    2-30+ seconds (up to 4 sequential ArcGIS REST calls), so every uncached
    call -- including repeat calls within a single run_historical() -- was
    paying that cost live every time."""
    key = f"wfigs_perims_{fire_name.upper().replace(' ', '_')}_{year}_{state}"
    cached = cache_get(key, category="static", ext="json")
    if cached is not None:
        return cached
    import sys
    if str(FPM_DIR) not in sys.path:
        sys.path.insert(0, str(FPM_DIR))
    from fire_analysis import fetch_perimeters
    gj = fetch_perimeters(fire_name, year=year, state=state)
    if gj.get("features"):
        cache_put(key, gj, ext="json")
    return gj


# CAL FIRE FRAP "California Historic Fire Perimeters" — the only public
# fire-history service that carries a COMPLEX_NAME grouping field, so it is the
# only one from which a *complex-level* (not sub-incident) perimeter can be
# reconstructed by a principled query. MTBS (get_mtbs_points and the per-year
# "Burned Area Boundaries" polygon layers) maps every constituent fire
# separately with no complex linkage — e.g. it carries HENNESSEY / WALBRIDGE /
# MEYERS as three independent 2020 burns with no field tying them back to the
# LNU Lightning Complex — so MTBS cannot answer "give me the whole complex."
_FRAP_HISTORIC = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/"
                  "services/California_Historic_Fire_Perimeters/"
                  "FeatureServer/0/query")


def get_complex_perimeter_frap(complex_name, year=None, state="CA"):
    """Opt-in alternative ground-truth source: the full perimeter of a named
    fire *complex* from CAL FIRE FRAP's California Historic Fire Perimeters.

    Motivation: WFIGS/GeoMAC (get_wfigs_perimeters) keys on an individual
    incident name, so for a lightning *complex* it frequently resolves to a
    single small constituent sub-fire rather than the whole burn (the LNU
    Lightning Complex, for instance, resolves to the ~2.4k-acre MEYERS fire
    instead of the ~363k-acre complex). FRAP is the only public service that
    tags each constituent fire with the COMPLEX_NAME it belonged to, so
    unioning every polygon that shares that exact complex name reconstructs
    the true complex-scale perimeter.

    Selection logic (principled — no acreage number is hardcoded anywhere):
      * exact, case-insensitive match on COMPLEX_NAME (an `=` test, not a LIKE
        substring — that avoids collisions such as "SOUTHERN LNU COMPLEX" /
        "CENTRAL LNU COMPLEX" vs "LNU LIGHTNING COMPLEX"),
      * optional YEAR_ filter to disambiguate complexes that reuse a name in
        different years,
      * optional STATE filter (FRAP is California-only, but the field exists),
      * every matching polygon is returned; the caller unions them. The summed
        GIS_ACRES of the returned features is a built-in sanity signal that the
        match is complex-scale rather than one stray sub-fire.

    Returns a GeoJSON FeatureCollection (one feature per constituent fire, with
    FIRE_NAME / YEAR_ / GIS_ACRES / COMPLEX_NAME / ALARM_DATE / CONT_DATE
    properties) in EPSG:4326, or an empty FeatureCollection if nothing matches.
    Cached forever (category="static") — a contained historical complex's
    perimeter never changes.

    NOT wired into validation.py or any default code path: this is an
    available, unused-by-default alternative for callers that specifically
    want a complex-level ground truth.
    """
    key = (f"frap_complex_{complex_name.upper().replace(' ', '_')}"
           f"_{year}_{state}")
    cached = cache_get(key, category="static", ext="json")
    if cached is not None:
        return cached
    empty = {"type": "FeatureCollection", "features": []}
    where = f"UPPER(COMPLEX_NAME) = '{complex_name.upper()}'"
    if year:
        where += f" AND YEAR_ = {int(year)}"
    if state:
        where += f" AND UPPER(STATE) = '{state.upper()}'"
    q = urllib.parse.urlencode({
        "where": where,
        "outFields": "FIRE_NAME,YEAR_,GIS_ACRES,COMPLEX_NAME,ALARM_DATE,CONT_DATE",
        "returnGeometry": "true", "outSR": "4326", "f": "geojson",
    })
    try:
        raw = http_get(f"{_FRAP_HISTORIC}?{q}", timeout=90)
        if not raw:
            return empty
        gj = json.loads(raw)
    except Exception as e:
        log.warning(f"FRAP complex-perimeter query failed "
                    f"({complex_name} {year}): {e}")
        return empty
    if not gj.get("features"):
        return empty
    gj.setdefault("type", "FeatureCollection")
    cache_put(key, gj, ext="json")
    return gj


_MTBS_PTS = ("https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/"
             "MapServer/62/query")


def get_mtbs_points(bbox, min_year=1984):
    """MTBS ignition points in bbox (layer 62 has NO `year` field — filter
    client-side on ig_date epoch-ms)."""
    key = f"mtbs_pts_{bbox[0]:.1f}_{bbox[1]:.1f}_{bbox[2]:.1f}_{bbox[3]:.1f}_{min_year}"
    cached = cache_get(key, "static")
    if cached is not None:
        return cached
    pts, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1",
            "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "fire_id,fire_name,ig_date,acres,latitude,longitude",
            "f": "json", "resultRecordCount": 2000, "resultOffset": offset,
            "outSR": "4326",
        })
        raw = http_get(f"{_MTBS_PTS}?{q}", timeout=90)
        if not raw:
            break
        d = json.loads(raw)
        feats = d.get("features", [])
        for f in feats:
            a = f.get("attributes", {})
            g = f.get("geometry") or {}
            lon = a.get("longitude") if a.get("longitude") is not None else g.get("x")
            lat = a.get("latitude") if a.get("latitude") is not None else g.get("y")
            if lon is None or lat is None:
                continue
            ig = a.get("ig_date")
            year = None
            if isinstance(ig, (int, float)) and ig > 1e9:
                year = datetime.fromtimestamp(ig / 1000, tz=timezone.utc).year
            if year is not None and year < min_year:
                continue
            pts.append({**a, "year": year, "lon": lon, "lat": lat})
        if len(feats) < 2000 and not d.get("exceededTransferLimit"):
            break
        offset += len(feats) or 2000
    if pts:
        cache_put(key, pts)
    return pts


def get_red_flag_geoms():
    key = "red_flag"
    d = cache_get(key, "alerts")
    if d is None:
        raw = http_get("https://api.weather.gov/alerts/active?event=Red%20Flag%20Warning",
                       timeout=45, headers={"Accept": "application/geo+json"})
        d = json.loads(raw).get("features", []) if raw else []
        cache_put(key, d)
    from shapely.geometry import shape
    geoms = []
    for f in d:
        g = f.get("geometry")
        if g:
            try:
                geoms.append(shape(g))
            except Exception:
                continue
    return geoms


# ── Ignition point resolution chain ───────────────────────────────────────────

def _arcgis_name_query(url, name_field, fire_name, year=None, year_field=None,
                       extra_where="", out_fields="*"):
    where = f"UPPER({name_field}) LIKE '%{fire_name.upper()}%'"
    if year and year_field:
        where += f" AND {year_field}={year}"
    if extra_where:
        where += f" AND {extra_where}"
    q = urllib.parse.urlencode({
        "where": where, "outFields": out_fields, "f": "json",
        "resultRecordCount": 10, "outSR": "4326",
    })
    raw = http_get(f"{url}?{q}", timeout=45)
    if not raw:
        return []
    try:
        return json.loads(raw).get("features", [])
    except Exception:
        return []


def _pick_latlon(attrs, geom=None):
    """Find lat/lon in arbitrary ArcGIS attribute dicts."""
    lat = lon = None
    for k, v in attrs.items():
        kl = k.lower()
        if v is None:
            continue
        if lat is None and ("latitude" in kl or kl in ("lat", "y")):
            try: lat = float(v)
            except (TypeError, ValueError): pass
        if lon is None and ("longitude" in kl or kl in ("lon", "lng", "x")):
            try: lon = float(v)
            except (TypeError, ValueError): pass
    if (lat is None or lon is None) and geom:
        lat = lat if lat is not None else geom.get("y")
        lon = lon if lon is not None else geom.get("x")
    if lat and lon and -90 <= lat <= 90 and -180 <= lon <= 0:
        return lat, lon
    return None


def resolve_ignition(fire_name, year=None, state=None, fire_key=None):
    """
    Resolve a historical fire's ignition point. Sources in priority order:
      1 curated historical_fires.json   2 IRWIN   3 WFIGS discovery point
      4 CAL FIRE discovery fields       5 InciWeb 6 MTBS ignition point
      7 FIRMS earliest-detection centroid (last resort, logged WARNING)
    Returns {lat, lon, datetime, source, confidence} or None.
    Result cached to sim_cache/ignitions/{name}_{year}.json.

    fire_key: exact historical_fires.json key (e.g. "CAMP_2018"), if the
    caller already has it (the /api/sim/fires dropdown does). Without it,
    step 1 falls back to a normalized guess from fire_name+year, which
    fails for anything with "Fire"/"Complex" in the display name (e.g.
    "Woolsey Fire" -> normalized "WOOLSEY" vs raw slug "WOOLSEY FIRE") —
    so always pass fire_key when it's available.
    """
    slug = f"{fire_name.upper().strip()}_{year or 'X'}"
    cpath = IGN_DIR / f"{''.join(c if c.isalnum() or c in '._-' else '_' for c in slug)}.json"
    if cpath.exists():
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _done(res):
        atomic_write_json(cpath, res)
        return res

    # 1 — curated index: exact fire_key first, then raw slug, then a
    # normalized slug with FIRE/COMPLEX/trailing punctuation stripped
    try:
        idx = json.loads((SIM_DIR / "historical_fires.json").read_text(encoding="utf-8"))
        fires = idx.get("fires", {})
        # strip FIRE/COMPLEX from the NAME before appending _year — "_" is a
        # word char, so matching against the finished slug ("CALDOR
        # FIRE_2021") never finds a \b after FIRE and silently misses
        name_norm = re.sub(r"\s+(FIRE|COMPLEX|INCIDENT)\s*$", "",
                           fire_name.upper().strip())
        norm_slug = f"{name_norm}_{year or 'X'}"
        hit = (fires.get(fire_key) if fire_key else None) \
            or fires.get(slug) or fires.get(norm_slug)
        if hit:
            return _done({"lat": hit["lat"], "lon": hit["lon"],
                          "datetime": hit["datetime"], "source": "curated_index",
                          "confidence": "verified"})
    except Exception as e:
        log.warning(f"historical_fires.json read failed: {e}")

    # 2 — IRWIN
    try:
        feats = _arcgis_name_query(CONFIG["irwin_api_base"], "IncidentName",
                                   fire_name)
        for f in feats:
            a = f.get("attributes", {})
            fy = a.get("FireDiscoveryDateTime")
            if year and isinstance(fy, (int, float)) and fy > 1e9:
                if datetime.fromtimestamp(fy / 1000, tz=timezone.utc).year != year:
                    continue
            ll = _pick_latlon(a, f.get("geometry"))
            if ll:
                dt = (datetime.fromtimestamp(fy / 1000, tz=timezone.utc).isoformat()
                      if isinstance(fy, (int, float)) and fy > 1e9 else None)
                return _done({"lat": ll[0], "lon": ll[1], "datetime": dt,
                              "source": "irwin", "confidence": "high"})
    except Exception as e:
        log.warning(f"IRWIN query failed: {e}")

    # 3 — WFIGS incident locations (discovery points)
    try:
        url = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
               "WFIGS_Incident_Locations/FeatureServer/0/query")
        feats = _arcgis_name_query(url, "IncidentName", fire_name)
        for f in feats:
            a = f.get("attributes", {})
            fy = a.get("FireDiscoveryDateTime")
            if year and isinstance(fy, (int, float)) and fy > 1e9:
                if datetime.fromtimestamp(fy / 1000, tz=timezone.utc).year != year:
                    continue
            ll = _pick_latlon(a, f.get("geometry"))
            if ll:
                dt = (datetime.fromtimestamp(fy / 1000, tz=timezone.utc).isoformat()
                      if isinstance(fy, (int, float)) and fy > 1e9 else None)
                return _done({"lat": ll[0], "lon": ll[1], "datetime": dt,
                              "source": "wfigs_discovery", "confidence": "high"})
    except Exception as e:
        log.warning(f"WFIGS discovery query failed: {e}")

    # 4 — CAL FIRE incidents
    try:
        url = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
               "California_Fire_Perimeters/FeatureServer/0/query")
        feats = _arcgis_name_query(url, "FIRE_NAME", fire_name,
                                   year=year, year_field="YEAR_")
        for f in feats:
            ll = _pick_latlon(f.get("attributes", {}), f.get("geometry"))
            if ll:
                return _done({"lat": ll[0], "lon": ll[1], "datetime": None,
                              "source": "calfire", "confidence": "medium"})
    except Exception as e:
        log.warning(f"CAL FIRE query failed: {e}")

    # 5 — InciWeb
    try:
        raw = http_get(CONFIG["inciweb_api_base"]
                       + "?" + urllib.parse.urlencode({"search": fire_name}),
                       timeout=30)
        if raw:
            items = json.loads(raw)
            items = items.get("data", items) if isinstance(items, dict) else items
            for it in (items or []):
                if not isinstance(it, dict):
                    continue
                ll = _pick_latlon(it)
                if ll:
                    return _done({"lat": ll[0], "lon": ll[1],
                                  "datetime": it.get("date_of_origin"),
                                  "source": "inciweb", "confidence": "medium"})
    except Exception as e:
        log.warning(f"InciWeb query failed: {e}")

    # 6 — MTBS ignition point
    try:
        feats = _arcgis_name_query(_MTBS_PTS, "fire_name", fire_name)
        for f in feats:
            a = f.get("attributes", {})
            ig = a.get("ig_date")
            if year and isinstance(ig, (int, float)) and ig > 1e9:
                if datetime.fromtimestamp(ig / 1000, tz=timezone.utc).year != year:
                    continue
            ll = _pick_latlon(a, f.get("geometry"))
            if ll:
                dt = (datetime.fromtimestamp(ig / 1000, tz=timezone.utc).isoformat()
                      if isinstance(ig, (int, float)) and ig > 1e9 else None)
                return _done({"lat": ll[0], "lon": ll[1], "datetime": dt,
                              "source": "mtbs", "confidence": "medium"})
    except Exception as e:
        log.warning(f"MTBS ignition query failed: {e}")

    # 7 — FIRMS earliest-detection centroid (last resort)
    log.warning(f"resolve_ignition({fire_name} {year}): all authoritative "
                "sources failed — falling back to FIRMS earliest-detection "
                "centroid (low confidence)")
    try:
        gj = get_wfigs_perimeters(fire_name, year=year, state=state)
        feats = gj.get("features", [])
        if feats:
            from shapely.geometry import shape
            from shapely.ops import unary_union
            u = unary_union([shape(f["geometry"]) for f in feats if f.get("geometry")])
            minx, miny, maxx, maxy = u.bounds
            start_d = date(year, 1, 1) if year else date.today() - timedelta(days=30)
            end_d = date(year, 12, 31) if year else date.today()
            hs = get_firms_archive((minx, miny, maxx, maxy), start_d, end_d)
            hs = sorted(hs, key=lambda h: h.get("datetime", ""))
            if hs:
                first = hs[:20]
                return _done({"lat": float(np.median([h["lat"] for h in first])),
                              "lon": float(np.median([h["lon"] for h in first])),
                              "datetime": hs[0].get("datetime"),
                              "source": "firms_fallback", "confidence": "low"})
    except Exception as e:
        log.warning(f"FIRMS fallback failed: {e}")
    return None


# ── Derived meteorology ───────────────────────────────────────────────────────

def dead_fuel_moisture_1h(temp_f, rh_pct):
    """Fosberg & Deeming (1971) EMC → 1-hr dead FM fraction. Vectorized."""
    T = np.asarray(temp_f, dtype=np.float64)
    H = np.asarray(rh_pct, dtype=np.float64)
    emc = np.where(
        H < 10, 0.03229 + 0.281073 * H - 0.000578 * H * T,
        np.where(H < 50, 2.22749 + 0.160107 * H - 0.014784 * T,
                 21.0606 + 0.005565 * H ** 2 - 0.00035 * H * T - 0.483199 * H))
    return np.clip(1.03 * emc, 1.0, 35.0) / 100.0


def vpd_kpa(temp_c, rh_pct):
    T = np.asarray(temp_c, dtype=np.float64)
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    return np.clip(es * (1.0 - np.asarray(rh_pct, dtype=np.float64) / 100.0), 0, None)


def kbdi_series(temps_f_daily, precip_in_daily, annual_rain_in=25.0):
    Q, out, wet_run = 400.0, [], 0.0
    for T, P in zip(temps_f_daily, precip_in_daily):
        net = 0.0
        if P and P > 0:
            wet_run += P
            net = max(0.0, wet_run - 0.2) if wet_run > 0.2 else 0.0
            net = min(net, P)
        else:
            wet_run = 0.0
        Q = max(0.0, Q - net * 100.0)
        if T and T > 50:
            dQ = ((800.0 - Q) * (0.968 * math.exp(0.0486 * T) - 8.30)
                  / (1.0 + 10.88 * math.exp(-0.0441 * annual_rain_in))) * 1e-3
            Q = min(800.0, Q + max(0.0, dQ))
        out.append(Q)
    return out


def kbdi_to_fm1h(kbdi):
    """Monotonic KBDI(0–800) → dead 1-hr FM fraction (0.25 wet → 0.04 drought)."""
    k = np.clip(np.asarray(kbdi, dtype=np.float64), 0, 800)
    return 0.04 + (1.0 - k / 800.0) * 0.21


def kbdi_for_bbox(bbox, end_date=None):
    """
    Real KBDI at the bbox center: Open-Meteo 30-day daily max-temp/precip →
    kbdi_series, last value. Shared by spread_engine._kbdi_center and
    sim_data.fetch_fuel — the latter used to pass kbdi_val=None (neutral
    300 → a uniform 17.1% dead FM, above every grass fuel's 15% moisture of
    extinction, which made grass fires in the /simulator engine unable to
    spread at all). Returns 300.0 on any failure. Cached 60 min per
    bbox+date.
    """
    end = end_date or date.today()
    key = f"kbdi_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{end}"
    cached = cache_get(key, category="weather", ext="json")
    if cached is not None:
        return float(cached["kbdi"])
    try:
        clat = (bbox[1] + bbox[3]) / 2
        clon = (bbox[0] + bbox[2]) / 2
        base = _OM_HIST if end >= date(2016, 1, 31) else _OM_ERA5
        res = om_multi(base, [(clat, clon)], {
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "UTC", "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "start_date": str(end - timedelta(days=30)),
            "end_date": str(end - timedelta(days=1)),
        })
        dd = res[0]["daily"]
        k = kbdi_series([t or 70 for t in dd["temperature_2m_max"]],
                        [p or 0 for p in dd["precipitation_sum"]],
                        CONFIG["danger_norms"]["kbdi_annual_rain_in"])[-1]
        cache_put(key, {"kbdi": float(k)})
        return float(k)
    except Exception as e:
        log.warning(f"KBDI fetch failed ({e}) — neutral 300")
        return 300.0


# ── Fused fuel moisture grid (RAWS → KBDI → NDMI) ─────────────────────────────

def fuel_moisture_grid(bbox, lats, lons, kbdi_val=None):
    """
    1-hr dead fuel moisture fraction on the grid, fused per spec:
      RAWS IDW within raws_radius_km  →  KBDI-derived  →  NDMI raster.
    kbdi_val: scalar or 2D grid of KBDI; if None, a neutral 300 is used.
    Returns (fm_grid, sources_used list).
    """
    ny, nx = len(lats), len(lons)
    sources = []
    kb = np.full((ny, nx), 300.0) if kbdi_val is None else \
        (np.full((ny, nx), float(kbdi_val)) if np.isscalar(kbdi_val) else np.asarray(kbdi_val))
    fm = kbdi_to_fm1h(kb)
    sources.append("kbdi")

    # NDMI refinement where a raster exists
    ndmi_tif = CACHE_DIR / "ndmi.tif"
    if ndmi_tif.exists():
        try:
            import rasterio
            with rasterio.open(ndmi_tif) as src:
                gy, gx = np.meshgrid(lats, lons, indexing="ij")
                rows, cols = raster_rowcol(src, gx.ravel(), gy.ravel())
                rows = np.clip(rows, 0, src.height - 1)
                cols = np.clip(cols, 0, src.width - 1)
                nd = src.read(1)[rows, cols].reshape(gy.shape).astype(np.float64)
            fm_ndmi = np.clip(0.03 + (nd + 1) / 2 * 0.25, 0.02, 0.35)
            fm = 0.5 * fm + 0.5 * fm_ndmi
            sources.append("ndmi")
        except Exception as e:
            log.warning(f"NDMI read failed: {e}")

    # RAWS IDW overrides where stations are close enough
    stations = [s for s in get_raws_obs(buffer_bbox(bbox, km=75))
                if s.get("fuel_moisture_pct") is not None
                or (s.get("temp_f") is not None and s.get("rh") is not None)]
    if stations:
        sources.append("raws")
        radius = CONFIG["raws_radius_km"]
        gy, gx = np.meshgrid(lats, lons, indexing="ij")
        num = np.zeros((ny, nx)); den = np.zeros((ny, nx))
        for s in stations:
            if s["fuel_moisture_pct"] is not None:
                s_fm = float(s["fuel_moisture_pct"]) / 100.0
            else:
                s_fm = float(dead_fuel_moisture_1h(
                    np.array([s["temp_f"]]), np.array([s["rh"]]))[0])
            d_km = np.hypot((gy - s["lat"]) * 111.0,
                            (gx - s["lon"]) * 111.0 * math.cos(math.radians(s["lat"])))
            w = np.where(d_km < radius, 1.0 / np.maximum(d_km, 0.5) ** 2, 0.0)
            num += w * s_fm
            den += w
        mask = den > 0
        fm[mask] = num[mask] / den[mask]
    return np.clip(fm, 0.02, 0.35), sources
