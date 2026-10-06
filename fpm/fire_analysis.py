#!/usr/bin/env python3
"""
Fire History Analyzer
Generates an interactive HTML report for any named wildfire.

Usage:
  python fire_analysis.py CAMP   --year 2018 --state CA
  python fire_analysis.py CALDOR --year 2021 --state CA --key YOUR_FIRMS_KEY
  python fire_analysis.py DIXIE  --year 2021 --state CA

NASA FIRMS key (free, adds hotspot layer):
  1. Go to https://firms.modaps.eosdis.nasa.gov/api/
  2. Click Get MAP_KEY
  3. Run with:  --key YOUR_KEY_HERE
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

import time, base64, io, argparse, math, json, re
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# -- auto-install missing packages --------------------------------------------
def _ensure(*pkgs):
    import subprocess
    for pkg in pkgs:
        mod = pkg.replace("-", "_")
        try:
            __import__(mod)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                           check=True)

_ensure("folium", "matplotlib")

import folium
from folium.plugins import TimestampedGeoJson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -- Config -------------------------------------------------------------------
FIRMS_KEY = os.environ.get("FIRMS_MAP_KEY", "")   # NASA FIRMS MAP_KEY (set FIRMS_MAP_KEY)

# -- NIFC fire perimeter services ---------------------------------------------
# Daily perimeter snapshots (best source for recent fires)
WFIGS_DAILY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Daily_Perimeters_Public/FeatureServer/0/query"
)
# Consolidated perimeters fallback (recent fires 2019-present)
WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)
# Year-specific GeoMAC services (2000-2019): have DAILY operational snapshots
# Use these instead of the combined service which only has final perimeters
def _geomac_url(year):
    name = f"Historic_Geomac_Perimeters_{year}"
    if year == 2019:
        name = "Historic_GeoMAC_Perimeters_2019"
    return (
        f"https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        f"{name}/FeatureServer/0/query"
    )

def _archive_url(year):
    # Layer numbers by year (verify at NIFC's ArcGIS if a new year breaks)
    layer = {2020: 6, 2021: 6, 2022: 7, 2023: 7, 2024: 8, 2025: 8}.get(year, 8)
    return (
        f"https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        f"Operational_Data_Archive_{year}/FeatureServer/{layer}/query"
    )


def _parse_date(v):
    if isinstance(v, (int, float)) and v > 1e9:
        return datetime.fromtimestamp(v / 1000)
    if isinstance(v, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(v[:19], fmt)
            except ValueError:
                pass
    return None


def _norm(p, source):
    return {
        "poly_IncidentName":          p.get("poly_IncidentName") or p.get("incidentname"),
        "poly_GISAcres":              p.get("poly_GISAcres")     or p.get("gisacres"),
        "poly_DateCurrent":           p.get("poly_DateCurrent")  or p.get("datecurrent"),
        "poly_PolygonDateTime":       p.get("poly_PolygonDateTime") or p.get("perimeterdatetime"),
        "attr_PercentContained":      p.get("attr_PercentContained"),
        "attr_PredominantFuelModel":  p.get("attr_PredominantFuelModel"),
        "attr_PrimaryFuelModel":      p.get("attr_PrimaryFuelModel"),
        "attr_FireDiscoveryDateTime": p.get("attr_FireDiscoveryDateTime"),
        "attr_ContainmentDateTime":   p.get("attr_ContainmentDateTime"),
        "attr_ControlDateTime":       p.get("attr_ControlDateTime"),
        "_source": source,
    }


def fetch_perimeters(fire_name, year=None, state=None):
    """
    Fetch daily perimeter snapshots:
      2000-2019  -> year-specific GeoMAC  (true daily operational snapshots)
      2020+      -> WFIGS consolidated to get IRWIN ID, then WFIGS Daily by ID
      2020-2021  -> no daily source; returns final perimeter only
    """
    clean    = fire_name.upper().replace(" FIRE", "").strip()
    features = []

    # GeoMAC year-specific (2000-2019)
    if year and 2000 <= year <= 2019:
        try:
            where = f"incidentname LIKE '%{clean}%'"
            if state:
                where += f" AND state = '{state.upper()}'"
            r = requests.get(_geomac_url(year), params={
                "where": where, "outFields": "*", "f": "geojson",
                "resultRecordCount": 500, "orderByFields": "perimeterdatetime ASC",
            }, timeout=20)
            for feat in r.json().get("features", []):
                feat["properties"] = _norm(feat.get("properties", {}), "geomac")
                features.append(feat)
            print(f"     (GeoMAC {year}: {len(features)} snapshots)")
        except Exception as e:
            print(f"  GeoMAC error: {e}")
        return {"type": "FeatureCollection", "features": features}

    # ── Step 1: WFIGS consolidated — finds IRWIN ID for most fires ───────────
    irwin_id = None
    try:
        where = f"poly_IncidentName LIKE '%{clean}%'"
        if state:
            where += f" AND attr_POOState LIKE '%{state.upper()}%'"
        r = requests.get(WFIGS_URL, params={
            "where": where, "outFields": "*", "f": "geojson",
            "resultRecordCount": 100, "orderByFields": "poly_GISAcres DESC",
        }, timeout=20)
        # Filter out tiny incidents: if searching for a major fire name (year <= 2015
        # or "Fire"/"Complex" in original name), skip matches under 100 acres to avoid
        # picking up small incidents with similar names (e.g., "Camp 8" vs "Camp Fire").
        min_acres = 100 if (not year or year <= 2015 or "FIRE" in fire_name.upper() or "COMPLEX" in fire_name.upper()) else 10
        for feat in r.json().get("features", []):
            p = feat.get("properties", {})
            acres = p.get("poly_GISAcres")
            if acres is not None and isinstance(acres, (int, float)):
                if acres < min_acres:
                    continue  # Skip tiny incidents
            if year:
                dt_cur  = _parse_date(p.get("poly_DateCurrent"))
                dt_disc = _parse_date(p.get("attr_FireDiscoveryDateTime"))
                # Accept if the perimeter filing date OR the actual discovery
                # date matches — some fires are archived years after the incident.
                if not any(dt and dt.year == year for dt in [dt_cur, dt_disc]):
                    continue
            irwin_id = p.get("poly_IRWINID")
            feat["properties"] = _norm(p, "wfigs")
            features.append(feat)
            break   # largest matching fire (after filtering)
        if irwin_id:
            print(f"     (WFIGS consolidated: IRWIN={irwin_id}, acres={p.get('poly_GISAcres')})")
        else:
            print(f"     (WFIGS consolidated: no match for '{clean}' (min {min_acres} acres))")
    except Exception as e:
        print(f"  WFIGS consolidated error: {e}")

    # ── Step 2: WFIGS Daily by IRWIN ID (if we found one) ────────────────────
    if irwin_id:
        try:
            r = requests.get(WFIGS_DAILY_URL, params={
                "where": f"poly_IRWINID = '{irwin_id}'",
                "outFields": "*", "f": "geojson",
                "resultRecordCount": 500, "orderByFields": "poly_DateCurrent ASC",
            }, timeout=20)
            daily = r.json().get("features", [])
            for feat in daily:
                feat["properties"] = _norm(feat.get("properties", {}), "wfigs_daily")
                features.append(feat)
            if daily:
                print(f"     (WFIGS Daily by IRWIN: +{len(daily)} snapshots)")
        except Exception as e:
            print(f"  WFIGS Daily error: {e}")

    # ── Step 2b: WFIGS Daily by NAME — catches fires missing from consolidated ─
    # Recent / smaller fires often live only in WFIGS Daily, not the consolidated
    # service.  Search directly so we don't miss fires like Eaton 2025.
    if not features:
        try:
            where = f"poly_IncidentName LIKE '%{clean}%'"
            if state:
                where += f" AND attr_POOState LIKE '%{state.upper()}%'"
            if year:
                where += f" AND poly_DateCurrent >= DATE '{year}-01-01'"
                where += f" AND poly_DateCurrent <  DATE '{year+1}-01-01'"
            r = requests.get(WFIGS_DAILY_URL, params={
                "where": where, "outFields": "*", "f": "geojson",
                "resultRecordCount": 500, "orderByFields": "poly_DateCurrent ASC",
            }, timeout=25)
            daily = r.json().get("features", [])
            # Filter by acreage like Step 1: if major fire, skip tiny matches
            min_acres = 100 if (not year or year <= 2015 or "FIRE" in fire_name.upper() or "COMPLEX" in fire_name.upper()) else 10
            for feat in daily:
                p = feat.get("properties", {})
                acres = p.get("poly_GISAcres")
                if acres is not None and isinstance(acres, (int, float)):
                    if acres < min_acres:
                        continue  # Skip tiny incidents
                if not irwin_id:
                    irwin_id = p.get("poly_IRWINID")
                feat["properties"] = _norm(p, "wfigs_daily")
                features.append(feat)
            if features:
                print(f"     (WFIGS Daily by name: {len(features)} snapshots, IRWIN={irwin_id})")
            else:
                print(f"     (WFIGS Daily by name: no match — try a different name/year)")
        except Exception as e:
            print(f"  WFIGS Daily name-search error: {e}")

    # ── Step 3: Operational Data Archive (2020+) ──────────────────────────────
    # Try multiple layer numbers in case NIFC changes them for a new year
    if irwin_id and year and year >= 2020:
        arch_found = False
        for layer_try in [_archive_url(year)] + [
            f"https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            f"Operational_Data_Archive_{year}/FeatureServer/{l}/query"
            for l in [6, 7, 8, 9] if l != int(_archive_url(year).split("/FeatureServer/")[1].split("/")[0])
        ]:
            try:
                r = requests.get(layer_try, params={
                    "where": (f"IRWINID = '{irwin_id}'"
                              f" AND FeatureCategory = 'Wildfire Daily Fire Perimeter'"),
                    "outFields": "*", "f": "geojson",
                    "resultRecordCount": 500, "orderByFields": "DateCurrent ASC",
                }, timeout=25)
                arch = r.json().get("features", [])
                if arch:
                    for feat in arch:
                        p = feat.get("properties", {})
                        feat["properties"] = {
                            "poly_IncidentName":         p.get("IncidentName"),
                            "poly_GISAcres":             p.get("GISAcres"),
                            "poly_DateCurrent":          p.get("DateCurrent"),
                            "poly_PolygonDateTime":      p.get("PolygonDateTime"),
                            "attr_PercentContained":     None,
                            "attr_PredominantFuelModel": None,
                            "attr_PrimaryFuelModel":     None,
                            "_source": "archive",
                        }
                        features.append(feat)
                    print(f"     (Op Archive {year}: +{len(arch)} snapshots)")
                    arch_found = True
                    break
            except Exception:
                pass
        if not arch_found:
            print(f"     (Op Archive {year}: no snapshots found across tried layers)")

    return {"type": "FeatureCollection", "features": features}


# -- NASA FIRMS hotspots ------------------------------------------------------
def fetch_hotspots(map_key, bbox, start, end):
    """Fetch VIIRS + MODIS hotspots in 5-day chunks.
    Uses SP (archive) products for historical dates, adds NRT products when
    the end date is within the last 14 days (FIRMS SP lags ~2 weeks)."""
    base     = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    area_str = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    products = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]
    # For very recent fires, SP data may not be processed yet — also try NRT
    if (date.today() - end).days <= 14:
        products += ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
    seen     = set()
    all_pts  = []

    for product in products:
        cur = start
        while cur <= end:
            ndays = min((end - cur).days + 1, 5)   # API max is 5
            url   = f"{base}/{map_key}/{product}/{area_str}/{ndays}/{cur.strftime('%Y-%m-%d')}"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and r.text.strip():
                    lines   = r.text.strip().split("\n")
                    headers = lines[0].split(",")
                    for line in lines[1:]:
                        vals = line.split(",")
                        if len(vals) < len(headers):
                            continue
                        row = dict(zip(headers, vals))
                        try:
                            lat = float(row["latitude"])
                            lon = float(row["longitude"])
                            t   = row.get("acq_time", "0000").zfill(4)
                            dt  = f"{row['acq_date']}T{t[:2]}:{t[2:]}:00"
                            key = (round(lat, 3), round(lon, 3), dt)
                            if key not in seen:
                                seen.add(key)
                                all_pts.append({
                                    "lat":      lat,
                                    "lon":      lon,
                                    "frp":      float(row.get("frp", 0)),
                                    "datetime": dt,
                                })
                        except (KeyError, ValueError):
                            pass
            except Exception as e:
                print(f"  FIRMS {product} chunk {cur}: {e}")
            cur += timedelta(days=5)
            time.sleep(0.2)

    return all_pts


# -- Open-Meteo weather -------------------------------------------------------
# Use best_match model (picks HRRR 3 km for US, GFS 25 km elsewhere) which is
# dramatically more accurate than ERA5 25 km for fire-weather conditions.
# ERA5 routinely under-reports peak wind by 2-5× during Santa Ana / Diablo
# events.  Fall back to ERA5 archive only if the forecast archive is unavailable.

_HIST_URL  = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_ERA5_URL  = "https://archive-api.open-meteo.com/v1/archive"
_HIST_AVAIL = date(2016, 1, 1)   # best_match / gfs_seamless starts here


def _fetch_wx_point(lat, lon, start_d, end_d, hourly_vars):
    """Fetch hourly weather at one point, preferring best_match over ERA5."""
    base = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start_d.strftime("%Y-%m-%d"),
        "end_date":         end_d.strftime("%Y-%m-%d"),
        "hourly":           hourly_vars,
        "windspeed_unit":   "mph",
        "temperature_unit": "fahrenheit",
        "timezone":         "America/Los_Angeles",
    }
    if start_d >= _HIST_AVAIL:
        try:
            r = requests.get(_HIST_URL, params={**base, "models": "best_match"}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if d.get("hourly", {}).get("time"):
                    return d
        except Exception:
            pass
    r = requests.get(_ERA5_URL, params=base, timeout=30)
    return r.json() if r.status_code == 200 else None


def fetch_weather(lat, lon, start, end):
    return _fetch_wx_point(lat, lon, start, end,
                           "temperature_2m,relativehumidity_2m,"
                           "windspeed_10m,winddirection_10m,precipitation")


# -- Open-Meteo weather grid (N×N points over fire bbox, real spatial data) ---
def fetch_weather_grid(bbox, start, end, n=5):
    """
    Fetch hourly weather at an n×n grid of real geographic points using the
    best available model (HRRR/GFS via best_match, ERA5 as fallback).
    Covers the full incident date range so the time slider always has matching
    grid data across the entire fire timeline.
    """
    minLon, minLat, maxLon, maxLat = bbox
    cap_start = start
    cap_end   = end

    lats = [round(minLat + (maxLat - minLat) * i / (n - 1), 5) for i in range(n)]
    lngs = [round(minLon + (maxLon - minLon) * j / (n - 1), 5) for j in range(n)]

    def _fetch_one(lat, lng):
        try:
            d = _fetch_wx_point(lat, lng, cap_start, cap_end,
                                "temperature_2m,relativehumidity_2m,"
                                "windspeed_10m,winddirection_10m")
            if d:
                h = d.get("hourly", {})
                return {
                    "lat":  lat,
                    "lng":  lng,
                    "time": h.get("time", []),
                    "temp": h.get("temperature_2m", []),
                    "rh":   h.get("relativehumidity_2m", []),
                    "ws":   h.get("windspeed_10m", []),
                    "wd":   h.get("winddirection_10m", []),
                }
        except Exception as e:
            print(f"  Grid ({lat},{lng}): {e}")
        return {"lat": lat, "lng": lng, "time": [], "temp": [], "rh": [], "ws": [], "wd": []}

    points = [None] * (n * n)
    pairs  = [(i * n + j, lats[i], lngs[j]) for i in range(n) for j in range(n)]

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_fetch_one, lat, lng): idx for idx, lat, lng in pairs}
        for fut in as_completed(futs):
            points[futs[fut]] = fut.result()

    return {
        "n":      n,
        "lats":   lats,
        "lngs":   lngs,
        "points": points,   # row-major: points[i*n+j] = grid[lat_i][lng_j]
    }


# -- OpenTopoData elevation (free, no key) ------------------------------------
def fetch_elevation(lat, lon):
    r = requests.get(
        f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}",
        timeout=10,
    )
    if r.status_code == 200:
        res = r.json().get("results", [])
        return res[0]["elevation"] if res else None
    return None


# -- Geometry helpers ---------------------------------------------------------
def _collect_coords(obj, lons, lats):
    if not obj:
        return
    if isinstance(obj[0], (int, float)):
        lons.append(obj[0]); lats.append(obj[1])
    else:
        for sub in obj:
            _collect_coords(sub, lons, lats)


def geojson_bbox(gj, pad=0.05):
    lons, lats = [], []
    for f in gj.get("features", []):
        _collect_coords(f.get("geometry", {}).get("coordinates", []), lons, lats)
    if not lons:
        return None
    return (min(lons)-pad, min(lats)-pad, max(lons)+pad, max(lats)+pad)


def bbox_center(bb):
    return ((bb[1]+bb[3])/2, (bb[0]+bb[2])/2)


# -- Weather danger color ------------------------------------------------------
def _danger_color(temp_f, rh_pct, wind_mph):
    """Blue=low, yellow=moderate, orange=high, red=extreme fire danger."""
    score = 0
    if temp_f  is not None and temp_f  > 90:  score += 1
    if temp_f  is not None and temp_f  > 100: score += 1
    if rh_pct  is not None and rh_pct  < 25:  score += 1
    if rh_pct  is not None and rh_pct  < 15:  score += 1
    if wind_mph is not None and wind_mph > 20: score += 1
    if wind_mph is not None and wind_mph > 35: score += 1
    return ["#3498db", "#f1c40f", "#e67e22", "#e74c3c"][min(score // 2, 3)]


# -- Weather chart (embedded PNG) ---------------------------------------------
def weather_chart(w):
    h     = w.get("hourly", {})
    times = [datetime.fromisoformat(t) for t in h.get("time", [])]
    if not times:
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig.patch.set_facecolor("#1a1a1a")

    rows = [
        (h.get("temperature_2m",      []), "Temp (F)",     "#e74c3c"),
        (h.get("relativehumidity_2m", []), "Humidity (%)", "#3498db"),
        (h.get("windspeed_10m",       []), "Wind (mph)",   "#2ecc71"),
    ]
    for ax, (vals, ylabel, color) in zip(axes, rows):
        ax.set_facecolor("#222")
        ax.plot(times, vals, color=color, linewidth=1.2)
        ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=30, color="#666", fontsize=8)
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#1a1a1a", dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# -- Build HTML report --------------------------------------------------------
def build_report(name, perimeters_gj, hotspots, weather, elevation, center):

    m = folium.Map(location=center, zoom_start=10,
                   tiles="CartoDB dark_matter", prefer_canvas=True)

    # Deduplicate: drop snapshots where acreage hasn't changed since the last
    # snapshot (these are identical re-publishes of the same geometry).
    # Keep everything else so the full progression is visible.
    raw = [f for f in perimeters_gj.get("features", []) if f.get("geometry")]
    # sort by timestamp first
    def _feat_dt(f):
        p  = f.get("properties", {})
        dt = _parse_date(p.get("poly_DateCurrent") or p.get("poly_PolygonDateTime"))
        return dt or datetime.min
    raw.sort(key=_feat_dt)
    # drop snapshots where rounded acreage is same as the immediately previous one
    features, prev_ac = [], None
    for feat in raw:
        p  = feat.get("properties", {})
        ac = round((p.get("poly_GISAcres") or 0) / 10) * 10   # bucket to nearest 10 ac
        if ac != prev_ac:
            features.append(feat)
            prev_ac = ac

    ts_feats      = []   # perimeters + hotspots (accumulate)
    daily_readings = []  # noon readings for popup table

    # -- perimeter polygon features --
    for feat in features:
        p      = feat.get("properties", {})
        dt     = _parse_date(p.get("poly_DateCurrent") or p.get("poly_PolygonDateTime"))
        iso    = dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else "1970-01-01T00:00:00"
        acres  = p.get("poly_GISAcres", "?")
        pct    = p.get("attr_PercentContained")
        fuel   = p.get("attr_PredominantFuelModel") or p.get("attr_PrimaryFuelModel") or ""
        label  = f"{iso[:10]}"
        if isinstance(acres, (int, float)): label += f" -- {acres:,.0f} ac"
        if pct is not None:                 label += f" -- {pct:.0f}% contained"
        if fuel:                            label += f" -- Fuel: {fuel}"

        ts_feats.append({
            "type":     "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "time":   iso,
                "popup":  label,
                "style":  {
                    "color":       "#e74c3c",
                    "fillColor":   "#e74c3c",
                    "fillOpacity": 0.25,
                    "weight":      2,
                },
            },
        })

    # -- VIIRS hotspot features — separate rolling layer for clarity --
    # Keep top-1500 by FRP so dense fires don't turn into a solid blob
    hs_feats = []
    top_hotspots = sorted(hotspots, key=lambda h: h["frp"], reverse=True)[:1500]
    for h in top_hotspots:
        frp    = h["frp"]
        radius = max(2, min(7, frp / 20))   # smaller, tighter dots
        if   frp > 200: hcolor = "#c0392b"
        elif frp > 80:  hcolor = "#e74c3c"
        elif frp > 30:  hcolor = "#e67e22"
        else:           hcolor = "#f39c12"
        hs_feats.append({
            "type":     "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [h["lon"], h["lat"]]},
            "properties": {
                "time":   h["datetime"],
                "popup":  f"{h['datetime']} — FRP {frp:.0f} MW",
                "icon":   "circle",
                "iconstyle": {
                    "fillColor":   hcolor,
                    "fillOpacity": 0.65,
                    "stroke":      False,
                    "radius":      radius,
                },
            },
        })

    all_feats = ts_feats + hs_feats
    if all_feats:
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": all_feats},
            period="PT1H",
            duration="P365D",
            auto_play=False,
            loop=False,
            time_slider_drag_update=True,
        ).add_to(m)

    # -- weather: JS-driven live panel that reacts to the time slider --
    if weather:
        h_data  = weather.get("hourly", {})
        times_h = h_data.get("time", [])
        temp_h  = h_data.get("temperature_2m",      [])
        rh_h    = h_data.get("relativehumidity_2m", [])
        ws_h    = h_data.get("windspeed_10m",        [])
        wd_h    = h_data.get("winddirection_10m",    [])

        wx_hourly = []
        daily_map = {}   # day → index of noon-closest reading
        for i, t in enumerate(times_h):
            t_v  = temp_h[i] if i < len(temp_h) else None
            if t_v is None:
                continue
            rh_v = rh_h[i] if i < len(rh_h) else None
            ws_v = ws_h[i] if i < len(ws_h) else None
            wd_v = wd_h[i] if i < len(wd_h) else None
            iso  = t if len(t) > 15 else t + ":00"
            wx_hourly.append({
                "time": iso,
                "temp": round(t_v, 1),
                "rh":   round(rh_v, 0) if rh_v is not None else None,
                "ws":   round(ws_v, 1) if ws_v is not None else None,
                "wd":   round(wd_v, 0) if wd_v is not None else None,
            })
            # track daily noon for popup table
            day  = t[:10]
            hour = int(t[11:13])
            if day not in daily_map or abs(hour - 12) < abs(daily_map[day][0] - 12):
                daily_map[day] = (hour, i, iso, t_v, rh_v, ws_v, wd_v)

        for day in sorted(daily_map):
            _, _, iso, t_v, rh_v, ws_v, wd_v = daily_map[day]
            daily_readings.append((day, iso, t_v, rh_v, ws_v, wd_v))

        map_var  = m.get_name()
        wx_json  = json.dumps(wx_hourly)

        # build hourly popup table rows (all hours, scrollable)
        tbl_rows = ""
        for row in wx_hourly:
            clr = _danger_color(row["temp"], row["rh"] or 50, row["ws"] or 0)
            tbl_rows += (
                f'<tr style="border-bottom:1px solid #222">'
                f'<td style="padding:2px 6px;color:#888">{row["time"][:16].replace("T"," ")}</td>'
                f'<td style="padding:2px 6px;color:#e74c3c">{row["temp"]:.0f}°F</td>'
                f'<td style="padding:2px 6px;color:#3498db">{int(row["rh"] or 0)}%</td>'
                f'<td style="padding:2px 6px;color:#2ecc71">{int(row["ws"] or 0)} mph</td>'
                f'<td style="padding:2px 6px;color:#ccc">{int(row["wd"] or 0)}°</td>'
                f'<td style="padding:2px 6px"><span style="color:{clr}">&#9679;</span></td>'
                f'</tr>'
            )

        wx_station_popup = (
            '<div style="background:#1a1a1a;color:#eee;font-size:11px;'
            'max-height:340px;overflow-y:auto;padding:10px 12px;min-width:340px">'
            '<b style="font-size:12px">Hourly Weather at Fire Center</b><br><br>'
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="color:#555;font-size:10px;border-bottom:1px solid #333">'
            '<th style="padding:2px 6px;text-align:left">Time</th>'
            '<th>Temp</th><th>RH</th><th>Wind</th><th>Dir</th><th>Risk</th>'
            '</tr>'
            + tbl_rows +
            '</table></div>'
        )

        # weather station marker — offset slightly NW so fire icon stays separate
        wx_fg = folium.FeatureGroup(name="Weather Station", show=True)
        folium.CircleMarker(
            location=[center[0] + 0.005, center[1] - 0.005],
            radius=10,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color="#2980b9",
            fill_opacity=0.8,
            tooltip="Weather Station — click for hourly temp / humidity / wind table",
            popup=folium.Popup(wx_station_popup, max_width=400),
        ).add_to(wx_fg)
        wx_fg.add_to(m)

        # -- fire bbox for overlay grid (padded perimeter extent) --
        lons_b, lats_b = [], []
        for feat in features:
            _collect_coords(feat.get("geometry", {}).get("coordinates", []), lons_b, lats_b)
        if lons_b:
            pad  = 0.15
            fx1  = min(lons_b) - pad   # west
            fy1  = min(lats_b) - pad   # south
            fx2  = max(lons_b) + pad   # east
            fy2  = max(lats_b) + pad   # north
        else:
            fx1, fy1 = center[1] - 0.3, center[0] - 0.3
            fx2, fy2 = center[1] + 0.3, center[0] + 0.3

        map_var = m.get_name()

        # windy.com-style overlay: canvas wind particles + temp/humidity colour layers + switcher
        wx_overlay = f"""
<style>
#wx-sw {{
  position:absolute; left:10px; top:50%; transform:translateY(-50%);
  z-index:9999; display:flex; flex-direction:column; gap:5px;
}}
.wx-sw-btn {{
  background:rgba(18,18,18,.92); border:1px solid #555; border-radius:8px;
  color:#bbb; font-size:11px; font-weight:600; padding:9px 8px;
  cursor:pointer; text-align:center; width:64px;
  letter-spacing:.3px; transition:all .18s; user-select:none;
}}
.wx-sw-btn.on  {{ background:#1a6ea8; border-color:#2980b9; color:#fff; }}
.wx-sw-btn:hover:not(.on) {{ background:rgba(50,50,50,.95); color:#fff; }}
#wx-bar {{
  position:absolute; bottom:42px; left:50%; transform:translateX(-50%);
  z-index:9999; background:rgba(14,14,14,.93); border:1px solid #3a3a3a;
  border-radius:10px; padding:8px 16px; display:flex; gap:12px;
  align-items:center; pointer-events:none;
  box-shadow:0 2px 10px rgba(0,0,0,.6); font-family:sans-serif;
}}
#wx-bar .b-item {{ text-align:center; min-width:52px; }}
#wx-bar .b-val  {{ font-size:20px; font-weight:700; color:#eee; line-height:1.1; }}
#wx-bar .b-lbl  {{ font-size:9px; color:#555; text-transform:uppercase; letter-spacing:.5px; }}
#wx-bar .b-sep  {{ width:1px; height:36px; background:#2a2a2a; }}
#wx-bar .b-arr  {{ font-size:26px; display:inline-block; transition:transform .3s; line-height:1; }}
#wx-legend {{
  position:absolute; right:10px; top:50%; transform:translateY(-50%);
  z-index:9999; background:rgba(14,14,14,.88); border:1px solid #333;
  border-radius:8px; padding:10px 8px; display:none;
  flex-direction:column; align-items:center; gap:0; width:32px;
}}
#wx-legend .leg-lbl {{ font-size:9px; color:#888; white-space:nowrap; margin:3px 0; writing-mode:horizontal-tb; }}
</style>
<script>
(function(){{
  var WX    = {wx_json};
  var FBBOX = [{fy1},{fx1},{fy2},{fx2}];  // [S,W,N,E]
  var MV    = "{map_var}";

  // ── colour helpers ────────────────────────────────────────────────────────
  function lerpHex(a,b,t){{
    function ch(c,o){{return parseInt(c.slice(o,o+2),16);}}
    function hh(v){{return('0'+Math.round(v).toString(16)).slice(-2);}}
    return '#'+hh(ch(a,1)+(ch(b,1)-ch(a,1))*t)
              +hh(ch(a,3)+(ch(b,3)-ch(a,3))*t)
              +hh(ch(a,5)+(ch(b,5)-ch(a,5))*t);
  }}
  function scaleColor(stops,t){{
    t=Math.max(0,Math.min(1,t));
    for(var i=1;i<stops.length;i++){{
      if(t<=stops[i][0]){{
        var f=(t-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
        return lerpHex(stops[i-1][1],stops[i][1],f);
      }}
    }}
    return stops[stops.length-1][1];
  }}
  var T_SC=[
    [0,'#053061'],[.12,'#2166ac'],[.28,'#4393c3'],[.44,'#92c5de'],
    [.56,'#fddbc7'],[.68,'#f4a582'],[.80,'#d6604d'],[.92,'#b2182b'],[1,'#67001f']
  ];
  var RH_SC=[[0,'#ffffd4'],[.25,'#c2e699'],[.50,'#78c679'],[.75,'#31a354'],[1,'#006837']];
  var T_MIN=30,T_MAX=115;
  function tCol(t){{ return scaleColor(T_SC,(t-T_MIN)/(T_MAX-T_MIN)); }}
  function rhCol(r){{ return scaleColor(RH_SC,r/100); }}

  // wind speed → rgb array (calm=blue → moderate=green → strong=yellow → extreme=red)
  function windRGB(ws){{
    var stops=[[0,[100,190,255]],[15,[80,220,120]],[28,[250,220,50]],[45,[255,120,20]],[70,[230,40,40]]];
    for(var i=1;i<stops.length;i++){{
      if(ws<=stops[i][0]){{
        var f=(ws-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
        var a=stops[i-1][1],b=stops[i][1];
        return[Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f)];
      }}
    }}
    return [230,40,40];
  }}

  // ── risk helpers ──────────────────────────────────────────────────────────
  var RCOL=['#3498db','#f1c40f','#e67e22','#e74c3c'];
  var RLBL=['Low','Moderate','High','Extreme'];
  function risk(t,rh,ws){{
    var s=0;
    if(t>90)s++;if(t>100)s++;
    if((rh||50)<25)s++;if((rh||50)<15)s++;
    if((ws||0)>20)s++;if((ws||0)>35)s++;
    return Math.min(Math.floor(s/2),3);
  }}

  // ── find closest hourly reading ───────────────────────────────────────────
  function findWx(ms){{
    if(!WX.length)return null;
    var best=WX[0],bd=Infinity;
    for(var i=0;i<WX.length;i++){{
      var d=Math.abs(new Date(WX[i].time).getTime()-ms);
      if(d<bd){{bd=d;best=WX[i];}}
    }}
    return bd<7200000?best:null;
  }}

  // ── readout bar ───────────────────────────────────────────────────────────
  function updateBar(r){{
    var el=document.getElementById('wx-bar');
    if(!el||!r)return;
    var sc=risk(r.temp,r.rh,r.ws),col=RCOL[sc];
    var deg=((r.wd||0)+180)%360;
    el.innerHTML=
      '<div class="b-item"><div class="b-lbl">'+r.time.slice(0,16).replace('T',' ')+'</div>'+
        '<div style="height:3px;border-radius:2px;background:'+col+';margin:3px 0"></div>'+
        '<div style="font-size:9px;color:'+col+'">'+RLBL[sc]+' risk</div></div>'+
      '<div class="b-sep"></div>'+
      '<div class="b-item"><div class="b-lbl">Temp</div>'+
        '<div class="b-val" style="color:#e74c3c">'+r.temp.toFixed(0)+'&#176;F</div></div>'+
      '<div class="b-sep"></div>'+
      '<div class="b-item"><div class="b-lbl">Humidity</div>'+
        '<div class="b-val" style="color:#2ecc71">'+(r.rh!=null?r.rh.toFixed(0):'?')+'%</div></div>'+
      '<div class="b-sep"></div>'+
      '<div class="b-item"><div class="b-lbl">Wind</div>'+
        '<div class="b-val" style="color:#3498db">'+(r.ws!=null?r.ws.toFixed(0):'?')+' mph</div></div>'+
      '<div class="b-sep"></div>'+
      '<div class="b-item"><div class="b-lbl">Direction</div>'+
        '<div class="b-arr" style="transform:rotate('+deg+'deg);color:'+col+'">&#8679;</div>'+
        '<div style="font-size:9px;color:#777">'+(r.wd!=null?r.wd.toFixed(0):'?')+'&#176; from</div></div>';
  }}

  // ── colour legend ─────────────────────────────────────────────────────────
  function showLegend(stops,topLbl,botLbl){{
    var el=document.getElementById('wx-legend');
    if(!el)return;
    var parts=[];
    for(var i=stops.length-1;i>=0;i--) parts.push(stops[i][1]+' '+((1-stops[i][0])*100).toFixed(0)+'%');
    el.innerHTML=
      '<div class="leg-lbl">'+topLbl+'</div>'+
      '<div style="width:14px;height:140px;border-radius:3px;background:linear-gradient(to bottom,'+parts.join(',')+')" ></div>'+
      '<div class="leg-lbl">'+botLbl+'</div>';
    el.style.display='flex';
  }}
  function hideLegend(){{ var el=document.getElementById('wx-legend');if(el)el.style.display='none'; }}

  // ── native canvas wind particle system ───────────────────────────────────
  function WindCanvas(container, mapObj){{
    var self=this;
    this.ws=5; this.wd=0;
    this.running=false;
    this._frame=null;
    this._ps=[];
    this._map=mapObj;

    var cv=document.createElement('canvas');
    cv.style.cssText=
      'position:absolute;top:0;left:0;width:100%;height:100%;'+
      'pointer-events:none;z-index:450;';
    container.appendChild(cv);
    this._cv=cv;
    this._ctx=cv.getContext('2d');

    function sizeCanvas(){{
      var w=container.offsetWidth||container.clientWidth||900;
      var h=container.offsetHeight||container.clientHeight||500;
      if(w>10&&h>10){{ cv.width=w; cv.height=h; self._init(w,h); }}
      else setTimeout(sizeCanvas,100);
    }}
    if(mapObj&&mapObj.on) mapObj.on('resize',sizeCanvas);
    setTimeout(sizeCanvas,120);
  }}

  WindCanvas.prototype={{
    _init:function(w,h){{
      this._ps=[];
      for(var i=0;i<500;i++)
        this._ps.push({{x:Math.random()*w,y:Math.random()*h,
                        age:Math.floor(Math.random()*80),
                        life:60+Math.floor(Math.random()*70),
                        hist:[]}});
    }},
    show:function(){{
      this.running=true;
      this._cv.style.display='block';
      if(!this._frame)this._tick();
    }},
    hide:function(){{
      this.running=false;
      this._cv.style.display='none';
      if(this._frame){{cancelAnimationFrame(this._frame);this._frame=null;}}
      this._ctx.clearRect(0,0,this._cv.width,this._cv.height);
    }},
    set:function(ws,wd){{ this.ws=ws||0; this.wd=wd||0; }},
    _tick:function(){{
      if(!this.running){{this._frame=null;return;}}
      var self=this,ctx=this._ctx,cv=this._cv;
      var w=cv.width,h=cv.height;
      ctx.clearRect(0,0,w,h);
      ctx.lineCap='round';

      var baseWs=Math.max(1,this.ws), baseWd=this.wd;
      var mapObj=this._map;
      var zoom=mapObj&&mapObj.getZoom?mapObj.getZoom():10;
      var speedScale=Math.pow(2,Math.max(-1,zoom-10)*0.85);
      var TRAIL=Math.max(10,26-Math.max(0,zoom-10)*2);

      for(var i=0;i<this._ps.length;i++){{
        var p=this._ps[i];
        var t=p.age/p.life;
        var bright=t<0.15?t/0.15:t>0.72?(1-t)/0.28:1.0;

        var localWs=baseWs, localWd=baseWd;

        var ang=((localWd+180)%360)*Math.PI/180;
        var spd=Math.min(32,Math.max(0.2,localWs*0.20*speedScale));
        var vx=Math.sin(ang)*spd, vy=-Math.cos(ang)*spd;

        p.hist.push([p.x,p.y]);
        if(p.hist.length>TRAIL)p.hist.shift();
        p.x+=vx; p.y+=vy; p.age++;

        if(p.x<-30)p.x=w+30; if(p.x>w+30)p.x=-30;
        if(p.y<-30)p.y=h+30; if(p.y>h+30)p.y=-30;

        var rgb=windRGB(localWs);
        var rgbStr=rgb[0]+','+rgb[1]+','+rgb[2];
        var n=p.hist.length;
        for(var j=1;j<n;j++){{
          var alpha=(j/n)*bright*0.93;
          ctx.beginPath();
          ctx.moveTo(p.hist[j-1][0],p.hist[j-1][1]);
          ctx.lineTo(p.hist[j][0],  p.hist[j][1]);
          ctx.strokeStyle='rgba('+rgbStr+','+alpha+')';
          ctx.lineWidth=0.9+(j/n)*1.7;
          ctx.stroke();
        }}

        if(p.age>=p.life){{
          p.x=Math.random()*w; p.y=Math.random()*h;
          p.age=0; p.life=60+Math.floor(Math.random()*70); p.hist=[];
        }}
      }}
      this._frame=requestAnimationFrame(function(){{self._tick();}});
    }}
  }};

  // ── phase 1: build DOM elements immediately ───────────────────────────────
  function buildUI(){{
    var container=document.getElementById(MV)
      ||document.querySelector('.leaflet-container');
    if(!container){{setTimeout(buildUI,150);return;}}

    var r0=WX.length?WX[0]:{{ws:8,wd:225,temp:75,rh:35}};
    var mapObj=window[MV];

    // wind canvas
    var wind=new WindCanvas(container,mapObj||{{on:function(){{}}}});
    wind.set(r0.ws||0,r0.wd||0);
    wind.show();

    // temp + humidity colour overlays (need real map for L.rectangle)
    var tempRect=null,rhRect=null;
    if(mapObj){{
      tempRect=L.rectangle([[FBBOX[0],FBBOX[1]],[FBBOX[2],FBBOX[3]]],
        {{fillColor:tCol(r0.temp||70),fillOpacity:0.40,stroke:false,interactive:false}});
      rhRect=L.rectangle([[FBBOX[0],FBBOX[1]],[FBBOX[2],FBBOX[3]]],
        {{fillColor:rhCol(r0.rh||40),fillOpacity:0.40,stroke:false,interactive:false}});
    }}

    // switcher sidebar
    var sw=document.createElement('div'); sw.id='wx-sw';
    sw.innerHTML=
      '<div class="wx-sw-btn on" id="sw-wind" onclick="window._wxSw(\'wind\')">&#x1F32C;<br>Wind</div>'+
      '<div class="wx-sw-btn"    id="sw-temp" onclick="window._wxSw(\'temp\')">&#x1F321;<br>Temp</div>'+
      '<div class="wx-sw-btn"    id="sw-rh"   onclick="window._wxSw(\'rh\')">&#x1F4A7;<br>Humid</div>';
    container.appendChild(sw);

    // readout bar — show first reading right away
    var bar=document.createElement('div'); bar.id='wx-bar';
    container.appendChild(bar);
    updateBar(r0);

    // legend strip
    var leg=document.createElement('div'); leg.id='wx-legend';
    container.appendChild(leg);

    var mode='wind';
    window._wxSw=function(m){{
      mode=m;
      ['wind','temp','rh'].forEach(function(k){{
        document.getElementById('sw-'+k).classList.toggle('on',k===m);
      }});
      if(m==='wind'){{
        wind.show();
        if(tempRect)tempRect.remove();
        if(rhRect)rhRect.remove();
        hideLegend();
      }} else if(m==='temp'){{
        wind.hide();
        if(tempRect&&mapObj)tempRect.addTo(mapObj);
        if(rhRect)rhRect.remove();
        showLegend(T_SC,'Hot','Cold');
      }} else {{
        wind.hide();
        if(rhRect&&mapObj)rhRect.addTo(mapObj);
        if(tempRect)tempRect.remove();
        showLegend(RH_SC,'Wet','Dry');
      }}
    }};

    // phase 2: hook up time slider when timeDimension is ready
    hookTime(mapObj,wind,tempRect,rhRect,function(){{return mode;}});
  }}

  // ── phase 2: connect time slider ──────────────────────────────────────────
  function hookTime(mapObj,wind,tempRect,rhRect,getMode){{
    if(!mapObj||!mapObj.timeDimension){{setTimeout(function(){{
      hookTime(window[MV],wind,tempRect,rhRect,getMode);
    }},300);return;}}
    mapObj.timeDimension.on('timeload',function(e){{
      var r=findWx(e.time);
      if(!r)return;
      updateBar(r);
      wind.set(r.ws||0,r.wd||0);
      var m=getMode();
      if(m==='temp'&&tempRect)tempRect.setStyle({{fillColor:tCol(r.temp||70)}});
      if(m==='rh'&&rhRect)    rhRect.setStyle({{fillColor:rhCol(r.rh||50)}});
    }});
  }}

  // Run as soon as this script tag is parsed — no DOMContentLoaded needed
  // because Folium puts this script LAST in the body, after all map scripts.
  setTimeout(buildUI, 50);
}})();
</script>
"""
        m.get_root().html.add_child(folium.Element(wx_overlay))

    # -- fire center marker --
    elev_str = (f"Elevation: {elevation:.0f} m ({elevation*3.28:.0f} ft)"
                if elevation else "")
    folium.Marker(
        location=center,
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        popup=f"{center[0]:.3f}N {abs(center[1]):.3f}W<br>{elev_str}",
    ).add_to(m)

    # -- map legend (bottom-left) --
    legend_html = """
    <div style="position:fixed;bottom:30px;left:86px;z-index:1000;
                background:#1a1a1a;border:1px solid #444;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#ccc;min-width:160px">
      <b style="color:#fff">Fire Perimeter</b><br>
      <span style="color:#e74c3c">&#9632;</span> Daily (accumulates)<br><br>
      <b style="color:#fff">Hotspots (FRP)</b><br>
      <span style="color:#f1c40f">&#9679;</span> Low &nbsp;
      <span style="color:#e67e22">&#9679;</span> Med<br>
      <span style="color:#e74c3c">&#9679;</span> High &nbsp;
      <span style="color:#c0392b">&#9679;</span> Extreme<br><br>
      <b style="color:#fff">Weather left bar:</b><br>
      &#x1F32C; Wind particles<br>
      &#x1F321; Temp colour overlay<br>
      &#x1F4A7; Humidity overlay<br>
      &#9711; Station = click for table
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    # -- stats --
    n_perims  = len(features)
    max_acres = max(
        (f.get("properties", {}).get("poly_GISAcres") or 0 for f in features),
        default=0,
    )
    dts = [_parse_date(f["properties"].get("poly_PolygonDateTime") or
                        f["properties"].get("poly_DateCurrent"))
           for f in features]
    dts = [d for d in dts if d]
    date_range = (f"{min(dts).strftime('%b %d')} to {max(dts).strftime('%b %d, %Y')}"
                  if dts else "unknown")

    wx_b64   = weather_chart(weather) if weather else ""
    wx_block = (f'<img src="data:image/png;base64,{wx_b64}" style="width:100%">'
                if wx_b64 else "<p style='color:#555'>Weather unavailable.</p>")

    # Render Folium to standalone HTML, then inline it (no iframe/srcdoc).
    map_id  = m.get_name()
    raw     = m.get_root().render()

    head_src = (re.search(r'<head[^>]*>(.*?)</head>', raw, re.DOTALL | re.IGNORECASE) or re.search(r'()', '')).group(1)
    body_src = (re.search(r'<body[^>]*>(.*?)</body>', raw, re.DOTALL | re.IGNORECASE) or re.search(r'()', '')).group(1)

    # Split head: external CDN links stay; inline scripts move to end of body
    inline_scripts = re.findall(r'<script(?![^>]*\bsrc\b)[^>]*>.*?</script>',
                                head_src, re.DOTALL | re.IGNORECASE)
    head_clean     = re.sub(r'<script(?![^>]*\bsrc\b)[^>]*>.*?</script>', '',
                            head_src, flags=re.DOTALL | re.IGNORECASE)
    # Remove Folium's full-page height styles that break when inlined
    head_clean = re.sub(r'<style[^>]*>\s*html\s*,\s*body\s*\{[^}]*\}\s*</style>',
                        '', head_clean, flags=re.DOTALL | re.IGNORECASE)
    head_clean = re.sub(r'<style[^>]*>\s*#map\s*\{[^}]*\}\s*</style>',
                        '', head_clean, flags=re.DOTALL | re.IGNORECASE)

    # Inline scripts first (L_NO_TOUCH etc.) then map body (Leaflet init + WX overlay)
    map_body_html  = '\n'.join(inline_scripts) + '\n' + body_src

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{name} -- Fire Analysis</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#eee;font-family:sans-serif}}
  header{{background:#1a1a1a;padding:18px 28px;border-bottom:1px solid #2a2a2a}}
  header h1{{color:#e74c3c;font-size:22px;font-weight:600}}
  header p{{color:#777;font-size:13px;margin-top:5px}}
  .stats{{display:flex;gap:14px;padding:16px 28px;flex-wrap:wrap}}
  .stat{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;
         padding:12px 18px;min-width:120px}}
  .stat .v{{font-size:19px;font-weight:700;color:#e74c3c}}
  .stat .l{{font-size:11px;color:#666;margin-top:3px}}
  .sec{{padding:0 28px 28px}}
  .sec h2{{color:#bbb;font-size:13px;font-weight:600;letter-spacing:.5px;
           text-transform:uppercase;padding:10px 0;border-bottom:1px solid #222;
           margin-bottom:12px}}
  .note{{color:#555;font-size:11px;margin-top:8px}}
  /* map container — fixed pixel height so Leaflet gets real dimensions */
  #fire-map-wrap {{
    position:relative; width:100%; height:640px;
    background:#0d1117; border-radius:8px; overflow:hidden;
  }}
</style>
{head_clean}
</head>
<body>
<header>
  <h1>Fire: {name}</h1>
  <p>Use the time slider to step through perimeter growth hour by hour.
     Left bar = Wind / Temp / Humidity layers. Bottom bar = live weather readout.</p>
</header>
<div class="stats">
  <div class="stat"><div class="v">{n_perims}</div>
    <div class="l">Perimeter snapshots</div></div>
  <div class="stat"><div class="v">{max_acres:,.0f}</div>
    <div class="l">Peak acres</div></div>
  <div class="stat"><div class="v">{len(hotspots)}</div>
    <div class="l">VIIRS hotspots</div></div>
  <div class="stat"><div class="v">{f"{elevation:.0f} m" if elevation else "N/A"}</div>
    <div class="l">Center elevation</div></div>
  <div class="stat">
    <div class="v" style="font-size:13px;padding-top:3px">{date_range}</div>
    <div class="l">Date range</div></div>
</div>
<div class="sec">
  <h2>Fire Spread Map</h2>
  <div id="fire-map-wrap">
{map_body_html}
  </div>
  <p class="note">
    Red polygon = fire perimeter (accumulates as slider advances hour by hour).
    Left bar: Wind / Temp / Humidity layers. Blue circle = weather station (click for table).
  </p>
</div>
<div class="sec">
  <h2>Hourly Weather at Fire Center</h2>
  {wx_block}
</div>
</body>
</html>"""


# -- Main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate an interactive fire history report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fire_analysis.py CAMP   --year 2018 --state CA\n"
            "  python fire_analysis.py CALDOR --year 2021 --state CA\n"
            "  python fire_analysis.py DIXIE  --year 2021 --state CA --key ABC123\n"
            "\n"
            "NASA FIRMS key (free -- adds hourly hotspot layer):\n"
            "  https://firms.modaps.eosdis.nasa.gov/api/  ->  Get MAP_KEY"
        )
    )
    ap.add_argument("fire",           help="Fire name, e.g. CAMP or CALDOR")
    ap.add_argument("--year",  "-y",  type=int, help="Year, e.g. 2018")
    ap.add_argument("--state", "-s",  help="State abbreviation, e.g. CA")
    ap.add_argument("--key",   "-k", default=FIRMS_KEY,
                    help="NASA FIRMS MAP_KEY (default: pre-configured)")
    ap.add_argument("--out",   "-o",  help="Output HTML filename")
    args = ap.parse_args()

    print(f"\nFire: {args.fire}  year={args.year}  state={args.state}")
    print("=" * 50)

    print("1/4  Fetching perimeters from NIFC...")
    gj = fetch_perimeters(args.fire, args.year, args.state)
    n  = len(gj.get("features", []))
    if n == 0:
        print(f"\nNo perimeters found for '{args.fire}'.")
        print("  Tips:")
        print("  - Use the official short name without 'Fire' (e.g. CAMP not 'Camp Fire')")
        print("  - Add --year and --state to narrow the search")
        print("  - Try: CAMP --year 2018 --state CA")
        sys.exit(1)
    print(f"     {n} perimeter(s) found")

    bbox   = geojson_bbox(gj)
    center = bbox_center(bbox)

    dts = []
    for f in gj.get("features", []):
        p  = f.get("properties", {})
        dt = _parse_date(p.get("poly_DateCurrent") or p.get("poly_PolygonDateTime"))
        if dt:
            dts.append(dt)
    start_d = min(dts).date() if dts else date.today() - timedelta(30)
    end_d   = max(dts).date() if dts else date.today()
    print(f"     {start_d} to {end_d}  |  center {center[0]:.3f}N {abs(center[1]):.3f}W")

    # 2. Hotspots
    hotspots = []
    if args.key:
        print("2/4  Fetching VIIRS hotspots from NASA FIRMS...")
        hotspots = fetch_hotspots(args.key, bbox, start_d, end_d)
        print(f"     {len(hotspots)} hotspot detection(s)")
    else:
        print("2/4  Skipping hotspots (no --key provided)")
        print("     Get a free key: https://firms.modaps.eosdis.nasa.gov/api/")

    # 3. Weather
    print("3/4  Fetching hourly weather from Open-Meteo...")
    weather = fetch_weather(center[0], center[1], start_d, end_d)
    hrs = len(weather.get("hourly", {}).get("time", [])) if weather else 0
    print(f"     {hrs} hours" if hrs else "     unavailable")

    # 4. Terrain
    print("4/4  Fetching elevation from OpenTopoData...")
    elev = fetch_elevation(center[0], center[1])
    print(f"     {elev:.0f} m  ({elev*3.28:.0f} ft)" if elev else "     unavailable")

    # Build
    print("\nBuilding report...")
    html     = build_report(args.fire, gj, hotspots, weather, elev, center)
    out_path = Path(args.out or f"fire_{args.fire.replace(' ', '_')}.html")
    out_path.write_text(html, encoding="utf-8")

    print(f"\nSaved: {out_path.resolve()}")
    print("Open that file in your browser.")


if __name__ == "__main__":
    main()
