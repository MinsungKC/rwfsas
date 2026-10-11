"""Charted-airport locations, for rejecting airport-pattern/approach traffic
misidentified as working a fire (rebuild-spec Section 8 item 5: "reject...
any aircraft whose track is consistent with an instrument approach to a
charted airport. Add an airport/approach-corridor exclusion layer.").

Data: OurAirports' free, public, unauthenticated CSV
(https://ourairports.com/data/, mirrored at
davidmegginson.github.io/ourairports-data/airports.csv) -- ~80k landing
facilities worldwide with lat/lon/elevation. Verified live and working
2026-09-17 (the FAA's own airport data is behind the same Akamai bot
protection that blocked the aircraft registry probe; this is a documented
substitute, not a design preference).
"""
import csv
import json
import math
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '_cache')
RAW_CSV = os.path.join(CACHE_DIR, 'ourairports_airports.csv')
INDEX_JSON = os.path.join(CACHE_DIR, 'airport_index.json')
RAW_URL = 'https://davidmegginson.github.io/ourairports-data/airports.csv'
RAW_MAX_AGE_DAYS = 90  # airports open/close rarely

EXCLUDE_TYPES = {'closed', 'balloonport'}
GRID_DEG = 0.5  # ~55 km bins at mid latitudes -- coarse but fine for an 8-15 km search radius


def _fresh(path, max_age_days):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_days * 86400


def ensure_raw_csv(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and _fresh(RAW_CSV, RAW_MAX_AGE_DAYS):
        return RAW_CSV
    tmp = RAW_CSV + '.part'
    req = urllib.request.Request(RAW_URL, headers={'User-Agent': 'fire-research'})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, RAW_CSV)
    return RAW_CSV


def build_index(force=False, force_download=False):
    """{ "lat,lon_grid_key": [{ident, lat, lon, elev_ft, type}, ...] }."""
    if not force and _fresh(INDEX_JSON, RAW_MAX_AGE_DAYS):
        with open(INDEX_JSON, encoding='utf-8') as fh:
            return json.load(fh)
    raw = ensure_raw_csv(force=force_download)
    grid = {}
    with open(raw, encoding='utf-8', errors='replace', newline='') as fh:
        for row in csv.DictReader(fh):
            t = (row.get('type') or '').strip()
            if t in EXCLUDE_TYPES or not t:
                continue
            try:
                lat = float(row['latitude_deg']); lon = float(row['longitude_deg'])
            except (KeyError, ValueError, TypeError):
                continue
            elev = row.get('elevation_ft')
            try:
                elev_ft = float(elev) if elev not in (None, '') else 0.0
            except ValueError:
                elev_ft = 0.0
            key = f'{round(lat/GRID_DEG)},{round(lon/GRID_DEG)}'
            grid.setdefault(key, []).append({
                'ident': row.get('ident'), 'lat': lat, 'lon': lon,
                'elev_ft': elev_ft, 'type': t,
            })
    os.makedirs(os.path.dirname(INDEX_JSON), exist_ok=True)
    with open(INDEX_JSON, 'w', encoding='utf-8') as fh:
        json.dump(grid, fh)
    return grid


def _hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(a)))


_INDEX_CACHE = None


def nearest_airport(lat, lon, max_km=15.0):
    """(airport_dict, distance_km) for the nearest charted airport/heliport
    within max_km, or None. Lazily builds/loads the grid index on first call
    in this process. A build/download failure propagates as an exception --
    callers must treat that as "cannot evaluate," not "no airport nearby"
    (the same Ground-Rule-#1 distinction as everywhere else in this pass)."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = build_index()
    gx, gy = round(lat/GRID_DEG), round(lon/GRID_DEG)
    span = max(1, int(math.ceil(max_km / (GRID_DEG*111.0))) + 1)
    best = None
    for dx in range(-span, span+1):
        for dy in range(-span, span+1):
            for apt in _INDEX_CACHE.get(f'{gx+dx},{gy+dy}', []):
                d = _hav_km(lat, lon, apt['lat'], apt['lon'])
                if d <= max_km and (best is None or d < best[1]):
                    best = (apt, d)
    return best


if __name__ == '__main__':
    import sys
    idx = build_index(force='--force' in sys.argv, force_download='--redownload' in sys.argv)
    n = sum(len(v) for v in idx.values())
    print(f'{n} charted airports/heliports indexed -> {INDEX_JSON}')
    if len(sys.argv) > 2 and sys.argv[1] == '--near':
        lat, lon = float(sys.argv[2]), float(sys.argv[3])
        print(nearest_airport(lat, lon, max_km=30))
