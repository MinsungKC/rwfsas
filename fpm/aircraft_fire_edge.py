"""Estimate a fire's ACTIVE perimeter from firefighting-aircraft ADS-B tracks.

Premise (user's idea): air tankers and lead planes fly low, slow runs and drop
retardant on the FLANKS/HEAD to steer the fire. Their low-altitude track points
therefore trace the active, DEFENDED edge -- a signal independent of satellites
and immune to GOES's 2 km footprint bloat.

Access reality (probed 2026-09-12):
  * OpenSky live states/all: OPEN (anonymous) -> works for live fires now.
  * OpenSky historical flights/tracks: 403 without an account -> to replay a
    past fire (e.g. Lucas Sep 10-11) you must set OPENSKY_CLIENT_ID/SECRET
    (free OAuth2 client creds) and this module will use the authed endpoints.
  * ADS-B Exchange historical: paid. adsb.fi/airplanes.live: live only.

So: run live against an active fire, or provide OpenSky creds to replay.
"""
import os, math, time, json, urllib.request, urllib.parse
import numpy as np

# --- firefighting callsign heuristics (OpenSky callsign field, 8 chars) ---
# NOTE: most firefighting aircraft squawk their TAIL NUMBER, not "TANKER", so
# callsign is a WEAK identifier (a bare 'N' prefix matches every US civil
# aircraft -- do NOT use it). Real identification needs a curated registry
# (TANKER_REGISTRY, ICAO24 hex) and/or behavioral loiter detection over the
# fire (loitering_over). These prefixes only catch aircraft that DO broadcast a
# fire callsign.
FF_PREFIXES = ('TANKER', 'TNKR', 'AIRATK', 'AIRAT', 'ATTACK',
               'LEAD', 'BRONCO', 'COPTER', 'HELITK', 'CALFIRE', 'GRIZZLY',
               'MALIBU', 'NEPTUNE')
# Curated CA/federal aerial-firefighting ICAO24 hex addresses go here (stable,
# unlike callsigns). Populate from tail numbers of the 10 Tanker/Coulson/
# Neptune/CAL FIRE fleet. Empty until filled; loiter detection covers the gap.
TANKER_REGISTRY = set()
# drops happen LOW and SLOW; air attack orbits higher. Tune per platform.
DROP_MAX_ALT_M = 900.0      # ~3000 ft AGL band for a drop run (terrain-relative ideally)
DROP_MAX_SPD_MS = 90.0      # ~175 kt; tankers slow to ~120-140 kt on the line


def _get(url, auth=None, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': 'fire-research'})
    if auth:
        req.add_header('Authorization', 'Bearer ' + auth)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300]
    except Exception as e:
        return 'ERR', str(e)[:300]


def metar_altimeter(lat, lon, radius_deg=1.0):
    """Nearest METAR's altimeter setting, for correcting ADS-B barometric
    altitude to true MSL (rebuild-spec Section 8 item 1).

    Uses the free aviationweather.gov JSON API (no auth/key). Its `altim`
    field is in hPa (verified live 2026-09-17 against KAUN's raw METAR
    "A3015" -> 1021.1 hPa / 33.8639 = 30.16 inHg -- do not assume inHg from
    the field name alone, that would silently reintroduce this exact bug).

    Returns {'altim_inhg', 'station', 'distance_km', 'age_min'} for the
    nearest station with a report, or None if the query failed or returned no
    stations -- callers must not treat None as "standard pressure" (that is
    precisely the uncorrected-baro bug this function exists to fix).
    """
    url = (f'https://aviationweather.gov/api/data/metar?bbox='
           f'{lat-radius_deg:.3f},{lon-radius_deg:.3f},{lat+radius_deg:.3f},{lon+radius_deg:.3f}'
           '&format=json')
    try:
        stations = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={'User-Agent': 'fire-research'}), timeout=20).read())
    except Exception as e:
        print(f'  ! METAR altimeter unavailable ({lat:.3f},{lon:.3f}): {e!r}'[:150], flush=True)
        return None
    cands = [s for s in stations if s.get('altim') is not None
             and s.get('lat') is not None and s.get('lon') is not None]
    if not cands:
        return None
    mx = 111.32 * math.cos(math.radians(lat))
    def d_km(s): return math.hypot((s['lon']-lon)*mx, (s['lat']-lat)*111.32)
    best = min(cands, key=d_km)
    age_min = None
    try:
        age_min = (time.time() - best['obsTime']) / 60.0
    except Exception:
        pass
    return {'altim_inhg': float(best['altim']) / 33.8639, 'station': best.get('icaoId'),
            'distance_km': d_km(best), 'age_min': age_min}


def baro_to_msl_m(baro_alt_m, altim_inhg):
    """Correct 29.92-inHg-referenced barometric altitude to true MSL.

    Standard aviation approximation: 0.01 inHg of altimeter-setting
    departure from 29.92 (standard) corresponds to ~10 ft of true altitude.
    """
    return baro_alt_m + (altim_inhg - 29.92) * 1000.0 * 0.3048


def opensky_token():
    """OAuth2 client-credentials token if OPENSKY_CLIENT_ID/SECRET are set,
    else None (anonymous, live-only)."""
    cid = os.environ.get('OPENSKY_CLIENT_ID'); sec = os.environ.get('OPENSKY_CLIENT_SECRET')
    if not (cid and sec):
        return None
    data = urllib.parse.urlencode({'grant_type': 'client_credentials',
                                   'client_id': cid, 'client_secret': sec}).encode()
    url = 'https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token'
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=30)
        return json.loads(r.read())['access_token']
    except Exception as e:
        print('opensky auth failed:', e); return None


def live_states(bbox, auth=None, errors=None):
    """Current aircraft state vectors in bbox=(W,S,E,N). Anonymous OK.

    ``errors``: optional list. A non-200 response or a request-level failure
    appends a short reason instead of returning the same bare ``[]`` as "no
    aircraft in the bbox right now" -- a rate-limited/unreachable OpenSky
    otherwise looks identical to genuinely clear airspace (rebuild-spec Ground
    Rule #1). Kept optional/default-None so existing callers are unaffected.
    """
    W, S, E, N = bbox
    url = (f'https://opensky-network.org/api/states/all?lamin={S}&lomin={W}'
           f'&lamax={N}&lomax={E}')
    st, b = _get(url, auth)
    if st != 200 or not isinstance(b, bytes):
        if errors is not None:
            detail = b if isinstance(b, str) else (b[:150].decode('utf-8', 'replace') if isinstance(b, bytes) else repr(b))
            errors.append(f'opensky states/all: HTTP {st} {detail}')
        return []
    out = []
    for s in (json.loads(b).get('states') or []):
        # OpenSky state vector layout
        icao, cs, _, _, _, lon, lat, baro, ongnd, vel, hdg, vsi, _, geo = s[:14]
        if lat is None or lon is None:
            continue
        # Rebuild-spec Section 8 item 1: ADS-B barometric altitude is
        # referenced to the 29.92 inHg standard atmosphere, not true MSL --
        # using it uncorrected can be off by 500-2000 ft away from standard
        # pressure. GNSS altitude (`geo`) needs no such correction, so prefer
        # it, but RECORD which one is in play so a caller can apply (or skip)
        # the METAR altimeter correction accordingly -- see
        # aircraft_fire_edge.baro_to_msl_m / aircraft_tracker.poll_once.
        alt_source = 'geo' if geo is not None else ('baro' if baro is not None else None)
        out.append({'icao24': icao, 'callsign': (cs or '').strip(),
                    'lat': lat, 'lon': lon, 'alt_m': geo if geo is not None else baro,
                    'alt_source': alt_source, 'baro_alt_m': baro, 'geo_alt_m': geo,
                    'spd_ms': vel, 'hdg': hdg, 'on_ground': ongnd})
    return out


# ---------------- multi-feed aggregation (rebuild-spec Section 8 item 6) ----
# "OpenSky has poor low-altitude mountain coverage -- exactly where tankers
# fly. Aggregate multiple feeds (OpenSky + adsb.lol + adsb.fi +
# airplanes.live)." adsb.lol and adsb.fi are free community ADS-B aggregators
# (readsb/tar1090-family, no auth) verified live 2026-09-17; airplanes.live
# requires emailing for API permission (its endpoint returned an explicit
# "contact us" response, not data) -- a real, documented gate, kept as an
# honest stub below rather than worked around.

def _bbox_to_point_radius_nm(bbox):
    """These feeds take a center point + radius (nautical miles), not a
    bbox. Use the distance from the bbox centroid to its farthest corner so
    the circle fully covers the original box (over-covers the corners, which
    just means a few extra irrelevant aircraft to filter, not a coverage
    gap)."""
    W, S, E, N = bbox
    lat0 = (S+N)/2.0; lon0 = (W+E)/2.0
    mx = 111.32*math.cos(math.radians(lat0)); my = 111.32
    dx = max(abs(E-lon0), abs(W-lon0)) * mx
    dy = max(abs(N-lat0), abs(S-lat0)) * my
    r_km = math.hypot(dx, dy)
    return lat0, lon0, min(r_km * 0.539957, 250.0)  # km->nm; these APIs cap around 250nm


def _map_community_row(ac):
    """Map an adsb.lol/adsb.fi aircraft record to this module's common state
    shape (same keys live_states() produces, so callers don't care which
    feed an aircraft came from)."""
    icao24 = (ac.get('hex') or '').strip().lower()
    lat, lon = ac.get('lat'), ac.get('lon')
    if not icao24 or lat is None or lon is None:
        return None
    alt_baro = ac.get('alt_baro')
    on_ground = (alt_baro == 'ground')
    baro_m = None if on_ground or alt_baro is None else float(alt_baro) * 0.3048
    geo_ft = ac.get('alt_geom')
    geo_m = float(geo_ft) * 0.3048 if geo_ft is not None else None
    alt_source = 'geo' if geo_m is not None else ('baro' if baro_m is not None else None)
    gs = ac.get('gs')
    return {'icao24': icao24, 'callsign': (ac.get('flight') or '').strip(),
            'lat': lat, 'lon': lon, 'alt_m': geo_m if geo_m is not None else baro_m,
            'alt_source': alt_source, 'baro_alt_m': baro_m, 'geo_alt_m': geo_m,
            'spd_ms': float(gs) * 0.514444 if gs is not None else None,
            'hdg': ac.get('track'), 'on_ground': on_ground,
            # extra, not used by is_firefighting()/poll_once() yet, but free
            # from these feeds and a useful cross-check against
            # aircraft_registry's own cached type/operator if ever needed:
            'adsb_type_code': ac.get('t'), 'adsb_desc': ac.get('desc')}


def _adsblol_states(bbox, errors=None):
    lat0, lon0, r_nm = _bbox_to_point_radius_nm(bbox)
    url = f'https://api.adsb.lol/v2/point/{lat0:.4f}/{lon0:.4f}/{r_nm:.0f}'
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={'User-Agent': 'fire-research'}), timeout=20).read())
    except Exception as e:
        if errors is not None: errors.append(f'adsb.lol: {e!r}')
        return []
    return [r for r in (_map_community_row(ac) for ac in (d.get('ac') or [])) if r]


def _adsbfi_states(bbox, errors=None):
    lat0, lon0, r_nm = _bbox_to_point_radius_nm(bbox)
    url = f'https://opendata.adsb.fi/api/v2/lat/{lat0:.4f}/lon/{lon0:.4f}/dist/{r_nm:.0f}'
    try:
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={'User-Agent': 'fire-research'}), timeout=20).read())
    except Exception as e:
        if errors is not None: errors.append(f'adsb.fi: {e!r}')
        return []
    return [r for r in (_map_community_row(ac) for ac in (d.get('aircraft') or [])) if r]


AIRPLANES_LIVE_GATE_NOTE = ('airplanes.live: access requires emailing contact@airplanes.live '
                             'for API permission (not attempted)')


def _airplaneslive_states(bbox, errors=None):
    """Gated: airplanes.live requires emailing contact@airplanes.live for API
    permission (probed 2026-09-17: the endpoint returns a 200 with an
    explicit "contact us" body, not aircraft data, for an unregistered
    caller). Returns [] WITHOUT appending to `errors` -- this is a permanent,
    already-known limitation, not a transient failure, and appending it every
    poll would spam a live tracking loop with the same "error" forever,
    burying real problems. See AIRPLANES_LIVE_GATE_NOTE for the one-time
    startup message instead. Set AIRPLANES_LIVE_API_KEY if/when access is
    granted and this can be wired to their authenticated endpoint the same
    way opensky_token() is."""
    return []


def live_states_multi(bbox, opensky_auth=None, errors=None):
    """Aggregate every feed above. First-seen-wins by icao24 in the order
    (opensky, adsb.lol, adsb.fi, airplanes.live) -- OpenSky first because
    the rest of this module (alt_source, geo/baro handling) was built and
    tested against its exact shape. A per-feed failure is recorded in
    `errors` and does not block the others; this only returns [] if every
    feed failed or genuinely found nothing, and `errors` explains which.
    This is the function aircraft_tracker.poll_once() should call instead of
    live_states() directly, so OpenSky's known poor low-altitude mountain
    coverage (exactly where tankers fly) is backfilled by the other feeds.
    """
    merged = {}
    for feed_name, fn, extra in (
        ('opensky', live_states, opensky_auth),
        ('adsb.lol', _adsblol_states, None),
        ('adsb.fi', _adsbfi_states, None),
        ('airplanes.live', _airplaneslive_states, None),
    ):
        try:
            rows = fn(bbox, extra, errors=errors) if feed_name == 'opensky' else fn(bbox, errors=errors)
        except Exception as e:
            if errors is not None: errors.append(f'{feed_name}: {e!r}')
            continue
        for row in rows:
            merged.setdefault(row['icao24'], row)
    return list(merged.values())


def is_firefighting(callsign, icao24=None):
    c = (callsign or '').upper().strip()
    if icao24 and icao24.lower() in TANKER_REGISTRY:
        return True
    if icao24:
        try:
            import aircraft_registry as AR
            if AR.lookup(icao24) is not None:
                return True
        except Exception as e:
            # Registry build/download failed -- fall through to callsign
            # heuristics. This is NOT "not a firefighting aircraft"; print so
            # a silent registry outage isn't mistaken for "no fire fleet
            # match" (rebuild-spec Ground Rule #1).
            print(f'  ! aircraft_registry unavailable: {e!r}'[:150], flush=True)
    return any(c.startswith(p) for p in FF_PREFIXES) and len(c) >= 4


def loitering_over(points, fire_bbox, min_points=8, max_spd_ms=110.0,
                   max_spread_km=6.0):
    """Behavioral firefighting detector: group points by aircraft, keep those
    that (a) sit inside the fire bbox, (b) fly slow, and (c) stay CONCENTRATED
    (small spatial spread = orbiting/working the fire, not transiting). This
    catches tankers/air-attack that squawk only a tail number.
    Returns {icao24: [points]} for aircraft judged to be working the fire."""
    W, S, E, N = fire_bbox
    by = {}
    for p in points:
        if not (W <= p['lon'] <= E and S <= p['lat'] <= N):
            continue
        by.setdefault(p['icao24'], []).append(p)
    out = {}
    for icao, ps in by.items():
        if len(ps) < min_points:
            continue
        lat0 = np.mean([q['lat'] for q in ps])
        xs = np.array([q['lon'] for q in ps]) * 111.32 * math.cos(math.radians(lat0))
        ys = np.array([q['lat'] for q in ps]) * 111.32
        spread = math.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        spds = [q['spd_ms'] for q in ps if q.get('spd_ms') is not None]
        if spread <= max_spread_km and (not spds or np.mean(spds) <= max_spd_ms):
            out[icao] = ps
    return out


def poll_live_tracks(bbox, minutes=30, interval_s=20, auth=None, verbose=True):
    """Accumulate firefighting-aircraft positions over a live window by polling
    states/all. Returns list of point dicts. Use on an ACTIVE fire now."""
    auth = auth or opensky_token()
    pts, seen = [], 0
    t_end = time.time() + minutes * 60
    while time.time() < t_end:
        for a in live_states(bbox, auth):
            if a['on_ground'] or not is_firefighting(a['callsign']):
                continue
            pts.append({**a, 't': time.time()})
            seen += 1
        if verbose:
            print(f'  polled; firefighting points so far: {seen}', flush=True)
        time.sleep(interval_s)
    return pts


def drop_points(points, max_alt_m=DROP_MAX_ALT_M, max_spd_ms=DROP_MAX_SPD_MS,
                ground_elev_m=0.0):
    """Keep only LOW+SLOW points (retardant-run candidates). ground_elev_m lets
    you make the altitude test terrain-relative (AGL) instead of MSL."""
    out = []
    for p in points:
        alt = p.get('alt_m')
        spd = p.get('spd_ms')
        if alt is None:
            continue
        agl = alt - ground_elev_m
        if agl <= max_alt_m and (spd is None or spd <= max_spd_ms):
            out.append(p)
    return out


def active_edge(points, alpha_km=0.6):
    """Concave hull (alpha shape) of drop points -> active-edge polygon + the
    dominant drop-line orientation (the fire edge the aircraft are working).
    Falls back to convex hull if too few points. Returns dict."""
    if len(points) < 3:
        return None
    lat0 = float(np.mean([p['lat'] for p in points]))
    mx = 111.32 * math.cos(math.radians(lat0)); my = 111.32
    xy = np.array([[(p['lon']) * mx, (p['lat']) * my] for p in points])  # km-ish
    from shapely.geometry import MultiPoint, LineString
    mp = MultiPoint([tuple(v) for v in xy])
    hull = mp.convex_hull
    # crude alpha shape: union of buffered points minus interior -> boundary band
    from shapely.ops import unary_union
    band = unary_union([p.buffer(alpha_km) for p in mp.geoms]).buffer(-alpha_km*0.5)
    edge = band if (band.geom_type == 'Polygon' and band.area > 0) else hull
    # dominant orientation via PCA of the points (the drop line direction)
    c = xy - xy.mean(0)
    if len(c) >= 2:
        _, _, V = np.linalg.svd(c, full_matrices=False)
        ang = math.degrees(math.atan2(V[0][1], V[0][0])) % 180
    else:
        ang = None
    def to_ll(g):
        from shapely.ops import transform
        return transform(lambda x, y: (x / mx, y / my), g)
    return {'edge_ll': to_ll(edge), 'convex_hull_ll': to_ll(hull),
            'n_drop_points': len(points), 'drop_line_bearing_deg': ang}


if __name__ == '__main__':
    import sys
    # live probe over a bbox (default: a wide CA box) to see what's flying now
    bbox = (-124.0, 38.5, -121.5, 40.5)
    if len(sys.argv) > 4:
        bbox = tuple(float(x) for x in sys.argv[1:5])
    auth = opensky_token()
    print('OpenSky auth:', 'YES (historical enabled)' if auth else 'anonymous (live only)')
    states = live_states(bbox, auth)
    ff = [a for a in states if is_firefighting(a['callsign'])]
    print(f'{len(states)} aircraft in bbox, {len(ff)} match firefighting callsigns:')
    for a in ff[:20]:
        print(f"  {a['callsign']:9s} {a['icao24']} ({a['lat']:.3f},{a['lon']:.3f}) "
              f"alt {a['alt_m']} m spd {a['spd_ms']} m/s")
