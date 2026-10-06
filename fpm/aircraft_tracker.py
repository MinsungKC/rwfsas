"""Persistent firefighting-aircraft TRACKER.

Runs continuously (daemon or cron) and records the PATHS of firefighting
aircraft over a fire, with timestamps. Classifies DROP RUNS -- the low+slow
segments where tankers lay retardant (phoscheck) or helicopters drop water --
because those paths trace the ACTIVE/DEFENDED perimeter. The fusion then uses
the RECENT drop paths (last ~N hours), not the current aircraft location.

Store: <store>/<fire>.jsonl, one JSON position per line:
  {icao, callsign, t(unix), lat, lon, alt_m, spd_ms, alt_source, alt_msl_corrected,
   metar_station, metar_altim_inhg, agl_m, agl_unavailable, registry_operator,
   registry_model, drop, kind}
alt_m is true MSL: GNSS (alt_source='geo') needs no correction; barometric
(alt_source='baro', std-pressure-referenced) is corrected to MSL via the
nearest METAR's altimeter setting when available (alt_msl_corrected=True) --
see aircraft_fire_edge.metar_altimeter / baro_to_msl_m and rebuild-spec
Section 8 item 1.
`kind` is the aircraft_registry role (vlat/lat/sat/scooper/leadplane_or_
airattack/helitanker_type1-3) when the icao24 resolves to a known wildland-
fire-operator airframe (Section 8 item 2), else a coarse water(heli)/
retardant(tanker)/unknown guess from the speed+AGL gate alone.

Usage:
  poll_once(fire, lat, lon)          # one sample -> append (run from cron/loop)
  run(fire, lat, lon, minutes=…)     # blocking loop (daemon)
  recent_paths(fire, hours=2)        # -> [{callsign, kind, drop, coords, t0, t1}]
"""
import os, sys, json, time, math, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aircraft_fire_edge as AF
import aircraft_registry as AR
import airport_db as APT

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_aircraft_tracks')
# drop-run gates: tanker retardant runs ~150-500ft AGL, ~120-160kt; heli water
# drops very low + slow. AGL from the fire's local ground elevation.
DROP_MAX_AGL_M = 600.0
TANKER_SPD = (55, 100)     # m/s (~107-194 kt) on a run
HELI_SPD   = (0, 45)       # m/s slow/hover for water drops
# airport-pattern/approach exclusion (Section 8 item 5): within this radius
# of a charted field AND within this height above ITS elevation looks like
# ordinary traffic-pattern/approach work, not fire-orbit behavior.
AIRPORT_EXCLUDE_KM = 8.0
AIRPORT_EXCLUDE_AGL_M = 500.0


def _filter_airport_pattern_loiterers(loit, states_by_icao):
    """Drop icao24s from the (weak) loitering-behavior set whose CURRENT
    position looks like ordinary airport traffic-pattern/approach work near
    a real charted airport, rather than orbiting a fire -- rebuild-spec
    Section 8 item 5. This ONLY touches the loitering fallback: an aircraft
    already confirmed by aircraft_registry or a firefighting callsign prefix
    is never filtered here, because a real airtanker base sitting at an
    airport (most of them do) must not be excluded for being near one."""
    out = {}
    for icao, pts in loit.items():
        s = states_by_icao.get(icao)
        if s is None:
            out[icao] = pts; continue   # not in this live snapshot, can't evaluate -- keep
        try:
            nearest = APT.nearest_airport(s['lat'], s['lon'], max_km=AIRPORT_EXCLUDE_KM)
        except Exception as e:
            print(f'  ! airport_db unavailable, skipping pattern-exclusion check: {e!r}'[:150], flush=True)
            out[icao] = pts; continue   # DB unavailable is not "no airport nearby" -- keep, don't guess
        if nearest is None:
            out[icao] = pts; continue
        apt, dist_km = nearest
        alt = s.get('alt_m')
        if alt is not None and (alt - apt['elev_ft']*0.3048) <= AIRPORT_EXCLUDE_AGL_M:
            continue   # pattern/approach altitude within AIRPORT_EXCLUDE_KM of a real field -- excluded
        out[icao] = pts
    return out


def _ground_elev(lat, lon):
    """Returns None on fetch failure. Callers must NOT treat that as sea
    level (0 m): a silently-wrong ground elevation is exactly how the AGL
    misclassification in FIRE_PERIMETER_REBUILD_PROMPT.md Section 8 item 1
    happens -- in mountainous terrain a wrong baseline of hundreds to
    thousands of feet flips a real drop run to "orbit" or vice versa."""
    try:
        d = json.loads(urllib.request.urlopen(
            f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}', timeout=15).read())
        return float(d['elevation'][0])
    except Exception as e:
        print(f'  ! ground elevation unavailable ({lat:.3f},{lon:.3f}): {e!r}'[:150], flush=True)
        return None


def _recent_points(fire, minutes=20):
    """This fire's own stored position samples from the last `minutes` --
    real repeated observations, for loitering_over's concentration test.
    (A fresh fire has no history yet; that's fine, min_points=4 just means
    nothing loiters on the first few polls until history accumulates.)"""
    p = os.path.join(STORE, f'{fire}.jsonl')
    if not os.path.exists(p):
        return []
    cut = time.time() - minutes*60
    out = []
    for line in open(p, encoding='utf-8'):
        try:
            d = json.loads(line)
            if d.get('t', 0) >= cut:
                out.append(d)
        except Exception:
            pass
    return out


def poll_once(fire, lat, lon, radius_km=45, ground_elev=None, metar=None):
    """One snapshot -> classify + append firefighting-aircraft positions."""
    os.makedirs(STORE, exist_ok=True)
    if ground_elev is None: ground_elev = _ground_elev(lat, lon)
    elev_known = ground_elev is not None
    # One regional METAR altimeter setting per poll (aircraft in-bbox share
    # essentially the same local pressure) to correct any BARO-sourced
    # altitude to true MSL. metar=None (not fetched by caller) is
    # distinguished from metar={} (fetch attempted, unavailable) so run()
    # can retry only when genuinely needed.
    if metar is None:
        metar = AF.metar_altimeter(lat, lon) or {}
    altim_inhg = metar.get('altim_inhg')
    bbox = (lon-radius_km/111.0/math.cos(math.radians(lat)), lat-radius_km/111.0,
            lon+radius_km/111.0/math.cos(math.radians(lat)), lat+radius_km/111.0)
    errors = []
    # Multi-feed (Section 8 item 6): OpenSky + adsb.lol + adsb.fi (+
    # airplanes.live once permitted), deduplicated by icao24 -- OpenSky alone
    # has known poor low-altitude mountain coverage, exactly where tankers
    # fly. A per-feed failure lands in `errors` without blocking the others.
    states = AF.live_states_multi(bbox, AF.opensky_token(), errors=errors)
    if errors:
        print('  ! ADS-B feed issue(s):', '; '.join(errors)[:200], flush=True)
    # loitering_over needs REPEATED observations to mean anything ("stays
    # CONCENTRATED" per its own docstring); calling it on just this one live
    # snapshot made every aircraft trivially "concentrated" (a single point
    # has zero spread by definition), so min_points=1 here silently accepted
    # almost any slow-moving aircraft as "loitering" -- a real, separate
    # contributor to the aircraft-misidentification problem this whole
    # Section 8 pass is fixing. Feed it the fire's own recent stored history
    # (real repeated samples) plus this snapshot, and require several points.
    history = _recent_points(fire, minutes=20)
    loit_input = ([{'icao24': d['icao'], 'lat': d['lat'], 'lon': d['lon'], 'spd_ms': d.get('spd_ms')}
                   for d in history]
                  + [{'icao24': s['icao24'], 'lat': s['lat'], 'lon': s['lon'], 'spd_ms': s.get('spd_ms')}
                     for s in states])
    loit = AF.loitering_over(loit_input, bbox, min_points=4, max_spread_km=10)
    states_by_icao = {s['icao24']: s for s in states}
    loit = _filter_airport_pattern_loiterers(loit, states_by_icao)
    now = time.time(); rows = []
    for s in states:
        if s['on_ground']: continue
        ff = AF.is_firefighting(s['callsign'], s['icao24']) or s['icao24'] in loit
        if not ff: continue
        try:
            reg = AR.lookup(s['icao24'])
        except Exception as e:
            reg = None
            print(f'  ! aircraft_registry lookup failed: {e!r}'[:150], flush=True)
        alt = s.get('alt_m'); spd = s.get('spd_ms')
        alt_corrected = False
        if alt is not None and s.get('alt_source') == 'baro' and altim_inhg is not None:
            # GNSS altitude needs no correction; std-pressure baro does.
            alt = AF.baro_to_msl_m(alt, altim_inhg); alt_corrected = True
        # AGL is None (not a wrong MSL-baselined number) whenever the ground
        # elevation fetch failed -- see _ground_elev docstring.
        agl = (alt - ground_elev) if (alt is not None and elev_known) else None
        reg_role = reg['role'] if reg else None
        # A confirmed registry role (rebuild-spec Section 8 item 2) is a much
        # stronger platform-class signal than the old callsign-prefix guess;
        # use it when we have it, and only fall back to the guess otherwise.
        if reg_role:
            is_heli = reg_role.startswith('helitanker_')
        else:
            cs = (s['callsign'] or '').upper()
            is_heli = cs.startswith(('COPTER','HELI','H')) and not cs.startswith('HL')
        kind = reg_role or 'unknown'; drop = False
        if agl is not None and agl <= DROP_MAX_AGL_M and spd is not None:
            if is_heli and HELI_SPD[0] <= spd <= HELI_SPD[1]:
                drop = True
                if kind == 'unknown': kind = 'water(heli)'
            elif TANKER_SPD[0] <= spd <= TANKER_SPD[1]:
                drop = True
                if kind == 'unknown': kind = 'retardant(tanker)'
        rows.append({'icao': s['icao24'], 'callsign': s['callsign'], 't': now,
                     'lat': s['lat'], 'lon': s['lon'], 'alt_m': alt, 'spd_ms': spd,
                     'alt_source': s.get('alt_source'), 'alt_msl_corrected': alt_corrected,
                     'metar_station': metar.get('station'), 'metar_altim_inhg': altim_inhg,
                     'agl_m': agl, 'agl_unavailable': not elev_known,
                     'registry_operator': reg['operator'] if reg else None,
                     'registry_model': reg['model'] if reg else None,
                     'drop': drop, 'kind': kind})
    if rows:
        with open(os.path.join(STORE, f'{fire}.jsonl'), 'a', encoding='utf-8') as fh:
            for r in rows: fh.write(json.dumps(r)+'\n')
    return rows


def run(fire, lat, lon, minutes=120, interval_s=30, radius_km=45):
    ge = _ground_elev(lat, lon)
    metar = AF.metar_altimeter(lat, lon) or {}; metar_t = time.time()
    t_end = time.time()+minutes*60
    altim_desc = 'unknown' if not metar else f'{metar["altim_inhg"]:.2f}inHg@{metar.get("station")}'
    print(f'tracking {fire} (ground {"unknown" if ge is None else f"{ge:.0f}m"}, altim {altim_desc}) '
          f'every {interval_s}s for {minutes}min', flush=True)
    print(f'  note: {AF.AIRPLANES_LIVE_GATE_NOTE} -- feeds in use: opensky, adsb.lol, adsb.fi', flush=True)
    while time.time() < t_end:
        try:
            if ge is None:
                ge = _ground_elev(lat, lon)   # retry each cycle until it resolves,
                                               # rather than staying wrong all session
            # Pressure drifts over hours, not seconds -- refresh at most every
            # ~20min, but retry immediately every cycle while unavailable.
            if not metar or (time.time()-metar_t) > 1200:
                metar = AF.metar_altimeter(lat, lon) or {}; metar_t = time.time()
            r = poll_once(fire, lat, lon, radius_km, ge, metar)
            nd = sum(1 for x in r if x['drop'])
            print(f'  {time.strftime("%H:%M:%S")} {len(r)} a/c ({nd} dropping)', flush=True)
        except Exception as e: print('  poll err', repr(e)[:60], flush=True)
        time.sleep(interval_s)


def recent_paths(fire, hours=2, seed_from_history=None):
    """Recent per-aircraft paths from the store, split into transit vs drop
    segments. If the store is thin and seed_from_history=(lat,lon) is given,
    backfill from OpenSky historical /tracks for the firefighting aircraft."""
    p = os.path.join(STORE, f'{fire}.jsonl')
    pts = []
    if os.path.exists(p):
        cut = time.time()-hours*3600
        for line in open(p, encoding='utf-8'):
            try:
                d = json.loads(line)
                if d['t'] >= cut: pts.append(d)
            except Exception: pass
    if seed_from_history is not None:
        pts += _seed_history(fire, *seed_from_history, hours)
    by = {}
    for d in pts: by.setdefault(d['icao'], []).append(d)
    out = []
    for icao, ps in by.items():
        ps.sort(key=lambda x: x['t'])
        seg = []; segdrop = ps[0].get('drop', False)
        for d in ps:
            if d.get('drop', False) != segdrop and seg:
                out.append(_mkseg(seg, segdrop)); seg = []; segdrop = d.get('drop', False)
            seg.append(d)
        if seg: out.append(_mkseg(seg, segdrop))
    return [s for s in out if len(s['coords']) >= 2]


def _mkseg(seg, drop):
    return {'callsign': seg[-1].get('callsign'), 'icao': seg[-1]['icao'],
            'kind': seg[-1].get('kind','unknown'), 'drop': bool(drop),
            'coords': [(d['lon'], d['lat']) for d in seg],
            't0': seg[0]['t'], 't1': seg[-1]['t']}


def _seed_history(fire, lat, lon, hours):
    """Backfill from OpenSky historical /tracks for firefighting a/c near the fire.

    Returns [] both when no OAuth creds are configured (expected; anonymous
    OpenSky has no historical access) and when a request fails -- those are
    different situations, so the latter is printed rather than silent."""
    tok = AF.opensky_token()
    if not tok: return []
    import re
    beg = int(time.time()-hours*3600); end = int(time.time())
    def g(u):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'x','Authorization':'Bearer '+tok}), timeout=40).read())
        except Exception as e:
            print(f'  ! OpenSky historical unavailable ({u.split("?")[0]}): {e!r}'[:150], flush=True)
            return None
    fl = g(f'https://opensky-network.org/api/flights/all?begin={beg}&end={end}') or []
    ge = _ground_elev(lat, lon); out = []; bbox=(lon-0.4,lat-0.4,lon+0.4,lat+0.4)
    cand = [f for f in fl if AF.is_firefighting((f.get('callsign') or ''))][:15]
    for f in cand:
        tr = g(f'https://opensky-network.org/api/tracks/all?icao24={f["icao24"]}&time={f["firstSeen"]}')
        if not tr: continue
        for w in tr.get('path', []):
            t,la,lo,alt,trk,ong = w[0],w[1],w[2],w[3],w[4],w[5]
            if la is None or lo is None or ong: continue
            if not (bbox[1]<=la<=bbox[3] and bbox[0]<=lo<=bbox[2]): continue
            agl = (alt-ge) if alt is not None else None
            out.append({'icao': f['icao24'], 'callsign': (f.get('callsign') or '').strip(),
                        't': t, 'lat': la, 'lon': lo, 'alt_m': alt, 'spd_ms': None,
                        'agl_m': agl, 'drop': bool(agl is not None and agl<=DROP_MAX_AGL_M), 'kind': 'hist'})
    return out


def _hav_km(p1, p2):
    """Great-circle distance in km between two (lon,lat) points."""
    lon1, lat1 = p1; lon2, lat2 = p2
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(a)))


def _greedy_cluster(points, radius_km):
    """points: [(lon,lat), ...]. Returns clusters (largest first), each
    {'members': [idx,...], 'centroid': (lon,lat)}. Deliberately simple
    (single-pass, order-dependent) -- good enough to separate two
    well-spaced repeat locations, which is all a shuttle needs."""
    clusters = []
    for i, p in enumerate(points):
        best, best_d = None, radius_km
        for c in clusters:
            d = _hav_km(p, c['centroid'])
            if d <= best_d:
                best, best_d = c, d
        if best is not None:
            best['members'].append(i)
            lons = [points[j][0] for j in best['members']]
            lats = [points[j][1] for j in best['members']]
            best['centroid'] = (sum(lons)/len(lons), sum(lats)/len(lats))
        else:
            clusters.append({'members': [i], 'centroid': p})
    clusters.sort(key=lambda c: -len(c['members']))
    return clusters


def detect_shuttles(fire, hours=6, min_cycles=2, cluster_radius_km=1.2):
    """Detect a repeating water/base <-> fire shuttle in one aircraft's
    recent DROP-segment locations (rebuild-spec Section 8 item 4): "detecting
    the repeating shuttle between a fixed water point and a varying drop
    point is a very strong, low-false-positive fire-location signal."

    A real shuttle alternates between a TIGHT cluster (reload base/water
    source -- fixed) and a looser cluster near the working fire; this is
    deliberately not "any two drops, anywhere" (min_cycles defaults to 2 real
    round trips, and the alternation itself is checked, not just visit
    counts -- two isolated one-off drops at two random places would not
    pass).

    Returns a list of {icao, callsign, kind, n_round_trips, base:
    {centroid, n_visits, spread_km}, away: {centroid, n_visits, spread_km,
    points}, t0, t1}, sorted by n_round_trips descending. The 'away' cluster
    is the useful fire-location evidence: the base end is already known
    (it's fixed), so what's informative is where the *other* end keeps
    landing -- that's the aircraft's own repeated confirmation of where the
    active edge is.
    """
    segs = recent_paths(fire, hours=hours)
    by_icao = {}
    for s in segs:
        if not s['drop']:
            continue
        by_icao.setdefault(s['icao'], []).append(s)
    out = []
    for icao, dsegs in by_icao.items():
        dsegs.sort(key=lambda s: s['t0'])
        if len(dsegs) < 2*min_cycles:
            continue
        centroids = [(sum(c[0] for c in s['coords'])/len(s['coords']),
                      sum(c[1] for c in s['coords'])/len(s['coords'])) for s in dsegs]
        clusters = _greedy_cluster(centroids, cluster_radius_km)
        if len(clusters) < 2 or len(clusters[1]['members']) < min_cycles:
            continue
        top2 = clusters[:2]
        # Label each drop segment by which of the two dominant stops it's
        # at; segments absorbed into smaller/noise clusters are dropped from
        # the alternation count (a genuine two-stop shuttle should already
        # cover most observations in these top two).
        member_label = {m: ci for ci, c in enumerate(top2) for m in c['members']}
        seq = sorted(((dsegs[i]['t0'], member_label[i]) for i in member_label), key=lambda x: x[0])
        switches = sum(1 for a, b in zip(seq, seq[1:]) if a[1] != b[1])
        if switches < 2*min_cycles - 1:
            continue

        def spread_km(c):
            pts = [centroids[i] for i in c['members']]
            return max((_hav_km(p, c['centroid']) for p in pts), default=0.0)
        c0, c1 = top2
        base, away = (c0, c1) if spread_km(c0) <= spread_km(c1) else (c1, c0)
        rep = dsegs[0]
        out.append({
            'icao': icao, 'callsign': rep.get('callsign'), 'kind': rep.get('kind'),
            'n_round_trips': switches // 2,
            'base': {'centroid': base['centroid'], 'n_visits': len(base['members']),
                     'spread_km': round(spread_km(base), 3)},
            'away': {'centroid': away['centroid'], 'n_visits': len(away['members']),
                     'spread_km': round(spread_km(away), 3),
                     'points': [centroids[i] for i in away['members']]},
            't0': dsegs[0]['t0'], 't1': dsegs[-1]['t1'],
        })
    out.sort(key=lambda r: -r['n_round_trips'])
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run','poll','paths','shuttles'])
    ap.add_argument('fire'); ap.add_argument('lat', type=float); ap.add_argument('lon', type=float)
    ap.add_argument('--minutes', type=int, default=120); ap.add_argument('--hours', type=float, default=2)
    a = ap.parse_args()
    if a.cmd=='run': run(a.fire, a.lat, a.lon, a.minutes)
    elif a.cmd=='poll': print(len(poll_once(a.fire, a.lat, a.lon)), 'a/c logged')
    elif a.cmd=='shuttles': print(json.dumps(detect_shuttles(a.fire, a.hours), indent=2))
    else: print(json.dumps(recent_paths(a.fire, a.hours, seed_from_history=(a.lat,a.lon)), indent=2)[:2000])
