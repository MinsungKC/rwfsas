"""Pattern + terrain aware firefighting-aircraft DROP detection (rebuild spec 8).

Fixes the two defects in aircraft_tracker's per-sample drop flag:
  1. AGL was alt - the FIRE's single ground elevation, so a plane over higher/lower
     terrain got a wrong AGL (and DROP_MAX_AGL_M=600 let cruise passes count). Here
     AGL = alt - the DEM elevation UNDER THE AIRCRAFT, from a cached coarse grid.
  2. Classification was instantaneous. A real drop run sits at a LOCAL ALTITUDE
     MINIMUM (descend in / climb out), low AGL, speed in tanker/heli range, near the
     fire. (Poll cadence ~45s is too coarse for a full HMM; this is the sparse-data
     form of the pattern test.)

recent_drop_paths(fire, hours) -> [{coords, kind, icao, callsign, t0, t1}] for use
by the perimeter fusion. DEM via Open-Meteo elevation (no auth); falls back to a
flat plane if it can't be fetched (then AGL == the old behaviour, fail-safe).
"""
import os, sys, json, math, time, urllib.request, urllib.error

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_aircraft_tracks')
TANKER_MAX_AGL = 180.0     # m -- LAT/VLAT retardant run height (~150-500 ft)
HELI_MAX_AGL = 90.0        # m -- helicopter water drop, very low
TANKER_SPD = (48.0, 108.0) # m/s (~93-210 kt) on a run
HELI_SPD = (0.0, 42.0)     # m/s slow/hover
NEAR_FIRE_KM = 10.0


class DEM:
    """Coarse elevation grid over a bbox (fetched once, bilinear-sampled)."""
    def __init__(self, bbox, n=20):
        self.ok = False
        W, S, E, N = bbox
        self.W, self.S, self.E, self.N = W, S, E, N
        self.n = n
        lats = [S + (N-S)*i/(n-1) for i in range(n)]
        lons = [W + (E-W)*j/(n-1) for j in range(n)]
        self.lats, self.lons = lats, lons
        pts = [(la, lo) for la in lats for lo in lons]
        # disk cache keyed on rounded bbox (terrain is static) -> avoid re-fetch/429
        os.makedirs(STORE, exist_ok=True)
        ck = os.path.join(STORE, '_dem_%.2f_%.2f_%.2f_%.2f_%d.json' % (W, S, E, N, n))
        if os.path.exists(ck):
            try:
                self.grid = json.load(open(ck)); self.ok = True; return
            except Exception:
                pass
        elev = []
        try:
            for k in range(0, len(pts), 100):
                chunk = pts[k:k+100]
                la = ','.join(f'{p[0]:.4f}' for p in chunk)
                lo = ','.join(f'{p[1]:.4f}' for p in chunk)
                u = f'https://api.open-meteo.com/v1/elevation?latitude={la}&longitude={lo}'
                d = None
                for attempt in range(4):                    # retry/backoff on 429
                    try:
                        d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())
                        break
                    except urllib.error.HTTPError as he:
                        if he.code == 429 and attempt < 3:
                            time.sleep(6*(attempt+1)); continue
                        raise
                elev += list(d['elevation'])
            if len(elev) == len(pts):
                self.grid = [elev[i*n:(i+1)*n] for i in range(n)]   # [lat_idx][lon_idx]
                self.ok = True
                try:
                    json.dump(self.grid, open(ck, 'w'))
                except Exception:
                    pass
        except Exception as e:
            print('  DEM fetch failed, flat fallback:', repr(e)[:60], flush=True)

    def sample(self, lat, lon):
        if not self.ok:
            return 0.0
        n = self.n
        fi = (lat - self.S)/(self.N - self.S)*(n-1) if self.N > self.S else 0
        fj = (lon - self.W)/(self.E - self.W)*(n-1) if self.E > self.W else 0
        i0 = max(0, min(n-2, int(fi))); j0 = max(0, min(n-2, int(fj)))
        di = fi - i0; dj = fj - j0
        g = self.grid
        return (g[i0][j0]*(1-di)*(1-dj) + g[i0][j0+1]*(1-di)*dj +
                g[i0+1][j0]*di*(1-dj) + g[i0+1][j0+1]*di*dj)


def _load(fire, hours):
    p = os.path.join(STORE, f'{fire}.jsonl')
    if not os.path.exists(p):
        return []
    cut = time.time() - hours*3600
    rows = []
    for ln in open(p, encoding='utf-8'):
        try:
            d = json.loads(ln)
            if d['t'] >= cut and d.get('lat') is not None:
                rows.append(d)
        except Exception:
            pass
    return rows


def classify_track(pts, dem):
    """One aircraft's time-ordered points -> drop point indices (with kind)."""
    pts = sorted(pts, key=lambda x: x['t'])
    agl = []
    for p in pts:
        a = p.get('alt_m')
        agl.append(None if a is None else a - dem.sample(p['lat'], p['lon']))
    hits = []
    for i, p in enumerate(pts):
        a = agl[i]; s = p.get('spd_ms')
        if a is None or s is None:
            continue
        cs = (p.get('callsign') or '').upper()
        is_heli = cs.startswith(('COPTER', 'HELI', 'HELO'))
        prev_a = agl[i-1] if i > 0 else None
        nxt_a = agl[i+1] if i < len(pts)-1 else None
        # local altitude minimum (within 40m of being the lowest of its neighbours)
        localmin = (prev_a is None or a <= prev_a + 40) and (nxt_a is None or a <= nxt_a + 40)
        if not localmin:
            continue
        if is_heli and a <= HELI_MAX_AGL and HELI_SPD[0] <= s <= HELI_SPD[1]:
            hits.append((i, 'water(heli)'))
        elif (not is_heli) and a <= TANKER_MAX_AGL and TANKER_SPD[0] <= s <= TANKER_SPD[1]:
            hits.append((i, 'retardant(tanker)'))
    return pts, hits


def _segments(pts, hits):
    """Group consecutive drop hits into path segments (coords)."""
    if not hits:
        return []
    out = []; run = [hits[0]]
    for h in hits[1:]:
        if h[0] == run[-1][0] + 1:
            run.append(h)
        else:
            out.append(run); run = [h]
    out.append(run)
    segs = []
    for run in out:
        i0 = max(0, run[0][0]-1); i1 = min(len(pts)-1, run[-1][0]+1)   # pad by one for a line
        coords = [[pts[k]['lon'], pts[k]['lat']] for k in range(i0, i1+1)]
        if len(coords) < 2:
            # single-sample drop: make a tiny 2-pt stub so it renders/buffers
            lo, la = pts[run[0][0]]['lon'], pts[run[0][0]]['lat']
            coords = [[lo, la], [lo+1e-4, la+1e-4]]
        segs.append({'coords': coords, 'kind': run[-1][1],
                     'icao': pts[run[0][0]]['icao'], 'callsign': pts[run[0][0]].get('callsign'),
                     't0': pts[run[0][0]]['t'], 't1': pts[run[-1][0]]['t']})
    return segs


def recent_drop_paths(fire, lat, lon, hours=3.0):
    rows = _load(fire, hours)
    if not rows:
        return []
    m = math.cos(math.radians(lat))
    bbox = (lon-NEAR_FIRE_KM/111.0/m, lat-NEAR_FIRE_KM/111.0,
            lon+NEAR_FIRE_KM/111.0/m, lat+NEAR_FIRE_KM/111.0)
    dem = DEM(bbox)
    by = {}
    for d in rows:
        by.setdefault(d['icao'], []).append(d)
    out = []
    for icao, pts in by.items():
        p2, hits = classify_track(pts, dem)
        out += _segments(p2, hits)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('fire'); ap.add_argument('lat', type=float); ap.add_argument('lon', type=float)
    ap.add_argument('--hours', type=float, default=6.0)
    a = ap.parse_args()
    rows = _load(a.fire, a.hours)
    old = sum(1 for r in rows if r.get('drop'))
    segs = recent_drop_paths(a.fire, a.lat, a.lon, a.hours)
    print(f'{a.fire}: {len(rows)} rows / {len({r["icao"] for r in rows})} aircraft in last {a.hours}h')
    print(f'  OLD per-sample flag: {old} drop points')
    print(f'  NEW pattern+DEM:     {len(segs)} drop segments')
    for s in segs:
        print(f'    {s["kind"]:18s} {s.get("callsign") or s["icao"]:10s} {len(s["coords"])} pts')
