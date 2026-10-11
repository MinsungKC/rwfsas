"""LIVE learned-field v3 perimeter for a fire (the validated best model, run live).

Pipeline per call:
  1. official mapped base + its true as-of time (poly_PolygonDateTime) = anchor
  2. live detections GOES-19 + GOES-18 + VIIRS over [t_map, now]  (champion_live feed)
  3. learned_field_v3 -> per-pixel burned-probability field -> TOPK perimeter
  4. conformal p10/p90 acreage envelope (champion_live's fitted quantiles)
  5. render: Esri satellite basemap + all detections + base(dashed) + predicted(green)
             + p10/p90 envelope + origin  (the standing image convention)

Deployable model trained on ALL cohort fires with ZeroDem -> inference also uses
ZeroDem so the 55 features match. Dome/Floriston (Sept 2026) are not in the
older cohort, so this is genuinely out-of-sample.
"""
import sys, os, json, math, pickle, importlib.util, urllib.request, urllib.parse
from datetime import datetime, timezone

FSL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
sys.path.insert(0, FSL)                       # `data.abi_fire_area` (GOES reader) resolves here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))  # firms_fixed (VIIRS)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(HERE, '_frozen', 'deployable_field_v3.pkl')
ABI_CACHE = os.path.join(HERE, '_frozen', '_abi_live')

import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import transform, unary_union

# learned-field inference core (import by path; register for py3.14 dataclass)
_spec = importlib.util.spec_from_file_location('learned_field_v3', os.path.join(FSL, 'models', 'learned_field_v3.py'))
LF = importlib.util.module_from_spec(_spec); sys.modules['learned_field_v3'] = LF; _spec.loader.exec_module(LF)

CONF_LO, CONF_HI = -0.34, 0.29                # champion_live conformal half-widths (log-acre), ~0.80 coverage


def _get(url):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'x'}), timeout=40).read())
    except Exception:
        return None


def mapped_base(lat, lon, pad=0.06):
    """Official perimeter + its true as-of time (poly_PolygonDateTime)."""
    env = json.dumps({'xmin': lon-pad, 'ymin': lat-pad, 'xmax': lon+pad, 'ymax': lat+pad,
                      'spatialReference': {'wkid': 4326}})
    for base in ['https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query',
                 'https://bz1uwwpkuinzbk94.svcs5.arcgis.com/bz1uwWPKUInZBK94/arcgis/rest/services/CA_Perimeters_NIFC_FIRIS_public_view/FeatureServer/0/query']:
        d = _get(base + '?' + urllib.parse.urlencode({'where': '1=1', 'geometry': env, 'geometryType': 'esriGeometryEnvelope',
                  'inSR': '4326', 'spatialRel': 'esriSpatialRelIntersects', 'outFields': '*', 'f': 'geojson',
                  'geometryPrecision': '6', 'resultRecordCount': '5'}))
        if d and d.get('features'):
            f = max(d['features'], key=lambda x: shape(x['geometry']).area)
            p = f.get('properties', {})
            t_map = None
            for kf in ('poly_PolygonDateTime', 'PolygonDateTime', 'poly_DateCurrent', 'DateCurrent', 'poly_CreateDate', 'CreateDate'):
                v = p.get(kf)
                if isinstance(v, (int, float)) and v > 1e11:
                    t_map = datetime.fromtimestamp(v/1000, timezone.utc); break
            return shape(f['geometry']), p, t_map
    return None, None, None


def _live_dets(bbox, t_map, now):
    """Point detections {lat,lon,frp,sensor,acq} from the same sources the live
    loop uses: GOES-18/19 FDCC (data.abi_fire_area) + VIIRS FIRMS (firms_fixed)."""
    dets = []
    try:
        from data.abi_fire_area import granules, fetch_granule, read_mask_granule
        os.makedirs(ABI_CACHE, exist_ok=True)
        for sat in ('GOES-19', 'GOES-18'):
            try:
                for u in sorted(granules(sat, t_map, now))[::2][:60]:
                    try:
                        for c in read_mask_granule(fetch_granule(u, ABI_CACHE), bbox=bbox, include_nonfire=False):
                            if c.get('state') != 'fire':
                                continue
                            frp = c.get('power_mw')
                            frp = float(frp) if isinstance(frp, (int, float)) and math.isfinite(frp) else 1.0
                            dets.append({'lat': c['lat'], 'lon': c['lon'], 'frp': max(frp, 0.0),
                                         'sensor': c.get('platform', sat), 'acq': c.get('acquisition_start')})
                    except Exception:
                        pass
            except Exception as e:
                print('  GOES fetch', sat, repr(e)[:50], flush=True)
    except Exception as e:
        print('  GOES reader import fail', repr(e)[:60], flush=True)
    try:
        import firms_fixed as F2
        for d in F2.fetch(bbox, t_map.date(), now.date()):
            if d.get('lat') is None:
                continue
            dets.append({'lat': d['lat'], 'lon': d['lon'], 'frp': float(d.get('frp') or 0.0),
                         'sensor': 'VIIRS', 'acq': d.get('acq')})
    except Exception as e:
        print('  VIIRS fetch', repr(e)[:60], flush=True)
    return dets


def _acres(geom, lon, lat):
    mx = 111320*math.cos(math.radians(lat)); my = 111320
    return transform(lambda x, y, z=None: ((x-lon)*mx, (y-lat)*my), geom).area/4046.86 if geom else 0.0


def _drop_stray(perim, base, dets):
    """Keep the base-connected component; keep a DISCONNECTED component only if it
    is genuinely corroborated (>=4 detections from >=2 sensors inside it -- a real
    spot fire). Drops scatter blobs the top-k paints around a few lone GOES pixels
    with no VIIRS/second-satellite agreement (the Floriston 'spot fire')."""
    if perim is None or perim.is_empty or perim.geom_type == 'Polygon':
        return perim
    parts = [g for g in perim.geoms if not g.is_empty and g.area > 0]
    if not parts:
        return perim
    bb = base.buffer(0.5/111.0)   # ~0.5 km around the mapped base
    pts = [(Point(d['lon'], d['lat']), d.get('sensor', '')) for d in dets]

    def corroborated(comp):
        ins = [s for (pt, s) in pts if comp.contains(pt)]
        return len(ins) >= 4 and len({s for s in ins}) >= 2

    keep = [p for p in parts if p.intersects(bb) or corroborated(p)]
    return unary_union(keep) if keep else max(parts, key=lambda p: p.area)


def _smooth(g, r_km=0.35):
    """Round the 250m-grid stair-steps into a natural boundary (cosmetic; the
    lab measured contour-vs-square-union as a wash in area terms)."""
    if g is None or g.is_empty:
        return g
    r = r_km/111.0
    out = g.buffer(r, join_style=1).buffer(-1.6*r, join_style=1).buffer(0.6*r, join_style=1)
    out = out.simplify(0.12/111.0)
    return out if (not out.is_empty and out.area > 0) else g


def _parse_dt(s):
    s = str(s).replace('Z', '+00:00')
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _gather_drops(name, lat, lon, base):
    """Recent aircraft DROP paths near the fire, via the PATTERN+DEM detector
    (aircraft_drops, rebuild spec 8): AGL from the DEM under the aircraft + a
    local-altitude-minimum test, instead of the old per-sample flag. Returns
    serialisable [{coords, kind, t1}]; a final length/proximity guard is kept."""
    out = []
    try:
        sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fpm')))
        import aircraft_drops as AD
        from shapely.geometry import LineString
        segs = AD.recent_drop_paths(name, lat, lon, hours=3.0)
        fb = base.buffer(6.0/111.0)
        def _len_km(c):
            tot = 0.0
            for (ax_, ay_), (bx_, by_) in zip(c, c[1:]):
                dx = (bx_-ax_)*111.32*math.cos(math.radians(ay_)); dy = (by_-ay_)*111.32
                tot += (dx*dx + dy*dy)**0.5
            return tot
        for s in segs:
            c = s.get('coords', [])
            if len(c) >= 2 and _len_km(c) <= 4.0 and fb.contains(LineString(c).centroid):
                out.append({'coords': [list(pt) for pt in c], 'kind': s.get('kind', 'drop'), 't1': s.get('t1')})
    except Exception as e:
        print('  aircraft', repr(e)[:60], flush=True)
    return out


def run_field(name, lat, lon, out_dir, k=1.0, pad_deg=None, inputs=None):
    """One cycle. inputs=None -> fetch live + snapshot; inputs=dict -> deterministic
    REPLAY from a snapshot (no network), for reproducibility/debugging (spec 9)."""
    os.makedirs(out_dir, exist_ok=True)
    from shapely import wkb as _wkb
    with open(MODEL_PKL, 'rb') as fh:
        M = pickle.load(fh)
    model, rnf = M['model'], M['retained_negative_fraction']
    if inputs is None:
        base, props, t_map = mapped_base(lat, lon)
        if base is None:
            print(f'{name}: no mapped base found', flush=True); return None
        now = datetime.now(timezone.utc)
        if t_map is None:
            t_map = now.replace(hour=0, minute=0, second=0, microsecond=0)
        base_ac = _acres(base, lon, lat)
        diam_km = 2*math.sqrt(max(base_ac, 1)*4046.86/math.pi)/1000.0   # adaptive pad (bbox-truncation fix)
        pad = pad_deg if pad_deg is not None else max(0.08, min(0.6, diam_km/111.0*1.5))
        b = base.bounds; bbox = (b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad)
        dets = _live_dets(bbox, t_map, now)
        drop_paths = _gather_drops(name, lat, lon, base)
        # SNAPSHOT the exact input bundle -> this perimeter is reproducible offline
        snap = {'name': name, 'lat': lat, 'lon': lon, 'base_wkb': base.wkb_hex,
                't_map': t_map.isoformat(), 'now': now.isoformat(), 'k': k, 'dets': dets,
                'drop_paths': drop_paths, 'model_pkl': os.path.basename(MODEL_PKL),
                'model_mtime': round(os.path.getmtime(MODEL_PKL), 3)}
        with open(os.path.join(out_dir, f'{name}_inputs.json'), 'w') as fh:
            json.dump(snap, fh)
    else:
        base = _wkb.loads(inputs['base_wkb'], hex=True)
        t_map = _parse_dt(inputs['t_map']); now = _parse_dt(inputs['now'])
        dets = inputs['dets']; drop_paths = inputs.get('drop_paths', [])
        base_ac = _acres(base, lon, lat)
    by = {}
    for d in dets: by[d['sensor']] = by.get(d['sensor'], 0)+1
    print(f'{name}: base {base_ac:.0f} ac as-of {t_map:%m-%d %H:%MZ}, {len(dets)} dets {by}'
          + ('  [REPLAY]' if inputs is not None else ''), flush=True)

    step = {'fire': name, 'step': 0, 'base_wkb': base.wkb_hex,
            'start': t_map.isoformat(), 'end': now.isoformat(), 'dets': dets}
    grid = LF.build_prediction_grid(step, now, dem=LF.ZeroDem())
    if grid is None:
        print(f'{name}: no prediction grid -> persistence', flush=True); perim = base
    else:
        p = LF.predict_probabilities(model, grid, rnf)
        perim = LF.geometry_from_probabilities(grid, p, k=k, close_km=0.2)
        perim = _drop_stray(perim, base, dets)   # drop uncorroborated scatter blobs
        perim = _smooth(perim)                   # round the 250m stair-steps
    # aircraft drops (already fetched/filtered) confirm the worked ACTIVE EDGE:
    # extend the perimeter only where a drop hugs the predicted edge (<=0.6km).
    if drop_paths and perim is not None:
        from shapely.geometry import LineString
        dg = unary_union([LineString(dp['coords']).buffer(300/111320.0) for dp in drop_paths])
        near = dg.intersection(perim.buffer(0.6/111.0))
        if not near.is_empty and near.area > 0:
            perim = unary_union([perim, near])
        print(f'{name}: +{len(drop_paths)} recent drop path(s) (worked edge)', flush=True)

    pred_ac = _acres(perim, lon, lat)
    lo_ac, hi_ac = pred_ac*math.exp(CONF_LO), pred_ac*math.exp(CONF_HI)
    print(f'{name}: FIELD pred {pred_ac:.0f} ac  [{lo_ac:.0f}, {hi_ac:.0f}]  (base {base_ac:.0f}, k={k})', flush=True)

    out = {'name': name, 'lat': lat, 'lon': lon, 't_map': t_map.isoformat(), 'now': now.isoformat(),
           'base_acres': round(base_ac), 'pred_acres': round(pred_ac),
           'lo_acres': round(lo_ac), 'hi_acres': round(hi_ac),
           'n_dets': len(dets), 'dets_by_sensor': by, 'k': k, 'n_drop_paths': len(drop_paths)}
    with open(os.path.join(out_dir, f'{name}_field.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(out_dir, f'{name}_field.geojson'), 'w') as fh:
        json.dump({'type': 'Feature', 'properties': out,
                   'geometry': shape(perim.__geo_interface__).__geo_interface__}, fh)
    _render(name, lat, lon, base, perim, dets, lo_ac, hi_ac, pred_ac, out_dir, drop_paths, now)
    return out


def _render(name, lat, lon, base, perim, dets, lo_ac, hi_ac, pred_ac, out_dir, drop_paths=(), now=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(11, 10)); asp = 1/math.cos(math.radians(lat))
    pb = perim.bounds; pad = max(0.02, (pb[2]-pb[0]), (pb[3]-pb[1]))*0.5
    vx = (min(pb[0], lon)-pad, max(pb[2], lon)+pad); vy = (min(pb[1], lat)-pad, max(pb[3], lat)+pad)
    try:
        import fusion_v3 as FV                        # reuse the Esri satellite basemap
        img, ext = FV.satellite_basemap((vx[0], vy[0], vx[1], vy[1]),
                                        zoom=13 if (vx[1]-vx[0]) > 0.12 else 14)
        if img is not None: ax.imshow(img, extent=ext, origin='upper', aspect=asp, zorder=0)
    except Exception as e:
        print('  basemap fail', repr(e)[:60], flush=True)

    def poly(g, **kw):
        if g is None or g.is_empty: return
        for gg in ([g] if g.geom_type == 'Polygon' else list(g.geoms)):
            if gg.is_empty: continue
            xs, ys = gg.exterior.xy; ax.plot(xs, ys, **kw)
    # detections colored by sensor
    for sat, c, lab in (('GOES-19', '#ff6666', 'GOES-19 East'), ('GOES-18', '#ffaa33', 'GOES-18 West'), ('VIIRS', '#ff00ff', 'VIIRS 375m')):
        pts = [(d['lon'], d['lat']) for d in dets if d['sensor'] == sat]
        if pts: ax.scatter([x for x, _ in pts], [y for _, y in pts], s=10, c=c, edgecolor='k', linewidth=0.15, zorder=4, label=f'{lab} ({len(pts)})')
    # aircraft DROP paths (worked edge) -- pink, with kind + age
    import time as _t
    for i, p in enumerate(drop_paths or ()):
        xs = [c[0] for c in p['coords']]; ys = [c[1] for c in p['coords']]
        ax.plot(xs, ys, color='#ff4fd0', lw=2.2, zorder=5,
                label='aircraft drop paths' if i == 0 else None)
        age_min = (_t.time() - p.get('t1', _t.time()))/60.0
        ax.annotate(f"{p.get('kind','drop')} {age_min:.0f}m", (xs[-1], ys[-1]),
                    fontsize=7, color='#ff9fe6', zorder=6)
    poly(base, color='cyan', lw=1.6, ls='--', zorder=5)
    # ACTIVE vs QUIET edge: red where recent/hot detections hug the boundary,
    # green where the edge is cold (backing / contained). Activity = a detection
    # seen in the last ~1.5h OR in the top 30% FRP, within ~1.2km of the edge.
    from datetime import datetime as _dt, timezone as _tz
    ref = now or _dt.now(_tz.utc)
    def _age_h(d):
        try:
            s = str(d.get('acq') or '').replace('Z', '')
            t = _dt.fromisoformat(s); t = t if t.tzinfo else t.replace(tzinfo=_tz.utc)
            return (ref - t).total_seconds()/3600.0
        except Exception:
            return 99.0
    # "Active" = the HOT/running front: a detection in the top quartile of FRP, or
    # a genuinely recent GOES cell (5-min cadence). VIIRS recency is NOT used as
    # activity -- a single pass stamps the whole fire at once and would paint
    # everything red. Heat is what distinguishes the head from the backing flanks.
    frps = [float(d.get('frp') or 0) for d in dets]
    p75 = float(np.percentile(frps, 75)) if frps else 0.0
    act = np.array([(d['lon'], d['lat']) for d in dets
                    if (float(d.get('frp') or 0) >= p75 and p75 > 0)
                    or (str(d.get('sensor', '')).startswith('GOES') and _age_h(d) < 0.75)])
    cosl = math.cos(math.radians(lat)); THR = 1.0
    def _active(x, y):
        if len(act) == 0: return False
        dx = (act[:, 0]-x)*111.32*cosl; dy = (act[:, 1]-y)*111.32
        return bool(np.sqrt(np.min(dx*dx+dy*dy)) < THR)
    for gg in ([perim] if perim.geom_type == 'Polygon' else list(perim.geoms)):
        if gg.is_empty: continue
        xs, ys = gg.exterior.xy
        for i in range(len(xs)-1):
            c = '#ff2a2a' if (_active(xs[i], ys[i]) or _active(xs[i+1], ys[i+1])) else '#00ff66'
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=c, lw=2.6, zorder=6, solid_capstyle='round')
    ax.scatter([lon], [lat], marker='*', s=240, c='yellow', edgecolor='k', zorder=7)
    ax.set_xlim(*vx); ax.set_ylim(*vy); ax.set_xlabel('lon'); ax.set_ylabel('lat')
    handles = ax.get_legend_handles_labels()[0] + [
        Patch(facecolor='none', edgecolor='#ff2a2a', label='ACTIVE edge (recent/hot)'),
        Patch(facecolor='none', edgecolor='#00ff66', label='quiet edge'),
        Patch(facecolor='none', edgecolor='cyan', label='mapped base (as-of)'),
        Patch(facecolor='none', edgecolor='none', label=f'FIELD {pred_ac:.0f} ac  [{lo_ac:.0f}, {hi_ac:.0f}]')]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.85)
    ax.set_title(f'{name} — learned-field v3 (GOES+VIIRS), {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'{name}_field.png'), dpi=115); plt.close(fig)


if __name__ == '__main__':
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fpm')))   # for satellite_basemap reuse
    if len(sys.argv) > 1 and sys.argv[1] == '--replay':
        # deterministic replay from a snapshot: deploy_live_field.py --replay <inputs.json> <out_dir>
        snap = json.load(open(sys.argv[2]))
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(sys.argv[2])
        run_field(snap['name'], snap['lat'], snap['lon'], out, k=snap.get('k', 1.0), inputs=snap)
    else:
        name, lat, lon, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        k = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        run_field(name, lat, lon, out, k=k)
