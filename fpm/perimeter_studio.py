"""Perimeter Studio -- a runnable tool (not an app) to pull every data layer for
a fire and run the 5-minute perimeter model on it.

Flow:
  1. pick a CURRENT active fire (live WFIGS list) or a PAST fire (name + date)
  2. pick a time: 'present' or a specific date/time
  3. pick which MODEL to run
  4. it fetches all relevant layers (VIIRS, GOES FDCC crossed, GOES C07 hot,
     GOES ADP smoke, Sentinel-2 SWIR, mapped/past perimeter, wind, terrain)
  5. opens an interactive map with a checkbox for every layer, and overlays
     the model's predicted perimeter

Run:  python perimeter_studio.py
      python perimeter_studio.py --fire "Little Giant"      # jump straight in
      python perimeter_studio.py --lat 48.1 --lon -120.3 --model fusion_v3
"""
import os, sys, math, json, argparse, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta, date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../sim')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab-claude/scripts')))

WFIGS = ('https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
         'WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query')

MODELS = {
    'fusion_v3':    'Full multi-source 5-min fusion (VIIRS+GOES+S2+cameras+aircraft+physics) -- the live deployable model',
    'goes_crossed': 'GOES-18 x GOES-19 crossed footprints only -- the sharp 5-min GOES extent',
    'viirs_only':   'VIIRS 375m active-fire footprint only',
    'layers_only':  'Fetch and view every layer, run no model',
}


def _get(url, timeout=40):
    try:
        return json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={'User-Agent': 'perimeter-studio'}), timeout=timeout).read())
    except Exception as e:
        print('   ! fetch failed:', repr(e)[:90]); return None


def list_current_fires(min_acres=500, n=20):
    """Live active large fires from WFIGS, with centroids."""
    q = {'where': f'attr_IncidentSize>{min_acres}', 'outFields': 'attr_IncidentName,attr_IncidentSize,attr_POOState',
         'returnGeometry': 'true', 'geometryPrecision': '4', 'outSR': '4326',
         'resultRecordCount': str(n), 'orderByFields': 'attr_IncidentSize DESC', 'f': 'json'}
    d = _get(WFIGS + '?' + urllib.parse.urlencode(q))
    fires = []
    for f in (d or {}).get('features', []):
        a = f['attributes']; g = f.get('geometry')
        if not g or 'rings' not in g:
            continue
        pts = [p for ring in g['rings'] for p in ring]
        lon = sum(p[0] for p in pts) / len(pts); lat = sum(p[1] for p in pts) / len(pts)
        fires.append({'name': (a.get('attr_IncidentName') or '?').strip(), 'lat': lat, 'lon': lon,
                      'acres': a.get('attr_IncidentSize') or 0, 'state': a.get('attr_POOState') or ''})
    return fires


def geocode_fire(name):
    """Find a fire by name in WFIGS current perimeters -> centroid."""
    q = {'where': f"attr_IncidentName LIKE '%{name.upper()}%'", 'outFields': 'attr_IncidentName,attr_IncidentSize',
         'returnGeometry': 'true', 'geometryPrecision': '4', 'outSR': '4326', 'resultRecordCount': '1', 'f': 'json'}
    d = _get(WFIGS + '?' + urllib.parse.urlencode(q))
    for f in (d or {}).get('features', []):
        g = f.get('geometry')
        if g and 'rings' in g:
            pts = [p for ring in g['rings'] for p in ring]
            return {'name': name, 'lat': sum(p[1] for p in pts)/len(pts), 'lon': sum(p[0] for p in pts)/len(pts),
                    'acres': f['attributes'].get('attr_IncidentSize') or 0}
    return None


# ---------------- layer fetching ----------------
def fetch_layers(lat, lon, when=None, name='fire', want_cameras=False):
    """Return {name: {'kind','data','note'}} for every available layer.
    when=None -> present; a datetime -> historical where the source allows."""
    import fire_fusion as FF
    from firms_fixed import fetch as viirs_fetch
    half = 0.11
    bbox = (lon - half/math.cos(math.radians(lat)), lat - half,
            lon + half/math.cos(math.radians(lat)), lat + half)
    ABI = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_studio_abi'); os.makedirs(ABI, exist_ok=True)
    layers = {}
    def add(name, kind, data, note=''):
        layers[name] = {'kind': kind, 'data': data, 'note': note}
        print(f'   {"+" if data is not None else "~"} {name:20s} {note}', flush=True)

    print('  fetching layers...', flush=True)
    # VIIRS points
    try:
        d0 = (when.date() if when else date.today()); d1 = d0 + timedelta(days=2)
        v = viirs_fetch(bbox, d0 - timedelta(days=2), d1)
        add('VIIRS 375m', 'points', [(x['lon'], x['lat'], x.get('frp', 0)) for x in v] if v else None,
            f'{len(v)} detections' if v else 'none')
    except Exception as e: add('VIIRS 375m', 'points', None, repr(e)[:50])
    # GOES crossed
    try:
        o = FF.obs_goes(bbox, ABI, since_dt=when)
        add('GOES crossed (5min)', 'poly', getattr(o, 'geom', None) if _obs(o) else None, getattr(o,'note','') if _obs(o) else 'none')
    except Exception as e: add('GOES crossed (5min)', 'poly', None, repr(e)[:50])
    # GOES C07 hot
    try:
        o = FF.obs_goes_c07_hot(bbox, ABI)
        add('GOES C07 hot', 'poly', getattr(o,'geom',None) if _obs(o) else None, getattr(o,'note','') if _obs(o) else 'none')
    except Exception as e: add('GOES C07 hot', 'poly', None, repr(e)[:50])
    # GOES ADP smoke (direction)
    try:
        o = FF.obs_goes_adp_smoke(bbox, ABI)
        add('GOES ADP smoke dir', 'bearing', getattr(o,'bearing',None) if _obs(o) else None, getattr(o,'note','') if _obs(o) else 'none')
    except Exception as e: add('GOES ADP smoke dir', 'bearing', None, repr(e)[:50])
    # Sentinel-2 SWIR
    try:
        o = FF.obs_sentinel2_swir_fire(bbox)
        add('Sentinel-2 SWIR 20m', 'poly', getattr(o,'geom',None) if _obs(o) else None, getattr(o,'note','') if _obs(o) else 'none')
    except Exception as e: add('Sentinel-2 SWIR 20m', 'poly', None, repr(e)[:50])
    # mapped/past perimeter
    try:
        import fusion_v3
        base, props, status = fusion_v3.mapped_perimeter(lat, lon)
        add('Mapped perimeter', 'poly', base, status)
    except Exception as e: add('Mapped perimeter', 'poly', None, repr(e)[:50])
    # wind
    try:
        wj = _get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
                  '&current=wind_speed_10m,wind_direction_10m')
        if wj and 'current' in wj:
            c = wj['current']; add('Wind', 'wind', (c['wind_direction_10m'], c['wind_speed_10m']),
                                   f"{c['wind_speed_10m']} at {c['wind_direction_10m']}deg")
        else: add('Wind', 'wind', None, 'none')
    except Exception as e: add('Wind', 'wind', None, repr(e)[:50])
    # aircraft drop paths (ADS-B, fast)
    try:
        import aircraft_tracker as ATR
        paths = ATR.recent_paths(name, hours=3, seed_from_history=(lat, lon))
        drops = [p['coords'] for p in paths if p.get('drop') and len(p.get('coords', [])) >= 2]
        add('Aircraft drops', 'lines', drops or None, f'{len(drops)} drop runs' if drops else 'none')
    except Exception as e: add('Aircraft drops', 'lines', None, repr(e)[:50])
    # terrain / fuel model (LANDFIRE raster) as a context layer
    try:
        import data_ingest as DI
        import numpy as _np
        lats = _np.linspace(bbox[1], bbox[3], 80); lons = _np.linspace(bbox[0], bbox[2], 80)
        fuel = DI.get_fbfm40(bbox, lats, lons)
        add('Fuel model (LANDFIRE)', 'raster', (fuel, [bbox[0], bbox[2], bbox[1], bbox[3]]) if fuel is not None else None,
            'FBFM40 grid' if fuel is not None else 'none')
    except Exception as e: add('Fuel model (LANDFIRE)', 'raster', None, repr(e)[:50])
    # cameras (slow -- optional)
    if want_cameras:
        try:
            o = FF.obs_camera(bbox)
            add('Cameras (fire fix)', 'poly', getattr(o, 'geom', None) if _obs(o) else None,
                getattr(o, 'note', '') if _obs(o) else 'none')
        except Exception as e: add('Cameras (fire fix)', 'poly', None, repr(e)[:50])
    # roadmap products (available per DATA_PRODUCTS.md, not yet wired to a live fetch here)
    for rp in ['Sentinel-1 SAR', 'Sentinel-3 SLSTR FRP', 'VIIRS Day/Night Band',
               'Landsat 8/9 SWIR', 'MODIS MOD14/MYD14', 'GOES GLM lightning', 'GOES AOD (smoke)']:
        layers[rp] = {'kind': 'roadmap', 'data': None, 'note': 'available (DATA_PRODUCTS) — not wired to live fetch'}
    print('   (roadmap layers listed but not fetched: SAR, Sentinel-3, DNB, Landsat, MODIS, GLM, AOD)', flush=True)
    return bbox, layers


def _obs(o):
    """True if o is a real Observation with usable content (not None/Unavailable)."""
    return o is not None and hasattr(o, 'kind')


# ---------------- model run ----------------
def run_model(model, name, lat, lon):
    if model in ('layers_only',):
        return None
    if model == 'fusion_v3':
        import fusion_v3
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_studio_out', name.replace(' ', '_'),
                           datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        print('  running fusion_v3 (this fetches + runs the full model)...', flush=True)
        try:
            r = fusion_v3.run(name, lat, lon, out)
            gj = os.path.join(out, name.replace(' ', '_') + '_perimeter.geojson')
            # fusion_v3 keys geojson on the raw name
            for cand in (os.path.join(out, name + '_perimeter.geojson'), gj):
                if os.path.exists(cand):
                    from shapely.geometry import shape
                    fc = json.load(open(cand)); return shape(fc['features'][0]['geometry'])
        except Exception as e:
            print('   ! fusion_v3 failed:', repr(e)[:120]); return None
    return 'USE_LAYER'   # goes_crossed / viirs_only -> use the fetched layer as the "prediction"


# ---------------- interactive map ----------------
def show(bbox, layers, prediction, lat, lon, title):
    import matplotlib
    matplotlib.use('TkAgg') if _has_tk() else matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    import numpy as np
    fig, ax = plt.subplots(figsize=(11, 9)); ax.set_facecolor('#0d0b0a')
    fig.subplots_adjust(left=0.30)
    W, S, E, N = bbox; ax.set_xlim(W, E); ax.set_ylim(S, N)
    ax.set_aspect(1/math.cos(math.radians(lat)))
    # try a satellite basemap
    try:
        import fusion_v3
        img, ext = fusion_v3.satellite_basemap((W, S, E, N), zoom=12)
        if img is not None: ax.imshow(img, extent=ext, origin='upper', aspect='auto', zorder=0)
    except Exception: pass

    artists = {}   # label -> list of artists
    def poly(g, color, lw=2, fill=False, z=4):
        arts = []
        if g is None: return arts
        for gg in ([g] if g.geom_type == 'Polygon' else getattr(g, 'geoms', [])):
            if gg.is_empty or gg.geom_type != 'Polygon': continue
            xs, ys = gg.exterior.xy
            if fill: arts += ax.fill(xs, ys, color=color, alpha=0.22, zorder=z)
            arts += ax.plot(xs, ys, color=color, lw=lw, zorder=z+1)
        return arts

    palette = {'VIIRS 375m':'#ff3b3b','GOES crossed (5min)':'#ff9d3b','GOES C07 hot':'#ffe23b',
               'Sentinel-2 SWIR 20m':'#ff3bff','Mapped perimeter':'#00e5ff','Cameras (fire fix)':'#00ffaa',
               'Aircraft drops':'#ff1493'}
    for name, L in layers.items():
        c = palette.get(name, '#9ad')
        if L['data'] is None: artists[name] = []; continue
        if L['kind'] == 'points':
            xs = [p[0] for p in L['data']]; ys = [p[1] for p in L['data']]
            artists[name] = [ax.scatter(xs, ys, s=14, c=c, edgecolor='k', linewidth=0.2, zorder=6)]
        elif L['kind'] == 'poly':
            artists[name] = poly(L['data'], c, fill=(name=='Mapped perimeter'))
        elif L['kind'] == 'lines':
            arts = []
            for coords in L['data']:
                xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
                arts += ax.plot(xs, ys, '-', color=c, lw=3, zorder=7)
            artists[name] = arts
        elif L['kind'] == 'raster':
            grid, ext = L['data']
            artists[name] = [ax.imshow(grid, extent=ext, origin='lower', cmap='YlOrBr', alpha=0.35, zorder=1, aspect='auto')]
        elif L['kind'] == 'roadmap':
            artists[name] = []   # listed in the panel, no live data
        elif L['kind'] == 'bearing':
            br = math.radians(L['data']); artists[name] = [ax.annotate('', xy=(lon+0.03*math.sin(br), lat+0.03*math.cos(br)),
                xytext=(lon, lon and lat), arrowprops=dict(fc='#cccccc', ec='k', width=2, headwidth=9), zorder=7)]
        elif L['kind'] == 'wind':
            wdir, wspd = L['data']; br = math.radians((wdir+180) % 360)
            artists[name] = [ax.annotate('', xy=(lon+0.02*math.sin(br), lat+0.02*math.cos(br)),
                xytext=(lon, lat), arrowprops=dict(fc='cyan', ec='k', width=2.5, headwidth=11), zorder=8)]
        else: artists[name] = []
    # prediction
    pred_arts = []
    if prediction is not None and prediction != 'USE_LAYER' and hasattr(prediction, 'geom_type'):
        pred_arts = poly(prediction, '#39ff14', lw=3.2, fill=True, z=9)
    ax.plot(lon, lat, '*', color='yellow', ms=16, markeredgecolor='k', zorder=10)
    ax.set_title(title, color='#f0e8de', fontsize=11)
    ax.tick_params(colors='#8a7f70')
    for s in ax.spines.values(): s.set_color('#333')

    # checkboxes
    labels = list(artists.keys()) + (['PREDICTED perimeter'] if pred_arts else [])
    groups = list(artists.values()) + ([pred_arts] if pred_arts else [])
    init = [bool(g) for g in groups]
    rax = fig.add_axes([0.02, 0.25, 0.24, 0.5]); rax.set_facecolor('#1a1714')
    check = CheckButtons(rax, labels, init)
    for t in check.labels: t.set_color('#e8e0d5'); t.set_fontsize(9)
    def toggle(lbl):
        i = labels.index(lbl)
        for a in groups[i]:
            a.set_visible(not a.get_visible())
        fig.canvas.draw_idle()
    check.on_clicked(toggle)
    fig.text(0.02, 0.80, 'LAYERS  (click to toggle)', color='#f4772f', fontsize=10, weight='bold')
    fig.text(0.02, 0.18, 'yellow star = fire origin\ngreen = model prediction', color='#8a7f70', fontsize=8)

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_studio_out', 'studio_view.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor='#0d0b0a', bbox_inches='tight')
    print(f'\n  saved a static copy -> {out_png}', flush=True)
    if _has_tk():
        print('  opening interactive window (toggle layers with the checkboxes)...', flush=True)
        plt.show()
    else:
        print('  (no display backend -- see the saved PNG)', flush=True)


def _has_tk():
    try:
        import tkinter; return True
    except Exception:
        return False


# ---------------- CLI ----------------
def menu(title, options):
    print('\n' + title)
    for i, o in enumerate(options, 1): print(f'  {i}. {o}')
    while True:
        s = input('  > ').strip()
        if s.isdigit() and 1 <= int(s) <= len(options): return int(s) - 1
        print('  pick a number.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fire'); ap.add_argument('--lat', type=float); ap.add_argument('--lon', type=float)
    ap.add_argument('--model', choices=list(MODELS)); ap.add_argument('--date', help='YYYY-MM-DD[THH:MM] or "present"')
    a = ap.parse_args()

    print('='*64 + '\n  PERIMETER STUDIO -- fire data + 5-minute model runner\n' + '='*64)

    # --- fire ---
    name = a.fire; lat = a.lat; lon = a.lon
    if lat is None or lon is None:
        if name:
            g = geocode_fire(name)
            if g: lat, lon = g['lat'], g['lon']
        if lat is None:
            src = menu('Fire source:', ['Current active fire (live list)', 'Past / by name', 'Enter coordinates'])
            if src == 0:
                print('\n  loading current active fires...')
                fires = list_current_fires()
                if not fires: print('  (no live fires reachable)'); return
                i = menu('Active fires (largest first):',
                         [f"{f['name']}  ({round(f['acres']):,} ac, {f['state']})" for f in fires])
                name, lat, lon = fires[i]['name'], fires[i]['lat'], fires[i]['lon']
            elif src == 1:
                name = input('  fire name: ').strip()
                g = geocode_fire(name)
                if g: lat, lon = g['lat'], g['lon']
                else: print('  not found in current WFIGS; enter coordinates.'); lat = None
            if lat is None:
                lat = float(input('  latitude: ')); lon = float(input('  longitude: '))
                name = name or f'fire_{lat:.2f}_{lon:.2f}'
    name = name or f'fire_{lat:.2f}_{lon:.2f}'

    # --- time ---
    when = None
    dstr = a.date
    if dstr is None:
        ti = menu('Time:', ['Present (now)', 'Specific date/time'])
        if ti == 1: dstr = input('  date (YYYY-MM-DD or YYYY-MM-DDTHH:MM): ').strip()
    if dstr and dstr.lower() != 'present':
        try: when = datetime.fromisoformat(dstr).replace(tzinfo=timezone.utc)
        except Exception: print('  couldn\'t parse date; using present.'); when = None

    # --- model ---
    model = a.model
    if model is None:
        keys = list(MODELS)
        mi = menu('Model to run:', [f'{k} -- {MODELS[k]}' for k in keys])
        model = keys[mi]

    print(f'\n  FIRE: {name}  ({lat:.4f}, {lon:.4f})   TIME: {when or "present"}   MODEL: {model}')
    bbox, layers = fetch_layers(lat, lon, when, name=name, want_cameras=(model == 'fusion_v3'))
    pred = run_model(model, name, lat, lon)
    if pred == 'USE_LAYER':
        pred = layers.get('GOES crossed (5min)' if model == 'goes_crossed' else 'VIIRS 375m', {}).get('data')
        if model == 'viirs_only': pred = None  # viirs is points, shown as its own layer
    title = f'{name}  |  {when or "present"}  |  model: {model}'
    show(bbox, layers, pred, lat, lon, title)


if __name__ == '__main__':
    main()
