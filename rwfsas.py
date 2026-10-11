"""rwfsas command line.  No server, no browser: run a command, get an image.

    python rwfsas.py fires                         list active large fires (live, WFIGS)
    python rwfsas.py run "Park" --open             live prediction for a fire, by name
    python rwfsas.py run "Park" --lat 40.1 --lon -121.2
    python rwfsas.py stages                        list replayable historical steps
    python rwfsas.py stage "Fire Name" 3           predicted vs observed perimeter, growth-IoU

`run` fetches the official mapped perimeter, live GOES-18/19 + VIIRS detections, and
runs learned_field_v3. It writes <out>/<name>_field.png (+ .json, .geojson) and prints
the base -> predicted acreage and growth factor.  `stage` replays a past step and also
shows the observed end perimeter, so it needs fire-spread-lab/_frozen/steps_viirs.jsonl
and the frozen predictions (not shipped in the repo).
"""
import argparse, json, os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ('fpm', 'fire-spread-lab', os.path.join('fire-spread-lab', 'scripts'), 'sim'):
    sys.path.insert(0, os.path.join(ROOT, sub))


def _open(path):
    if sys.platform == 'win32':
        os.startfile(path)
    else:
        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', path], check=False)


def _slug(s):
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_')


# A fire is "active" only below this containment. Contained fires are not predicted.
MAX_ACTIVE_CONTAINED_PCT = 90


def _is_active(pct, ended):
    return not ended and (pct is None or pct < MAX_ACTIVE_CONTAINED_PCT)


def fire_status(lat, lon, pad=0.06):
    """Official containment for the mapped fire at lat/lon -> (pct_contained, ended) or None."""
    import json as _j, urllib.parse, urllib.request
    from shapely.geometry import shape
    env = _j.dumps({'xmin': lon - pad, 'ymin': lat - pad, 'xmax': lon + pad, 'ymax': lat + pad,
                    'spatialReference': {'wkid': 4326}})
    url = ('https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
           'WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?' + urllib.parse.urlencode({
               'where': '1=1', 'geometry': env, 'geometryType': 'esriGeometryEnvelope', 'inSR': '4326',
               'spatialRel': 'esriSpatialRelIntersects', 'outFields': '*', 'f': 'geojson',
               'geometryPrecision': '6', 'resultRecordCount': '5'}))
    try:
        d = _j.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'rwfsas'}), timeout=40).read())
        f = max(d['features'], key=lambda x: shape(x['geometry']).area)
    except Exception:
        return None
    p = f.get('properties', {})
    return (p.get('attr_PercentContained'),
            any(p.get(k) for k in ('attr_FireOutDateTime', 'attr_ContainmentDateTime', 'attr_ControlDateTime')))


def cmd_fires(a):
    import perimeter_studio as PS
    fires = PS.list_current_fires(min_acres=a.min_acres, n=a.n if a.all else 200)
    if not fires:
        sys.exit('no fires returned (offline, or WFIGS is unreachable)')
    if not a.all:
        fires = [f for f in fires if _is_active(f['pct_contained'], f['ended'])][:a.n]
    print(f'{"name":<28}{"state":<8}{"acres":>10}{"contained":>11}   lat, lon')
    for f in fires:
        pc = '?' if f['pct_contained'] is None else f'{f["pct_contained"]:.0f}%'
        print(f'{f["name"][:27]:<28}{f["state"]:<8}{f["acres"]:>10,.0f}{pc:>11}   {f["lat"]:.4f}, {f["lon"]:.4f}')


def cmd_run(a):
    if not os.path.exists(os.path.join(ROOT, 'fire-spread-lab', '_frozen', 'deployable_field_v3.pkl')):
        sys.exit('missing fire-spread-lab/_frozen/deployable_field_v3.pkl')
    lat, lon = a.lat, a.lon
    if lat is None or lon is None:
        import perimeter_studio as PS
        hit = PS.geocode_fire(a.name)
        if not hit:
            sys.exit(f'could not find "{a.name}" in WFIGS; pass --lat and --lon')
        lat, lon = hit['lat'], hit['lon']
        print(f'{a.name}: found at {lat:.4f}, {lon:.4f} ({hit["acres"]:,.0f} ac mapped)')
    st = fire_status(lat, lon)
    if st is not None and not a.force and not _is_active(*st):
        pct, ended = st
        why = 'declared out/contained' if ended else f'{pct:.0f}% contained'
        sys.exit(f'{a.name} is not active ({why}); predictions are only made for active fires. '
                 f'Use --force to run anyway.')
    import deploy_live_field as D
    name = _slug(a.name)
    out_dir = os.path.join(a.out, name)
    res = D.run_field(name, lat, lon, out_dir, k=a.k)
    if not res:
        sys.exit('no result (no mapped perimeter near that point)')
    base, pred = res['base_acres'], res['pred_acres']
    print(f'\n  first perimeter (mapped) : {base:>9,} ac   as-of {res["t_map"][:16]}Z')
    print(f'  predicted now            : {pred:>9,} ac   80% range {res["lo_acres"]:,} - {res["hi_acres"]:,}')
    if base:
        print(f'  growth                   : {pred / base:>9.2f}x   (+{pred - base:,} ac)')
    print(f'  detections               : {res["n_dets"]}  {res["dets_by_sensor"]}')
    png = os.path.join(out_dir, f'{name}_field.png')
    print(f'\n  image: {png}')
    if a.open and os.path.exists(png):
        _open(png)


def _need_steps():
    p = os.path.join(ROOT, 'fire-spread-lab', '_frozen', 'steps_viirs.jsonl')
    if not os.path.exists(p):
        sys.exit('historical replay needs fire-spread-lab/_frozen/steps_viirs.jsonl and the frozen '
                 'predictions, which are not in this repo. `run` works without them.')


def cmd_stages(a):
    _need_steps()
    import stages as CS
    for fire, steps in sorted(CS.list_stages().items()):
        print(f'{fire}: steps ' + ', '.join(str(s['step']) for s in steps))


def cmd_stage(a):
    _need_steps()
    import stages as CS
    st = CS.list_stages().get(a.fire)
    if not st:
        sys.exit(f'no replayable steps for "{a.fire}"; try `stages`')
    step = a.step if a.step is not None else st[0]['step']
    match = [s for s in st if s['step'] == step]
    if not match:
        sys.exit(f'step {step} not found for {a.fire}; available: {[s["step"] for s in st]}')
    res = CS.run_stage(match[0]['key'], os.path.join(a.out, _slug(a.fire)))
    print(json.dumps({k: v for k, v in res.items() if k != 'png'}, indent=2))
    print(f'\n  image: {res["png"]}')
    if a.open:
        _open(res['png'])


def main():
    ap = argparse.ArgumentParser(prog='rwfsas', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('fires', help='list active large fires')
    p.add_argument('--min-acres', type=int, default=1000); p.add_argument('--all', action='store_true', help='include contained fires'); p.add_argument('-n', type=int, default=25)
    p.set_defaults(fn=cmd_fires)
    p = sub.add_parser('run', help='live perimeter prediction -> image')
    p.add_argument('name'); p.add_argument('--lat', type=float); p.add_argument('--lon', type=float)
    p.add_argument('--k', type=float, default=1.0, help='threshold scale (default 1.0)')
    p.add_argument('--out', default=os.path.join(ROOT, 'output')); p.add_argument('--open', action='store_true')
    p.add_argument('--force', action='store_true', help='run even if the fire is contained')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('stages', help='list replayable historical steps'); p.set_defaults(fn=cmd_stages)
    p = sub.add_parser('stage', help='replay a past step: predicted vs observed')
    p.add_argument('fire'); p.add_argument('step', nargs='?', type=int)
    p.add_argument('--out', default=os.path.join(ROOT, 'output')); p.add_argument('--open', action='store_true')
    p.set_defaults(fn=cmd_stage)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
