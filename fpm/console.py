"""FIRE PLATFORM CONSOLE  --  one local app to run everything.

    python fpm/console.py            # opens http://localhost:8095 in your browser

Three panels in one page:
  1. INVENTORY      -- every model / data source / weight file the platform has
  2. PERIMETER      -- pick a fire (live list or coords), pick a model, RUN it
                       live, get a real predicted perimeter + acreage + figure
  3. LIVE SMOKE     -- run the best_seg_v3 smoke detector on the AlertCalifornia
                       camera network live, ranked by detection confidence

Self-contained http.server (same pattern as goes19_replay_v3 / review_server).
Heavy jobs (perimeter run, camera scan) run in background threads; the page
polls /api/job/<id> for progress.  Nothing is precomputed -- every RUN hits the
real models and the real feeds.
"""
import os, sys, io, json, time, math, threading, traceback, urllib.request, urllib.parse
import importlib.util
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FDV = os.path.dirname(HERE)
FSL = os.path.join(FDV, 'fire-spread-lab')
FSLC = FSL
for p in (HERE, FDV, FSL, FSLC, os.path.join(FSLC, 'scripts'), os.path.join(FDV, 'sim')):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT_DIR = os.path.join(HERE, '_console_out')
SMOKE_DIR = os.path.join(OUT_DIR, 'smoke')
os.makedirs(SMOKE_DIR, exist_ok=True)
PORT = 8095

AC_LIST = 'https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json'
def frame_url(cid): return f'https://cameras.alertcalifornia.org/public-camera-data/{cid}/latest-frame.jpg'

PERIMETER_MODELS = {
    'learned_field_v3': 'Learned-field v3 (validated best model) — run LIVE, out-of-sample. Base + GOES-18/19 + VIIRS → probability field → perimeter + conformal envelope.',
    'fusion_v3':        'Full multi-source Bayesian fusion (VIIRS+GOES+S2+cameras+aircraft+physics).',
}
SMOKE_WEIGHTS = 'best_seg_v3.pt'

# ---------------- job registry ----------------
JOBS = {}
JLOCK = threading.Lock()
def new_job(kind):
    jid = f'{kind}-{int(time.time()*1000)}'
    with JLOCK:
        JOBS[jid] = {'id': jid, 'kind': kind, 'status': 'running', 'progress': 0,
                     'log': [], 'result': None, 'error': None, 'started': time.time()}
    return jid
def jlog(jid, msg, progress=None):
    with JLOCK:
        j = JOBS.get(jid)
        if not j: return
        j['log'].append(msg)
        if progress is not None: j['progress'] = progress
def jdone(jid, result=None, error=None):
    with JLOCK:
        j = JOBS.get(jid)
        if not j: return
        j['status'] = 'error' if error else 'done'
        j['result'] = result; j['error'] = error; j['progress'] = 100

# ---------------- shared model handles ----------------
_YOLO = None
_YLOCK = threading.Lock()
def get_smoke_model():
    global _YOLO
    with _YLOCK:
        if _YOLO is None:
            from ultralytics import YOLO
            _YOLO = YOLO(os.path.join(HERE, SMOKE_WEIGHTS))
    return _YOLO

_deploy = None
def get_deploy():
    global _deploy
    if _deploy is None:
        spec = importlib.util.spec_from_file_location('deploy_live_field', os.path.join(FSLC, 'deploy_live_field.py'))
        m = importlib.util.module_from_spec(spec); sys.modules['deploy_live_field'] = m; spec.loader.exec_module(m)
        _deploy = m
    return _deploy

# ---------------- data helpers ----------------
def _get(url, timeout=40):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'console'}), timeout=timeout).read())
    except Exception:
        return None

def list_fires(min_acres=1000, n=25):
    import perimeter_studio as PS
    try:
        return PS.list_current_fires(min_acres=min_acres, n=n)
    except Exception:
        return []

_CAMS = {'ts': 0, 'data': None}
def all_cameras():
    if _CAMS['data'] and time.time() - _CAMS['ts'] < 300:
        return _CAMS['data']
    d = _get(AC_LIST, timeout=40)
    feats = d if isinstance(d, list) else (d or {}).get('features', [])
    cams = []
    for f in feats:
        pr = f.get('properties', {}) if isinstance(f, dict) else {}
        g = f.get('geometry', {}) if isinstance(f, dict) else {}
        coords = (g or {}).get('coordinates') or [None, None]
        lon, lat = (coords + [None, None])[:2]
        cams.append({'id': pr.get('id'), 'name': pr.get('name') or pr.get('id'),
                     'lat': lat, 'lon': lon, 'ts': pr.get('last_frame_ts') or 0,
                     'az': pr.get('az_current')})
    cams = [c for c in cams if c['id']]
    _CAMS['data'] = cams; _CAMS['ts'] = time.time()
    return cams

def cameras_for(lat=None, lon=None, radius_km=50, limit=24):
    cams = all_cameras()
    if lat is not None and lon is not None:
        near = []
        for c in cams:
            if c['lat'] is None or c['lon'] is None: continue
            dkm = math.hypot((c['lat']-lat)*111.0, (c['lon']-lon)*111.0*math.cos(math.radians(lat)))
            if dkm <= radius_km:
                near.append((dkm, c))
        near.sort(key=lambda x: x[0])
        near = near if limit <= 0 else near[:limit]
        return [dict(c, dist_km=round(d, 1)) for d, c in near]
    cams = [c for c in cams if c['ts']]
    cams.sort(key=lambda c: c['ts'], reverse=True)
    return cams if limit <= 0 else cams[:limit]

# ---------------- background workers ----------------
def run_perimeter_job(jid, name, lat, lon, model):
    try:
        out_dir = os.path.join(OUT_DIR, 'perim', f"{name.replace(' ','_')}_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        jlog(jid, f'starting {model} on {name} ({lat:.4f},{lon:.4f})', 10)
        if model == 'learned_field_v3':
            dp = get_deploy()
            jlog(jid, 'fetching base perimeter + live GOES-18/19 + VIIRS detections…', 30)
            res = dp.run_field(name, lat, lon, out_dir, k=1.0)
            png = os.path.join(out_dir, f'{name}_field.png')
            res['png'] = png
        elif model == 'fusion_v3':
            import fusion_v3
            jlog(jid, 'running fusion_v3 (fetch + fuse all sources)…', 30)
            fusion_v3.run(name, lat, lon, out_dir)
            png = None
            for f in os.listdir(out_dir):
                if f.lower().endswith('.png'): png = os.path.join(out_dir, f); break
            gj = None
            for f in os.listdir(out_dir):
                if f.endswith('.geojson'): gj = os.path.join(out_dir, f); break
            res = {'name': name, 'png': png, 'geojson': gj, 'note': 'fusion_v3 complete'}
        else:
            raise ValueError('unknown model ' + model)
        jlog(jid, 'done', 100)
        jdone(jid, result=res)
    except Exception as e:
        jdone(jid, error=f'{type(e).__name__}: {e}\n' + traceback.format_exc()[-1200:])

def _fetch_frame(c):
    """Download + decode one camera's latest frame. I/O bound -> run in a pool."""
    import cv2, numpy as np
    try:
        raw = urllib.request.urlopen(urllib.request.Request(frame_url(c['id']), headers={'User-Agent': 'console'}), timeout=15).read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return c, arr
    except Exception as e:
        return c, None

def run_stage_job(jid, key):
    try:
        import console_stages as CS
        jlog(jid, f'rendering stage {key} (learned_field_v3, frozen)…', 30)
        out_dir = os.path.join(OUT_DIR, 'stage')
        res = CS.run_stage(key, out_dir)
        jlog(jid, f"growth-IoU {res['growth_iou']}", 100)
        jdone(jid, result=res)
    except Exception as e:
        jdone(jid, error=f'{type(e).__name__}: {e}\n' + traceback.format_exc()[-1200:])

def scan_smoke_job(jid, cams, conf, workers=32, batch=16):
    """Scale to the whole network: prefetch frames concurrently, run the smoke
    model in batches. Only cameras WITH a smoke detection keep an annotated image
    (so scanning thousands doesn't write thousands of files)."""
    try:
        import cv2, numpy as np
        from concurrent.futures import ThreadPoolExecutor
        jlog(jid, f'loading smoke model {SMOKE_WEIGHTS}…', 3)
        model = get_smoke_model()
        total = len(cams)
        results = []; done = 0; n_hit = 0
        jlog(jid, f'scanning {total} cameras ({workers} fetch workers)…', 5)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = []   # (cam, frame)
            def flush():
                nonlocal done, n_hit
                if not pending: return
                frames = [f for _, f in pending]
                preds = model.predict(frames, conf=conf, verbose=False)
                for (c, _), pred in zip(pending, preds):
                    n = 0 if pred.boxes is None else len(pred.boxes)
                    maxc = float(pred.boxes.conf.max()) if n else 0.0
                    rec = {**c, 'ok': True, 'n': n, 'maxconf': round(maxc, 3)}
                    if n:                                   # save annotated frame only on a hit
                        fn = f"{c['id']}.jpg".replace('/', '_').replace('\\', '_')
                        cv2.imwrite(os.path.join(SMOKE_DIR, fn), pred.plot())
                        rec['img'] = fn; n_hit += 1
                    results.append(rec)
                pending.clear()
            for c, frame in pool.map(_fetch_frame, cams):
                done += 1
                if frame is None:
                    results.append({**c, 'ok': False, 'note': 'no frame'})
                else:
                    pending.append((c, frame))
                    if len(pending) >= batch:
                        flush()
                if done % 25 == 0 or done == total:
                    jlog(jid, f'[{done}/{total}] scanned · {n_hit} smoke hits', 5 + int(90 * done / max(total, 1)))
            flush()
        results.sort(key=lambda r: (r.get('n', 0) > 0, r.get('maxconf', 0)), reverse=True)
        # cap the payload returned to the page (hits first, then a sample of clears)
        hits = [r for r in results if r.get('n', 0) > 0]
        clears = [r for r in results if r.get('n', 0) == 0]
        shown = hits + clears[:max(0, 60 - len(hits))]
        jdone(jid, result={'cameras': shown, 'n_scanned': total, 'n_smoke': len(hits),
                           'n_clear': len(clears), 'truncated': len(shown) < len(results)})
    except Exception as e:
        jdone(jid, error=f'{type(e).__name__}: {e}\n' + traceback.format_exc()[-1200:])

# ---------------- inventory ----------------
def inventory():
    def files(pat_dir, exts):
        out = []
        for f in sorted(os.listdir(pat_dir)) if os.path.isdir(pat_dir) else []:
            if f.lower().endswith(exts): out.append(f)
        return out
    return {
        'perimeter_models': PERIMETER_MODELS,
        'smoke_weights': files(HERE, ('.pt', '.onnx')),
        'active_smoke_model': SMOKE_WEIGHTS,
        'data_sources': [
            'WFIGS Interagency Perimeters (live official perimeters + as-of time)',
            'GOES-19 East FDCC fire detections (5-min)',
            'GOES-18 West FDCC fire detections (5-min, crossing)',
            'VIIRS 375m active fire (FIRMS)',
            'Sentinel-2 SWIR / Sentinel-1 SAR (on-demand)',
            'Open-Meteo / RTMA→HRRR wind + fuel moisture',
            'LANDFIRE FBFM40 fuel model',
            'AlertCalifornia camera network (2249 cameras, live frames)',
        ],
        'key_files': {
            'learned_field_v3 model': 'fire-spread-lab/_frozen/deployable_field_v3.pkl',
            'live runner': 'fire-spread-lab/deploy_live_field.py',
            'fusion_v3': 'fpm/fusion_v3.py',
            'smoke detector': f'fpm/{SMOKE_WEIGHTS}',
            'perimeter studio': 'fpm/perimeter_studio.py',
        },
    }

# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == '/':
            return self._send(200, PAGE, 'text/html; charset=utf-8')
        if u.path == '/api/inventory':
            return self._send(200, inventory())
        if u.path == '/api/fires':
            return self._send(200, {'fires': list_fires()})
        if u.path == '/api/stages':
            try:
                import console_stages as CS
                return self._send(200, {'stages': CS.list_stages()})
            except Exception as e:
                return self._send(200, {'stages': {}, 'error': str(e)})
        if u.path == '/api/cameras':
            lat = float(q['lat'][0]) if 'lat' in q else None
            lon = float(q['lon'][0]) if 'lon' in q else None
            r = float(q.get('radius', ['50'])[0]); lim = int(q.get('limit', ['24'])[0])
            return self._send(200, {'cameras': cameras_for(lat, lon, r, lim)})
        if u.path.startswith('/api/job/'):
            jid = u.path.split('/')[-1]
            with JLOCK: j = JOBS.get(jid)
            return self._send(200, j or {'error': 'no such job'})
        if u.path.startswith('/out/'):
            return self._serve_file(os.path.join(OUT_DIR, u.path[len('/out/'):]))
        if u.path.startswith('/smoke/'):
            return self._serve_file(os.path.join(SMOKE_DIR, u.path[len('/smoke/'):]))
        if u.path.startswith('/frame/'):   # proxy a live raw camera frame
            cid = u.path[len('/frame/'):]
            try:
                raw = urllib.request.urlopen(urllib.request.Request(frame_url(cid), headers={'User-Agent': 'console'}), timeout=20).read()
                return self._send(200, raw, 'image/jpeg')
            except Exception:
                return self._send(502, b'', 'image/jpeg')
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        ln = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(ln) or b'{}')
        if u.path == '/api/perimeter/run':
            jid = new_job('perim')
            if body.get('mode') == 'stage' and body.get('stage_key'):
                threading.Thread(target=run_stage_job, args=(jid, body['stage_key']), daemon=True).start()
            else:
                threading.Thread(target=run_perimeter_job, args=(jid, body.get('name', 'fire'),
                                 float(body['lat']), float(body['lon']), body.get('model', 'learned_field_v3')),
                                 daemon=True).start()
            return self._send(200, {'job': jid})
        if u.path == '/api/smoke/scan':
            if body.get('camera_ids'):
                allc = {c['id']: c for c in all_cameras()}
                cams = [allc[i] for i in body['camera_ids'] if i in allc]
            else:
                lat = body.get('lat'); lon = body.get('lon')
                cams = cameras_for(lat, lon, float(body.get('radius', 50)), int(body.get('limit', 24)))
            jid = new_job('smoke')
            threading.Thread(target=scan_smoke_job, args=(jid, cams, float(body.get('conf', 0.25))), daemon=True).start()
            return self._send(200, {'job': jid, 'n': len(cams)})
        return self._send(404, {'error': 'not found'})

    def _serve_file(self, path):
        if not os.path.isfile(path): return self._send(404, {'error': 'no file'})
        ct = 'image/png' if path.endswith('.png') else 'image/jpeg' if path.endswith(('.jpg', '.jpeg')) else \
             'application/geo+json' if path.endswith('.geojson') else 'application/octet-stream'
        with open(path, 'rb') as fh: data = fh.read()
        self._send(200, data, ct)


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Fire Platform Console</title><style>
:root{--bg:#ffffff;--panel:#ffffff;--ink:#111111;--mut:#5b5b5b;--line:#cfcfcf;--line2:#111111;--hdr:#f2f2f2;--red:#b3261e;--sel:#eef2f6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 "Helvetica Neue",Arial,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:"SFMono-Regular",Consolas,"Roboto Mono",ui-monospace,monospace}
header{padding:0;border-bottom:2px solid var(--line2);background:#fff}
header .top{display:flex;align-items:baseline;gap:16px;max-width:1180px;margin:0 auto;padding:14px 22px}
header h1{font-size:17px;margin:0;font-weight:700;letter-spacing:.3px;white-space:nowrap}
header .tag{font-weight:400;color:var(--mut)}
header .id{color:var(--mut);font-size:12px}
header .id b{color:var(--ink);font-weight:600}
.wrap{max-width:1180px;margin:0 auto;padding:22px 22px 70px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line2);padding:0}
.card h2{margin:0;font-size:12px;color:var(--ink);background:var(--hdr);border-bottom:1px solid var(--line2);
  text-transform:uppercase;letter-spacing:.8px;padding:9px 14px;font-weight:700}
.card .body{padding:16px 16px}
label{display:block;color:var(--mut);font-size:11px;margin:12px 0 4px;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
input,select,button{font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:0;padding:8px 10px;width:100%}
input::placeholder{color:#9a9a9a}
input:focus,select:focus{outline:0;border-color:var(--line2);box-shadow:inset 0 -2px 0 var(--line2)}
button{background:#fff;color:var(--ink);border:1px solid var(--line2);font-weight:600;cursor:pointer;width:auto;padding:9px 18px;letter-spacing:.4px;text-transform:uppercase;font-size:12px}
button:hover{background:var(--ink);color:#fff}
button.sec{border-color:var(--line)}
button:disabled{opacity:.45;cursor:default;background:#fff;color:var(--mut);border-color:var(--line)}
.seg{display:flex;margin:2px 0 6px}
.seg button{flex:1;width:auto;border:1px solid var(--line2);border-right:0;padding:8px 6px}
.seg button:last-child{border-right:1px solid var(--line2)}
.seg button.on{background:var(--ink);color:#fff}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}.row>*{flex:1;min-width:90px}
.mut{color:var(--mut)}
.pill{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:0;padding:3px 9px;margin:3px 4px 0 0;font-size:12px;color:var(--ink)}
.pill b{color:#000}
.out{margin-top:14px}.out img{width:100%;border-radius:0;border:1px solid var(--line2)}
.stat{display:inline-block;margin:0 18px 8px 0;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut)}
.stat b{color:var(--ink);font-size:20px;display:block;font-weight:700;font-family:"SFMono-Regular",Consolas,ui-monospace,monospace}
.bar{height:6px;background:#eee;border:1px solid var(--line);border-radius:0;overflow:hidden;margin:14px 0 6px}
.bar i{display:block;height:100%;background:var(--ink);width:0;transition:width .3s}
.log{font-family:"SFMono-Regular",Consolas,ui-monospace,monospace;font-size:11.5px;color:var(--mut);max-height:96px;overflow:auto;white-space:pre-wrap;background:#fafafa;border:1px solid var(--line);padding:6px 8px;margin-top:8px}
.cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(184px,1fr));gap:10px;margin-top:14px}
.cam{background:#fff;border:1px solid var(--line);border-radius:0;overflow:hidden}
.cam.hit{border:2px solid var(--red)}
.cam img{width:100%;display:block;aspect-ratio:16/10;object-fit:cover;background:#000}
.cam .m{padding:6px 8px;font-size:11.5px;border-top:1px solid var(--line)}.cam .m .c{color:var(--red);font-weight:700}
small{color:var(--mut);font-size:11.5px}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);margin:0 0 6px;font-weight:700}
</style></head><body>
<header><div class=top>
  <h1>Fire Platform <span class=tag>· Console</span></h1>
  <div class=id id=inv>Initializing…</div>
</div></header>
<div class=wrap>
<div class=grid>
  <div class=card>
    <h2>1 &nbsp;·&nbsp; Perimeter model</h2>
    <div class=body>
    <div class=seg><button id=mlive class=on onclick="setMode('live')">Live forecast</button><button id=mstage onclick="setMode('stage')">Historical stage</button></div>
    <div id=livebox>
      <div class=row>
        <div style="flex:2"><label>Fire target</label><select id=fire><option value="">— PICK / LOAD LIVE FIRES —</option></select></div>
        <div><button class=sec onclick=loadFires()>Load fires</button></div>
      </div>
      <div class=row>
        <div><label>Name</label><input id=pname placeholder="DOME"></div>
        <div><label>Lat</label><input id=plat placeholder="37.5731"></div>
        <div><label>Lon</label><input id=plon placeholder="-119.6125"></div>
      </div>
      <label>Model</label><select id=pmodel></select>
    </div>
    <div id=stagebox style=display:none>
      <label>Fire (validation cohort)</label><select id=sfire></select>
      <label>Stage — start perimeter → end perimeter</label><select id=sstep></select>
      <small class=mut id=snote>learned_field_v3 vs observed truth · growth-IoU</small>
    </div>
    <div style="margin-top:14px"><button id=prun onclick=runPerim()>Run model</button>
      <small id=pnote> real GOES/VIIRS fetch — ~1–3 MIN</small></div>
    <div class=bar id=pbarw style=display:none><i id=pbar></i></div>
    <div class=log id=plog></div>
    <div class=out id=pout></div>
    </div>
  </div>

  <div class=card>
    <h2>2 &nbsp;·&nbsp; Live smoke detector</h2>
    <div class=body>
    <div class=row>
      <div><label>Near lat (opt)</label><input id=slat placeholder="fire lat"></div>
      <div><label>Near lon (opt)</label><input id=slon placeholder="fire lon"></div>
    </div>
    <div class=row>
      <div><label>Radius km</label><input id=srad value=60></div>
      <div><label>N cameras</label><input id=slim value=18></div>
      <div><label>Conf</label><input id=sconf value=0.25></div>
    </div>
    <div style="margin-top:14px"><button id=srun onclick=scan()>Scan cameras</button>
      <small class=mut> N=0 scans the WHOLE network · blank lat/lon = network-wide</small></div>
    <div class=bar id=sbarw style=display:none><i id=sbar></i></div>
    <div class=log id=slog></div>
    <div id=sres></div>
    <div class=cams id=scams></div>
    </div>
  </div>
</div>

<div class=card style="margin-top:22px"><h2>System inventory</h2><div class=body><div id=invfull class=mut>…</div></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
async function j(u,o){const r=await fetch(u,o);return r.json()}
async function boot(){
  const inv=await j('/api/inventory');
  $('inv').textContent=`${Object.keys(inv.perimeter_models).length} perimeter models · ${inv.smoke_weights.length} smoke weights · ${inv.data_sources.length} data sources`;
  const ps=$('pmodel');for(const[k,v]of Object.entries(inv.perimeter_models)){const o=document.createElement('option');o.value=k;o.textContent=k;o.title=v;ps.appendChild(o)}
  let h='<b>Perimeter models:</b> '+Object.keys(inv.perimeter_models).map(x=>`<span class=pill>${x}</span>`).join('');
  h+='<br><b>Active smoke model:</b> <span class=pill>'+inv.active_smoke_model+'</span> &nbsp; <b>weights:</b> '+inv.smoke_weights.map(x=>`<span class=pill>${x}</span>`).join('');
  h+='<br><b>Data sources:</b> '+inv.data_sources.map(x=>`<span class=pill>${x}</span>`).join('');
  h+='<br><b>Key files:</b> '+Object.entries(inv.key_files).map(([k,v])=>`<span class=pill>${k}: ${v}</span>`).join('');
  $('invfull').innerHTML=h;
}
async function loadFires(){
  $('fire').innerHTML='<option>loading…</option>';
  const d=await j('/api/fires');const s=$('fire');s.innerHTML='<option value="">— pick a fire —</option>';
  for(const f of d.fires){const o=document.createElement('option');o.value=JSON.stringify(f);o.textContent=`${f.name} (${Math.round(f.acres).toLocaleString()} ac, ${f.state})`;s.appendChild(o)}
}
$('fire').onchange=e=>{if(!e.target.value)return;const f=JSON.parse(e.target.value);$('pname').value=f.name;$('plat').value=f.lat.toFixed(4);$('plon').value=f.lon.toFixed(4);$('slat').value=f.lat.toFixed(4);$('slon').value=f.lon.toFixed(4)}
async function poll(jid,onp){for(;;){const s=await j('/api/job/'+jid);onp(s);if(s.status!=='running')return s;await new Promise(r=>setTimeout(r,1500))}}
let MODE='live', STAGES=null;
function setMode(m){MODE=m;
  $('mlive').className=m==='live'?'on':'';$('mstage').className=m==='stage'?'on':'';
  $('livebox').style.display=m==='live'?'':'none';$('stagebox').style.display=m==='stage'?'':'none';
  $('pnote').textContent=m==='live'?' real GOES/VIIRS fetch — ~1–3 MIN':' frozen learned_field_v3 vs truth — instant';
  if(m==='stage'&&!STAGES)loadStages();
}
async function loadStages(){
  const sf=$('sfire');sf.innerHTML='<option>loading…</option>';
  const d=await j('/api/stages');STAGES=d.stages||{};
  const fires=Object.keys(STAGES).sort();
  sf.innerHTML=fires.map(f=>`<option value="${f}">${f} (${STAGES[f].length} stage${STAGES[f].length>1?'s':''})</option>`).join('');
  sf.onchange=fillSteps;fillSteps();
}
function fillSteps(){
  const f=$('sfire').value,ss=$('sstep');if(!STAGES[f]){ss.innerHTML='';return}
  ss.innerHTML=STAGES[f].map(s=>`<option value="${s.key}">step ${s.step} · ${s.base.toLocaleString()} → ${s.real.toLocaleString()} ac (${s.gr}×, ${s.stratum})</option>`).join('');
}
async function runPerim(){
  let body;
  if(MODE==='stage'){
    const key=$('sstep').value;if(!key){alert('pick a stage');return}
    body={mode:'stage',stage_key:key};
  }else{
    const lat=parseFloat($('plat').value),lon=parseFloat($('plon').value);
    if(isNaN(lat)||isNaN(lon)){alert('need lat & lon');return}
    body={name:$('pname').value||'fire',lat,lon,model:$('pmodel').value};
  }
  $('prun').disabled=true;$('pbarw').style.display='block';$('pout').innerHTML='';
  const {job}=await j('/api/perimeter/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const s=await poll(job,s=>{$('pbar').style.width=s.progress+'%';$('plog').textContent=s.log.slice(-6).join('\n')});
  $('prun').disabled=false;
  if(s.status==='error'){$('pout').innerHTML='<div class=mut>error: '+s.error+'</div>';return}
  const r=s.result;let html='';
  if(r.growth_iou!=null)html+=`<div><span class=stat>start <b>${r.base_acres.toLocaleString()}</b> ac</span><span class=stat>end/observed <b>${r.real_acres.toLocaleString()}</b> ac</span><span class=stat>predicted <b>${r.pred_acres.toLocaleString()}</b> ac</span><span class=stat>growth-IoU <b>${r.growth_iou}</b></span></div>`;
  else if(r.pred_acres!=null)html+=`<div><span class=stat>base <b>${r.base_acres.toLocaleString()}</b> ac</span><span class=stat>predicted <b>${r.pred_acres.toLocaleString()}</b> ac</span><span class=stat>80% <b>[${r.lo_acres.toLocaleString()}–${r.hi_acres.toLocaleString()}]</b></span><span class=stat>dets <b>${r.n_dets}</b></span></div>`;
  if(r.png)html+=`<div class=out><img src="/out/${r.png.split(/[\\/]/).slice(-2).join('/')}?t=${Date.now()}"></div>`;
  $('pout').innerHTML=html||'<div class=mut>done: '+(r.note||'')+'</div>';
}
async function scan(){
  const lat=parseFloat($('slat').value),lon=parseFloat($('slon').value);
  const body={radius:parseFloat($('srad').value),limit:parseInt($('slim').value),conf:parseFloat($('sconf').value)};
  if(!isNaN(lat)&&!isNaN(lon)){body.lat=lat;body.lon=lon}
  $('srun').disabled=true;$('sbarw').style.display='block';$('scams').innerHTML='';$('sres').innerHTML='';
  const {job,n}=await j('/api/smoke/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  $('slog').textContent='scanning '+n+' cameras…';
  const s=await poll(job,s=>{$('sbar').style.width=s.progress+'%';$('slog').textContent=s.log.slice(-4).join('\n')});
  $('srun').disabled=false;
  if(s.status==='error'){$('sres').innerHTML='<div class=mut>error: '+s.error+'</div>';return}
  const r=s.result;$('sres').innerHTML=`<div style=margin-top:10px><span class=stat>scanned <b>${r.n_scanned}</b></span><span class=stat>smoke on <b>${r.n_smoke}</b> cameras</span></div>`;
  $('scams').innerHTML=r.cameras.map(c=>{
    const hit=c.n>0;const src=c.img?`/smoke/${c.img}?t=${Date.now()}`:`/frame/${c.id}`;
    return `<div class="cam ${hit?'hit':''}"><img src="${src}" loading=lazy><div class=m>${c.name||c.id}<br>${hit?`<span class=c>SMOKE ×${c.n} · ${(c.maxconf*100).toFixed(0)}%</span>`:'<span class=mut>clear</span>'}${c.dist_km!=null?` · ${c.dist_km}km`:''}</div></div>`;
  }).join('');
}
boot();
</script></body></html>"""


def start_server(block=False, open_browser=False):
    """Start the HTTP server. block=True runs forever (CLI); block=False starts
    it in a daemon thread and returns (srv, running) for the native-window app.
    running=False means the port was already bound (another instance)."""
    url = f'http://localhost:{PORT}'
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    except OSError:
        return None, False          # already running -> caller just shows the window
    if open_browser:
        try:
            import webbrowser; webbrowser.open(url)
        except Exception:
            pass
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    if block:
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            srv.shutdown()
    return srv, True


def main():
    print('=' * 60)
    print(f'  FIRE PLATFORM CONSOLE  ->  http://localhost:{PORT}')
    print('=' * 60)
    start_server(block=True, open_browser=True)


if __name__ == '__main__':
    main()
