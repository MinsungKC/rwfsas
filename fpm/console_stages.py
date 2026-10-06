"""Historical STAGE runner for the console: pick a fire + a step (start perimeter
-> end perimeter) and render learned_field_v3's prediction against the observed
truth, with the growth-IoU accuracy decomposition. Uses the frozen causal LOFO
predictions -- no network, deterministic."""
import os, sys, json, glob, math
from functools import lru_cache

FDV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSLC = os.path.join(FDV, 'fire-spread-lab')
STEPS = os.path.join(FSLC, '_frozen', 'steps_viirs.jsonl')
PRED_GLOB = os.path.join(FDV, 'fire-spread-lab', 'pipeline', 'runs', 'v3_causal_port', '*', 'predictions.jsonl')
MODEL = 'learned_field_v3_causal_viirs_zero_dem_k1'


def _step_key(fire, step, start):
    return f"{fire}:{step}:{start.replace('+00:00', 'Z')}"


@lru_cache(maxsize=1)
def _pred_index():
    idx = {}
    for f in glob.glob(PRED_GLOB):
        for l in open(f):
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get('model_id') == MODEL and d.get('prediction_wkb'):
                idx[d['step_key']] = d['prediction_wkb']
    return idx


@lru_cache(maxsize=1)
def list_stages():
    """{fire: [ {step, key, base, real, gr, stratum, start, end} ... ]} for every
    step that has a frozen v3 prediction (i.e. is runnable)."""
    preds = _pred_index()
    out = {}
    for l in open(STEPS):
        d = json.loads(l)
        key = _step_key(d['fire'], d['step'], d['start'])
        if key not in preds:
            continue
        out.setdefault(d['fire'], []).append({
            'step': d['step'], 'key': key,
            'base': round(d['base_acres']), 'real': round(d['real_acres']),
            'gr': round(d['gr'], 1), 'stratum': d.get('stratum'),
            'start': d['start'][:16], 'end': d.get('end', '')[:16]})
    for fire in out:
        out[fire].sort(key=lambda s: s['step'])
    return out


def _load_step(key):
    for l in open(STEPS):
        d = json.loads(l)
        if _step_key(d['fire'], d['step'], d['start']) == key:
            return d
    return None


def run_stage(key, out_dir):
    """Render base / observed truth / learned_field_v3 prediction + growth-IoU
    decomposition for one step. Returns a summary dict incl. the png path."""
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely import wkb
    from shapely.geometry import mapping, shape
    from shapely.ops import transform

    d = _load_step(key)
    if d is None:
        raise ValueError('unknown stage ' + key)
    pred_wkb = _pred_index().get(key)
    base = wkb.loads(bytes.fromhex(d['base_wkb']))
    real = wkb.loads(bytes.fromhex(d['real_wkb']))
    pred = wkb.loads(bytes.fromhex(pred_wkb))
    clean = lambda g: g.buffer(0) if not g.is_valid else g
    base, real, pred = clean(base), clean(real), clean(pred)
    real_g = real.difference(base); pred_g = pred.difference(base)
    inter = real_g.intersection(pred_g).area; union = real_g.union(pred_g).area
    giou = inter / union if union > 0 else 0.0
    c = real.centroid; klat = 111320.0; klon = 111320.0 * math.cos(math.radians(c.y))
    acres = lambda g: transform(lambda x, y, z=None: ((x - c.x) * klon, (y - c.y) * klat), g).area / 4046.86
    overlap = real_g.intersection(pred_g); missed = real_g.difference(pred_g); over = pred_g.difference(real_g)
    dets = [dt for dt in d.get('dets', []) if isinstance(dt, dict) and 'lat' in dt and 'lon' in dt]

    xs, ys = [], []
    for g in (base, real, pred):
        x0, y0, x1, y1 = g.bounds; xs += [x0, x1]; ys += [y0, y1]
    W, E, S, N = min(xs), max(xs), min(ys), max(ys)
    mx = (E - W) * 0.07 or 0.01; my = (N - S) * 0.07 or 0.01
    W, E, S, N = W - mx, E + mx, S - my, N + my; latm = (S + N) / 2

    fig, ax = plt.subplots(figsize=(11, 9)); fig.patch.set_facecolor('white'); ax.set_facecolor('#eef1f4')
    ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_aspect(1 / math.cos(math.radians(latm)))
    def fill(g, col, a, z):
        for gg in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if gg.is_empty or gg.geom_type != 'Polygon': continue
            x, y = gg.exterior.xy; ax.fill(x, y, color=col, alpha=a, zorder=z, ec='none')
    def outline(g, ec, lw, ls='-', z=6):
        for gg in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if gg.is_empty or gg.geom_type != 'Polygon': continue
            x, y = gg.exterior.xy; ax.plot(x, y, color=ec, lw=lw, ls=ls, zorder=z)
    fill(overlap, '#2ecc71', 0.55, 3); fill(missed, '#3fb8e0', 0.45, 3); fill(over, '#f39c12', 0.45, 3); fill(base, '#8a8079', 0.35, 2)
    if dets:
        hlon = np.array([x['lon'] for x in dets]); hlat = np.array([x['lat'] for x in dets])
        hfrp = np.array([max(x.get('frp', 0), 0) for x in dets])
        if len(hfrp) > 1400:
            i = np.argsort(hfrp)[-1400:]; hlon, hlat, hfrp = hlon[i], hlat[i], hfrp[i]
        o = np.argsort(hfrp)
        sc = ax.scatter(hlon[o], hlat[o], c=hfrp[o], cmap='plasma', s=10, alpha=0.6, linewidths=0.2,
                        edgecolor='#1a1a1a', zorder=6, vmin=0, vmax=np.percentile(hfrp, 96) if len(hfrp) else 1)
        cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.01); cb.set_label('Detection FRP (MW)', fontsize=9)
    outline(base, '#4d4640', 2.0, ls=(0, (6, 4)), z=7); outline(real, '#c81e3a', 3.2, z=8); outline(pred, '#127a2e', 3.0, z=9)
    ba, ra, pa = acres(base), acres(real), acres(pred); mi, ov = acres(missed), acres(over)
    perim = [Line2D([0], [0], color='#4d4640', lw=2, ls=(0, (6, 4)), label=f'Start perimeter (base) — {ba:,.0f} ac'),
             Line2D([0], [0], color='#c81e3a', lw=3.2, label=f'End perimeter (observed) — {ra:,.0f} ac'),
             Line2D([0], [0], color='#127a2e', lw=3, label=f'PREDICTED (learned_field_v3) — {pa:,.0f} ac')]
    l1 = ax.legend(handles=perim, loc='lower left', fontsize=9, framealpha=1, facecolor='white', edgecolor='#bbb',
                   title='PERIMETERS', title_fontsize=10); l1.set_zorder(20); ax.add_artist(l1)
    acc = [Patch(fc='#2ecc71', alpha=.7, label=f'correct growth — {ra-ba-mi:,.0f} ac'),
           Patch(fc='#3fb8e0', alpha=.7, label=f'missed (recall) — {mi:,.0f} ac'),
           Patch(fc='#f39c12', alpha=.7, label=f'over-painted (precision) — {ov:,.0f} ac')]
    ax.legend(handles=acc, loc='upper right', fontsize=8.5, framealpha=1, facecolor='white', edgecolor='#bbb',
              title='GROWTH ACCURACY', title_fontsize=9.5).set_zorder(20)
    ax.set_xlabel('Longitude', fontsize=10); ax.set_ylabel('Latitude', fontsize=10); ax.tick_params(labelsize=8)
    ax.set_title(f"{d['fire']} — step {d['step']}  ({d.get('stratum','')})\n"
                 f"{ba:,.0f} → {ra:,.0f} ac ({ra/ba:.1f}×)   learned_field_v3   growth-IoU {giou:.3f}",
                 fontsize=13, weight='bold', pad=10)
    ax.grid(True, color='white', lw=0.6, alpha=0.7)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{d['fire']}_step{d['step']}_v3.png")
    fig.savefig(png, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return {'fire': d['fire'], 'step': d['step'], 'stratum': d.get('stratum'),
            'base_acres': round(ba), 'real_acres': round(ra), 'pred_acres': round(pa),
            'growth_iou': round(giou, 3), 'missed_acres': round(mi), 'over_acres': round(ov),
            'n_dets': len(dets), 'png': png}
