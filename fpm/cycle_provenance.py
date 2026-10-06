"""Per-cycle input snapshot + structured metrics with a physics-feasibility
alert (rebuild-spec Phase 9 items 3 and 4).

Item 3: "Snapshot inputs every cycle. Any perimeter must be reproducible
offline months later." The spec suggests inputs.tar.zst; zstd isn't a
guaranteed dependency here, so this writes a plain JSON bundle of every
observation actually folded into the belief grid (source, kind, weight,
sigma, note, provenance, and the geometry as WKT) plus the scalar
inputs (wind/fuel/moisture/base) -- the things you need to re-derive the
perimeter, in a format that needs nothing to read back.

Item 4: "Structured logging + metrics (per-cycle: source availability,
counts, runtime, perimeter delta area, ensemble spread). Alert when
perimeter delta exceeds the physics-feasible maximum -- that's a bug
detector." This computes the cycle's growth vs. the previous cycle and
flags it when it exceeds what even an extreme head ROS could burn in the
elapsed time.
"""
import json
import math
import os
import time
from datetime import datetime, timezone


# An extreme sustained head rate of spread. Real wind-driven crown runs top
# out around 100-180 m/min instantaneously; 200 m/min sustained over a whole
# 5-min cycle is a deliberately generous ceiling, so anything above it is a
# near-certain bug (a contamination blob, a bad base, a geometry error),
# not real fire behavior -- which is exactly the "bug detector" the spec wants.
MAX_HEAD_ROS_M_MIN = 200.0


def _fire_dir(out_dir):
    return os.path.dirname(os.path.normpath(out_dir))


def snapshot_inputs(out_dir, name, layers, scalars):
    """Write <out_dir>/<name>_input_bundle.json: every observation used this
    cycle plus the scalar model inputs. `layers` is fusion_v3's source->obs
    dict; `scalars` is any JSON-able dict of the cycle's other inputs
    (wind, fuel, moisture, base acreage, timestamps...)."""
    obs_records = []
    for src, obs in layers.items():
        if src.startswith('_'):          # internal helpers like _aircraft_paths
            continue
        rec = {'source': src}
        for attr in ('kind', 'weight', 'cond_w', 'sigma_m', 'bearing', 'note', 'provenance', 't'):
            v = getattr(obs, attr, None)
            if v is not None:
                rec[attr] = v if attr not in ('t',) else str(v)
        geom = getattr(obs, 'geom', None)
        if geom is not None:
            try:
                rec['geom_wkt'] = geom.wkt
            except Exception:
                rec['geom_wkt'] = None
        obs_records.append(rec)
    bundle = {
        'fire': name,
        'written_utc': datetime.now(timezone.utc).isoformat(),
        'scalars': scalars,
        'observations': obs_records,
    }
    path = os.path.join(out_dir, f'{name}_input_bundle.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(bundle, fh, indent=1, default=str)
    return path


def _prev_cycle(out_dir, name):
    """The most recent prior history.jsonl record for this fire, or None."""
    hist = os.path.join(_fire_dir(out_dir), 'history.jsonl')
    if not os.path.exists(hist):
        return None
    last = None
    for line in open(hist, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except Exception:
            pass
    return last


def feasible_growth_acres(base_acres, elapsed_min, max_ros_m_min=MAX_HEAD_ROS_M_MIN):
    """Upper bound on new acres in `elapsed_min`, treating the fire as a
    circle of area base_acres whose entire perimeter advances outward at the
    max head ROS. That is far more than a real fire (only the head advances
    that fast), so exceeding it is a genuine red flag. Uses a floor radius so
    a tiny/zero base still permits a plausible new-ignition disc."""
    base_m2 = max(base_acres, 0.0) * 4046.86
    r0 = max(math.sqrt(base_m2 / math.pi), 200.0)     # floor 200 m
    dr = max_ros_m_min * max(elapsed_min, 0.0)
    grown_m2 = math.pi * (r0 + dr) ** 2
    return max(grown_m2 - base_m2, 0.0) / 4046.86


def record_cycle_metrics(out_dir, name, *, pred_acres, base_acres, runtime_s,
                         source_health_summary, extra=None):
    """Append one structured metrics row to <fire>/cycle_metrics.jsonl and
    return it (with a `physics_alert` flag set when this cycle's growth over
    the previous cycle exceeds the physics-feasible maximum)."""
    now = time.time()
    prev = _prev_cycle(out_dir, name)
    prev_acres = prev.get('acres') if prev else None
    prev_t = None
    if prev and prev.get('t'):
        try:
            prev_t = datetime.fromisoformat(prev['t'].replace('Z', '+00:00')).timestamp()
        except Exception:
            prev_t = None
    elapsed_min = ((now - prev_t) / 60.0) if prev_t else None

    delta_acres = None
    physics_alert = False
    feasible = None
    if prev_acres is not None:
        delta_acres = pred_acres - prev_acres
        if elapsed_min and elapsed_min > 0:
            # Bound growth against the PREVIOUS perimeter over the real
            # elapsed time between cycles.
            feasible = feasible_growth_acres(prev_acres, elapsed_min)
            if delta_acres > feasible:
                physics_alert = True

    n_unavailable = sum(1 for s in (source_health_summary or {}).values()
                        if s.get('unavailable', 0) > 0)
    row = {
        't': datetime.now(timezone.utc).isoformat(),
        'fire': name,
        'pred_acres': round(pred_acres, 1),
        'prev_acres': prev_acres,
        'base_acres': round(base_acres, 1) if base_acres is not None else None,
        'delta_acres': round(delta_acres, 1) if delta_acres is not None else None,
        'elapsed_min': round(elapsed_min, 2) if elapsed_min else None,
        'feasible_growth_acres': round(feasible, 1) if feasible is not None else None,
        'physics_alert': physics_alert,
        'runtime_s': round(runtime_s, 1) if runtime_s is not None else None,
        'sources_unavailable': n_unavailable,
        'sources_total': len(source_health_summary or {}),
    }
    if extra:
        row.update(extra)
    path = os.path.join(_fire_dir(out_dir), 'cycle_metrics.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row) + '\n')
    if physics_alert:
        print(f'  ** PHYSICS ALERT: {name} grew {delta_acres:.0f} ac in '
              f'{elapsed_min:.1f} min (feasible max {feasible:.0f} ac) -- '
              f'likely contamination/bad-base/geometry bug, not real fire',
              flush=True)
    return row


if __name__ == '__main__':
    # tiny self-check of the feasibility bound
    print('feasible growth from 1000 ac in 5 min:',
          round(feasible_growth_acres(1000, 5), 1), 'ac')
    print('feasible growth from 100 ac in 60 min:',
          round(feasible_growth_acres(100, 60), 1), 'ac')
