"""Per-cycle source-availability log + summary.

Phase 1 (data layer) of FIRE_PERIMETER_REBUILD_PROMPT.md asks for "a
dashboard per incident showing, for the last 24h, per source: fetch success
rate, latency distribution, record count, and last error." This module is
the logging half of that: one append-only JSONL file per fire, plus a
`summarize()` that computes exactly those fields. Nothing here talks to the
network; callers (fusion_v3.run) record after each adapter call.
"""
import json, os, time
from datetime import datetime, timezone


def fire_dir(out_dir):
    """out_dir passed to fusion_v3.run() is a per-cycle timestamped dir
    (<outbase>/<name>/<ts>/); the health log lives one level up, alongside
    run_5min.py's history.jsonl, so it is stable across cycles."""
    return os.path.dirname(os.path.normpath(out_dir))


def log_path(out_dir):
    return os.path.join(fire_dir(out_dir), 'source_health.jsonl')


def record(out_dir, source, status, reason='', latency_s=None, record_count=None):
    """status: 'ok' (evidence added) | 'no_data' (queried fine, nothing to add)
    | 'unavailable' (could not reach/parse the source -- see fire_fusion.Unavailable)."""
    if status not in ('ok', 'no_data', 'unavailable'):
        raise ValueError(f'unknown source_health status {status!r}')
    p = log_path(out_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    row = {
        't': datetime.now(timezone.utc).isoformat(),
        'source': source,
        'status': status,
        'reason': (reason or '')[:400],
        'latency_s': latency_s,
        'record_count': record_count,
    }
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row) + '\n')
    return row


def record_observation(out_dir, name, obs, latency_s=None):
    """Convenience: log the outcome of one obs_* adapter call.

    ``obs`` is whatever the adapter returned: an Observation, an
    Unavailable, a dict (the legacy non-Observation adapters), or None.
    """
    import fire_fusion as FF
    if isinstance(obs, FF.Unavailable):
        return record(out_dir, name, 'unavailable', reason=obs.reason, latency_s=latency_s)
    if obs is None:
        return record(out_dir, name, 'no_data', latency_s=latency_s)
    if isinstance(obs, FF.Observation):
        return record(out_dir, name, 'ok', reason=obs.note, latency_s=latency_s, record_count=1)
    # legacy dict-returning adapters (obs_evacuation_zones, obs_pge_psps, ...)
    return record(out_dir, name, 'ok', reason=json.dumps(obs)[:200], latency_s=latency_s)


def summarize(out_dir, hours=24):
    """Per source: attempts, ok/no_data/unavailable counts, success rate,
    mean latency, and the most recent error -- the Phase 1 dashboard fields."""
    p = log_path(out_dir)
    if not os.path.exists(p):
        return {}
    cutoff = time.time() - hours * 3600
    by_source = {}
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = datetime.fromisoformat(row['t']).timestamp()
            except Exception:
                continue
            if t < cutoff:
                continue
            s = by_source.setdefault(row['source'], {
                'attempts': 0, 'ok': 0, 'no_data': 0, 'unavailable': 0,
                'last_error': None, 'last_error_t': None, '_latencies': [],
                'last_record_count': None,
            })
            s['attempts'] += 1
            status = row.get('status', 'unavailable')
            s[status] = s.get(status, 0) + 1
            if status == 'unavailable':
                s['last_error'] = row.get('reason'); s['last_error_t'] = row['t']
            if row.get('latency_s') is not None:
                s['_latencies'].append(row['latency_s'])
            if row.get('record_count') is not None:
                s['last_record_count'] = row['record_count']
    for s in by_source.values():
        lat = s.pop('_latencies')
        s['success_rate'] = round((s['ok'] + s['no_data']) / s['attempts'], 3) if s['attempts'] else None
        s['mean_latency_s'] = round(sum(lat) / len(lat), 2) if lat else None
    return by_source


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>{fire} -- source health</title>
<style>
  body {{ font: 14px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
          background:#111; color:#e8e8e8; margin:24px; }}
  h1 {{ font-size:18px; margin:0 0 2px; }}
  .meta {{ color:#999; margin-bottom:16px; }}
  table {{ border-collapse:collapse; width:100%; max-width:920px; }}
  th,td {{ padding:6px 10px; border-bottom:1px solid #333; text-align:left; }}
  th {{ color:#aaa; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
  tr:hover {{ background:#1a1a1a; }}
  .src {{ font-weight:600; }}
  .rate {{ font-variant-numeric:tabular-nums; }}
  .bar {{ display:inline-block; height:8px; border-radius:4px; background:#2a2a2a; width:80px; vertical-align:middle; margin-right:6px; overflow:hidden; }}
  .bar > span {{ display:block; height:100%; }}
  .ok {{ color:#39d353; }} .warn {{ color:#e3b341; }} .bad {{ color:#f85149; }}
  .reason {{ color:#999; font-size:12px; max-width:360px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .empty {{ color:#777; padding:24px 0; }}
  .refresh {{ color:#555; font-size:11px; margin-top:20px; }}
</style>
</head><body>
<h1>{fire} &mdash; source health</h1>
<div class="meta">last {hours:g}h, generated {generated}</div>
{body}
<div class="refresh">fpm/source_health.py -- FIRE_PERIMETER_REBUILD_PROMPT.md Phase 1 dashboard slice</div>
</body></html>
"""


def _rate_class(rate):
    if rate is None: return 'warn'
    if rate >= 0.9: return 'ok'
    if rate >= 0.5: return 'warn'
    return 'bad'


def render_html(out_dir, hours=24, fire=None):
    """A small, dependency-free HTML table: per source, attempts / ok /
    no_data / unavailable / success rate / mean latency / last error --
    exactly the Phase 1 Definition-of-Done fields. Sorted worst-first so a
    failing source is the first thing visible, not buried alphabetically."""
    import html as _html
    summary = summarize(out_dir, hours=hours)
    fire = fire or os.path.basename(fire_dir(out_dir)) or '(unknown fire)'
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
    if not summary:
        body = '<div class="empty">No source_health.jsonl records yet for this fire/window.</div>'
    else:
        rows = sorted(summary.items(), key=lambda kv: (kv[1]['success_rate'] if kv[1]['success_rate'] is not None else -1))
        trs = []
        for source, s in rows:
            rate = s['success_rate']
            rate_pct = '?' if rate is None else f'{rate*100:.0f}%'
            cls = _rate_class(rate)
            bar_pct = 0 if rate is None else int(rate * 100)
            bar_color = {'ok': '#39d353', 'warn': '#e3b341', 'bad': '#f85149'}[cls]
            last_err = _html.escape(s['last_error'] or '')
            last_err_t = s['last_error_t'] or ''
            lat = '-' if s['mean_latency_s'] is None else f"{s['mean_latency_s']:.2f}s"
            trs.append(
                f'<tr><td class="src">{_html.escape(source)}</td>'
                f'<td>{s["attempts"]}</td>'
                f'<td class="ok">{s["ok"]}</td>'
                f'<td>{s["no_data"]}</td>'
                f'<td class="{"bad" if s["unavailable"] else ""}">{s["unavailable"]}</td>'
                f'<td class="rate {cls}"><span class="bar"><span style="width:{bar_pct}%;background:{bar_color}"></span></span>{rate_pct}</td>'
                f'<td>{lat}</td>'
                f'<td class="reason" title="{last_err} ({last_err_t})">{last_err}</td></tr>'
            )
        body = (
            '<table><tr><th>Source</th><th>Attempts</th><th>OK</th><th>No data</th>'
            '<th>Unavailable</th><th>Success rate</th><th>Mean latency</th><th>Last error</th></tr>'
            + ''.join(trs) + '</table>'
        )
    return _HTML_TEMPLATE.format(fire=_html.escape(fire), hours=hours, generated=generated, body=body)


def write_html(out_dir, hours=24, fire=None):
    """Write the dashboard next to source_health.jsonl (<outbase>/<name>/source_health.html)."""
    p = os.path.join(fire_dir(out_dir), 'source_health.html')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(render_html(out_dir, hours=hours, fire=fire))
    return p


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--html':
        print(write_html(sys.argv[2] if len(sys.argv) > 2 else '.'))
    else:
        print(json.dumps(summarize(sys.argv[1] if len(sys.argv) > 1 else '.'), indent=2))
