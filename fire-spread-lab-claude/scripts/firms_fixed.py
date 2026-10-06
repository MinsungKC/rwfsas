"""FIXED FIRMS fetch: select products by their ADVERTISED availability window.

The bug (measured, see AGENT_COMMS 2026-09-10 18:10): the champion path requests
the SP/archive VIIRS products always and the NRT products only when the step is
within 14 days of today. But FIRMS' SP archives have stopped:

    VIIRS_SNPP_SP    2012-01-20 .. 2026-04-27
    VIIRS_SNPP_NRT   2026-04-28 .. today
    VIIRS_NOAA20_SP  2018-04-01 .. 2026-05-31
    VIIRS_NOAA20_NRT 2026-06-01 .. today
    VIIRS_NOAA21_NRT 2024-01-17 .. today      <-- never requested at all

98% of our cohort falls after SNPP_SP ends and 94% after NOAA20_SP ends, so the
14-day rule blocks exactly the products that would cover it. Measured on
tartar:3 (2026-08-01): pipeline receives 0 VIIRS detections; 2,674 are available.

The fix is to ask FIRMS what it actually has (`data_availability`) and request
each product over its own window, instead of hard-coding an age rule. NOAA-21 is
added -- note VIIRS_NOAA21_SP does not exist (HTTP 400 "Invalid source"), NRT is
the only route to it.

VIIRS matters here specifically: 375 m is ~28x finer in area than GOES's 2 km,
and the measured root cause of this project's ceiling is a ~2 km point-spread
function (median out-of-perimeter detection offset 1.69 km).
"""
import os, sys, json, time
from datetime import date, timedelta

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../fpm')))
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     '_frozen', 'firms_avail.json')
BASE = 'https://fir ms.modaps.eosdis.nasa.gov/api/area/csv'
WANT = ('VIIRS_SNPP', 'VIIRS_NOAA20', 'VIIRS_NOAA21')


def availability(force=False):
    """{data_id: (min_date, max_date)} straight from FIRMS, cached to disk."""
    if not force and os.path.exists(CACHE):
        age = time.time() - os.path.getmtime(CACHE)
        if age < 6 * 3600:
            return {k: tuple(v) for k, v in json.load(open(CACHE)).items()}
    import requests
    from fire_analysis import FIRMS_KEY
    url = f'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{FIRMS_KEY}/ALL'
    out = {}
    try:
        r = requests.get(url, timeout=60)
        for line in r.text.strip().split('\n')[1:]:
            p = line.split(',')
            if len(p) >= 3:
                out[p[0].strip()] = (p[1].strip(), p[2].strip())
        json.dump({k: list(v) for k, v in out.items()}, open(CACHE, 'w'), indent=1)
    except Exception as e:
        print(f'availability fetch failed ({e}); falling back to cache/defaults')
        if os.path.exists(CACHE):
            return {k: tuple(v) for k, v in json.load(open(CACHE)).items()}
    return out


def products_for(start_d, end_d, avail=None, families=WANT):
    """Every product whose advertised window overlaps [start_d, end_d]."""
    avail = avail or availability()
    s, e = str(start_d), str(end_d)
    keep = []
    for pid, (lo, hi) in avail.items():
        if not any(pid.startswith(f) for f in families):
            continue
        if lo <= e and hi >= s:            # windows overlap
            keep.append(pid)
    return sorted(keep)


def fetch(bbox, start_d, end_d, families=WANT, pad_deg=0.0, verbose=False, errors=None):
    """VIIRS detections over [start_d, end_d], using every covering product.

    ``errors``: optional list. If given, every request/window that fails
    (network error, non-200, or an "Invalid ..." FIRMS response body) appends
    a short description instead of being silently dropped. Callers use this
    to tell "queried FIRMS, genuinely nothing there" from "FIRMS was
    unreachable" -- see fire_fusion.Unavailable and rebuild-spec Ground Rule
    #1. Kept optional/default-None so existing callers are unaffected.
    """
    import requests
    from fire_analysis import FIRMS_KEY
    x0, y0, x1, y1 = bbox
    area = (f'{x0-pad_deg:.4f},{y0-pad_deg:.4f},'
            f'{x1+pad_deg:.4f},{y1+pad_deg:.4f}')
    prods = products_for(start_d, end_d, families=families)
    if verbose:
        print(f'  products: {prods}')
    seen, out = set(), []
    for product in prods:
        cur = start_d
        while cur <= end_d:
            ndays = min((end_d - cur).days + 1, 5)
            url = f'{BASE}/{FIRMS_KEY}/{product}/{area}/{ndays}/{cur.strftime("%Y-%m-%d")}'
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and r.text.strip() and \
                        not r.text.startswith('Invalid'):
                    lines = r.text.strip().split('\n')
                    hdr = lines[0].split(',')
                    for line in lines[1:]:
                        vals = line.split(',')
                        if len(vals) < len(hdr):
                            continue
                        row = dict(zip(hdr, vals))
                        try:
                            lat = float(row['latitude']); lon = float(row['longitude'])
                            t = row.get('acq_time', '0000').zfill(4)
                            dt = f"{row['acq_date']}T{t[:2]}:{t[2:]}:00"
                            frp = float(row.get('frp') or 0.0)
                        except Exception:
                            continue
                        k = (round(lat, 4), round(lon, 4), dt)
                        if k in seen:
                            continue
                        seen.add(k)
                        out.append({'lat': lat, 'lon': lon, 'frp': frp,
                                    'sensor': 'VIIRS', 'product': product,
                                    'acq': dt})
                elif errors is not None:
                    errors.append(f'{product} {cur}: HTTP {r.status_code} {r.text[:60]!r}')
            except Exception as e:
                if errors is not None:
                    errors.append(f'{product} {cur}: {e!r}')
            cur += timedelta(days=ndays)
    return out


if __name__ == '__main__':
    av = availability(force=True)
    print('FIRMS advertised availability (VIIRS):')
    for k in sorted(av):
        if k.startswith('VIIRS'):
            print(f'   {k:22s} {av[k][0]} .. {av[k][1]}')
    print()
    for s, e in ((date(2025, 8, 14), date(2025, 8, 17)),
                 (date(2026, 5, 10), date(2026, 5, 13)),
                 (date(2026, 8, 1), date(2026, 8, 4))):
        print(f'{s} .. {e}  ->  {products_for(s, e, av)}')
