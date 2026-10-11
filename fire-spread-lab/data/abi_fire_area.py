"""
ABI L2 Fire-Area ingest (System scope: the fill-fraction measurement).

Pulls the raw ABI-L2-FDCC granules from the public GOES S3 buckets and
extracts, per 2 km pixel: Area (m2 actively burning), Temp (K), Power
(MW/FRP), Mask (incl. saturation), each geolocated to lat/lon. Area /
pixel-footprint would estimate active subpixel area fraction. This experiment
currently divides by a NOMINAL nadir footprint, not a geolocated pixel area;
it must not be interpreted as measured cumulative burned fraction. Saturated
retrievals are censored/unknown, never imputed as fully burning pixels.

No credentials: the noaa-goes19 / noaa-goes18 buckets are public over
plain HTTPS. Requires netCDF4 (installed).

  granules(sat, start, end)     -> list of granule URLs in the window
  read_fire_pixels(path, bbox)  -> [{lat,lon,area_m2,temp_k,power_mw,mask,saturated}]
  fill_field(pixels, tile_km)   -> per-tile fill fraction at the latest obs
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os, math, urllib.request
from datetime import datetime, timezone, timedelta

BUCKET = {'GOES-19': 'noaa-goes19', 'GOES-18': 'noaa-goes18'}
PIXEL_M2 = 2000.0 * 2000.0            # nominal 2 km ABI footprint
SAT_MASK_CODES = {11, 31}  # confirmed against cached Mask.flag_meanings
FIRE_MASK_CODES = set(range(10, 16)) | set(range(30, 36))


@lru_cache(maxsize=512)
def _list_prefix(bucket, prefix, max_keys=200):
    url = (f'https://{bucket}.s3.amazonaws.com/?list-type=2'
           f'&prefix={prefix}&max-keys={max_keys}')
    with urllib.request.urlopen(url, timeout=30) as r:
        xml = r.read().decode()
    import re
    return re.findall(r'<Key>([^<]+)</Key>', xml)


def granules(satellite, start_dt, end_dt):
    """FDCC granule URLs whose scan interval overlaps a UTC window.

    NOAA objects are listed once per UTC day (not once per hour): a single
    satellite/day stays below the 1,000-key response cap and this makes cohort
    source snapshotting practical.  Filename timestamps are then used to
    filter scan intervals locally.
    """
    bucket = BUCKET[satellite]
    sat_tag = 'G19' if satellite == 'GOES-19' else 'G18'
    urls, t = [], start_dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_dt.astimezone(timezone.utc)
    days = []
    while t <= end:
        days.append(t)
        t += timedelta(days=1)

    def list_day(day):
        prefix = f'ABI-L2-FDCC/{day.year}/{day.timetuple().tm_yday:03d}/'
        try:
            return _list_prefix(bucket, prefix, max_keys=1000)
        except Exception:
            return []

    # Public S3 listing is I/O-bound.  Bounded concurrency makes a long
    # historical manifest feasible while avoiding a request per step.
    with ThreadPoolExecutor(max_workers=min(8, len(days) or 1)) as pool:
        listings = pool.map(list_day, days)
        for keys in listings:
            for k in keys:
                if f'_{sat_tag}_' not in k:
                    continue
                import re
                m = re.search(r'_s(\d{14})_e(\d{14})_', k)
                if not m:
                    continue
                parse = lambda v: datetime.strptime(v[:13], '%Y%j%H%M%S').replace(tzinfo=timezone.utc)
                s, e = parse(m.group(1)), parse(m.group(2))
                if e >= start_dt.astimezone(timezone.utc) and s <= end:
                    urls.append(f'https://{bucket}.s3.amazonaws.com/{k}')
    return urls


def fetch_granule(url, cache_dir=None):
    """Download a granule to a local file (cached). Returns the path."""
    name = url.rsplit('/', 1)[-1]
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '_abi_cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _fixed_grid_to_latlon(x, y, proj):
    """GOES-R ABI fixed-grid scan angles (rad) -> lat/lon (deg). Standard
    algorithm from the ABI L1b/L2 PUG, vectorised over 1-D x / y meshes."""
    import numpy as np
    H = proj['perspective_point_height'] + proj['semi_major_axis']
    r_eq = proj['semi_major_axis']
    r_pol = proj['semi_minor_axis']
    lon0 = math.radians(proj['longitude_of_projection_origin'])
    X, Y = np.meshgrid(x, y)
    sinx, cosx = np.sin(X), np.cos(X)
    siny, cosy = np.sin(Y), np.cos(Y)
    a = sinx**2 + cosx**2 * (cosy**2 + (r_eq**2 / r_pol**2) * siny**2)
    b = -2.0 * H * cosx * cosy
    c = H**2 - r_eq**2
    disc = b**2 - 4*a*c
    valid = disc >= 0
    rs = np.where(valid, (-b - np.sqrt(np.where(valid, disc, 0))) / (2*a), np.nan)
    sx = rs * cosx * cosy
    sy = -rs * sinx
    sz = rs * cosx * siny
    lat = np.degrees(np.arctan((r_eq**2 / r_pol**2) * (sz / np.sqrt((H - sx)**2 + sy**2))))
    lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return lat, lon


def _fixed_points_to_latlon(x, y, proj):
    """Fixed-grid scan angles to lat/lon for paired, sparse coordinates.

    FDCC frames are continental grids but normally contain only a handful of
    fire pixels near one incident.  This avoids allocating full 2-D grids when
    the caller requests fire evidence only.
    """
    import numpy as np
    X, Y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    H = proj['perspective_point_height'] + proj['semi_major_axis']
    r_eq, r_pol = proj['semi_major_axis'], proj['semi_minor_axis']
    lon0 = math.radians(proj['longitude_of_projection_origin'])
    sinx, cosx, siny, cosy = np.sin(X), np.cos(X), np.sin(Y), np.cos(Y)
    a = sinx**2 + cosx**2 * (cosy**2 + (r_eq**2 / r_pol**2) * siny**2)
    b, c = -2.0 * H * cosx * cosy, H**2 - r_eq**2
    disc = b**2 - 4*a*c
    valid = disc >= 0
    rs = np.where(valid, (-b - np.sqrt(np.where(valid, disc, 0))) / (2*a), np.nan)
    sx, sy, sz = rs * cosx * cosy, -rs * sinx, rs * cosx * siny
    lat = np.degrees(np.arctan((r_eq**2 / r_pol**2) * (sz / np.sqrt((H - sx)**2 + sy**2))))
    lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return lat, lon


def _cell_edges(values):
    """Return fixed-grid cell edges from ABI centre coordinates.

    ABI's x/y coordinates are scan angles, not metres.  Constructing corners
    here (then projecting each corner to the ellipsoid) is deliberately used
    instead of buffering a centre point or assuming a constant 2-km square.
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('ABI coordinate needs at least two cells')
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - (values[1] - values[0]) / 2.0
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2.0
    return edges


def _iso_utc(value):
    """Normalise a netCDF ISO timestamp without inventing delivery time."""
    if value is None:
        return None
    text = str(value)
    return text if text.endswith('Z') or '+' in text[10:] else text + 'Z'


def read_mask_granule(nc_path, bbox=None, include_nonfire=True):
    """Read ABI FDCC mask cells with native ground footprints and provenance.

    This is the System-A ingestion boundary.  A returned ``state`` is one of
    ``fire``, ``observed_nonfire`` or ``unavailable``; unavailable must never
    be treated as a negative fire observation.  Receipt time is intentionally
    null: public archive metadata supplies product creation, not delivery.
    Footprints are WGS84 corner tuples in lon/lat order and retain their native
    varying size and shape.
    """
    import netCDF4
    import numpy as np
    ds = netCDF4.Dataset(nc_path)
    try:
        p = ds.variables['goes_imager_projection']
        proj = {'perspective_point_height': float(p.perspective_point_height),
                'semi_major_axis': float(p.semi_major_axis),
                'semi_minor_axis': float(p.semi_minor_axis),
                'longitude_of_projection_origin': float(p.longitude_of_projection_origin)}
        x, y = np.asarray(ds.variables['x'][:]), np.asarray(ds.variables['y'][:])
        # Some FDCC files encode these masks as unsigned bytes.  Cast before
        # applying the unavailable sentinel so a masked uint8 value does not
        # reject ``-1`` (which previously made otherwise valid scans fail).
        mask = np.ma.filled(ds.variables['Mask'][:].astype(np.int16), -1).astype(int)
        dqf = (np.ma.filled(ds.variables['DQF'][:].astype(np.int16), -1).astype(int)
               if 'DQF' in ds.variables else None)
        area_var = ds.variables.get('Area')
        power_var = ds.variables.get('Power')
        temp_var = ds.variables.get('Temp')
        xe, ye = _cell_edges(x), _cell_edges(y)
        if include_nonfire:
            lat, lon = _fixed_grid_to_latlon(x, y, proj)
            indices = np.where(np.isfinite(lat) & np.isfinite(lon))
            center_lat, center_lon = lat[indices], lon[indices]
            clat, clon = _fixed_grid_to_latlon(xe, ye, proj)
            corners_at = lambda i, yy, xx: [(float(clon[yy, xx]), float(clat[yy, xx])),
                                             (float(clon[yy, xx+1]), float(clat[yy, xx+1])),
                                             (float(clon[yy+1, xx+1]), float(clat[yy+1, xx+1])),
                                             (float(clon[yy+1, xx]), float(clat[yy+1, xx]))]
        else:
            indices = np.where(np.isin(mask, list(FIRE_MASK_CODES)))
            yyv, xxv = indices
            center_lat, center_lon = _fixed_points_to_latlon(x[xxv], y[yyv], proj)
            corner_latlon = [_fixed_points_to_latlon(xe[xxv + dx], ye[yyv + dy], proj)
                             for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
            corners_at = lambda i, yy, xx: [(float(lo[i]), float(la[i])) for la, lo in corner_latlon]
        raw_platform = str(getattr(ds, 'platform_ID', ''))
        platform = {'G19': 'GOES-19', 'G18': 'GOES-18'}.get(
            raw_platform, 'GOES-19' if '_G19_' in os.path.basename(nc_path) else 'GOES-18')
        acquisition_start = _iso_utc(getattr(ds, 'time_coverage_start', None))
        acquisition_end = _iso_utc(getattr(ds, 'time_coverage_end', None))
        created = _iso_utc(getattr(ds, 'date_created', None))
        product = 'ABI-L2-FDCC'
        out = []
        for i, (yy, xx) in enumerate(zip(*indices)):
            la, lo = float(center_lat[i]), float(center_lon[i])
            if not (math.isfinite(la) and math.isfinite(lo)):
                continue
            if bbox and not (bbox[0] <= lo <= bbox[2] and bbox[1] <= la <= bbox[3]):
                continue
            code = int(mask[yy, xx])
            quality = None if dqf is None or dqf[yy, xx] < 0 else int(dqf[yy, xx])
            # FDCC's documented DQF 2+ means a cloud/invalid condition;
            # Mask code 0 is unprocessed.  Neither is observed clear ground.
            unavailable = code <= 0 or (quality is not None and quality >= 2)
            fire = code in FIRE_MASK_CODES
            if not include_nonfire and not fire:
                continue
            corners = corners_at(i, yy, xx)
            if not all(math.isfinite(v) for pair in corners for v in pair):
                unavailable = True
            def measured(var):
                if var is None:
                    return None
                value = var[yy, xx]
                return (None if np.ma.is_masked(value) or not np.isfinite(value)
                        else float(value))
            area_m2 = measured(area_var)
            power_mw = measured(power_var)
            temp_k = measured(temp_var)
            out.append({'lat': la, 'lon': lo, 'footprint': corners,
                        'mask': code, 'dqf': quality,
                        'state': 'fire' if fire else ('unavailable' if unavailable else 'observed_nonfire'),
                        'saturated': code in SAT_MASK_CODES,
                        'sensor': platform, 'platform': platform,
                        'product_id': product, 'acquisition_start': acquisition_start,
                        'acquisition_end': acquisition_end, 'product_created': created,
                        'receipt_time': None, 'receipt_by_cutoff_verified': False,
                        'granule_id': os.path.basename(nc_path),
                        'timing_mode': 'retrospective_acquisition_only',
                        'area_m2': area_m2, 'power_mw': power_mw,
                        'temp_k': temp_k,
                        'area_censored': code in SAT_MASK_CODES,
                        'fill_fraction': (None if area_m2 is None or code in SAT_MASK_CODES
                                          else min(area_m2 / PIXEL_M2, 1.0)),
                        'fill_basis': 'nominal_nadir_4km2; active area, not cumulative burned area'})
        return out
    finally:
        ds.close()


def read_fire_pixels(nc_path, bbox=None):
    """Extract fire pixels (Area/Temp/Power/Mask) geolocated to lat/lon,
    optionally clipped to bbox=(lonW,latS,lonE,latN)."""
    import netCDF4, numpy as np
    ds = netCDF4.Dataset(nc_path)
    p = ds.variables['goes_imager_projection']
    proj = {'perspective_point_height': float(p.perspective_point_height),
            'semi_major_axis': float(p.semi_major_axis),
            'semi_minor_axis': float(p.semi_minor_axis),
            'longitude_of_projection_origin': float(p.longitude_of_projection_origin)}
    area = ds.variables['Area'][:]
    mask = ds.variables['Mask'][:]
    # Retain fire-mask evidence even when the area retrieval is missing.
    # In particular, censoring must not turn saturation into "no fire".
    fy, fx = np.where(np.isin(np.ma.filled(mask, -1), list(FIRE_MASK_CODES)))
    if len(fy) == 0:
        ds.close(); return []
    x = ds.variables['x'][:]; y = ds.variables['y'][:]
    lat_g, lon_g = _fixed_grid_to_latlon(x, y, proj)
    temp = ds.variables['Temp'][:]; power = ds.variables['Power'][:]
    t_mid = float(ds.variables['t'][:])
    epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(seconds=t_mid)
    out = []
    for yy, xx in zip(fy.tolist(), fx.tolist()):
        la, lo = float(lat_g[yy, xx]), float(lon_g[yy, xx])
        if not (math.isfinite(la) and math.isfinite(lo)):
            continue
        if bbox and not (bbox[0] <= lo <= bbox[2] and bbox[1] <= la <= bbox[3]):
            continue
        def measured(value):
            return None if np.ma.is_masked(value) or not np.isfinite(value) else float(value)
        a = measured(area[yy, xx])
        mk = int(mask[yy, xx]) if np.isfinite(mask[yy, xx]) else 0
        out.append({'lat': la, 'lon': lo, 'area_m2': a,
                    'temp_k': measured(temp[yy, xx]),
                    'power_mw': measured(power[yy, xx]),
                    'mask': mk, 'saturated': mk in SAT_MASK_CODES,
                    'time': epoch.isoformat(),
                    'fill_fraction': None if a is None or mk in SAT_MASK_CODES else min(a / PIXEL_M2, 1.0),
                    'fill_basis': 'nominal_nadir_4km2; not cumulative burned fraction',
                    'area_censored': mk in SAT_MASK_CODES,
                    'granule': os.path.basename(nc_path),
                    'time_coverage_end': getattr(ds, 'time_coverage_end', None),
                    'date_created': getattr(ds, 'date_created', None)})
    ds.close()
    return out


def fill_field(pixels, tile_km=2.0):
    """Per-tile fill fraction at the LATEST observation (Area is a state,
    not a time-integral -- so take the most recent, not the sum). A
    saturated pixel retains unknown fill. Nominal footprint only: this is
    not a calibrated spatial perimeter or a cumulative burn fraction."""
    tiles = {}
    for px in pixels:
        i = round(px['lat'] / (tile_km / 111.32))
        j = round(px['lon'] / (tile_km / (111.32 * math.cos(math.radians(px['lat'])))))
        key = (i, j)
        prev = tiles.get(key)
        if prev is None or px['time'] > prev['time']:
            fill = None if px['saturated'] else px['fill_fraction']
            tiles[key] = {'lat': px['lat'], 'lon': px['lon'], 'fill': fill,
                          'time': px['time'], 'saturated': px['saturated']}
    return list(tiles.values())


if __name__ == '__main__':
    import sys
    # test on a downloaded granule
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path and os.path.exists(path):
        px = read_fire_pixels(path)
        print(f'{len(px)} fire pixels')
        for p in px[:5]:
            print(f"  ({p['lat']:.3f},{p['lon']:.3f}) area={p['area_m2']}m2 "
                  f"nominal_fill={p['fill_fraction']} temp={p['temp_k']} "
                  f"power={p['power_mw']}MW sat={p['saturated']}")
