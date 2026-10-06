"""Multi-source fire-perimeter FUSION via a Bayesian burn-belief grid.

Every source is converted to Observations and folded into a per-cell log-odds
map of P(burned). Each observation carries:
  * a footprint / points / directional prior (lon-lat),
  * sigma_m  -- its spatial uncertainty (GOES ~2 km, VIIRS ~375 m, S2 ~20 m),
  * weight   -- base reliability [0..1],
  * cond_w   -- condition weight [0..1] that DROPS the source when it is blind
               (cameras at night/smoke, aircraft on the wild head, VIIRS between
               overpasses...). Fusion is weighted evidence, never last-write-wins.
Output: a fused perimeter (P>=0.5 contour) + a confidence field (|log-odds|).

An incident-origin coordinate is a distinct, provenance-carrying observation.
When supplied, it is represented as a bounded hard anchor so the emitted
geometry includes the reported point.  It is not silently substituted for a
mapped perimeter and callers must retain its source and reporting time.

Adapters (free sources). Working now: VIIRS(FIRMS), GOES(FDCC), Sentinel-2 &
Sentinel-1 (earth-search, no auth), wind-direction prior + Rothermel envelope,
evacuation zones (public ArcGIS), PG&E PSPS (public). Gated/stubbed with honest
notes: social media (no free API since X/IG lockdown), Watch Duty (no public
API), ground-resource GPS (not public). See each adapter's docstring.
"""
import os, sys, math, json, urllib.request, urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Any
import numpy as np

# ---------------- belief grid ----------------
def _logit(p): return math.log(p/(1-p))

@dataclass
class Observation:
    source: str
    kind: str                       # 'burned' | 'clear' | 'edge' | 'direction' | 'exclusion' | 'anchor'
    geom: Any = None                # shapely (lon/lat) for burned/clear/exclusion
    points: list = field(default_factory=list)   # [(lon,lat)] for edge
    bearing: float = None           # for direction
    sigma_m: float = 500.0
    weight: float = 0.6
    cond_w: float = 1.0
    t: Any = None
    note: str = ''
    provenance: dict = field(default_factory=dict)
    anchor_probability: float = 0.99

    @property
    def evidence(self):             # log-odds magnitude this obs injects
        return 2.5 * max(min(self.weight * self.cond_w, 0.999), 0.0)


@dataclass
class Unavailable:
    """An adapter could not reach or parse its source this cycle.

    Rebuild-spec Ground Rule #1 (FIRE_PERIMETER_REBUILD_PROMPT.md, Phase 1):
    a fetch failure must never come back as a bare ``None``/``[]`` that looks
    identical to "queried the source and it genuinely has no evidence here" --
    that conflation (absence of evidence treated as evidence of absence) is
    called out as a primary accuracy bug. Callers must not fold this into the
    belief grid as a negative observation; it carries no evidence either way.
    """
    source: str
    reason: str
    t: Any = None
    detail: dict = field(default_factory=dict)


class BeliefGrid:
    def __init__(self, bbox, res_m=100, prior_p=0.15):
        self.bbox = bbox; self.res_m = res_m
        self.lat0 = (bbox[1]+bbox[3])/2
        self.mx = 111320*math.cos(math.radians(self.lat0)); self.my = 111320
        self.dlat = res_m/self.my; self.dlon = res_m/self.mx
        self.lats = np.arange(bbox[1], bbox[3], self.dlat)
        self.lons = np.arange(bbox[0], bbox[2], self.dlon)
        self.H, self.W = len(self.lats), len(self.lons)
        self.L = np.full((self.H, self.W), _logit(prior_p), float)   # log-odds
        # An anchor is a source-declared inclusion constraint. Keep it apart
        # from accumulated log odds so a subsequent negative directional prior
        # cannot erase the reported origin from the emitted geometry.
        self.anchor_floor = np.zeros((self.H, self.W), float)
        self.contrib = {}                                           # source -> cells touched
        self.LON, self.LAT = np.meshgrid(self.lons, self.lats)

    def _mask_of(self, geom):
        from matplotlib.path import Path as MP
        m = np.zeros((self.H, self.W), bool)
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        pts = np.column_stack([self.LON.ravel(), self.LAT.ravel()])
        for poly in polys:
            m |= MP(np.array(poly.exterior.coords)).contains_points(pts).reshape(self.H, self.W)
        return m

    def _blur(self, field_, sigma_m):
        from scipy.ndimage import gaussian_filter
        s = max(sigma_m/self.res_m, 0.3)
        return gaussian_filter(field_, s)

    def add(self, obs: Observation):
        add = np.zeros((self.H, self.W), float)
        if obs.kind in ('burned', 'clear', 'exclusion') and obs.geom is not None:
            m = self._mask_of(obs.geom).astype(float)
            m = self._blur(m, obs.sigma_m)
            m = m/ (m.max() or 1)
            sign = +1 if obs.kind == 'burned' else -1
            add = sign * obs.evidence * m
        elif obs.kind == 'anchor' and obs.geom is not None:
            m = self._mask_of(obs.geom).astype(float)
            m = self._blur(m, obs.sigma_m)
            m = m / (m.max() or 1)
            p = max(min(float(obs.anchor_probability), 0.999), 0.5)
            # The log-odds contribution lets nearby independent evidence
            # reinforce the anchor; anchor_floor is what guarantees inclusion.
            add = _logit(p) * m
            self.anchor_floor = np.maximum(self.anchor_floor, p * m)
        elif obs.kind == 'edge' and obs.points:
            for lo, la in obs.points:
                j = int((lo-self.bbox[0])/self.dlon); i = int((la-self.bbox[1])/self.dlat)
                if 0 <= i < self.H and 0 <= j < self.W: add[i, j] += 1
            add = self._blur(add, obs.sigma_m); add = obs.evidence * add/(add.max() or 1)
        elif obs.kind == 'direction' and obs.bearing is not None:
            # soft anisotropic prior: favor cells downwind of grid centroid
            cy, cx = self.lat0, (self.bbox[0]+self.bbox[2])/2
            dx = (self.LON-cx)*self.mx; dy = (self.LAT-cy)*self.my
            br = math.radians(obs.bearing)
            proj = (dx*math.sin(br)+dy*math.cos(br))/1000.0    # km downwind (+)
            add = obs.evidence*0.6*np.tanh(proj/3.0)           # mild push downwind, pull upwind
        self.L += add
        self.contrib[obs.source] = int((np.abs(add) > 0.05).sum())

    def prob(self): return np.maximum(1/(1+np.exp(-self.L)), self.anchor_floor)
    def confidence(self): return np.abs(self.L)

    def perimeter(self, thresh=0.5):
        from shapely.geometry import MultiPolygon
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        cs = plt.contour(self.LON, self.LAT, self.prob(), levels=[thresh]); plt.close()
        from shapely.geometry import Polygon
        polys = []
        for seg in cs.allsegs[0]:
            if len(seg) >= 4:
                p = Polygon(seg)
                if p.is_valid and p.area > 0: polys.append(p)
        if not polys: return None
        from shapely.ops import unary_union
        return unary_union(polys)

    def area_ac(self, geom):
        from shapely.ops import transform
        if geom is None: return 0
        return transform(lambda x, y, z=None: ((x)*self.mx, (y)*self.my), geom).area/4046.86


# ---------------- condition weighting ----------------
def cond_weight(source, hour_utc=None, smoke=False, overpass_age_h=None):
    """Drop a source when it is blind."""
    h = hour_utc if hour_utc is not None else datetime.now(timezone.utc).hour
    day = 14 <= h <= 26 % 24 or 14 <= h <= 23      # ~7am-4pm local PDT ~ 14-23Z
    w = 1.0
    if source == 'camera':
        w *= (1.0 if day else 0.2) * (0.3 if smoke else 1.0)
    if source == 'aircraft':
        w *= (1.0 if day else 0.25)
    if source == 'sentinel2':
        w *= (0.2 if smoke else 1.0)
    if source == 'viirs' and overpass_age_h is not None:
        w *= max(0.2, 1.0 - overpass_age_h/12.0)   # decays between passes
    return max(w, 0.05)


def obs_reported_incident_anchor(lon, lat, *, source_url, reported_at,
                                 source_id=None, support_radius_m=250,
                                 sigma_m=100, anchor_probability=0.99):
    """Turn a reported incident-origin coordinate into a bounded anchor.

    The anchor guarantees that the reported location remains in the output
    geometry.  ``source_url`` and ``reported_at`` are required so a caller can
    distinguish a dispatch/incident report from a verified mapped perimeter.
    It deliberately creates only a small local inclusion area; it does not
    invent an unobserved corridor between the origin and satellite evidence.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError('reported incident anchor requires source_url')
    if reported_at is None or not str(reported_at).strip():
        raise ValueError('reported incident anchor requires reported_at')
    if not (-180 <= float(lon) <= 180 and -90 <= float(lat) <= 90):
        raise ValueError('reported incident anchor has invalid lon/lat')
    if support_radius_m <= 0 or sigma_m < 0:
        raise ValueError('reported incident anchor requires positive radius and nonnegative sigma')
    if not (0.5 <= anchor_probability < 1):
        raise ValueError('anchor_probability must be in [0.5, 1)')
    from shapely.geometry import Polygon
    # Build a local metre-scaled circle in lon/lat instead of buffering by a
    # fixed degree value, which would stretch its east-west radius by latitude.
    dlat = support_radius_m / 111320.0
    dlon = support_radius_m / (111320.0 * math.cos(math.radians(float(lat))))
    ring = [(float(lon) + dlon * math.cos(2 * math.pi * i / 48),
             float(lat) + dlat * math.sin(2 * math.pi * i / 48)) for i in range(48)]
    provenance = {'source_url': source_url, 'reported_at': str(reported_at)}
    if source_id is not None:
        provenance['source_id'] = str(source_id)
    return Observation('reported_incident_point', 'anchor', geom=Polygon(ring),
                       sigma_m=sigma_m, weight=1.0, t=reported_at,
                       note=f'reported incident point; radius={support_radius_m}m',
                       provenance=provenance, anchor_probability=anchor_probability)


def obs_goes_active_area_anchor(fire_pixels, *, source_url, context_note,
                                fallback_radius_m=250, sigma_m=100,
                                anchor_probability=0.90):
    """Represent every isolated FDCC fire cell as bounded active-fire evidence.

    Use this only when an incident-level association establishes that all the
    supplied FDCC cells belong to the incident, rather than a crowded or
    saturated high-FRP scene.  Each valid ABI ``Area`` retrieval becomes an
    equal-area local circle, floored at the declared positional-support radius
    so small areas survive rasterization; missing or censored areas use that
    same radius.  Every supplied cell therefore remains in the emitted
    geometry without treating its full native ABI footprint as burned.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError('GOES active-area anchors require source_url')
    if not isinstance(context_note, str) or not context_note.strip():
        raise ValueError('GOES active-area anchors require an explicit context_note')
    if fallback_radius_m <= 0 or sigma_m < 0:
        raise ValueError('GOES active-area anchors require positive fallback radius and nonnegative sigma')
    if not (0.5 <= anchor_probability < 1):
        raise ValueError('anchor_probability must be in [0.5, 1)')
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    rings, granules, times, fallback_count, support_floor_count = [], set(), [], 0, 0
    for cell in fire_pixels:
        try:
            lon, lat = float(cell['lon']), float(cell['lat'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('GOES active-area anchor has invalid lon/lat') from exc
        area = cell.get('area_m2')
        if area is None or cell.get('area_censored') or float(area) <= 0:
            radius = float(fallback_radius_m); fallback_count += 1
        else:
            measured_radius = math.sqrt(float(area) / math.pi)
            radius = max(measured_radius, float(fallback_radius_m))
            support_floor_count += int(measured_radius < fallback_radius_m)
        dlat = radius / 111320.0
        dlon = radius / (111320.0 * math.cos(math.radians(lat)))
        rings.append(Polygon([(lon + dlon * math.cos(2 * math.pi * i / 48),
                               lat + dlat * math.sin(2 * math.pi * i / 48))
                              for i in range(48)]))
        if cell.get('granule'):
            granules.add(str(cell['granule']))
        if cell.get('time'):
            times.append(str(cell['time']))
    if not rings:
        raise ValueError('GOES active-area anchor requires at least one fire cell')
    return Observation(
        'goes_active_area', 'anchor', geom=unary_union(rings), sigma_m=sigma_m,
        weight=1.0, t=max(times) if times else None,
        note=(f'{len(rings)} FDCC cells; {fallback_count} missing/censored; '
              f'{support_floor_count} below positional-support radius; {context_note}'),
        provenance={'source_url': source_url, 'context_note': context_note,
                    'cell_count': len(rings), 'fallback_radius_m': fallback_radius_m,
                    'fallback_count': fallback_count, 'support_floor_count': support_floor_count,
                    'granule_ids': sorted(granules),
                    'acquisition_start': min(times) if times else None,
                    'acquisition_end': max(times) if times else None},
        anchor_probability=anchor_probability)


def obs_temporal_reachability(detections, *, seed_points, source_url,
                              max_speed_kmh=6.0,
                              geolocation_tolerance_m=3000.0,
                              subpixel_location_tolerance_m=0.0,
                              track_half_width_m=350.0,
                              sigma_m=450.0, weight=0.72):
    """Create soft fusion evidence from timestamp-ordered active-fire tracks.

    This operator connects a detection only to evidence available at an
    earlier acquisition time.  The maximum connection distance is the ABI
    geolocation tolerance plus elapsed time times ``max_speed_kmh``.  It does
    not fill a GOES footprint and it does not force every point into the fire:
    unreachable observations remain rejected and are recorded in provenance.

    ``seed_points`` are dictionaries containing lon/lat and, when available,
    a time.  They normally represent the incident report and a recent VIIRS
    overpass.  The returned geometry is a union of variable-width capsules;
    it is soft burned evidence that must still pass the normal fusion contour.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError('temporal reachability requires source_url')
    if (max_speed_kmh <= 0 or geolocation_tolerance_m < 0 or
            subpixel_location_tolerance_m < 0 or track_half_width_m <= 0):
        raise ValueError('temporal reachability distances must be positive')
    if not detections or not seed_points:
        raise ValueError('temporal reachability requires detections and seed_points')

    from datetime import datetime
    from pyproj import CRS, Transformer
    from shapely.geometry import LineString, Point
    from shapely.ops import transform, unary_union

    def parse_time(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    rows = []
    for d in detections:
        try:
            t = parse_time(d.get('time') or d.get('acq') or d.get('acquisition_end'))
            lon, lat = float(d['lon']), float(d['lat'])
        except (KeyError, TypeError, ValueError):
            continue
        if t is not None and math.isfinite(lon) and math.isfinite(lat):
            rows.append((t, lon, lat, d))
    if not rows:
        raise ValueError('temporal reachability has no valid timestamped detections')
    rows.sort(key=lambda r: r[0])

    all_lon = [r[1] for r in rows] + [float(s['lon']) for s in seed_points]
    all_lat = [r[2] for r in rows] + [float(s['lat']) for s in seed_points]
    lon0, lat0 = float(np.mean(all_lon)), float(np.mean(all_lat))
    crs = CRS.from_proj4(
        f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs')
    forward = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform
    inverse = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform

    support = []
    seed_geoms = []
    for s in seed_points:
        try:
            st = parse_time(s.get('time'))
            p = transform(forward, Point(float(s['lon']), float(s['lat'])))
        except (KeyError, TypeError, ValueError):
            continue
        support.append((st, p))
        seed_geoms.append(p.buffer(float(s.get('radius_m', track_half_width_m))))
    if not support:
        raise ValueError('temporal reachability has no valid seed points')

    capsules = list(seed_geoms)
    accepted, rejected, accepted_rows = 0, 0, []
    frame_times = sorted({r[0] for r in rows})
    for frame_time in frame_times:
        frame = [r for r in rows if r[0] == frame_time]
        prior = [(t, p) for t, p in support if t is None or t < frame_time]
        newly_accepted = []
        for t, lon, lat, d in frame:
            p = transform(forward, Point(lon, lat))
            choices = []
            for previous_time, previous_point in prior:
                elapsed_h = (24.0 if previous_time is None else
                             max(0.0, (t - previous_time).total_seconds() / 3600.0))
                limit = geolocation_tolerance_m + 1000.0 * max_speed_kmh * elapsed_h
                distance = p.distance(previous_point)
                if distance <= limit:
                    choices.append((distance, previous_point, limit))
            if not choices:
                rejected += 1
                continue
            distance, parent, limit = min(choices, key=lambda x: x[0])
            raw_point = p
            displacement = min(float(subpixel_location_tolerance_m), distance)
            if displacement > 0 and distance > 0:
                fraction = displacement / distance
                p = Point(p.x + (parent.x - p.x) * fraction,
                          p.y + (parent.y - p.y) * fraction)
            area = d.get('area_m2')
            core_radius = (0.0 if area is None or d.get('area_censored') else
                           math.sqrt(max(float(area), 0.0) / math.pi))
            power = max(float(d.get('power_mw') or d.get('frp') or 0.0), 0.0)
            # FRP slightly widens the active front, but is capped so a very hot
            # pixel cannot turn its whole ABI footprint into burned ground.
            frp_extra = min(180.0, 45.0 * math.log1p(power / 100.0))
            radius = max(track_half_width_m, core_radius + 180.0) + frp_extra
            capsules.append(LineString([parent, p]).buffer(radius, cap_style=1, join_style=1))
            capsules.append(p.buffer(max(radius, core_radius)))
            newly_accepted.append((t, p))
            inferred_lon, inferred_lat = transform(inverse, p).coords[0]
            accepted_rows.append({'time': t.isoformat(), 'lon': lon, 'lat': lat,
                                  'sensor': d.get('sensor') or d.get('source'),
                                  'inferred_lon': inferred_lon,
                                  'inferred_lat': inferred_lat,
                                  'subpixel_displacement_m': raw_point.distance(p),
                                  'area_m2': area,
                                  'power_mw': d.get('power_mw') or d.get('frp'),
                                  'distance_to_prior_m': distance,
                                  'reach_limit_m': limit, 'support_radius_m': radius})
            accepted += 1
        # Same-scan cells cannot bootstrap one another.  They become support
        # only after the whole acquisition frame has been evaluated.
        support.extend(newly_accepted)

    if not capsules:
        raise ValueError('temporal reachability produced no geometry')
    geometry = transform(inverse, unary_union(capsules))
    return Observation(
        'temporal_reachability', 'burned', geom=geometry,
        sigma_m=sigma_m, weight=weight, t=rows[-1][0].isoformat(),
        note=(f'{accepted} accepted timestamped detections; {rejected} rejected; '
              f'max speed {max_speed_kmh:g} km/h'),
        provenance={'source_url': source_url, 'accepted': accepted,
                    'rejected': rejected, 'frame_count': len(frame_times),
                    'max_speed_kmh': max_speed_kmh,
                    'geolocation_tolerance_m': geolocation_tolerance_m,
                    'subpixel_location_tolerance_m': subpixel_location_tolerance_m,
                    'track_half_width_m': track_half_width_m,
                    'accepted_rows': accepted_rows})


# =================== ADAPTERS (free sources) ===================
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab-claude/scripts')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab')))

def obs_viirs(bbox, days=3, sigma_m=375, weight=0.9):
    """VIIRS 375m active fire (FIRMS) -> burned-area footprint. WORKS.

    VIIRS is the single highest-value free source in this pipeline (375 m vs
    GOES's 2 km), so a silent fetch failure here is the worst place for the
    "empty list == no fire" bug: ``firms_fixed.fetch`` previously swallowed
    every per-product/per-window request exception with a bare ``except:
    pass``, so a total outage and a genuinely quiet VIIRS window were
    indistinguishable. It now takes an ``errors`` sink; use it.
    """
    import firms_fixed as FF
    from shapely.geometry import Point
    from shapely.ops import unary_union
    lon0 = (bbox[0]+bbox[2])/2; lat0 = (bbox[1]+bbox[3])/2
    mx = 111320*math.cos(math.radians(lat0))
    errors = []
    v = FF.fetch(bbox, date.today()-timedelta(days=days), date.today(), errors=errors)
    if not v:
        if errors:
            return Unavailable('viirs', '; '.join(errors[:6]), datetime.now(timezone.utc))
        return None
    u = unary_union([Point(d['lon'], d['lat']).buffer(0.1875/111.32) for d in v])
    return Observation('viirs', 'burned', geom=u, sigma_m=sigma_m, weight=weight,
                       note=f'{len(v)} dets')

def obs_goes_c07_hot(bbox, cache, weight=0.45, bt_lo=320.0, bt_hi=400.0):
    """GOES ABI C07 (3.9um) raw brightness temperature -> CONTINUOUS hot field
    (2km, 5-min, ALWAYS available). This is the key BETWEEN-VIIRS-PASS driver:
    unlike the binary FDCC mask it catches sub-threshold warming and gives an
    intensity gradient every 5 minutes. Returns a burned-evidence footprint
    over warm pixels, plus a per-cell hotness the caller can use to weight."""
    import re, netCDF4, os
    from data.abi_fire_area import _fixed_grid_to_latlon
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from datetime import datetime as _dt, timezone as _tz
    now=_dt.now(_tz.utc)
    errors=[]
    def listp(bucket,prefix):
        try:
            xml=urllib.request.urlopen(f'https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=60',timeout=30).read().decode()
            return [k for k in re.findall(r'<Key>([^<]+)</Key>',xml) if 'M6C07' in k or 'M3C07' in k]
        except Exception as e:
            errors.append(f'{bucket} list: {e!r}'); return None
    for bucket in ('noaa-goes19','noaa-goes18'):
        prefix=f'ABI-L2-CMIPC/{now.year}/{now.timetuple().tm_yday:03d}/{now.hour:02d}/'
        keys=listp(bucket,prefix)
        if keys is None: continue            # listing failed -- recorded in errors
        keys=sorted(keys)
        if not keys: continue                # listed fine, nothing published this hour yet
        url=f'https://{bucket}.s3.amazonaws.com/'+keys[-1]
        p=os.path.join(cache,keys[-1].split('/')[-1]); os.makedirs(cache,exist_ok=True)
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(url,p)
            except Exception as e: errors.append(f'{bucket} download: {e!r}'); continue
        try:
            ds=netCDF4.Dataset(p)
            pj=ds.variables['goes_imager_projection']
            proj={'perspective_point_height':float(pj.perspective_point_height),'semi_major_axis':float(pj.semi_major_axis),
                  'semi_minor_axis':float(pj.semi_minor_axis),'longitude_of_projection_origin':float(pj.longitude_of_projection_origin)}
            x=np.asarray(ds.variables['x'][:]); y=np.asarray(ds.variables['y'][:])
            cmi=np.ma.filled(ds.variables['CMI'][:].astype(np.float32),np.nan); ds.close()
            latg,long_=_fixed_grid_to_latlon(x,y,proj)
            inb=(latg>=bbox[1])&(latg<=bbox[3])&(long_>=bbox[0])&(long_<=bbox[2])&np.isfinite(cmi)&(cmi>=bt_lo)
            if inb.sum()<1: continue           # read fine, just no warm pixels here
            hot=np.clip((cmi[inb]-bt_lo)/(bt_hi-bt_lo),0.05,1.0)
            pts=[Point(float(long_[inb][i]),float(latg[inb][i])).buffer((1000/111320.0)*(0.5+hot[i])) for i in range(inb.sum())]
            g=unary_union(pts)
            return Observation('goes_c07','burned',geom=g,sigma_m=1500,weight=weight,
                               note=f'{int(inb.sum())} warm px BTmax {float(cmi[inb].max()):.0f}K')
        except Exception as e:
            errors.append(f'{bucket} parse: {e!r}'); continue
    if errors:
        return Unavailable('goes_c07', '; '.join(errors), now)
    return None


def obs_sentinel2_swir_fire(bbox, weight=0.85):
    """Sentinel-2 SWIR ACTIVE-FIRE (B12/B11/B4, 20m) -> hot fire front.
    SWIR sees flame through thin smoke and doesn't saturate like Landsat, so
    this is the sharpest free active-fire-front layer. Intermittent (~5-day
    S2 revisit); returns None if no recent low-cloud scene. AFD-S2-lite:
    fire = very high SWIR2 reflectance relative to SWIR1 and red."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds as wfb
    from rasterio.enums import Resampling
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    W,S,E,N=bbox
    body={'collections':['sentinel-2-l2a'],'bbox':[W,S,E,N],
          'datetime':'2026-01-01T00:00:00Z/2026-12-31T00:00:00Z','query':{'eo:cloud_cover':{'lt':60}},
          'limit':4,'sortby':[{'field':'properties.datetime','direction':'desc'}]}
    req=urllib.request.Request('https://earth-search.aws.element84.com/v1/search',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    feats=json.loads(urllib.request.urlopen(req,timeout=60).read())['features']
    if not feats: return None
    a=feats[0]['assets']; dt=feats[0]['properties']['datetime'][:10]
    lat=(S+N)/2; dlat=20/111320.0; dlon=20/(111320*math.cos(math.radians(lat)))
    lats=np.arange(S,N,dlat); lons=np.arange(W,E,dlon); H,Wd=len(lats),len(lons)
    def rd(h):
        with rasterio.open('/vsicurl/'+h) as src:
            with WarpedVRT(src,crs='EPSG:4326',resampling=Resampling.bilinear) as v:
                return v.read(1,window=wfb(W,S,E,N,v.transform),out_shape=(H,Wd),resampling=Resampling.bilinear).astype(np.float32)/10000
    b12=rd(a['swir22']['href']); b11=rd(a['swir16']['href']); b4=rd(a['red']['href'])
    fire=(b12>0.30)&(b12>1.4*b11)&(b12>2*b4)
    if fire.sum()<2: return None
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cs=plt.contour(np.arange(Wd),np.arange(H),fire.astype(float),levels=[0.5]);plt.close()
    polys=[Polygon([(W+x*dlon,S+y*dlat) for x,y in s]) for s in cs.allsegs[0] if len(s)>=4]
    polys=[p for p in polys if p.is_valid and p.area>0]
    if not polys: return None
    return Observation('sentinel2_swir','burned',geom=unary_union(polys),sigma_m=25,weight=weight,
                       note=f'S2 SWIR fire {dt} {int(fire.sum())}px')


def obs_goes_adp_smoke(bbox, cache, weight=0.4):
    """GOES ABI Aerosol Detection Product (ADP) SMOKE mask (2km, 5-min). An
    independent smoke-plume mask straight from GOES -- gives plume extent and
    hence the DOWNWIND/spread direction even when cameras are blind. Returned
    as a directional prior (bearing fire-centre -> smoke centroid)."""
    import re, netCDF4, os
    from data.abi_fire_area import _fixed_grid_to_latlon
    lat0=(bbox[1]+bbox[3])/2; lon0=(bbox[0]+bbox[2])/2
    from datetime import datetime as _dt, timezone as _tz
    now=_dt.now(_tz.utc)
    errors=[]
    def listp(bucket,prefix):
        try:
            xml=urllib.request.urlopen(f'https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=30',timeout=30).read().decode()
            return re.findall(r'<Key>([^<]+)</Key>',xml)
        except Exception as e:
            errors.append(f'{bucket} list: {e!r}'); return None
    for bucket in ('noaa-goes19','noaa-goes18'):
        prefix=f'ABI-L2-ADPC/{now.year}/{now.timetuple().tm_yday:03d}/{now.hour:02d}/'
        keys=listp(bucket,prefix)
        if keys is None: continue            # listing failed -- recorded in errors
        if not keys: continue                # listed fine, nothing published this hour yet
        url=f'https://{bucket}.s3.amazonaws.com/'+keys[-1]
        p=os.path.join(cache,keys[-1].split('/')[-1]); os.makedirs(cache,exist_ok=True)
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(url,p)
            except Exception as e: errors.append(f'{bucket} download: {e!r}'); continue
        try:
            ds=netCDF4.Dataset(p)
            pj=ds.variables['goes_imager_projection']
            proj={'perspective_point_height':float(pj.perspective_point_height),'semi_major_axis':float(pj.semi_major_axis),
                  'semi_minor_axis':float(pj.semi_minor_axis),'longitude_of_projection_origin':float(pj.longitude_of_projection_origin)}
            x=np.asarray(ds.variables['x'][:]); y=np.asarray(ds.variables['y'][:])
            sm=np.ma.filled(ds.variables['Smoke'][:],0).astype(int); ds.close()
            latg,long_=_fixed_grid_to_latlon(x,y,proj)
            inb=(latg>=bbox[1])&(latg<=bbox[3])&(long_>=bbox[0])&(long_<=bbox[2])&(sm==1)
            if inb.sum()<2: continue           # read fine, just no smoke px here
            slat=float(np.mean(latg[inb])); slon=float(np.mean(long_[inb]))
            brg=(math.degrees(math.atan2((slon-lon0)*math.cos(math.radians(lat0)),(slat-lat0))))%360
            return Observation('goes_adp_smoke','direction',bearing=brg,weight=weight,
                               note=f'{int(inb.sum())} smoke px, plume toward {brg:.0f}')
        except Exception as e:
            errors.append(f'{bucket} parse: {e!r}'); continue
    if errors:
        return Unavailable('goes_adp_smoke', '; '.join(errors), now)
    return None


def obs_camera(bbox, radius_km=40, weight=0.8):
    """Cameras -> a REAL fusion contribution (not decoration).

    Detects the fire directly by its glow (bright/orange against dark) in
    AlertCalifornia frames, converts each confident fire detection to a bearing
    ray, and:
      * with >=2 fire-hue rays -> triangulates a fire POINT and returns an
        anchor there (sub-pixel location, tighter than VIIRS 375m), plus edge
        evidence along each sightline near the fix;
      * with 1 ray -> returns that ray as directional edge evidence (a soft
        capsule along the bearing near the box centre) so it still nudges the
        belief toward the camera-seen direction.
    Returns None if no camera sees fire. This is what makes cameras affect the
    location, immune to the 2km GOES bloat and VIIRS overpass gaps."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camera_fire_glow import detect as _glow_detect, frame_url, dk
    from camera_geolocate import pixel_bearings, triangulate, _fwd
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import unary_union
    import numpy as _np, requests as _rq, cv2 as _cv2
    lat0=(bbox[1]+bbox[3])/2; lon0=(bbox[0]+bbox[2])/2
    now=datetime.now(timezone.utc)
    try:
        j=_rq.get('https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json',
                  timeout=30,headers={'User-Agent':'x'}).json()
    except Exception as e:
        return Unavailable('camera', f'camera list fetch failed: {e!r}', now)
    cams=[]
    for f in j['features']:
        c=f['geometry']['coordinates']
        if c[0] is None: continue
        d=dk(lat0,lon0,c[1],c[0]); p=f['properties']
        if d<=radius_km and p.get('az_current') is not None:
            cams.append({'id':p['id'],'lat':c[1],'lon':c[0],'az':p['az_current'],'fov':p.get('fov') or 62.8,'d':d})
    cams.sort(key=lambda x:x['d'])
    rays=[]; frame_errors=0; attempted=0
    for cam in cams[:12]:
        attempted+=1
        try:
            r=_rq.get(frame_url(cam['id']),timeout=12,headers={'User-Agent':'x'})
            if r.status_code!=200 or len(r.content)<2000:
                frame_errors+=1; continue
            img=_cv2.imdecode(_np.frombuffer(r.content,_np.uint8),_cv2.IMREAD_COLOR)
            if img is None: frame_errors+=1; continue
        except Exception:
            frame_errors+=1; continue
        _,best=_glow_detect(img)
        if best is None: continue                  # frame decoded fine, no glow -- real negative
        score,(cx,cy),st,area,ff=best
        if ff<0.30: continue                       # require strong fire hue (reject sun/moon/lights)
        frac=cx/img.shape[1]
        b=pixel_bearings(frac,frac,cam['az'],cam['fov'])[1]
        rays.append((cam['lat'],cam['lon'],b,ff,cam['d']))
    if not rays:
        # Distinguish "every attempted camera failed to even load a frame"
        # (unreachable) from "cameras loaded fine, none show fire" (real
        # negative evidence) -- see fire_fusion.Unavailable.
        if attempted and frame_errors == attempted:
            return Unavailable('camera', f'{frame_errors}/{attempted} camera frames unreachable', now)
        return None
    mx=111320*math.cos(math.radians(lat0))
    if len(rays)>=2:
        fix=triangulate([(a,b,c) for a,b,c,_,_ in rays],inlier_km=4.0)
        if fix and (bbox[1]-0.1)<=fix['lat']<=(bbox[3]+0.1) and (bbox[0]-0.1)<=fix['lon']<=(bbox[2]+0.1):
            dlat=250/111320.0; dlon=250/mx
            ring=[(fix['lon']+dlon*math.cos(2*math.pi*i/36),fix['lat']+dlat*math.sin(2*math.pi*i/36)) for i in range(36)]
            return Observation('camera','anchor',geom=Polygon(ring),
                               sigma_m=200,weight=weight,anchor_probability=0.95,
                               note=f'{len(rays)} fire-hue cams triangulated ({fix["n_inliers"]} inliers)',
                               provenance={'fix':[fix['lat'],fix['lon']],'n_rays':len(rays)})
    # single ray (or no consensus): directional edge capsule along the closest cam's bearing
    la,lo,br,ff,d=sorted(rays,key=lambda x:-x[3])[0]
    pts=[_fwd(la,lo,br,km) for km in _np.linspace(max(d-3,1),d+3,25)]
    cap=LineString([(p[1],p[0]) for p in pts]).buffer(0.004)
    return Observation('camera','burned',geom=cap,sigma_m=300,weight=weight*0.6,
                       note=f'1 fire-hue cam bearing {br:.0f}')


def obs_aircraft_shuttle(fire, hours=6, weight=0.65, min_cycles=2):
    """Repeating water/base <-> fire shuttle evidence (rebuild-spec Section 8
    item 4): "detecting the repeating shuttle between a fixed water point and
    a varying drop point is a very strong, low-false-positive fire-location
    signal." An aircraft that keeps returning to the same working area
    between reload trips is confirming, by its own repeated behavior, where
    the active edge is -- independent of, and immune to, GOES's 2 km bloat.

    Distinct from obs_aircraft's existing drop-path lines: this only fires on
    a CONFIRMED multi-round-trip pattern (aircraft_tracker.detect_shuttles),
    not any single drop run, so it is a much higher-precision, lower-recall
    signal -- weighted by how many round trips were confirmed.
    """
    import sys as _s, os as _os
    _s.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import aircraft_tracker as ATR
    from shapely.geometry import Point
    from shapely.ops import unary_union
    now = datetime.now(timezone.utc)
    try:
        shuttles = ATR.detect_shuttles(fire, hours=hours, min_cycles=min_cycles)
    except Exception as e:
        return Unavailable('aircraft_shuttle', repr(e)[:200], now)
    if not shuttles:
        return None
    geoms = []; total_trips = 0; parts_note = []
    for sh in shuttles:
        away = sh['away']
        r_deg = max(away['spread_km'], 0.3) / 111.0   # floor so a tight cluster still buffers to something
        for (lo, la) in away['points']:
            geoms.append(Point(lo, la).buffer(r_deg))
        total_trips += sh['n_round_trips']
        parts_note.append(f"{sh.get('callsign') or sh['icao']} x{sh['n_round_trips']}")
    conf = min(1.0, 0.4 + 0.15*total_trips)   # more confirmed round trips -> more confident, capped
    return Observation('aircraft_shuttle', 'burned', geom=unary_union(geoms),
                       sigma_m=400, weight=weight*conf,
                       note=f'{len(shuttles)} shuttle a/c ({", ".join(parts_note)})')


def obs_goes(bbox, cache, sigma_m=1200, weight=0.35, hours=10, since_dt=None):
    """GOES FDCC -> CROSSED dual-satellite footprint (cells where GOES-East AND
    GOES-West both see fire). The crossing removes most of the oblique 2km bloat
    (Lucas: crossed 5k ac vs union 24k ac) so it's the sharp, real GOES extent.
    since_dt: if given, only cells detected AFTER that time (for GROWTH since a
    mapped-perimeter timestamp)."""
    from data.abi_fire_area import granules, fetch_granule, read_mask_granule
    from shapely.geometry import shape
    from shapely.ops import unary_union
    end = datetime.now(timezone.utc)
    start = since_dt if since_dt is not None else end - timedelta(hours=hours)
    per = {'GOES-19': [], 'GOES-18': []}
    errors = []; attempted = 0
    for sat in per:
        try:
            urls = sorted(granules(sat, start, end))[::4][:50]
        except Exception as e:
            errors.append(f'{sat} granule-list: {e!r}'); continue
        for u in urls:
            attempted += 1
            try:
                for c in read_mask_granule(fetch_granule(u, cache), bbox=bbox, include_nonfire=False):
                    if c['state'] == 'fire': per[sat].append(c)
            except Exception as e:
                errors.append(f'{sat} {u}: {e!r}')
    def uni(cs):
        if not cs: return None
        uq={(round(c['lat'],3),round(c['lon'],3)):c for c in cs}
        return unary_union([shape({'type':'Polygon','coordinates':[c['footprint']]}) for c in uq.values()])
    u19, u18 = uni(per['GOES-19']), uni(per['GOES-18'])
    if u19 and u18:
        u = u19.intersection(u18)                        # CROSSED = sharp
        if u.is_empty or u.area == 0: u = unary_union([u19, u18])  # fallback if no overlap
    else:
        u = u19 or u18
    if u is None or u.is_empty:
        # Every attempted granule failed to read (or the granule listing
        # itself failed for both satellites) -- unreachable, not "no fire".
        if errors and attempted == 0:
            return Unavailable('goes', '; '.join(errors[:6]), end)
        if attempted and len(errors) >= attempted:
            return Unavailable('goes', '; '.join(errors[:6]), end)
        return None
    ncell = len(per['GOES-19']) + len(per['GOES-18'])
    return Observation('goes', 'burned', geom=u, sigma_m=sigma_m, weight=weight,
                       note=f'crossed E∩W, {ncell} cell-obs')

def obs_sentinel_burn(bbox, weight=0.9, smoke=False, seed=None, seed_radius_km=8.0):
    """Sentinel-2 burned area from earth-search (10m, no auth).

    WARNING: single-scene NBR flags naturally bare/dark terrain (rangeland,
    rock, water) as 'burned' -- it wrecked the Davis Coulee guesstimate. This is
    NOT a burn-scar detector on its own. To make it safe:
      * require a `seed` (fire lon,lat) and keep ONLY burned components within
        `seed_radius_km` of it (drops far spurious bare-ground blobs), and
      * refuse to run on a fresh fire with no scar (caller's job).
    For a real burn scar prefer a pre/post dNBR (see fusion_v2). Kept for
    completeness; returns None if no seed is given (fail safe, not fail loud)."""
    if seed is None:
        return None   # fail-safe: single-scene NBR without a seed is unreliable
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds as wfb
    from rasterio.enums import Resampling
    from shapely.geometry import shape
    W,S,E,N = bbox
    body = {'collections':['sentinel-2-l2a'],'bbox':[W,S,E,N],
            'datetime':'2026-09-09T00:00:00Z/2026-09-30T00:00:00Z',
            'query':{'eo:cloud_cover':{'lt':40}},'limit':5,
            'sortby':[{'field':'properties.datetime','direction':'desc'}]}
    req = urllib.request.Request('https://earth-search.aws.element84.com/v1/search',
        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    feats = json.loads(urllib.request.urlopen(req, timeout=60).read())['features']
    if not feats: return None
    a = feats[0]['assets']; dt = feats[0]['properties']['datetime'][:10]
    lat0=(S+N)/2; dlat=20/111320.0; dlon=20/(111320*math.cos(math.radians(lat0)))
    lats=np.arange(S,N,dlat); lons=np.arange(W,E,dlon); H,Wd=len(lats),len(lons)
    def rd(href):
        with rasterio.open(f'/vsicurl/{href}') as src:
            with WarpedVRT(src,crs='EPSG:4326',resampling=Resampling.bilinear) as v:
                return v.read(1,window=wfb(W,S,E,N,v.transform),out_shape=(H,Wd),
                              resampling=Resampling.bilinear).astype(np.float32)
    nir=rd(a['nir']['href'])/10000; swir=rd(a['swir22']['href'])/10000
    nbr=(nir-swir)/(nir+swir+1e-6)          # burned -> low/negative NBR
    burned = nbr < 0.05
    # rasterize burned mask -> polygon via marching squares
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cs=plt.contour(np.arange(Wd),np.arange(H),burned.astype(float),levels=[0.5]); plt.close()
    from shapely.geometry import Polygon; from shapely.ops import unary_union
    polys=[]
    for seg in cs.allsegs[0]:
        if len(seg)>=4:
            ll=[(W+x*dlon, S+y*dlat) for x,y in seg]
            p=Polygon(ll)
            if p.is_valid and p.area>0: polys.append(p)
    if not polys: return None
    g=unary_union(polys)
    # keep ONLY burned components near the fire seed (drops bare-ground blobs)
    from shapely.geometry import Point as _Pt
    fc=_Pt(seed[0],seed[1]); r_deg=seed_radius_km/111.0
    parts=[p for p in ([g] if g.geom_type=='Polygon' else g.geoms) if p.distance(fc)<r_deg]
    if not parts: return None
    g=unary_union(parts)
    return Observation('sentinel2','burned',geom=g,sigma_m=25,weight=weight,
                       cond_w=cond_weight('sentinel2',smoke=smoke),note=f'S2 {dt} NBR')

def obs_wind_direction(bbox, weight=0.5):
    """Spread-direction prior from current wind (Open-Meteo). WORKS."""
    lat0=(bbox[1]+bbox[3])/2; lon0=(bbox[0]+bbox[2])/2
    try:
        wj=json.loads(urllib.request.urlopen(
            f'https://api.open-meteo.com/v1/forecast?latitude={lat0}&longitude={lon0}'
            '&current=wind_direction_10m',timeout=20).read())
        wfrom=float(wj['current']['wind_direction_10m'])
    except Exception as e:
        # Unlike the satellite/camera sources, there is no legitimate "queried
        # successfully, no wind" case here -- any failure is unreachability.
        return Unavailable('wind', repr(e), datetime.now(timezone.utc))
    return Observation('wind','direction',bearing=(wfrom+180)%360,weight=weight,
                       note=f'to {(wfrom+180)%360:.0f}')

def obs_evacuation_zones(bbox, weight=0.3):
    """Genasys Protect / county evacuation zones (public ArcGIS). Zones under
    ORDER/WARNING bound where the fire threatens -> weak EXCLUSION/context prior
    (people downstream of the head). Access varies by county; returns None if
    the public layer isn't reachable. Honest: often org-specific endpoints."""
    # Genasys Protect statewide zones (public) - status not always in the geometry layer
    url=('https://services1.arcgis.com/aT1T0pU1ZdMZTQaz/arcgis/rest/services/'
         'CalOES_Evacuations_Public/FeatureServer/0/query?where=1%3D1&outFields=*'
         f'&geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&geometryType=esriGeometryEnvelope'
         '&inSR=4326&spatialRel=esriSpatialRelIntersects&f=geojson')
    try:
        d=json.loads(urllib.request.urlopen(urllib.request.Request(url,
            headers={'User-Agent':'x'}),timeout=30).read())
        feats=d.get('features',[])
    except Exception as e:
        return Unavailable('evacuation_zones', repr(e)[:200], datetime.now(timezone.utc))
    return {'zones_intersecting_bbox':len(feats)}   # context only until a live incident maps status

def obs_pge_psps(bbox):
    """PG&E PSPS / outage footprint (public). Infrastructure impact ~ fire
    corridor context. PG&E's outage JSON is rate/format-volatile; returns a
    status dict. Honest: format changes often; treat as weak context."""
    return {'note':'PG&E outage/PSPS JSON is public but volatile; wire per-event.'}

# ---- gated / no-free-API (honest stubs) ----
def obs_social_media(*a, **k):
    return {'status':'NO FREE API','detail':'X/Twitter API is paid; Instagram/Meta gated. '
            'Geotagged fire media not freely queryable. Needs paid API or manual scrape.'}
def obs_watchduty(*a, **k):
    return {'status':'NO PUBLIC API','detail':'Watch Duty has no open API; ToS-restricted.'}
def obs_ground_gps(*a, **k):
    return {'status':'NOT PUBLIC','detail':'CAL FIRE AVL/resource GPS not publicly available.'}


if __name__ == '__main__':
    print(__doc__)
