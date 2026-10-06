"""End-to-end: live AlertCalifornia frames -> smoke masks -> triangulated fire location.

Pulls current frames from cameras near a point, runs the segmentation model on
each, and for every frame with a confident smoke mask derives a bearing to the
plume base, then triangulates across cameras.

The plume BASE (lowest smoke pixels, at the horizontal centre of the mask's
lower edge) is used as the fire-indicative ray, not the plume centroid -- smoke
drifts downwind, so its bulk leans away from the fire while its base stays over
the source.
"""
import sys, math, io, argparse
import numpy as np
import requests
from ultralytics import YOLO
from camera_geolocate import pixel_bearings, triangulate

AC_LIST = 'https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json'
MODEL = 'best_seg_v1.pt'


def dist_km(a, b, c, e):
    R = 6371.0; la1, la2 = math.radians(a), math.radians(c)
    dla = la2-la1; dlo = math.radians(e-b)
    h = math.sin(dla/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return 2*R*math.asin(math.sqrt(h))


def cameras_near(lat, lon, radius_km, cams_json):
    out = []
    for f in cams_json['features']:
        c = f['geometry']['coordinates']
        if c[0] is None:
            continue
        clon, clat = c[0], c[1]
        d = dist_km(lat, lon, clat, clon)
        if d <= radius_km:
            p = f['properties']
            if p.get('az_current') is not None:
                out.append({'id': p['id'], 'lat': clat, 'lon': clon,
                            'az': p['az_current'], 'fov': p.get('fov') or 62.8,
                            'dist_km': d})
    return sorted(out, key=lambda x: x['dist_km'])


def latest_frame_url(cam_id):
    # AlertCalifornia public still-image endpoint pattern
    return f'https://cameras.alertcalifornia.org/public-camera-data/{cam_id}/latest-frame.jpg'


def plume_base_extent(mask, conf):
    """From a binary mask (H,W) return (x_left_frac, x_center_frac, x_right_frac)
    of the plume BASE -- the lowest 15% band of mask rows."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    y_lo = ys.max()
    band = ys >= (y_lo - 0.15 * (y_lo - ys.min()) - 1)
    bx = xs[band]
    if len(bx) == 0:
        bx = xs
    return bx.min() / W, np.median(bx) / W, bx.max() / W


def upwind_source_frac(xl, xc, xr, cam_az, wind_from_deg):
    """Pick the plume-base edge nearest the SOURCE using wind geometry.

    Smoke drifts TOWARD (wind_from + 180). Projected onto the camera's
    horizontal image axis (image-x increases with bearing, i.e. to the right),
    a drift toward bearing D shifts the plume tail by sign sin(D - cam_az):
      >0  tail moves right  -> source is the LEFT edge  (use xl)
      <0  tail moves left   -> source is the RIGHT edge (use xr)
    Near-along-view drift (|sin| small) is ambiguous; fall back to the centre.
    """
    drift = (wind_from_deg + 180.0) % 360.0
    s = math.sin(math.radians(drift - cam_az))
    if abs(s) < 0.15:
        return xc, 'centre(along-view drift)'
    return (xl, 'left(upwind)') if s > 0 else (xr, 'right(upwind)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', type=float, default=39.2385)
    ap.add_argument('--lon', type=float, default=-123.012)
    ap.add_argument('--radius', type=float, default=40.0)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--max-cams', type=int, default=40)
    a = ap.parse_args()

    print(f'fetching camera list...', flush=True)
    cams_json = requests.get(AC_LIST, timeout=30, headers={'User-Agent': 'x'}).json()
    cams = cameras_near(a.lat, a.lon, a.radius, cams_json)[:a.max_cams]
    print(f'{len(cams)} cameras with azimuth within {a.radius} km', flush=True)

    # wind, for upwind-edge bearing extraction (Open-Meteo; the codebase's
    # documented fallback since NOMADS OpenDAP was retired)
    try:
        wj = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': a.lat, 'longitude': a.lon,
            'current': 'wind_direction_10m', 'wind_speed_unit': 'mph'},
            timeout=20).json()
        wind_from = float(wj['current']['wind_direction_10m'])
        print(f'wind FROM {wind_from:.0f} deg (smoke drifts toward '
              f'{(wind_from+180)%360:.0f})\n', flush=True)
    except Exception:
        wind_from = None
        print('wind unavailable -> using plume-base centre (no upwind correction)\n',
              flush=True)

    model = YOLO(MODEL)
    rays = []
    detections = []
    for cam in cams:
        try:
            r = requests.get(latest_frame_url(cam['id']), timeout=15,
                             headers={'User-Agent': 'x'})
            if r.status_code != 200 or len(r.content) < 2000:
                continue
            import cv2
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue
        res = model.predict(img, conf=a.conf, verbose=False, retina_masks=True)[0]
        if res.masks is None or len(res.masks) == 0:
            continue
        confs = res.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        m = res.masks.data[best].cpu().numpy() > 0.5
        ext = plume_base_extent(m, confs[best])
        if ext is None:
            continue
        xl, xc, xr = ext
        if wind_from is not None:
            src_frac, which = upwind_source_frac(xl, xc, xr, cam['az'], wind_from)
        else:
            src_frac, which = xc, 'centre'
        b_src = pixel_bearings(src_frac, src_frac, cam['az'], cam['fov'])[1]
        rays.append((cam['lat'], cam['lon'], b_src))
        detections.append((cam['id'], cam['dist_km'], float(confs[best]), b_src))
        print(f'  {cam["id"]:26s} {cam["dist_km"]:5.1f}km  conf={confs[best]:.2f}  '
              f'source-bearing {b_src:5.1f} deg  [{which}]', flush=True)

    print(f'\n{len(rays)} cameras with a confident smoke detection')
    if len(rays) < 2:
        print('need >=2 cameras to triangulate; with 1, only a bearing is available.')
        return
    fix = triangulate(rays, inlier_km=2.0)
    if fix is None:
        print('bearings did not intersect (fire may be outside camera sightlines).')
        return
    print(f'\nTRIANGULATED FIRE LOCATION: ({fix["lat"]:.4f}, {fix["lon"]:.4f})')
    print(f'  consensus={fix["consensus"]}  inliers {fix["n_inliers"]}/{fix["n_rays"]}  '
          f'spread {fix["spread_km"]*1000:.0f} m')
    kept = [detections[k][0] for k in fix['inlier_idx']]
    print(f'  inlier cameras: {kept}')
    d = dist_km(a.lat, a.lon, fix['lat'], fix['lon'])
    print(f'  {d:.1f} km from the query point ({a.lat},{a.lon})')


if __name__ == '__main__':
    main()
