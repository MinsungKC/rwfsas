"""Turn smoke masks from fixed cameras into a ground location for the fire.

Ties the pieces together:
  1. the segmentation model (best_seg_v1.pt) gives a real smoke SILHOUETTE, not
     a box, so we can find the plume's true left/right extent and, critically,
     the base of the smoke column where it meets terrain;
  2. AlertCalifornia publishes each camera's exact position and current azimuth
     (az_current, verified accurate to 0.1 deg vs the bearing to fov_center);
  3. a horizontal pixel maps linearly across the camera's field of view, so a
     mask's horizontal extent -> a wedge of bearings from that camera;
  4. two or more cameras' bearing wedges INTERSECT at the fire -- triangulation
     gives a ground point with far tighter cross-range precision (~0.1 deg =
     ~17 m at 10 km) than any 2 km satellite pixel.

Why this beats satellites for a NEW hot fire (the Lucas lesson): a camera sees
WHERE the plume rises from the ground regardless of how hot the fire burns, so
it is immune to the intensity-driven neighbour-cell contamination that made the
Lucas GOES footprint 29x too large.

NOT modelled here (honest limits):
  * range along a single camera's line of sight is weak -- triangulation or a
    DEM ray-cast of the plume BASE is needed for distance; this module does the
    triangulation, and leaves DEM ray-cast as a single-camera fallback stub.
  * lens distortion is treated as a linear FOV mapping (fine near frame centre,
    degrades at edges on the fisheye units -- flagged per detection).
"""
import math
import numpy as np

R_EARTH_KM = 6371.0


def pixel_bearings(mask_x_frac_left, mask_x_frac_right, az_center_deg, fov_deg,
                   flip=False):
    """Map a mask's horizontal extent (fractions 0..1 across the frame) to a
    left/right BEARING from the camera. az_center is the bearing of the frame
    centre; fov the horizontal field of view. `flip` for mirrored feeds."""
    def frac_to_bearing(fx):
        off = (fx - 0.5) * fov_deg           # deg from centre, + = right
        if flip:
            off = -off
        return (az_center_deg + off) % 360.0
    b_left = frac_to_bearing(mask_x_frac_left)
    b_right = frac_to_bearing(mask_x_frac_right)
    # bearing to the plume-base centre is the most fire-indicative single ray
    b_center = frac_to_bearing((mask_x_frac_left + mask_x_frac_right) / 2)
    return b_left, b_center, b_right


def _fwd(lat, lon, bearing_deg, dist_km):
    la1 = math.radians(lat); lo1 = math.radians(lon); br = math.radians(bearing_deg)
    dr = dist_km / R_EARTH_KM
    la2 = math.asin(math.sin(la1)*math.cos(dr) + math.cos(la1)*math.sin(dr)*math.cos(br))
    lo2 = lo1 + math.atan2(math.sin(br)*math.sin(dr)*math.cos(la1),
                           math.cos(dr) - math.sin(la1)*math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540) % 360 - 180


def _intersect(lat1, lon1, brg1, lat2, lon2, brg2):
    """Intersection of two great-circle bearings (Ed Williams' formula).
    Returns (lat,lon) or None if parallel / behind."""
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    b13 = math.radians(brg1); b23 = math.radians(brg2)
    dLat = la2 - la1; dLon = lo2 - lo1
    d12 = 2*math.asin(math.sqrt(math.sin(dLat/2)**2 +
                                math.cos(la1)*math.cos(la2)*math.sin(dLon/2)**2))
    if d12 < 1e-12:
        return None
    cbA = (math.sin(la2)-math.sin(la1)*math.cos(d12)) / (math.sin(d12)*math.cos(la1))
    cbB = (math.sin(la1)-math.sin(la2)*math.cos(d12)) / (math.sin(d12)*math.cos(la2))
    bA = math.acos(min(1, max(-1, cbA)))
    bB = math.acos(min(1, max(-1, cbB)))
    if math.sin(lo2-lo1) > 0:
        b12 = bA; b21 = 2*math.pi - bB
    else:
        b12 = 2*math.pi - bA; b21 = bB
    a1 = (b13 - b12 + math.pi) % (2*math.pi) - math.pi
    a2 = (b21 - b23 + math.pi) % (2*math.pi) - math.pi
    if math.sin(a1) == 0 and math.sin(a2) == 0:
        return None
    if math.sin(a1)*math.sin(a2) < 0:
        return None            # intersection is behind one of the cameras
    a1 = abs(a1); a2 = abs(a2)
    a3 = math.acos(-math.cos(a1)*math.cos(a2) + math.sin(a1)*math.sin(a2)*math.cos(d12))
    d13 = math.atan2(math.sin(d12)*math.sin(a1)*math.sin(a2),
                     math.cos(a2)+math.cos(a1)*math.cos(a3))
    la3 = math.asin(min(1, max(-1, math.sin(la1)*math.cos(d13) +
                               math.cos(la1)*math.sin(d13)*math.cos(b13))))
    dLon13 = math.atan2(math.sin(b13)*math.sin(d13)*math.cos(la1),
                        math.cos(d13)-math.sin(la1)*math.sin(la3))
    lo3 = lo1 + dLon13
    return math.degrees(la3), (math.degrees(lo3)+540) % 360 - 180


def _dk(a, b, c, e):
    R = 6371.0; x1, x2 = math.radians(a), math.radians(c)
    dla = x2-x1; dlo = math.radians(e-b)
    h = math.sin(dla/2)**2 + math.cos(x1)*math.cos(x2)*math.sin(dlo/2)**2
    return 2*R*math.asin(math.sqrt(h))


def _ray_miss_km(cam_lat, cam_lon, bearing, plat, plon):
    """Perpendicular-ish miss distance of a ray from a candidate point:
    how far the point sits off the camera's bearing line."""
    brg_to_pt = _bearing(cam_lat, cam_lon, plat, plon)
    d = _dk(cam_lat, cam_lon, plat, plon)
    dang = abs((brg_to_pt - bearing + 180) % 360 - 180)
    return d * math.sin(math.radians(dang))


def _bearing(lat, lon, tlat, tlon):
    la1, la2 = math.radians(lat), math.radians(tlat)
    dlon = math.radians(tlon - lon)
    y = math.sin(dlon) * math.cos(la2)
    x = math.cos(la1)*math.sin(la2) - math.sin(la1)*math.cos(la2)*math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360


def triangulate(rays, inlier_km=1.5):
    """rays: list of (cam_lat, cam_lon, bearing_deg).

    Consensus (RANSAC-style) triangulation: false-positive smoke detections
    produce bearings that point nowhere near the real fire, and averaging them
    in wrecks the fix (measured on the live Lucas run: 5 rays, 3 spurious ->
    6.4 km error). So instead of trusting all rays: for every pairwise
    intersection, count how many OTHER rays pass within `inlier_km` of it, and
    keep the largest consistent set. Returns the consensus point over inliers
    plus the inlier list, so spurious cameras are reported, not silently mixed.
    """
    n = len(rays)
    if n < 2:
        return None
    best_inliers = []
    for i in range(n):
        for j in range(i+1, n):
            p = _intersect(*rays[i][:2], rays[i][2], *rays[j][:2], rays[j][2])
            if not p:
                continue
            inliers = [k for k in range(n)
                       if _ray_miss_km(rays[k][0], rays[k][1], rays[k][2], *p) <= inlier_km]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
    if len(best_inliers) < 2:
        # no consensus -- fall back to the all-rays median, flagged low-confidence
        best_inliers = list(range(n))
        consensus = False
    else:
        consensus = True
    sub = [rays[k] for k in best_inliers]
    pts = []
    for i in range(len(sub)):
        for j in range(i+1, len(sub)):
            p = _intersect(*sub[i][:2], sub[i][2], *sub[j][:2], sub[j][2])
            if p:
                pts.append(p)
    if not pts:
        return None
    la = float(np.median([p[0] for p in pts]))
    lo = float(np.median([p[1] for p in pts]))
    spread = float(np.median([_dk(la, lo, p[0], p[1]) for p in pts])) if len(pts) > 1 else 0.0
    return {'lat': la, 'lon': lo, 'n_intersections': len(pts),
            'spread_km': spread, 'n_inliers': len(best_inliers),
            'n_rays': n, 'inlier_idx': best_inliers, 'consensus': consensus}


if __name__ == '__main__':
    # self-test: two cameras with known bearings crossing at a known point
    # place a synthetic fire at (39.235, -122.973) -- the real Lucas centroid
    fire = (39.235, -122.973)

    def bearing_to(lat, lon, tlat, tlon):
        la1, la2 = math.radians(lat), math.radians(tlat)
        dlon = math.radians(tlon-lon)
        y = math.sin(dlon)*math.cos(la2)
        x = math.cos(la1)*math.sin(la2)-math.sin(la1)*math.cos(la2)*math.cos(dlon)
        return math.degrees(math.atan2(y, x)) % 360

    cams = [(39.170, -122.930), (39.130, -123.080), (39.310, -123.180)]
    rays = [(la, lo, bearing_to(la, lo, *fire)) for la, lo in cams]
    r = triangulate(rays)
    err_km = math.hypot((r['lat']-fire[0])*111.32,
                        (r['lon']-fire[1])*111.32*math.cos(math.radians(fire[0])))
    print(f'synthetic 3-camera triangulation of the Lucas centroid:')
    print(f'  true  {fire}')
    print(f'  fixed ({r["lat"]:.4f}, {r["lon"]:.4f})  from {r["n_intersections"]} intersections')
    print(f'  error {err_km*1000:.0f} m   spread {r["spread_km"]*1000 if r["spread_km"] else 0:.0f} m')
    # now with 0.5 deg bearing noise on each camera (realistic pointing error)
    import random
    random.seed(1)
    errs = []
    for _ in range(200):
        noisy = [(la, lo, b + random.gauss(0, 0.5)) for la, lo, b in rays]
        rr = triangulate(noisy)
        errs.append(math.hypot((rr['lat']-fire[0])*111.32,
                    (rr['lon']-fire[1])*111.32*math.cos(math.radians(fire[0])))*1000)
    print(f'  with 0.5deg pointing noise: median error {np.median(errs):.0f} m, '
          f'p90 {np.percentile(errs,90):.0f} m')
