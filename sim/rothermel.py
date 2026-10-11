#!/usr/bin/env python3
"""
Rothermel (1972) surface fire spread physics + fire ellipse geometry +
Albini-style spotting distances. Pure math — no I/O, no network.

Fuel parameters are the official Scott & Burgan (2005, RMRS-GTR-153)
constants for the 40 standard fire behavior fuel models. LANDFIRE's FBFM40
raster stores the fuel MODEL code per cell; the per-model parameters are
defined constants of the classification itself (there is no per-cell
parameter raster to fetch), so this table IS the authoritative source.
The old LandFireProductScalar GPServer the spec referenced is defunct
(now a Next.js SPA) — the raster codes come from the LFPS job API instead.

The wind adjustment factor (WAF) blends the standard unsheltered
Baughman & Albini (1980) formula with a sheltered-canopy value using the
LANDFIRE canopy cover (CC) raster.
"""
import math

import numpy as np

# (load_1h t/ac, load_herb t/ac, sav 1/ft, depth ft, Mx_dead %) per S&B code
FUEL40 = {
    101: (0.10, 0.30, 2200, 0.4, 15), 102: (0.10, 1.00, 2000, 1.0, 15),
    103: (0.10, 1.50, 1500, 2.0, 30), 104: (0.25, 1.90, 2000, 2.0, 15),
    105: (0.40, 2.50, 1800, 1.5, 40), 106: (0.10, 3.40, 2200, 1.5, 40),
    107: (1.00, 5.40, 2000, 3.0, 15), 108: (0.50, 7.30, 1500, 4.0, 30),
    109: (1.00, 9.00, 1800, 5.0, 40),
    121: (0.20, 0.50, 2000, 0.9, 15), 122: (0.50, 0.60, 2000, 1.5, 15),
    123: (0.30, 1.45, 1800, 1.8, 40), 124: (1.90, 3.40, 1800, 2.1, 40),
    141: (0.25, 0.15, 2000, 1.0, 15), 142: (1.35, 0.00, 2000, 1.0, 15),
    143: (0.45, 0.00, 1600, 2.4, 40), 144: (0.85, 0.00, 2000, 3.0, 30),
    145: (3.60, 0.00, 750, 6.0, 15),  146: (2.90, 0.00, 750, 2.0, 30),
    147: (3.50, 0.00, 750, 6.0, 15),  148: (2.05, 0.00, 750, 3.0, 40),
    149: (4.50, 1.55, 750, 4.4, 40),
    161: (0.20, 0.20, 2000, 0.6, 20), 162: (0.95, 0.00, 2000, 1.0, 30),
    163: (1.10, 0.65, 1800, 1.3, 30), 164: (4.50, 0.00, 2300, 0.5, 12),
    165: (4.00, 0.00, 1500, 1.0, 25),
    181: (1.00, 0.00, 2000, 0.2, 30), 182: (1.40, 0.00, 2000, 0.2, 25),
    183: (0.50, 0.00, 2000, 0.3, 20), 184: (0.50, 0.00, 2000, 0.4, 25),
    185: (1.15, 0.00, 2000, 0.6, 25), 186: (2.40, 0.00, 2000, 0.3, 25),
    187: (0.30, 0.00, 2000, 0.4, 25), 188: (5.80, 0.00, 1800, 0.3, 35),
    189: (6.65, 0.00, 1800, 0.6, 35),
    201: (1.50, 0.00, 2000, 1.0, 25), 202: (4.50, 0.00, 2000, 1.0, 25),
    203: (5.50, 0.00, 2000, 1.2, 25), 204: (5.25, 0.00, 2000, 2.7, 25),
    # ── conditional fuels (not in Scott & Burgan) ────────────────────────
    # 91 urban/developed: structures as fuel — heavy load, low SAV,
    #    slow-ish spread but HIGH intensity once burning. The engines gate
    #    entry: the wildland fire attacking the WUI edge must exceed
    #    spread.urban.ignition_kw_m before structures ignite (Camp,
    #    Woolsey, Palisades, Eaton all burned deep into towns this way).
    91:  (6.00, 0.00, 1200, 1.2, 20),
    # 93 agriculture: crop/stubble — light flashy fuel
    93:  (1.50, 1.00, 1800, 0.9, 20),
}
URBAN = 91
# truly unburnable regardless of intensity: snow/ice, water, barren rock,
# nodata. Urban (91) and agriculture (93) are conditional — see FUEL40.
NONBURN = frozenset({92, 98, 99, 0, -9999, -32768})

_T_AC_TO_LB_FT2 = 0.0459137
HEAT = 8000.0                      # BTU/lb
FT_MIN_TO_M_MIN = 0.3048


def waf(fuel_depth_ft, canopy_cover_pct):
    """Midflame wind adjustment factor.
    Unsheltered: Baughman & Albini (1980). Sheltered under canopy: blended
    toward ~0.12 by canopy cover fraction (canopy height raster not
    available, so the standard sheltered formula's crown-fill term is
    approximated by cover alone)."""
    d = max(fuel_depth_ft, 0.1)
    unsh = 1.83 / math.log((20.0 + 0.36 * d) / (0.13 * d))
    cc = min(max((canopy_cover_pct or 0.0) / 100.0, 0.0), 1.0)
    if cc < 0.05:
        return unsh
    return unsh * (1.0 - cc) + 0.12 * cc


def rothermel_point(fuel_code, fm1h, wind_mph_10m, wind_dir_to_deg,
                    slope_deg, aspect_deg, canopy_cover_pct=0.0,
                    cured_herb_frac=0.667):
    """
    Full Rothermel spread at one point.
    Returns None for nonburnable fuel, else a dict:
      ros_head_m_min, theta_head_deg (direction of max spread),
      lb (length/breadth), ecc, ros_back_m_min, ros_flank_m_min,
      intensity_kw_m, flame_len_m, ir_btu_ft2_min, r0_m_min
    """
    if fuel_code in NONBURN:
        return None
    params = FUEL40.get(int(fuel_code))
    if params is None:
        return None
    l1, lh, sigma, depth, mx = params
    w0 = (l1 + cured_herb_frac * lh) * _T_AC_TO_LB_FT2
    if w0 <= 0:
        return None

    rho_b = w0 / depth
    beta = rho_b / 32.0
    beta_op = 3.348 * sigma ** -0.8189
    rat = beta / beta_op
    A = 133.0 * sigma ** -0.7913
    gmax = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    gamma = gmax * rat ** A * math.exp(A * (1.0 - rat))

    mf = min(max(fm1h, 0.01), mx / 100.0 * 0.99)
    rm = mf / (mx / 100.0)
    eta_m = max(0.05, 1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    eta_s = 0.42
    wn = w0 * (1.0 - 0.0555)
    ir = gamma * wn * HEAT * eta_m * eta_s            # BTU/ft²/min

    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma)) * (beta + 0.1)) \
         / (192.0 + 0.2595 * sigma)
    eps = math.exp(-138.0 / sigma)
    qig = 250.0 + 1116.0 * mf
    r0 = ir * xi / (rho_b * eps * qig)                # ft/min, no wind/slope

    # wind factor at midflame speed (WAF from canopy + fuel depth)
    u_mid = max(wind_mph_10m, 0.0) * waf(depth, canopy_cover_pct) * 88.0  # ft/min
    # Effective wind speed limit (Rothermel 1972; reaffirmed by Andrews,
    # Cruz & Rothermel 2013, Int. J. Wildland Fire 22:959-969): the wind
    # correlation was fit to wind-tunnel data and has no natural ceiling,
    # so extrapolating it to real gale-force midflame winds produces
    # spread rates the fuel's own energy release can't physically sustain
    # (500+ m/min surface ROS). Cap the wind speed fed into phi_w at the
    # point where its effect saturates relative to the reaction intensity.
    u_mid = min(u_mid, 0.9 * ir)
    C = 7.47 * math.exp(-0.133 * sigma ** 0.55)
    B = 0.02526 * sigma ** 0.54
    E = 0.715 * math.exp(-3.59e-4 * sigma)
    phi_w = C * u_mid ** B * rat ** -E if u_mid > 0 else 0.0

    tan_phi = math.tan(math.radians(max(slope_deg, 0.0)))
    phi_s = 5.275 * beta ** -0.3 * tan_phi ** 2

    # combine wind + slope as vectors -> head direction + effective factor
    wd = math.radians(wind_dir_to_deg)
    up = math.radians((aspect_deg + 180.0) % 360.0)   # upslope direction
    vx = phi_w * math.sin(wd) + phi_s * math.sin(up)
    vy = phi_w * math.cos(wd) + phi_s * math.cos(up)
    phi_e = math.hypot(vx, vy)
    theta_head = math.degrees(math.atan2(vx, vy)) % 360.0 if phi_e > 1e-9 \
        else wind_dir_to_deg

    r_head_ft = r0 * (1.0 + phi_e)

    # effective windspeed (invert phi_w) -> length/breadth -> eccentricity
    if phi_e > 0 and C > 0:
        u_eff_ft = (phi_e * rat ** E / C) ** (1.0 / B)
    else:
        u_eff_ft = 0.0
    u_ms = min(u_eff_ft / 88.0, 60.0) * 0.44704
    lb = min(max(0.936 * math.exp(0.2566 * u_ms)
                 + 0.461 * math.exp(-0.1548 * u_ms) - 0.397, 1.0), 8.0)
    ecc = math.sqrt(1.0 - 1.0 / (lb * lb))

    r_head = r_head_ft * FT_MIN_TO_M_MIN
    r_back = r_head * (1.0 - ecc) / (1.0 + ecc)
    r_flank = r_head * (1.0 - ecc) / math.sqrt(1.0 - ecc * ecc) \
        if ecc < 1.0 else r_head * 0.1

    # Byram fireline intensity + flame length at the head
    tr = 384.0 / sigma                                 # residence time, min
    ib_btu = ir * r_head_ft * tr / 60.0                # BTU/ft/s
    intensity_kw_m = ib_btu * 3.4613
    flame_len_m = 0.45 * max(ib_btu, 0.0) ** 0.46 * 0.3048

    return {
        "ros_head_m_min": r_head, "theta_head_deg": theta_head,
        "lb": lb, "ecc": ecc,
        "ros_back_m_min": r_back, "ros_flank_m_min": r_flank,
        "intensity_kw_m": intensity_kw_m, "flame_len_m": flame_len_m,
        "ir_btu_ft2_min": ir, "r0_m_min": r0 * FT_MIN_TO_M_MIN,
    }


def ellipse_params(res, dt_min):
    """
    Huygens wavelet ellipse for one timestep from a rothermel_point result.
    Fire sits at the rear focus. Returns (a, b, c, theta_deg):
      a = semi-major (m), b = semi-minor (m),
      c = distance from vertex to ellipse CENTER along theta (m).
    """
    fwd = res["ros_head_m_min"] * dt_min
    back = res["ros_back_m_min"] * dt_min
    a = (fwd + back) / 2.0
    b = max(a / res["lb"], 0.5)
    c = a - back            # center sits ahead of the ignition focus
    return a, b, c, res["theta_head_deg"]


def spotting_distance_m(intensity_kw_m, wind_mph_10m):
    """
    Albini-inspired maximum spot-fire distance. Scales with firebrand
    lofting (via fireline intensity) and wind transport, capped at 3 km —
    the regime that matters for wind-driven fires (Camp, Woolsey).
    """
    if intensity_kw_m <= 0 or wind_mph_10m <= 0:
        return 0.0
    u_ms = wind_mph_10m * 0.44704
    d_km = 0.077 * u_ms ** 0.6 * (intensity_kw_m / 1000.0) ** 0.3
    return min(d_km, 3.0) * 1000.0


def fm_10h(fm1h):
    """Standard +1% offset. np.minimum (not builtin min) so this works for
    both scalars and grid arrays."""
    return np.minimum(fm1h + 0.01, 0.40)


def fm_100h(fm1h):
    """Standard +3% offset. np.minimum (not builtin min) so this works for
    both scalars and grid arrays."""
    return np.minimum(fm1h + 0.03, 0.40)
