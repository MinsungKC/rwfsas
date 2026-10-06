"""Causal, target-free inference core for the transferred learned-field v3.

This is deliberately a new implementation, not a loader for the legacy
``field_probs3.pkl`` cache.  At inference time the only accepted transition
state is a known base perimeter, a decision window, and frozen detections.
Target geometry, target acres, growth ratio, and stratum are never read.

The feature schema and classifier settings match the documented legacy v3
candidate.  ``push_*`` is a causal detection-derived frontier proxy; it is not
the old champion prediction cache.  A caller may supply a versioned DEM sampler
to enable the terrain features; the default explicitly emits zero terrain.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy, points, wkb
from shapely.geometry import box
from shapely.ops import transform, unary_union


GRID_KM = 0.25
MAX_CELLS = 30_000
NEGATIVE_RATIO = 6.0
CLASSIFIER_KWARGS = dict(max_iter=220, max_depth=7, learning_rate=0.08,
                         min_samples_leaf=60, l2_regularization=1.0,
                         random_state=0)

FEATS = (
    "d_near", "frp_near", "t_near", "d_edge", "d_edge_rel", "d_near_rel",
    "n1", "frpmax1", "frpsum1", "tmax1", "viirs1",
    "n2", "frpmax2", "frpsum2", "tmax2", "viirs2",
    "n3", "frpmax3", "frpsum3", "tmax3", "viirs3",
    "n5", "frpmax5", "frpsum5", "tmax5", "viirs5",
    "base_r", "log_base", "log_win", "log_ndet", "frp_med", "frp_p90",
    "frp_near_rel", "frpmax2_rel", "grid_km",
    "d_edge_minus_det", "toward_base", "toward_base_norm",
)
FEATS2 = FEATS + ("surround2", "surround4", "push_here", "in_champ", "push_rel",
                  "aniso_axis", "aniso_centroid")
FEATURE_NAMES = FEATS2 + (
    "surround05", "surround1", "surround8", "dens_r12", "dens_r25",
    "t_first_near", "t_first2", "elev_rel", "slope_deg", "elev_minus_det",
)
assert len(FEATURE_NAMES) == 55


def parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("timestamp must be an ISO string or datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DemSampler(Protocol):
    """Named, frozen DEM source supplied by the replay/training layer."""
    source_id: str

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ZeroDem:
    """Explicit no-terrain fallback; it is not a hidden network DEM query."""
    source_id: str = "none/zero-dem-v1"

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return np.zeros(len(lon), dtype=np.float32)


@dataclass(frozen=True)
class RejectedObservation:
    index: int
    reason: str


@dataclass(frozen=True)
class FieldGrid:
    fire: str
    step: int
    cutoff_utc: str
    base: Any
    center_lon: float
    center_lat: float
    km_per_lon: float
    grid_km: float
    points_km: np.ndarray
    features: np.ndarray
    rejected: tuple[RejectedObservation, ...]


@dataclass(frozen=True)
class TrainingBlock:
    """A target-labelled block, constructed only in the training adapter."""
    fire: str
    step: int
    grid: FieldGrid
    labels: np.ndarray


def _base(step: Mapping[str, Any]):
    try:
        return wkb.loads(step["base_wkb"], hex=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("step must include a valid known base_wkb") from exc


def _projection(base: Any) -> tuple[float, float, float]:
    c = base.centroid
    lon, lat = float(c.x), float(c.y)
    return lon, lat, 111.320 * max(math.cos(math.radians(lat)), 1e-6)


def _to_km(lon: np.ndarray, lat: np.ndarray, center_lon: float, center_lat: float,
           km_per_lon: float) -> np.ndarray:
    return np.column_stack(((lon - center_lon) * km_per_lon, (lat - center_lat) * 111.320))


def _geometry_to_km(geom: Any, center_lon: float, center_lat: float, km_per_lon: float):
    return transform(lambda x, y: ((np.asarray(x) - center_lon) * km_per_lon,
                                   (np.asarray(y) - center_lat) * 111.320), geom)


def _from_km(geom: Any, center_lon: float, center_lat: float, km_per_lon: float):
    return transform(lambda x, y: (center_lon + np.asarray(x) / km_per_lon,
                                   center_lat + np.asarray(y) / 111.320), geom)


def cutoff_detections(step: Mapping[str, Any], cutoff_utc: str | datetime) -> tuple[list[dict[str, Any]], tuple[RejectedObservation, ...]]:
    """Validate and retain detections no later than the explicit cutoff.

    Missing/invalid timestamps are rejected instead of silently assigned a
    convenient time.  This is an acquisition-time proxy: receipt is not
    inferred from these frozen observations.
    """
    cutoff = parse_utc(cutoff_utc)
    kept: list[dict[str, Any]] = []
    rejected: list[RejectedObservation] = []
    for i, raw in enumerate(step.get("dets", ())):
        try:
            acquired = parse_utc(raw["acq"])
            lon, lat = float(raw["lon"]), float(raw["lat"])
            frp = float(raw.get("frp", 0.0))
            if not (-180 <= lon <= 180 and -90 <= lat <= 90 and math.isfinite(frp) and frp >= 0):
                raise ValueError("invalid coordinates or FRP")
        except (KeyError, TypeError, ValueError):
            rejected.append(RejectedObservation(i, "missing_or_invalid_acquisition_record"))
            continue
        if acquired > cutoff:
            rejected.append(RejectedObservation(i, "post_cutoff_acquisition"))
            continue
        # Copy only prediction-time observation fields; do not retain arbitrary
        # input payload where target material could travel alongside a detection.
        kept.append({"acq": acquired, "lon": lon, "lat": lat, "frp": frp,
                     "sensor": str(raw.get("sensor", ""))})
    return kept, tuple(rejected)


def _surroundedness(tree: cKDTree, det_xy: np.ndarray, pts: np.ndarray, radius: float) -> np.ndarray:
    indexes = tree.query_ball_point(pts, radius)
    out = np.zeros(len(pts), dtype=float)
    for i, ids in enumerate(indexes):
        if not ids:
            continue
        delta = det_xy[ids] - pts[i]
        norm = np.hypot(delta[:, 0], delta[:, 1])
        valid = norm > 1e-8
        if valid.any():
            out[i] = 1.0 - float(np.hypot(*np.mean(delta[valid] / norm[valid, None], axis=0)))
        else:
            out[i] = 1.0
    return out


def build_prediction_grid(step: Mapping[str, Any], cutoff_utc: str | datetime, *,
                          dem: DemSampler | None = None, grid_km: float = GRID_KM,
                          max_cells: int = MAX_CELLS) -> FieldGrid | None:
    """Construct the 55-column prediction grid without consulting target fields."""
    base = _base(step)
    cutoff = parse_utc(cutoff_utc)
    start = parse_utc(step["start"])
    dets, rejected = cutoff_detections(step, cutoff)
    if not dets:
        return None
    center_lon, center_lat, km_per_lon = _projection(base)
    base_km = _geometry_to_km(base, center_lon, center_lat, km_per_lon)
    lon = np.asarray([d["lon"] for d in dets]); lat = np.asarray([d["lat"] for d in dets])
    det_xy = _to_km(lon, lat, center_lon, center_lat, km_per_lon)
    frp = np.asarray([d["frp"] for d in dets], dtype=float)
    duration = max((cutoff - start).total_seconds(), 1.0)
    times = np.asarray([(d["acq"] - start).total_seconds() / duration for d in dets], dtype=float)
    viirs = np.asarray([float(d["sensor"].upper() == "VIIRS") for d in dets])
    bx0, by0, bx1, by1 = base_km.bounds
    x0, x1 = min(bx0, det_xy[:, 0].min()) - 2.0, max(bx1, det_xy[:, 0].max()) + 2.0
    y0, y1 = min(by0, det_xy[:, 1].min()) - 2.0, max(by1, det_xy[:, 1].max()) + 2.0
    gk = float(grid_km)
    while max(int((x1 - x0) / gk), 1) * max(int((y1 - y0) / gk), 1) > max_cells:
        gk *= 1.4
    xs = x0 + (np.arange(max(int((x1 - x0) / gk), 1)) + .5) * gk
    ys = y0 + (np.arange(max(int((y1 - y0) / gk), 1)) + .5) * gk
    px, py = np.meshgrid(xs, ys)
    px, py = px.ravel(), py.ravel()
    outside = ~contains_xy(base_km, px, py)
    pts = np.column_stack((px[outside], py[outside]))
    if len(pts) < 10:
        return None
    tree = cKDTree(det_xy)
    d1, i1 = tree.query(pts)
    boundary = base_km.boundary
    d_edge = np.asarray(boundary.distance(points(pts)), dtype=float)
    d_edge_det = np.asarray(boundary.distance(points(det_xy)), dtype=float)
    base_r = max(math.sqrt(max(base_km.area, 0.0) / math.pi), .1)
    values: dict[str, np.ndarray] = {
        "d_near": d1, "frp_near": frp[i1], "t_near": times[i1], "d_edge": d_edge,
        "d_edge_rel": d_edge / base_r, "d_near_rel": d1 / base_r,
        "d_edge_minus_det": d_edge - d_edge_det[i1],
    }
    for radius in (1, 2, 3, 5):
        ids = tree.query_ball_point(pts, radius)
        tag = str(radius)
        values[f"n{tag}"] = np.asarray([len(x) for x in ids], dtype=float)
        values[f"frpmax{tag}"] = np.asarray([frp[x].max() if x else 0. for x in ids])
        values[f"frpsum{tag}"] = np.asarray([frp[x].sum() if x else 0. for x in ids])
        values[f"tmax{tag}"] = np.asarray([times[x].max() if x else 0. for x in ids])
        values[f"viirs{tag}"] = np.asarray([viirs[x].sum() if x else 0. for x in ids])
    centroid = base_km.centroid
    q = det_xy[i1]
    toward = np.column_stack((centroid.x - q[:, 0], centroid.y - q[:, 1]))
    outward = pts - q
    tn = np.hypot(toward[:, 0], toward[:, 1]) + 1e-9
    on = np.hypot(outward[:, 0], outward[:, 1]) + 1e-9
    dot = outward[:, 0] * toward[:, 0] + outward[:, 1] * toward[:, 1]
    values["toward_base"] = dot / tn
    values["toward_base_norm"] = dot / (tn * on)
    for radius, name in ((.5, "surround05"), (1., "surround1"), (2., "surround2"),
                         (4., "surround4"), (8., "surround8")):
        values[name] = _surroundedness(tree, det_xy, pts, radius)
    values["dens_r12"] = values["n1"] / (values["n2"] + 1.)
    values["dens_r25"] = values["n2"] / (values["n5"] + 1.)
    ids2 = tree.query_ball_point(pts, 2.)
    values["t_first_near"] = times[i1]
    values["t_first2"] = np.asarray([times[x].min() if x else 1. for x in ids2])
    # Causal frontier proxy: radial 90th-percentile detection reach in each
    # nearest grid angle.  It uses only base + accepted detections.
    rel_det = det_xy - np.array((centroid.x, centroid.y))
    det_angle = np.arctan2(rel_det[:, 1], rel_det[:, 0])
    det_radius = np.hypot(rel_det[:, 0], rel_det[:, 1])
    rel_pts = pts - np.array((centroid.x, centroid.y))
    pt_angle = np.arctan2(rel_pts[:, 1], rel_pts[:, 0])
    bins = np.floor((det_angle + math.pi) / (2 * math.pi) * 72).astype(int) % 72
    reach = np.full(72, base_r)
    for b in range(72):
        available = det_radius[bins == b]
        if len(available): reach[b] = max(base_r, float(np.percentile(available, 90)))
    pt_bins = np.floor((pt_angle + math.pi) / (2 * math.pi) * 72).astype(int) % 72
    push = reach[pt_bins]
    pt_radius = np.hypot(rel_pts[:, 0], rel_pts[:, 1])
    values["push_here"] = push
    values["push_rel"] = push / base_r
    values["in_champ"] = (pt_radius <= push).astype(float)
    if len(rel_det) > 2:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(rel_det.T))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    else:
        axis = np.asarray((1., 0.))
    direction = rel_det.mean(axis=0)
    direction /= np.hypot(*direction) + 1e-9
    pr = pt_radius + 1e-9
    values["aniso_axis"] = np.abs((rel_pts[:, 0] * axis[0] + rel_pts[:, 1] * axis[1]) / pr)
    values["aniso_centroid"] = (rel_pts[:, 0] * direction[0] + rel_pts[:, 1] * direction[1]) / pr
    # ``shapely.points`` is vectorized and returns an ndarray, whereas
    # ``ops.transform`` accepts one geometry.  Coordinate arrays are both
    # faster and unambiguous here.
    point_lon = center_lon + pts[:, 0] / km_per_lon
    point_lat = center_lat + pts[:, 1] / 111.320
    sample = dem or ZeroDem()
    elev = np.asarray(sample.sample(point_lon, point_lat), dtype=float)
    det_elev = np.asarray(sample.sample(lon, lat), dtype=float)
    base_elev = float(sample.sample(np.asarray([center_lon]), np.asarray([center_lat]))[0])
    values["elev_rel"] = elev - base_elev
    values["elev_minus_det"] = elev - det_elev[i1]
    # A portable sampler need not expose a raster gradient.  Set slope to zero
    # unless it intentionally provides this optional method.
    slope = getattr(sample, "slope", None)
    values["slope_deg"] = (np.asarray(slope(point_lon, point_lat), dtype=float)
                           if callable(slope) else np.zeros(len(pts)))
    constants = {"base_r": base_r, "log_base": math.log1p(max(base_km.area * 247.105, 0.0)),
                 "log_win": math.log1p(duration / 3600.), "log_ndet": math.log1p(len(dets)),
                 "frp_med": float(np.median(frp)), "frp_p90": float(np.percentile(frp, 90)), "grid_km": gk}
    for key, value in constants.items(): values[key] = np.full(len(pts), value)
    values["frp_near_rel"] = values["frp_near"] / max(constants["frp_med"], 1e-6)
    values["frpmax2_rel"] = values["frpmax2"] / max(constants["frp_med"], 1e-6)
    matrix = np.column_stack([np.asarray(values[name], dtype=np.float32) for name in FEATURE_NAMES])
    return FieldGrid(str(step.get("fire", "")), int(step.get("step", 0)), cutoff.isoformat().replace("+00:00", "Z"),
                     base, center_lon, center_lat, km_per_lon, gk, pts, matrix, rejected)


def make_training_block(step: Mapping[str, Any], target_wkb_hex: str, cutoff_utc: str | datetime, **kwargs: Any) -> TrainingBlock | None:
    """Label an already target-free grid. Only trainer code should call this."""
    grid = build_prediction_grid(step, cutoff_utc, **kwargs)
    if grid is None:
        return None
    target = wkb.loads(target_wkb_hex, hex=True)
    target_km = _geometry_to_km(target, grid.center_lon, grid.center_lat, grid.km_per_lon)
    labels = contains_xy(target_km, grid.points_km[:, 0], grid.points_km[:, 1]).astype(np.int8)
    return TrainingBlock(grid.fire, grid.step, grid, labels)


def corrected_probability(probability: np.ndarray, retained_negative_fraction: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    r = max(float(retained_negative_fraction), 1e-9)
    return (p * r) / (p * r + 1. - p)


def fit_lofo(blocks: Sequence[TrainingBlock], holdout_fire: str):
    """Fit one deterministic outer-LOFO model; holdout cells cannot enter fit."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    train = [b for b in blocks if b.fire != holdout_fire]
    if not train or any(b.fire == holdout_fire for b in train):
        raise ValueError("LOFO training set is empty or includes the held-out fire")
    rng = np.random.default_rng(0)
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []
    all_negative = kept_negative = 0
    for block in train:
        pos = np.flatnonzero(block.labels == 1); neg = np.flatnonzero(block.labels == 0)
        all_negative += len(neg)
        count = min(len(neg), max(int(len(pos) * NEGATIVE_RATIO), 200))
        if count < len(neg): neg = rng.choice(neg, count, replace=False)
        kept_negative += len(neg)
        select = np.concatenate((pos, neg))
        xs.append(block.grid.features[select]); ys.append(block.labels[select])
    x, y = np.vstack(xs), np.concatenate(ys)
    if len(np.unique(y)) != 2: raise ValueError("LOFO training needs both classes")
    model = HistGradientBoostingClassifier(**CLASSIFIER_KWARGS).fit(x, y)
    return model, kept_negative / max(all_negative, 1)


def predict_probabilities(model: Any, grid: FieldGrid, retained_negative_fraction: float) -> np.ndarray:
    """Prediction accepts a target-free FieldGrid only."""
    return corrected_probability(model.predict_proba(grid.features)[:, 1], retained_negative_fraction)


def geometry_from_probabilities(grid: FieldGrid, probabilities: np.ndarray, *, k: float = 1.0,
                                close_km: float = 0.0):
    """TOPK sum(p)*k selector. ``k`` must be trained outside the held-out fire."""
    p = np.asarray(probabilities, dtype=float)
    take = min(len(p), max(0, int(p.sum() * float(k))))
    if take == 0: return grid.base
    ids = np.argpartition(p, len(p) - take)[-take:]
    cells = [box(x - grid.grid_km / 2, y - grid.grid_km / 2, x + grid.grid_km / 2, y + grid.grid_km / 2)
             for x, y in grid.points_km[ids]]
    result = unary_union(cells)
    if close_km > 0: result = result.buffer(close_km).buffer(-close_km)
    if result.is_empty: return grid.base
    grown = _from_km(result, grid.center_lon, grid.center_lat, grid.km_per_lon)
    return unary_union((grid.base.buffer(0), grown.buffer(0)))
