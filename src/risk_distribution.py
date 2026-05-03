"""Incident risk distributions over the water grid."""

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any, Callable

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import transform

from .grid import METERS_PER_DEG_LAT, METERS_PER_DEG_LON


@dataclass
class GaussianKernel:
    lat: float
    lon: float
    weight: float = 1.0
    sigma_m: float = 1000.0


@dataclass
class AnisotropicGaussianKernel:
    lat: float
    lon: float
    weight: float = 1.0
    sigma_x_m: float = 1000.0
    sigma_y_m: float = 500.0
    angle_deg: float = 0.0


@dataclass
class IncidentDistribution:
    """Discrete approximation of lambda, q, and Q on the app grid."""

    name: str
    lats: np.ndarray
    lons: np.ndarray
    lambda_values: np.ndarray
    q_values: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_intensity(
        cls,
        name: str,
        lats: np.ndarray,
        lons: np.ndarray,
        lambda_values: np.ndarray,
        cell_areas: np.ndarray | None = None,
    ) -> "IncidentDistribution":
        q_values = intensity_to_density(lambda_values, cell_areas=cell_areas)
        weights = density_to_weights(q_values, cell_areas=cell_areas)
        return cls(
            name=name,
            lats=lats,
            lons=lons,
            lambda_values=np.asarray(lambda_values, dtype=np.float64),
            q_values=q_values,
            weights=weights,
        )

    @classmethod
    def from_scenario(
        cls,
        name: str,
        lats: np.ndarray,
        lons: np.ndarray,
        scenarios: dict[str, Any],
        water_polygon=None,
        shoreline=None,
        cell_areas: np.ndarray | None = None,
    ) -> "IncidentDistribution":
        lambda_values = scenario_intensity(
            lats,
            lons,
            name,
            scenarios,
            water_polygon=water_polygon,
            shoreline=shoreline,
            cell_areas=cell_areas,
        )
        return cls.from_intensity(name, lats, lons, lambda_values, cell_areas=cell_areas)

    def sample(self, n: int, rng: int | np.random.Generator | None = None) -> np.ndarray:
        """Sample grid cell indices from Q."""
        generator = _as_generator(rng)
        return generator.choice(len(self.weights), size=n, replace=True, p=self.weights)

    def sample_points(
        self,
        n: int,
        rng: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample incident coordinates as (lats, lons)."""
        indices = self.sample(n, rng=rng)
        return self.lats[indices], self.lons[indices]

    def probability(self, mask: np.ndarray) -> float:
        """Return Q(mask)."""
        return float(self.weights[np.asarray(mask, dtype=bool)].sum())

    def expected_time(self, min_times: np.ndarray, finite_only: bool = False) -> float:
        """Expected response time under Q.

        If ``finite_only`` is false, positive probability of unreachable cells
        makes the expectation infinite.  If it is true, Q is renormalized over
        finite cells before averaging.
        """
        times = np.asarray(min_times, dtype=np.float64)
        if finite_only:
            mask = np.isfinite(times)
            mass = self.weights[mask].sum()
            if mass <= 0:
                return float("inf")
            return float(np.sum(self.weights[mask] * times[mask]) / mass)

        if np.any((~np.isfinite(times)) & (self.weights > 0)):
            return float("inf")
        return float(np.sum(self.weights * times))

    def expected_survival(
        self,
        min_times: np.ndarray,
        survival_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Expected rescue success probability under Q."""
        times = np.asarray(min_times, dtype=np.float64)
        values = np.zeros_like(times, dtype=np.float64)
        mask = np.isfinite(times)
        values[mask] = survival_fn(times[mask])
        return float(np.sum(self.weights * values))


def gaussian_kernel(
    lats: np.ndarray,
    lons: np.ndarray,
    kernel: GaussianKernel,
) -> np.ndarray:
    """Evaluate one isotropic Gaussian risk kernel on the grid."""
    if kernel.sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    dx, dy = coordinate_offsets_m(lats, lons, kernel.lat, kernel.lon)
    dist_sq = dx * dx + dy * dy
    return kernel.weight * np.exp(-dist_sq / (2.0 * kernel.sigma_m**2))


def anisotropic_gaussian_kernel(
    lats: np.ndarray,
    lons: np.ndarray,
    kernel: AnisotropicGaussianKernel,
) -> np.ndarray:
    """Evaluate one rotated anisotropic Gaussian risk kernel on the grid."""
    if kernel.sigma_x_m <= 0 or kernel.sigma_y_m <= 0:
        raise ValueError("sigma_x_m and sigma_y_m must be positive")

    dx, dy = coordinate_offsets_m(lats, lons, kernel.lat, kernel.lon)
    angle = radians(kernel.angle_deg)
    c, s = cos(angle), sin(angle)
    x_rot = c * dx + s * dy
    y_rot = -s * dx + c * dy
    exponent = (x_rot / kernel.sigma_x_m) ** 2 + (y_rot / kernel.sigma_y_m) ** 2
    return kernel.weight * np.exp(-0.5 * exponent)


def gaussian_mixture(
    lats: np.ndarray,
    lons: np.ndarray,
    kernels: list[GaussianKernel],
) -> np.ndarray:
    """Evaluate a sum of isotropic Gaussian kernels."""
    out = np.zeros(len(lats), dtype=np.float64)
    for kernel in kernels:
        out += gaussian_kernel(lats, lons, kernel)
    return out


def anisotropic_gaussian_mixture(
    lats: np.ndarray,
    lons: np.ndarray,
    kernels: list[AnisotropicGaussianKernel],
) -> np.ndarray:
    """Evaluate a sum of anisotropic Gaussian kernels."""
    out = np.zeros(len(lats), dtype=np.float64)
    for kernel in kernels:
        out += anisotropic_gaussian_kernel(lats, lons, kernel)
    return out


def shore_distance_intensity(
    lats: np.ndarray,
    lons: np.ndarray,
    shoreline,
    sigma_m: float,
) -> np.ndarray:
    """Risk component that decays with distance from the real shoreline."""
    if shoreline is None:
        raise ValueError("shoreline is required for shore_distance components")
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")

    distances = distance_to_geometry_m(lats, lons, shoreline)
    return np.exp(-(distances * distances) / (2.0 * sigma_m**2))


def line_distance_mixture(lats: np.ndarray, lons: np.ndarray, lines: list[dict]) -> np.ndarray:
    """Evaluate line-based components with per-line weight and sigma."""
    out = np.zeros(len(lats), dtype=np.float64)
    for line in lines:
        sigma_m = float(line["sigma_m"])
        if sigma_m <= 0:
            raise ValueError("line sigma_m must be positive")

        points = line["points"]
        geometry = LineString([(p["lon"], p["lat"]) for p in points])
        distances = distance_to_geometry_m(lats, lons, geometry)
        out += float(line.get("weight", 1.0)) * np.exp(
            -(distances * distances) / (2.0 * sigma_m**2)
        )
    return out


def coordinate_offsets_m(
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate coordinate offsets from a center in meters."""
    dx = (np.asarray(lons, dtype=np.float64) - center_lon) * METERS_PER_DEG_LON
    dy = (np.asarray(lats, dtype=np.float64) - center_lat) * METERS_PER_DEG_LAT
    return dx, dy


def distance_to_geometry_m(lats: np.ndarray, lons: np.ndarray, geometry) -> np.ndarray:
    """Approximate point-to-geometry distances in meters."""
    geometry_m = transform(_lonlat_to_xy_m, geometry)
    out = np.empty(len(lats), dtype=np.float64)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        out[i] = geometry_m.distance(Point(lon * METERS_PER_DEG_LON, lat * METERS_PER_DEG_LAT))
    return out


def intensity_to_density(
    lambda_values: np.ndarray,
    cell_areas: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize lambda values into grid density q."""
    values = _validated_nonnegative(lambda_values, name="lambda_values")
    if cell_areas is None:
        total = values.sum()
    else:
        areas = _validated_cell_areas(cell_areas, len(values))
        total = np.sum(values * areas)
    if total <= 0:
        raise ValueError("intensity must have positive total mass")
    return values / total


def density_to_weights(
    q_values: np.ndarray,
    cell_areas: np.ndarray | None = None,
) -> np.ndarray:
    """Convert grid density q into discrete probability weights Q({x_j})."""
    q = _validated_nonnegative(q_values, name="q_values")
    if cell_areas is None:
        weights = q.copy()
    else:
        areas = _validated_cell_areas(cell_areas, len(q))
        weights = q * areas

    total = weights.sum()
    if total <= 0:
        raise ValueError("density must have positive total mass")
    return weights / total


def intensity_to_weights(
    lambda_values: np.ndarray,
    cell_areas: np.ndarray | None = None,
) -> np.ndarray:
    """Direct lambda -> Q helper."""
    return density_to_weights(
        intensity_to_density(lambda_values, cell_areas=cell_areas),
        cell_areas=cell_areas,
    )


def scenario_intensity(
    lats: np.ndarray,
    lons: np.ndarray,
    name: str,
    scenarios: dict[str, Any],
    water_polygon=None,
    shoreline=None,
    cell_areas: np.ndarray | None = None,
    _stack: tuple[str, ...] = (),
) -> np.ndarray:
    """Build lambda values for a named configured scenario."""
    if name in _stack:
        cycle = " -> ".join([*_stack, name])
        raise ValueError(f"cyclic scenario reference: {cycle}")
    if name not in scenarios:
        raise KeyError(f"unknown risk scenario: {name}")

    cfg = scenarios[name]
    kind = cfg.get("kind", "component_mixture")
    if kind == "scenario_mixture":
        return _scenario_mixture_intensity(
            lats,
            lons,
            cfg,
            scenarios,
            water_polygon=water_polygon,
            shoreline=shoreline,
            cell_areas=cell_areas,
            stack=(*_stack, name),
        )
    if kind != "component_mixture":
        raise ValueError(f"unsupported scenario kind: {kind}")

    return _component_mixture_intensity(
        lats,
        lons,
        cfg,
        water_polygon=water_polygon,
        shoreline=shoreline,
        cell_areas=cell_areas,
    )


def component_intensity(
    lats: np.ndarray,
    lons: np.ndarray,
    component: dict[str, Any],
    water_polygon=None,
    shoreline=None,
) -> np.ndarray:
    """Build raw lambda values for one configured component."""
    kind = component.get("kind", "gaussian_mixture")
    if kind == "gaussian_mixture":
        kernels = [
            GaussianKernel(
                lat=float(raw["lat"]),
                lon=float(raw["lon"]),
                weight=float(raw.get("weight", 1.0)),
                sigma_m=float(raw.get("sigma_m", component.get("sigma_m", 1000.0))),
            )
            for raw in component.get("kernels", [])
        ]
        return gaussian_mixture(lats, lons, kernels)

    if kind == "anisotropic_gaussian_mixture":
        kernels = [
            AnisotropicGaussianKernel(
                lat=float(raw["lat"]),
                lon=float(raw["lon"]),
                weight=float(raw.get("weight", 1.0)),
                sigma_x_m=float(raw["sigma_x_m"]),
                sigma_y_m=float(raw["sigma_y_m"]),
                angle_deg=float(raw.get("angle_deg", 0.0)),
            )
            for raw in component.get("kernels", [])
        ]
        return anisotropic_gaussian_mixture(lats, lons, kernels)

    if kind == "shore_distance":
        return shore_distance_intensity(
            lats,
            lons,
            shoreline if shoreline is not None else water_polygon.boundary,
            sigma_m=float(component["sigma_m"]),
        )

    if kind == "line_distance":
        return line_distance_mixture(lats, lons, component.get("lines", []))

    if kind == "uniform":
        return np.ones(len(lats), dtype=np.float64)

    raise ValueError(f"unsupported risk component kind: {kind}")


def _component_mixture_intensity(
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: dict[str, Any],
    water_polygon=None,
    shoreline=None,
    cell_areas: np.ndarray | None = None,
) -> np.ndarray:
    normalize_components = bool(cfg.get("normalize_components", True))
    out = np.zeros(len(lats), dtype=np.float64)

    for component in cfg.get("components", []):
        weight = float(component.get("weight", 1.0))
        values = component_intensity(
            lats,
            lons,
            component,
            water_polygon=water_polygon,
            shoreline=shoreline,
        )
        if normalize_components:
            values = intensity_to_density(values, cell_areas=cell_areas)
        out += weight * values

    if np.all(out == 0):
        raise ValueError("risk scenario produced zero intensity")
    return out


def _scenario_mixture_intensity(
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: dict[str, Any],
    scenarios: dict[str, Any],
    water_polygon=None,
    shoreline=None,
    cell_areas: np.ndarray | None = None,
    stack: tuple[str, ...] = (),
) -> np.ndarray:
    out = np.zeros(len(lats), dtype=np.float64)
    for item in cfg.get("scenarios", []):
        weight = float(item.get("weight", 1.0))
        child_name = item["name"]
        child_values = scenario_intensity(
            lats,
            lons,
            child_name,
            scenarios,
            water_polygon=water_polygon,
            shoreline=shoreline,
            cell_areas=cell_areas,
            _stack=stack,
        )
        child_density = intensity_to_density(child_values, cell_areas=cell_areas)
        out += weight * child_density

    if np.all(out == 0):
        raise ValueError("risk scenario mixture produced zero intensity")
    return out


def _validated_nonnegative(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _validated_cell_areas(cell_areas: np.ndarray, n: int) -> np.ndarray:
    areas = np.asarray(cell_areas, dtype=np.float64)
    if areas.shape != (n,):
        raise ValueError("cell_areas must have the same shape as lambda_values")
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("cell_areas must be positive and finite")
    return areas


def _as_generator(rng: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _lonlat_to_xy_m(lon: float, lat: float, z: float | None = None):
    x = lon * METERS_PER_DEG_LON
    y = lat * METERS_PER_DEG_LAT
    if z is None:
        return x, y
    return x, y, z
