"""Generate a regular grid over the water area."""

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep


# Approximate meters per degree at ~60°N latitude
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LON = 55_660.0  # 111320 * cos(60°)


def generate_grid(
    water_polygon,
    cell_size_m: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate grid points that lie within the water polygon.

    Parameters
    ----------
    water_polygon : shapely geometry
        Union of all water zone polygons.
    cell_size_m : float
        Grid cell size in meters.

    Returns
    -------
    lats : ndarray of shape (N,)
        Latitudes of water grid cells.
    lons : ndarray of shape (N,)
        Longitudes of water grid cells.
    dlat : float
        Grid step in latitude degrees.
    dlon : float
        Grid step in longitude degrees.
    """
    dlat = cell_size_m / METERS_PER_DEG_LAT
    dlon = cell_size_m / METERS_PER_DEG_LON

    minx, miny, maxx, maxy = water_polygon.bounds
    # Extend slightly to catch boundary cells
    lat_range = np.arange(miny, maxy + dlat, dlat)
    lon_range = np.arange(minx, maxx + dlon, dlon)

    # Use prepared geometry for fast containment checks
    prepared = prep(water_polygon)

    # Vectorized point generation
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    # Check containment for all points
    mask = np.array(
        [prepared.contains(Point(lon, lat)) for lon, lat in zip(lon_flat, lat_flat)],
        dtype=bool,
    )

    return lat_flat[mask], lon_flat[mask], dlat, dlon


def snap_to_grid(
    lat: float,
    lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> int:
    """Find the nearest grid cell index for a given coordinate.

    Uses approximate Euclidean distance weighted for latitude.
    """
    dlat = grid_lats - lat
    dlon = (grid_lons - lon) * (METERS_PER_DEG_LON / METERS_PER_DEG_LAT)
    dist_sq = dlat**2 + dlon**2
    return int(np.argmin(dist_sq))
