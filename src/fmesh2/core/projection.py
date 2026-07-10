"""Spherical stereographic projection (pure numpy, R in km).

Conformal, so point-in-polygon and nearest-point queries done in the plane
are topologically exact; metric distances are measured back on the sphere.
Singular at the antipode of the projection center — callers keep polygons
well inside one hemisphere around the center.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def ring_center(lon_deg, lat_deg) -> tuple[float, float]:
    """Lon/lat of the normalized mean of the points' unit-sphere positions."""
    lon = np.deg2rad(np.asarray(lon_deg, float))
    lat = np.deg2rad(np.asarray(lat_deg, float))
    x = (np.cos(lat) * np.cos(lon)).mean()
    y = (np.cos(lat) * np.sin(lon)).mean()
    z = np.sin(lat).mean()
    norm = np.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:  # degenerate (e.g. points evenly covering a great circle)
        return 0.0, 90.0 if z >= 0 else -90.0
    return float(np.rad2deg(np.arctan2(y, x))), float(np.rad2deg(np.arcsin(z / norm)))


def stereo_xy(lon_deg, lat_deg, lon0: float, lat0: float, radius: float = EARTH_RADIUS_KM):
    """Project to the stereographic plane centered at (lon0, lat0); km."""
    lam = np.deg2rad(np.asarray(lon_deg, float))
    phi = np.deg2rad(np.asarray(lat_deg, float))
    lam0, phi0 = np.deg2rad(lon0), np.deg2rad(lat0)
    cosc = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(lam - lam0)
    k = 2.0 * radius / np.maximum(1.0 + cosc, 1e-12)
    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lam - lam0))
    return x, y


def stereo_inverse(x, y, lon0: float, lat0: float, radius: float = EARTH_RADIUS_KM):
    """Inverse of `stereo_xy`, returning (lon, lat) degrees."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lam0, phi0 = np.deg2rad(lon0), np.deg2rad(lat0)
    rho = np.hypot(x, y)
    c = 2.0 * np.arctan2(rho, 2.0 * radius)
    safe_rho = np.where(rho == 0.0, 1.0, rho)
    phi = np.arcsin(np.cos(c) * np.sin(phi0) + y * np.sin(c) * np.cos(phi0) / safe_rho)
    lam = lam0 + np.arctan2(
        x * np.sin(c), rho * np.cos(phi0) * np.cos(c) - y * np.sin(phi0) * np.sin(c)
    )
    lon = np.rad2deg(np.where(rho == 0.0, lam0, lam))
    lat = np.rad2deg(np.where(rho == 0.0, phi0, phi))
    return (lon + 180.0) % 360.0 - 180.0, lat


def haversine_km(lon1, lat1, lon2, lat2, radius: float = EARTH_RADIUS_KM):
    """Great-circle distance between point arrays, km."""
    lam1, phi1 = np.deg2rad(np.asarray(lon1, float)), np.deg2rad(np.asarray(lat1, float))
    lam2, phi2 = np.deg2rad(np.asarray(lon2, float)), np.deg2rad(np.asarray(lat2, float))
    a = (
        np.sin((phi2 - phi1) / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
