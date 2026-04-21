"""
Ground-Based Early Warning Radar (GBEWR) system for missile defense footprint calculation.

Computes whether an incoming ballistic missile is detectable by a forward-deployed radar,
and the resulting effective detection range from the ILP (Interceptor Launch Platform) perspective.
"""

import json
import os
import numpy as np
from math import radians, degrees
from geographiclib.geodesic import Geodesic

from fcc_constants import R_e

# Default path for GBEWR radar specs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GBEWR_DATA_PATH = os.path.join(BASE_DIR, "gbewr_data.json")
ROCKET_DATA_PATH = os.path.join(BASE_DIR, "rocket_data.json")

_geod = Geodesic(R_e, 0)


def load_gbewr_data(path=None):
    """Load GBEWR radar specifications from JSON."""
    path = path or GBEWR_DATA_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("gbewr_radars", data) if isinstance(data, dict) else data


def get_radar_spec(radar_key, path=None):
    """Get a single radar specification by key."""
    radars = load_gbewr_data(path)
    for r in radars:
        if r.get("r_key") == radar_key:
            return r
    return None


def _mlp_from_omegashift(ilp_lat, ilp_lon, center_dir_deg, omega_rad, shift_m, mrange_m):
    """
    Compute MLP (lat, lon) from (omega, shift) geometry.
    Uses law of cosines for spherical geometry.
    """
    from math import cos, sin, acos, asin
    f_alfa = mrange_m / R_e
    f_gamma = shift_m / R_e
    angle_eps = 1e-9

    if abs(omega_rad) <= angle_eps:
        fi_deg = 0
    else:
        f_delta = acos(cos(f_gamma) * cos(f_alfa) + sin(f_gamma) * sin(f_alfa) * cos(omega_rad))
        fi = acos((cos(f_alfa) - cos(f_gamma) * cos(f_delta)) / (sin(f_gamma) * sin(f_delta) + 1e-15))
        fi_deg = degrees(fi)
        if omega_rad < 0:
            fi_deg = 360 - fi_deg

    bearing = (center_dir_deg + fi_deg) % 360
    result = _geod.Direct(ilp_lat, ilp_lon, bearing, shift_m)
    return result["lat2"], result["lon2"]


def _trajectory_points_geographic(trj, mlp_lat, mlp_lon, ilp_lat, ilp_lon):
    """
    Convert missile trajectory (time, R, alfa) to geographic (lat, lon) points.
    trj: trajectory from balmis, columns [time, R, alfa]
    MLP is launch point, trajectory flows toward impact (ILP).
    """
    inv = _geod.Inverse(mlp_lat, mlp_lon, ilp_lat, ilp_lon)
    az_mlp_to_ilp = inv["azi1"]

    points = []
    for i in range(len(trj)):
        alfa_rad = trj[i, 2]
        arc_dist_m = alfa_rad * R_e
        direct = _geod.Direct(mlp_lat, mlp_lon, az_mlp_to_ilp, arc_dist_m)
        points.append((direct["lat2"], direct["lon2"]))
    return points


def check_gbewr_detection(ilp_lat, ilp_lon, mlp_lat, mlp_lon, trj, radar_lat, radar_lon, radar_range_km):
    """
    Check if an incoming missile (from MLP toward ILP) is detectable by a GBEWR radar.

    Parameters
    ----------
    ilp_lat, ilp_lon : float
        ILP (Interceptor Launch Platform) position in degrees
    mlp_lat, mlp_lon : float
        MLP (Missile Launch Point) position in degrees
    trj : ndarray
        Missile trajectory from balmis: time, R, alfa (radians)
    radar_lat, radar_lon : float
        radar position in degrees
    radar_range_km : float
        radar detection range in km

    Returns
    -------
    detected : bool
        True if missile enters radar's detection volume
    effective_det_range_m : float
        Distance from ILP to missile when radar first detects it (meters).
        From ILP perspective: this is the effective detection range.
        If not detected, returns 0.
    """
    radar_range_m = radar_range_km * 1000.0
    traj_points = _trajectory_points_geographic(trj, mlp_lat, mlp_lon, ilp_lat, ilp_lon)

    for i, (mis_lat, mis_lon) in enumerate(traj_points):
        inv = _geod.Inverse(radar_lat, radar_lon, mis_lat, mis_lon)
        dist_radar_to_mis = inv["s12"]
        if dist_radar_to_mis <= radar_range_m:
            inv_ilp = _geod.Inverse(ilp_lat, ilp_lon, mis_lat, mis_lon)
            effective_det_range_m = inv_ilp["s12"]
            return True, effective_det_range_m

    return False, 0.0


def effective_det_range_for_trajectory(ilp_lat, ilp_lon, center_dir_deg, omega_rad, shift_m,
                                       trj, radar_lat, radar_lon, radar_range_km):
    """
    Compute effective detection range for a specific trajectory (omega, shift), with GBEWR support.

    Parameters
    ----------
    ilp_lat, ilp_lon : float
        ILP position
    center_dir_deg : float
        ILP center direction (azimuth) in degrees
    omega_rad : float
        Mode2 dihedral angle in radians
    shift_m : float
        ILP-MLP distance in meters
    trj : ndarray
        Missile trajectory
    radar_lat, radar_lon : float
        radar position
    radar_range_km : float
        radar range in km

    Returns
    -------
    effective_det_range_m : float
        Effective det_range to use for this trajectory. If GBEWR detects earlier,
        returns the larger value. Otherwise returns 0 (use base/satellite logic).
    """
    mrange = trj[len(trj) - 1, 2] * R_e
    mlp_lat, mlp_lon = _mlp_from_omegashift(ilp_lat, ilp_lon, center_dir_deg, omega_rad, shift_m, mrange)
    detected, eff_range = check_gbewr_detection(
        ilp_lat, ilp_lon, mlp_lat, mlp_lon, trj,
        radar_lat, radar_lon, radar_range_km
    )
    return eff_range if detected else 0.0


def make_det_range_func(ilp_lat, ilp_lon, center_dir_deg, base_det_range_m, trj,
                         gbewr_list, base_uses_sat_delay=False):
    """
    Create a det_range callback for footprint_mode2 that incorporates GBEWR.

    gbewr_list : list of dicts with keys:
        lat, lon : radar position
        range_km : radar detection range (km)
        r_key : optional radar key for logging

    Returns a function f(omega, shift) -> det_range_m.
    """
    def det_range_func(omega, shift):
        best = base_det_range_m
        if base_uses_sat_delay and best <= 0:
            return 0.0

        for gb in gbewr_list:
            lat = gb.get("lat")
            lon = gb.get("lon")
            range_km = gb.get("range_km", 0)
            if lat is None or lon is None or range_km <= 0:
                continue
            eff = effective_det_range_for_trajectory(
                ilp_lat, ilp_lon, center_dir_deg, omega, shift,
                trj, lat, lon, range_km
            )
            if eff > best:
                best = eff

        return best

    return det_range_func
