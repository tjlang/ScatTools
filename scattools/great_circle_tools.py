import numpy as np
from math import atan2 as atan2

"""
Timothy James Lang
timothy.j.lang@nasa.gov

Major Changes:
07/11/2015: Made code pep8 compliant
"""

def gc_latlon_bear_dist(lat1, lon1, bear, dist):
    """
    Input lat1/lon1 as decimal degrees, as well as bearing and distance from
    the coordinate. Returns lat2/lon2 of final destination. Cannot be
    vectorized due to atan2.
    """
    re = 6371.1  # km
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    bearr = np.deg2rad(bear)
    lat2r = np.arcsin((np.sin(lat1r) * np.cos(dist/re)) +
                      (np.cos(lat1r) * np.sin(dist/re) * np.cos(bearr)))
    lon2r = lon1r + atan2(np.sin(bearr) * np.sin(dist/re) *
                          np.cos(lat1r), np.cos(dist/re) - np.sin(lat1r) *
                          np.sin(lat2r))
    return np.rad2deg(lat2r), np.rad2deg(lon2r)


def gc_bear(lat1, lon1, lat2, lon2):
    """
    Input lat1/lon1 and lat2/lon2 as decimal degrees.
    Returns initial bearing (deg) from lat1/lon1 to lat2/lon2.
    Cannot be vectorized due to atan2
    """
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    thetar = atan2(np.sin(lon2r-lon1r) * np.cos(lat2r),
                   np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) *
                   np.cos(lat2r) * np.cos(lon2r-lon1r))
    theta = np.rad2deg(thetar)
    return (theta + 360.0) % 360.0


def gc_dist(lat1, lon1, lat2, lon2):
    """
    Input lat1/lon1 and lat2/lon2 as decimal degrees.
    Returns great circle distance in km. Can run in vectorized form!
    """
    re = np.float64(6371.1)  # km
    lat1r = lat1 * np.pi / np.float64(180.0)
    lon1r = lon1 * np.pi / np.float64(180.0)
    lat2r = lat2 * np.pi / np.float64(180.0)
    lon2r = lon2 * np.pi / np.float64(180.0)
    dist = np.arccos(np.sin(lat1r) * np.sin(lat2r) + np.cos(lat1r) *
                     np.cos(lat2r) * np.cos(lon2r-lon1r))*re
    return dist
