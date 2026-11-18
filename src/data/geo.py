"""WGS84 lon/lat degrees -> per-trip azimuthal equidistant x/y metres.

Projection is network-derived and never reference-derived. Circular longitude mean
handles the dateline; reject networks >1500 km from centre rather than silently
using a distorted local system. Bearings are geodesic, clockwise from north.
"""
from dataclasses import dataclass
import numpy as np
from pyproj import CRS, Transformer, Geod

GEOD = Geod(ellps='WGS84')

def valid_lonlat(a):
    a = np.asarray(a, dtype=float)
    return np.isfinite(a).all(axis=-1) & (np.abs(a[..., 0]) <= 180) & (np.abs(a[..., 1]) <= 90)

def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2-p1
    dl = np.radians(np.asarray(lon2)-np.asarray(lon1))
    h = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def bearing_deg(lon1, lat1, lon2, lat2):
    az, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return np.where(np.asarray(distance) > 1e-6, np.asarray(az) % 360, np.nan)

def angular_difference(a, b):
    return np.abs((np.asarray(a)-np.asarray(b)+180) % 360-180)

@dataclass
class TripProjection:
    lon0: float
    lat0: float

    def __post_init__(self):
        self.crs = CRS.from_proj4(f'+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m')
        self.forward = Transformer.from_crs(4326, self.crs, always_xy=True)
        self.backward = Transformer.from_crs(self.crs, 4326, always_xy=True)

    @classmethod
    def from_lonlat(cls, points):
        a = np.asarray(points)
        if not len(a) or not valid_lonlat(a).all():
            raise ValueError('Projection needs a nonempty valid network (invalid nodes cannot be renumbered)')
        lon0 = np.degrees(np.arctan2(np.sin(np.radians(a[:,0])).mean(), np.cos(np.radians(a[:,0])).mean()))
        obj = cls(float(lon0), float(a[:,1].mean()))
        if np.linalg.norm(obj.project(a), axis=1).max() > 1_500_000:
            raise ValueError('Network exceeds supported local projection extent (1500 km radius)')
        return obj

    def project(self, points):
        a = np.asarray(points, dtype=float).reshape(-1, 2)
        result = np.full(a.shape, np.nan)
        mask = valid_lonlat(a)
        result[mask] = np.column_stack(self.forward.transform(a[mask,0].tolist(), a[mask,1].tolist()))
        return result

    def inverse(self, points):
        a = np.asarray(points).reshape(-1, 2)
        return np.column_stack(self.backward.transform(a[:,0].tolist(), a[:,1].tolist()))
