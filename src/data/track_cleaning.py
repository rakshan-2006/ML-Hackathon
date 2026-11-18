"""Keep all raw identities; no smoothing or fitted transformations."""
import numpy as np
from src.common.schema import C1, typed_table
from .geo import valid_lonlat, bearing_deg

def clean_track(track, projection, trip_id, partition='train', config=None):
    cfg = config or {}
    a = track[['lon','lat']].to_numpy()
    xy = projection.project(a)
    t = track.t.to_numpy()
    n = len(track)
    coord = valid_lonlat(a)
    timestamp = np.isfinite(t)
    valid = coord & timestamp
    dt = np.r_[np.nan,np.diff(t)]
    gap = dt > cfg.get('gap_threshold_s',30)
    nonpositive = np.isfinite(dt) & (dt <= 0)
    step = np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)]
    eligible = valid & np.r_[False,valid[:-1]] & (dt>0) & ~gap
    speed = np.full(n,np.nan)
    speed[eligible] = step[eligible]/dt[eligible]
    heading = np.full(n,np.nan)
    if n>1:
        heading[1:] = bearing_deg(a[:-1,0],a[:-1,1],a[1:,0],a[1:,1])
    heading[~eligible | (step < cfg.get('stationary_m',1.0))] = np.nan
    return typed_table(dict(trip_id=[str(trip_id)]*n,partition=[partition]*n,obs_idx=np.arange(n),t=t,lon=a[:,0],lat=a[:,1],x=xy[:,0],y=xy[:,1],dt_prev=dt,speed_mps=speed,heading_deg=heading,is_valid=valid,gap_flag=gap,n_candidates=np.zeros(n,dtype=int),step_dist_m=step,coordinate_valid=coord,timestamp_valid=timestamp,nonpositive_dt=nonpositive,suspicious_speed=speed>cfg.get('suspicious_speed_mps',70)),C1)
