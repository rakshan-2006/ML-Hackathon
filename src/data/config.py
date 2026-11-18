"""Route-free data configuration loader shared by offline and inference paths."""
from pathlib import Path
import yaml

def load_config(path='configs/data.yaml'):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    required = ['seed','radius_m','max_candidates','gap_threshold_s','alignment_radius_m']
    if not isinstance(cfg,dict) or any(k not in cfg for k in required):
        raise ValueError(f'Config requires {required}')
    if cfg.get('smoothing','none') != 'none' or cfg.get('projection') != 'network_centered_aeqd_wgs84':
        raise ValueError('Unsupported projection/smoothing policy')
    if cfg['radius_m']<=0 or not isinstance(cfg['max_candidates'],int) or cfg['max_candidates']<1:
        raise ValueError('Positive radius and integer max_candidates required')
    if cfg['gap_threshold_s']<=0 or cfg['alignment_radius_m']<=0:
        raise ValueError('Positive gap and alignment thresholds required')
    return cfg
