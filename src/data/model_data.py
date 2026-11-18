"""Load frozen preprocessing outputs for model training; no fitted preprocessing.

M2 must fit imputation/scaling within trip-group CV folds, never on student_test.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.schema import SCHEMA_VERSION,FEATURES

def load_model_data(processed_root='data/processed', partition='train'):
    """Return (X, y, groups, keys), preserving candidate order.

    Only confident derived labels are retained. X keeps explicit NaNs and exact
    schema feature order; y is int8; groups are eight-character trip strings.
    keys contains trip_id, obs_idx, arc_id, cand_rank for scoring/join diagnostics.
    This convenience loader materializes the selected partition in memory.
    """
    if partition not in ('train','student_test'):
        raise ValueError('Only permitted development partitions may be loaded')
    root = Path(processed_root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['schema_version']!=SCHEMA_VERSION:
        raise ValueError('Dataset/schema mismatch; regenerate or use matching code')
    candidates = pd.read_parquet(root/f'{partition}_candidates.parquet')
    trusted = candidates.loc[candidates.label.notna()].reset_index(drop=True)
    X = trusted[FEATURES].copy()
    y = trusted.label.astype('int8')
    groups = trusted.trip_id.copy()
    keys = trusted[['trip_id','obs_idx','arc_id','cand_rank']].copy()
    if not len(X) or set(y.unique())!={0,1}:
        raise ValueError('No usable two-class supervised data in selected partition')
    return X,y,groups,keys
