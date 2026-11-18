"""Versioned C1/C2/C3 contract; sole ordered model feature definition."""
import numpy as np
import pandas as pd

SCHEMA_VERSION = '1.0.0-m1-provisional'
# (dtype, unit, meaning); NaN floats are explicit unknowns for fold-fitted M2 imputation.
FEATURE_SPEC = {
    'dist_m': ('float64', 'm', 'Exact point-to-directed-segment distance'),
    'cand_rank': ('int64', 'count', 'Distance then arc-ID rank starting at zero'),
    'dist_minus_min': ('float64', 'm', 'Distance above nearest candidate'),
    'heading_diff_deg': ('float64', 'degree', 'Absolute circular directional heading difference'),
    'speed_mps': ('float64', 'm/s', 'Previous valid step speed; unknown at gap/stationary boundary'),
    'arc_len_m': ('float64', 'm', 'Directed segment length'),
    'proj_frac': ('float64', 'fraction', 'Clamped fraction from source to target'),
    'n_candidates': ('int64', 'count', 'Retained candidates for observation'),
    'is_oneway': ('bool', 'boolean', 'No reverse endpoint pair exists'),
    'source_degree': ('int64', 'count', 'Source incoming plus outgoing arc count'),
    'target_degree': ('int64', 'count', 'Target incoming plus outgoing arc count'),
    'prev_dist_m': ('float64', 'm', 'Previous observation distance to this same arc; unknown across gap'),
    'next_dist_m': ('float64', 'm', 'Next observation distance to this same arc; whole-trip contract'),
}
FEATURES = list(FEATURE_SPEC)
C1 = dict(trip_id='string', partition='string', obs_idx='int64', t='float64', lon='float64', lat='float64', x='float64', y='float64', dt_prev='float64', speed_mps='float64', heading_deg='float64', is_valid='bool', gap_flag='bool', n_candidates='int64')
# Additional C1 diagnostics do not become model inputs.
C1.update(step_dist_m='float64', coordinate_valid='bool', timestamp_valid='bool', nonpositive_dt='bool', suspicious_speed='bool')
C2_BASE = dict(trip_id='string', obs_idx='int64', arc_id='int64', cand_rank='int64', dist_m='float64', proj_frac='float64', proj_x='float64', proj_y='float64')
C2 = C2_BASE | {k:v[0] for k,v in FEATURE_SPEC.items()}
C3 = dict(trip_id='string', obs_idx='int64', ref_arc_id='int64', ref_route_pos='int64', align_dist_m='float64', align_status='string', ref_in_candidates='bool')

def typed_table(data, spec):
    df = pd.DataFrame(data)
    return df.reindex(columns=list(spec)).astype(spec)

def validate_tables(obs, cand, alignment, graph, radius, k):
    for df, spec, keys in [(obs,C1,['trip_id','obs_idx']), (cand,C2,['trip_id','obs_idx','arc_id']), (alignment,C3,['trip_id','obs_idx'])]:
        if df.duplicated(keys).any():
            raise ValueError(f'Duplicate keys {keys}')
        for name, dtype in spec.items():
            if str(df[name].dtype) != dtype:
                raise ValueError(f'{name}: expected {dtype}, got {df[name].dtype}')
    if obs.obs_idx.tolist() != list(range(len(obs))) or not obs[['trip_id','obs_idx']].equals(alignment[['trip_id','obs_idx']]):
        raise ValueError('Observation/alignment identity or coverage mismatch')
    if len(cand):
        if not cand.trip_id.isin(obs.trip_id).all() or not cand.obs_idx.isin(obs.obs_idx).all():
            raise ValueError('Candidate foreign key')
        if not cand.arc_id.between(0,len(graph.arc_from)-1).all() or not cand.proj_frac.between(0,1).all() or (cand.dist_m > radius+1e-8).any():
            raise ValueError('Candidate bounds')
        ranks = cand.groupby('obs_idx', sort=False).cumcount().to_numpy()
        if not np.array_equal(ranks,cand.cand_rank) or (cand.cand_rank >= k).any():
            raise ValueError('Candidate ranks')
        if not cand.sort_values(['trip_id','obs_idx','cand_rank']).index.equals(cand.index):
            raise ValueError('Candidate ordering')
        ids = cand.arc_id.to_numpy()
        xy = graph.node_xy[graph.arc_from[ids]] + cand.proj_frac.to_numpy()[:,None]*(graph.node_xy[graph.arc_to[ids]]-graph.node_xy[graph.arc_from[ids]])
        if not np.allclose(xy,cand[['proj_x','proj_y']],atol=1e-7):
            raise ValueError('Projection inconsistency')
    counts = cand.groupby('obs_idx').size().reindex(obs.obs_idx,fill_value=0).to_numpy()
    if not np.array_equal(counts,obs.n_candidates) or (obs.loc[~obs.is_valid,'n_candidates'] != 0).any():
        raise ValueError('C1/C2 counts')
    if not np.allclose(graph.projection.project(obs[['lon','lat']]), obs[['x','y']],equal_nan=True):
        raise ValueError('C1 projection mismatch')
    if 'label' in cand:
        if str(cand.label.dtype) != 'Int8':
            raise ValueError('Nullable label dtype')
        ref = alignment.set_index('obs_idx').loc[cand.obs_idx]
        trusted = ref.align_status.to_numpy() == 'ok'
        if not cand.label.isna().to_numpy()[~trusted].all():
            raise ValueError('Untrusted labels')
        expected = cand.arc_id.to_numpy() == ref.ref_arc_id.to_numpy()
        if not np.array_equal(cand.label[trusted].to_numpy(dtype=int), expected[trusted].astype(int)):
            raise ValueError('Label reference join')
