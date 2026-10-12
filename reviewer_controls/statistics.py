"""Joint common support: every reported cell uses the SAME strata and weights.

Bootstrap units are whole rollouts (paired rollout IDs for RQ2). Original
empirical stratum weights stay fixed. A bootstrap draw missing any required
cell in a supported stratum is invalid, rather than silently changing its
population. Sparse support can therefore legitimately yield no interval.
"""
import numpy as np
import pandas as pd
from e2e6.analysis import ClusterBootstrap

DISTANCES = ('1', '2–4', '5–16', '17–64', '>64')
BASE_STRATA = ['y', 'round_bin', 'position_bin', 'tree_depth', 'draft_k',
               'active_bin', 'exposure_bin']
SAME_STRATA = BASE_STRATA + ['same_events_exposure', 'same_y_exposure']


def estimate(value, draws, supported=True):
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    enough = supported and len(values) >= 20
    return dict(estimate=float(value) if value is not None and np.isfinite(value) else None,
                ci95=ClusterBootstrap.interval(values) if enough else [None, None],
                bootstrap_valid_resamples=len(values),
                ci_status='estimated' if enough else 'insufficient_clusters_or_bootstrap_support')


def joint_match(frame, cells, boot, strata):
    """Return standardized risks and all directional pairwise contrasts.

    `cell` identifies mutually exclusive observational groups, or arm/exposure
    cells. No individual event pairing is assumed. Bootstrap draws use the
    supplied common cluster weights, shared across ALL cells.
    """
    frame = frame[frame['cell'].isin(cells)].copy() if len(frame) else frame.copy()
    support = dict(eligible_events=len(frame), eligible_clusters=int(frame.rollout_id.nunique()) if len(frame) else 0,
                   matched_events=0, common_strata=0, matched_clusters=0, coverage=0.,
                   matching_variables=list(strata), stratum_weighting='pooled eligible event count, fixed across cells and resamples',
                   bootstrap_missing_cell_policy='invalidate entire draw; never impute or renormalize strata',
                   cells={cell:dict(eligible_events=int((frame.cell == cell).sum()) if len(frame) else 0,
                                    matched_events=0, matched_clusters=0) for cell in cells})
    risks = {cell:estimate(None, []) for cell in cells}
    contrasts = {f'{a}_minus_{b}':dict(risk_difference=estimate(None, []), relative_risk=estimate(None, []),
                                         relative_change=estimate(None, []))
                 for a in cells for b in cells if a != b}
    if not len(frame):return dict(support=support, risks=risks, contrasts=contrasts)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[strata]), sort=True)
    counts = pd.crosstab(codes, frame.cell.to_numpy()).reindex(columns=cells, fill_value=0)
    common = counts.index[(counts > 0).all(axis=1)].to_numpy()
    if not len(common):return dict(support=support, risks=risks, contrasts=contrasts)
    mask = np.isin(codes, common)
    matched = frame.loc[mask]
    remap = {code:i for i,code in enumerate(common)}
    s = np.array([remap[code] for code in codes[mask]])
    r = matched.rollout_id.map(boot.index).to_numpy(int)
    c = matched.cell.map({cell:i for i,cell in enumerate(cells)}).to_numpy(int)
    den = np.zeros((len(cells),len(boot.clusters),len(common)))
    num = np.zeros_like(den)
    np.add.at(den,(c,r,s),1.)
    np.add.at(num,(c,r,s),matched.is_miss.to_numpy(float))
    stratum_weights = den.sum(axis=(0,1));stratum_weights /= stratum_weights.sum()
    points = (num.sum(1)/den.sum(1)) @ stratum_weights
    draws = np.full((len(boot.weights),len(cells)),np.nan)
    for start in range(0,len(boot.weights),32):
        w = boot.weights[start:start+32]
        ds = np.einsum('br,crs->bcs',w,den)
        ns = np.einsum('br,crs->bcs',w,num)
        valid = (ds > 0).all(axis=(1,2))
        draws[start:start+len(w)][valid] = (ns[valid]/ds[valid]) @ stratum_weights
    cell_clusters = (den.sum(2)>0).sum(1)
    enough = bool((cell_clusters>=2).all())
    support.update(matched_events=len(matched),common_strata=len(common),matched_clusters=int(matched.rollout_id.nunique()),
                   coverage=len(matched)/len(frame),bootstrap_valid_resamples=int(np.isfinite(draws[:,0]).sum()))
    for i,cell in enumerate(cells):
        support['cells'][cell].update(matched_events=int(den[i].sum()), matched_clusters=int(cell_clusters[i]),
                                      coverage=int(den[i].sum())/max(support['cells'][cell]['eligible_events'],1))
        risks[cell] = estimate(points[i],draws[:,i],enough)
    for i,a in enumerate(cells):
        for j,b in enumerate(cells):
            if i == j:continue
            ratios = np.divide(draws[:,i],draws[:,j],out=np.full(len(draws),np.nan),where=draws[:,j]>0)
            ratio = points[i]/points[j] if points[j]>0 else None
            contrasts[f'{a}_minus_{b}'] = dict(
                risk_difference=estimate(points[i]-points[j],draws[:,i]-draws[:,j],enough),
                relative_risk=estimate(ratio,ratios,enough),
                relative_change=estimate(ratio-1 if ratio is not None else None,ratios-1,enough))
    return dict(support=support,risks=risks,contrasts=contrasts)


def inference(stat, positive, negative, inconclusive):
    lo, hi = stat['ci95']
    if lo is None:return 'Insufficient common support or independent rollout clusters for inference.'
    if lo > 0:return positive
    if hi < 0:return negative
    return inconclusive
