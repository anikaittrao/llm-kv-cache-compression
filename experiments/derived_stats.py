"""Derived statistics from the existing results JSONs (audit patch B6, 2026-11-22). Read-only on the inputs; seconds.

Run:  .venv\\Scripts\\python.exe experiments\\derived_stats.py
Writes experiments/results_derived_stats.json (a NEW file; no reported JSON is changed). Every number the audit asked
for in the text fixes U4/U8/U9 is computed here from the saved results, so the papers can cite one file:
  quality          (results_quality.json): Spearman of perplexity with the output (activation) error and with the weight
                   error, pooled over the 15 fixed-size runs (reported 0.97 / 0.43) and within each size (n = 5 each);
                   the cross-size counterexample (plain SVD 90%: output error 0.205, ppl 1689; activation-aware 50%:
                   0.260, 214); the output/weight error ratio per run and its range per family.
  timing_fig14     (results_kv_timing.json): per-token and attention-only ratios against full-cache matmul ALONE
                   (Fig 14 / report Fig 12 used the faster of matmul and SDPA at each T), the baseline that was used
                   per T, and the recorded CPU load.
  sink             (results_kv_sink_mech.json): k0_mean_comp + 1 = k_0 . mu / ||mu||^2 (k_0's own coordinate along
                   the mean, in units of ||mu||) per layer (median over heads), and k0_mean_comp = (k_0 - mu) . mu /
                   ||mu||^2; ranges and medians. Note: that run used 2 test windows.
  bound_level3     (results_kv_bound.json): median actual/level-3 ratio and the implied looseness factor.
  bound_levelV     (results_kv_bound3.json): level V (and 2', C) 5-95% spread as looseness factors (1/ratio).
  spearman_bounds  : the 10-setting Spearman values and the offline-vs-online pair at half cache.
  cpu_load         : recorded load in kv_timing (per point) and kv_cheapupdate (per window).
  online_timing    : corrected kv_online timing summary (setting-filtered; same as kv_online.py --resummarize).
  trivial_proxy    : fields of a trivial error proxy, if a results JSON has them (none of the reported ones does; the
                     audit's C9 computes them in experiments/kv_proxy.py).
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


def spearman(x, y):  # same definition as kv_bound.py (no ties in these data)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def rng(v):
    return [float(min(v)), float(max(v))]


def main():
    out = {'note': 'derived from the reported results JSONs by experiments/derived_stats.py (audit B6, 2026-11-22)'}

    # ---- quality (Regime C)
    q = load('results_quality.json')
    runs = [r for r in q['runs'] if r['label'].startswith('ratio=')]
    fam = {'svd': 'plain', 'rsvd_q0': 'plain', 'rsvd_q2': 'plain', 'wsvd': 'activation-aware', 'wrsvd_q2': 'activation-aware'}
    pooled = {'n': len(runs), 'spearman_output_err': spearman([r['act_rel_err_mean'] for r in runs], [r['ppl'] for r in runs]),
              'spearman_weight_err': spearman([r['weight_rel_err_mean'] for r in runs], [r['ppl'] for r in runs])}
    within = {}
    for lab in sorted({r['label'] for r in runs}, reverse=True):
        rr = [r for r in runs if r['label'] == lab]
        within[lab] = {'n': len(rr), 'spearman_output_err': spearman([r['act_rel_err_mean'] for r in rr], [r['ppl'] for r in rr]),
                       'spearman_weight_err': spearman([r['weight_rel_err_mean'] for r in rr], [r['ppl'] for r in rr])}
    ratio = {f"{r['label']} {r['method']}": r['act_rel_err_mean'] / r['weight_rel_err_mean'] for r in runs}
    by_fam = {f: rng([ratio[f"{r['label']} {r['method']}"] for r in runs if fam[r['method']] == f]) for f in ('plain', 'activation-aware')}
    pick = lambda lab, m: next(r for r in runs if r['label'] == lab and r['method'] == m)
    a, b = pick('ratio=0.9', 'svd'), pick('ratio=0.5', 'wsvd')
    out['quality'] = {'pooled_15_runs': pooled, 'within_size': within,
                      'output_over_weight_error_ratio_per_run': ratio,
                      'output_over_weight_error_ratio_range': by_fam,
                      'output_over_weight_error_ratio_range_rounded': {k: [round(v[0], 2), round(v[1], 2)] for k, v in by_fam.items()},
                      'cross_size_counterexample': {
                          'plain svd 90%': {'output_err': a['act_rel_err_mean'], 'weight_err': a['weight_rel_err_mean'], 'ppl': a['ppl']},
                          'activation-aware svd 50%': {'output_err': b['act_rel_err_mean'], 'weight_err': b['weight_rel_err_mean'], 'ppl': b['ppl']}}}

    # ---- timing (Fig 14 of final_report / Fig 12 of report)
    t = load('results_kv_timing.json')
    tf = {'by_T': {}, 'note': 'ratio_vs_matmul = time / full-cache matmul (full_bmm); reported ratios used the faster of '
                              'full_bmm and full_sdpa at each T (baseline_used)'}
    for p in t['points']:
        M = p['methods']
        base = min(('full_bmm', 'full_sdpa'), key=lambda k: M[k]['token_ms_est'])
        tf['by_T'][str(p['T'])] = {'baseline_used': base, 'cpu_load_pct': p.get('cpu_load_pct'),
                                   'token_ratio_vs_matmul': {k: M[k]['token_ms_est'] / M['full_bmm']['token_ms_est'] for k in M},
                                   'attn_ratio_vs_matmul': {k: M[k]['attn_ms'] / M['full_bmm']['attn_ms'] for k in M},
                                   'token_ratio_reported': {k: M[k].get('token_ratio') for k in M}}
    out['timing_fig14'] = tf

    # ---- sink geometry
    s = load('results_kv_sink_mech.json')
    mc = np.asarray(s['per_layer_head']['k0_mean_comp'])          # (k0 - mu).mu / mu.mu, L x H
    own = mc + 1                                                   # k0.mu / mu.mu
    out['sink'] = {'windows': s['windows'], 'r': s['r'],
                   'k0_coord_mu_per_layer_median': np.median(own, 1).tolist(),
                   'k0_coord_mu_range_of_layer_medians': rng(np.median(own, 1)),
                   'k0_coord_mu_median_all_heads': float(np.median(own)),
                   'k0_minus_mu_coord_per_layer_median': np.median(mc, 1).tolist(),
                   'k0_minus_mu_coord_range_of_layer_medians': rng(np.median(mc, 1)),
                   'k0_minus_mu_coord_median_all_heads': float(np.median(mc)),
                   'layers_with_k0_coord_mu_negative': int((np.median(own, 1) < 0).sum()),
                   'note': 'k0_coord_mu = k0_mean_comp + 1 (units of ||mu||); k0_mean_comp is the component of (k0 - mu)'}

    # ---- bounds
    b1 = load('results_kv_bound.json')
    r3 = b1['ratio_quantiles']['r3']
    out['bound_level3'] = {'median_actual_over_bound': r3['0.5'], 'looseness_factor_median': 1 / r3['0.5'],
                           'looseness_5_95': [1 / r3['0.95'], 1 / r3['0.05']]}
    b3 = load('results_kv_bound3.json')
    rq = b3['ratio_quantiles']
    out['bound_levelV'] = {k: {'median_ratio': rq[k]['0.5'], 'ratio_5_95': [rq[k]['0.05'], rq[k]['0.95']],
                               'looseness_median': 1 / rq[k]['0.5'], 'looseness_5_95': [1 / rq[k]['0.95'], 1 / rq[k]['0.05']]}
                           for k in ('rV', 'rC', 'rL2p')}
    b2 = load('results_kv_bound2.json')
    sb = {'kv_bound (level 2)': b1.get('spearman'), 'kv_bound2 (level 2prime)': b2.get('spearman'), 'kv_bound3 (level V)': b3.get('spearman')}
    half = {r['method']: r for r in b3['rows'] if r['setting'] == 'half'}
    sb['half_cache_offline_vs_online'] = {m: {'ppl': half[m]['ppl'], 'err_mean': half[m]['err_mean'], 'L2p_mean': half[m]['L2p_mean'],
                                              'V_mean': half[m]['V_mean']} for m in ('offline', 'online')}
    sb['n_settings'] = len(b3['rows'])
    out['spearman_bounds'] = sb

    # ---- CPU load
    cu = load('results_kv_cheapupdate.json')
    out['cpu_load'] = {'kv_timing_points_pct': [p.get('cpu_load_pct') for p in t['points']],
                       'kv_timing_start_pct': t.get('cpu_load_start_pct'),
                       'kv_cheapupdate_windows_pct': cu.get('cpu_load_pct'), 'kv_cheapupdate_start_pct': cu.get('cpu_load_start_pct')}
    vals = [x for x in out['cpu_load']['kv_timing_points_pct'] if x is not None]
    out['cpu_load']['kv_timing_range'] = rng(vals) if vals else None
    vals = [x for x in (out['cpu_load']['kv_cheapupdate_windows_pct'] or []) if x is not None]
    out['cpu_load']['kv_cheapupdate_range'] = rng(vals) if vals else None

    # ---- corrected online timing (same as kv_online.py --resummarize)
    import sys
    sys.path.insert(0, EXP)
    from kv_online import timing_summaries
    on = load('results_kv_online.json')
    ts = timing_summaries(on['timing'])
    out['online_timing'] = {'main_setting_KV256_512_P768': ts['timing_summary'], 'by_setting': ts['timing_summary_by_setting'],
                            'pooled_old_as_reported': ts['timing_summary_pooled_old']}

    # ---- trivial proxy fields (C9), if any reported JSON has them
    found = {}
    for fn in ('results_kv_bound.json', 'results_kv_bound2.json', 'results_kv_bound3.json'):
        rows = load(fn)['rows']
        keys = sorted({k for r in rows for k in r if 'proxy' in k.lower() or k.lower().startswith('res')})
        if keys:
            found[fn] = keys
    out['trivial_proxy'] = found or 'not present in the reported JSONs (see experiments/kv_proxy.py, audit C9)'

    with open(os.path.join(EXP, 'results_derived_stats.json'), 'w') as f:
        json.dump(out, f, indent=1)
    Q = out['quality']
    print('quality pooled:', Q['pooled_15_runs'])
    for k, v in Q['within_size'].items():
        print(' within', k, v)
    print('output/weight error ratio range:', Q['output_over_weight_error_ratio_range'])
    print('timing vs matmul at 65536:', {k: round(v, 3) for k, v in out['timing_fig14']['by_T']['65536']['token_ratio_vs_matmul'].items()})
    print('sink k0 coord along mu (layer medians):', [round(x, 2) for x in out['sink']['k0_coord_mu_per_layer_median']])
    print('sink (k0-mu) coord (layer medians):', [round(x, 2) for x in out['sink']['k0_minus_mu_coord_per_layer_median']])
    print('level 3:', out['bound_level3'])
    print('level V:', out['bound_levelV']['rV'])
    print('cpu load:', out['cpu_load']['kv_timing_range'], out['cpu_load']['kv_cheapupdate_range'])
    print('online timing K q=1 main:', out['online_timing']['main_setting_KV256_512_P768'].get('K R=256 q=1'))


if __name__ == '__main__':
    main()
