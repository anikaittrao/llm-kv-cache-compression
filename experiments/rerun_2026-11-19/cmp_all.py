"""Compare the idle-CPU reruns (this folder) with the reported JSONs: timings and (for reproducibility) quality numbers."""
import json
import numpy as np
O, N = 'experiments/', 'experiments/rerun_2026-11-19/'
ld = lambda p: json.load(open(p))

print('===== Regime A (results.json): SVD time, RSVD speedups')
a, b = ld(O + 'results.json'), ld(N + 'results.json')
print('runtime s', round(a['total_runtime_s']), round(b['total_runtime_s']), ' threads', a['hardware']['torch_threads'], b['hardware']['torch_threads'])
for q in ('0', '1', '2'):
    sa = [a['regimeA'][m]['by_k'][k]['q'][q]['speedup_vs_svd'] for m in a['regimeA'] for k in a['regimeA'][m]['by_k']]
    sb = [b['regimeA'][m]['by_k'][k]['q'][q]['speedup_vs_svd'] for m in b['regimeA'] for k in b['regimeA'][m]['by_k']]
    ea = [a['regimeA'][m]['by_k'][k]['q'][q]['err_ratio_to_optimal'] for m in a['regimeA'] for k in a['regimeA'][m]['by_k']]
    eb = [b['regimeA'][m]['by_k'][k]['q'][q]['err_ratio_to_optimal'] for m in b['regimeA'] for k in b['regimeA'][m]['by_k']]
    print(f' q={q} speedup min/median/max old {min(sa):.1f}/{np.median(sa):.1f}/{max(sa):.1f}  new {min(sb):.1f}/{np.median(sb):.1f}/{max(sb):.1f}'
          f'   err ratio old {min(ea):.4f}-{max(ea):.4f} new {min(eb):.4f}-{max(eb):.4f}')
for k in ('16', '32', '64', '128'):
    sa = [a['regimeA'][m]['by_k'][k]['q']['0']['speedup_vs_svd'] for m in a['regimeA']]
    sb = [b['regimeA'][m]['by_k'][k]['q']['0']['speedup_vs_svd'] for m in b['regimeA']]
    print(f'  k={k:>3} q=0 speedup max old {max(sa):.1f} new {max(sb):.1f}  median old {np.median(sa):.1f} new {np.median(sb):.1f}')
for typ in ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'):
    ta = [a['regimeA'][m]['svd_time_median_s'] for m in a['regimeA'] if m.endswith(typ)]
    tb = [b['regimeA'][m]['svd_time_median_s'] for m in b['regimeA'] if m.endswith(typ)]
    print(f'  SVD time {typ:12s} median old {np.median(ta):.2f}s new {np.median(tb):.2f}s')
print(' assertions', a['all_assertions_passed'], b['all_assertions_passed'],
      ' Thm-bound cases', a['assertions']['hmt_expected_error_bound_q0'], b['assertions']['hmt_expected_error_bound_q0']['n_holds'])
g = lambda r: [(x['k'], x['rho'], round(x.get('heldout', x.get('heldout_rel_err', -1)), 3)) for x in r['regimeB_grid']['runs']] if 'k' in r['regimeB_grid']['runs'][0] else None
print(' regimeB grid keys', list(a['regimeB_grid']['runs'][0].keys())[:10])
d = max(abs(a['regimeA'][m]['by_k'][k]['eckart_young_rel_err'] - b['regimeA'][m]['by_k'][k]['eckart_young_rel_err']) for m in a['regimeA'] for k in a['regimeA'][m]['by_k'])
print(' max |EY rel err diff| =', d)

print('\n===== Regime C (results_quality.json)')
a, b = ld(O + 'results_quality.json'), ld(N + 'results_quality.json')
print(' baseline ppl', a['baseline']['ppl'], b['baseline']['ppl'], ' runtime', round(a['total_runtime_s']), round(b['total_runtime_s']))
for ra, rb in zip(a['runs'], b['runs']):
    assert ra['label'] == rb['label'] and ra['method'] == rb['method']
    print(f"  {ra['label']:12s} {ra['method']:9s} ppl {ra['ppl']:9.3f} {rb['ppl']:9.3f}   factor_s {ra['factor_time_s']:6.1f} {rb['factor_time_s']:6.1f}")
print(' layer latency (shape, T, ratio): speedup old -> new')
for xa, xb in zip(a['layer_latency'], b['layer_latency']):
    print(f"  {str(xa['shape']):12s} T={xa['T']:5d} r={xa['ratio']:.1f} {xa['speedup']:.2f} -> {xb['speedup']:.2f}")
print(' decode latency')
for xa, xb in zip(a['decode_latency'], b['decode_latency']):
    print(f"  {xa['method']:6s} r={xa['ratio']:.1f} s/token {1e3*xa['s_per_token']:.1f}ms -> {1e3*xb['s_per_token']:.1f}ms  speedup {xa['speedup_vs_dense']:.3f} -> {xb['speedup_vs_dense']:.3f}")

print('\n===== Online basis (results_kv_online.json)')
a, b = ld(O + 'results_kv_online.json'), ld(N + 'results_kv_online.json')
for s in a['settings']:
    print(' ', s, {m: (round(a['settings'][s]['ppl'][m], 3), round(b['settings'][s]['ppl'][m], 3)) for m in a['settings'][s]['ppl']})
print(' decision', a['decision'], b['decision'])
for k in a['timing_summary']:
    x, y = a['timing_summary'][k], b['timing_summary'][k]
    print(f"  {k:12s} svd {x['svd_ms_median']:6.1f} -> {y['svd_ms_median']:6.1f} ms   rsvd {x['rsvd_ms_median']:6.1f} -> {y['rsvd_ms_median']:6.1f} ms")
for k in a['theory_summary']:
    print(f"  {k:12s} proj max {a['theory_summary'][k]['proj_ratio_max']:.3f} -> {b['theory_summary'][k]['proj_ratio_max']:.3f}")
