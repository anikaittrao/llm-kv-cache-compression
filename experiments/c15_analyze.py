"""C15 analysis (wave 3): SmolLM2 on fresh windows 16-31, baselines and the offline basis. Merges
  c15a_rope_w16-31.json            (kv_rope_port.py --window_offset 16 --windows 16: the 9 low-rank cells)
  c15b_rope_baselines_w16-31.json  (kv_rope_baselines.py: dense, int8/int4 + buffer 32, sink+window f=0.5/0.75, offline / buffer 32)
and writes experiments/rerun_2026-11-22/c15_summary.json. Rules fixed before any full run (e(x) = ppl(x)/dense - 1, 16 windows,
dense of c15a; the dense NLL of the two files must agree to < 1e-2 per window, reported as 'dense_merge_check'):
  R1 e(attn-late+update / buffer 32) <= 0.9 x e(update / buffer 32)          (confirms the post hoc cell of C13)
  R2 e(int4 + buffer 32) <= 0.01                                              (4-bit KV quantization is nearly free)
  R3 e(offline / buffer 32) <= e(update / buffer 32) + 0.005                 (offline is no worse than online + 0.5 point)
  R4 among all methods whose codes-only memory is <= 0.55 (c15b 'memory_codes_only_T': int4 + buffer 32 at 0.304,
     sink+window f=0.5 at 0.500, offline and the 8 low-rank cells of c15a at 0.505; int8 + buffer 32 computes to 0.552 in the
     smoke test and is therefore just OUTSIDE the cut, reported in the secondary <= 0.60 list), the one with the smallest excess
     is a low-rank method (any of static / update / attn-late / attn-late+update / offline). Reported with the full ranked list.
     Smoke-test note (window 0 only, not a result): sink+window had NEGATIVE excess there, i.e. R4 is expected to fail.
Each excess comes with a paired bootstrap CI; R1/R3 also with the CI of the ratio / difference (windows resampled).
"""
import numpy as np
import w3_stats as W

LOWRANK = ('static', 'update', 'attn-late', 'offline')


def main():
    a = W.load('c15a_rope_w16-31.json')
    b = W.load('c15b_rope_baselines_w16-31.json')
    assert a['windows'] == b['windows'], (a['windows'], b['windows'])
    tok = a['tokens_per_window']
    nll = dict(a['nll_per_window'])
    for m, v in b['nll_per_window'].items():
        if m != 'dense':
            nll[m] = v
    out = {'windows': a['windows'], 'tokens_per_window': tok}
    out['dense_merge_check_max_abs_nll_diff'] = float(np.max(np.abs(np.array(a['nll_per_window']['dense']) - np.array(b['nll_per_window']['dense']))))
    e = {m: W.excess(nll[m], nll['dense'], tok) for m in nll if m != 'dense'}
    out['excess'] = e
    out['excess_ci'] = {m: W.ex_ci(nll, m, 'dense', tok) for m in e}
    # memory (codes only)
    lr_mem = [v for k, v in b['memory_codes_only_T'].items() if k.startswith('offline')][0]
    mem = {m: (b['memory_codes_only_T'].get(m, lr_mem) if m in b['memory_codes_only_T'] else lr_mem) for m in e}
    out['memory_codes_only'] = mem
    n = len(nll['dense'])
    # rules
    R1 = e['attn-late+update / buffer 32'] <= 0.9 * e['update / buffer 32']
    out['R1'] = {'pass': bool(R1), 'ratio_of_excess': W.ex_ratio_ci(nll, 'attn-late+update / buffer 32', 'update / buffer 32', 'dense', tok),
                 'ppl_ratio': W.ppl_ratio_ci(nll, 'attn-late+update / buffer 32', 'update / buffer 32', tok)}
    out['R2'] = {'pass': bool(e['int4 + buffer 32'] <= 0.01), 'excess_int4': e['int4 + buffer 32'], 'excess_int8': e['int8 + buffer 32']}
    d3 = W.boot(lambda s: W.e_of(nll, 'offline / buffer 32', 'dense', s, tok) - W.e_of(nll, 'update / buffer 32', 'dense', s, tok), n)
    out['R3'] = {'pass': bool(e['offline / buffer 32'] <= e['update / buffer 32'] + 0.005),
                 'e_offline': e['offline / buffer 32'], 'e_update': e['update / buffer 32'], 'difference_ci': d3,
                 'ppl_ratio_offline_over_update': W.ppl_ratio_ci(nll, 'offline / buffer 32', 'update / buffer 32', tok)}
    cand = sorted([m for m in e if mem[m] <= 0.55 + 1e-12], key=lambda m: e[m])
    out['R4'] = {'pass': bool(cand and cand[0].startswith(LOWRANK)), 'best': cand[0] if cand else None,
                 'ranked_memory_le_0.55': [(m, mem[m], e[m]) for m in cand]}
    cand6 = sorted([m for m in e if mem[m] <= 0.60 + 1e-12], key=lambda m: e[m])   # secondary: int8 sits at 0.552, just above 0.55
    out['R4_secondary_memory_le_0.60'] = {'best': cand6[0] if cand6 else None, 'best_is_lowrank': bool(cand6 and cand6[0].startswith(LOWRANK)),
                                          'ranked': [(m, mem[m], e[m]) for m in cand6]}
    out['table'] = sorted([(m, mem[m], e[m]) for m in e], key=lambda x: x[1])
    W.save('c15_summary.json', out)
    for r in ('R1', 'R2', 'R3', 'R4'):
        print(r, out[r]['pass'])
    print('best at memory <= 0.55:', out['R4']['best'])
    for row in out['R4']['ranked_memory_le_0.55']:
        print('  ', row[0], round(row[1], 3), round(100 * row[2], 3))
    print('dense merge check', out['dense_merge_check_max_abs_nll_diff'])


if __name__ == '__main__':
    main()
