"""C16 analysis (wave 3): low rank + quantization on GPT-2. Reads c16_kv_quantlr.json (kv_quantlr.py), writes c16_summary.json.
Rules fixed before any full run (e(x) = ppl(x)/dense - 1, 16 windows 0-15), names as in kv_quantlr.py:
  (i)   static / buffer 32 + int8 codes            (ii) update / buffer 32 + int8 codes, master
  (iii) update / buffer 32 + int8 grouped codes    (iv) update / buffer 32 + int8 codes  (reference, C5: +2.54%)
  Q1 hypothesis A (compounding): e(ii) <= e(i) + 0.003.
  Q2 hypothesis B (per-token scale): e(iii) <= e(iv) - 0.005.
  Q3 frontier: e(offline K256/V512 / buffer 32 + int4 codes) <= 0.01;  Q3b: the same for K384/V768.
  Verdicts: A supported if Q1 and e(iv) - e(ii) >= 0.005; B supported if Q2 and e(iv) - e(iii) >= 0.005; the re-projection
  diagnostic supports A if the median relative code error of (iv) at 7 re-projections is >= 1.5 x that at 0 while (ii) is flat
  (7-vs-0 ratio <= 1.2).  Also reported: code checks vs C5 / kv_buffer.py (full 16 windows: static 24.997, update 24.778,
  offline 25.022, C5 update+int8 +2.54%, int4 +0.10%), all excesses with CIs, memory per method.
"""
import numpy as np
import w3_stats as W

NAMES = {'i': 'static / buffer 32 + int8 codes', 'ii': 'update / buffer 32 + int8 codes, master',
         'iii': 'update / buffer 32 + int8 grouped codes', 'iv': 'update / buffer 32 + int8 codes',
         'iiim': 'update / buffer 32 + int8 grouped codes, master', 'int4': 'int4 + buffer 32',
         'off_int4': 'offline K256/V512 / buffer 32 + int4 codes', 'off_big_int4': 'offline K384/V768 / buffer 32 + int4 codes',
         'off_grp_int4': 'offline K256/V512 / buffer 32 + int4 grouped codes'}


def main():
    r = W.load('c16_kv_quantlr.json')
    nllw, tok = r['nll_per_window'], r['tokens_per_window']
    e = r['excess']
    out = {'windows': r['windows'], 'excess': {m: e[m] for m in e if m != 'dense'}, 'memory_T1024': r['memory_T1024'],
           'excess_ci': {m: W.ex_ci(nllw, m, 'dense', tok) for m in e if m != 'dense'},
           'code_check_ppl_16_windows': {m: r['ppl'].get(m) for m in ('static / buffer 32', 'update / buffer 32', 'offline K256/V512 / buffer 32')}}
    g = lambda k: e[NAMES[k]]
    out['Q1'] = {'pass': bool(g('ii') <= g('i') + 0.003), 'e_ii': g('ii'), 'e_i': g('i'),
                 'diff_ci': W.boot(lambda s: W.e_of(nllw, NAMES['ii'], 'dense', s, tok) - W.e_of(nllw, NAMES['i'], 'dense', s, tok), len(r['windows']))}
    out['Q2'] = {'pass': bool(g('iii') <= g('iv') - 0.005), 'e_iii': g('iii'), 'e_iv': g('iv'),
                 'diff_ci': W.boot(lambda s: W.e_of(nllw, NAMES['iii'], 'dense', s, tok) - W.e_of(nllw, NAMES['iv'], 'dense', s, tok), len(r['windows']))}
    out['Q3'] = {'pass': bool(g('off_int4') <= 0.01), 'e': g('off_int4'), 'memory': r['memory_T1024'][NAMES['off_int4']]}
    out['Q3b'] = {'pass': bool(g('off_big_int4') <= 0.01), 'e': g('off_big_int4'), 'memory': r['memory_T1024'][NAMES['off_big_int4']]}
    out['frontier_vs_int4'] = {'int4_alone': {'e': g('int4'), 'memory': r['memory_T1024'][NAMES['int4']]},
                               'offline_K256V512_int4': out['Q3'], 'offline_K384V768_int4': out['Q3b'],
                               'offline_K256V512_int4_grouped': {'e': g('off_grp_int4'), 'memory': r['memory_T1024'][NAMES['off_grp_int4']]},
                               'offline_K384V768_int4_grouped': {'e': e['offline K384/V768 / buffer 32 + int4 grouped codes'],
                                                                 'memory': r['memory_T1024']['offline K384/V768 / buffer 32 + int4 grouped codes']}}
    # re-projection diagnostic
    diag = {}
    for m, rows in r.get('reproj_err', {}).items():
        d = {}
        for kind in 'KV':
            med = []
            for c in range(8):
                v = [x[kind][l][c] for x in rows for l in range(len(x[kind])) if x[kind][l] is not None and x[kind][l][c] is not None]
                med.append(float(np.median(v)) if v else None)
            d[kind] = med
        diag[m] = d
    out['reproj_err_median_by_count'] = diag
    ratio = lambda m, k: (diag[m][k][7] / diag[m][k][0]) if m in diag and diag[m][k][0] and diag[m][k][7] else None
    out['reproj_ratio_7_over_0'] = {m: {k: ratio(m, k) for k in 'KV'} for m in diag}
    dv = lambda k: ratio(NAMES['iv'], k)
    mv = lambda k: ratio(NAMES['ii'], k)
    out['diag_supports_A'] = bool(all(dv(k) is not None and dv(k) >= 1.5 for k in 'KV') and all(mv(k) is not None and mv(k) <= 1.2 for k in 'KV')) if diag else None
    out['A_supported'] = bool(out['Q1']['pass'] and g('iv') - g('ii') >= 0.005)
    out['B_supported'] = bool(out['Q2']['pass'] and g('iv') - g('iii') >= 0.005)
    W.save('c16_summary.json', out)
    for k in ('Q1', 'Q2', 'Q3', 'Q3b'):
        print(k, out[k]['pass'])
    print('A supported', out['A_supported'], 'B supported', out['B_supported'], 'diag supports A', out['diag_supports_A'])
    for m in out['excess']:
        print(f"{m:58s} mem {r['memory_T1024'][m]:.3f}  e {100 * out['excess'][m]:7.3f}%")


if __name__ == '__main__':
    main()
