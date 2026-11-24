"""C13 pooling: SmolLM2 16 windows (c11 windows 0-7 + c13a new methods on 0-7 + c13b windows 8-15, all methods). Writes c13_summary.json."""
import json, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from stats_utils import ppl_ratio
c11 = json.load(open(os.path.join(HERE, 'c11_rope_smollm2.json')))
a = json.load(open(os.path.join(HERE, 'c13a_rope_w0-7_attnupdate.json')))
b = json.load(open(os.path.join(HERE, 'c13b_rope_w8-15_all.json')))
tok = b['tokens_per_window']
nll = {}
for m in b['nll_per_window']:
    first = c11['nll_per_window'].get(m) or a['nll_per_window'].get(m)
    nll[m] = list(first) + list(b['nll_per_window'][m])
ppl = {m: math.exp(sum(v) / (tok * len(v))) for m, v in nll.items()}
e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
# code check: the old windows 0-7 dense vs c13a dense
chk = max(abs(x - y) for x, y in zip(c11['nll_per_window']['dense'], a['nll_per_window']['dense']))
chk2 = max(abs(x - y) for x, y in zip(c11['nll_per_window']['update / buffer 32'], b['nll_per_window']['update / buffer 32'][:0] or c11['nll_per_window']['update / buffer 32']))
n = {m: len(v) for m, v in nll.items()}
R = lambda x, y: ppl_ratio(nll[x], nll[y], tok)
cis = {
 'update/buf vs static/buf': R('update / buffer 32', 'static / buffer 32'),
 'attn-late/buf vs update/buf': R('attn-late / buffer 32', 'update / buffer 32'),
 'attn-late/buf vs static/buf': R('attn-late / buffer 32', 'static / buffer 32'),
 'attn-late/nobuf vs static/nobuf': R('attn-late / no buffer', 'static / no buffer'),
 'attn-late+update/buf vs attn-late/buf': R('attn-late+update / buffer 32', 'attn-late / buffer 32'),
 'attn-late+update/buf vs update/buf': R('attn-late+update / buffer 32', 'update / buffer 32'),
 'attn-late+update/nobuf vs attn-late/nobuf': R('attn-late+update / no buffer', 'attn-late / no buffer'),
}
d = {'Q1r_update_buf_le_0.5pct': e['update / buffer 32'] <= 0.005,
     'Q2r_order': e['update / buffer 32'] < e['static / buffer 32'] < e['update / no buffer'] < e['static / no buffer'],
     'Q3r_CI_below_1': cis['update/buf vs static/buf']['ci_hi'] < 1,
     'W1r': e['attn-late / no buffer'] <= 0.9 * e['static / no buffer'],
     'W3r': e['attn-late / buffer 32'] <= 0.9 * e['static / buffer 32'],
     'A1r': e['attn-late+update / buffer 32'] <= e['attn-late / buffer 32'],
     'A2r': e['attn-late+update / buffer 32'] <= 0.9 * e['update / buffer 32']}
out = {'windows': list(range(16)), 'n_windows': n, 'ppl': ppl, 'excess': e, 'cis': cis, 'decision': d,
       'dense_check_c11_vs_c13a_max_abs_nll_diff': chk}
json.dump(out, open(os.path.join(HERE, 'c13_summary.json'), 'w'), indent=1)
print('n', n); print('excess %', {k: round(100 * v, 3) for k, v in e.items()}); print(d); print('dense check', chk)
for k, v in cis.items(): print(k, round(v['ratio'], 5), round(v['ci_lo'], 5), round(v['ci_hi'], 5), v['windows_won'], '/', v['windows'], 'p', round(v['sign_p'], 4))
