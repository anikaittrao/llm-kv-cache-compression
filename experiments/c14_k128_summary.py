"""C14 add-on (2026-11-24): summary + paired CIs of the GPT-2 K128/V256 low-R/n run (c14c_kv_lowrn_K128V256.json),
using the same summarize() and PAIRS as c14_analyze.py. Adds key 'c14c_gpt2_lowrn_K128V256' to c14_summary.json
(other keys untouched). Run from the project root:  .venv\\Scripts\\python.exe experiments\\c14_k128_summary.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_stats as W
from c14_analyze import summarize, PAIRS

c = W.load('c14c_kv_lowrn_K128V256.json')
s = summarize(c, 'c14c_K128V256', PAIRS)
e = c['excess']
s['attnlate_over_static_buf'] = e['attn-late / buffer 32'] / e['static / buffer 32']
s['update_over_static_buf'] = e['update / buffer 32'] / e['static / buffer 32']
d = W.load('c14_summary.json')
d['c14c_gpt2_lowrn_K128V256'] = s
W.save('c14_summary.json', d)
print({k: round(100 * v, 3) for k, v in s['excess'].items()})
print('attn-late/static (buf):', round(s['attnlate_over_static_buf'], 4), ' update/static (buf):', round(s['update_over_static_buf'], 4))
for k, v in s['ratios'].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
