"""CIs for C12 (reads c2 and c12 JSONs, writes c12_cis.json)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from stats_utils import ppl_ratio
d = json.load(open(os.path.join(HERE, 'c12_kv_control.json'))); c = json.load(open(os.path.join(HERE, 'c2_kv_synergy.json')))
tok = d['tokens_per_window']; n = d['nll_per_window']; n2 = c['nll_per_window']
out = {}
for m in ('lagged 31 + own', 'random old 31 + own'):
    out[f'{m} vs own only (B=1)'] = ppl_ratio(n[m], n2['static / B=1'], tok)
    out[f'{m} vs sliding 32'] = ppl_ratio(n[m], n2['static / sliding 32'], tok)
    out[f'{m} vs random prompt 31 + own'] = ppl_ratio(n[m], n2['static / random 31 + own'], tok)
out['check own-only vs c2 B=1 max |nll diff|'] = max(abs(a - b) for a, b in zip(n['own only (check)'], n2['static / B=1']))
json.dump(out, open(os.path.join(HERE, 'c12_cis.json'), 'w'), indent=1)
for k, v in out.items(): print(k, v)
