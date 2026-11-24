import json
for f in ('experiments/results.json', 'experiments/rerun_2026-11-19/results.json'):
    r = json.load(open(f))['regimeA']
    sp = [r[m]['by_k'][k]['q'][q]['speedup_vs_svd'] for m in r if m.startswith('L5.') for k in r[m]['by_k'] for q in ('0', '1', '2')]
    k32 = {m: [round(r[m]['by_k']['32']['q'][q]['speedup_vs_svd'], 1) for q in ('0', '1', '2')] for m in r if m.startswith('L5.')}
    print(f, 'layer 5 speedup range %.1f-%.1f' % (min(sp), max(sp)), 'k=32:', k32)
