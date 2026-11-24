import json
a = json.load(open('experiments/results_kv_timing.json'))
b = json.load(open('experiments/rerun_2026-11-19/results_kv_timing.json'))
print('load start', a['cpu_load_start_pct'], b['cpu_load_start_pct'], ' runtime', round(a['total_runtime_s']), round(b['total_runtime_s']))
print('method keys:', list(a['points'][0]['methods']['g1'].keys()))
for pa, pb in zip(a['points'], b['points']):
    line = f"T={pa['T']:6d} load {pa['cpu_load_pct']:3d}/{pb['cpu_load_pct']:3d} |"
    for m in ('full_bmm', 'g1', 'g2', 'g4', 'g12'):
        ma, mb = pa['methods'][m], pb['methods'][m]
        ka = [k for k in ma if 'token' in k and 'ms' in k]
        k = ka[0] if ka else None
        if k:
            line += f" {m} {ma[k]:.2f}/{mb[k]:.2f}"
    print(line)
print('g1 ratio  old:', {k: round(v, 2) for k, v in a['decision']['g1_token_ratio_by_T'].items()})
print('g1 ratio  new:', {k: round(v, 2) for k, v in b['decision']['g1_token_ratio_by_T'].items()})
print('answer', a['decision']['answer_yes'], b['decision']['answer_yes'])
for m in ('g2', 'g4', 'g12'):
    for pa, pb in ((a['points'][-1], b['points'][-1]), (a['points'][1], b['points'][1])):
        print(m, pa['T'], {k: (round(pa['methods'][m][k], 3), round(pb['methods'][m][k], 3)) for k in pa['methods'][m] if 'ratio' in k})
