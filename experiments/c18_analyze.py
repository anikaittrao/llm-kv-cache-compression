"""C18 analysis (wave 3): two confound closers. Reads
  c18a_kv_equalcount.json          (kv_equalcount.py: per-step NLL of own only, sliding 16/32, block 16/32/64, dense)
  c18a_kv_control_perstep.json     (kv_control.py --per_step: own only (check), lagged 31 + own, random old 31 + own)
  c18b_kv_novelty_interleaved.json (kv_novelty_interleaved.py)
and writes c18_summary.json. Rules fixed before any full run (e(x) = ppl(x)/dense - 1 over the SCORED steps; windows 0-15;
primary scoring = continuation steps >= 63 (192 of the 255 steps), the all-steps values are reported next to them):
  E1 e(static / sliding 16) <= e(static / block 32)    (equal average exact count 16 vs 16.5; sliding no worse)
  E2 e(static / sliding 32) <= e(static / block 64)    (32 vs 32.5)
  Z3 e(lagged 31 + own) <= (e(own only) + e(sliding 32)) / 2, steps >= 63 (own only / sliding 32 / dense from c18a_equalcount,
     lagged from the control run; the control's own-only cell is the code check against the equalcount own-only cell)
  N4 interleaved held-out ratio within 25% of the contiguous one for keys AND values:
     |median ratio_heldout_inter / median ratio_heldout_contig - 1| <= 0.25 (then drift plays no role); else drift matters
     (smaller interleaved ratio) or the contiguous split is not the pessimistic case (larger).
Paired bootstrap CIs over windows (10,000 resamples) for every excess and for the E1/E2/Z3 differences.
"""
import numpy as np
import w3_stats as W

FROM = 63


def scored(steps, lo):
    """{method: per-window sums over steps lo..} and number of steps."""
    return {m: [float(sum(v[lo:])) for v in vv] for m, vv in steps.items()}, len(next(iter(steps.values()))[0]) - lo


def main():
    out = {}
    try:
        q = W.load('c18a_kv_equalcount.json')
        steps = q['nll_per_step']
        n = len(q['windows'])
        res = {}
        for tag, lo in (('from63', FROM), ('all', 0)):
            nl, nt = scored(steps, lo)
            e = {m: W.excess(nl[m], nl['dense'], nt) for m in nl if m != 'dense'}
            d = {'steps_scored': nt, 'excess': e, 'excess_ci': {m: W.ex_ci(nl, m, 'dense', nt) for m in e}}
            for name, A, Bm in (('E1', 'static / sliding 16', 'static / block 32'), ('E2', 'static / sliding 32', 'static / block 64')):
                diff = W.boot(lambda s: W.e_of(nl, A, 'dense', s, nt) - W.e_of(nl, Bm, 'dense', s, nt), n)
                d[name] = {'pass': bool(e[A] <= e[Bm]), 'e_sliding': e[A], 'e_block': e[Bm], 'difference_ci': diff,
                           'ppl_ratio': W.ppl_ratio_ci(nl, A, Bm, nt)}
            res[tag] = d
            print(tag, 'E1', d['E1']['pass'], 'E2', d['E2']['pass'], {k: round(100 * v, 3) for k, v in e.items()})
        out['c18a_equalcount'] = res
        out['E1_primary'] = res['from63']['E1']['pass']; out['E2_primary'] = res['from63']['E2']['pass']
        # Z3
        try:
            c = W.load('c18a_kv_control_perstep.json')
            assert c['windows'] == q['windows']
            cs = c['nll_per_step']
            nlc, nt = scored(cs, FROM)
            nlq, _ = scored(steps, FROM)
            dense = nlq['dense']
            all_ = dict(nlq); all_.update({'lagged 31 + own': nlc['lagged 31 + own'], 'random old 31 + own': nlc['random old 31 + own']})
            e = {m: W.excess(all_[m], dense, nt) for m in ('static / own only', 'static / sliding 32', 'lagged 31 + own', 'random old 31 + own')}
            h = (e['static / own only'] + e['static / sliding 32']) / 2
            f = lambda s, m: W.e_of(all_, m, 'dense', s, nt)
            z3 = {'excess_from63': e, 'threshold_h': h, 'Z3_pass': bool(e['lagged 31 + own'] <= h),
                  'random_old_le_h': bool(e['random old 31 + own'] <= h),
                  'lagged_minus_h_ci': W.boot(lambda s: f(s, 'lagged 31 + own') - (f(s, 'static / own only') + f(s, 'static / sliding 32')) / 2, n),
                  'check_own_only_control_vs_equalcount_max_abs_step_diff': float(np.max(np.abs(np.array(cs['own only (check)']) - np.array(steps['static / own only']))))}
            out['Z3'] = z3
            print('Z3', z3['Z3_pass'], {k: round(100 * v, 3) for k, v in e.items()}, 'h', round(100 * h, 3))
        except FileNotFoundError as ex:
            print('missing', ex)
    except FileNotFoundError as ex:
        print('missing', ex)
    try:
        b = W.load('c18b_kv_novelty_interleaved.json')
        s = b['summary']
        n4 = {}
        for k in 'KV':
            n4[k] = {'heldout_contig': s[f'{k} ratio_heldout_contig median'], 'heldout_inter': s[f'{k} ratio_heldout_inter median'],
                     'cont_contig': s[f'{k} ratio_cont_contig median'], 'cont_inter': s[f'{k} ratio_cont_inter median'],
                     'rel_diff_heldout': s[f'{k} ratio_heldout_inter median'] / s[f'{k} ratio_heldout_contig median'] - 1}
            rr = [x for x in b['rows'] if x['kind'] == k]
            dd = np.array([x['ratio_heldout_inter'] - x['ratio_heldout_contig'] for x in rr])
            n4[k]['paired_diff_median_5_95'] = [float(np.median(dd)), float(np.quantile(dd, 0.05)), float(np.quantile(dd, 0.95))]
        n4['N4_pass'] = bool(all(abs(n4[k]['rel_diff_heldout']) <= 0.25 for k in 'KV'))
        n4['direction'] = {k: ('interleaved smaller by >25% (drift contributes)' if n4[k]['rel_diff_heldout'] < -0.25 else
                               'interleaved larger by >25% (contiguous split not pessimistic)' if n4[k]['rel_diff_heldout'] > 0.25
                               else 'within 25%') for k in 'KV'}
        out['N4'] = n4
        print('N4', n4['N4_pass'], n4['direction'], {k: (round(n4[k]['heldout_contig'], 3), round(n4[k]['heldout_inter'], 3)) for k in 'KV'})
    except FileNotFoundError as ex:
        print('missing', ex)
    W.save('c18_summary.json', out)


if __name__ == '__main__':
    main()
