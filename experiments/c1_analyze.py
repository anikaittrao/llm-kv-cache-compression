"""C1 analysis (audit 2026-11-22): fresh-window confirmation of Section 8 with paired bootstrap CIs.

Run (after the wave-1 runs):  .venv\\Scripts\\python.exe experiments\\c1_analyze.py
Reads  experiments/rerun_2026-11-22/c1_kv_buffer.json     (kv_buffer.py, windows 16-31, main setting)
       experiments/rerun_2026-11-22/c1_kv_update.json     (kv_update.py, windows 16-31, main setting B=32)
       experiments/rerun_2026-11-22/c1_kv_attnbasis.json  (kv_attnbasis.py, windows 16-31, main setting)
       experiments/rerun_2026-11-22/c1b_kv_buffer_skip0.json (kv_buffer.py --calib_skip0, windows 0-15, offline only)
Writes experiments/rerun_2026-11-22/c1_summary.json.

Rules fixed before the run (main setting: P = 768, K 256 / V 512, half the codes; windows 16-31; e(x) = ppl(x)/dense - 1;
CIs = paired bootstrap over the 16 windows, 10,000 resamples, 95% percentile, experiments/stats_utils.py):
  Q1 both small      : e(update / buffer B=32) <= 0.005.
  Q2 order           : e(update / buffer B=32) < e(static / buffer B=32) < e(update / no buffer) < e(static / no buffer).
  Q3 both vs buffer  : the CI of ppl(update / buffer B=32) / ppl(static / buffer B=32) lies below 1.
  Q4 descriptive     : CIs of update vs update decay / update no-reproj / oracle refit (kv_update.py) and of
                       attn-late vs online, buffer off and on (kv_attnbasis.py); these are expected to contain 1
                       ("indistinguishable at this resolution").
C1b (descriptive): offline / no buffer and offline / buffer B=32 with calibration statistics without position 0,
windows 0-15, against the reported 25.368 / 25.022.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import ppl_ratio, excess_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22')


def load(name):
    p = os.path.join(RR, name)
    if not os.path.exists(p):
        print('missing', p); return None
    with open(p) as f:
        return json.load(f)


def main():
    out = {}
    kb = load('c1_kv_buffer.json')
    if kb:
        s = kb['settings']['KV 256/512 P=768']
        nw, tok, ppl = s['nll_per_window'], s['tokens_per_window'], s['ppl']
        e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
        out['kv_buffer'] = {'windows': s['windows'], 'ppl': ppl, 'excess': e,
                            'excess_ci': {m: excess_ci(nw[m], nw['dense'], tok) for m in nw if m != 'dense'}}
        ci = ppl_ratio(nw['update / buffer B=32'], nw['static / buffer B=32'], tok)
        out['kv_buffer']['both_over_buffer_only'] = ci
        out['kv_buffer']['pairs'] = {f'{A} | {B}': ppl_ratio(nw[A], nw[B], tok) for A, B in (
            ('update / no buffer', 'static / no buffer'), ('static / buffer B=32', 'static / no buffer'),
            ('update / buffer B=32', 'update / no buffer'), ('offline / buffer B=32', 'static / buffer B=32'))}
        out['decision'] = {
            'Q1_both_le_0.5pct': e['update / buffer B=32'] <= 0.005,
            'Q2_order': e['update / buffer B=32'] < e['static / buffer B=32'] < e['update / no buffer'] < e['static / no buffer'],
            'Q3_both_over_buffer_CI_below_1': ci['ci_hi'] < 1}
    ku = load('c1_kv_update.json')
    if ku:
        s = ku['settings']['KV 256/512 P=768 B=32']
        nw, tok = s['nll_per_window'], s['tokens_per_window']
        out['kv_update_Q4'] = {f'{A} | update': ppl_ratio(nw[A], nw['update'], tok)
                               for A in ('update decay', 'update no-reproj', 'oracle refit', 'rsvd q=1 + update') if A in nw}
        out['kv_update_ppl'] = s['ppl']
    ka = load('c1_kv_attnbasis.json')
    if ka:
        s = ka['settings']['KV 256/512 P=768']
        nw, tok = s['nll_per_window'], s['tokens_per_window']
        out['kv_attnbasis_Q4'] = {f'attn-late / {b} | online / {b}': ppl_ratio(nw[f'attn-late / {b}'], nw[f'online / {b}'], tok)
                                  for b in ('no buffer', 'buffer')}
        out['kv_attnbasis_ppl'] = s['ppl']
    k0 = load('c1b_kv_buffer_skip0.json')
    if k0:
        s = k0['settings']['KV 256/512 P=768']
        out['C1b_offline_skip0_windows_0_15'] = {'ppl': s['ppl'], 'reported_offline_no_buffer': 25.368, 'reported_offline_buffer32': 25.022}
    with open(os.path.join(RR, 'c1_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out.get('decision', {}), indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'kv_buffer'}, indent=1)[:4000])


if __name__ == '__main__':
    main()
