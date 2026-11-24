"""C18a (wave 3, 2026-11-24): sliding window versus block buffer at an EQUAL average number of exact tokens, with per-step NLL.

Why. C2 found static / sliding 32 (+0.77%) better than static / block 32 (+1.09%), but a block buffer of size B has an exact
window of only 1..B tokens (average (B+1)/2, "effective exact window ~ B/2"), a sliding window of 32 always has 32. The fair
comparison is at equal AVERAGE exact count: sliding 16 (16 exact tokens at every step: own + 15 previous) versus block 32
(1..32 exact tokens, average 16.5), and sliding 32 versus block 64 (average 32.5). Setting = kv_synergy.py main setting
(static prompt basis, K256/V512, P = 768, 256 continuation tokens, first token exact, windows 0-15).
Methods (kvlab): dense; static / own only (run_tokens window 1 = C2 'B=1'); static / sliding 16; static / sliding 32 (C2);
static / block 16; static / block 32 (C2 / kv_buffer); static / block 64 (run_block B, buffer on, no update).
Per-step NLL: step s (0..254) is the prediction of continuation token P+s+1; stored per window for every method
(nll_per_step), plus the window sums over all 255 steps (nll_per_window) and over steps >= 63 (nll_from63; 192 steps, so that
a ramp of 62 steps, e.g. of the C12 'lagged' control, is excluded; the same steps are scored for every method).
Code checks (--smoke, window 0): own only, sliding 32 and block 32 against c2_kv_synergy.json ('static / B=1',
'static / sliding 32', 'static / block 32'), and block 32 against kv_buffer.py's patched smoke JSON.
Rules fixed before any full run (e(x) = ppl(x)/dense - 1 with ppl over the SCORED steps; windows 0-15; evaluated by
c18_analyze.py; the primary scoring is steps >= 63, the all-steps version is reported next to it):
  E1 equal budget 16: e(sliding 16) <= e(block 32)   (sliding no worse at equal average exact count).
  E2 equal budget 32: e(sliding 32) <= e(block 64).
  Z3 (C12 recency controls with the ramp removed, kv_control.py --per_step; uses this script's own only, sliding 32 and dense):
     e(lagged 31 + own) <= (e(own only) + e(sliding 32)) / 2, all scored from step 63.
  Descriptive: paired bootstrap CIs (stats_utils.ppl_ratio on the scored NLL sums) for sliding 16 / block 32 and
  sliding 32 / block 64; the whole ordering of the 6 compressed cells.
Run:  .venv\\Scripts\\python.exe experiments\\kv_equalcount.py --threads 8 --out experiments\\rerun_2026-11-22\\c18a_kv_equalcount.json
      smoke: --windows 1 --smoke --out <scratch>.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, E, D64, ROOT, prompt_state, kinds_of, run_block, run_tokens, base_result, save_json,  # noqa: E402
                   rss_mb)

P, RK, RV, SEQ, FROM = 768, 256, 512, 1024, 63
METHODS = ['dense', 'static / own only', 'static / sliding 16', 'static / sliding 32', 'static / block 16',
           'static / block 32', 'static / block 64']


def step_nll(lg, w):
    lp = torch.log_softmax(lg[:-1].double(), -1)
    tgt = w[P + 1:]
    return (-lp[torch.arange(tgt.numel()), tgt]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c18a_kv_equalcount.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    tok = SEQ - P - 1
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'windows': win_idx, 'tokens_per_window': tok,
                          'tokens_from63': tok - FROM, 'from_step': FROM, 'avg_exact_tokens':
                          {'static / own only': 1, 'static / sliding 16': 16, 'static / sliding 32': 32,
                           'static / block 16': 8.5, 'static / block 32': 16.5, 'static / block 64': 32.5}})
    steps = {m: [] for m in methods}
    tw = {m: [] for m in methods}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        st = prompt_state(ctx, pre, kinds)
        for m in methods:
            t1 = time.perf_counter()
            if m == 'dense':
                _, lg = run_block(ctx, w, P, kinds, None, B=32, keep_logits=True)
            elif m == 'static / own only':
                _, lg = run_tokens(ctx, w, P, kinds, st, window=1, keep_logits=True)
            elif m.startswith('static / sliding'):
                _, lg = run_tokens(ctx, w, P, kinds, st, window=int(m.split()[-1]), keep_logits=True)
            else:
                _, lg = run_block(ctx, w, P, kinds, st, B=int(m.split()[-1]), buf=True, upd=False, keep_logits=True)
            steps[m].append(step_nll(lg, w)); tw[m].append(time.perf_counter() - t1)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(sum(steps[m][-1]) / tok):.3f}' for m in methods), flush=True)
    sums_all = {m: [float(sum(v)) for v in steps[m]] for m in methods}
    sums63 = {m: [float(sum(v[FROM:])) for v in steps[m]] for m in methods}
    ppl_all = {m: math.exp(sum(sums_all[m]) / (tok * len(win_idx))) for m in methods}
    ppl63 = {m: math.exp(sum(sums63[m]) / ((tok - FROM) * len(win_idx))) for m in methods}
    e_all = {m: ppl_all[m] / ppl_all['dense'] - 1 for m in methods}
    e63 = {m: ppl63[m] / ppl63['dense'] - 1 for m in methods}
    res.update({'nll_per_step': steps, 'nll_per_window': sums_all, 'nll_from63': sums63, 'ppl': ppl_all, 'excess': e_all,
                'ppl_from63': ppl63, 'excess_from63': e63, 'timing_s': tw})
    dec = {}
    try:
        dec['E1_sliding16_le_block32_from63'] = e63['static / sliding 16'] <= e63['static / block 32']
        dec['E1_all_steps'] = e_all['static / sliding 16'] <= e_all['static / block 32']
        dec['E2_sliding32_le_block64_from63'] = e63['static / sliding 32'] <= e63['static / block 64']
        dec['E2_all_steps'] = e_all['static / sliding 32'] <= e_all['static / block 64']
    except KeyError as ex:
        dec['incomplete'] = f'missing {ex}'
    res['decision'] = dec
    c2p = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c2_kv_synergy.json')
    if a.smoke and os.path.exists(c2p) and win_idx == [0]:
        c2 = json.load(open(c2p))['nll_per_window']
        pair = {'static / own only': 'static / B=1', 'static / sliding 32': 'static / sliding 32', 'static / block 32': 'static / block 32',
                'dense': 'dense'}
        res['code_check_vs_c2_window0'] = {m: (sums_all[m][0], c2[c][0]) for m, c in pair.items() if m in sums_all}
        print('code checks vs c2 (window 0 NLL sums):', res['code_check_vs_c2_window0'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('excess % all steps:', {k: round(100 * v, 3) for k, v in e_all.items()})
    print('excess % from step 63:', {k: round(100 * v, 3) for k, v in e63.items()})
    print('seconds per window and method:', {k: round(float(np.mean(v)), 1) for k, v in tw.items()})
    print('decision:', dec)


if __name__ == '__main__':
    main()
