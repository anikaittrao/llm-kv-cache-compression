"""C4 (audit 2026-11-22): the half-cache result when most cached tokens are NEW (prompt of 256 tokens, 768 continued).

Run:  .venv\\Scripts\\python.exe experiments\\kv_p256.py --out experiments\\rerun_2026-11-22\\c4_kv_p256.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 55 min: calibration 75 s + ~200 s per window, smoke 2026-11-22)
      smoke test: --windows 1      --replot: redraw the figure from the JSON.

Why. In the reported main setting (P = 768) 768 of the 1024 cached tokens are the prompt that the online basis was fitted
to, so they are compressed in-sample; only 256 tokens are new, and the value rank 512 was fitted on 767 rows. Here the
prompt is P = 256 tokens and the continuation 768 tokens, so 3/4 of the cache is new text.
Setting: kv_buffer.py recipe (one basis per layer for all 12 heads, keys query-aware with the prompt's G_K, values
output-aware, centered, first token exact), K 256 / V 512 = half of the stored codes, buffer B = 32 (block buffer) for
every method. The prompt has n = 255 usable rows (positions 1..255), so the online basis has rank min(R, n - 1) = 254 at
the start (the 255th centered direction is numerically zero); the update raises the rank toward R = 256 / 512 as rows
arrive (the sketch [S U^T ; A_blk] has 254 + 32 rows after one block). Memory is counted at the nominal R (codes
padded), see experiments/kv_memory.py.
Methods: dense; static (prompt basis, never updated); update (incremental SVD every block, old rows re-projected, as
kv_update.py 'update'); offline (calibration basis, the method of Section 5.5); offline + update (calibration basis,
then the same updates, S^2 = calibration eigenvalues x (P - 1)).
Rules fixed before the run, e(x) = ppl(x)/dense - 1, 16 windows:
  P1 quality      : e(update) <= 0.01.
  P2 beats offline: ppl(update) <= ppl(offline).
Descriptive: static and offline + update.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, ROOT, prompt_state, offline_state, kinds_of, run_block, base_result, ppl_table,  # noqa: E402
                   save_json, rss_mb)

RK, RV, B = 256, 512, 32
METHODS = ['dense', 'static', 'update', 'offline', 'offline + update']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--P', type=int, default=256)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c4_kv_p256.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    P = a.P
    ctx = Ctx(a.threads)
    off = ctx.offline()
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    tok = 1024 - P - 1
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'windows': win_idx, 'tokens_per_window': tok,
                          'initial_online_rank': min(RK, P - 2), 'final_ranks': []})
    nllw = {m: [] for m in methods}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        for m in methods:
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B)
            elif m in ('static', 'update'):
                st = prompt_state(ctx, pre, kinds, rank_cap=P - 2)
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=(m == 'update'))
                res['final_ranks'].append({'window': wi, 'method': m, 'K': int(st[('K', 0)]['U'].shape[1]), 'V': int(st[('V', 0)]['U'].shape[1])})
            elif m == 'offline':
                v, _ = run_block(ctx, w, P, kinds, offline_state(off, kinds), B=B, buf=True, upd=False)
            elif m == 'offline + update':
                v, _ = run_block(ctx, w, P, kinds, offline_state(off, kinds, n_rows_equiv=P - 1), B=B, buf=True, upd=True)
            nllw[m].append(v)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / tok):.3f}' for m in methods), flush=True)
    ppl = ppl_table(nllw, tok)
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e})
    d = {}
    try:
        d['P1_update_within_1pct'] = e['update'] <= 0.01
        d['P2_update_beats_offline'] = ppl['update'] <= ppl['offline']
    except KeyError as ex:
        d['incomplete'] = f'missing {ex}'
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('ppl:', {k: round(v, 4) for k, v in ppl.items()})
    print('excess %:', {k: round(100 * v, 3) for k, v in e.items()})
    print('final ranks (layer 0):', res['final_ranks'][-2:])
    print('decision:', d)
    try:
        make_figures(a, res)
    except Exception as ex:
        print('figure skipped:', ex)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    os.makedirs(a.figdir, exist_ok=True)
    e = res['excess']
    ms = [m for m in METHODS if m in e and m != 'dense']
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar(range(len(ms)), [100 * e[m] for m in ms], color=dv.C['blue'])
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, fontsize=7)
    ax.set_ylabel('perplexity increase (%)'); ax.set_title(f"half cache, P = {res['P']}, buffer 32", fontsize=8)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c4_kv_p256.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
