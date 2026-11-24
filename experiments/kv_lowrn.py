"""C14c (wave 3, 2026-11-24): GPT-2 at a LOW rank-to-prompt ratio R/n. Does attention weighting of the prompt basis start to
help on GPT-2 when R/n is as small as on SmolLM2?

Why. C11/C13 found that on SmolLM2-135M (RoPE, P = 2048, K64/V128 of 192 channels: R/n = 0.031 keys, 0.063 values) the
attention-weighted prompt basis HELPS (attn-late / buffer 32 removes 67% of the static/buffer excess), while on GPT-2
(K256/V512 at P = 768: R/n = 0.33 / 0.67) it does not (kv_attnbasis.py, C1). Candidate explanation: R is far below n, so
the basis must choose which directions to keep and the attention weights say which rows matter. This script puts GPT-2
at K 64 / V 128 (R/n = 0.083 / 0.167 at P = 768, the same R/n as SmolLM2 at P = 768) with the SAME recipe as kv_buffer.py
(one basis per layer for all 12 heads, G_K from prompt queries, G_V = W_O W_O^T, prompt mean, first token exact, B = 32,
256 continuation tokens, 255 scored predictions), 16 windows 0-15. Rank 64/128 of the 768-dim rows = 1/12 + 1/6 of the
2 x 768 numbers per token = 0.125 of the stored codes (not half).
Methods (names as in kv_rope_port.py): dense; static / no buffer; static / buffer 32; update / no buffer;
update / buffer 32; attn-late / no buffer; attn-late / buffer 32; attn-late+update / buffer 32 (the attn-late weighted S, U
combined with the block update [S U^T ; A_blk]; new block rows unweighted, as in C13). attn-late = prompt rows scaled by
sqrt of the attention received from the last 128 prompt queries (head mean, / mean), as kv_attnbasis.py.
Code checks: with --RK 256 --RV 512 the cells static / no buffer, static / buffer 32, update / buffer 32 reproduce
kv_buffer.py (window 0: smoke/patched_kv_buffer_w1.json; 16 windows 25.775 / 24.997 / 24.778).

Rules fixed before any full run (e(x) = ppl(x)/dense - 1, 16 windows 0-15; evaluated by c14_analyze.py):
  L1 weighting helps at low R/n (GPT-2, K64/V128): e(attn-late / buffer 32) <= 0.9 x e(static / buffer 32).
  Descriptive (no rule): the same for the no-buffer cells; e(update / buffer 32) / e(static / buffer 32); paired bootstrap
  CIs (stats_utils.ppl_ratio) of attn-late / buffer vs static / buffer and of attn-late+update vs update.
(The C14 rules G1 and K1 on SmolLM2 are in c14_analyze.py; they use kv_rope_port.py runs.)
Run:  .venv\\Scripts\\python.exe experiments\\kv_lowrn.py --threads 8 --out experiments\\rerun_2026-11-22\\c14c_kv_lowrn.json
      smoke: --windows 1 --RK 256 --RV 512 --only static update --out <scratch>.json
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, E, D64, ROOT, prompt_state, kinds_of, run_block, base_result, ppl_table, save_json,  # noqa: E402
                   rss_mb)
from kv_attnbasis import attention, LATE  # noqa: E402

P, B, SEQ = 768, 32, 1024
METHODS = ['dense', 'static / no buffer', 'static / buffer 32', 'update / no buffer', 'update / buffer 32',
           'attn-late / no buffer', 'attn-late / buffer 32', 'attn-late+update / buffer 32']


def attn_late_weights(pre):
    wl = {}
    for l in range(L):
        At = attention(pre[l])
        c = At[:, P - LATE:, :].sum(1).mean(0)[1:]
        wl[l] = c / c.mean()
    return wl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--RK', type=int, default=64)
    ap.add_argument('--RV', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c14c_kv_lowrn.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    kinds = kinds_of(a.RK, a.RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    tok = SEQ - P - 1
    res = base_result(a, {'P': P, 'RK': a.RK, 'RV': a.RV, 'B': B, 'windows': win_idx, 'tokens_per_window': tok,
                          'R_over_n': {'K': a.RK / (P - 1), 'V': a.RV / (P - 1)}})
    nllw = {m: [] for m in methods}
    tw = {m: [] for m in methods}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        wl = attn_late_weights(pre) if any(m.startswith('attn-late') for m in methods) else None
        for m in methods:
            t1 = time.perf_counter()
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B)
            else:
                st = prompt_state(ctx, pre, kinds, row_weights=wl if m.startswith('attn-late') else None)
                buf = m.endswith('buffer 32') and 'no buffer' not in m
                upd = 'update' in m
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=buf, upd=upd)
            nllw[m].append(v); tw[m].append(time.perf_counter() - t1)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / tok):.3f}' for m in methods), flush=True)
    ppl = ppl_table(nllw, tok)
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e, 'timing_s': tw})
    dec = {}
    try:
        dec['L1_attnlate_buf_le_0.9_static_buf'] = e['attn-late / buffer 32'] <= 0.9 * e['static / buffer 32']
    except KeyError as ex:
        dec['incomplete'] = f'missing {ex}'
    res['decision'] = dec
    if a.windows == 1 and a.window_offset == 0 and (a.RK, a.RV) == (256, 512):
        ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
        if os.path.exists(ref):
            import json
            kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
            res['code_checks_window0'] = {'dense': (ppl['dense'], kb['dense']),
                                          'static / no buffer': (ppl.get('static / no buffer'), kb['static / no buffer']),
                                          'static / buffer 32': (ppl.get('static / buffer 32'), kb['static / buffer B=32']),
                                          'update / buffer 32': (ppl.get('update / buffer 32'), kb['update / buffer B=32'])}
            print('code checks (this script, kv_buffer.py), window 0:', res['code_checks_window0'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('ppl:', {k: round(v, 4) for k, v in ppl.items()})
    print('excess %:', {k: round(100 * v, 3) for k, v in e.items()})
    print('seconds per window and method:', {k: round(float(np.mean(v)), 1) for k, v in tw.items()})
    print('decision:', dec)


if __name__ == '__main__':
    main()
