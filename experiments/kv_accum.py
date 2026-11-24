"""C17 (wave 3, 2026-11-24): AttSVD's "accumulating" mode on GPT-2. Does attention weighting of the prompt basis help when the
continuation tokens are NEVER compressed?

Why. kv_attnbasis.py / C2 found that, when new tokens are compressed (static, buffer <= 32), 93-99% of the attention-read
error sits on continuation tokens, so weighting the PROMPT rows cannot help (W1-W3 FAIL). The AttSVD paper's main mode keeps
new tokens exact; the audit predicted that weighting does help THERE. Test: the prompt (P tokens, prefilled exactly) is
compressed once with a static basis; every continuation token stays exact (one block that covers the whole continuation:
kvlab.run_block with B = seq - P, buf = True, upd = False; no later compression, so the basis is never updated and no new
token is ever compressed). Perplexity on the continuation predictions.
Bases: 'unweighted' = online prompt basis (kv_buffer.py recipe: G_K from prompt queries, G_V = W_O W_O^T, prompt mean, first
token exact, rows 1..P-1); 'attn-late' = the same with the prompt rows scaled by sqrt of the attention received from the last
128 prompt queries (head mean, / mean; kv_attnbasis.py 'attn-late'). One basis per layer for all 12 heads.
Settings: main = P 768, K256/V512, 1024 - 768 = 256 continuation tokens (255 predictions); keys-only = P 384, K 256, 640
continuation tokens (639 predictions; the second setting of kv_attnbasis.py). Windows 0-15, dense reference.
Code checks (--smoke): kvlab.run_block static / buffer 32 = 24.997 / window-0 value of kv_buffer.py's patched smoke JSON
(the accumulating run reuses the same code path with B = seq - P).
Rules fixed before any full run (e(x) = ppl(x)/dense - 1; evaluated by c17_analyze.py):
  A1 weighting helps when new tokens are exact (main setting, P 768): e(attn-late, accumulating) <= 0.9 x
     e(unweighted, accumulating).
  Descriptive (no rule): the same ratio at P = 384 keys only; paired bootstrap CI (stats_utils) of ppl(attn-late) /
  ppl(unweighted) in both settings; e of both against the buffered/compressed references of C1 (static / buffer 32 +1.09%).
Run:  .venv\\Scripts\\python.exe experiments\\kv_accum.py --threads 8 --out experiments\\rerun_2026-11-22\\c17_kv_accum.json
      smoke: --windows 1 --smoke --out <scratch>.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, D64, ROOT, prompt_state, kinds_of, run_block, base_result, save_json, rss_mb)  # noqa: E402
from kv_attnbasis import attention, LATE  # noqa: E402

SEQ = 1024
SETTINGS = {'KV256/512 P=768': (768, 256, 512), 'K256 P=384': (384, 256, 0)}
METHODS = ['dense', 'unweighted accumulating', 'attn-late accumulating']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--settings', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c17_kv_accum.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    res = base_result(a, {'windows': win_idx, 'settings': {}})
    for name, (P, RK, RV) in SETTINGS.items():
        if a.settings and name not in a.settings:
            continue
        kinds = kinds_of(RK, RV)
        tok = SEQ - P - 1
        nllw = {m: [] for m in METHODS + (['static / buffer 32 (check)'] if a.smoke else [])}
        tw = {m: [] for m in nllw}
        for wi, w in zip(win_idx, wins):
            pre = ctx.prefill_stats(w, P)
            wl = {}
            for l in range(L):
                c = attention(pre[l])[:, P - LATE:, :].sum(1).mean(0)[1:]
                wl[l] = c / c.mean()
            for m in nllw:
                t1 = time.perf_counter()
                if m == 'dense':
                    v, _ = run_block(ctx, w, P, kinds, None, B=SEQ - P)
                elif m == 'static / buffer 32 (check)':
                    v, _ = run_block(ctx, w, P, kinds, prompt_state(ctx, pre, kinds), B=32, buf=True, upd=False)
                else:
                    st = prompt_state(ctx, pre, kinds, row_weights=wl if m.startswith('attn-late') else None)
                    v, _ = run_block(ctx, w, P, kinds, st, B=SEQ - P, buf=True, upd=False)
                nllw[m].append(v); tw[m].append(time.perf_counter() - t1)
            print(f'[{name}] window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
                  ' '.join(f'{m}: {math.exp(nllw[m][-1] / tok):.3f}' for m in nllw), flush=True)
        ppl = {m: math.exp(sum(v) / (tok * len(v))) for m, v in nllw.items()}
        e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
        s = {'P': P, 'RK': RK, 'RV': RV, 'tokens_per_window': tok, 'nll_per_window': nllw, 'ppl': ppl, 'excess': e,
             'timing_s': tw}
        if a.smoke and name == 'KV256/512 P=768' and win_idx == [0]:
            ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
            if os.path.exists(ref):
                kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
                s['code_check_window0'] = {'static / buffer 32': (ppl['static / buffer 32 (check)'], kb['static / buffer B=32']),
                                           'dense': (ppl['dense'], kb['dense'])}
                print('code checks:', s['code_check_window0'])
        res['settings'][name] = s
        print(name, 'ppl', {k: round(v, 4) for k, v in ppl.items()}, 'excess %',
              {k: round(100 * v, 3) for k, v in e.items()}, flush=True)
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)


if __name__ == '__main__':
    main()
