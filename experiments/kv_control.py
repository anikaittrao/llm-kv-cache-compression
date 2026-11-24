"""C12 (follow-up to C2, 2026-11-23): recency versus novelty control for the exact buffer.

C2 left open whether the exact NEWEST tokens help because they are recent or because they are NEW to the basis
(continuation tokens, residual 2.5x K / 6.2x V of the prompt's, C3) while the random / top-attended exact tokens of C2
were PROMPT tokens. Control: exact OLD continuation tokens (continuation tokens that are NOT among the newest).
Setting = kv_synergy.py main setting (static prompt basis, K256/V512, P=768, 256 continuation tokens, token by token,
first token exact, own token exact while processed = B=1 baseline). Extra exact tokens at the step of query t:
  lagged 31 + own        the continuation tokens of age 32..62 (t-62 .. t-32), i.e. old but new-to-the-basis
  random old 31 + own    31 tokens drawn at random (fresh draw per step, seed = window) from continuation ages >= 32
  own only (check)       no extra tokens; must reproduce c2 'static / B=1' (code check, --smoke)
Exact rows are restored from an exact copy (a control, not a realistic cache). For BOTH controls no extra token exists
before t = P+32 and fewer than 31 exist before t = P+62 (corrected 2026-11-23; the random-old pool np.arange(P, t-31)
also needs 62 steps); this budget ramp is a confound against the controls (disclosed).
Reference values (same windows 0-15, same dense) come from c2_kv_synergy.json: e(B=1), e(sliding 32), e(random prompt).
Rules fixed before the run, e(x) = ppl(x)/dense - 1, h = (e(B=1) + e(sliding 32))/2:
  Z1 lagged     : e(lagged 31 + own) <= h   (old continuation tokens recover at least half of the sliding-window gain
                                              => novelty of the tokens, not recency, explains the buffer)
  Z2 random old : e(random old 31 + own) <= h
If both fail the gain needs recency (or the ramp confound hides it); descriptive: paired bootstrap ratios vs B=1 and sliding.
Wave 3 (C18a, 2026-11-24): option --per_step also stores nll_per_step[method][window] (255 values) so that the ramp
confound can be removed by scoring from continuation step 63 on; rule Z3 (c18_analyze.py): e(lagged 31 + own) <=
(e(own only) + e(sliding 32)) / 2, all scored from step 63 (own / sliding 32 / dense from kv_equalcount.py, same windows).
Run: .venv\\Scripts\\python.exe experiments\\kv_control.py --out experiments\\rerun_2026-11-22\\c12_kv_control.json
     smoke: add --windows 1 --smoke --out <scratch>.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, prompt_state, kinds_of, get_rows, nll_of, Pi, base_result,  # noqa: E402
                   ppl_table, save_json, rss_mb)

P, RK, RV = 768, 256, 512
METHODS = ['own only (check)', 'lagged 31 + own', 'random old 31 + own']


def run_control(ctx, w, P, kinds, st, variant, seed, steps=False):
    """steps=True (wave 3, C18a) additionally returns the per-step NLL vector (step s predicts token P+s+1); default
    behaviour and output unchanged."""
    seq = w.numel()
    m = ctx.model
    Pm = {(kd, l): Pi(st[(kd, l)]) for kd, _, _, _ in kinds for l in range(L)}
    rng = np.random.default_rng(seed)
    ex, cp = {}, {}

    def put(past, l, attr, j, x):
        getattr(past.layers[l], attr)[0, :, j, :] = x.float().reshape(H, Dh)

    with torch.inference_mode():
        ctx.mode['proj'] = None
        past = m(w[:P][None], use_cache=True).past_key_values
        for kd, R, attr, _ in kinds:
            for l in range(L):
                X = get_rows(past, l, attr, P)
                s = st[(kd, l)]
                ex[(kd, l)] = torch.zeros(seq, E, dtype=D64)
                cp[(kd, l)] = torch.zeros(seq, E, dtype=D64)
                ex[(kd, l)][:P] = X
                cp[(kd, l)][:P] = s['mu'] + (X - s['mu']) @ Pm[(kd, l)]
                cp[(kd, l)][0] = X[0]
                for j in range(1, P):
                    put(past, l, attr, j, cp[(kd, l)][j])
        cur = {0}  # positions currently exact in the cache (below the query)
        logits = []
        for t in range(P, seq):
            if variant == 'own only (check)':
                S = set()
            elif variant == 'lagged 31 + own':
                S = set(range(max(P, t - 62), t - 31)) if t - 32 >= P else set()
            else:
                pool = np.arange(P, t - 31) if t - 32 >= P else np.array([], dtype=int)
                S = set(rng.choice(pool, 31, replace=False).tolist()) if len(pool) > 31 else set(pool.tolist())
            target = {0} | S
            for j in cur - target:
                if j >= 1:
                    for kd, R, attr, _ in kinds:
                        for l in range(L):
                            put(past, l, attr, j, cp[(kd, l)][j])
            for j in target - cur:
                for kd, R, attr, _ in kinds:
                    for l in range(L):
                        put(past, l, attr, j, ex[(kd, l)][j])
            cur = set(target)
            out = m(w[t:t + 1][None], past_key_values=past)
            logits.append(out.logits[0])
            past = out.past_key_values
            for kd, R, attr, _ in kinds:           # row t is exact now; store it and its compressed version
                for l in range(L):
                    x = get_rows(past, l, attr, t + 1)[t]
                    s = st[(kd, l)]
                    ex[(kd, l)][t] = x
                    cp[(kd, l)][t] = s['mu'] + (x - s['mu']) @ Pm[(kd, l)]
            cur.add(t)
        lg = torch.cat(logits)
    if steps:
        lp = torch.log_softmax(lg[:-1].double(), -1)
        tgt = w[P + 1:]
        return nll_of(lg, w, P), (-lp[torch.arange(tgt.numel()), tgt]).tolist()
    return nll_of(lg, w, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--per_step', action='store_true', help='C18a: also store nll_per_step[method][window] (255 values)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c12_kv_control.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [x for x in METHODS if not a.only or any(o in x for o in a.only)]
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'windows': win_idx, 'tokens_per_window': 1024 - P - 1})
    nllw = {x: [] for x in methods}
    steps_ = {x: [] for x in methods}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        st = prompt_state(ctx, pre, kinds)
        for x in methods:
            if a.per_step:
                v, vec = run_control(ctx, w, P, kinds, st, x, wi, steps=True)
                nllw[x].append(v); steps_[x].append(vec)
            else:
                nllw[x].append(run_control(ctx, w, P, kinds, st, x, wi))
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{x}: {math.exp(nllw[x][-1] / (1024 - P - 1)):.3f}' for x in methods), flush=True)
    tok = 1024 - P - 1
    ppl = ppl_table(nllw, tok)
    res.update({'nll_per_window': nllw, 'ppl': ppl})
    if a.per_step:
        res['nll_per_step'] = steps_
    c2p = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c2_kv_synergy.json')
    if os.path.exists(c2p):
        c2 = json.load(open(c2p))
        if a.smoke and 'own only (check)' in nllw and win_idx == [0]:
            print('check own-only vs c2 B=1, window 0 NLL:', nllw['own only (check)'][0],
                  c2['nll_per_window']['static / B=1'][0])
        if c2['windows'][:len(win_idx)] == win_idx and len(win_idx) == 16:
            dense = c2['ppl']['dense']
            e = {x: ppl[x] / dense - 1 for x in ppl}
            h = (c2['excess']['static / B=1'] + c2['excess']['static / sliding 32']) / 2
            res['excess'] = e
            res['threshold_h'] = h
            res['reference_excess'] = {k: c2['excess'][k] for k in
                                       ('static / B=1', 'static / sliding 32', 'static / random 31 + own',
                                        'static / block 32')}
            dec = {}
            if 'lagged 31 + own' in e:
                dec['Z1_lagged_le_h'] = e['lagged 31 + own'] <= h
            if 'random old 31 + own' in e:
                dec['Z2_random_old_le_h'] = e['random old 31 + own'] <= h
            res['decision'] = dec
            print('excess %:', {k: round(100 * v, 3) for k, v in e.items()}, 'h %:', round(100 * h, 3), dec)
    res['total_runtime_s'] = time.perf_counter() - T0
    save_json(a.out, res)


if __name__ == '__main__':
    main()
