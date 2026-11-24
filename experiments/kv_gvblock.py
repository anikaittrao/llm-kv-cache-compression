"""C8 (audit 2026-11-22): is the shared output-aware value weight G_V = W_O W_O^T the right one for a basis shared by
all 12 heads?

Run:  .venv\\Scripts\\python.exe experiments\\kv_gvblock.py --out experiments\\rerun_2026-11-22\\c8_kv_gvblock.json
            (about 25 min: calibration 80 s + part (a) 130 s + ~70 s per window of part (b))      smoke test: --windows 1 --eval_windows 1

Background. For one head, the value error e_h reaches the output as e_h W_O^h, so G = W_O^h W_O^h^T is exact. For one
basis shared by all heads, the full G_V = W_O W_O^T (768 x 768) also weights the cross-head blocks W_O^h W_O^g^T, which
is right only if all heads attend alike (the output is sum_h a_h e_h W_O^h with different attention a_h per head).
The alternative that does not assume equal attention is the block-diagonal G_V = blockdiag_h(W_O^h W_O^h^T).
(a) Option 2 offline (Section 5.5, kv_adaptive.py protocol: calibration of 16 train windows, 8 test windows, full
    passes with every position compressed): 'V joint output-aware uniform R=512' (reported 34.287) and
    'KV joint aware uniform R=256/512' (reported 34.979), full G_V vs block-diagonal G_V.
(b) Section 8 main setting (kv_buffer.py, P = 768, K 256 / V 512, 16 windows): update / buffer 32 (reported 24.778),
    full vs block-diagonal G_V (prompt basis, prompt G_K for keys in both).
Rule fixed before the run:
  G1 : |ppl(block-diagonal) / ppl(full) - 1| <= 0.003 in (a) V512 and in (b) => keep the reported recipe, with the
       caveat in the text. Otherwise the block-diagonal weight is reported as the better/worse choice.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, Pi, psd_pow, qblock, prompt_state, kinds_of, run_block,  # noqa: E402
                   base_result, ppl_table, save_json, rss_mb)

P, RK, RV, B = 768, 256, 512, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--eval_windows', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c8_kv_gvblock.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    ctx.offline()
    cal = ctx.calib; n = cal['n']
    print(f'calibration ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B})

    # ---- (a) Option 2 offline, full-window protocol of kv_adaptive.py
    muK = [cal['S1']['K'][l] / n for l in range(L)]; muV = [cal['S1']['V'][l] / n for l in range(L)]
    CK = [cal['S2']['K'][l] / n - torch.outer(muK[l], muK[l]) for l in range(L)]
    CV = [cal['S2']['V'][l] / n - torch.outer(muV[l], muV[l]) for l in range(L)]
    projK = []
    for l in range(L):
        G = qblock(cal['S2']['Q'][l] / n); T, Ti = psd_pow(G, 0.5), psd_pow(G, -0.5)
        U = torch.linalg.eigh(T @ CK[l] @ T)[1].flip(1)[:, :RK]
        projK.append((muK[l].float(), (T @ U @ U.T @ Ti).float()))

    def projV(gv):
        out = []
        for l in range(L):
            G = ctx.WO[l] @ ctx.WO[l].T
            if gv == 'blockdiag':
                G = torch.block_diag(*[G[h * Dh:(h + 1) * Dh, h * Dh:(h + 1) * Dh] for h in range(H)])
            T, Ti = psd_pow(G, 0.5), psd_pow(G, -0.5)
            U = torch.linalg.eigh(T @ CV[l] @ T)[1].flip(1)[:, :RV]
            out.append((muV[l].float(), (T @ U @ U.T @ Ti).float()))
        return out
    evalw = [ctx.ids_test[i * 1024:(i + 1) * 1024] for i in range(a.eval_windows)]

    def evaluate(proj):
        ctx.mode['proj'] = proj
        nll, cnt = 0.0, 0
        with torch.inference_mode():
            for w in evalw:
                out = ctx.model(w[None], labels=w[None]); k_ = w.numel() - 1
                nll += float(out.loss) * k_; cnt += k_
        ctx.mode['proj'] = None
        return math.exp(nll / cnt)
    pa = {'dense': evaluate(None)}
    for gv in ('full', 'blockdiag'):
        pv = projV(gv)
        pa[f'V512 offline {gv}'] = evaluate({'V': pv})
        pa[f'K256+V512 offline {gv}'] = evaluate({'K': projK, 'V': pv})
        print(gv, {k: round(v, 3) for k, v in pa.items()}, f'({time.perf_counter()-T0:.0f}s)', flush=True)
    res['part_a_ppl'] = pa
    if a.eval_windows == 8:
        res['part_a_code_checks'] = {'V512 offline full': (pa['V512 offline full'], 34.2866918547425),
                                     'K256+V512 offline full': (pa['K256+V512 offline full'], 34.97891862168307)}

    # ---- (b) update + buffer 32, prompt basis
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    nllw = {'dense': [], 'update / buffer 32 full G_V': [], 'update / buffer 32 blockdiag G_V': []}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        nllw['dense'].append(run_block(ctx, w, P, kinds, None, B=B)[0])
        for gv in ('full', 'blockdiag'):
            ctx.set_gv(gv)
            st = prompt_state(ctx, pre, kinds)
            nllw[f'update / buffer 32 {gv} G_V'].append(run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=True)[0])
        ctx.set_gv('full')
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s)', {k: round(math.exp(v[-1] / (1024 - P - 1)), 3) for k, v in nllw.items()}, flush=True)
    tok = 1024 - P - 1
    pb = ppl_table(nllw, tok)
    res.update({'windows': win_idx, 'tokens_per_window': tok, 'nll_per_window': nllw, 'part_b_ppl': pb})
    ra = pa['V512 offline blockdiag'] / pa['V512 offline full']
    rb = pb['update / buffer 32 blockdiag G_V'] / pb['update / buffer 32 full G_V']
    res['decision'] = {'ratio_a_V512': ra, 'ratio_a_K256V512': pa['K256+V512 offline blockdiag'] / pa['K256+V512 offline full'],
                       'ratio_b_update_buffer': rb, 'G1_within_0.3pct': abs(ra - 1) <= 0.003 and abs(rb - 1) <= 0.003}
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('part a:', pa); print('part b:', pb); print('decision:', res['decision'])


if __name__ == '__main__':
    main()
