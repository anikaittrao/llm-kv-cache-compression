"""C18b (wave 3, 2026-11-24): is the held-out-prompt gap of C3 a temporal-drift effect? Interleaved versus contiguous split.

Why. C3 (kv_novelty.py) fitted the basis on the FIRST 383 prompt rows (rank scaled to keep R/n) and measured the residual
ratio on the held-out SECOND half of the prompt (rows 384..767): K 3.21 / V 5.60, about the continuation's 4.27 / 5.88, and
concluded "ordinary out-of-sample error, not novelty". A sceptic: the held-out rows are LATER text than the fit rows, so
slow drift inside the prompt could mimic out-of-sample error. Control: fit on a random INTERLEAVED half of the prompt rows
(383 of the rows 1..767, drawn without replacement, no temporal ordering) and measure the held-out ratio on the OTHER 384
rows. If drift mattered, the interleaved held-out ratio would be clearly smaller than the contiguous one.
Same recipe as kv_novelty.py Part A (dense pass of each window; per window, layer, kind): basis = online recipe from the fit
rows only (mean, G_K from the fit rows' queries, G_V = W_O W_O^T), rank scaled to keep R/n: R = 128 (K) / 256 (V) for 383
rows; residual = row norm after projection in the weighted metric. Quantities (medians over windows x layers):
  ratio_heldout_contig   rows 384..767 over in-sample rows 1..383  (contiguous fit on rows 1..383; code check = C3
                         'ratio_heldout_split': K 3.21, V 5.60 in c3_kv_novelty.json, all 16 windows)
  ratio_cont_contig      continuation rows 768..1023 over the same in-sample RMS (C3 'ratio_cont_split': 4.27 / 5.88)
  ratio_heldout_inter    the other half of rows 1..767 over in-sample RMS of the interleaved fit half
  ratio_cont_inter       continuation rows over in-sample RMS of the interleaved fit
The interleaved subset is drawn per (window, layer) with seed 1000 * window + layer + 900000 (the same subset for K and V).
Rule fixed before any full run (windows 0-15, medians over windows and layers, keys AND values):
  N4 drift plays no role: |median ratio_heldout_inter / median ratio_heldout_contig - 1| <= 0.25 for K and for V.
     If the interleaved ratio is smaller by more than 25% for either kind, drift contributes; if it is larger by more than
     25%, the contiguous split is not the pessimistic case (reported as such). Descriptive: the same for the continuation
     ratio, and paired per (window, layer) differences with a 5-95% range.
Run:  .venv\\Scripts\\python.exe experiments\\kv_novelty_interleaved.py --threads 8 --out experiments\\rerun_2026-11-22\\c18b_kv_novelty_interleaved.json
      smoke: --windows 1 --smoke --out <scratch>.json     (prints the code check against c3 rows of window 0)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, E, D64, ROOT, qblock, psd_pow, top_right, base_result, save_json, rss_mb)  # noqa: E402

P, RK, RV, NSPLIT, SEQ = 768, 256, 512, 384, 1024


def resid_rms(X, mu, T0, U):
    Ew = (X - mu) @ T0
    return (Ew - (Ew @ U) @ U.T).norm(dim=1)


def fit(Xq, Xr, rows, T_V, R):
    """Online recipe on the given prompt rows (index tensor). Returns (mu, T0, U)."""
    if T_V is None:
        Q = Xq[rows]
        GK = qblock(Q.T @ Q / Q.shape[0]); T = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
    else:
        T = T_V
    mu = Xr[rows].mean(0)
    U = top_right((Xr[rows] - mu) @ T[0], R)[0]
    return mu, T[0], U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c18b_kv_novelty_interleaved.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    res = base_result(a, {'P': P, 'windows': win_idx, 'split_rows': NSPLIT - 1, 'split_ranks': {'K': 128, 'V': 256}, 'rows': []})
    for wi, w in zip(win_idx, wins):
        ctx.mode['proj'] = None
        with torch.inference_mode():
            ctx.model(w[None])
        for l in range(L):
            X = ctx.cap[l].to(D64)
            Xq = X[:, :E]
            all_rows = torch.arange(1, P)
            contig = torch.arange(1, NSPLIT)
            gen = torch.Generator().manual_seed(1000 * wi + l + 900000)
            perm = all_rows[torch.randperm(P - 1, generator=gen)]
            inter = perm[:NSPLIT - 1].sort().values            # 383 rows, random subset of 1..767
            other = perm[NSPLIT - 1:].sort().values            # the remaining 384 prompt rows
            held_contig = torch.arange(NSPLIT, P)
            cont = torch.arange(P, SEQ)
            for kind, R, offs in (('K', RK, E), ('V', RV, 2 * E)):
                Xr = X[:, offs:offs + E]
                Rs = round(R * (NSPLIT - 1) / (P - 1))
                tv = None if kind == 'K' else ctx.TV[l]
                row = {'window': wi, 'layer': l, 'kind': kind, 'R_split': Rs}
                for tag, fit_rows, held in (('contig', contig, held_contig), ('inter', inter, other)):
                    mu, T0m, U = fit(Xq, Xr, fit_rows, tv, Rs)
                    es = resid_rms(Xr, mu, T0m, U)
                    ins = float(es[fit_rows].pow(2).mean().sqrt())
                    row[f'ratio_heldout_{tag}'] = float(es[held].pow(2).mean().sqrt()) / ins
                    row[f'ratio_cont_{tag}'] = float(es[cont].pow(2).mean().sqrt()) / ins
                res['rows'].append(row)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB)', flush=True)
    summ = {}
    for kind in 'KV':
        rr = [x for x in res['rows'] if x['kind'] == kind]
        for q in ('ratio_heldout_contig', 'ratio_cont_contig', 'ratio_heldout_inter', 'ratio_cont_inter'):
            v = [x[q] for x in rr]
            summ[f'{kind} {q} median'] = float(np.median(v))
            summ[f'{kind} {q} 5-95'] = [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
    res['summary'] = summ
    res['decision'] = {'N4_interleaved_within_25pct_of_contiguous':
                       all(abs(summ[f'{k} ratio_heldout_inter median'] / summ[f'{k} ratio_heldout_contig median'] - 1) <= 0.25 for k in 'KV')}
    c3p = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c3_kv_novelty.json')
    if a.smoke and os.path.exists(c3p):
        c3 = json.load(open(c3p))
        chk = []
        for x in res['rows']:
            for y in c3['rows']:
                if (y['window'], y['layer'], y['kind']) == (x['window'], x['layer'], x['kind']):
                    chk.append(max(abs(x['ratio_heldout_contig'] - y['ratio_heldout_split']),
                                   abs(x['ratio_cont_contig'] - y['ratio_cont_split'])))
        res['code_check_vs_c3_max_abs_diff'] = max(chk) if chk else None
        print('code check vs c3 rows (max abs diff of contiguous ratios):', res['code_check_vs_c3_max_abs_diff'], 'n', len(chk))
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('summary:', json.dumps(summ, indent=1))
    print('decision:', res['decision'])


if __name__ == '__main__':
    main()
