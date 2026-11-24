"""C3 (audit 2026-11-22): is the "new-token gap" just the ordinary out-of-sample error of a basis fitted to n rows?

Run:  .venv\\Scripts\\python.exe experiments\\kv_novelty.py --out experiments\\rerun_2026-11-22\\c3_kv_novelty.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 35 min: calibration 80 s + ~125 s per window, smoke 2026-11-22)
      smoke test: --windows 1 --smoke        --replot: redraw the figure from the JSON.

Background. Section 7 measured that the online prompt basis fits later tokens 2.4x (keys) / 6.2x (values) worse than
the prompt rows it was computed from (RMS weighted residual, continuation / prompt; kv_bound.py 'frob'). Section 8
called this gap "intrinsic". A sceptic's reading: it is the usual in-sample vs out-of-sample gap of PCA with a rank R
close to the number of rows n (R/n = 256/767 = 0.33 keys, 512/767 = 0.67 values), so it should appear on ANY unseen
rows, not specially on later text.
Part A (dense pass of each window, no compression feedback; per window, layer, kind; online recipe as kv_bound.py):
  full fit  : basis from prompt rows 1..767 (R = 256 / 512); ratio_full = RMS residual on rows 768..1023 / RMS on rows
              1..767. Code check: medians = kv_bound.py 'half online' cont_over_prompt_rms (2.467 K, 6.187 V).
  split fit : basis from rows 1..383 only, rank scaled to keep R/n: R = 128 (K) / 256 (V) for n = 383 rows (mean and
              G_K from the same rows). ratio_heldout = RMS on held-out prompt rows 384..767 / RMS on rows 1..383;
              ratio_cont_split = RMS on rows 768..1023 / RMS on rows 1..383 (same basis).
  Gaussian  : rows drawn i.i.d. from N(0, C), C = weighted centered covariance of rows 1..1023 of the same window and
              layer (metric = the prompt's G^{1/2}); fit the online recipe to n = 767 draws (rank 256 / 512), and
              predicted ratio = sqrt(E||(x - m)(I - UU^T)||^2) / in-sample RMS, computed exactly for x ~ N(0, C)
              (m = sample mean of the draws). One draw per window and layer (seed = 1000 window + layer).
              Secondary: the same with C = calibration covariance (16 train windows, offline metric).
Part B (perplexity, kv_buffer.py main setting, buffer B = 32): dense, static, update (code checks 24.997 / 24.778 on
16 windows), offline (25.022), offline + update: the calibration basis, updated like 'update' with the sketch
[S U^T ; A_blk], S^2 = calibration eigenvalues x (P - 1) (i.e. the calibration covariance with the weight of a
P-1-row prompt); mean and G stay the calibration ones.
Rules fixed before the run (medians over windows and layers):
  N1 ordinary out-of-sample : |median ratio_heldout / median ratio_cont_split - 1| <= 0.25, for keys AND values.
  N2 Gaussian prediction    : |median predicted / median ratio_full - 1| <= 0.30 for keys AND values, and the predicted
                              value ratio > predicted key ratio.
  N3 descriptive            : e(offline + update) against e(update) and e(offline).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, qblock, psd_pow, top_right, prompt_state, offline_state, kinds_of,  # noqa: E402
                   run_block, base_result, ppl_table, save_json, rss_mb)

P, RK, RV, B = 768, 256, 512, 32
NSPLIT = 384  # split fit on rows 1..383
METHODS = ['dense', 'static', 'update', 'offline', 'offline + update']


def resid_rms(X, mu, T0, U):
    Ew = (X - mu) @ T0
    e = Ew - (Ew @ U) @ U.T
    return e.norm(dim=1)


def predicted_ratio(C, n, R, gen):
    """Online recipe on n i.i.d. N(0, C) rows; exact expected out-of-sample residual over in-sample RMS."""
    w, Q = torch.linalg.eigh((C + C.T) / 2)
    w = w.clamp_min(0)
    Z = torch.randn(n, C.shape[0], generator=gen, dtype=D64) * w.sqrt()
    X = Z @ Q.T
    m = X.mean(0)
    U = top_right(X - m, R)[0]
    ins = (X - m) - ((X - m) @ U) @ U.T
    Pp = torch.eye(C.shape[0], dtype=D64) - U @ U.T
    out2 = float(torch.trace(Pp @ C @ Pp)) + float((m @ Pp).pow(2).sum())
    return math.sqrt(out2) / float(ins.pow(2).sum(1).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c3_kv_novelty.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    off = ctx.offline()
    calibC = {}
    cal = ctx.calib
    for l in range(L):
        for kind in 'KV':
            mu = cal['S1'][kind][l] / cal['n']
            C = cal['S2'][kind][l] / cal['n'] - torch.outer(mu, mu)
            T0m = off[(kind, l)][1][0]
            calibC[(kind, l)] = T0m @ C @ T0m
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'windows': win_idx, 'tokens_per_window': 1024 - P - 1,
                          'split_rows': NSPLIT - 1, 'split_ranks': {'K': 128, 'V': 256}, 'rows': []})
    nllw = {m: [] for m in methods}
    for wi, w in zip(win_idx, wins):
        # ---- Part A: dense statistics
        ctx.mode['proj'] = None
        with torch.inference_mode():
            ctx.model(w[None])
        full = {l: ctx.cap[l].to(D64) for l in range(L)}
        for l in range(L):
            X = full[l]
            Xq = X[:, :E]
            GKf = qblock(Xq[1:P].T @ Xq[1:P] / (P - 1)); TKf = (psd_pow(GKf, 0.5), psd_pow(GKf, -0.5))
            GKs = qblock(Xq[1:NSPLIT].T @ Xq[1:NSPLIT] / (NSPLIT - 1)); TKs = (psd_pow(GKs, 0.5), psd_pow(GKs, -0.5))
            for kind, R, offs in (('K', RK, E), ('V', RV, 2 * E)):
                Xr = X[:, offs:offs + E]
                Tf = TKf if kind == 'K' else ctx.TV[l]
                Ts = TKs if kind == 'K' else ctx.TV[l]
                mu = Xr[1:P].mean(0); U = top_right((Xr[1:P] - mu) @ Tf[0], R)[0]
                en = resid_rms(Xr, mu, Tf[0], U); en[0] = 0
                ratio_full = float(en[P:].pow(2).mean().sqrt() / en[1:P].pow(2).mean().sqrt())
                Rs = round(R * (NSPLIT - 1) / (P - 1))
                mus = Xr[1:NSPLIT].mean(0); Us = top_right((Xr[1:NSPLIT] - mus) @ Ts[0], Rs)[0]
                es = resid_rms(Xr, mus, Ts[0], Us)
                ins = float(es[1:NSPLIT].pow(2).mean().sqrt())
                ratio_held = float(es[NSPLIT:P].pow(2).mean().sqrt()) / ins
                ratio_cont_s = float(es[P:].pow(2).mean().sqrt()) / ins
                Aw = (Xr[1:] - Xr[1:].mean(0)) @ Tf[0]
                Cw = Aw.T @ Aw / Aw.shape[0]
                gen = torch.Generator().manual_seed(1000 * wi + l)
                pred_w = predicted_ratio(Cw, P - 1, R, gen)
                gen = torch.Generator().manual_seed(1000 * wi + l + 500000)
                pred_c = predicted_ratio(calibC[(kind, l)], P - 1, R, gen)
                res['rows'].append({'window': wi, 'layer': l, 'kind': kind, 'R': R, 'R_split': Rs, 'ratio_full': ratio_full,
                                    'ratio_heldout_split': ratio_held, 'ratio_cont_split': ratio_cont_s,
                                    'pred_gauss_window': pred_w, 'pred_gauss_calib': pred_c})
        # ---- Part B: perplexity
        pre = ctx.prefill_stats(w, P)
        for m in methods:
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B)
            elif m in ('static', 'update'):
                st = prompt_state(ctx, pre, kinds)
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=(m == 'update'))
            elif m == 'offline':
                v, _ = run_block(ctx, w, P, kinds, offline_state(off, kinds), B=B, buf=True, upd=False)
            elif m == 'offline + update':
                v, _ = run_block(ctx, w, P, kinds, offline_state(off, kinds, n_rows_equiv=P - 1), B=B, buf=True, upd=True)
            nllw[m].append(v)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / (1024 - P - 1)):.3f}' for m in methods), flush=True)
    tok = 1024 - P - 1
    ppl = ppl_table(nllw, tok)
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e})
    summ = {}
    for kind in 'KV':
        rr = [x for x in res['rows'] if x['kind'] == kind]
        for q in ('ratio_full', 'ratio_heldout_split', 'ratio_cont_split', 'pred_gauss_window', 'pred_gauss_calib'):
            summ[f'{kind} {q} median'] = float(np.median([x[q] for x in rr]))
            summ[f'{kind} {q} 5-95'] = [float(np.quantile([x[q] for x in rr], 0.05)), float(np.quantile([x[q] for x in rr], 0.95))]
    res['summary'] = summ
    d = {}
    d['N1_heldout_like_continuation'] = all(abs(summ[f'{k} ratio_heldout_split median'] / summ[f'{k} ratio_cont_split median'] - 1) <= 0.25 for k in 'KV')
    d['N2_gaussian_predicts_gap'] = (all(abs(summ[f'{k} pred_gauss_window median'] / summ[f'{k} ratio_full median'] - 1) <= 0.30 for k in 'KV')
                                     and summ['V pred_gauss_window median'] > summ['K pred_gauss_window median'])
    d['code_check_ratio_full_vs_kv_bound'] = {'K': (summ['K ratio_full median'], 2.467481336369209),
                                             'V': (summ['V ratio_full median'], 6.186708334537716),
                                             'note': 'kv_bound.py values are over windows 0-15; equal only for the full run'}
    try:
        d['N3_offline_update_excess'] = {'offline + update': e['offline + update'], 'update': e['update'], 'offline': e['offline']}
    except KeyError as ex:
        d['N3_incomplete'] = f'missing {ex}'
    res['decision'] = d
    if a.smoke and win_idx == [0]:
        ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
        if os.path.exists(ref):
            kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
            res['code_checks_window0'] = {'static': (ppl.get('static'), kb['static / buffer B=32']),
                                          'update': (ppl.get('update'), kb['update / buffer B=32']),
                                          'offline': (ppl.get('offline'), kb['offline / buffer B=32']), 'dense': (ppl['dense'], kb['dense'])}
            print('code checks (this script, kv_buffer.py) on window 0:', res['code_checks_window0'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('ppl:', {k: round(v, 4) for k, v in ppl.items()})
    print('summary:', json.dumps(summ, indent=1))
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
    s = res['summary']
    qs = [('ratio_full', 'continuation\n(full fit)'), ('ratio_cont_split', 'continuation\n(split fit)'),
          ('ratio_heldout_split', 'held-out prompt\n(split fit)'), ('pred_gauss_window', 'Gaussian\nprediction')]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(qs))
    for j, (kind, c) in enumerate((('K', dv.C['blue']), ('V', dv.C['orange']))):
        ax.bar(x + (j - 0.5) * 0.36, [s[f'{kind} {q} median'] for q, _ in qs], 0.36, color=c, label='keys' if kind == 'K' else 'values')
    ax.axhline(1, color='gray', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l_ for _, l_ in qs], fontsize=6.5)
    ax.set_ylabel('RMS residual / in-sample RMS (median)'); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c3_kv_novelty.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
