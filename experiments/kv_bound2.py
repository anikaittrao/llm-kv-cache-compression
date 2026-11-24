"""Step (2026-11-19): a tighter, attention-weighted bound on the attention output, checked on GPT-2.
Follow-up to kv_bound.py (report.pdf Section 10), whose level-2 bound was about 25x too large.

Run:  .venv\\Scripts\\python.exe experiments\\kv_bound2.py            (about 35 min)
      .venv\\Scripts\\python.exe experiments\\kv_bound2.py --replot
Writes experiments/results_kv_bound2.json and figures/fig17_kv_bound2.{pdf,png}.
Random stress test of the inequalities: experiments/bound2_random_test.py.

Theory (report.pdf Section 10, after Corollary 10.3). One head, one query; a = softmax(s), a^ = softmax(s + delta).
  d_i = delta_i - sum_j a_j delta_j   (score error relative to its attention-weighted mean)
  phi(x) = e^x - 1 - x >= 0,  R = sum_i a_i phi(d_i) >= 0,  m1 = sum_i a_i |d_i|,  s2m = sum_{d_i<0} a_i d_i^2.
  Exact:          a^_i = a_i e^{d_i} / (1 + R).
  Lemma 10.4      (m1 - s2m/2 - R)/(1 + R)  <=  ||a^ - a||_1  <=  (m1 + 2R)/(1 + R).
  Prop. 10.5      u_i = ||(v_i - o)W||, eps_i = ||(v^_i - v_i)W||:
                  ||(o^ - o)W|| <= sum_i |a^_i - a_i| u_i + sum_i a^_i eps_i                                   (level 1')
                               <= 1/(1+R) sum_i a_i [ (|d_i| + phi(d_i) + R) u_i + e^{d_i} eps_i ]             (level 2')
  Level 2' needs only the exact attention a, the score errors and the value errors (like level 2 of Prop 10.2),
  but weights every token by its attention instead of taking the worst pair / worst token.

Setting: identical to kv_bound.py (online setting of kv_online.py, P = 768, bases offline / online / rsvd q=0,1,2,
half and third of the cache, one layer at a time, 16 windows, all heads and continuation queries).

Rules fixed before the run:
  T1 correctness : Lemma 10.4 (both sides) and levels 1', 2' hold in every case where ||a^ - a||_1 > 1e-9
                   (relative violation <= 1e-9; below 1e-9 float64 round-off in a^ - a dominates).
  T2 usefulness  : level 2' is within 10x of the actual error (median actual / bound >= 0.1; the target B2 that the
                   level-2 bound of Prop. 10.2 failed with 0.039).
  T3 lemma       : the upper side of Lemma 10.4 is within 25% of ||a^ - a||_1 (median ratio >= 0.8).
  T4 ranking     : Spearman(mean level 2' summed over layers and heads, perplexity from results_kv_online.json) >= 0.8.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_online import load_wikitext, psd_pow, qblock, rsvd_right, ROOT, D64, L, H, E, Dh  # noqa: E402
from kv_bound import SETTINGS, METHODS, PPL_KEY, spearman  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--layers', type=int, default=L, help='smoke test: only the first n layers')
    ap.add_argument('--P', type=int, default=768)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_bound2.json'))
    ap.add_argument('--online_json', default=os.path.join(ROOT, 'experiments', 'results_kv_online.json'))
    ap.add_argument('--first_window', type=int, default=0, help='diagnostics: start at this window')
    ap.add_argument('--dump_worst', default=None, help='diagnostics: save the worst Lemma 10.4 upper-side case (s, delta) to this .pt file')
    ap.add_argument('--tv_floor', type=float, default=1e-9,
                    help='Lemma 10.4 is checked only where ||a^-a||_1 > tv_floor (default 1e-9 = the pre-registered T1). '
                         'A post hoc floor of 1e-6 (audit 2026-11-22) is always reported in addition as T1prime_posthoc.')
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    torch.set_num_threads(a.threads)
    T0 = time.perf_counter()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    seq, P = 1024, a.P
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
    TV = [(psd_pow(WO[l] @ WO[l].T, 0.5), psd_pow(WO[l] @ WO[l].T, -0.5)) for l in range(L)]
    WOh = [[WO[l][h * Dh:(h + 1) * Dh] for h in range(H)] for l in range(L)]

    cap = {}
    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: cap.__setitem__(l, o[0].detach()))
             for l in range(L)]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids_train.numel() - seq, (16,), generator=g).tolist()
    S1 = {k: [torch.zeros(E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    S2 = {k: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    with torch.inference_mode():
        for s in starts:
            model(ids_train[s:s + seq][None])
            for l in range(L):
                X = cap[l].to(D64)
                for i, k in enumerate('QKV'):
                    Xi = X[:, i * E:(i + 1) * E]; S1[k][l] += Xi.sum(0); S2[k][l] += Xi.T @ Xi
    n = 16 * seq
    off = {}
    for l in range(L):
        muK, muV = S1['K'][l] / n, S1['V'][l] / n
        CK = S2['K'][l] / n - torch.outer(muK, muK); CV = S2['V'][l] / n - torch.outer(muV, muV)
        GK = qblock(S2['Q'][l] / n); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
        off[l] = {'K': (muK, TK, torch.linalg.eigh(TK[0] @ CK @ TK[0])[1].flip(1)),
                  'V': (muV, TV[l], torch.linalg.eigh(TV[l][0] @ CV @ TV[l][0])[1].flip(1))}
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    tri = torch.ones(seq, seq, dtype=torch.bool).tril()
    acc = {}
    viol = {'lemma_up': 0.0, 'lemma_lo': 0.0, 'L1p': 0.0, 'L2p': 0.0, 'L2_old': 0.0}
    FLOOR_POSTHOC = 1e-6
    viol_ph = {'lemma_up': 0.0, 'lemma_lo': 0.0}  # post hoc: only cases with ||a^-a||_1 > 1e-6 (float64 round-off floor)
    keys = ('err', 'L2', 'L2p', 'r2', 'r1p', 'r2p', 'rup', 'rlo', 'rfo')

    worst = {'v': -1.0}
    for wi in range(a.first_window, a.first_window + a.windows):
        w = ids_test[wi * seq:(wi + 1) * seq]
        with torch.inference_mode():
            model(w[None])
        for l in range(a.layers):
            X = cap[l].to(D64)
            Xq, Xk, Xv = X[:, :E], X[:, E:2 * E], X[:, 2 * E:]
            q = Xq[P:].reshape(seq - P, H, Dh).transpose(0, 1)
            k = Xk.reshape(seq, H, Dh).transpose(0, 1)
            v = Xv.reshape(seq, H, Dh).transpose(0, 1)
            mask = tri[P:]
            s = ((q @ k.transpose(1, 2)) / 8.0).masked_fill(~mask, float('-inf'))
            att = torch.softmax(s, -1)                                       # H x Tq x T
            vW = torch.stack([v[h] @ WOh[l][h] for h in range(H)])           # H x T x 768
            o = att @ vW                                                     # H x Tq x 768
            # u_i = ||(v_i - o)W|| for every query: ||vW_i||^2 - 2 vW_i.o + ||o||^2
            u = ((vW.pow(2).sum(-1)[:, None, :] - 2 * (o @ vW.transpose(1, 2)) + o.pow(2).sum(-1, keepdim=True))
                 .clamp_min(0).sqrt())                                       # H x Tq x T
            diam = torch.empty(H, seq - P, dtype=D64)
            for h in range(H):
                Dm = torch.cdist(vW[h], vW[h]).masked_fill(~tri, 0).amax(1).cummax(0).values
                diam[h] = Dm[P:]
            pq, pk, pv = Xq[1:P], Xk[1:P], Xv[1:P]
            GKon = qblock(pq.T @ pq / pq.shape[0]); TKon = (psd_pow(GKon, 0.5), psd_pow(GKon, -0.5))
            onl = {}
            for kind, Xr, T in (('K', pk, TKon), ('V', pv, TV[l])):
                mu = Xr.mean(0); A = (Xr - mu) @ T[0]
                onl[kind] = (mu, T, torch.linalg.svd(A, full_matrices=False)[2].T, A)
            for sname, RK, RV in SETTINGS:
                for m in METHODS:
                    xh = {}
                    for kind, R, Xfull in (('K', RK, Xk), ('V', RV, Xv)):
                        if m == 'offline':
                            mu, T, U = off[l][kind]; Ur = U[:, :R]
                        else:
                            mu, T, U, A = onl[kind]
                            if m == 'online':
                                Ur = U[:, :R]
                            else:
                                qq = int(m[-1])
                                gen = torch.Generator().manual_seed(1000 * wi + 10 * l + (0 if kind == 'K' else 1) + 100000 * qq)
                                Ur = rsvd_right(A, R, a.p, qq, gen)[0]
                        Ew = (Xfull - mu) @ T[0]
                        e = Ew - (Ew @ Ur) @ Ur.T
                        e[0] = 0
                        xh[kind] = Xfull - e @ T[1]
                    kh = xh['K'].reshape(seq, H, Dh).transpose(0, 1)
                    vh = xh['V'].reshape(seq, H, Dh).transpose(0, 1)
                    sh = ((q @ kh.transpose(1, 2)) / 8.0).masked_fill(~mask, float('-inf'))
                    ah = torch.softmax(sh, -1)
                    vWh = torch.stack([vh[h] @ WOh[l][h] for h in range(H)])
                    err = (ah @ vWh - o).norm(dim=-1)                                # H x Tq
                    delta = (sh - s).masked_fill(~mask, 0.0)
                    dd = (delta - (att * delta).sum(-1, keepdim=True)).masked_fill(~mask, 0.0)
                    phi = (torch.expm1(dd) - dd).masked_fill(~mask, 0.0)
                    Rr = (att * phi).sum(-1)                                         # H x Tq
                    m1 = (att * dd.abs()).sum(-1)
                    s2m = (att * dd.pow(2) * (dd < 0)).sum(-1)
                    tv = (ah - att).abs().sum(-1)
                    up = (m1 + 2 * Rr) / (1 + Rr); lo = (m1 - 0.5 * s2m - Rr) / (1 + Rr)
                    eps = (vWh - vW).norm(dim=-1)                                    # H x T
                    L1p = ((ah - att).abs() * u).sum(-1) + (ah * eps[:, None, :]).sum(-1)
                    L2p = (att * ((dd.abs() + phi + Rr[..., None]) * u + torch.exp(dd).masked_fill(~mask, 0.0) * eps[:, None, :])).sum(-1) / (1 + Rr)
                    osc = delta.masked_fill(~mask, float('-inf')).amax(-1) - delta.masked_fill(~mask, float('inf')).amin(-1)
                    vmax = eps[:, None, :].masked_fill(~mask[None], 0).amax(-1)
                    L2 = 0.25 * osc * diam + vmax
                    ok = tv > a.tv_floor
                    okp = tv > FLOOR_POSTHOC
                    if okp.any():
                        viol_ph['lemma_up'] = max(viol_ph['lemma_up'], float(((tv - up) / tv)[okp].max()))
                        viol_ph['lemma_lo'] = max(viol_ph['lemma_lo'], float(((lo - tv) / tv)[okp].max()))
                    scale = err.abs().max() + 1e-12
                    viol['lemma_up'] = max(viol['lemma_up'], float(((tv - up) / tv)[ok].max()) if ok.any() else 0.0)
                    if a.dump_worst and ok.any():
                        rel = ((tv - up) / tv).masked_fill(~ok, -1.0)
                        j = int(rel.argmax()); hh, tq = divmod(j, rel.shape[1])
                        if float(rel[hh, tq]) > worst['v']:
                            nvalid = P + tq + 1
                            worst = {'v': float(rel[hh, tq]), 'tv': float(tv[hh, tq]), 'up': float(up[hh, tq]),
                                     's': s[hh, tq, :nvalid].clone(), 'sh': sh[hh, tq, :nvalid].clone(),
                                     'where': (wi, l, sname, m, hh, tq)}
                            torch.save(worst, a.dump_worst)
                    viol['lemma_lo'] = max(viol['lemma_lo'], float(((lo - tv) / tv)[ok].max()) if ok.any() else 0.0)
                    viol['L1p'] = max(viol['L1p'], float((err - L1p).max() / scale))
                    viol['L2p'] = max(viol['L2p'], float((err - L2p).max() / scale))
                    viol['L2_old'] = max(viol['L2_old'], float((err - L2).max() / scale))
                    r = acc.setdefault((sname, m), {k_: [] for k_ in keys})
                    r['err'].append(float(err.mean())); r['L2'].append(float(L2.mean())); r['L2p'].append(float(L2p.mean()))
                    sub = lambda t_: t_.flatten()[::7].tolist()
                    r['r2'] += sub(err / L2); r['r1p'] += sub(err / L1p); r['r2p'] += sub(err / L2p)
                    r['rup'] += sub((tv / up)[ok]); r['rlo'] += sub((lo / tv)[ok]); r['rfo'] += sub((tv / m1.clamp_min(1e-300))[ok])
        print(f'window {wi+1}/{a.windows} done ({time.perf_counter()-T0:.0f}s)  max violations {viol}', flush=True)
    for h_ in hooks:
        h_.remove()

    with open(a.online_json) as f:
        ppl = json.load(f)['settings']
    rows = []
    LH = a.layers * H  # heads summed over (144 for the full model; fewer in --layers smoke tests)
    for (sname, m), r in acc.items():
        rows.append({'setting': sname, 'method': m, 'ppl': ppl[PPL_KEY[sname]]['ppl'][m],
                     'err_mean': LH * float(np.mean(r['err'])), 'L2_mean': LH * float(np.mean(r['L2'])),
                     'L2p_mean': LH * float(np.mean(r['L2p'])),
                     'ratio_L2_median': float(np.median(r['r2'])), 'ratio_L1p_median': float(np.median(r['r1p'])),
                     'ratio_L2p_median': float(np.median(r['r2p'])), 'ratio_up_median': float(np.median(r['rup']))})
    allr = {k_: np.concatenate([np.asarray(r[k_]) for r in acc.values()]) for k_ in ('r2', 'r1p', 'r2p', 'rup', 'rlo', 'rfo')}
    res = {'config': vars(a), 'violations_rel': viol, 'violations_rel_posthoc_floor_1e-6': viol_ph, 'rows': rows,
           'ratio_quantiles': {k_: {str(qq): float(np.quantile(v_, qq)) for qq in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)} for k_, v_ in allr.items()}}
    y = [r_['ppl'] for r_ in rows]
    res['spearman'] = {'L2p_vs_ppl': spearman([r_['L2p_mean'] for r_ in rows], y),
                       'L2_vs_ppl': spearman([r_['L2_mean'] for r_ in rows], y),
                       'err_vs_ppl': spearman([r_['err_mean'] for r_ in rows], y)}
    res['decision'] = {'T1_all_hold': all(v_ <= 1e-9 for k_, v_ in viol.items()),
                       'T2_L2p_within_10x': res['ratio_quantiles']['r2p']['0.5'] >= 0.1,
                       'T3_lemma_up_within_25pct': res['ratio_quantiles']['rup']['0.5'] >= 0.8,
                       'T4_spearman_ge_0.8': res['spearman']['L2p_vs_ppl'] >= 0.8}
    # post hoc (audit 2026-11-22, not pre-registered): T1 with the round-off floor ||a^-a||_1 > 1e-6
    res['decision_posthoc'] = {'T1prime_floor_1e-6_all_hold': all(v_ <= 1e-9 for v_ in viol_ph.values()) and
                               all(viol[k_] <= 1e-9 for k_ in ('L1p', 'L2p', 'L2_old'))}
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k_: res[k_] for k_ in ('violations_rel', 'ratio_quantiles', 'spearman', 'decision')}, indent=1))
    for r_ in rows:
        print(r_)
    make_figures(a, res)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    q = res['ratio_quantiles']
    labs = [('rlo', 'Lemma 10.4\nlower /\nactual'), ('rup', 'Lemma 10.4\nactual /\nupper'),
            ('r2', 'old level 2\n(Prop 10.2)'), ('r1p', "level 1'"), ('r2p', "level 2'")]
    for i, (k, lab) in enumerate(labs):
        ax[0].plot([i, i], [max(q[k]['0.05'], 1e-4), q[k]['0.95']], color=dv.C['blue'], lw=1)
        ax[0].plot([i, i], [max(q[k]['0.25'], 1e-4), q[k]['0.75']], color=dv.C['blue'], lw=5, solid_capstyle='butt')
        ax[0].plot(i, q[k]['0.5'], 'o', color=dv.C['orange'], ms=4)
    ax[0].set_yscale('log'); ax[0].axhline(1, color='gray', lw=0.8); ax[0].axhline(0.1, color='gray', ls=':', lw=0.8)
    ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels([l_ for _, l_ in labs], fontsize=6.5)
    ax[0].set_ylabel('ratio (1 = exact)'); ax[0].set_title('(a) tightness')
    cols = {'offline': dv.C['black'], 'online': dv.C['blue'], 'rsvd q=0': dv.C['yellow'], 'rsvd q=1': dv.C['orange'], 'rsvd q=2': dv.C['aqua']}
    for r in res['rows']:
        mk = 'o' if r['setting'] == 'half' else 's'
        ax[1].plot(r['L2p_mean'], r['ppl'], mk, color=cols[r['method']], ms=5, mfc='none' if r['setting'] == 'third' else cols[r['method']])
    for m, c in cols.items():
        ax[1].plot([], [], 'o', color=c, label=m)
    ax[1].plot([], [], 'o', color='gray', label='half cache'); ax[1].plot([], [], 's', color='gray', mfc='none', label='third')
    ax[1].set_xlabel("level-2' bound (sum over layers, heads)"); ax[1].set_ylabel('perplexity (continuation)')
    ax[1].set_title(f"(b) bound vs. perplexity (Spearman {res['spearman']['L2p_vs_ppl']:.2f})")
    ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig17_kv_bound2.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
