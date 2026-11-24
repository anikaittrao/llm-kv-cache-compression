"""Step (2026-11-19): a variance-form bound on the attention output (report.pdf Section 10, Proposition 10.6), on GPT-2.
Follow-up to kv_bound2.py, whose level-2' bound (Prop 10.5) was a median 3.5x the actual error; the remaining looseness
came from the triangle inequality across tokens.

Run:  .venv\\Scripts\\python.exe experiments\\kv_bound3.py            (about 20-30 min; needs no idle CPU)
      .venv\\Scripts\\python.exe experiments\\kv_bound3.py --replot
Writes experiments/results_kv_bound3.json and figures/fig21_kv_bound3.{pdf,png}.
Random stress test and sharpness check: experiments/bound3_random_test.py.

Theory. One head, one query; a = softmax(s), a^ = softmax(s + delta), o = sum a_i v_i, W = W_O^h.
  y_i = (v_i - o) W,  eps_i = (v^_i - v_i) W,  C_y = sum a_i y_i^T y_i,  C_e = sum a^_i eps_i^T eps_i,
  chi2 = sum (a^_i - a_i)^2 / a_i = Var_a(e^d) / (1 + r)^2   (to first order sum a_i d_i^2).
  Prop 10.6 ("level V"):  ||(o^ - o)W|| <= sqrt(lmax(C_y) chi2) + sqrt(lmax(C_e)).
  Combined ("level C"): each of the two terms replaced by the smaller of its Prop 10.5 level-2' form
  (1/(1+r)) sum a_i (|d_i| + phi(d_i) + r) u_i  resp.  sum a^_i ||eps_i||  and its Prop 10.6 form. Both levels need only
  the exact attention, the score errors and the value errors.
In 64 dimensions: lmax(C_y) = lmax(M C_v M) with C_v = sum a_i (v_i - vbar)(v_i - vbar)^T, M = (W W^T)^{1/2}; same for C_e.

Setting: identical to kv_bound2.py (online setting, P = 768, bases offline / online / rsvd q = 0, 1, 2, half and third of
the cache, one layer at a time, 16 windows), but only every 4th continuation query (64 per window, layer and head).

Rules fixed before the run:
  V1 correctness : level V >= actual error in every case (relative violation <= 1e-9 of the largest error, as for T1).
  V2 tighter     : median over cases of level 2' / level V >= 1.5.
  V3 combined    : level C is within 2x of the actual error (median actual / level C >= 0.5).
  V4 ranking     : Spearman(mean level V summed over layers and heads, perplexity from results_kv_online.json) >= 0.8.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_online import load_wikitext, psd_pow, qblock, rsvd_right, ROOT, D64, L, H, E, Dh  # noqa: E402
from kv_bound import SETTINGS, METHODS, PPL_KEY, spearman  # noqa: E402


def lmax(C, M):
    """Largest eigenvalue of M C M for a batch of 64 x 64 matrices C (..., 64, 64) and one symmetric M."""
    S = M @ C @ M
    return torch.linalg.eigvalsh((S + S.transpose(-1, -2)) / 2)[..., -1].clamp_min(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--layers', type=int, default=L)
    ap.add_argument('--stride', type=int, default=4, help='use every stride-th continuation query')
    ap.add_argument('--P', type=int, default=768)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_bound3.json'))
    ap.add_argument('--online_json', default=os.path.join(ROOT, 'experiments', 'results_kv_online.json'))
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
    Mh = [torch.stack([psd_pow(WOh[l][h] @ WOh[l][h].T, 0.5, ridge=0.0) for h in range(H)]) for l in range(L)]  # H x 64 x 64

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
    qidx = torch.arange(P, seq, a.stride)
    mask = tri[qidx]                                                       # Tq x T
    acc = {}
    viol = {'V': 0.0, 'C': 0.0, 'L2p': 0.0}
    keys = ('err', 'L2p', 'V', 'C', 'rL2p', 'rV', 'rC', 'gain', 'erank_y', 'erank_e', 'share_mix_V', 'win1', 'win2')
    for wi in range(a.windows):
        w = ids_test[wi * seq:(wi + 1) * seq]
        with torch.inference_mode():
            model(w[None])
        for l in range(a.layers):
            X = cap[l].to(D64)
            Xq, Xk, Xv = X[:, :E], X[:, E:2 * E], X[:, 2 * E:]
            q = Xq[qidx].reshape(-1, H, Dh).transpose(0, 1)                 # H x Tq x 64
            k = Xk.reshape(seq, H, Dh).transpose(0, 1)
            v = Xv.reshape(seq, H, Dh).transpose(0, 1)                      # H x T x 64
            s = ((q @ k.transpose(1, 2)) / 8.0).masked_fill(~mask, float('-inf'))
            att = torch.softmax(s, -1)                                       # H x Tq x T
            vW = torch.stack([v[h] @ WOh[l][h] for h in range(H)])           # H x T x 768
            o = att @ vW
            u = ((vW.pow(2).sum(-1)[:, None, :] - 2 * (o @ vW.transpose(1, 2)) + o.pow(2).sum(-1, keepdim=True))
                 .clamp_min(0).sqrt())                                       # H x Tq x T
            vv = (v[..., :, None] * v[..., None, :]).reshape(H, seq, Dh * Dh)
            vbar = att @ v                                                   # H x Tq x 64
            Cv = (att @ vv).reshape(H, -1, Dh, Dh) - vbar[..., :, None] * vbar[..., None, :]
            lam_y = torch.stack([lmax(Cv[h], Mh[l][h]) for h in range(H)])   # H x Tq
            tr_y = (att * u.pow(2)).sum(-1)                                  # trace of C_y
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
                    Rr = (att * phi).sum(-1)
                    eps = (vWh - vW).norm(dim=-1)                                    # H x T
                    t1_2p = (att * (dd.abs() + phi + Rr[..., None]) * u).sum(-1) / (1 + Rr)
                    t2 = (ah * eps[:, None, :]).sum(-1)
                    L2p = t1_2p + t2
                    chi2 = torch.where(att > 0, (ah - att).pow(2) / att.clamp_min(1e-300), torch.zeros_like(att)).sum(-1)
                    ev = vh - v                                                      # H x T x 64
                    ee = (ev[..., :, None] * ev[..., None, :]).reshape(H, seq, Dh * Dh)
                    Ce = (ah @ ee).reshape(H, -1, Dh, Dh)
                    lam_e = torch.stack([lmax(Ce[h], Mh[l][h]) for h in range(H)])
                    V1 = (lam_y * chi2).sqrt(); V2 = lam_e.sqrt()
                    Vb = V1 + V2
                    Cb = torch.minimum(t1_2p, V1) + torch.minimum(t2, V2)
                    scale = float(err.max()) + 1e-12
                    viol['V'] = max(viol['V'], float((err - Vb).max()) / scale)
                    viol['C'] = max(viol['C'], float((err - Cb).max()) / scale)
                    viol['L2p'] = max(viol['L2p'], float((err - L2p).max()) / scale)
                    r = acc.setdefault((sname, m), {k_: [] for k_ in keys})
                    r['err'].append(float(err.mean())); r['L2p'].append(float(L2p.mean()))
                    r['V'].append(float(Vb.mean())); r['C'].append(float(Cb.mean()))
                    sub = lambda t_: t_.flatten()[::3].tolist()
                    r['rL2p'] += sub(err / L2p); r['rV'] += sub(err / Vb); r['rC'] += sub(err / Cb); r['gain'] += sub(L2p / Vb)
                    r['erank_y'] += sub(tr_y / lam_y.clamp_min(1e-300))
                    r['erank_e'] += sub((ah * eps[:, None, :].pow(2)).sum(-1) / lam_e.clamp_min(1e-300))
                    r['share_mix_V'] += sub(V1 / Vb)
                    r['win1'] += sub((V1 < t1_2p).double()); r['win2'] += sub((V2 < t2).double())
        print(f'window {wi+1}/{a.windows} done ({time.perf_counter()-T0:.0f}s)  max violations {viol}', flush=True)
    for h_ in hooks:
        h_.remove()

    with open(a.online_json) as f:
        ppl = json.load(f)['settings']
    rows = []
    LH = a.layers * H  # heads summed over (144 for the full model; fewer in --layers smoke tests)
    for (sname, m), r in acc.items():
        rows.append({'setting': sname, 'method': m, 'ppl': ppl[PPL_KEY[sname]]['ppl'][m],
                     'err_mean': LH * float(np.mean(r['err'])), 'L2p_mean': LH * float(np.mean(r['L2p'])),
                     'V_mean': LH * float(np.mean(r['V'])), 'C_mean': LH * float(np.mean(r['C'])),
                     'ratio_L2p_median': float(np.median(r['rL2p'])), 'ratio_V_median': float(np.median(r['rV'])),
                     'ratio_C_median': float(np.median(r['rC'])), 'gain_median': float(np.median(r['gain']))})
    qk = ('rL2p', 'rV', 'rC', 'gain', 'erank_y', 'erank_e', 'share_mix_V')
    allr = {k_: np.concatenate([np.asarray(r[k_]) for r in acc.values()]) for k_ in qk + ('win1', 'win2')}
    res = {'config': vars(a), 'violations_rel': viol, 'rows': rows, 'n_cases_sampled': int(allr['rV'].size),
           'ratio_quantiles': {k_: {str(qq): float(np.quantile(allr[k_], qq)) for qq in (0.05, 0.25, 0.5, 0.75, 0.95)} for k_ in qk},
           'frac_V_term_smaller': {'mixing': float(allr['win1'].mean()), 'value': float(allr['win2'].mean())}}
    y = [r_['ppl'] for r_ in rows]
    res['spearman'] = {'V_vs_ppl': spearman([r_['V_mean'] for r_ in rows], y),
                       'C_vs_ppl': spearman([r_['C_mean'] for r_ in rows], y),
                       'L2p_vs_ppl': spearman([r_['L2p_mean'] for r_ in rows], y),
                       'err_vs_ppl': spearman([r_['err_mean'] for r_ in rows], y)}
    rq = res['ratio_quantiles']
    res['decision'] = {'V1_holds': viol['V'] <= 1e-9,
                       'V2_1.5x_tighter_than_L2p': rq['gain']['0.5'] >= 1.5,
                       'V3_combined_within_2x': rq['rC']['0.5'] >= 0.5,
                       'V4_spearman_ge_0.8': res['spearman']['V_vs_ppl'] >= 0.8}
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k_: res[k_] for k_ in ('violations_rel', 'ratio_quantiles', 'frac_V_term_smaller', 'spearman', 'decision')}, indent=1))
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
    labs = [('rL2p', "level 2'\n(Prop 10.5)"), ('rV', 'level V\n(Prop 10.6)'), ('rC', 'level C\n(smaller term)')]
    for i, (k, lab) in enumerate(labs):
        ax[0].plot([i, i], [q[k]['0.05'], q[k]['0.95']], color=dv.C['blue'], lw=1)
        ax[0].plot([i, i], [q[k]['0.25'], q[k]['0.75']], color=dv.C['blue'], lw=5, solid_capstyle='butt')
        ax[0].plot(i, q[k]['0.5'], 'o', color=dv.C['orange'], ms=4)
    ax[0].set_yscale('log'); ax[0].axhline(1, color='gray', lw=0.8); ax[0].axhline(0.5, color='gray', ls=':', lw=0.8)
    ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels([l_ for _, l_ in labs], fontsize=7)
    ax[0].set_xlim(-0.5, len(labs) - 0.5)
    ax[0].set_ylabel('actual error / bound (1 = exact)'); ax[0].set_title('(a) tightness')
    cols = {'offline': dv.C['black'], 'online': dv.C['blue'], 'rsvd q=0': dv.C['yellow'], 'rsvd q=1': dv.C['orange'], 'rsvd q=2': dv.C['aqua']}
    for r in res['rows']:
        mk = 'o' if r['setting'] == 'half' else 's'
        ax[1].plot(r['V_mean'], r['ppl'], mk, color=cols[r['method']], ms=5, mfc='none' if r['setting'] == 'third' else cols[r['method']])
    for m, c in cols.items():
        ax[1].plot([], [], 'o', color=c, label=m)
    ax[1].plot([], [], 'o', color='gray', label='half cache'); ax[1].plot([], [], 's', color='gray', mfc='none', label='third')
    ax[1].set_xlabel('level-V bound (sum over layers, heads)'); ax[1].set_ylabel('perplexity (continuation)')
    ax[1].set_title(f"(b) bound vs. perplexity (Spearman {res['spearman']['V_vs_ppl']:.2f})")
    ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig21_kv_bound3.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
