"""Low-rank compression of GPT-2 weights: RSVD (dense) and ALS (masked).

Run:  .venv\\Scripts\\python.exe experiments\\llm_compression.py
Writes experiments/results.json and figures/fig{1,2,3,4}_*.{pdf,png}.
Function-space evaluation (perplexity, activation-aware SVD, latency) is in experiments/model_quality.py.

Convention: GPT-2 uses HF Conv1D, whose weight is stored as (in_features, out_features).
We define W in R^{m x n} as exactly that stored tensor (no transpose), e.g. mlp.c_fc: m=768, n=3072.
Everything numerical is done in float64 on CPU.
"""
import argparse, json, os, platform, statistics, time, math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = torch.float64


# ----------------------------------------------------------------------------- algorithms
def rsvd(A, k, p=10, q=0, gen=None):
    """Algorithm 2 (Halko-Martinsson-Tropp). Returns U_k, S_k, Vt_k, Q.
    Omega ~ N(0,1) n x l, l=k+p; Y=(A A^T)^q A Omega with QR re-orthonormalization."""
    m, n = A.shape
    l = min(k + p, min(m, n))
    Om = torch.randn(n, l, dtype=A.dtype, generator=gen)
    Q, _ = torch.linalg.qr(A @ Om)
    for _ in range(q):
        Z, _ = torch.linalg.qr(A.T @ Q)
        Q, _ = torch.linalg.qr(A @ Z)
    B = Q.T @ A
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    return (Q @ Ub[:, :k]), S[:k], Vt[:k], Q


def balanced(U, S, Vt):
    s = S.sqrt()
    return U * s, Vt.T * s  # U_f = U_k S^{1/2}, V_f = V_k S^{1/2}


def J_obj(W, mask, U, V, lam):
    R = (W - U @ V.T) * mask
    return float((R * R).sum() + lam * ((U * U).sum() + (V * V).sum()))


def als_half(W, mask, F, lam):
    """Closed-form ridge update of the rows of the 'other' factor given fixed F (n x k).
    G_i = sum_j mask_ij f_j f_j^T ; rhs_i = sum_j mask_ij W_ij f_j ; solve (G_i + lam I) x_i = rhs_i, batched."""
    k = F.shape[1]
    G = torch.einsum('ij,jab->iab', mask, F[:, :, None] * F[:, None, :])
    rhs = torch.einsum('ij,jk->ik', mask * W, F)
    G = G + lam * torch.eye(k, dtype=W.dtype)
    return torch.linalg.solve(G, rhs[:, :, None])[:, :, 0]


def als(W, mask, U0, V0, lam, sweeps):
    """Algorithm 1. Returns U, V and J log: [J_init, J after U-step, J after V-step, ...]."""
    U, V = U0.clone(), V0.clone()
    log = [J_obj(W, mask, U, V, lam)]
    for _ in range(sweeps):
        U = als_half(W, mask, V, lam); log.append(J_obj(W, mask, U, V, lam))
        V = als_half(W.T, mask.T, U, lam); log.append(J_obj(W, mask, U, V, lam))
    return U, V, log


def med_time(fn, repeats):
    fn()  # warmup
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), ts


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--layers', type=int, nargs='+', default=list(range(12)))
    ap.add_argument('--err_draws', type=int, default=5, help='independent RSVD draws for the error ratio (median reported)')
    ap.add_argument('--grid_ks', type=int, nargs='+', default=[8, 32, 128])
    ap.add_argument('--grid_rhos', type=float, nargs='+', default=[0.3, 0.5, 0.8])
    ap.add_argument('--ks', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--qs', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k_masked', type=int, default=32)
    ap.add_argument('--sweeps', type=int, default=100)
    ap.add_argument('--lam_rel', type=float, default=0.1, help='lambda = lam_rel * mean(W^2) (masked regime)')
    ap.add_argument('--init_std', type=float, default=0.1)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results.json'))
    ap.add_argument('--replot', action='store_true',
                    help='only regenerate the figures from the saved --out json (no recomputation)')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            saved = json.load(f)
        a.ks, a.qs = saved['config']['ks'], saved['config']['qs']
        os.makedirs(a.figdir, exist_ok=True)
        make_figures(a, saved['regimeA'], saved['regimeB'], None)
        print('figures regenerated from', a.out)
        return
    T0 = time.perf_counter()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.figdir, exist_ok=True)
    assertions = {}

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(a.model).eval()
    names = {'attn.c_attn': lambda b: b.attn.c_attn, 'attn.c_proj': lambda b: b.attn.c_proj,
             'mlp.c_fc': lambda b: b.mlp.c_fc, 'mlp.c_proj': lambda b: b.mlp.c_proj}
    mats = {}
    for L in a.layers:
        blk = model.transformer.h[L]
        for nm, f in names.items():
            mats[f'L{L}.{nm}'] = f(blk).weight.detach().to(DT).clone()  # stored (in,out) = (m,n)
    del model

    res = {'config': vars(a) | {'dtype': 'float64'}, 'model': a.model, 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                        'cpu_count': os.cpu_count(), 'torch_threads': torch.get_num_threads(), 'device': 'cpu'},
           'convention': 'W = stored Conv1D weight tensor, shape (in,out) = (m,n); no transpose.'}

    # ------------------------------------------------------------------ Regime A
    A = {}
    max_bal_err = 0.0
    for name, W in mats.items():
        m, n = W.shape
        nW = float(torch.linalg.norm(W))
        t_svd, ts_svd = med_time(lambda: torch.linalg.svd(W, full_matrices=False), a.repeats)
        Ue, Se, Vte = torch.linalg.svd(W, full_matrices=False)
        s2 = Se ** 2; tot = float(s2.sum()); cum = torch.cumsum(s2, 0) / tot
        ent = {'shape': [m, n], 'fro_norm': nW, 'svd_time_median_s': t_svd, 'svd_times_s': ts_svd,
               'breakeven_rank': m * n / (m + n),
               'k_for_energy': {str(int(100 * e)): int((cum >= e).nonzero()[0]) + 1 for e in (0.90, 0.95, 0.99)},
               'sigma': Se.tolist() if name.startswith('L5.') else None,
               'cum_energy': cum.tolist() if name.startswith('L5.') else None, 'by_k': {}}
        # also store curves for all matrices in compact form (every singular value) for fig1
        ent['sigma_all'] = Se.tolist(); ent['cum_energy_all'] = cum.tolist()
        ent.pop('sigma'); ent.pop('cum_energy')
        for k in a.ks:
            tail2 = float(s2[k:].sum()); ey = math.sqrt(tail2) / nW
            Wk = (Ue[:, :k] * Se[:k]) @ Vte[:k]
            # balanced factors reproduce W_k
            Uf, Vf = balanced(Ue[:, :k], Se[:k], Vte[:k])
            be = float(torch.linalg.norm(Uf @ Vf.T - Wk) / torch.linalg.norm(Wk)); max_bal_err = max(max_bal_err, be)
            kd = {'exact_energy': float(cum[k - 1]), 'eckart_young_rel_err': ey, 'balanced_factor_rel_err_exact': be,
                  'param_ratio': k * (m + n) / (m * n), 'q': {}}
            # q=0 expected-error bound over seeds
            bound = (1 + k / (a.p - 1)) * tail2
            sq = []
            for sd in range(a.seeds):
                g = torch.Generator().manual_seed(1000 + sd)
                _, _, _, Q = rsvd(W, k, a.p, 0, g)
                sq.append(float(((W - Q @ (Q.T @ W)) ** 2).sum()))
            sq = np.array(sq)
            kd['bound_q0'] = {'mean_sq_proj_err': float(sq.mean()), 'max_sq_proj_err': float(sq.max()),
                              'hmt_bound': bound, 'ratio_mean_to_bound': float(sq.mean() / bound),
                              'holds': bool(sq.mean() <= bound)}
            epsd = {}
            for delta in (0.01, 0.1, 0.5):
                eps_impl = k / ((a.p - 1) * delta)  # smallest eps with p >= 1 + k/(eps*delta)
                # empirical fraction of seeds with ||A-QQ^T A||^2 <= (1+eps) tail^2
                epsd[str(delta)] = {'implied_eps': eps_impl,
                                    'empirical_frac_within': float((sq <= (1 + eps_impl) * tail2).mean()),
                                    'p_needed_for_eps0.5': 1 + k / (0.5 * delta)}
            kd['cor_epsdelta'] = epsd
            for q in a.qs:
                g = torch.Generator().manual_seed(a.seed)
                fn = lambda: rsvd(W, k, a.p, q, torch.Generator().manual_seed(a.seed))
                t_r, ts_r = med_time(fn, a.repeats)
                Uk, Sk, Vtk, Q = rsvd(W, k, a.p, q, g)
                What = (Uk * Sk) @ Vtk
                Uf, Vf = balanced(Uk, Sk, Vtk)
                be2 = float(torch.linalg.norm(Uf @ Vf.T - What) / torch.linalg.norm(What)); max_bal_err = max(max_bal_err, be2)
                # error ratio over independent draws (seed a.seed + d); the reported value is the median
                errs, caps = [], []
                for d in range(a.err_draws):
                    Uk, Sk, Vtk, Q = rsvd(W, k, a.p, q, torch.Generator().manual_seed(a.seed + d))
                    errs.append(float(torch.linalg.norm(W - (Uk * Sk) @ Vtk) / nW))
                    caps.append(float((Q.T @ W).pow(2).sum() / nW ** 2))
                ratios_d = [e_ / ey for e_ in errs]
                kd['q'][str(q)] = {'time_median_s': t_r, 'times_s': ts_r, 'speedup_vs_svd': t_svd / t_r,
                                   'rel_err': float(np.median(errs)), 'err_ratio_to_optimal': float(np.median(ratios_d)),
                                   'err_ratio_draws': ratios_d, 'err_ratio_min': min(ratios_d), 'err_ratio_max': max(ratios_d),
                                   'captured_energy': float(np.median(caps)),
                                   'balanced_factor_rel_err': be2}
            ent['by_k'][str(k)] = kd
        A[name] = ent
        print(f'[A] {name} done  t_svd={t_svd:.3f}s  ({time.perf_counter()-T0:.0f}s elapsed)', flush=True)
    assertions['balanced_factors_reproduce_Wk_float64'] = {'max_rel_err': max_bal_err, 'passed': bool(max_bal_err < 1e-10)}
    holds = [v['bound_q0']['holds'] for e in A.values() for v in e['by_k'].values()]
    assertions['hmt_expected_error_bound_q0'] = {'n_cases': len(holds), 'n_holds': int(sum(holds)), 'passed': bool(all(holds))}
    ratios = [r_ for e in A.values() for v in e['by_k'].values() for q in a.qs for r_ in v['q'][str(q)]['err_ratio_draws']]
    assertions['rsvd_error_ge_optimal'] = {'min_ratio': min(ratios), 'passed': bool(min(ratios) >= 1 - 1e-9)}
    res['regimeA'] = A

    # ------------------------------------------------------------------ Regime B
    B = {}
    k = a.k_masked
    for name in ('L5.mlp.c_fc', 'L5.attn.c_proj'):
        W = mats[name]; m, n = W.shape
        lam = a.lam_rel * float((W ** 2).mean())
        g = torch.Generator().manual_seed(a.seed)
        masks = {'random_bernoulli': (torch.rand(m, n, generator=g, dtype=DT) < a.rho).to(DT)}
        thr = torch.kthvalue(W.abs().flatten(), int(W.numel() * (1 - a.rho))).values
        masks['magnitude_top50'] = (W.abs() > thr).to(DT)
        B[name] = {'lambda': lam, 'k': k, 'masks': {}}
        for mname, M in masks.items():
            rho = float(M.mean()); un = 1 - M
            nobs = float(torch.linalg.norm(W * M)); nun = float(torch.linalg.norm(W * un))
            def metrics(U, V):
                R = U @ V.T
                return {'rel_err_observed': float(torch.linalg.norm((W - R) * M) / nobs),
                        'rel_rmse_heldout': float(torch.linalg.norm((W - R) * un) / nun)}
            # baselines: RSVD of zero-filled matrix
            base = {}
            for tag, scale in (('zero_fill_no_rescale', 1.0), ('zero_fill_rescaled', 1 / rho)):
                Uk, Sk, Vtk, _ = rsvd(W * M * scale, k, a.p, 1, torch.Generator().manual_seed(a.seed))
                base[tag] = metrics(Uk * Sk, Vtk.T)  # product U S V^T (= balanced product)
            # inits
            g2 = torch.Generator().manual_seed(a.seed + 1)
            Ur = a.init_std * torch.randn(m, k, generator=g2, dtype=DT); Vr = a.init_std * torch.randn(n, k, generator=g2, dtype=DT)
            Uk, Sk, Vtk, _ = rsvd(W * M / rho, k, a.p, 1, torch.Generator().manual_seed(a.seed))
            Us, Vs = balanced(Uk, Sk, Vtk)
            runs = {}
            for iname, (U0, V0) in {'random': (Ur, Vr), 'rsvd': (Us, Vs)}.items():
                t = time.perf_counter()
                U, V, log = als(W, M, U0, V0, lam, a.sweeps)
                el = time.perf_counter() - t
                lg = np.array(log)
                mono = bool(np.all(lg[1:] <= lg[:-1] * (1 + 1e-9)))
                worst = float(np.max((lg[1:] - lg[:-1]) / lg[:-1]))
                runs[iname] = {'J_log': log, 'J_final': log[-1], 'monotone': mono, 'max_rel_increase': worst,
                               'rel_drop_last_10_sweeps': float((log[-21] - log[-1]) / log[-1]) if len(log) > 21 else None,
                               'time_s': el, 'init_J': log[0], **metrics(U, V)}
                assertions[f'als_monotone_{name}_{mname}_{iname}'] = {'max_rel_increase': worst, 'passed': mono}
            Jbest = min(r['J_final'] for r in runs.values())
            for r in runs.values():
                lg = np.array(r['J_log'])
                full = np.arange(2, len(lg), 2)  # indices after V-step of sweep s
                hit = [int(i // 2) for i in full if lg[i] <= 1.01 * Jbest]
                r['sweeps_to_1pct_of_best_final'] = hit[0] if hit else None
            B[name]['masks'][mname] = {'density': rho, 'baselines': base, 'als': runs, 'J_best_final': Jbest}
            print(f'[B] {name} {mname} done ({time.perf_counter()-T0:.0f}s elapsed)', flush=True)
    res['regimeB'] = B

    # ------------------------------------------------------------------ Regime B grid: does held-out error ever beat 1?
    # L5.attn.c_proj, random mask, RSVD init (Rem. 6.3), over rank k and density rho
    name = 'L5.attn.c_proj'; W = mats[name]; m, n = W.shape
    lam = a.lam_rel * float((W ** 2).mean())
    grid = []
    for rho_t in a.grid_rhos:
        M = (torch.rand(m, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=DT) < rho_t).to(DT)
        rho = float(M.mean()); un = 1 - M
        nobs = float(torch.linalg.norm(W * M)); nun = float(torch.linalg.norm(W * un))
        for k in a.grid_ks:
            Uk, Sk, Vtk, _ = rsvd(W * M / rho, k, a.p, 1, torch.Generator().manual_seed(a.seed))
            U0, V0 = balanced(Uk, Sk, Vtk)
            U, V, log = als(W, M, U0, V0, lam, a.sweeps)
            lg = np.array(log); mono = bool(np.all(lg[1:] <= lg[:-1] * (1 + 1e-9)))
            R = U @ V.T
            Uz, Sz, Vtz, _ = rsvd(W * M, k, a.p, 1, torch.Generator().manual_seed(a.seed))
            Rz = (Uz * Sz) @ Vtz
            Uo, So, Vto = torch.linalg.svd(W, full_matrices=False)
            Ro = (Uo[:, :k] * So[:k]) @ Vto[:k]  # oracle: best rank-k fit of the FULL matrix (sees the held-out entries)
            grid.append({'k': k, 'rho_target': rho_t, 'density': rho, 'monotone': mono, 'J_final': log[-1],
                         'rel_err_observed': float(torch.linalg.norm((W - R) * M) / nobs),
                         'rel_rmse_heldout': float(torch.linalg.norm((W - R) * un) / nun),
                         'zero_fill_heldout': float(torch.linalg.norm((W - Rz) * un) / nun),
                         'oracle_svd_heldout': float(torch.linalg.norm((W - Ro) * un) / nun)})
            assertions[f'als_monotone_grid_k{k}_rho{rho_t}'] = {'passed': mono}
            print(f'[B-grid] k={k} rho={rho_t} heldout={grid[-1]["rel_rmse_heldout"]:.3f} '
                  f'oracle={grid[-1]["oracle_svd_heldout"]:.3f} ({time.perf_counter()-T0:.0f}s elapsed)', flush=True)
    res['regimeB_grid'] = {'matrix': name, 'mask': 'random_bernoulli', 'init': 'rsvd', 'lambda': lam, 'runs': grid}
    res['assertions'] = assertions
    res['all_assertions_passed'] = bool(all(v['passed'] for v in assertions.values()))
    res['total_runtime_s'] = time.perf_counter() - T0

    # ------------------------------------------------------------------ figures
    make_figures(a, A, B, mats)
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('all assertions passed:', res['all_assertions_passed'], ' total %.1fs' % res['total_runtime_s'])
    for k_, v in assertions.items():
        if not v['passed']: print('FAILED', k_, v)


def make_figures(a, A, B, mats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    OI = {'blue': dv.C['blue'], 'orange': dv.C['yellow'], 'green': dv.C['aqua'], 'verm': dv.C['orange'], 'purple': dv.C['magenta'], 'sky': dv.C['violet'], 'black': dv.C['black']}
    def save(fig, stem):
        fig.savefig(os.path.join(a.figdir, stem + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(a.figdir, stem + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # fig1
    tcol = {'attn.c_attn': OI['blue'], 'attn.c_proj': OI['orange'], 'mlp.c_fc': OI['green'], 'mlp.c_proj': OI['verm']}
    lsty = {0: ':', 5: '-', 11: '--'}
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
    from matplotlib.lines import Line2D
    tmark = {'attn.c_attn': 'o', 'attn.c_proj': 's', 'mlp.c_fc': '^', 'mlp.c_proj': 'D'}
    k95lab = {}
    for name, e in A.items():
        L = int(name.split('.')[0][1:]); t = name.split('.', 1)[1]
        if L not in lsty:  # panel (a) shows layers 0, 5, 11 only; all 12 layers are summarized in fig4
            continue
        s = np.array(e['sigma_all']); c = np.array(e['cum_energy_all']); kk = np.arange(1, len(s) + 1)
        # (a): all twelve matrices (colour = matrix type, line style = layer)
        ax[0].semilogy(kk, s, color=tcol[t], ls=lsty.get(L, '-'), lw=1.2 if L == 5 else 0.9)
        # (b): layer 5 only, so that the four curves and their 95% points can be told apart
        if L == 5:
            k95 = e['k_for_energy']['95']; k95lab[t] = k95
            ax[1].plot(kk, c, color=tcol[t], lw=1.2)
            ax[1].plot([k95], [0.95], tmark[t], color=tcol[t], mec='black', ms=6, zorder=5, ls='none')
    ax[1].axhline(0.95, color='gray', lw=0.7, ls='-.')
    ax[0].set_xlabel('index $i$'); ax[0].set_ylabel(r'singular value $\sigma_i$'); ax[0].set_title('(a) spectrum, layers 0, 5, 11')
    ax[1].set_xlabel('rank $k$'); ax[1].set_ylabel(r'$\sum_{i\leq k}\sigma_i^2/\|W\|_F^2$'); ax[1].set_title('(b) cumulative energy, layer 5')
    h = [Line2D([], [], color=c, lw=1.5, label=t) for t, c in tcol.items()] + \
        [Line2D([], [], color='k', ls=lsty[L], lw=1, label=f'layer {L}') for L in (0, 5, 11)]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    h2 = [Line2D([], [], marker=tmark[t], color=tcol[t], mec='black', ls='none', ms=5, label=f'{t}: $k_{{0.95}}={k95lab[t]}$') for t in tcol]
    ax[1].legend(handles=h2, loc='lower right', frameon=False, fontsize=6.5)
    fig.tight_layout(); save(fig, 'fig1_spectrum_energy')

    # fig2
    qcol = {0: OI['blue'], 1: OI['orange'], 2: OI['green']}
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
    ref = 'L5.mlp.c_fc'; ks = a.ks
    ax[0].plot(ks, [A[ref]['svd_time_median_s']] * len(ks), color='k', ls='--', label='exact SVD')
    for q in a.qs:
        ax[0].plot(ks, [A[ref]['by_k'][str(k)]['q'][str(q)]['time_median_s'] for k in ks], 'o-', color=qcol[q], ms=4, label=f'RSVD $q={q}$')
    ax[0].set_yscale('log'); ax[0].set_xscale('log', base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
    ax[0].set_xlabel('target rank $k$'); ax[0].set_ylabel('median wall-clock (s)'); ax[0].set_title('(a) runtime, layer 5 mlp.c_fc'); ax[0].legend(frameon=False)
    mk = {16: 'o', 32: 's', 64: '^', 128: 'D'}; kc = [OI['blue'], OI['orange'], OI['green'], OI['verm']]
    for k, c in zip(ks, kc):
        arr = np.array([[e['by_k'][str(k)]['q'][str(q)]['err_ratio_to_optimal'] for q in a.qs]
                        for nm, e in A.items() if nm.startswith('L5.')])  # layer 5 only, as in Table 2
        ax[1].plot(a.qs, np.median(arr, 0), marker=mk.get(k, 'o'), color=c, ms=4, label=f'$k={k}$')
        ax[1].fill_between(a.qs, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
    ax[1].axhline(1, color='k', lw=0.7, ls='--'); ax[1].set_xticks(a.qs)
    ax[1].set_xlabel('power iterations $q$'); ax[1].set_ylabel('RSVD error / optimal error'); ax[1].set_title('(b) error ratio, layer 5 (median, min-max over 4 matrices)')
    ax[1].legend(frameon=False); fig.tight_layout(); save(fig, 'fig2_rsvd_vs_svd')

    # fig3
    mn = {'random_bernoulli': 'random mask ($\\rho=0.5$)', 'magnitude_top50': 'magnitude mask (top 50%)'}
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
    for r, name in enumerate(B):
        for c, mname in enumerate(mn):
            x = ax[r, c]; d = B[name]['masks'][mname]['als']
            Jb = B[name]['masks'][mname]['J_best_final']
            for iname, col, lab in (('random', OI['blue'], 'random init'), ('rsvd', OI['orange'], 'RSVD init')):
                lg = np.array(d[iname]['J_log'])
                ex = (lg - Jb) / Jb                      # relative excess over the best final J
                ex = np.where(ex > 0, ex, np.nan)        # the best run's last point is exactly 0: not drawable on a log axis
                x.semilogy(np.arange(len(lg)), ex, color=col, lw=1.2, label=lab)
                s = d[iname]['sweeps_to_1pct_of_best_final']
                if s is not None:
                    x.plot([2 * s], [ex[2 * s]], marker='*', ms=10, color=col, mec='black', zorder=5, ls='none')
            x.axhline(0.01, color='gray', lw=0.7, ls='-.')
            x.set_title(f"layer {name[1:].split('.', 1)[0]} {name.split('.', 1)[1]}: {mn[mname]}", fontsize=8)
            if r == 1: x.set_xlabel('half-sweep (U-step, V-step alternate)')
            if c == 0: x.set_ylabel(r'$(J-J_{\mathrm{best}})/J_{\mathrm{best}}$')
    h, l = ax[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], marker='*', ls='none', color='gray', mec='black', ms=9)); l.append('within 1% of best final $J$')
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'fig3_als_convergence')

    # fig4: rank for 90/95/99% energy relative to break-even, every layer and matrix type
    layers = sorted({int(nm.split('.')[0][1:]) for nm in A})
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8))
    for t, c in tcol.items():
        y95 = [A[f'L{L}.{t}']['k_for_energy']['95'] / A[f'L{L}.{t}']['breakeven_rank'] for L in layers]
        y90 = [A[f'L{L}.{t}']['k_for_energy']['90'] / A[f'L{L}.{t}']['breakeven_rank'] for L in layers]
        ax.plot(layers, y95, marker=tmark[t], color=c, ms=4, lw=1.2, label=t)
        ax.plot(layers, y90, marker=tmark[t], color=c, ms=3, lw=0.8, ls=':', mfc='white')
    ax.axhline(1, color='k', lw=0.8, ls='--')
    ax.set_xticks(layers); ax.set_xlabel('layer'); ax.set_ylabel(r'$k_{e}/k_{\mathrm{be}}$')
    ax.set_title('rank for 95% (solid) and 90% (dotted) energy / break-even rank')
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
    fig.tight_layout(); save(fig, 'fig4_rank_vs_breakeven')


if __name__ == '__main__':
    main()
