"""Regime C: does weight-space error predict model quality? GPT-2 perplexity, activation-aware SVD, inference timing.

Run:  .venv\\Scripts\\python.exe experiments\\model_quality.py          (about 20-30 min on a laptop CPU)
      .venv\\Scripts\\python.exe experiments\\model_quality.py --replot (figures only, from the saved json)
Writes experiments/results_quality.json and figures/fig5_perplexity.{pdf,png}, figures/fig6_latency.{pdf,png}.

All 48 block matrices of GPT-2 small (12 layers x attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) are replaced by
rank-k factors U V^T. Embeddings and the LM head are left dense. Convention as in llm_compression.py: W is the stored
Conv1D tensor, shape (in, out) = (m, n), and the layer computes y = x W + b.

Methods (all factorizations in float64, model run in float32):
  svd      plain truncated SVD of W                     -> minimizes ||W - W_hat||_F           (Thm 2.4)
  rsvd_q0  RSVD of W, p=10, q=0                          (Alg. 2)
  rsvd_q2  RSVD of W, p=10, q=2                          (Alg. 2 + Rem. 5.10)
  wsvd     activation-aware ("whitened") SVD: with C = X^T X / N = L L^T from calibration inputs X,
           ||X (W - W_hat)||_F^2 = N ||L^T (W - W_hat)||_F^2, so the optimum over rank-k W_hat is
           W_hat = L^{-T} [L^T W]_k  (Thm 2.4 applied to L^T W; same idea as SVD-LLM)
  wrsvd_q2 the same with RSVD (q=2) in place of the exact SVD of L^T W
Rank choice: uniform parameter ratio r = k(m+n)/(mn) per matrix, and (plain SVD) energy targets via Prop. 3.4,
where a matrix whose energy rank reaches the break-even rank mn/(m+n) is left dense.
"""
import argparse, json, math, os, platform, statistics, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_compression import rsvd, balanced  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64
TYPES = ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


class LowRank(nn.Module):
    """y = (x U) V^T + b, the factored replacement of a Conv1D layer."""
    def __init__(self, U, V, bias):
        super().__init__()
        self.U = nn.Parameter(U.float().contiguous(), requires_grad=False)          # (m, k)
        self.Vt = nn.Parameter(V.T.float().contiguous(), requires_grad=False)       # (k, n)
        self.bias = nn.Parameter(bias.detach().float().clone(), requires_grad=False)

    def forward(self, x):
        return torch.addmm(self.bias, (x.reshape(-1, x.shape[-1]) @ self.U), self.Vt).reshape(*x.shape[:-1], -1)


def get_mod(block, t):
    a, b = t.split('.')
    return getattr(getattr(block, a), b)


def set_mod(block, t, mod):
    a, b = t.split('.')
    setattr(getattr(block, a), b, mod)


def load_wikitext(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download('Salesforce/wikitext', f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset')
    return '\n\n'.join(pq.read_table(f).column('text').to_pylist())


@torch.inference_mode()
def perplexity(model, windows):
    nll, ntok = 0.0, 0
    for w in windows:
        out = model(w[None], labels=w[None])
        n = w.numel() - 1
        nll += float(out.loss) * n; ntok += n
    return math.exp(nll / ntok)


def factor(W, k, method, L=None, p=10, seed=0, svd_cache=None):
    """Return balanced factors U (m,k), V (n,k) with U V^T = W_hat (Prop. 3.2).
    svd_cache: precomputed thin SVD (U, S, Vt) of M = W (plain) or M = L^T W (whitened), reused across ranks."""
    M = W if L is None else L.T @ W
    if method in ('svd', 'wsvd'):
        Uf, S, Vt = svd_cache if svd_cache is not None else torch.linalg.svd(M, full_matrices=False)
        Uk, Sk, Vtk = Uf[:, :k], S[:k], Vt[:k]
    else:
        q = int(method[-1])
        Uk, Sk, Vtk, _ = rsvd(M, k, p, q, torch.Generator().manual_seed(seed))
    U, V = balanced(Uk, Sk, Vtk)
    if L is not None:
        U = torch.linalg.solve_triangular(L.T, U, upper=True)  # L^{-T} U
    return U, V


def med(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.9, 0.7, 0.5])
    ap.add_argument('--energies', type=float, nargs='+', default=[0.95, 0.99])
    ap.add_argument('--methods', nargs='+', default=['svd', 'rsvd_q0', 'rsvd_q2', 'wsvd', 'wrsvd_q2'])
    ap.add_argument('--eval_windows', type=int, default=8)
    ap.add_argument('--calib_windows', type=int, default=16)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--damp', type=float, default=1e-4, help='C += damp * mean(diag C) * I before Cholesky')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_quality.json'))
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--cache', default=os.path.join(tempfile.gettempdir(), 'llm_compression_quality_cache.pt'),
                    help='calibration/SVD cache (about 1 GB; kept outside the OneDrive project folder)')
    ap.add_argument('--fresh', action='store_true', help='ignore the cache and any partial results')
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    partial = a.out + '.partial'
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated from', a.out)
        return
    T0 = time.perf_counter()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    blocks = model.transformer.h
    dense = {(l, t): get_mod(blocks[l], t) for l in range(len(blocks)) for t in TYPES}
    Ws = {key: m.weight.detach().to(D64).clone() for key, m in dense.items()}

    # ---- data: non-overlapping windows; eval = first windows of the test split, calibration = random train windows
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    evalw = [ids_test[i * a.seq:(i + 1) * a.seq] for i in range(a.eval_windows)]
    g = torch.Generator().manual_seed(a.seed)
    starts = torch.randint(0, ids_train.numel() - a.seq, (a.calib_windows,), generator=g).tolist()
    calw = [ids_train[s:s + a.seq] for s in starts]

    # ---- input second moments C = X^T X / N of every block matrix (dense model): calibration (train) and eval (test)
    def second_moments(windows):
        S = {key: torch.zeros(W.shape[0], W.shape[0], dtype=D64) for key, W in Ws.items()}
        cnt = {key: 0 for key in Ws}
        hooks = []
        for key, m in dense.items():
            def hk(mod, args, key=key):
                x = args[0].reshape(-1, args[0].shape[-1]).to(D64)
                S[key] += x.T @ x; cnt[key] += x.shape[0]
            hooks.append(m.register_forward_pre_hook(hk))
        with torch.inference_mode():
            for w in windows:
                model(w[None])
        for h in hooks:
            h.remove()
        return {key: S[key] / cnt[key] for key in S}
    fp = {k_: getattr(a, k_) for k_ in ('eval_windows', 'calib_windows', 'seq', 'damp', 'seed')}
    cache = torch.load(a.cache) if (os.path.exists(a.cache) and not a.fresh) else None
    if cache is not None and cache.get('fingerprint') == fp:
        Ls, Ceval, svd_plain, svd_white = cache['Ls'], cache['Ceval'], cache['svd_plain'], cache['svd_white']
        print(f'calibration and SVDs loaded from cache {a.cache}', flush=True)
    else:
        C = second_moments(calw)                              # used to build the activation-aware factors
        Ceval = {k_: v.float() for k_, v in second_moments(evalw).items()}  # only for reporting held-out output error
        Ls = {key: torch.linalg.cholesky(Cn + a.damp * Cn.diagonal().mean() * torch.eye(Cn.shape[0], dtype=D64))
              for key, Cn in C.items()}
        del C  # only the Cholesky factors are needed from here on (saves ~1 GB)
        svd_plain = {key: tuple(torch.linalg.svd(W, full_matrices=False)) for key, W in Ws.items()}
        svd_white = {key: tuple(torch.linalg.svd(Ls[key].T @ W, full_matrices=False)) for key, W in Ws.items()}
        torch.save({'fingerprint': fp, 'Ls': Ls, 'Ceval': Ceval, 'svd_plain': svd_plain, 'svd_white': svd_white}, a.cache)
        print(f'calibration and SVDs done ({time.perf_counter()-T0:.0f}s), cached to {a.cache}', flush=True)
    del cache

    def install(factors):
        for (l, t), (U, V) in factors.items():
            set_mod(blocks[l], t, LowRank(U, V, dense[(l, t)].bias) if U is not None else dense[(l, t)])

    def restore():
        for (l, t), m in dense.items():
            set_mod(blocks[l], t, m)

    def act_err(W, What, Cn):
        D = W - What; Cn = Cn.to(D64)
        return math.sqrt(float(torch.trace(D.T @ Cn @ D)) / float(torch.trace(W.T @ Cn @ W)))

    total_block = sum(W.numel() for W in Ws.values())
    res = {'config': vars(a), 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                        'torch_threads': torch.get_num_threads(), 'device': 'cpu', 'model_dtype': 'float32'},
           'eval_tokens': sum(w.numel() for w in evalw), 'calib_tokens': sum(w.numel() for w in calw),
           'block_params_dense': total_block, 'model_params_total': sum(p.numel() for p in model.parameters()),
           'runs': []}

    done = {}
    if os.path.exists(partial) and not a.fresh:
        with open(partial) as f:
            old = json.load(f)
        if {k_: old['config'].get(k_) for k_ in fp} == fp:
            res['baseline'] = old['baseline']
            done = {(r['label'], r['method']): r for r in old['runs']}
            print(f'resuming: {len(done)} configurations already done', flush=True)
    if 'baseline' not in res:
        t = time.perf_counter(); ppl0 = perplexity(model, evalw)
        res['baseline'] = {'ppl': ppl0, 'eval_time_s': time.perf_counter() - t}
    print(f'baseline ppl {res["baseline"]["ppl"]:.2f} ({time.perf_counter()-T0:.0f}s)', flush=True)

    def run(label, method, ks):
        if (label, method) in done:
            res['runs'].append(done[(label, method)])
            return done[(label, method)]
        t = time.perf_counter()
        fac, werr, aerr, params = {}, [], [], 0
        for key, W in Ws.items():
            m, n = W.shape; k = ks[key]
            if k is None:
                fac[key] = (None, None); params += m * n; continue
            wh = method.startswith('w')
            U, V = factor(W, k, method, Ls[key] if wh else None, seed=a.seed,
                          svd_cache=(svd_white if wh else svd_plain)[key])
            What = U @ V.T
            werr.append(float(torch.linalg.norm(W - What) / torch.linalg.norm(W)))
            aerr.append(act_err(W, What, Ceval[key]))
            fac[key] = (U.float(), V.float()); params += k * (m + n)
        tf = time.perf_counter() - t
        install(fac); del fac
        t = time.perf_counter(); ppl = perplexity(model, evalw); te = time.perf_counter() - t
        restore()
        r = {'label': label, 'method': method, 'ppl': ppl, 'param_frac_blocks': params / total_block,
             'n_dense_kept': sum(1 for k in ks.values() if k is None),
             'weight_rel_err_mean': float(np.mean(werr)) if werr else 0.0,
             'act_rel_err_mean': float(np.mean(aerr)) if aerr else 0.0,
             'weight_rel_err_all': werr, 'act_rel_err_all': aerr,
             'ranks': {f'L{l}.{t_}': ks[(l, t_)] for (l, t_) in ks}, 'factor_time_s': tf, 'eval_time_s': te}
        res['runs'].append(r)
        with open(partial, 'w') as f:
            json.dump(res, f)
        print(f'{label:>18s} {method:>9s} ppl={ppl:10.2f} params={r["param_frac_blocks"]:.3f} '
              f'werr={r["weight_rel_err_mean"]:.3f} aerr={r["act_rel_err_mean"]:.3f} ({time.perf_counter()-T0:.0f}s)', flush=True)
        return r

    for r_ in a.ratios:
        ks = {key: int(r_ * W.shape[0] * W.shape[1] / sum(W.shape)) for key, W in Ws.items()}
        for meth in a.methods:
            run(f'ratio={r_}', meth, ks)
    for e in a.energies:
        ks = {}
        for key, W in Ws.items():
            s2 = svd_plain[key][1] ** 2; cum = torch.cumsum(s2, 0) / s2.sum()
            k = int((cum >= e).nonzero()[0]) + 1
            ks[key] = k if k < W.shape[0] * W.shape[1] / sum(W.shape) else None
        run(f'energy={e}', 'svd', ks)

    # ---- inference timing (float32): one layer, dense x W vs factored (x U) V^T
    lat = []
    shapes = sorted({tuple(W.shape) for W in Ws.values()})
    for (m, n) in shapes:
        Wd = torch.randn(m, n); b = torch.zeros(n)
        for T in (1, 1024):
            x = torch.randn(T, m)
            reps = 200 if T == 1 else 20
            td = med(lambda: torch.addmm(b, x, Wd), reps)
            for r_ in [1.0] + a.ratios:
                k = int(r_ * m * n / (m + n)); U = torch.randn(m, k); Vt = torch.randn(k, n)
                tl = med(lambda: torch.addmm(b, x @ U, Vt), reps)
                lat.append({'shape': [m, n], 'T': T, 'ratio': r_, 'k': k, 'dense_s': td, 'lowrank_s': tl, 'speedup': td / tl})
    res['layer_latency'] = lat
    # ---- end-to-end greedy decoding: 32-token prompt, 64 new tokens, per-token time
    prompt = ids_test[:32][None]

    def decode_time():
        with torch.inference_mode():
            model.generate(prompt, max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    e2e = [{'ratio': 1.0, 'method': 'dense', 's_per_token': med(decode_time, 3) / 64}]
    for r_ in a.ratios:
        ks = {key: int(r_ * W.shape[0] * W.shape[1] / sum(W.shape)) for key, W in Ws.items()}
        install({key: factor(W, ks[key], 'wsvd', Ls[key], svd_cache=svd_white[key]) for key, W in Ws.items()})
        e2e.append({'ratio': r_, 'method': 'wsvd', 's_per_token': med(decode_time, 3) / 64})
        restore()
    for d in e2e:
        d['speedup_vs_dense'] = e2e[0]['s_per_token'] / d['s_per_token']
    res['decode_latency'] = e2e
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    make_figures(a, res)
    print('done, total %.0fs' % res['total_runtime_s'])


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    OI = {'svd': dv.C['black'], 'rsvd_q0': dv.C['violet'], 'rsvd_q2': dv.C['blue'], 'wsvd': dv.C['orange'], 'wrsvd_q2': dv.C['yellow']}
    lab = {'svd': 'SVD', 'rsvd_q0': 'RSVD $q=0$', 'rsvd_q2': 'RSVD $q=2$', 'wsvd': 'act.-aware SVD', 'wrsvd_q2': 'act.-aware RSVD $q=2$'}
    mk = {'svd': 'o', 'rsvd_q0': 'v', 'rsvd_q2': '^', 'wsvd': 's', 'wrsvd_q2': 'D'}

    def save(fig, stem):
        fig.savefig(os.path.join(a.figdir, stem + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(a.figdir, stem + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    runs = [r for r in res['runs'] if r['label'].startswith('ratio')]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for meth in res['config']['methods']:
        rr = sorted([r for r in runs if r['method'] == meth], key=lambda r: r['param_frac_blocks'])
        x = [r['param_frac_blocks'] for r in rr]
        ax[0].semilogy(x, [r['ppl'] for r in rr], marker=mk[meth], color=OI[meth], ms=4, lw=1.2, label=lab[meth])
        ax[1].plot([r['weight_rel_err_mean'] for r in rr], [r['act_rel_err_mean'] for r in rr], marker=mk[meth],
                   color=OI[meth], ms=4, lw=1.0, ls='none')
    for r in res['runs']:
        if r['label'].startswith('energy'):
            ax[0].plot([r['param_frac_blocks']], [r['ppl']], marker='*', ms=9, color=dv.C['aqua'], mec='black', ls='none')
            ax[0].annotate(f"{int(round(100 * float(r['label'].split('=')[1])))}% energy", (r['param_frac_blocks'], r['ppl']),
                           textcoords='offset points', xytext=(-4, 6), fontsize=7, ha='right')
    ax[0].axhline(res['baseline']['ppl'], color='gray', lw=0.8, ls='--', label='dense GPT-2')
    ax[0].set_xlabel('stored parameters / dense (48 block matrices)'); ax[0].set_ylabel('WikiText-2 perplexity')
    ax[0].set_title('(a) perplexity vs. size'); ax[0].legend(frameon=False, fontsize=6.5)
    lim = [0, 1]
    ax[1].plot(lim, lim, color='gray', lw=0.7, ls='--')
    ax[1].set_xlabel(r'weight error $\|W-\hat W\|_F/\|W\|_F$ (mean)'); ax[1].set_ylabel(r'output error $\|X(W-\hat W)\|_F/\|XW\|_F$ (mean)')
    ax[1].set_title('(b) weight vs. output error'); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    fig.tight_layout(); save(fig, 'fig5_perplexity')

    lat = res['layer_latency']
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    cols = [dv.C['blue'], dv.C['yellow'], dv.C['aqua'], dv.C['orange']]
    shapes = sorted({tuple(d['shape']) for d in lat})
    for j, T in enumerate((1, 1024)):
        for c, sh in zip(cols, shapes):
            dd = sorted([d for d in lat if d['T'] == T and tuple(d['shape']) == sh], key=lambda d: d['ratio'])
            ax[j].plot([d['ratio'] for d in dd], [d['speedup'] for d in dd], 'o-', color=c, ms=4, label=f'{sh[0]}$\\times${sh[1]}')
        ax[j].axhline(1, color='gray', lw=0.7, ls='--'); ax[j].set_xlabel('stored parameters / dense')
        ax[j].set_title(f'({"ab"[j]}) one layer, {T} token{"s" if T > 1 else ""}')
    ax[0].set_ylabel('speedup (dense time / factored time)'); ax[0].legend(frameon=False)
    fig.tight_layout(); save(fig, 'fig6_latency')


if __name__ == '__main__':
    main()
