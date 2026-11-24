"""KV-cache go/no-go test: are GPT-2's cached keys and values low rank, and does a low-rank cache keep perplexity?

Run:  .venv\\Scripts\\python.exe experiments\\kv_spectra.py            (about 15-25 min on a laptop CPU)
      .venv\\Scripts\\python.exe experiments\\kv_spectra.py --replot
Writes experiments/results_kv.json and figures/fig7_kv_spectra.{pdf,png}, figures/fig8_kv_perplexity.{pdf,png}.

For every layer l and head h, the cache holds K_{l,h}, V_{l,h} in R^{T x d_head} (GPT-2 small: 12 layers, 12 heads,
d_head = 64; no rotary embedding, so the cached keys are exactly the c_attn outputs).
  1. Spectra (per sequence): singular values of K and V for each test window; rank for 90/95/99% energy (Prop. 3.4),
     reported as a fraction of d_head. Also per layer with all heads joined (T x 768).
  2. Offline basis (as in Eigen Attention / Palu): P_{l,h} = top-r right singular vectors of the K (or V) rows stacked
     over calibration windows from the train split; energy that P captures on held-out test windows.
  3. Function space: replace k by k P P^T and/or v by v P_v P_v^T inside the model (a rank-r cache stores T*r numbers
     per head instead of T*64) and measure WikiText-2 perplexity and the relative attention-output error.
Go criterion (from meeting_prep.pdf): most layers need rank < d_head/2 = 32 for 95% energy.
"""
import argparse, json, math, os, platform, statistics, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_compression import rsvd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64


def load_wikitext(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download('Salesforce/wikitext', f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset')
    return '\n\n'.join(pq.read_table(f).column('text').to_pylist())


def energy_rank(s, e):
    c = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return int((c >= e).nonzero()[0]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_windows', type=int, default=8)
    ap.add_argument('--calib_windows', type=int, default=16)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--ranks', type=int, nargs='+', default=[8, 16, 32, 48])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv.json'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated from', a.out)
        return
    T0 = time.perf_counter()
    torch.set_num_threads(a.threads)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    cfg = model.config
    L, H = cfg.n_layer, cfg.n_head
    d = cfg.n_embd // H
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    evalw = [ids_test[i * a.seq:(i + 1) * a.seq] for i in range(a.eval_windows)]
    g = torch.Generator().manual_seed(a.seed)
    starts = torch.randint(0, ids_train.numel() - a.seq, (a.calib_windows,), generator=g).tolist()
    calw = [ids_train[s:s + a.seq] for s in starts]
    E = cfg.n_embd

    def capture(windows):
        """Returns K[l], V[l]: list over windows of (T, H, d) float32 tensors (cast to float64 for every SVD)."""
        K = [[] for _ in range(L)]; V = [[] for _ in range(L)]
        hooks = []
        for l in range(L):
            def hk(mod, args, out, l=l):
                o = out[0].float()
                K[l].append(o[:, E:2 * E].reshape(-1, H, d).clone()); V[l].append(o[:, 2 * E:].reshape(-1, H, d).clone())
            hooks.append(model.transformer.h[l].attn.c_attn.register_forward_hook(hk))
        with torch.inference_mode():
            for w in windows:
                model(w[None])
        for h_ in hooks:
            h_.remove()
        return K, V

    Kt, Vt = capture(evalw)
    Kc, Vc = capture(calw)  # ~1.8 GB in float32 together with Kt, Vt; Kc, Vc are freed after the basis is built
    print(f'captured caches ({time.perf_counter()-T0:.0f}s)', flush=True)

    res = {'config': vars(a), 'torch': torch.__version__, 'model': 'gpt2', 'n_layer': L, 'n_head': H, 'd_head': d,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                        'torch_threads': torch.get_num_threads()},
           'eval_tokens': sum(w.numel() for w in evalw), 'calib_tokens': sum(w.numel() for w in calw)}

    # ---- 1. per-sequence spectra
    energies = (0.90, 0.95, 0.99)
    spec = {}
    for kind, C in (('K', Kt), ('V', Vt)):
        per = np.zeros((L, H, len(energies), len(evalw)))
        joint = np.zeros((L, len(energies), len(evalw)))
        centered95 = np.zeros((L, H, len(evalw)))
        mean_sv = np.zeros((L, H, d))
        for l in range(L):
            for w, X in enumerate(C[l]):
                X = X.to(D64)
                for h in range(H):
                    s = torch.linalg.svdvals(X[:, h]); mean_sv[l, h] += (s / s[0]).numpy() / len(evalw)
                    for j, e in enumerate(energies):
                        per[l, h, j, w] = energy_rank(s, e)
                    Xc = X[:, h] - X[:, h].mean(0)
                    centered95[l, h, w] = energy_rank(torch.linalg.svdvals(Xc), 0.95)
                sj = torch.linalg.svdvals(X.reshape(X.shape[0], -1))
                for j, e in enumerate(energies):
                    joint[l, j, w] = energy_rank(sj, e)
        spec[kind] = {'rank_per_head_mean_over_windows': {str(e): per[:, :, j].mean(-1).tolist() for j, e in enumerate(energies)},
                      'rank_joint_mean_over_windows': {str(e): joint[:, j].mean(-1).tolist() for j, e in enumerate(energies)},
                      'rank95_centered_per_head': centered95.mean(-1).tolist(),
                      'normalized_sv_mean': mean_sv.tolist()}
        r95 = per[:, :, 1].mean(-1)
        spec[kind]['layer_median_rank95'] = np.median(r95, 1).tolist()
        spec[kind]['frac_heads_rank95_below_half'] = float((r95 < d / 2).mean())
        spec[kind]['n_layers_median_below_half'] = int((np.median(r95, 1) < d / 2).sum())
    res['spectra'] = spec
    res['go'] = {kind: spec[kind]['n_layers_median_below_half'] > L / 2 for kind in ('K', 'V')}
    print('go/no-go (most layers median rank95 < d/2):', res['go'], f'({time.perf_counter()-T0:.0f}s)', flush=True)

    # ---- 2. offline basis from calibration, captured energy on held-out windows
    P = {}
    held = {}
    for kind, Cc, Ct in (('K', Kc, Kt), ('V', Vc, Vt)):
        cap = np.zeros((L, H, len(a.ranks)))
        for l in range(L):
            Xc = torch.cat(Cc[l], 0).to(D64)  # (N, H, d)
            Xt = torch.cat(Ct[l], 0).to(D64)
            for h in range(H):
                _, _, Vh = torch.linalg.svd(Xc[:, h], full_matrices=False)
                P[(kind, l, h)] = Vh.T  # (d, d), columns ordered by singular value
                tot = float((Xt[:, h] ** 2).sum())
                for j, r in enumerate(a.ranks):
                    cap[l, h, j] = float(((Xt[:, h] @ Vh.T[:, :r]) ** 2).sum()) / tot
        held[kind] = {str(r): cap[:, :, j].tolist() for j, r in enumerate(a.ranks)}
    res['offline_basis_heldout_energy'] = held
    del Kc, Vc

    # ---- 3. function space: perplexity and attention-output error with rank-r K and/or V
    mode = {'r': None, 'which': ''}
    Pt = {key: v.float() for key, v in P.items()}

    def proj_hook(mod, args, out, l):
        if mode['r'] is None:
            return out
        o = out.clone()
        r = mode['r']
        for kind, off in (('K', E), ('V', 2 * E)):
            if kind not in mode['which']:
                continue
            X = o[..., off:off + E].reshape(*o.shape[:-1], H, d)
            for h in range(H):
                Pr = Pt[(kind, l, h)][:, :r]
                X[..., h, :] = (X[..., h, :] @ Pr) @ Pr.T
            o[..., off:off + E] = X.reshape(*o.shape[:-1], E)
        return o

    attn_out = {}

    def attn_hook(mod, args, l):  # forward *pre*-hook: receives (module, args) only
        if mode.get('record'):
            attn_out.setdefault(l, []).append(args[0].detach().float().clone())  # input to attn.c_proj = merged heads

    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: proj_hook(m, a_, o, l)) for l in range(L)]
    hooks += [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m, a_, l=l: attn_hook(m, a_, l)) for l in range(L)]

    @torch.inference_mode()
    def evaluate(record=False):
        mode['record'] = record
        nll, n = 0.0, 0
        for w in evalw:
            out = model(w[None], labels=w[None]); k = w.numel() - 1
            nll += float(out.loss) * k; n += k
        mode['record'] = False
        return math.exp(nll / n)

    ppl0 = evaluate(record=True)
    ref = {l: torch.cat(v, 0) for l, v in attn_out.items()}; attn_out.clear()
    runs = [{'which': 'none', 'r': d, 'ppl': ppl0, 'cache_frac': 1.0, 'attn_rel_err_mean': 0.0}]
    print(f'baseline ppl {ppl0:.2f} ({time.perf_counter()-T0:.0f}s)', flush=True)
    for which in ('K', 'V', 'KV'):
        for r in a.ranks:
            mode['r'], mode['which'] = r, which
            ppl = evaluate(record=True)
            errs = [float(torch.linalg.norm(torch.cat(attn_out[l], 0) - ref[l]) / torch.linalg.norm(ref[l])) for l in range(L)]
            attn_out.clear()
            frac = (r / d + 1) / 2 if which != 'KV' else r / d
            runs.append({'which': which, 'r': r, 'ppl': ppl, 'cache_frac': frac,
                         'attn_rel_err_per_layer': errs, 'attn_rel_err_mean': float(np.mean(errs))})
            print(f'  {which:>2s} r={r:3d} ppl={ppl:9.2f} cache={frac:.3f} attn_err={np.mean(errs):.3f} ({time.perf_counter()-T0:.0f}s)', flush=True)
    mode['r'] = None
    for h_ in hooks:
        h_.remove()
    res['function_space'] = runs

    # ---- 4. cost of computing a basis online for one layer's joined cache (T x 768): exact SVD vs RSVD
    X = Kt[5][0].reshape(a.seq, -1).to(D64)  # first test window, layer 5, all heads joined: T x 768
    tm = []
    for fn, lab in ((lambda: torch.linalg.svd(X, full_matrices=False), 'svd'),
                    (lambda: rsvd(X, 128, 10, 1, torch.Generator().manual_seed(0)), 'rsvd_k128_q1')):
        fn(); ts = []
        for _ in range(5):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        tm.append({'method': lab, 'median_s': statistics.median(ts)})
    res['basis_timing_layer5_joint'] = tm
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    make_figures(a, res)
    print('done, total %.0fs' % res['total_runtime_s'])


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    d, L = res['d_head'], res['n_layer']
    col = {'K': dv.C['blue'], 'V': dv.C['orange'], 'KV': dv.C['aqua']}

    def save(fig, stem):
        fig.savefig(os.path.join(a.figdir, stem + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(a.figdir, stem + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
    layers = np.arange(L)
    for kind in ('K', 'V'):
        r95 = np.array(res['spectra'][kind]['rank_per_head_mean_over_windows']['0.95'])
        ax[0].plot(layers, np.median(r95, 1), 'o-', color=col[kind], ms=4, label=f'{kind}: median head')
        ax[0].fill_between(layers, r95.min(1), r95.max(1), color=col[kind], alpha=0.15, lw=0)
        sv = np.array(res['spectra'][kind]['normalized_sv_mean']).mean((0, 1))
        ax[1].semilogy(np.arange(1, d + 1), sv, color=col[kind], lw=1.3, label=kind)
    ax[0].axhline(d / 2, color='k', lw=0.8, ls='--', label='go threshold $d_{head}/2$')
    ax[0].set_xlabel('layer'); ax[0].set_ylabel('rank for 95% energy'); ax[0].set_ylim(0, d); ax[0].set_xticks(layers)
    ax[0].set_title('(a) per-head rank (median, min-max over heads)'); ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].set_xlabel('index $i$'); ax[1].set_ylabel(r'$\sigma_i/\sigma_1$ (mean over layers, heads)')
    ax[1].set_title('(b) normalized spectrum'); ax[1].legend(frameon=False)
    fig.tight_layout(); save(fig, 'fig7_kv_spectra')

    fs = res['function_space']; base = fs[0]['ppl']
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for which in ('K', 'V', 'KV'):
        rr = sorted([r for r in fs if r['which'] == which], key=lambda r: r['r'])
        ax[0].semilogy([r['r'] for r in rr], [r['ppl'] for r in rr], 'o-', color=col[which], ms=4, label=f'rank-$r$ {which}')
        ax[1].plot([r['r'] for r in rr], [r['attn_rel_err_mean'] for r in rr], 'o-', color=col[which], ms=4, label=which)
    ax[0].axhline(base, color='gray', ls='--', lw=0.8, label='full cache')
    ax[0].set_xlabel('rank $r$ per head (of 64)'); ax[0].set_ylabel('WikiText-2 perplexity'); ax[0].set_title('(a) perplexity')
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].set_xlabel('rank $r$ per head (of 64)'); ax[1].set_ylabel('relative attention-output error (mean over layers)')
    ax[1].set_title('(b) attention-output error'); ax[1].legend(frameon=False)
    fig.tight_layout(); save(fig, 'fig8_kv_perplexity')


if __name__ == '__main__':
    main()
