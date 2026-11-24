"""Follow-up to the KV no-go (kv_spectra.py): (1) a different rank for every head, keys only; (2) one basis per layer
shared by all 12 heads ("joint" basis). Both are tested on WikiText-2 perplexity.

Run:  .venv\\Scripts\\python.exe experiments\\kv_adaptive.py            (about 20-30 min on a laptop CPU)
      .venv\\Scripts\\python.exe experiments\\kv_adaptive.py --resume   (skips runs already saved in the JSON)
      .venv\\Scripts\\python.exe experiments\\kv_adaptive.py --replot
Writes experiments/results_kv_adaptive.json, figures/fig9_kv_perhead.{pdf,png}, figures/fig10_kv_joint.{pdf,png}.

One recipe covers every variant. For a group of cached coordinates x (one head: 64 dims, or a whole layer: 768 dims)
with calibration mean mu, covariance C and a weight matrix G (positive definite), the rank-r compressed cache stores
z = (x - mu) G^{1/2} U_r (r numbers per token) and the model uses
        x_hat = mu + (x - mu) G^{1/2} U_r U_r^T G^{-1/2},      G^{1/2} C G^{1/2} = U Lambda U^T.
This is the best rank-r linear map for the weighted error E||(x - x_hat) G^{1/2}||^2 (Eckart-Young after a change of
variables, as in activation-aware SVD); the dropped eigenvalues Lambda_{r+1..} are that error.
  * G = I, mu = 0   : plain projection on the top singular vectors, the same as kv_spectra.py.
  * G = I           : centered. Keys: the mean adds q.mu to every score of a query, which softmax ignores, so it is free.
                      Values: attention weights sum to 1, so mu passes through unchanged and is stored once per layer.
  * keys, G = Sigma_q (second moment of the queries of the same head/layer): the error is E (q.(k - k_hat))^2, the error
                      in the attention score itself, assuming q and k are independent ("query-aware").
  * values, G = W_O W_O^T (block of attn.c_proj for the head/layer): the error after the output projection ("output-aware").
Allocation of ranks:
  * uniform  : the same r for every group.
  * threshold: per head, the smallest r keeping a fixed share e of the (centered) calibration energy.
  * greedy   : keep the globally largest eigenvalues Lambda over all groups, for a given average rank. With the
               query-aware weight every eigenvalue is in the same unit (squared score error), so this is a fair
               water-filling across heads and layers. Groups may get rank 0 (then x_hat = mu).
  * in-layer : greedy, but every layer keeps the same total rank; ranks move only between the heads of one layer
               (local error units need not be comparable across layers).
Cache fraction: keys-only per-head runs store avg_r/64 of the keys and all values: (avg_r/64 + 1)/2 of the cache.
Joint runs with average rank R store R/768 of that kind. A joint basis keeps the attention exact to compute: the score
for head h is (q_h B_h)(z)^T with B_h the head's rows of the basis, so each score costs R multiply-adds instead of 64.

Decision rules, fixed before the run (dense perplexity p0 on the same 8 test windows):
  Option 1 GO  if some per-head key method reaches ppl <= 1.05 p0 with average key rank <= 32 (keys halved).
  Option 2 GO  if some joint K+V method reaches ppl <= 1.05 p0 at total cache <= 50%.
           partial if only the keys reach ppl <= 1.05 p0 at key cache <= 50% (R <= 384).
  Groups (added 2026-11-18, run with --resume --only group): one basis per group of 12/ng consecutive heads,
           ng in {1,2,3,4,6,12}. Same pass rule (ppl <= 1.05 p0 at total cache <= 50%). Reported with the attention
           arithmetic factor (avg rank_K + avg rank_V) / (2*64): multiply-adds per cached token and head relative to the
           full cache (the per-query cost of q_h B_h is ignored; it does not grow with the context).
Writes figures/fig11_kv_groups.{pdf,png} as well when group runs exist.
  Attention sink (added 2026-11-18, run with --resume --only sink): why does centering hurt per-head keys? Plain vs
           centered per-head keys (r = 16, 32, 48) with the calibration statistics computed without position 0 of each
           window ('no0'), with position 0's key stored exactly ('keep0'), or both. Rule fixed in advance: the sink
           explains it if under some treatment centered <= plain at all three ranks. Sink diagnostics (norm of the
           position-0 key relative to the others, its share of the key variance) are saved as 'sink_diag'; fig13.
"""
import argparse, json, math, os, platform, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64


def load_wikitext(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download('Salesforce/wikitext', f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset')
    return '\n\n'.join(pq.read_table(f).column('text').to_pylist())


def psd_pow(G, p, ridge=1e-6):
    G = (G + G.T) / 2
    w, Q = torch.linalg.eigh(G)
    w = w.clamp_min(ridge * float(w.clamp_min(0).mean()))
    return (Q * w ** p) @ Q.T


def group_basis(C, G):
    """Returns (T, Tinv, U, lam) with T = G^{1/2}, eigen-decomposition T C T = U diag(lam) U^T, lam descending."""
    if G is None:
        T = Tinv = torch.eye(C.shape[0], dtype=D64)
    else:
        T, Tinv = psd_pow(G, 0.5), psd_pow(G, -0.5)
    lam, U = torch.linalg.eigh(T @ C @ T)
    lam, U = lam.flip(0).clamp_min(0), U.flip(1)
    return T, Tinv, U, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_windows', type=int, default=8)
    ap.add_argument('--calib_windows', type=int, default=16)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_adaptive.json'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval_offset', type=int, default=0, help='first test window (8 = fresh windows for a confirmation run)')
    ap.add_argument('--only', nargs='*', default=None, help='run only labels containing one of these strings')
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
    L, H, E = cfg.n_layer, cfg.n_head, cfg.n_embd
    d = E // H
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    evalw = [ids_test[i * a.seq:(i + 1) * a.seq] for i in range(a.eval_offset, a.eval_offset + a.eval_windows)]
    g = torch.Generator().manual_seed(a.seed)  # same calibration windows as kv_spectra.py
    starts = torch.randint(0, ids_train.numel() - a.seq, (a.calib_windows,), generator=g).tolist()
    calw = [ids_train[s:s + a.seq] for s in starts]

    # ---- calibration statistics: sums and second moments of q, k, v per layer (768 x 768, float64)
    S1 = {kind: [torch.zeros(E, dtype=D64) for _ in range(L)] for kind in 'QKV'}
    S2 = {kind: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for kind in 'QKV'}
    P1 = {kind: [torch.zeros(E, dtype=D64) for _ in range(L)] for kind in 'QKV'}     # position 0 of each window only
    P2 = {kind: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for kind in 'QKV'}
    N = [0, 0]
    hooks = []
    for l in range(L):
        def hk(mod, args, out, l=l):
            o = out[0].to(D64)
            for i, kind in enumerate('QKV'):
                X = o[:, i * E:(i + 1) * E]
                S1[kind][l] += X.sum(0); S2[kind][l] += X.T @ X
                P1[kind][l] += X[0]; P2[kind][l] += torch.outer(X[0], X[0])
        hooks.append(model.transformer.h[l].attn.c_attn.register_forward_hook(hk))
    with torch.inference_mode():
        for w in calw:
            model(w[None]); N[0] += w.numel(); N[1] += 1
    for h_ in hooks:
        h_.remove()
    n, n0 = N
    # two versions of the statistics: 'all' tokens, and 'no0' = without position 0 of each window (the attention sink)
    STATS = {}
    for sv, (A1, A2, m) in (('all', (S1, S2, n)), ('no0', ({k: [S1[k][l] - P1[k][l] for l in range(L)] for k in 'QKV'},
                                                             {k: [S2[k][l] - P2[k][l] for l in range(L)] for k in 'QKV'}, n - n0))):
        mu_ = {k: [A1[k][l] / m for l in range(L)] for k in 'QKV'}
        m2_ = {k: [A2[k][l] / m for l in range(L)] for k in 'QKV'}
        cv_ = {k: [m2_[k][l] - torch.outer(mu_[k][l], mu_[k][l]) for l in range(L)] for k in 'KV'}
        STATS[sv] = (mu_, m2_, cv_)
    mean, mom2, cov = STATS['all']
    # sink diagnostics per layer and head (keys): RMS norm of the position-0 key over the RMS norm of the other keys, and the
    # share of the centered key variance (trace of cov, all tokens) that the position-0 tokens contribute
    sink_diag = {'key_norm_ratio': [], 'pos0_var_share': [], 'mean_shift_rel': []}
    for l in range(L):
        nr, vs, ms = [], [], []
        for h in range(H):
            s = slice(h * d, (h + 1) * d)
            e0 = float(P2['K'][l][s, s].trace()) / n0
            er = float((S2['K'][l][s, s] - P2['K'][l][s, s]).trace()) / (n - n0)
            nr.append(math.sqrt(e0 / er))
            mu = mean['K'][l][s]
            d0 = float(P2['K'][l][s, s].trace() - 2 * mu @ P1['K'][l][s] + n0 * mu @ mu)  # sum over pos-0 tokens of ||k0 - mu||^2
            vs.append(d0 / (n * float(cov['K'][l][s, s].trace())))
            ms.append(float((STATS['all'][0]['K'][l][s] - STATS['no0'][0]['K'][l][s]).norm() / STATS['no0'][0]['K'][l][s].norm()))
        sink_diag['key_norm_ratio'].append(nr); sink_diag['pos0_var_share'].append(vs); sink_diag['mean_shift_rel'].append(ms)
    print('sink: key norm ratio pos0/rest, median per layer:', [round(float(np.median(x)), 1) for x in sink_diag['key_norm_ratio']])
    print('sink: pos-0 share of centered key variance, max per layer:', [round(max(x), 3) for x in sink_diag['pos0_var_share']], flush=True)
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]  # Conv1D: y = x W, W is (in, out)
    print(f'calibration stats from {n} tokens ({time.perf_counter()-T0:.0f}s)', flush=True)

    # ---- bases. key: (kind, scope, weighting, centered) -> per layer list of groups (T, Tinv, U, lam)
    def weight(kind, wt, l, sl, sv='all'):
        if wt == 'I':
            return None
        if wt == 'q':  # sum over the heads in the slice of score errors: block-diagonal query weight
            return torch.block_diag(*[STATS[sv][1]['Q'][l][h * d:(h + 1) * d, h * d:(h + 1) * d]
                                      for h in range(sl.start // d, sl.stop // d)])
        if wt == 'o':
            return WO[l][sl] @ WO[l][sl].T
        raise ValueError(wt)

    def n_groups(scope):
        """'head' = 12 groups of 1 head, 'joint' = 1 group of all heads, 'g4' = 4 groups of 3 consecutive heads."""
        return H if scope == 'head' else 1 if scope == 'joint' else int(scope[1:])

    bases = {}

    def get_bases(kind, scope, wt, centered, sv='all'):
        key = (kind, scope, wt, centered, sv)
        if key not in bases:
            per = []
            ng = n_groups(scope)
            assert H % ng == 0
            w = E // ng
            slices = [slice(j * w, (j + 1) * w) for j in range(ng)]
            _, m2_, cv_ = STATS[sv]
            for l in range(L):
                C = cv_[kind][l] if centered else m2_[kind][l]
                per.append([(sl,) + group_basis(C[sl, sl], weight(kind, wt, l, sl, sv)) for sl in slices])
            bases[key] = per
        return bases[key]

    def allocate(B, alloc, param):
        """Ranks per layer and group."""
        if alloc == 'uniform':
            return [[int(param)] * len(B[l]) for l in range(L)]
        if alloc == 'threshold':
            out = []
            for l in range(L):
                rr = []
                for (_, _, _, _, lam) in B[l]:
                    c = torch.cumsum(lam, 0) / lam.sum()
                    rr.append(int((c >= param).nonzero()[0]) + 1)
                out.append(rr)
            return out
        if alloc == 'greedy':  # param = average rank per group
            items = [(float(v), l, j) for l in range(L) for j, grp in enumerate(B[l]) for v in grp[4]]
            n_groups = sum(len(B[l]) for l in range(L))
            budget = int(round(param * n_groups))
            items.sort(key=lambda t: -t[0])
            out = [[0] * len(B[l]) for l in range(L)]
            for _, l, j in items[:budget]:
                out[l][j] += 1
            return out
        if alloc == 'greedy_layer':  # same total rank in every layer, moved between the heads of that layer only
            out = []
            for l in range(L):
                items = sorted(((float(v), j) for j, grp in enumerate(B[l]) for v in grp[4]), key=lambda t: -t[0])
                rr = [0] * len(B[l])
                for _, j in items[:int(round(param * len(B[l])))]:
                    rr[j] += 1
                out.append(rr)
            return out
        raise ValueError(alloc)

    def projector(B, ranks, kind, centered, sv='all'):
        """Per layer: (mu, Pi) with x_hat = mu + (x - mu) Pi, Pi block-diagonal for per-head groups (float32)."""
        out = []
        for l in range(L):
            Pi = torch.zeros(E, E, dtype=D64)
            for (sl, T, Tinv, U, lam), r in zip(B[l], ranks[l]):
                Ur = U[:, :r]
                Pi[sl, sl] = T @ Ur @ Ur.T @ Tinv
            mu = STATS[sv][0][kind][l] if centered else torch.zeros(E, dtype=D64)
            out.append((mu.float(), Pi.float()))
        return out

    # ---- runs (label, {kind: (scope, weighting, centered, alloc, param)})
    hd, jt = 'head', 'joint'
    R = []
    # Option 1: per-head, keys only
    for r in (16, 32, 48):
        R.append((f'K head plain uniform r={r}', {'K': (hd, 'I', False, 'uniform', r)}))
    for r in (16, 24, 32, 40, 48):
        R.append((f'K head centered uniform r={r}', {'K': (hd, 'I', True, 'uniform', r)}))
    for e in (0.90, 0.95, 0.98, 0.99):
        R.append((f'K head centered threshold e={e}', {'K': (hd, 'I', True, 'threshold', e)}))
    for r in (8, 16, 24, 32):
        R.append((f'K head centered greedy r={r}', {'K': (hd, 'I', True, 'greedy', r)}))
    for r in (16, 24, 32):
        R.append((f'K head query-aware uniform r={r}', {'K': (hd, 'q', True, 'uniform', r)}))
    for r in (4, 8, 16, 24, 32):
        R.append((f'K head query-aware greedy r={r}', {'K': (hd, 'q', True, 'greedy', r)}))
    # Option 2: joint basis per layer
    for r in (64, 128, 192, 256, 384):
        R.append((f'K joint centered uniform R={r}', {'K': (jt, 'I', True, 'uniform', r)}))
        R.append((f'K joint query-aware uniform R={r}', {'K': (jt, 'q', True, 'uniform', r)}))
    for r in (32, 64, 128, 192):
        R.append((f'K joint query-aware greedy R={r}', {'K': (jt, 'q', True, 'greedy', r)}))
    for r in (128, 256, 384, 512):
        R.append((f'V joint centered uniform R={r}', {'V': (jt, 'I', True, 'uniform', r)}))
        R.append((f'V joint output-aware uniform R={r}', {'V': (jt, 'o', True, 'uniform', r)}))
    for r in (128, 256, 384):
        R.append((f'V joint output-aware greedy R={r}', {'V': (jt, 'o', True, 'greedy', r)}))
    for r in (16, 24, 32):
        R.append((f'K head query-aware in-layer r={r}', {'K': (hd, 'q', True, 'greedy_layer', r)}))
    for r in (16, 32):
        R.append((f'V head output-aware greedy r={r}', {'V': (hd, 'o', True, 'greedy', r)}))
        R.append((f'V head output-aware in-layer r={r}', {'V': (hd, 'o', True, 'greedy_layer', r)}))
    for rk, rv in ((64, 128), (128, 256), (192, 384)):
        R.append((f'KV joint aware greedy R={rk}/{rv}', {'K': (jt, 'q', True, 'greedy', rk), 'V': (jt, 'o', True, 'greedy', rv)}))
    # added after the first pass (chosen from the K-only and V-only results on the same test windows; post hoc)
    for rk, rv in ((192, 384), (256, 384), (256, 512)):
        R.append((f'KV joint aware uniform R={rk}/{rv}', {'K': (jt, 'q', True, 'uniform', rk), 'V': (jt, 'o', True, 'uniform', rv)}))
    for rk, rv in ((8, 16), (16, 32), (24, 48)):
        R.append((f'KV head aware greedy r={rk}/{rv}', {'K': (hd, 'q', True, 'greedy', rk), 'V': (hd, 'o', True, 'greedy', rv)}))
    # Step 3 (2026-11-18): bases shared by groups of consecutive heads, ng groups of 12/ng heads (width W = 768/ng).
    # Main setting: the split that passed for the joint basis, keys 1/3 and values 2/3 of each group's width, which is
    # exactly 50% of the cache. Variants: an even split (1/2, 1/2), and in-layer allocation (rank moved between the groups
    # of a layer). ng = 1 repeats the joint result and ng = 12 is the per-head case on the same footing.
    for ng in (1, 2, 3, 4, 6, 12):
        W = E // ng
        rk = round(W / 3); rv = W - rk
        sc = f'g{ng}'
        R.append((f'KV group{ng} aware uniform R={rk}/{rv}', {'K': (sc, 'q', True, 'uniform', rk), 'V': (sc, 'o', True, 'uniform', rv)}))
        R.append((f'KV group{ng} aware uniform R={W // 2}/{W // 2}', {'K': (sc, 'q', True, 'uniform', W // 2), 'V': (sc, 'o', True, 'uniform', W // 2)}))
        if ng > 1:
            R.append((f'KV group{ng} aware in-layer R={rk}/{rv}', {'K': (sc, 'q', True, 'greedy_layer', rk), 'V': (sc, 'o', True, 'greedy_layer', rv)}))
        R.append((f'K group{ng} query-aware uniform R={rk}', {'K': (sc, 'q', True, 'uniform', rk)}))
        R.append((f'V group{ng} output-aware uniform R={rv}', {'V': (sc, 'o', True, 'uniform', rv)}))
    # Step 5 (2026-11-18): does the attention sink (position 0 of each window) explain why centering hurts per-head keys?
    # Treatments: 'no0' = calibration statistics without position 0; 'keep0' = position 0's key stored exactly (1/1024 of
    # the key cache extra); 'both'. Rule fixed in advance: explained if under some treatment centered <= plain at r=16,32,48.
    SINK = {'no0': {'stats': 'no0'}, 'keep0': {'keep0': True}, 'both': {'stats': 'no0', 'keep0': True}}
    for r in (16, 32, 48):
        for c, cen in (('plain', False), ('centered', True)):
            for tv, opt in SINK.items():
                R.append((f'sink K head {c} uniform r={r} {tv}', {'K': (hd, 'I', cen, 'uniform', r, opt)}))
    for tv in ('keep0', 'both'):  # does it also help the best per-head method and the passing per-layer setting?
        R.append((f'sink K head query-aware in-layer r=32 {tv}', {'K': (hd, 'q', True, 'greedy_layer', 32, SINK[tv])}))
        R.append((f'sink KV joint aware uniform R=256/512 {tv}', {'K': (jt, 'q', True, 'uniform', 256, SINK[tv]),
                                                                  'V': (jt, 'o', True, 'uniform', 512, SINK[tv])}))

    # ---- evaluation hook
    active = {}
    keep0 = set()  # kinds whose position-0 entry is stored exactly

    def proj_hook(mod, args, out, l):
        if not active:
            return out
        o = out.clone()
        for kind, off in (('K', E), ('V', 2 * E)):
            if kind in active:
                mu, Pi = active[kind][l]
                X = o[..., off:off + E]
                o[..., off:off + E] = mu + (X - mu) @ Pi
                if kind in keep0:
                    o[..., 0, off:off + E] = out[..., 0, off:off + E]
        return o

    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: proj_hook(m, a_, o, l)) for l in range(L)]

    @torch.inference_mode()
    def evaluate():
        nll, cnt = 0.0, 0
        for w in evalw:
            out = model(w[None], labels=w[None]); k = w.numel() - 1
            nll += float(out.loss) * k; cnt += k
        return math.exp(nll / cnt)

    res = None
    if a.resume and os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)
    if res is None:
        res = {'config': vars(a), 'torch': torch.__version__, 'model': 'gpt2', 'n_layer': L, 'n_head': H, 'd_head': d,
               'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                            'torch_threads': torch.get_num_threads()},
               'eval_tokens': sum(w.numel() for w in evalw), 'calib_tokens': n, 'runs': []}
        active.clear()
        res['dense_ppl'] = evaluate()
    res['sink_diag'] = sink_diag
    done = {r['label'] for r in res['runs']}
    p0 = res['dense_ppl']
    print(f'dense ppl {p0:.2f} ({time.perf_counter()-T0:.0f}s)', flush=True)

    def save():
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)

    save()
    for label, spec in R:
        if label in done or (a.only and not any(s in label for s in a.only)):
            continue
        active.clear(); keep0.clear()
        info = {'label': label}
        frac = {'K': 1.0, 'V': 1.0}
        arith = {'K': 1.0, 'V': 1.0}  # multiply-adds per cached token and head, relative to the full 64
        for kind, (scope, wt, centered, alloc, param, *opt) in spec.items():
            opt = opt[0] if opt else {}
            sv = opt.get('stats', 'all')
            B = get_bases(kind, scope, wt, centered, sv)
            ranks = allocate(B, alloc, param)
            active[kind] = projector(B, ranks, kind, centered, sv)
            if opt.get('keep0'):
                keep0.add(kind)
            if opt:
                info.setdefault('sink_options', {})[kind] = opt
            width = E // n_groups(scope)
            avg = float(np.mean([r for rr in ranks for r in rr]))
            frac[kind] = avg / width
            arith[kind] = avg / d  # each head works on its group's rank-r code (groups have equal size)
            # weighted error left on calibration data (sum of dropped eigenvalues / total), all groups together
            kept = sum(float(grp[4][:r].sum()) for l in range(L) for grp, r in zip(B[l], ranks[l]))
            tot = sum(float(grp[4].sum()) for l in range(L) for grp in B[l])
            info[kind] = {'scope': scope, 'weight': wt, 'centered': centered, 'alloc': alloc, 'param': param,
                          'avg_rank': avg, 'ranks': ranks, 'calib_rel_err': 1 - kept / tot}
        info['cache_frac'] = {'K': frac['K'], 'V': frac['V'], 'total': (frac['K'] + frac['V']) / 2}
        info['attn_arith_factor'] = {'K': arith['K'], 'V': arith['V'], 'total': (arith['K'] + arith['V']) / 2}
        t = time.perf_counter()
        info['ppl'] = evaluate()
        info['ppl_ratio'] = info['ppl'] / p0
        res['runs'].append(info)
        save()
        print(f"  {label:42s} ppl={info['ppl']:9.2f} ({info['ppl_ratio']:.3f}x) cache={info['cache_frac']['total']:.3f} "
              f"[{time.perf_counter()-t:.0f}s, total {time.perf_counter()-T0:.0f}s]", flush=True)
    for h_ in hooks:
        h_.remove()

    # ---- decisions (rules from the docstring)
    tol = 1.05 * p0
    runs = res['runs']
    o1 = [r for r in runs if r['label'].startswith('K head') and r['K']['avg_rank'] <= 32 and r['ppl'] <= tol]
    o2 = [r for r in runs if r['label'].startswith('KV joint') and r['cache_frac']['total'] <= 0.5 and r['ppl'] <= tol]
    o2k = [r for r in runs if r['label'].startswith('K joint') and r['cache_frac']['K'] <= 0.5 and r['ppl'] <= tol]
    o3 = [r for r in runs if r['label'].startswith('KV group') and r['cache_frac']['total'] <= 0.5 + 1e-9 and r['ppl'] <= tol]
    byl = {r['label']: r['ppl'] for r in runs}
    sink_dec = {}
    for tv in ('no0', 'keep0', 'both'):
        pairs = {r: (byl.get(f'sink K head centered uniform r={r} {tv}'), byl.get(f'sink K head plain uniform r={r} {tv}'))
                 for r in (16, 32, 48)}
        if all(None not in p for p in pairs.values()):
            sink_dec[tv] = {'centered_vs_plain': pairs, 'explained': all(c <= p for c, p in pairs.values())}
    res['sink_decision'] = sink_dec
    res['decision'] = {'tolerance_ppl': tol,
                       'groups_passing': [r['label'] for r in o3],
                       'groups_min_arith_passing': min((r['attn_arith_factor']['total'] for r in o3), default=None),
                       'option1_go': bool(o1), 'option1_passing': [r['label'] for r in o1],
                       'option2_go': bool(o2), 'option2_passing': [r['label'] for r in o2],
                       'option2_keys_partial': bool(o2k), 'option2_keys_passing': [r['label'] for r in o2k]}
    res['total_runtime_s'] = res.get('total_runtime_s', 0) + time.perf_counter() - T0
    save()
    print('decision:', json.dumps(res['decision'], indent=1))
    make_figures(a, res)
    print('done, total %.0fs' % (time.perf_counter() - T0))


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    p0, L = res['dense_ppl'], res['n_layer']
    runs = res['runs']
    cols = [dv.C['blue'], dv.C['orange'], dv.C['aqua'], dv.C['magenta'], dv.C['yellow'], dv.C['violet'], dv.C['black'], dv.C['gray']]

    def save(fig, stem):
        fig.savefig(os.path.join(a.figdir, stem + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(a.figdir, stem + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    def series(prefix, kind):
        rr = [r for r in runs if r['label'].startswith(prefix)]
        rr.sort(key=lambda r: r[kind]['avg_rank'])
        return rr

    def ref_lines(ax):
        ax.axhline(p0, color='gray', ls='--', lw=0.8, label=f'full cache ({p0:.1f})')
        ax.axhline(1.05 * p0, color='gray', ls=':', lw=0.8, label='+5% (pass line)')

    # fig 9: option 1, per-head keys
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for i, (prefix, lab) in enumerate([('K head plain uniform', 'plain, same rank'),
                                       ('K head centered uniform', 'centered, same rank'),
                                       ('K head centered threshold', 'centered, energy threshold'),
                                       ('K head centered greedy', 'centered, greedy'),
                                       ('K head query-aware uniform', 'query-aware, same rank'),
                                       ('K head query-aware greedy', 'query-aware, greedy'),
                                       ('K head query-aware in-layer', 'query-aware, in-layer')]):
        rr = series(prefix, 'K')
        if rr:
            ax[0].semilogy([r['K']['avg_rank'] for r in rr], [r['ppl'] for r in rr], 'o-', ms=3.5, color=cols[i], label=lab)
    ref_lines(ax[0])
    ax[0].axvline(32, color='k', lw=0.6, ls='--')
    ax[0].set_xlabel('average key rank per head (of 64)'); ax[0].set_ylabel('WikiText-2 perplexity')
    ax[0].set_title('(a) keys only, per-head bases')
    ax[0].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    best = [r for r in runs if r['label'].startswith('K head query-aware greedy') and r['K']['param'] == 16]
    if best:
        m = np.array(best[0]['K']['ranks'])
        im = ax[1].imshow(m.T, aspect='auto', cmap='viridis', vmin=0, vmax=64, origin='lower')
        ax[1].set_xlabel('layer'); ax[1].set_ylabel('head'); ax[1].grid(False)
        ax[1].set_xticks(range(L)); ax[1].set_title('(b) ranks chosen (query-aware greedy, avg 16)')
        fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); save(fig, 'fig9_kv_perhead')

    # fig 10: option 2, joint bases, ppl against total cache fraction
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for i, (prefix, lab) in enumerate([('K joint centered uniform', 'joint, centered'),
                                       ('K joint query-aware uniform', 'joint, query-aware'),
                                       ('K joint query-aware greedy', 'joint, query-aware, greedy'),
                                       ('K head query-aware greedy', 'per-head, query-aware, greedy')]):
        rr = series(prefix, 'K')
        if rr:
            ax[0].semilogy([r['cache_frac']['K'] for r in rr], [r['ppl'] for r in rr], 'o-', ms=3.5, color=cols[i], label=lab)
    ref_lines(ax[0])
    ax[0].set_xlabel('fraction of the key cache kept'); ax[0].set_ylabel('WikiText-2 perplexity')
    ax[0].set_title('(a) keys only'); ax[0].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    for i, (prefix, kind, lab) in enumerate([('V joint centered uniform', 'V', 'V joint, centered'),
                                             ('V joint output-aware uniform', 'V', 'V joint, output-aware'),
                                             ('V joint output-aware greedy', 'V', 'V joint, output-aware, greedy'),
                                             ('V head output-aware greedy', 'V', 'V per-head, output-aware, greedy'),
                                             ('V head output-aware in-layer', 'V', 'V per-head, output-aware, in-layer'),
                                             ('KV joint aware greedy', 'K', 'K+V joint, aware, greedy'),
                                             ('KV joint aware uniform', 'K', 'K+V joint, aware (post hoc)'),
                                             ('KV head aware greedy', 'K', 'K+V per-head, aware, greedy')]):
        rr = series(prefix, kind)
        if rr:
            ax[1].semilogy([r['cache_frac']['total'] for r in rr], [r['ppl'] for r in rr], 'o-', ms=3.5, color=cols[i], label=lab)
    ref_lines(ax[1])
    ax[1].axvline(0.5, color='k', lw=0.6, ls='--')
    ax[1].set_xlabel('fraction of the whole KV cache kept'); ax[1].set_title('(b) values, and keys + values')
    ax[1].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout(); save(fig, 'fig10_kv_joint')

    make_fig13(a, res, plt, save, cols)
    # fig 11: bases shared by groups of heads (step 3)
    gr = [r for r in runs if ' group' in r['label']]
    if not gr:
        return
    ngs = sorted({int(r['label'].split()[1][5:]) for r in gr})

    def pick(prefix, ng):
        rr = [r for r in gr if r['label'].startswith(f'{prefix.split()[0]} group{ng} {" ".join(prefix.split()[1:])}')]
        return rr[0] if rr else None

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for i, (prefix, lab) in enumerate([('KV aware uniform R', 'K 1/3 + V 2/3, same rank'),
                                       ('KV aware in-layer', 'K 1/3 + V 2/3, in-layer'),
                                       ]):
        pts = [(ng, pick(prefix, ng)) for ng in ngs]
        pts = [(ng, r) for ng, r in pts if r and (prefix != 'KV aware uniform R' or r['cache_frac']['K'] < 0.4)]
        if pts:
            xs = [r['attn_arith_factor']['total'] for _, r in pts]
            ax[0].semilogy(xs, [r['ppl'] for _, r in pts], 'o-', ms=3.5, color=cols[i], label=lab)
            if i == 0:
                for (ng, r), x in zip(pts, xs):
                    ax[0].annotate(f'{ng}', (x, r['ppl']), textcoords='offset points', xytext=(4, 3), fontsize=6)
    half = [(ng, r) for ng in ngs for r in gr if r['label'].startswith(f'KV group{ng} aware uniform') and abs(r['cache_frac']['K'] - 0.5) < 0.01]
    if half:
        ax[0].semilogy([r['attn_arith_factor']['total'] for _, r in half], [r['ppl'] for _, r in half], 's--', ms=3,
                       color=cols[2], label='K 1/2 + V 1/2, same rank')
    ref_lines(ax[0])
    from matplotlib.ticker import NullFormatter, NullLocator
    ax[0].set_xscale('log'); ax[0].xaxis.set_minor_locator(NullLocator()); ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_xticks([0.5, 1, 1.5, 2, 3, 6]); ax[0].set_xticklabels(['0.5', '1', '1.5', '2', '3', '6'])
    ax[0].set_xlabel('attention arithmetic vs. full cache (x)'); ax[0].set_ylabel('WikiText-2 perplexity')
    ax[0].set_title('(a) half the cache; labels = number of groups')
    ax[0].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    for i, (prefix, lab) in enumerate([('K query-aware uniform', 'keys only, 1/3 kept'),
                                       ('V output-aware uniform', 'values only, 2/3 kept')]):
        pts = [(ng, pick(prefix, ng)) for ng in ngs]
        pts = [(ng, r) for ng, r in pts if r]
        ax[1].semilogy([ng for ng, _ in pts], [r['ppl'] for _, r in pts], 'o-', ms=3.5, color=cols[i + 3], label=lab)
    ref_lines(ax[1])
    ax[1].set_xscale('log'); ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xticks(ngs); ax[1].set_xticklabels([str(x) for x in ngs])
    ax[1].set_xlabel('head groups per layer (1 = one basis per layer)')
    ax[1].set_ylabel('WikiText-2 perplexity'); ax[1].set_title('(b) keys and values separately')
    ax[1].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout(); save(fig, 'fig11_kv_groups')


def make_fig13(a, res, plt, save, cols):
    """Attention sink: (a) position-0 key norm relative to the other keys, per layer; (b) plain vs centered per-head keys."""
    if 'sink_diag' not in res or not any(r['label'].startswith('sink') for r in res['runs']):
        return
    p0, L = res['dense_ppl'], res['n_layer']
    byl = {r['label']: r['ppl'] for r in res['runs']}
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={'width_ratios': [1, 1.3]})
    mech_f = os.path.join(ROOT, 'experiments', 'results_kv_sink_mech.json')  # from kv_sink_check.py
    if os.path.exists(mech_f):
        with open(mech_f) as f:
            m = json.load(f)['per_layer_median']
        ls_ = range(L)
        tw = ax[0].twinx()
        tw.bar(ls_, m['attn0'], 0.7, color='#dddddd', zorder=0)
        tw.set_ylim(0, 1); tw.set_ylabel('attention on first token (bars)', color='#777777', fontsize=7)
        tw.spines['right'].set_visible(True); tw.grid(False)
        ax[0].set_zorder(tw.get_zorder() + 1); ax[0].patch.set_visible(False)
        for key, c, ls, lab in (('err0_centered', cols[1], '-', 'first key, centered'), ('err0_plain', cols[0], '-', 'first key, plain'),
                                ('errrest_centered', cols[1], ':', 'other keys, centered'), ('errrest_plain', cols[0], ':', 'other keys, plain')):
            ax[0].semilogy(ls_, m[key], marker='o', ms=3, ls=ls, color=c, label=lab)
        ax[0].set_xlabel('layer'); ax[0].set_ylabel('mean score error (logits), r=32')
        ax[0].set_xticks(list(ls_))
        ax[0].set_title('(a) where the error goes')
        ax[0].legend(frameon=False, fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    else:
        nr = np.array(res['sink_diag']['key_norm_ratio'])  # (L, H)
        ax[0].boxplot(nr.T, positions=range(L), widths=0.6, flierprops={'markersize': 2})
        ax[0].set_yscale('log'); ax[0].axhline(1, color='gray', lw=0.8)
        ax[0].set_xlabel('layer'); ax[0].set_ylabel('norm of first key / typical key')
        ax[0].set_title('(a) first-token keys, 12 heads per layer')
    treat = [('', 'none'), ('no0', 'stats\nwithout pos. 0'), ('keep0', 'pos. 0\nkept exact'), ('both', 'both')]
    x = np.arange(len(treat)); w = 0.13
    for i, r in enumerate((16, 32, 48)):
        for j, (c, base) in enumerate((('plain', 'K head plain uniform'), ('centered', 'K head centered uniform'))):
            ys = [byl.get(f'{base} r={r}') if t == '' else byl.get(f'sink K head {c} uniform r={r} {t}') for t, _ in treat]
            ys = [np.nan if y is None else y for y in ys]
            ax[1].bar(x + (2 * i + j - 2.5) * w, ys, w, color=cols[i], alpha=1.0 if j == 0 else 0.45,
                      label=f'r={r}, {c}', edgecolor='white', linewidth=0.3)
    ax[1].axhline(p0, color='gray', ls='--', lw=0.8)
    ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels([t for _, t in treat], fontsize=7)
    ax[1].set_ylabel('WikiText-2 perplexity'); ax[1].set_title('(b) per-head keys: plain vs centered')
    ax[1].legend(frameon=False, fontsize=6, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); save(fig, 'fig13_kv_sink')


if __name__ == '__main__':
    main()
