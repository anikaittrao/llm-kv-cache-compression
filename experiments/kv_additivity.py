"""C23 (wave 4, 2026-11-24): is the excess loss of compressing several layers the sum of the single-layer excesses?

Setting: GPT-2, offline per-layer basis (kvlab.Ctx.offline: 16 train windows, seed 0), K256 / V512 shared by the 12 heads of a layer,
query-aware keys / output-aware values, first token NOT kept exact, full 1024-token passes of test windows 0-7 (keys and values of
every position compressed inside c_attn, as kv_adaptive.py). Layers outside the subset get the identity map (mu = 0, Pi = I), which
is exact; the all-identity run is checked against the dense run (max |nll diff| must be 0).
d(S) = mean NLL(S compressed) - mean NLL(dense), nats/token over the 8 windows (8184 predictions).
Subsets: 12 singles; all 66 pairs; 30 random triples; 20 random subsets each of size 4, 6, 8 (seed 0); halves (layers 0-5,
6-11); even / odd layers; all 12.
Rules fixed before the run (s(S) = sum of single d(l), l in S; ratio(S) = d(S) / s(S)):
  A1 full model additive : |ratio(all 12) - 1| <= 0.20.
  A2 subsets additive    : the median over the random subsets of size 4, 6 and 8 (60 subsets) of |ratio - 1| <= 0.20.
  A3 pairs              : median over pairs of |d(i,j) - d(i) - d(j)| / (d(i) + d(j)) <= 0.20.
  A4 sign (descriptive) : fraction of pairs with d(i,j) > d(i) + d(j) (super-additive) is reported; no threshold.
Also a code check: all 12 layers = the kv_adaptive 'KV joint aware uniform R=256/512' result (34.98 on 8 windows) up to the small
difference of calibration weights (not asserted; printed).
Paired bootstrap CI over windows (stats_utils) for ratio(all 12) and for the median |ratio-1| of A2 is NOT computed; A2/A3 are
descriptive over subsets, the CI for A1 uses window resampling.
Run:  .venv\\Scripts\\python.exe experiments\\kv_additivity.py --threads 8 --out experiments\\rerun_2026-11-22\\c23_kv_additivity.json
      smoke: --windows 1 --smoke
"""
import argparse, itertools, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import Ctx, L, E, D64, ROOT, Pi, base_result, save_json, rss_mb, offline_state  # noqa: E402
from stats_utils import boot_idx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--RK', type=int, default=256)
    ap.add_argument('--RV', type=int, default=512)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c23_kv_additivity.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    off = ctx.offline()
    kinds = [('K', a.RK, 'keys', E), ('V', a.RV, 'values', 2 * E)]
    st = offline_state(off, kinds)
    full = {k: [(st[(k, l)]['mu'].float(), Pi(st[(k, l)]).float()) for l in range(L)] for k in 'KV'}
    ident = (torch.zeros(E), torch.eye(E))
    _, wins = ctx.windows(a.windows)
    print(f'calibration ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    def run(S):
        S = set(S)
        ctx.mode['proj'] = None if not S else {k: [full[k][l] if l in S else ident for l in range(L)] for k in 'KV'}
        out = []
        with torch.inference_mode():
            for w in wins:
                o = ctx.model(w[None], labels=w[None])
                out.append(float(o.loss) * (w.numel() - 1))
        ctx.mode['proj'] = None
        return out

    tok = wins[0].numel() - 1
    dense = run([])
    ident_run = None
    ctx.mode['proj'] = {k: [ident] * L for k in 'KV'}
    with torch.inference_mode():
        ident_run = [float(ctx.model(w[None], labels=w[None]).loss) * tok for w in wins]
    ctx.mode['proj'] = None
    chk = max(abs(x - y) for x, y in zip(dense, ident_run))
    print('identity vs dense max |nll diff| =', chk, flush=True)
    nd = np.array(dense)

    def delta(S):  # per-window nats/token excess
        return (np.array(run(S)) - nd) / tok

    rng = np.random.default_rng(0)
    subsets = {}
    for l in range(L):
        subsets[f's{l}'] = [l]
    pairs = list(itertools.combinations(range(L), 2))
    if a.smoke:
        pairs = pairs[:5]
    for i, j in pairs:
        subsets[f'p{i}_{j}'] = [i, j]
    trip = [tuple(sorted(rng.choice(L, 3, replace=False).tolist())) for _ in range(30 if not a.smoke else 3)]
    for t in trip:
        subsets['t' + '_'.join(map(str, t))] = list(t)
    rand = {}
    for k in (4, 6, 8):
        for r in range(20 if not a.smoke else 2):
            S = sorted(rng.choice(L, k, replace=False).tolist())
            rand[f'r{k}_{r}'] = S; subsets[f'r{k}_{r}'] = S
    named = {'first6': list(range(6)), 'last6': list(range(6, 12)), 'even': list(range(0, 12, 2)),
             'odd': list(range(1, 12, 2)), 'all12': list(range(12))}
    subsets.update(named)
    D = {}
    for n_, (key, S) in enumerate(subsets.items()):
        D[key] = delta(S)
        if n_ % 10 == 0:
            print(f'{n_}/{len(subsets)} {key} d={D[key].mean():.5f} ({time.perf_counter()-T0:.0f}s)', flush=True)
    single = np.array([D[f's{l}'].mean() for l in range(L)])
    mean = {k: float(v.mean()) for k, v in D.items()}

    def ratio(S, d=None):
        return (np.mean(d if d is not None else mean_of(S)) if False else None)
    rows = {}
    for key, S in subsets.items():
        s = float(sum(single[l] for l in S))
        rows[key] = {'layers': S, 'delta': mean[key], 'sum_singles': s, 'ratio': mean[key] / s if s != 0 else None}
    res = base_result(a, {'tokens_per_window': tok, 'dense_nll_per_window': dense, 'identity_vs_dense_max_abs_nll_diff': chk,
                          'single_delta_nats_per_token': single.tolist(), 'subsets': rows,
                          'delta_per_window': {k: v.tolist() for k, v in D.items()}})
    d = {}
    r12 = rows['all12']['ratio']
    sing_w = np.stack([D[f's{l}'] for l in range(L)])       # (12, windows)
    idx = boot_idx(len(wins), 10000, 0)
    bs = D['all12'][idx].mean(1) / sing_w[:, idx].mean(2).sum(0)
    d['A1_ratio_all12'] = {'ratio': r12, 'ci_lo': float(np.quantile(bs, .025)), 'ci_hi': float(np.quantile(bs, .975)),
                           'pass': abs(r12 - 1) <= 0.2}
    dev = [abs(rows[k]['ratio'] - 1) for k in rand]
    d['A2_median_abs_dev_random_subsets'] = {'median': float(np.median(dev)), 'by_size': {
        str(k): float(np.median([abs(rows[x]['ratio'] - 1) for x in rand if x.startswith(f'r{k}_')])) for k in (4, 6, 8)},
        'median_ratio_by_size': {str(k): float(np.median([rows[x]['ratio'] for x in rand if x.startswith(f'r{k}_')])) for k in (4, 6, 8)},
        'pass': float(np.median(dev)) <= 0.2}
    inter = [(mean[f'p{i}_{j}'] - single[i] - single[j]) / (single[i] + single[j]) for i, j in pairs]
    d['A3_pair_median_abs_rel_interaction'] = {'median': float(np.median(np.abs(inter))), 'pass': float(np.median(np.abs(inter))) <= 0.2}
    d['A4_fraction_pairs_superadditive'] = float(np.mean([x > 0 for x in inter]))
    d['A4_median_signed_rel_interaction'] = float(np.median(inter))
    d['named_ratios'] = {k: rows[k]['ratio'] for k in named}
    d['all12_ppl_8windows'] = math.exp(np.mean(dense) / tok + mean['all12'])
    d['dense_ppl'] = math.exp(np.mean(dense) / tok)
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('decision:', d)


if __name__ == '__main__':
    main()
