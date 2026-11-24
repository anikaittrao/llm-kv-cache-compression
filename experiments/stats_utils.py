"""Paired statistics over test windows (audit patch B2, 2026-11-22).

All Section 6-8 scripts compare methods on the SAME test windows, so the right uncertainty is paired: resample windows
(with replacement) and recompute the ratio of perplexities on each resample. Needs the per-window NLL that the patched
scripts store as settings[<name>]['nll_per_window'][<method>] (summed NLL of each window) and 'tokens_per_window'.
numpy only (scipy is not installed).

  ppl_ratio(A, B)    = exp((sum NLL_A - sum NLL_B) / tokens)  = ppl(A) / ppl(B)
  excess_ratio(A, B) = (ppl(A)/dense - 1) / (ppl(B)/dense - 1)  (used by rules of the form e(A) <= c e(B))
  windows_won        = number of windows with NLL_A < NLL_B (ties counted separately)
  sign_p             = exact two-sided binomial sign test p-value over the non-tied windows

CLI:
  .venv\\Scripts\\python.exe experiments\\stats_utils.py <results.json> --setting "KV 256/512 P=768" \\
        --pairs "update / buffer B=32|static / buffer B=32" "static / buffer B=32|dense"
  .venv\\Scripts\\python.exe experiments\\stats_utils.py <results.json> --setting ... --all_vs dense
Prints point value, 95% percentile CI (10,000 resamples, seed 0), windows won and sign-test p.
"""
import argparse, json, math
import numpy as np

N_BOOT = 10000


def _arr(x):
    return np.asarray(x, dtype=float)


def boot_idx(n, n_boot=N_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ppl_ratio(nll_a, nll_b, tok_per_window, n_boot=N_BOOT, seed=0, level=0.95):
    a, b = _arr(nll_a), _arr(nll_b)
    assert a.shape == b.shape and a.size > 0
    tok = _arr(tok_per_window) if np.ndim(tok_per_window) else np.full(a.size, float(tok_per_window))
    point = math.exp((a.sum() - b.sum()) / tok.sum())
    idx = boot_idx(a.size, n_boot, seed)
    bs = np.exp((a[idx].sum(1) - b[idx].sum(1)) / tok[idx].sum(1))
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    won, tie = int((a < b).sum()), int((a == b).sum())
    return {'ratio': point, 'ci_lo': float(lo), 'ci_hi': float(hi), 'windows': int(a.size),
            'windows_won': won, 'ties': tie, 'sign_p': sign_test(won, int(a.size) - tie)}


def excess_ratio(nll_a, nll_b, nll_dense, tok_per_window, n_boot=N_BOOT, seed=0, level=0.95):
    a, b, d = _arr(nll_a), _arr(nll_b), _arr(nll_dense)
    tok = _arr(tok_per_window) if np.ndim(tok_per_window) else np.full(a.size, float(tok_per_window))

    def er(ia):
        T = tok[ia].sum(-1)
        ea = np.exp((a[ia].sum(-1) - d[ia].sum(-1)) / T) - 1
        eb = np.exp((b[ia].sum(-1) - d[ia].sum(-1)) / T) - 1
        return ea / eb
    point = float(er(np.arange(a.size)))
    bs = er(boot_idx(a.size, n_boot, seed))
    lo, hi = np.nanquantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {'excess_ratio': point, 'ci_lo': float(lo), 'ci_hi': float(hi), 'windows': int(a.size)}


def excess_ci(nll_a, nll_dense, tok_per_window, n_boot=N_BOOT, seed=0, level=0.95):
    """ppl(A)/dense - 1 with a paired CI."""
    r = ppl_ratio(nll_a, nll_dense, tok_per_window, n_boot, seed, level)
    return {'excess': r['ratio'] - 1, 'ci_lo': r['ci_lo'] - 1, 'ci_hi': r['ci_hi'] - 1, 'windows': r['windows']}


def sign_test(k, n):
    """Exact two-sided binomial test of k successes in n trials with p = 1/2."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = pmf[k]
    return float(min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-12))))


def cells(res, setting):
    s = res['settings'][setting]
    return s['nll_per_window'], s['tokens_per_window']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    ap.add_argument('--setting', required=True)
    ap.add_argument('--pairs', nargs='*', default=[], help='"A|B" pairs: ratio ppl(A)/ppl(B)')
    ap.add_argument('--all_vs', default=None, help='compare every method with this one')
    a = ap.parse_args()
    with open(a.json) as f:
        res = json.load(f)
    nw, tok = cells(res, a.setting)
    pairs = [p.split('|') for p in a.pairs]
    if a.all_vs:
        pairs += [(m, a.all_vs) for m in nw if m != a.all_vs]
    for A, B in pairs:
        r = ppl_ratio(nw[A], nw[B], tok)
        print(f"{A:32s} / {B:28s} {r['ratio']:.5f}  95% CI [{r['ci_lo']:.5f}, {r['ci_hi']:.5f}]  "
              f"won {r['windows_won']}/{r['windows']}  sign p {r['sign_p']:.3g}")


if __name__ == '__main__':
    main()
