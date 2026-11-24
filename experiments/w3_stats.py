"""Small helpers shared by the wave-3 analysis scripts (c14_analyze.py ... c18_analyze.py). Paired bootstrap over test
windows (same resampling as stats_utils.py: 10,000 resamples, seed 0, 95% percentile CI)."""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats_utils import boot_idx, ppl_ratio  # noqa: E402,F401

RR = os.path.join(HERE, 'rerun_2026-11-22')


def load(name):
    p = name if os.path.isabs(name) else os.path.join(RR, name)
    with open(p) as f:
        return json.load(f)


def save(name, obj):
    with open(os.path.join(RR, name), 'w') as f:
        json.dump(obj, f, indent=1)


def excess(nll, dense, tok):
    """ppl(x)/dense - 1 from per-window NLL sums (tok = tokens per window, scalar or per-window)."""
    a, d = np.asarray(nll, float), np.asarray(dense, float)
    t = np.asarray(tok, float) if np.ndim(tok) else np.full(a.size, float(tok))
    return math.exp((a.sum() - d.sum()) / t.sum()) - 1


def boot(fn, n, n_boot=10000, seed=0):
    """fn(index array) -> float; returns point (all windows) and the 95% CI over resampled windows."""
    pt = fn(np.arange(n))
    idx = boot_idx(n, n_boot, seed)
    bs = np.array([fn(i) for i in idx])
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return {'point': float(pt), 'ci_lo': float(lo), 'ci_hi': float(hi), 'windows': int(n)}


def e_of(nllw, m, dense_key, sel, tok):
    a, d = np.asarray(nllw[m], float)[sel], np.asarray(nllw[dense_key], float)[sel]
    t = np.asarray(tok, float)[sel] if np.ndim(tok) else tok * len(sel)
    return math.exp((a.sum() - d.sum()) / (t.sum() if np.ndim(tok) else t)) - 1


def ex_ratio_ci(nllw, A, B, dense='dense', tok=1):
    """e(A)/e(B) with a paired bootstrap CI."""
    n = len(nllw[A])
    return boot(lambda s: e_of(nllw, A, dense, s, tok) / e_of(nllw, B, dense, s, tok), n)


def ex_ci(nllw, A, dense='dense', tok=1):
    n = len(nllw[A])
    return boot(lambda s: e_of(nllw, A, dense, s, tok), n)


def ppl_ratio_ci(nllw, A, B, tok):
    r = ppl_ratio(nllw[A], nllw[B], tok)
    return {k: r[k] for k in ('ratio', 'ci_lo', 'ci_hi', 'windows_won', 'windows', 'sign_p')}
