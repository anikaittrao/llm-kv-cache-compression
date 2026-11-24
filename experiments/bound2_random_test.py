"""Random stress test of the attention-weighted bounds (report.pdf Section 10, Lemma 10.4 / Prop 10.5), float64.
Run: .venv\\Scripts\\python.exe experiments\\bound2_random_test.py      (seconds)
Prints the worst ratio actual/upper bound (must be <= 1) and lower bound/actual (must be <= 1).
2026-11-22 (audit patch B3): ratios are only compared when they are finite; non-finite (degenerate) trials are counted
and printed instead of being dropped silently (max(worst, nan) would hide a NaN). The worst ratios over the finite
trials are unchanged."""
import math
import torch

torch.manual_seed(0)
D = torch.float64
worst = {'lem_up': 0.0, 'lem_lo': -1e9, 'out_L1p': 0.0, 'out_L2p': 0.0}
nonfinite = {'lem_up': 0, 'lem_lo': 0, 'out_L1p': 0, 'out_L2p': 0}
n_lemma_checked = 0
N_TRIALS = 20000


def upd(key, val, fn=max):
    if math.isfinite(val):
        worst[key] = fn(worst[key], val)
    else:
        nonfinite[key] += 1


for trial in range(N_TRIALS):
    n = int(torch.randint(2, 300, (1,)))
    sharp = float(10 ** torch.empty(1).uniform_(-1, 1.3))          # how peaked the attention is
    scale = float(10 ** torch.empty(1).uniform_(-4, 1.2))          # size of score errors
    s = torch.randn(n, dtype=D) * sharp
    delta = torch.randn(n, dtype=D) * scale
    if trial % 3 == 0:                                             # heavy-tailed errors
        delta = delta * torch.randn(n, dtype=D).abs() ** 3
    a = torch.softmax(s, 0); ah = torch.softmax(s + delta, 0)
    d = delta - (a * delta).sum()
    phi = torch.expm1(d) - d
    R = float((a * phi).sum())
    m1 = float((a * d.abs()).sum())
    sm = float((a * d.pow(2) * (d < 0)).sum())
    tv = float((ah - a).abs().sum())
    up = (m1 + 2 * R) / (1 + R); lo = (m1 - 0.5 * sm - R) / (1 + R)
    if tv > 1e-9:  # below this, float64 round-off in a^ - a (entries near 1) dominates
        n_lemma_checked += 1
        upd('lem_up', tv / up)
        upd('lem_lo', lo / tv)
        r_ = tv / up
        if math.isfinite(r_):
            worst['lem_up_min'] = min(worst.get('lem_up_min', 1.0), r_)
    # output bound with random values (dim 16) and value errors
    v = torch.randn(n, 16, dtype=D); vh = v + torch.randn(n, 16, dtype=D) * scale
    o = a @ v; oh = ah @ vh
    err = float((oh - o).norm())
    u = (v - o).norm(dim=1); eps = (vh - v).norm(dim=1)
    L1p = float(((ah - a).abs() * u).sum() + (ah * eps).sum())
    L2p = float((a * ((d.abs() + phi + R) * u + torch.exp(d) * eps)).sum() / (1 + R))
    upd('out_L1p', err / L1p)
    upd('out_L2p', err / L2p)
print({k: round(v, 6) for k, v in worst.items()})
print(f'trials {N_TRIALS}; lemma checked (||a^-a||_1 > 1e-9): {n_lemma_checked}; non-finite ratios skipped: {nonfinite}')
