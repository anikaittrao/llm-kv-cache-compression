"""Random stress test of the variance-form output bound (report.pdf Section 10, Proposition 10.6), float64.
Run: .venv\\Scripts\\python.exe experiments\\bound3_random_test.py      (seconds)

One head, one query, a = softmax(s), a^ = softmax(s + delta), output o = sum a_i v_i, W any matrix.
  y_i = (v_i - o) W,  eps_i = (v^_i - v_i) W  (row vectors)
  C_y = sum_i a_i y_i^T y_i,  C_e = sum_i a^_i eps_i^T eps_i,  chi2 = sum_i (a^_i - a_i)^2 / a_i
  Prop 10.6:  ||(o^ - o) W|| <= sqrt(lmax(C_y) chi2) + sqrt(lmax(C_e))
  identity:   chi2 = Var_a(e^d) / (1 + r)^2  with d, r as in Lemma 10.4
Prints the worst ratio actual / bound (must be <= 1), the worst relative error of the chi2 identity, and the ratio in
two constructed cases where each term is attained (should be 1).
2026-11-22 (audit patch B3): ratios are compared only when finite; non-finite (degenerate) trials are counted and
printed instead of being hidden by max(worst, nan). The worst ratio over the finite trials is unchanged.
"""
import math
import torch

torch.set_num_threads(1)
torch.manual_seed(0)
D = torch.float64


def bound(a, ah, v, vh, W, d):
    o = a @ v
    y = (v - o) @ W; ep = (vh - v) @ W
    Cy = (a[:, None] * y).T @ y; Ce = (ah[:, None] * ep).T @ ep
    chi2 = float(((ah - a) ** 2 / a).sum())
    # identity Var_a(e^d)/(1+r)^2 = E_a[e^{2d}]/(E_a e^d)^2 - 1 (1 + r = E_a e^d), in log space to avoid overflow
    la = torch.log(a)
    chi2_id = float(torch.expm1(torch.logsumexp(la + 2 * d, 0) - 2 * torch.logsumexp(la + d, 0)))
    t1 = (float(torch.linalg.eigvalsh(Cy)[-1].clamp_min(0)) * chi2) ** 0.5
    t2 = float(torch.linalg.eigvalsh(Ce)[-1].clamp_min(0)) ** 0.5
    err = float(((ah @ vh - o) @ W).norm())
    return err, t1, t2, chi2, chi2_id


worst, worst_id = 0.0, 0.0
n_nonfinite, n_nonfinite_id = 0, 0
for trial in range(20000):
    n = int(torch.randint(2, 300, (1,))); dv = int(torch.randint(1, 20, (1,))); dw = int(torch.randint(1, 30, (1,)))
    sharp = float(10 ** torch.empty(1).uniform_(-1, 1.3))
    scale = float(10 ** torch.empty(1).uniform_(-4, 1.2))
    s = torch.randn(n, dtype=D) * sharp
    delta = torch.randn(n, dtype=D) * scale
    if trial % 3 == 0:
        delta = delta * torch.randn(n, dtype=D).abs() ** 3
    a = torch.softmax(s, 0); ah = torch.softmax(s + delta, 0)
    d = delta - (a * delta).sum()
    v = torch.randn(n, dv, dtype=D); vh = v + torch.randn(n, dv, dtype=D) * scale
    if trial % 5 == 0:                                     # value errors all in one direction
        vh = v + torch.randn(n, 1, dtype=D) * torch.randn(1, dv, dtype=D) * scale
    W = torch.randn(dv, dw, dtype=D)
    err, t1, t2, chi2, chi2_id = bound(a, ah, v, vh, W, d)
    r_ = err / (t1 + t2) if (t1 + t2) > 0 else float('nan')
    if math.isfinite(r_):
        worst = max(worst, r_)
    else:
        n_nonfinite += 1
    if chi2 > 1e-8:
        ri = abs(chi2_id - chi2) / chi2
        if math.isfinite(ri):
            worst_id = max(worst_id, ri)
        else:
            n_nonfinite_id += 1
print('worst actual / bound over 20000 random cases:', f'{worst:.15f}')
print('worst relative error of the chi2 identity (chi2 > 1e-8):', f'{worst_id:.1e}')
print(f'non-finite ratios skipped: bound {n_nonfinite}, chi2 identity {n_nonfinite_id} (of 20000 trials)')

# sharpness 1: exact keys (a^ = a), every value error the same vector -> second term attained
n, dv = 50, 8
a = torch.softmax(torch.randn(n, dtype=D), 0); v = torch.randn(n, dv, dtype=D); e = torch.randn(1, dv, dtype=D)
err, t1, t2, _, _ = bound(a, a.clone(), v, v + e, torch.eye(dv, dtype=D), torch.zeros(n, dtype=D))
print('sharpness, value term (should be 1):', round(err / (t1 + t2), 12))
# sharpness 2: exact values, values on one line with y_i proportional to (a^_i - a_i)/a_i -> first term attained
s = torch.randn(n, dtype=D); delta = 0.3 * torch.randn(n, dtype=D)
a = torch.softmax(s, 0); ah = torch.softmax(s + delta, 0)
g = (ah - a) / a                                            # sum_i a_i g_i = 0, so o = 0 below
v = g[:, None] * torch.randn(1, dv, dtype=D)
d = delta - (a * delta).sum()
err, t1, t2, _, _ = bound(a, ah, v, v.clone(), torch.eye(dv, dtype=D), d)
print('sharpness, mixing term (should be 1):', round(err / (t1 + t2), 12))
