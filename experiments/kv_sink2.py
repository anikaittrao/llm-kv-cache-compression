"""C7 (audit 2026-11-22): the attention sink and centering, on 16 windows, with a fix to test and the keep0 puzzle.

Run:  .venv\\Scripts\\python.exe experiments\\kv_sink2.py --out experiments\\rerun_2026-11-22\\c7_kv_sink2.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 12-15 min; smoke with r = 32 only on 8 windows: 346 s)
      smoke test: --eval_windows 1 --mech_windows 1 --ranks 32     --replot: redraw the figure from the JSON.

Background (Section 5.7). Per-head key bases (Option 1 of kv_adaptive.py), rank r of 64. Centering hurt (centered
r = 32: 95.8 vs plain 70.6); storing position 0's key exactly ('keep0') fixed it (69.9). The audit found that the
reported geometry mislabelled (k_0 - mu) . mu_hat as k_0 . mu_hat, and that keep0 makes the PLAIN basis worse (70.58 ->
73.25 at r = 32), which the centering mechanism does not explain; the mechanism numbers used 2 test windows.
This script (same calibration as kv_adaptive.py: 16 train windows, seed 0, all positions; evaluation protocol of
kv_adaptive.py: full 1024-token passes of the first --eval_windows test windows, keys of every position compressed in
c_attn, perplexity over all predictions):
  bases per head: plain (top r eigenvectors of E[k k^T], mu = 0); centered (top r of the covariance, mu = mean key);
                  appended = centered with the unit mean direction mu_hat added: span{top r-1 centered directions,
                  mu_hat} (orthonormalised), mu = mean key (same rank r, same storage).
  treatments: none / keep0 (position 0's key exact). Ranks r = 16, 32, 48.
  Code checks (8 windows): plain r=32 70.585, centered r=32 95.845, centered keep0 69.946, plain keep0 73.246.
Mechanism on --mech_windows (default 16) test windows, dense pass, per layer (median over heads):
  k0_coord_mu   = k_0 . mu / ||mu||^2 (k_0's own coordinate along the mean, in units of ||mu||),
  k0mmu_coord   = (k_0 - mu) . mu / ||mu||^2 (= k0_coord_mu - 1; the quantity the report printed),
  Remark 5.1 at r = 32 (centered basis P): ||k_0 - k_hat_0|| >= |c| ||mu_hat (I - P)|| - ||w (I - P)||, with
  k_0 - mu = c mu_hat + w; violations counted (must be 0), and how often the lower bound is positive (informative).
keep0 puzzle (descriptive, K3): mean attention on position 0 (queries t >= 1, head mean) in the compressed runs vs the
  dense model, per layer, for plain / plain keep0 / centered / centered keep0 / appended (r = 32).
Rules fixed before the run:
  K1 fix        : ppl(appended) <= ppl(plain) at r = 16, 32 and 48 (no keep0 for either).
  K2 geometry   : the layer median of k0_coord_mu is < 0 in at least 9 of the 12 layers (16 windows).
  K3 descriptive: attention on position 0 under plain keep0 vs plain vs dense.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import Ctx, L, H, E, Dh, D64, ROOT, base_result, save_json, rss_mb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_windows', type=int, default=8)
    ap.add_argument('--eval_offset', type=int, default=0)
    ap.add_argument('--mech_windows', type=int, default=16)
    ap.add_argument('--ranks', type=int, nargs='*', default=[16, 32, 48])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c7_kv_sink2.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    a.windows = a.eval_windows
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    ctx.offline()  # fills ctx.calib with the sums over all positions of the 16 calibration windows
    n = ctx.calib['n']
    mu = [ctx.calib['S1']['K'][l] / n for l in range(L)]
    M2 = [ctx.calib['S2']['K'][l] / n for l in range(L)]
    C = [M2[l] - torch.outer(mu[l], mu[l]) for l in range(L)]
    print(f'calibration ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    def top(Mat, r):
        return torch.linalg.eigh(Mat)[1][:, -r:] if r > 0 else torch.zeros(Mat.shape[0], 0, dtype=D64)

    def projector(kind, r):
        out, Us = [], []
        for l in range(L):
            Pm = torch.zeros(E, E, dtype=D64); m_ = torch.zeros(E, dtype=D64); ul = []
            for h in range(H):
                s = slice(h * Dh, (h + 1) * Dh)
                if kind == 'plain':
                    U = top(M2[l][s, s], r)
                else:
                    U = top(C[l][s, s], r if kind == 'centered' else r - 1)
                    m_[s] = mu[l][s]
                    if kind == 'appended':
                        mh = mu[l][s] / mu[l][s].norm()
                        U = torch.linalg.qr(torch.cat([U, mh[:, None]], 1))[0]
                Pm[s, s] = U @ U.T; ul.append(U)
            out.append((m_.float(), Pm.float())); Us.append(ul)
        return out, Us

    # ---- mechanism (dense passes)
    mech = {k: np.zeros((L, H)) for k in ('k0_coord_mu', 'k0mmu_coord')}
    rem = {'violations': 0, 'cases': 0, 'lower_positive': 0}
    _, Uc32 = projector('centered', 32)
    ctx.mode['proj'] = None
    with torch.inference_mode():
        for wi in range(a.mech_windows):
            w = ctx.ids_test[wi * 1024:(wi + 1) * 1024]
            ctx.model(w[None])
            for l in range(L):
                k0 = ctx.cap[l][0, E:2 * E].to(D64)
                for h in range(H):
                    s = slice(h * Dh, (h + 1) * Dh)
                    m_ = mu[l][s]; mh = m_ / m_.norm()
                    mech['k0_coord_mu'][l, h] += float(k0[s] @ m_ / (m_ @ m_)) / a.mech_windows
                    mech['k0mmu_coord'][l, h] += float((k0[s] - m_) @ m_ / (m_ @ m_)) / a.mech_windows
                    U = Uc32[l][h]; Pm = U @ U.T
                    dk = k0[s] - m_; c_ = float(dk @ mh); wv = dk - c_ * mh
                    err = float((dk - dk @ Pm).norm())
                    lower = abs(c_) * float((mh - mh @ Pm).norm()) - float((wv - wv @ Pm).norm())
                    rem['cases'] += 1; rem['violations'] += int(err < lower - 1e-9 * max(1.0, abs(lower)))
                    rem['lower_positive'] += int(lower > 0)
    med = {k: np.median(v, 1).tolist() for k, v in mech.items()}
    print('mechanism done', {k: [round(x, 2) for x in v] for k, v in med.items()}, flush=True)

    # ---- perplexity runs (kv_adaptive.py protocol) with attention-on-position-0 diagnostic at r = 32
    evalw = [ctx.ids_test[i * 1024:(i + 1) * 1024] for i in range(a.eval_offset, a.eval_offset + a.eval_windows)]
    att0 = {}

    def on_hook_factory(key):
        def on_hook(l, out, o):
            q = o[0, :, :E].double(); k = o[0, :, E:2 * E].double()
            T_ = q.shape[0]
            qh = q.reshape(T_, H, Dh).transpose(0, 1); kh = k.reshape(T_, H, Dh).transpose(0, 1)
            sc = (qh @ kh.transpose(1, 2)) / 8.0
            sc = sc.masked_fill(torch.triu(torch.ones(T_, T_, dtype=torch.bool), 1), float('-inf'))
            p0 = torch.softmax(sc, -1)[:, 1:, 0].mean()
            att0.setdefault(key, np.zeros(L))[l] += float(p0) / a.eval_windows
        return on_hook

    def evaluate(proj=None, keep0=False, diag_key=None):
        ctx.mode['proj'] = {'K': proj} if proj is not None else None
        ctx.mode['keep0'] = ('K',) if keep0 else ()
        ctx.mode['on_hook'] = on_hook_factory(diag_key) if diag_key else None
        nll, cnt = 0.0, 0
        with torch.inference_mode():
            for w in evalw:
                out = ctx.model(w[None], labels=w[None]); k_ = w.numel() - 1
                nll += float(out.loss) * k_; cnt += k_
        ctx.mode.update({'proj': None, 'keep0': (), 'on_hook': None})
        return math.exp(nll / cnt)

    res = base_result(a, {'calib_tokens': n, 'mech_windows': a.mech_windows, 'mechanism_per_layer_median': med,
                          'mechanism_per_layer_head': {k: v.tolist() for k, v in mech.items()}, 'remark51': rem, 'runs': []})
    res['dense_ppl'] = evaluate(diag_key='dense')
    print(f"dense {res['dense_ppl']:.3f} ({time.perf_counter()-T0:.0f}s)", flush=True)
    for r in a.ranks:
        for kind in ('plain', 'centered', 'appended'):
            proj, _ = projector(kind, r)
            for keep0 in (False, True):
                lab = f'{kind} r={r}' + (' keep0' if keep0 else '')
                p = evaluate(proj, keep0, diag_key=lab if r == 32 else None)
                res['runs'].append({'label': lab, 'basis': kind, 'r': r, 'keep0': keep0, 'ppl': p})
                print(f'{lab:24s} {p:9.3f} ({time.perf_counter()-T0:.0f}s)', flush=True)
    res['attn_pos0_per_layer'] = {k: v.tolist() for k, v in att0.items()}
    res['attn_pos0_mean'] = {k: float(np.mean(v)) for k, v in att0.items()}
    pp = {x['label']: x['ppl'] for x in res['runs']}
    d = {}
    try:
        d['K1_appended_le_plain_all_ranks'] = all(pp[f'appended r={r}'] <= pp[f'plain r={r}'] for r in (16, 32, 48))
    except KeyError as ex:
        d['K1_incomplete'] = f'missing {ex}'
    d['K2_k0_coord_negative_layers'] = int(sum(x < 0 for x in med['k0_coord_mu']))
    d['K2_ge_9_of_12'] = d['K2_k0_coord_negative_layers'] >= 9
    d['remark51_no_violation'] = rem['violations'] == 0
    if a.eval_windows == 8 and a.eval_offset == 0:
        d['code_checks'] = {k: (pp.get(k), v) for k, v in (('plain r=32', 70.5846495006326), ('centered r=32', 95.84467778407233),
                                                           ('centered r=32 keep0', 69.94610450950351), ('plain r=32 keep0', 73.24554309541959))}
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('attention on position 0 (mean over layers):', {k: round(v, 4) for k, v in res['attn_pos0_mean'].items()})
    print('decision:', d)
    try:
        make_figures(a, res)
    except Exception as ex:
        print('figure skipped:', ex)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    os.makedirs(a.figdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    m = res['mechanism_per_layer_median']
    ax[0].plot(range(L), m['k0_coord_mu'], 'o-', ms=3, label=r'$k_0\cdot\hat\mu/\|\mu\|$')
    ax[0].plot(range(L), m['k0mmu_coord'], 's--', ms=3, label=r'$(k_0-\mu)\cdot\hat\mu/\|\mu\|$')
    ax[0].axhline(0, color='gray', lw=0.8); ax[0].set_xlabel('layer'); ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title(f"(a) first key along the mean ({res['mech_windows']} windows)")
    runs = res['runs']
    for kind, c in (('plain', dv.C['black']), ('centered', dv.C['blue']), ('appended', dv.C['orange'])):
        for keep0, ls in ((False, '-'), (True, ':')):
            rr = sorted([x for x in runs if x['basis'] == kind and x['keep0'] == keep0], key=lambda x: x['r'])
            if rr:
                ax[1].plot([x['r'] for x in rr], [x['ppl'] for x in rr], ls=ls, marker='o', ms=3, color=c,
                           label=kind + (' keep0' if keep0 else ''))
    ax[1].set_yscale('log'); ax[1].set_xlabel('rank r per head (of 64)'); ax[1].set_ylabel('perplexity')
    ax[1].set_title('(b) per-head keys'); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c7_kv_sink2.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
