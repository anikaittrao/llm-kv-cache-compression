"""Step (2026-11-19): a cheaper basis update during generation (follow-up to kv_update.py / kv_buffer.py, report.pdf Sec. 11).

Run:  .venv\\Scripts\\python.exe experiments\\kv_cheapupdate.py            (about 1.5-2 h, 16 windows; needs an idle CPU)
      .venv\\Scripts\\python.exe experiments\\kv_cheapupdate.py --replot
Writes experiments/results_kv_cheapupdate.json and figures/fig20_kv_cheapupdate.{pdf,png}.

Motivation. In Section 11 the update (top R right singular vectors of the (R+B) x 768 matrix [S U^T ; A_blk], by a plain
SVD, every B = 32 tokens) took about as long as generating the tokens. Here the same update is made cheaper in two ways.
1. Brand's incremental SVD (exact, same result). Write A_blk^T = U C + P with C = U^T A_blk^T (R x n) and the residual P
   (768 x n) outside the basis; QR: P = Q Rq. Then [S U^T ; A_blk]^T = [U Q] K with the small (R+n) x (R+n) matrix
   K = [[diag S, C], [0, Rq]]. The new basis is [U Q] times the top R left singular vectors of K, computed here by a
   symmetric eigendecomposition of K K^T (eigh, float64). Only small matrices are factorized.
   Variant k = 8: keep only the top 8 residual directions of each block (P ~ Q_8 Q_8^T P), so K is (R+8) x (R+n).
2. Rarer updates. The buffer stays B = 32 (each block is read exactly while processed, then compressed with the current
   basis), but the basis is updated only every U = 64 or 128 tokens:
     every U       : (U = 64 or 128; with P = 768 the 256-token continuation allows 3 or 1 updates, with P = 384 9 or 4)
                     from the exact rows of all tokens since the last update. These rows must be kept until the update
                     (not read by attention; extra memory of up to U exact rows). At the update the old cache is
                     re-projected and the tokens since the last update are re-compressed from their exact rows.
     every 128, last 32 : from the exact rows of the latest block only (no memory beyond the 32-token buffer).
Setting otherwise identical to kv_buffer.py 'update / buffer B=32' (one basis per layer for all heads, query-aware keys
with the prompt's G_K, output-aware values, centered with the prompt mean, first token exact, prompt of P tokens
prefilled exactly then compressed, perplexity on the continuation only). Code checks: static = 24.997 and
'svd every 32' = 24.778 (main setting, from kv_buffer.py / kv_update.py).

Timing. The basis computation of every update (the SVD, or C/QR/eigh/products of Brand's method; per layer, keys and
values) is timed with perf_counter inside the run and reported as ms per continuation token. For comparison, one dense
decode step of the whole model (one new token, about 1000 cached tokens) is timed in the same process after every
window (median of 10). Re-projecting the old cache is counted, not timed (as in Section 11). CPU load is recorded.

Rules fixed before the run (main setting: P = 768, keys R = 256, values R = 512, half cache, buffer B = 32);
e(x) = ppl(x)/dense - 1:
  C1 exact      : |ppl(brand every 32) - ppl(svd every 32)| <= 1e-4 x ppl(svd every 32).
  C2 faster     : median per-block update time (all layers, keys + values) of brand every 32 <= 0.5 x that of svd every 32.
  C3 cheap+good : brand every 128 costs <= 25% of a dense decode step per token (update ms per continuation token
                  / decode ms) AND e(brand every 128) <= 0.5 x e(static) in BOTH settings (main and keys only, R = 256, P = 384,
                  where the longer continuation allows 4 updates instead of 1).
  C4 no memory  : e(brand every 128, last 32) <= 0.5 x e(static) in BOTH settings.

Options added 2026-11-22 (audit patches B1/B5; the defaults reproduce the reported run exactly):
  --window_offset k : use test windows k .. k+windows-1 (16 = fresh windows 16-31 for the confirmation run C1).
  --only s1 s2 ...  : run only methods whose label contains one of the strings ('dense' always runs).
  --settings s1 ... : run only settings whose name contains one of the strings.
  The JSON now also stores, per setting, 'nll_per_window' (summed NLL of each window, per method), 'tokens_per_window'
  and 'windows' (the test-window indices), for paired bootstrap CIs (experiments/stats_utils.py).

C10 (audit 2026-11-22; only with --fd, so the default run is unchanged). Frequent Directions (FD; Liberty 2013,
Ghashami et al. 2016) as a baseline for the truncated update. Same sketch [S U^T ; A_blk] and SVD as 'svd every 32',
but the kept squared singular values are shrunk by the largest discarded one, S_i^2 <- S_i^2 - sigma_{R+1}^2 (the basis
U is the same at each step; only the weights of the old sketch in later updates change). Method 'fd every 32'.
Accumulated-error check (Prop 8.1 in the audit plan), for 'svd every 32' and 'fd every 32': with A_m all weighted
rows seen so far (prompt rows 1..P-1 and every updated block) and B_m = S U^T the sketch after update m,
        ||A_m^T A_m - B_m^T B_m||_2  <=  sum_{j=0..m} sigma_{R+1}(M_j)^2,
M_0 = the prompt matrix, M_j = [S U^T ; A_blk] at update j. Stored per layer/kind/update in 'prop81' (lhs, rhs).
Rule fixed before the C10 run (main setting, half cache, P = 768):
  FD1 : |e(fd every 32) - e(brand every 32)| <= 0.1 percentage points (100 x |difference of excess| <= 0.1).
  FD2 (correctness, not a hypothesis): lhs <= rhs x (1 + 1e-9) + 1e-9 x ||A_m||_F^2 in every recorded case.
Runtime with --fd: about 1.6 h for the full default method list plus FD; the C10 run uses
  --fd --only "svd every 32" "brand every 32" "fd every 32" (about 75 min for 16 windows: the substring
  "brand every 32" also selects "brand every 32, k=8"; smoke with svd + fd only: 204 s per window, both settings).
"""
import argparse, json, math, os, platform, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from kv_online import load_wikitext, psd_pow, qblock  # noqa: E402
from kv_timing import cpu_load  # noqa: E402

D64 = torch.float64
L, H, E, Dh = 12, 12, 768, 64
B = 32


def svd_update(U, S, A, R):
    M = torch.cat([S[:, None] * U.T, A])
    _, s, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[:R].T.contiguous(), s[:R].clone()


def fd_update(U, S, A, R):
    """Frequent Directions step (C10): same SVD as svd_update, kept squared values shrunk by sigma_{R+1}^2.
    Returns (U_new, S_new, sigma_{R+1}^2)."""
    M = torch.cat([S[:, None] * U.T, A])
    _, s, Vh = torch.linalg.svd(M, full_matrices=False)
    d2 = float(s[R] ** 2) if s.numel() > R else 0.0
    return Vh[:R].T.contiguous(), (s[:R] ** 2 - d2).clamp_min(0).sqrt(), d2


def svd_update_tail(U, S, A, R):
    """svd_update plus sigma_{R+1}^2 of the sketch (for the Prop 8.1 check only)."""
    M = torch.cat([S[:, None] * U.T, A])
    _, s, Vh = torch.linalg.svd(M, full_matrices=False)
    d2 = float(s[R] ** 2) if s.numel() > R else 0.0
    return Vh[:R].T.contiguous(), s[:R].clone(), d2


def brand_update(U, S, A, R, k=None):
    At = A.T  # 768 x n
    C = U.T @ At
    Pr = At - U @ C
    if k is None:
        Q, Rq = torch.linalg.qr(Pr)
    else:
        Qf = torch.linalg.svd(Pr, full_matrices=False)[0]
        Q = Qf[:, :k]; Rq = Q.T @ Pr
    m, n = Q.shape[1], At.shape[1]
    K = torch.zeros(R + m, R + n, dtype=A.dtype)
    K[:R, :R] = torch.diag(S); K[:R, R:] = C; K[R:, R:] = Rq
    w, V = torch.linalg.eigh(K @ K.T)
    V = V[:, -R:].flip(1); w = w[-R:].flip(0)
    Un = torch.cat([U, Q], 1) @ V
    return Un, w.clamp_min(0).sqrt()


def method_list(main, fd=False):
    # (label, kind, every, rows, k): kind None=dense, 'static', 'svd', 'brand'; rows 'all' since last update or 'last'
    m = [('dense', None, 0, None, None), ('static', 'static', 0, None, None),
         ('svd every 32', 'svd', 32, 'all', None), ('brand every 32', 'brand', 32, 'all', None)]
    if main:
        m += [('brand every 32, k=8', 'brand', 32, 'all', 8)]
    m += [('brand every 64', 'brand', 64, 'all', None), ('brand every 128', 'brand', 128, 'all', None),
          ('brand every 128, last 32', 'brand', 128, 'last', None)]
    if fd:
        m += [('fd every 32', 'fd', 32, 'all', None)]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_cheapupdate.json'))
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--fd', action='store_true', help='C10: add Frequent Directions and the Prop 8.1 check')
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--settings', nargs='*', default=None)
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    torch.set_num_threads(a.threads)
    T0 = time.perf_counter()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    seq = 1024
    win_idx = list(range(a.window_offset, a.window_offset + a.windows))
    wins = [ids_test[i * seq:(i + 1) * seq] for i in win_idx]
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
    TV = [(psd_pow(WO[l] @ WO[l].T, 0.5), psd_pow(WO[l] @ WO[l].T, -0.5)) for l in range(L)]

    cap = {}

    def hook(mod, args, out, l):
        cap[l] = out[0].detach()
        return out

    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: hook(m, a_, o, l)) for l in range(L)]

    settings = [('KV 256/512 P=768', 768, 256, 512, method_list(True, a.fd)),
                ('K 256 P=384', 384, 256, None, method_list(False, a.fd))]
    if a.settings:
        settings = [st_ for st_ in settings if any(x in st_[0] for x in a.settings)]
    if a.only:
        settings = [(n_, P_, rk_, rv_, [m_ for m_ in ms_ if m_[0] in ('dense', 'static') or any(x in m_[0] for x in a.only)])
                    for (n_, P_, rk_, rv_, ms_) in settings]
    res = {'config': vars(a), 'torch': torch.__version__, 'B': B,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads},
           'cpu_load_start_pct': cpu_load(), 'settings': {}, 'upd_times': [], 'decode_ms': [], 'cpu_load_pct': []}
    print('CPU load at start (%):', res['cpu_load_start_pct'], flush=True)
    nll = {(s[0], m[0]): 0.0 for s in settings for m in s[4]}
    nllw = {(s[0], m[0]): [] for s in settings for m in s[4]}  # per-window NLL (audit patch B1)
    check81 = a.fd  # Prop 8.1 accumulated-error check for 'svd every 32' and 'fd every 32'
    if check81:
        res['prop81'] = []
    cnt = {s[0]: 0 for s in settings}

    def Pi(s):
        return s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1]

    def compress(X, s):
        return s['mu'] + (X - s['mu']) @ Pi(s)

    for wi, w in enumerate(wins):
        for (sname, P, RK, RV, methods) in settings:
            kinds = [('K', RK, 'keys', E)] + ([('V', RV, 'values', 2 * E)] if RV else [])
            with torch.inference_mode():
                model(w[:P][None])
            pre = {l: cap[l].to(D64) for l in range(L)}
            st0 = {}
            for kind, R, _, offs in kinds:
                for l in range(L):
                    Xq = pre[l][1:, :E]; Xr = pre[l][1:, offs:offs + E]
                    if kind == 'K':
                        GK = qblock(Xq.T @ Xq / Xq.shape[0]); T = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
                    else:
                        T = TV[l]
                    mu = Xr.mean(0)
                    A0 = (Xr - mu) @ T[0]
                    _, s, Vh = torch.linalg.svd(A0, full_matrices=False)
                    st0[(kind, l)] = {'mu': mu, 'T': T, 'U': Vh[:R].T.contiguous(), 'S': s[:R].clone()}
                    if check81:
                        st0[(kind, l)]['G_all'] = A0.T @ A0  # running A_m^T A_m (768 x 768)
                        st0[(kind, l)]['tail_sum'] = float(s[R] ** 2) if s.numel() > R else 0.0
                        st0[(kind, l)]['fro2'] = float(A0.pow(2).sum())
            for (m, kind_m, every, rows, kk) in methods:
                st = {key: dict(v) for key, v in st0.items()}
                with torch.inference_mode():
                    out = model(w[:P][None], use_cache=True)
                    past = out.past_key_values
                    if kind_m is not None:
                        for kind, R, attr, _ in kinds:
                            for l in range(L):
                                lay = past.layers[l]
                                X = getattr(lay, attr)[0].transpose(0, 1).reshape(P, E).to(D64)
                                Xh = compress(X, st[(kind, l)]); Xh[0] = X[0]
                                setattr(lay, attr, Xh.float().reshape(P, H, Dh).transpose(0, 1)[None].contiguous())
                    logits = []
                    exact = {(kind, l): [] for kind, _, _, _ in kinds for l in range(L)}  # exact rows since last update
                    t = P; t_last = P
                    while t < seq:
                        blk = w[t:t + B]; nb = blk.numel()
                        out = model(blk[None], past_key_values=past)
                        logits.append(out.logits[0]); past = out.past_key_values
                        if kind_m is not None and t + nb < seq:
                            do_upd = kind_m in ('svd', 'brand', 'fd') and (t + nb - P) % every == 0
                            t_blk = 0.0
                            for kind, R, attr, offs in kinds:
                                for l in range(L):
                                    lay = past.layers[l]
                                    X = getattr(lay, attr)[0].transpose(0, 1).reshape(t + nb, E).to(D64)
                                    s = st[(kind, l)]
                                    Xnew = cap[l].to(D64)[:, offs:offs + E]
                                    if kind_m in ('svd', 'brand', 'fd') and rows == 'all':
                                        exact[(kind, l)].append(Xnew)
                                    if do_upd:
                                        if rows == 'all':
                                            Xu = torch.cat(exact[(kind, l)]); r0 = t_last; exact[(kind, l)] = []
                                        else:
                                            Xu = Xnew; r0 = t
                                        A = (Xu - s['mu']) @ s['T'][0]
                                        tt = time.perf_counter()
                                        d2 = None
                                        if kind_m == 'svd':
                                            if check81 and m == 'svd every 32':
                                                s['U'], s['S'], d2 = svd_update_tail(s['U'], s['S'], A, R)
                                            else:
                                                s['U'], s['S'] = svd_update(s['U'], s['S'], A, R)
                                        elif kind_m == 'fd':
                                            s['U'], s['S'], d2 = fd_update(s['U'], s['S'], A, R)
                                        else:
                                            s['U'], s['S'] = brand_update(s['U'], s['S'], A, R, kk)
                                        dt = time.perf_counter() - tt
                                        if check81 and d2 is not None:  # outside the timed part
                                            s['G_all'] = s['G_all'] + A.T @ A
                                            s['tail_sum'] += d2; s['fro2'] += float(A.pow(2).sum())
                                            Bm = s['S'][:, None] * s['U'].T
                                            lhs = float(torch.linalg.eigvalsh(s['G_all'] - Bm.T @ Bm).abs().max())
                                            res['prop81'].append({'setting': sname, 'method': m, 'window': wi, 'layer': l,
                                                                  'kind': kind, 't': t + nb, 'lhs': lhs, 'rhs': s['tail_sum'],
                                                                  'fro2': s['fro2']})
                                        t_blk += dt
                                        if wi < 4:
                                            res['upd_times'].append({'setting': sname, 'method': m, 'kind': kind, 'layer': l,
                                                                     'rows': int(A.shape[0]), 's': dt})
                                        Pn = Pi(s)
                                        X[1:r0] = s['mu'] + (X[1:r0] - s['mu']) @ Pn
                                        if rows == 'all':
                                            X[r0:] = s['mu'] + (Xu - s['mu']) @ Pn
                                        else:
                                            X[r0:] = s['mu'] + (Xnew - s['mu']) @ Pn
                                    else:
                                        X[t:] = compress(Xnew, s)
                                    setattr(lay, attr, X.float().reshape(t + nb, H, Dh).transpose(0, 1)[None].contiguous())
                            if do_upd:
                                t_last = t + nb
                                res['upd_times'].append({'setting': sname, 'method': m, 'kind': 'block', 'layer': -1,
                                                         'rows': -1, 's': t_blk, 'window': wi})
                        t += nb
                    lp = torch.log_softmax(torch.cat(logits)[:-1].double(), -1)
                    tgt = w[P + 1:]
                    v_ = float(-lp[torch.arange(tgt.numel()), tgt].sum())
                    nll[(sname, m)] += v_; nllw[(sname, m)].append(v_)
            cnt[sname] += seq - P - 1
        # one dense decode step with ~1000 cached tokens, in the same process
        with torch.inference_mode():
            past = model(w[:1000][None], use_cache=True).past_key_values
            ts = []
            for j in range(10):
                tt = time.perf_counter(); model(w[1000 + j:1001 + j][None], past_key_values=past); ts.append(time.perf_counter() - tt)
                past.crop(-1)  # drop the token just added, back to 1000 cached tokens
        res['decode_ms'].append(1e3 * float(np.median(ts)))
        res['cpu_load_pct'].append(cpu_load())
        print(f'window {wi+1}/{len(wins)} done ({time.perf_counter()-T0:.0f}s), decode {res["decode_ms"][-1]:.1f} ms, '
              f'load {res["cpu_load_pct"][-1]}', flush=True)

    for h_ in hooks:
        h_.remove()
    for (sname, P, RK, RV, methods) in settings:
        d = {m[0]: math.exp(nll[(sname, m[0])] / cnt[sname]) for m in methods}
        cf = ((RK or E) / E + (RV or E) / E) / 2
        ntok = seq - P
        cost = {}
        for (m, kind_m, every, rows, kk) in methods:
            if kind_m not in ('svd', 'brand', 'fd'):
                continue
            blocks = [x['s'] for x in res['upd_times'] if x['setting'] == sname and x['method'] == m and x['kind'] == 'block']
            per_win = sum(blocks) / a.windows
            cost[m] = {'block_ms_median': 1e3 * float(np.median(blocks)), 'updates_per_window': len(blocks) / a.windows,
                       'ms_per_token': 1e3 * per_win / ntok}
        res['settings'][sname] = {'P': P, 'RK': RK, 'RV': RV, 'cache_frac': cf, 'tokens': cnt[sname], 'ppl': d,
                                  'excess': {k: v / d['dense'] - 1 for k, v in d.items()}, 'cost': cost,
                                  'nll_per_window': {m[0]: nllw[(sname, m[0])] for m in methods},
                                  'tokens_per_window': seq - P - 1, 'windows': win_idx}
        print(sname, {k: round(v, 4) for k, v in d.items()}, flush=True)
        print(' cost:', {k: {q: round(x, 2) for q, x in v.items()} for k, v in cost.items()}, flush=True)
    dec = float(np.median(res['decode_ms']))
    res['decode_ms_median'] = dec
    res['decision'] = {}
    try:
        S_ = res['settings']['KV 256/512 P=768']
        p, e, c = S_['ppl'], S_['excess'], S_['cost']
        e2 = res['settings']['K 256 P=384']['excess']
        res['decision'] = {
            'C1_brand_exact': abs(p['brand every 32'] - p['svd every 32']) <= 1e-4 * p['svd every 32'],
            'C2_brand_2x_faster': c['brand every 32']['block_ms_median'] <= 0.5 * c['svd every 32']['block_ms_median'],
            'C3_every128_cheap_and_good': (c['brand every 128']['ms_per_token'] <= 0.25 * dec) and (e['brand every 128'] <= 0.5 * e['static']) and (e2['brand every 128'] <= 0.5 * e2['static']),
            'C4_last32_good': (e['brand every 128, last 32'] <= 0.5 * e['static']) and (e2['brand every 128, last 32'] <= 0.5 * e2['static'])}
    except KeyError as ex:  # --only / --settings runs lack some cells
        res['decision']['incomplete'] = f'missing {ex}'
    if a.fd:
        try:
            em = res['settings']['KV 256/512 P=768']['excess']
            res['decision_C10'] = {'FD1_fd_within_0.1pt_of_brand': 100 * abs(em['fd every 32'] - em['brand every 32']) <= 0.1}
        except KeyError as ex:
            res['decision_C10'] = {'incomplete': f'missing {ex}'}
        pr = res.get('prop81', [])
        res['decision_C10']['FD2_prop81_holds'] = all(x['lhs'] <= x['rhs'] * (1 + 1e-9) + 1e-9 * x['fro2'] for x in pr)
        res['prop81_summary'] = {m_: {'n': len([x for x in pr if x['method'] == m_]),
                                      'max_lhs_over_rhs': max([x['lhs'] / x['rhs'] for x in pr if x['method'] == m_ and x['rhs'] > 0], default=None),
                                      'median_lhs_over_rhs': float(np.median([x['lhs'] / x['rhs'] for x in pr if x['method'] == m_ and x['rhs'] > 0])) if any(x['method'] == m_ for x in pr) else None}
                                 for m_ in ('svd every 32', 'fd every 32')}
        print('C10:', res['decision_C10'], res['prop81_summary'], flush=True)
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decode ms median:', dec)
    print('decision:', res['decision'])
    try:
        make_figures(a, res)
    except KeyError as ex:
        print('figure skipped (subset run, missing', ex, ')')


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    dec = res['decode_ms_median']
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    names = ['svd every 32', 'brand every 32', 'brand every 32, k=8', 'brand every 64', 'brand every 128',
             'brand every 128, last 32']
    labels = ['plain SVD, every 32', 'Brand, every 32', 'Brand k=8, every 32', 'Brand, every 64', 'Brand, every 128',
              'Brand, every 128, last block only']
    cols = [dv.C['black'], dv.C['blue'], dv.C['violet'], dv.C['aqua'], dv.C['orange'], dv.C['magenta']]
    for i, (sname, title) in enumerate((('KV 256/512 P=768', '(a) half cache, P = 768 (1-7 updates)'),
                                        ('K 256 P=384', '(b) keys only, P = 384 (4-19 updates)'))):
        S = res['settings'][sname]
        for n_, l_, c_ in zip(names, labels, cols):
            if n_ in S['cost']:
                ax[i].scatter(S['cost'][n_]['ms_per_token'], 100 * S['excess'][n_], color=c_, s=24, label=l_, zorder=3)
        ax[i].axhline(100 * S['excess']['static'], color='gray', ls=':', lw=1, label='static basis (no update)')
        ax[i].axvline(0.25 * dec, color='gray', ls='--', lw=0.8, label='25% of a decode step')
        ax[i].set_xscale('log'); ticks = [2, 4, 8, 16, 32, 64]
        ax[i].xaxis.set_major_locator(FixedLocator(ticks)); ax[i].xaxis.set_major_formatter(FixedFormatter([str(t_) for t_ in ticks]))
        ax[i].xaxis.set_minor_locator(NullLocator()); ax[i].set_xlim(1.4, 90)
        ax[i].set_xlabel('update time per token (ms, CPU)'); ax[i].set_ylabel('perplexity increase (%)')
        ax[i].set_ylim(0, 100 * S['excess']['static'] * 1.12); ax[i].set_title(title)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig20_kv_cheapupdate.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
