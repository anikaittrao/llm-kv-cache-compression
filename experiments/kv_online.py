"""Step 6 (2026-11-18): an online basis for the KV cache, computed by RSVD from the text itself (no calibration data).

Run:  .venv\\Scripts\\python.exe experiments\\kv_online.py            (about 15-25 min)
      .venv\\Scripts\\python.exe experiments\\kv_online.py --replot
Writes experiments/results_kv_online.json and figures/fig14_kv_online.{pdf,png}.

Setting ("prompt basis"). A window of 1024 test tokens is split into a prompt of P tokens and a continuation. The prompt is
processed exactly (prefill), as in real use. Then, per layer, a basis is computed from the prompt's own keys and values,
the whole cache (prompt and continuation) is stored in it, and the continuation is run on the compressed cache.
Perplexity is measured on the continuation's predictions only (tokens P+1 .. 1023), so all methods are compared on the
same tokens. Same recipe as report.pdf Section 8: one basis per layer (all 12 heads), keys query-aware (G = blockdiag of
the heads' query second moments), values output-aware (G = W_O W_O^T), centered, first token stored exactly (the sink
result), x_hat = mu + (x - mu) G^{1/2} U U^T G^{-1/2}. The basis U is the top-R right singular vectors of the weighted,
centered prompt matrix A = (X - mu) G^{1/2} (positions 1..P-1; P-1 rows, 768 columns).
Methods for U:
  offline   : from 16 calibration windows of training text (the method of report.pdf), mean and G also from calibration.
  online    : exact SVD of A; mean and query weight from the prompt itself.
  rsvd q    : Algorithm 2 of background.pdf applied to A^T (range = right singular space), target rank k = R,
              oversampling p = 10, q power iterations with re-orthonormalization (Remark 5.10).
Theory check (background.pdf Thm 5.5): for every layer, window and kind, the projection error ||A^T - QQ^T A^T||_F of
the RSVD range Q (l = R + p columns) divided by the best rank-R error ||A - A_R||_F; Thm 5.5 bounds the expectation of this
ratio by sqrt(1 + R/(p-1)) for q = 0. Also the error of the rank-R output (Cor 5.2 / Remark 5.7), and wall time.

Rules fixed before the run (main setting: P = 768, keys R = 256, values R = 512, i.e. half the cache):
  R1 RSVD fidelity     : ppl(rsvd q=1) <= 1.01 x ppl(online exact).
  R2 no calibration    : ppl(online exact) <= ppl(offline).
  R3 quality           : ppl(rsvd q=1) <= 1.05 x dense, on the continuation tokens.
Extra settings: a smaller cache (keys 128 + values 384, a third), and keys only (R = 256) for prompts of 384, 512, 768.

Options added 2026-11-22 (audit patches B1/B5; the defaults reproduce the reported run exactly):
  --window_offset k : use test windows k .. k+windows-1 (16 = fresh windows 16-31 for the confirmation run C1).
  --only s1 s2 ...  : run only methods whose label contains one of the strings ('dense' always runs).
  --settings s1 ... : run only settings whose name contains one of the strings.
  --calib_skip0     : offline calibration statistics without position 0 of every calibration window (the online
                      statistics never use position 0).
  The JSON now also stores, per setting, 'nll_per_window' (summed NLL of each window, per method), 'tokens_per_window'
  and 'windows' (the test-window indices), for paired bootstrap CIs (experiments/stats_utils.py).
  Timing summary (audit patch B2, 2026-11-22): the reported run pooled the P = 768 and P = 384 timings of "K R=256 q=1"
  (keys-only P = 384 also runs RSVD q=1). 'timing_summary' is now the main setting (KV 256/512 P=768) only;
  'timing_summary_by_setting' gives every setting separately and 'timing_summary_pooled_old' the old pooled numbers.
  --resummarize reads an existing JSON (--out) and writes these summaries to --summary_out without running anything
  (the input JSON is never overwritten).
"""
import argparse, json, math, os, platform, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64
L, H, E, Dh = 12, 12, 768, 64


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


def qblock(M2Q):
    return torch.block_diag(*[M2Q[h * Dh:(h + 1) * Dh, h * Dh:(h + 1) * Dh] for h in range(H)])


def rsvd_right(A, k, p, q, gen):
    """Top-k right singular vectors of A (n x d) by Algorithm 2 of background.pdf on M = A^T. Returns (U_k, Q) with Q the range basis."""
    M = A.T
    Om = torch.randn(M.shape[1], k + p, generator=gen, dtype=A.dtype)
    Q = torch.linalg.qr(M @ Om)[0]
    for _ in range(q):
        Z = torch.linalg.qr(M.T @ Q)[0]
        Q = torch.linalg.qr(M @ Z)[0]
    Ub = torch.linalg.svd(Q.T @ M, full_matrices=False)[0]
    return Q @ Ub[:, :k], Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_online.json'))
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--settings', nargs='*', default=None)
    ap.add_argument('--calib_skip0', action='store_true')
    ap.add_argument('--resummarize', action='store_true', help='recompute the timing summaries of an existing --out JSON')
    ap.add_argument('--summary_out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'kv_online_timing_summary.json'))
    a = ap.parse_args()
    if a.resummarize:
        with open(a.out) as f:
            old = json.load(f)
        summ = timing_summaries(old['timing'])
        summ['source'] = a.out
        summ['note'] = ("timing_summary = main setting KV 256/512 P=768 only (as reported in the text); the reported JSON's "
                        "'timing_summary' pooled P=768 and P=384 for 'K R=256 q=1' (= timing_summary_pooled_old here).")
        summ['reported_json_timing_summary'] = old.get('timing_summary')
        os.makedirs(os.path.dirname(a.summary_out), exist_ok=True)
        with open(a.summary_out, 'w') as f:
            json.dump(summ, f, indent=1)
        print(json.dumps(summ, indent=1)); return
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
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    seq = 1024
    win_idx = list(range(a.window_offset, a.window_offset + a.windows))
    wins = [ids_test[i * seq:(i + 1) * seq] for i in win_idx]
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
    GV = [WO[l] @ WO[l].T for l in range(L)]
    TV = [(psd_pow(GV[l], 0.5), psd_pow(GV[l], -0.5)) for l in range(L)]

    cap, mode = {}, {'proj': None}

    def hook(mod, args, out, l):
        cap[l] = out[0].detach()
        P_ = mode['proj']
        if P_ is None:
            return out
        o = out.clone()
        for kind, off in (('K', E), ('V', 2 * E)):
            if P_.get(kind) is not None:
                mu, Pi = P_[kind][l]
                X = o[..., off:off + E]
                o[..., off:off + E] = mu + (X - mu) @ Pi
        return o

    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: hook(m, a_, o, l)) for l in range(L)]

    # ---- offline calibration statistics (same 16 windows as kv_adaptive.py)
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids_train.numel() - seq, (16,), generator=g).tolist()
    S1 = {k: [torch.zeros(E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    S2 = {k: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    sk = 1 if a.calib_skip0 else 0  # --calib_skip0: drop position 0 of every calibration window
    with torch.inference_mode():
        for s in starts:
            model(ids_train[s:s + seq][None])
            for l in range(L):
                X = cap[l].to(D64)
                for i, k in enumerate('QKV'):
                    Xi = X[sk:, i * E:(i + 1) * E]; S1[k][l] += Xi.sum(0); S2[k][l] += Xi.T @ Xi
    n = 16 * (seq - sk)
    off_stats = {l: {'muK': S1['K'][l] / n, 'muV': S1['V'][l] / n,
                     'CK': S2['K'][l] / n - torch.outer(S1['K'][l], S1['K'][l]) / n ** 2,
                     'CV': S2['V'][l] / n - torch.outer(S1['V'][l], S1['V'][l]) / n ** 2,
                     'GK': qblock(S2['Q'][l] / n)} for l in range(L)}
    off_basis = {}
    for l in range(L):
        st = off_stats[l]
        TK = (psd_pow(st['GK'], 0.5), psd_pow(st['GK'], -0.5))
        UK = torch.linalg.eigh(TK[0] @ st['CK'] @ TK[0])[1].flip(1)
        UV = torch.linalg.eigh(TV[l][0] @ st['CV'] @ TV[l][0])[1].flip(1)
        off_basis[l] = {'K': (st['muK'], TK, UK), 'V': (st['muV'], TV[l], UV)}
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    def projector(mu, T, U, R):
        Ur = U[:, :R]
        return mu.float(), (T[0] @ Ur @ Ur.T @ T[1]).float()

    settings = [  # (name, P, RK or None, RV or None)
        ('KV 256/512 P=768', 768, 256, 512),
        ('KV 128/384 P=768', 768, 128, 384),
        ('K 256 P=384', 384, 256, None),
        ('K 256 P=512', 512, 256, None),
        ('K 256 P=768', 768, 256, None),
    ]
    if a.settings:
        settings = [st_ for st_ in settings if any(x in st_[0] for x in a.settings)]

    def keep(m):
        return m == 'dense' or not a.only or any(x in m for x in a.only)
    methods = ['dense', 'offline', 'online', 'rsvd q=0', 'rsvd q=1', 'rsvd q=2']
    key_methods = ['dense', 'offline', 'online', 'rsvd q=1']
    res = {'config': vars(a), 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads},
           'settings': {}, 'theory': [], 'timing': []}
    nll = {(s[0], m): 0.0 for s in settings for m in methods}
    nllw = {(s[0], m): [] for s in settings for m in methods}  # per-window NLL (audit patch B1)
    cnt = {s[0]: 0 for s in settings}

    for wi, w in enumerate(wins):
        for (sname, P, RK, RV) in settings:
            # prefill (exact) once to get the prompt's q, k, v per layer for the online bases
            mode['proj'] = None
            with torch.inference_mode():
                model(w[:P][None])
            pre = {l: cap[l].to(D64) for l in range(L)}
            bases = {m: {} for m in methods if m not in ('dense', 'offline')}
            for l in range(L):
                Xq, Xk, Xv = pre[l][1:, :E], pre[l][1:, E:2 * E], pre[l][1:, 2 * E:]
                GK = qblock(Xq.T @ Xq / Xq.shape[0])
                TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
                for kind, Xr, T, R in (('K', Xk, TK, RK), ('V', Xv, TV[l], RV)):
                    if R is None:
                        continue
                    mu = Xr.mean(0)
                    A = (Xr - mu) @ T[0]
                    t = time.perf_counter(); _, sv, Vh = torch.linalg.svd(A, full_matrices=False); t_svd = time.perf_counter() - t
                    U = Vh.T
                    bases['online'][(kind, l)] = (mu, T, U)
                    tail = float(sv[R:].pow(2).sum().sqrt())
                    for q in (0, 1, 2):
                        m = f'rsvd q={q}'
                        if m not in bases:
                            continue
                        if m not in key_methods and RV is None:
                            continue
                        gen = torch.Generator().manual_seed(1000 * wi + 10 * l + (0 if kind == 'K' else 1) + 100000 * q)
                        t = time.perf_counter(); Ur, Q = rsvd_right(A, R, a.p, q, gen); t_r = time.perf_counter() - t
                        bases[m][(kind, l)] = (mu, T, Ur)
                        if sname.startswith('KV 256/512') or (sname == 'K 256 P=384'):
                            M = A.T
                            proj_err = float((M - Q @ (Q.T @ M)).norm())
                            out_err = float((M - Ur @ (Ur.T @ M)).norm())
                            res['theory'].append({'setting': sname, 'window': wi, 'layer': l, 'kind': kind, 'R': R, 'p': a.p,
                                                  'q': q, 'n_rows': A.shape[0], 'tail': tail,
                                                  'proj_ratio': proj_err / tail, 'out_ratio': out_err / tail,
                                                  'thm55_bound': math.sqrt(1 + R / (a.p - 1)),
                                                  'energy_kept_exact': 1 - tail ** 2 / float(sv.pow(2).sum())})
                            res['timing'].append({'setting': sname, 'kind': kind, 'R': R, 'q': q, 'svd_s': t_svd, 'rsvd_s': t_r})
            use = methods if RV is not None else key_methods
            for m in use:
                if not keep(m):
                    continue
                if m == 'dense':
                    proj = None
                else:
                    proj = {}
                    for kind, R in (('K', RK), ('V', RV)):
                        if R is None:
                            continue
                        proj[kind] = [projector(*(off_basis[l][kind] if m == 'offline' else bases[m][(kind, l)]), R) for l in range(L)]
                # prefill exactly, then compress the stored prompt cache (first token kept exact)
                mode['proj'] = None
                with torch.inference_mode():
                    out = model(w[:P][None], use_cache=True)
                    past = out.past_key_values
                    if proj is not None:
                        for l in range(L):
                            lay = past.layers[l]
                            for kind, attr in (('K', 'keys'), ('V', 'values')):
                                if kind not in proj:
                                    continue
                                mu, Pi = proj[kind][l]
                                X = getattr(lay, attr)[0].transpose(0, 1).reshape(P, E)  # (P, 768)
                                Xh = mu + (X - mu) @ Pi
                                Xh[0] = X[0]
                                setattr(lay, attr, Xh.reshape(P, H, Dh).transpose(0, 1)[None].contiguous())
                    mode['proj'] = proj  # the continuation's own keys and values are compressed as they are made
                    out2 = model(w[P:][None], past_key_values=past)
                    mode['proj'] = None
                    if m == 'dense' and wi == 0 and 'sanity_max_logit_diff' not in res:  # prefill + continuation == one pass
                        full = model(w[None]).logits[0, P:]
                        res['sanity_max_logit_diff'] = float((full - out2.logits[0]).abs().max())
                        print('sanity: max |logit diff| prefill+continuation vs one pass:', res['sanity_max_logit_diff'], flush=True)
                    lp = torch.log_softmax(out2.logits[0, :-1].double(), -1)
                    tgt = w[P + 1:]
                    v_ = float(-lp[torch.arange(tgt.numel()), tgt].sum())
                    nll[(sname, m)] += v_; nllw[(sname, m)].append(v_)
            cnt[sname] += seq - P - 1
        print(f'window {wi+1}/{len(wins)} done ({time.perf_counter()-T0:.0f}s)', flush=True)

    for h_ in hooks:
        h_.remove()
    for (sname, P, RK, RV) in settings:
        use = [m for m in (methods if RV is not None else key_methods) if keep(m)]
        d = {m: math.exp(nll[(sname, m)] / cnt[sname]) for m in use}
        cf = ((RK or E) / E + (RV or E) / E) / 2
        res['settings'][sname] = {'P': P, 'RK': RK, 'RV': RV, 'cache_frac': cf, 'tokens': cnt[sname], 'ppl': d,
                                  'ratio': {m: d[m] / d['dense'] for m in use},
                                  'nll_per_window': {m: nllw[(sname, m)] for m in use},
                                  'tokens_per_window': seq - P - 1, 'windows': win_idx}
        print(sname, {m: round(v, 3) for m, v in d.items()})
    try:
        main_ = res['settings']['KV 256/512 P=768']['ppl']
        res['decision'] = {'R1_rsvd_q1_within_1pct_of_online': main_['rsvd q=1'] <= 1.01 * main_['online'],
                           'R2_online_no_worse_than_offline': main_['online'] <= main_['offline'],
                           'R3_rsvd_q1_within_5pct_of_dense': main_['rsvd q=1'] <= 1.05 * main_['dense']}
    except KeyError as ex:  # --only / --settings runs lack some cells
        res['decision'] = {'incomplete': f'missing {ex}'}
    th = res['theory']
    if not any(t['setting'].startswith('KV') for t in th):
        th = []
    res['theory_summary'] = {} if not th else {f"{k} R={R} q={q}": {'proj_ratio_max': max(t['proj_ratio'] for t in th if t['kind'] == k and t['R'] == R and t['q'] == q and t['setting'].startswith('KV')),
                                                  'proj_ratio_median': float(np.median([t['proj_ratio'] for t in th if t['kind'] == k and t['R'] == R and t['q'] == q and t['setting'].startswith('KV')])),
                                                  'out_ratio_median': float(np.median([t['out_ratio'] for t in th if t['kind'] == k and t['R'] == R and t['q'] == q and t['setting'].startswith('KV')])),
                                                  'out_ratio_max': max(t['out_ratio'] for t in th if t['kind'] == k and t['R'] == R and t['q'] == q and t['setting'].startswith('KV')),
                                                  'thm55_bound': math.sqrt(1 + R / (a.p - 1))}
                             for k, R in (('K', 256), ('V', 512)) for q in (0, 1, 2)}
    res.update(timing_summaries(res['timing']))
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decision:', res['decision'])
    print('theory:', json.dumps(res['theory_summary'], indent=1))
    print('timing:', json.dumps(res['timing_summary'], indent=1))
    try:
        make_figures(a, res)
    except (KeyError, ValueError) as ex:
        print('figure skipped (subset run):', ex)


def timing_summaries(tm):
    """Median SVD / RSVD wall times. 'timing_summary' = main setting only (audit patch B2: the reported run pooled
    P = 768 and P = 384 for 'K R=256 q=1'); also per setting and the old pooled version."""
    def med(rows, key):
        return 1e3 * float(np.median([t[key] for t in rows])) if rows else None

    def summ(rows):
        out = {}
        for k, R, q in sorted({(t['kind'], t['R'], t['q']) for t in rows}):
            sel = [t for t in rows if t['kind'] == k and t['R'] == R and t['q'] == q]
            out[f"{k} R={R} q={q}"] = {'svd_ms_median': med(sel, 'svd_s'), 'rsvd_ms_median': med(sel, 'rsvd_s'),
                                       'n': len(sel), 'rsvd_over_svd': med(sel, 'rsvd_s') / med(sel, 'svd_s')}
        return out
    pooled = {f"{k} R={R} q={q}": {'svd_ms_median': med([t for t in tm if t['kind'] == k and t['R'] == R and t['q'] == q], 'svd_s'),
                                   'rsvd_ms_median': med([t for t in tm if t['kind'] == k and t['R'] == R and t['q'] == q], 'rsvd_s')}
              for k, R in (('K', 256), ('V', 512)) for q in (0, 1, 2)}
    by_setting = {s: summ([t for t in tm if t['setting'] == s]) for s in sorted({t['setting'] for t in tm})}
    return {'timing_summary': by_setting.get('KV 256/512 P=768', {}), 'timing_summary_by_setting': by_setting,
            'timing_summary_pooled_old': pooled}


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    cols = {'offline': dv.C['black'], 'online': dv.C['blue'], 'rsvd q=0': dv.C['yellow'], 'rsvd q=1': dv.C['orange'], 'rsvd q=2': dv.C['aqua']}
    S = res['settings']
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.2, 1, 1]})
    # (a) half and third of the cache, P = 768
    ms = ['offline', 'online', 'rsvd q=0', 'rsvd q=1', 'rsvd q=2']
    names = ['KV 256/512 P=768', 'KV 128/384 P=768']
    x = np.arange(len(names)); w = 0.15
    for i, m in enumerate(ms):
        ax[0].bar(x + (i - 2) * w, [100 * (S[s]['ratio'][m] - 1) for s in names], w, color=cols[m], label=m)
    ax[0].axhline(5, color='gray', ls=':', lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['half cache\n(K 256, V 512)', 'third\n(K 128, V 384)'], fontsize=7)
    ax[0].set_ylabel('perplexity increase (%)'); ax[0].set_title('(a) prompt of 768 tokens')
    ax[0].legend(frameon=False, fontsize=6)
    # (b) keys only vs prompt length
    Ps = [384, 512, 768]
    for m in ('offline', 'online', 'rsvd q=1'):
        ax[1].plot(Ps, [100 * (S[f'K 256 P={P}']['ratio'][m] - 1) for P in Ps], marker='o', ms=3, color=cols[m], label=m)
    ax[1].set_xticks(Ps); ax[1].set_xlabel('prompt length P'); ax[1].set_ylabel('perplexity increase (%)')
    ax[1].set_title('(b) keys only (R = 256)'); ax[1].legend(frameon=False, fontsize=6)
    # (c) theory check: RSVD projection error / best error, with the Thm 5.5 factor for q = 0
    th = [t for t in res['theory'] if t['setting'].startswith('KV 256/512')]
    data, labs = [], []
    for k, R in (('K', 256), ('V', 512)):
        for q in (0, 1, 2):
            data.append([t['proj_ratio'] for t in th if t['kind'] == k and t['q'] == q]); labs.append(f'{k}\nq={q}')
    ax[2].boxplot(data, widths=0.6, flierprops={'markersize': 2})
    ax[2].set_xticklabels(labs, fontsize=7); ax[2].axhline(1, color='gray', lw=0.8)
    ax[2].set_ylabel('RSVD error / best rank-R error'); ax[2].set_title('(c) RSVD vs. the best basis')
    ax[2].text(0.97, 0.97, 'Thm 5.5 bound (q=0):\nK 5.4, V 7.6', transform=ax[2].transAxes, ha='right', va='top', fontsize=6.5)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig14_kv_online.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
