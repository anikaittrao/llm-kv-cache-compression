"""Step (2026-11-19): a KV basis that is UPDATED during generation (follow-up to kv_online.py / report.pdf Sections 9-10).

Run:  .venv\\Scripts\\python.exe experiments\\kv_update.py            (about 60-90 min, 16 windows)
      .venv\\Scripts\\python.exe experiments\\kv_update.py --replot
Writes experiments/results_kv_update.json and figures/fig16_kv_update.{pdf,png}.

Motivation. Section 10 found that the online (prompt) basis fits later tokens 2.4x (keys) / 6.2x (values) worse than the
prompt it was computed from, which is why it lost to the offline basis (rule R2 of Section 9).

Setting. Same recipe as kv_online.py: one basis per layer for all 12 heads, keys query-aware (G = blockdiag of the prompt's
query second moments), values output-aware (G = W_O W_O^T), centered with the prompt mean, first token stored exactly.
The prompt (P tokens) is prefilled exactly, then compressed. The continuation is processed in blocks of B tokens:
  * while a block is processed, its own keys/values are exact (a recent buffer of at most B tokens; all methods, including
    the static ones, use this same scheme, so they are compared fairly);
  * at the end of the block, the basis may be updated from the block's exact weighted rows A_blk = (X_blk - mu) G^{1/2},
    the older cache rows may be re-projected onto the new basis, and the block is compressed.
Weighting, mean and G stay fixed (from the prompt); only the subspace U moves.
Methods:
  dense            : no compression.
  offline          : calibration basis (report.pdf Section 8), never updated.
  static           : online prompt basis (exact SVD), never updated.
  update           : incremental SVD.  [S U^T ; A_blk] -> SVD -> keep top R (U, S).  Old rows re-projected:
                     x_hat <- mu + (x_hat - mu) G^{1/2} U' U'^T G^{-1/2}   (= mapping the stored codes c -> c U^T U').
  update decay     : same, with the old sketch scaled by lambda = 0.9 per block (recent tokens weigh more).
  update no-reproj : the basis is updated for new tokens only; old rows keep their old basis (needs one basis per block in
                     practice; a diagnostic that isolates the loss from re-projection).
  rsvd q=1 + update: prompt basis by RSVD (Algorithm 2 of background.pdf, p = 10, q = 1), then the same updates.
  oracle refit     : exact SVD of ALL exact rows so far, whole cache re-compressed from exact values (not realizable: it
                     needs the exact old cache; an upper bound on what any update can do with this subspace size).
Diagnostic: the "novelty" of each block = RMS residual of its weighted rows outside the current basis (before the
update), divided by the RMS residual of the prompt rows under the prompt basis (the 2.4x / 6.2x quantity of Section 10).

Rules fixed before the run (main setting: P = 768, keys R = 256, values R = 512, half cache, B = 32):
  U1 update beats calibration : ppl(update) <= ppl(offline).
  U2 RSVD fidelity            : ppl(rsvd q=1 + update) <= 1.01 x ppl(update).
  U3 quality                  : ppl(rsvd q=1 + update) <= 1.05 x dense.
  U4 short prompt (keys only, R = 256, P = 384): the update removes at least half of the static basis's excess:
                                ppl(update) - dense <= 0.5 x (ppl(static) - dense).
Extras: block size B = 128 on the main setting (static, update, oracle).

Options added 2026-11-22 (audit patches B1/B5; the defaults reproduce the reported run exactly):
  --window_offset k : use test windows k .. k+windows-1 (16 = fresh windows 16-31 for the confirmation run C1).
  --only s1 s2 ...  : run only methods whose label contains one of the strings ('dense' always runs).
  --settings s1 ... : run only settings whose name contains one of the strings.
  --calib_skip0     : offline calibration statistics without position 0 of every calibration window (the online
                      statistics never use position 0).
  The JSON now also stores, per setting, 'nll_per_window' (summed NLL of each window, per method), 'tokens_per_window'
  and 'windows' (the test-window indices), for paired bootstrap CIs (experiments/stats_utils.py).
"""
import argparse, json, math, os, platform, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from kv_online import load_wikitext, psd_pow, qblock, rsvd_right  # noqa: E402

D64 = torch.float64
L, H, E, Dh = 12, 12, 768, 64
LAMBDA = 0.9


def top_right(A, R):
    _, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:R].T, s[:R], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_update.json'))
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--settings', nargs='*', default=None)
    ap.add_argument('--calib_skip0', action='store_true')
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
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    seq = 1024
    win_idx = list(range(a.window_offset, a.window_offset + a.windows))
    wins = [ids_test[i * seq:(i + 1) * seq] for i in win_idx]
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
    TV = [(psd_pow(WO[l] @ WO[l].T, 0.5), psd_pow(WO[l] @ WO[l].T, -0.5)) for l in range(L)]

    cap = {}
    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(
        lambda m, a_, o, l=l: cap.__setitem__(l, o[0].detach())) for l in range(L)]

    # ---- offline calibration (same 16 windows as kv_adaptive.py / kv_online.py)
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
    off = {}
    for l in range(L):
        muK, muV = S1['K'][l] / n, S1['V'][l] / n
        CK = S2['K'][l] / n - torch.outer(muK, muK); CV = S2['V'][l] / n - torch.outer(muV, muV)
        GK = qblock(S2['Q'][l] / n); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
        off[('K', l)] = (muK, TK, torch.linalg.eigh(TK[0] @ CK @ TK[0])[1].flip(1))
        off[('V', l)] = (muV, TV[l], torch.linalg.eigh(TV[l][0] @ CV @ TV[l][0])[1].flip(1))
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    settings = [  # (name, P, RK, RV, B, methods)
        ('KV 256/512 P=768 B=32', 768, 256, 512, 32,
         ['dense', 'offline', 'static', 'update', 'update decay', 'update no-reproj', 'rsvd q=1 + update', 'oracle refit']),
        ('K 256 P=384 B=32', 384, 256, None, 32,
         ['dense', 'offline', 'static', 'update', 'update decay', 'rsvd q=1 + update', 'oracle refit']),
        ('KV 256/512 P=768 B=128', 768, 256, 512, 128, ['static', 'update', 'oracle refit']),
    ]
    if a.settings:
        settings = [st_ for st_ in settings if any(x in st_[0] for x in a.settings)]
    if a.only:
        settings = [st_[:5] + ([m_ for m_ in st_[5] if m_ == 'dense' or any(x in m_ for x in a.only)],) for st_ in settings]
    res = {'config': vars(a), 'lambda': LAMBDA, 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads},
           'settings': {}, 'novelty': [], 'timing': []}
    nll = {(s[0], m): 0.0 for s in settings for m in s[5]}
    nllw = {(s[0], m): [] for s in settings for m in s[5]}  # per-window NLL (audit patch B1)
    cnt = {s[0]: 0 for s in settings}
    dense_nll_main = {}

    for wi, w in enumerate(wins):
        for (sname, P, RK, RV, B, methods) in settings:
            kinds = [('K', RK, 'keys')] + ([('V', RV, 'values')] if RV else [])
            # prompt statistics (exact prefill)
            with torch.inference_mode():
                model(w[:P][None])
            pre = {l: cap[l].to(D64) for l in range(L)}
            for m in methods:
                st = {}  # (kind, l) -> dict(mu, T, U, S, R)
                for kind, R, _ in kinds:
                    for l in range(L):
                        if m == 'offline':
                            mu, T, U = off[(kind, l)]
                            st[(kind, l)] = {'mu': mu, 'T': T, 'U': U[:, :R].clone(), 'S': None, 'R': R}
                            continue
                        Xq = pre[l][1:, :E]
                        Xr = pre[l][1:, E:2 * E] if kind == 'K' else pre[l][1:, 2 * E:]
                        if kind == 'K':
                            GK = qblock(Xq.T @ Xq / Xq.shape[0]); T = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
                        else:
                            T = TV[l]
                        mu = Xr.mean(0)
                        A = (Xr - mu) @ T[0]
                        U, S, sv = top_right(A, R)
                        prompt_rms = float(sv[R:].pow(2).sum().sqrt()) / math.sqrt(A.shape[0])
                        if m == 'rsvd q=1 + update':
                            gen = torch.Generator().manual_seed(1000 * wi + 10 * l + (0 if kind == 'K' else 1) + 100000)
                            U, _ = rsvd_right(A, R, a.p, 1, gen)
                            S = (A @ U).norm(dim=0)  # singular values of A along the RSVD directions
                        st[(kind, l)] = {'mu': mu, 'T': T, 'U': U, 'S': S, 'R': R, 'prompt_rms': prompt_rms}

                def Pi(s):
                    return (s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1])

                with torch.inference_mode():
                    out = model(w[:P][None], use_cache=True)
                    past = out.past_key_values
                    exact = {}
                    if m != 'dense':
                        for kind, R, attr in kinds:
                            for l in range(L):
                                lay = past.layers[l]
                                X = getattr(lay, attr)[0].transpose(0, 1).reshape(P, E).to(D64)
                                exact[(kind, l)] = X.clone()
                                s = st[(kind, l)]
                                Xh = s['mu'] + (X - s['mu']) @ Pi(s); Xh[0] = X[0]
                                setattr(lay, attr, Xh.float().reshape(P, H, Dh).transpose(0, 1)[None].contiguous())
                    logits = []
                    t, bi = P, 0
                    while t < seq:
                        blk = w[t:t + B]; nb = blk.numel()
                        out = model(blk[None], past_key_values=past)
                        logits.append(out.logits[0]); past = out.past_key_values
                        if m != 'dense' and t + nb < seq:  # compress the block (last block: no later token reads it)
                            for kind, R, attr in kinds:
                                for l in range(L):
                                    lay = past.layers[l]
                                    X = getattr(lay, attr)[0].transpose(0, 1).reshape(t + nb, E).to(D64)
                                    s = st[(kind, l)]
                                    Xnew = X[t:].clone()
                                    exact[(kind, l)] = torch.cat([exact[(kind, l)], Xnew])
                                    Ablk = (Xnew - s['mu']) @ s['T'][0]
                                    if 'prompt_rms' in s and m in ('static', 'update', 'oracle refit') and B == 32:
                                        r = Ablk - (Ablk @ s['U']) @ s['U'].T
                                        res['novelty'].append({'setting': sname, 'method': m, 'window': wi, 'block': bi,
                                                               'layer': l, 'kind': kind,
                                                               'ratio': float(r.norm()) / math.sqrt(nb) / s['prompt_rms']})
                                    if m in ('update', 'update decay', 'update no-reproj', 'rsvd q=1 + update'):
                                        lam = LAMBDA if m == 'update decay' else 1.0
                                        tt = time.perf_counter()
                                        M = torch.cat([(lam * s['S'])[:, None] * s['U'].T, Ablk])
                                        s['U'], s['S'], _ = top_right(M, R)
                                        res['timing'].append({'kind': kind, 'R': R, 'B': nb, 'what': 'update_svd',
                                                              's': time.perf_counter() - tt})
                                        if m != 'update no-reproj':
                                            Pn = Pi(s)
                                            X[1:t] = s['mu'] + (X[1:t] - s['mu']) @ Pn
                                        else:
                                            Pn = Pi(s)
                                        X[t:] = s['mu'] + (Xnew - s['mu']) @ Pn
                                    elif m == 'oracle refit':
                                        Xe = exact[(kind, l)]
                                        A = (Xe[1:] - s['mu']) @ s['T'][0]
                                        s['U'], s['S'], _ = top_right(A, R)
                                        Pn = Pi(s)
                                        X = Xe.clone(); X[1:] = s['mu'] + (Xe[1:] - s['mu']) @ Pn
                                    else:  # static / offline
                                        X[t:] = s['mu'] + (Xnew - s['mu']) @ Pi(s)
                                    setattr(lay, attr, X.float().reshape(t + nb, H, Dh).transpose(0, 1)[None].contiguous())
                        t += nb; bi += 1
                    lg = torch.cat(logits)
                    if m == 'dense' and wi == 0 and 'sanity_max_logit_diff' not in res:
                        full = model(w[None]).logits[0, P:]
                        res['sanity_max_logit_diff'] = float((full - lg).abs().max())
                        print('sanity: blockwise dense vs one pass, max |logit diff|:', res['sanity_max_logit_diff'], flush=True)
                    lp = torch.log_softmax(lg[:-1].double(), -1)
                    tgt = w[P + 1:]
                    v = float(-lp[torch.arange(tgt.numel()), tgt].sum())
                    nll[(sname, m)] += v; nllw[(sname, m)].append(v)
                    if m == 'dense':
                        dense_nll_main[(sname, wi)] = v
            cnt[sname] += seq - P - 1
        print(f'window {wi+1}/{len(wins)} done ({time.perf_counter()-T0:.0f}s)', flush=True)

    for h_ in hooks:
        h_.remove()
    # B=128 shares dense with B=32 (dense does not depend on the block size)
    for (sname, P, RK, RV, B, methods) in settings:
        d = {m: math.exp(nll[(sname, m)] / cnt[sname]) for m in methods}
        if 'dense' not in d:
            if 'KV 256/512 P=768 B=32' in res['settings']:
                d['dense'] = res['settings']['KV 256/512 P=768 B=32']['ppl']['dense']
            else:  # --settings run without the main setting: no dense reference for B=128
                d['dense'] = float('nan')
        cf = ((RK or E) / E + (RV or E) / E) / 2
        res['settings'][sname] = {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'cache_frac': cf, 'tokens': cnt[sname], 'ppl': d,
                                  'ratio': {k: v / d['dense'] for k, v in d.items()},
                                  'nll_per_window': {m: nllw[(sname, m)] for m in methods},
                                  'tokens_per_window': seq - P - 1, 'windows': win_idx}
        print(sname, {k: round(v, 3) for k, v in d.items()}, flush=True)
    try:
        mp = res['settings']['KV 256/512 P=768 B=32']['ppl']
        kp = res['settings']['K 256 P=384 B=32']['ppl']
        res['decision'] = {
            'U1_update_no_worse_than_offline': mp['update'] <= mp['offline'],
            'U2_rsvd_update_within_1pct_of_update': mp['rsvd q=1 + update'] <= 1.01 * mp['update'],
            'U3_rsvd_update_within_5pct_of_dense': mp['rsvd q=1 + update'] <= 1.05 * mp['dense'],
            'U4_P384_update_halves_static_excess': (kp['update'] - kp['dense']) <= 0.5 * (kp['static'] - kp['dense'])}
    except KeyError as ex:  # --only / --settings runs lack some cells
        res['decision'] = {'incomplete': f'missing {ex}'}
    nv = res['novelty']
    res['novelty_summary'] = {f'{s} | {m} | {k}': {'median': float(np.median([x['ratio'] for x in nv if x['setting'] == s and x['method'] == m and x['kind'] == k])),
                                                   'median_last_block': float(np.median([x['ratio'] for x in nv if x['setting'] == s and x['method'] == m and x['kind'] == k and x['block'] == max(y['block'] for y in nv if y['setting'] == s)]))}
                              for s in {x['setting'] for x in nv} for m in ('static', 'update', 'oracle refit') for k in 'KV'
                              if any(x['setting'] == s and x['method'] == m and x['kind'] == k for x in nv)}
    tm = res['timing']
    res['timing_summary'] = {f'{k} R={R} B={b}': 1e3 * float(np.median([x['s'] for x in tm if x['kind'] == k and x['R'] == R and x['B'] == b]))
                             for k, R, b in {(x['kind'], x['R'], x['B']) for x in tm}}
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decision:', res['decision'])
    print('novelty:', json.dumps(res['novelty_summary'], indent=1))
    print('update svd ms (median):', res['timing_summary'])
    try:
        make_figures(a, res)
    except (KeyError, ValueError) as ex:
        print('figure skipped (subset run):', ex)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    cols = {'offline': dv.C['black'], 'static': dv.C['blue'], 'update': dv.C['orange'], 'update decay': dv.C['yellow'],
            'update no-reproj': dv.C['magenta'], 'rsvd q=1 + update': dv.C['aqua'], 'oracle refit': dv.C['gray']}
    S = res['settings']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.4, 1]})
    names = ['KV 256/512 P=768 B=32', 'K 256 P=384 B=32']
    ms = ['offline', 'static', 'update', 'update decay', 'update no-reproj', 'rsvd q=1 + update', 'oracle refit']
    x = np.arange(len(names)); w = 0.115
    for i, m in enumerate(ms):
        vals = [100 * (S[s]['ratio'][m] - 1) if m in S[s]['ratio'] else np.nan for s in names]
        ax[0].bar(x + (i - 3) * w, vals, w, color=cols[m], label=m)
    ax[0].axhline(5, color='gray', ls=':', lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['half cache, P = 768\n(K 256, V 512)', 'keys only, P = 384\n(K 256)'], fontsize=7)
    ax[0].set_ylabel('perplexity increase (%)'); ax[0].set_title('(a) blocks of 32 tokens')
    ax[0].legend(frameon=False, fontsize=6, ncol=2)
    nv = [x_ for x_ in res['novelty'] if x_['setting'] == 'KV 256/512 P=768 B=32']
    for m, ls in (('static', '-'), ('update', '--'), ('oracle refit', ':')):
        for k, c in (('K', dv.C['blue']), ('V', dv.C['orange'])):
            bs = sorted({x_['block'] for x_ in nv})
            med = [np.median([x_['ratio'] for x_ in nv if x_['method'] == m and x_['kind'] == k and x_['block'] == b]) for b in bs]
            ax[1].plot([768 + 32 * b for b in bs], med, ls=ls, color=c, marker='o', ms=2, label=f'{k} {m}')
    ax[1].axhline(1, color='gray', lw=0.8)
    ax[1].set_xlabel('block start (token position)'); ax[1].set_ylabel('new-block residual / prompt residual')
    ax[1].set_title('(b) how new each block is'); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig16_kv_update.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
