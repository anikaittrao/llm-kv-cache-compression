"""Step (2026-11-19): attention-weighted online basis (suggested by Prop 10.5 / Lemma 10.4 of report.pdf).

Run:  .venv\\Scripts\\python.exe experiments\\kv_attnbasis.py            (about 60-90 min, 16 windows)
      .venv\\Scripts\\python.exe experiments\\kv_attnbasis.py --replot
Writes experiments/results_kv_attnbasis.json and figures/fig19_kv_attnbasis.{pdf,png}.

Motivation. Prop 10.5 bounds the attention-output error by sums over tokens weighted by the attention a_i each token
receives (sum_i a_i e^{d_i} eps_i for values, a_i |d_i| for keys). The plain online basis (Section 9) minimises the
unweighted sum of squared residuals over the prompt rows, so tokens that are rarely read count as much as tokens that are
read often. Here each prompt row of the SVD is scaled by the square root of the attention it receives in the prompt.

Setting: identical to kv_online.py / kv_buffer.py (one basis per layer for all 12 heads, keys query-aware with the
prompt's query weight G_K, values output-aware G_V = W_O W_O^T, centered with the plain prompt mean, first token exact,
prompt of P tokens prefilled exactly then compressed, continuation in blocks of 32, perplexity on the continuation only).
Attention weights come from the exact prefill of the prompt (float64, causal softmax of q k^T / 8 per head):
  c_i(all)  = mean over the 12 heads of sum over all prompt queries t >= i of a^h_{t,i}
  c_i(late) = same, but only the last 128 prompt queries (t = P-128 .. P-1); these see (almost) every prompt token,
              like the future continuation queries do, so they avoid the causal bias of c(all) toward early tokens.
Position 0 (stored exactly) is left out. Row weights w_i = c_i / mean(c) (rows scaled by sqrt(w_i)).
Basis methods:
  offline, online (unweighted; the Section 9 basis),
  attn-all  : w = c(all)/mean,
  attn-late : w = c(late)/mean          <- the method the rules are about (fixed before the run),
  mix-late  : w = 0.5 + 0.5 c(late)/mean (half uniform, half attention).
Buffer: off (new keys/values compressed before attention reads them, as kv_online.py) or on (B = 32 exact recent
buffer, as kv_buffer.py 'static / buffer B=32'). No basis update in this run.
Code checks: online/no buffer = 25.775 (main) and 29.314 (P=384); online/buffer = 24.997 (main); offline/no buffer = 25.368.

Mechanism diagnostic (no-buffer methods): from one dense pass of the whole window, the head-averaged attention
abar_{t,i} of the continuation queries t >= P, and the weighted residual r_i = ||(x_i - x_hat_i) T||^2 with a common
metric T (prompt G_K^{1/2} for keys, G_V^{1/2} for values). D_prompt = sum_{t>=P} sum_{1<=i<P} abar_{t,i} r_i,
D_cont = sum_{t>=P} sum_{P<=i<=t} abar_{t,i} r_i. Also N_eff = (sum w)^2 / sum w^2 / (P-1), the effective share of rows.

Rules fixed before the run, main setting (P = 768, keys R = 256, values R = 512, half cache), e(x) = ppl(x)/dense - 1:
  W1 helps, no buffer : e(attn-late / no buffer) <= 0.9 x e(online / no buffer).
  W2 beats offline    : ppl(attn-late / no buffer) <= ppl(offline / no buffer)   (rule R2 of Section 9 failed).
  W3 helps, buffer    : e(attn-late / buffer) <= 0.9 x e(online / buffer).
  W4 mechanism        : median over windows and layers of D_prompt(attn-late)/D_prompt(online) < 1, for keys and values.
Secondary (reported, not deciding): third of the cache (K 128 / V 384, P = 768) and keys only (R = 256, P = 384).

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
from kv_online import load_wikitext, psd_pow, qblock  # noqa: E402

D64 = torch.float64
L, H, E, Dh = 12, 12, 768, 64
B = 32
LATE = 128
BASES = ['online', 'attn-all', 'attn-late', 'mix-late']


def top_right(A, R):
    _, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:R].T


def attention(X):
    """Causal attention probabilities (H, n, n) in float64 from the c_attn output X (n, 3E)."""
    n = X.shape[0]
    q = X[:, :E].reshape(n, H, Dh).transpose(0, 1)
    k = X[:, E:2 * E].reshape(n, H, Dh).transpose(0, 1)
    s = q @ k.transpose(1, 2) / math.sqrt(Dh)
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float('-inf'))
    return torch.softmax(s, -1)


def method_list(main):
    # (label, basis, buffer)
    m = [('dense', None, False), ('offline / no buffer', 'offline', False)]
    m += [(f'{b} / no buffer', b, False) for b in BASES]
    m += [('online / buffer', 'online', True), ('attn-late / buffer', 'attn-late', True)]
    if main:
        m += [('mix-late / buffer', 'mix-late', True), ('offline / buffer', 'offline', True)]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_attnbasis.json'))
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

    cap, mode = {}, {'proj': None}

    def hook(mod, args, out, l):
        cap[l] = out[0].detach()
        P_ = mode['proj']
        if P_ is None:
            return out
        o = out.clone()
        for kind, offs in (('K', E), ('V', 2 * E)):
            if kind in P_:
                mu, Pi = P_[kind][l]
                o[..., offs:offs + E] = mu + (o[..., offs:offs + E] - mu) @ Pi
        return o

    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: hook(m, a_, o, l)) for l in range(L)]

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

    settings = [('KV 256/512 P=768', 768, 256, 512, method_list(True)),
                ('KV 128/384 P=768', 768, 128, 384, method_list(False)),
                ('K 256 P=384', 384, 256, None, method_list(False))]
    if a.settings:
        settings = [st_ for st_ in settings if any(x in st_[0] for x in a.settings)]
    if a.only:
        settings = [(n_, P_, rk_, rv_, [m_ for m_ in ms_ if m_[0] == 'dense' or any(x in m_[0] for x in a.only)])
                    for (n_, P_, rk_, rv_, ms_) in settings]
    res = {'config': vars(a), 'torch': torch.__version__, 'B': B, 'late_queries': LATE,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads},
           'settings': {}, 'diag': [], 'neff': []}
    nll = {(s[0], m[0]): 0.0 for s in settings for m in s[4]}
    nllw = {(s[0], m[0]): [] for s in settings for m in s[4]}  # per-window NLL (audit patch B1)
    cnt = {s[0]: 0 for s in settings}

    def Pi(s):
        return s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1]

    for wi, w in enumerate(wins):
        # one dense pass of the whole window for the mechanism diagnostic
        mode['proj'] = None
        with torch.inference_mode():
            model(w[None])
        full = {l: cap[l].to(D64) for l in range(L)}
        for (sname, P, RK, RV, methods) in settings:
            kinds = [('K', RK, 'keys', E)] + ([('V', RV, 'values', 2 * E)] if RV else [])
            mode['proj'] = None
            with torch.inference_mode():
                model(w[:P][None])
            pre = {l: cap[l].to(D64) for l in range(L)}
            # ---- bases for this window and setting
            bases = {}
            for l in range(L):
                At = attention(pre[l])  # (H, P, P)
                c = {'attn-all': At.sum(1).mean(0)[1:], 'attn-late': At[:, P - LATE:, :].sum(1).mean(0)[1:]}
                wts = {'online': None, 'attn-all': c['attn-all'] / c['attn-all'].mean(),
                       'attn-late': c['attn-late'] / c['attn-late'].mean()}
                wts['mix-late'] = 0.5 + 0.5 * wts['attn-late']
                if sname.startswith('KV 256'):
                    for b in BASES[1:]:
                        ww = wts[b]
                        res['neff'].append({'window': wi, 'layer': l, 'basis': b,
                                            'neff_frac': float(ww.sum() ** 2 / (ww * ww).sum() / ww.numel())})
                Xq = pre[l][1:, :E]
                GK = qblock(Xq.T @ Xq / Xq.shape[0]); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
                for kind, R, _, offs in kinds:
                    T = TK if kind == 'K' else TV[l]
                    Xr = pre[l][1:, offs:offs + E]
                    mu = Xr.mean(0)
                    A0 = (Xr - mu) @ T[0]
                    for b in BASES:
                        A = A0 if wts[b] is None else wts[b].sqrt()[:, None] * A0
                        bases[(b, kind, l)] = {'mu': mu, 'T': T, 'U': top_right(A, R)}
                    muo, To, Uo = off[(kind, l)]
                    bases[('offline', kind, l)] = {'mu': muo, 'T': To, 'U': Uo[:, :R].clone()}
                    bases[('metric', kind, l)] = T
            # ---- mechanism diagnostic (no-buffer semantics: every token but position 0 compressed)
            for l in range(L):
                abar = attention(full[l]).mean(0)[P:]  # (seq-P, seq)
                for kind, R, _, offs in kinds:
                    X = full[l][:, offs:offs + E]
                    Tm = bases[('metric', kind, l)][0]
                    for b in ['offline'] + BASES:
                        s = bases[(b, kind, l)]
                        Xh = s['mu'] + (X - s['mu']) @ Pi(s)
                        r = ((X - Xh) @ Tm).pow(2).sum(1); r[0] = 0
                        contrib = abar * r[None, :]
                        res['diag'].append({'setting': sname, 'window': wi, 'layer': l, 'kind': kind, 'basis': b,
                                            'D_prompt': float(contrib[:, 1:P].sum()), 'D_cont': float(contrib[:, P:].sum()),
                                            'res_prompt': float(r[1:P].sum()), 'res_cont': float(r[P:].sum())})
            # ---- perplexity runs
            for (m, basis, buf) in methods:
                st = {}
                if basis is not None:
                    for kind, R, _, _ in kinds:
                        for l in range(L):
                            st[(kind, l)] = bases[(basis, kind, l)]
                proj = {kind: [(st[(kind, l)]['mu'].float(), Pi(st[(kind, l)]).float()) for l in range(L)]
                        for kind, _, _, _ in kinds} if basis is not None else None
                with torch.inference_mode():
                    mode['proj'] = None
                    out = model(w[:P][None], use_cache=True)
                    past = out.past_key_values
                    if basis is not None:
                        for kind, R, attr, _ in kinds:
                            for l in range(L):
                                lay = past.layers[l]
                                X = getattr(lay, attr)[0].transpose(0, 1).reshape(P, E).to(D64)
                                s = st[(kind, l)]
                                Xh = s['mu'] + (X - s['mu']) @ Pi(s); Xh[0] = X[0]
                                setattr(lay, attr, Xh.float().reshape(P, H, Dh).transpose(0, 1)[None].contiguous())
                    hp = proj if (basis is not None and not buf) else None
                    logits = []
                    t = P
                    while t < seq:
                        blk = w[t:t + B]; nb = blk.numel()
                        mode['proj'] = hp
                        out = model(blk[None], past_key_values=past)
                        mode['proj'] = None
                        logits.append(out.logits[0]); past = out.past_key_values
                        if basis is not None and buf and t + nb < seq:
                            for kind, R, attr, offs in kinds:
                                for l in range(L):
                                    lay = past.layers[l]
                                    X = getattr(lay, attr)[0].transpose(0, 1).reshape(t + nb, E).to(D64)
                                    s = st[(kind, l)]
                                    X[t:] = s['mu'] + (X[t:] - s['mu']) @ Pi(s)
                                    setattr(lay, attr, X.float().reshape(t + nb, H, Dh).transpose(0, 1)[None].contiguous())
                        t += nb
                    lp = torch.log_softmax(torch.cat(logits)[:-1].double(), -1)
                    tgt = w[P + 1:]
                    v_ = float(-lp[torch.arange(tgt.numel()), tgt].sum())
                    nll[(sname, m)] += v_; nllw[(sname, m)].append(v_)
            cnt[sname] += seq - P - 1
        print(f'window {wi+1}/{len(wins)} done ({time.perf_counter()-T0:.0f}s)', flush=True)

    for h_ in hooks:
        h_.remove()
    for (sname, P, RK, RV, methods) in settings:
        d = {m[0]: math.exp(nll[(sname, m[0])] / cnt[sname]) for m in methods}
        cf = ((RK or E) / E + (RV or E) / E) / 2
        res['settings'][sname] = {'P': P, 'RK': RK, 'RV': RV, 'cache_frac': cf, 'tokens': cnt[sname], 'ppl': d,
                                  'excess': {k: v / d['dense'] - 1 for k, v in d.items()},
                                  'nll_per_window': {m[0]: nllw[(sname, m[0])] for m in methods},
                                  'tokens_per_window': seq - P - 1, 'windows': win_idx}
        print(sname, {k: round(v, 3) for k, v in d.items()}, flush=True)

    # diagnostic summary: median over windows and layers of ratio to online
    dg = res['diag']
    summ = {}
    for (sname, _, RK, RV, _) in settings:
        for kind in (['K', 'V'] if RV else ['K']):
            ref = {(x['window'], x['layer']): x for x in dg if x['setting'] == sname and x['kind'] == kind and x['basis'] == 'online'}
            for b in ['offline'] + BASES[1:]:
                rows = [x for x in dg if x['setting'] == sname and x['kind'] == kind and x['basis'] == b]
                summ[f'{sname} {kind} {b}'] = {
                    q: float(np.median([x[q] / ref[(x['window'], x['layer'])][q] for x in rows]))
                    for q in ('D_prompt', 'D_cont', 'res_prompt', 'res_cont')}
            rows = [x for x in dg if x['setting'] == sname and x['kind'] == kind and x['basis'] == 'online']
            summ[f'{sname} {kind} online: D_cont share'] = float(np.median([x['D_cont'] / (x['D_cont'] + x['D_prompt']) for x in rows]))
    res['diag_summary'] = summ
    res['neff_summary'] = {b: {'median': float(np.median([x['neff_frac'] for x in res['neff'] if x['basis'] == b])),
                               'min': float(min(x['neff_frac'] for x in res['neff'] if x['basis'] == b))} for b in BASES[1:]
                           if any(x['basis'] == b for x in res['neff'])}  # empty for --settings runs without the main setting
    res['decision'] = {}
    try:
        e = res['settings']['KV 256/512 P=768']['excess']
        p = res['settings']['KV 256/512 P=768']['ppl']
        res['decision'] = {
            'W1_helps_no_buffer': e['attn-late / no buffer'] <= 0.9 * e['online / no buffer'],
            'W2_beats_offline': p['attn-late / no buffer'] <= p['offline / no buffer'],
            'W3_helps_buffer': e['attn-late / buffer'] <= 0.9 * e['online / buffer']}
    except KeyError as ex:  # --only / --settings runs lack some cells
        res['decision']['incomplete'] = f'missing {ex}'
    try:
        res['decision']['W4_mechanism'] = summ['KV 256/512 P=768 K attn-late']['D_prompt'] < 1 and summ['KV 256/512 P=768 V attn-late']['D_prompt'] < 1
    except KeyError as ex:
        res['decision']['incomplete_W4'] = f'missing {ex}'

    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decision:', res['decision'])
    print('diag:', json.dumps(summ, indent=1))
    print('neff:', res['neff_summary'])
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
    S = res['settings']
    cols = {'offline': dv.C['black'], 'online': dv.C['blue'], 'attn-all': dv.C['aqua'], 'attn-late': dv.C['orange'], 'mix-late': dv.C['yellow']}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'width_ratios': [1.4, 1]})
    names = ['KV 256/512 P=768', 'KV 128/384 P=768', 'K 256 P=384']
    ms = ['offline', 'online', 'attn-all', 'attn-late', 'mix-late']
    x = np.arange(len(names)); wd = 0.16
    for i, m in enumerate(ms):
        vals = [100 * S[s]['excess'][f'{m} / no buffer'] for s in names]
        ax[0].bar(x + (i - 2) * wd, vals, wd, color=cols[m], label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['half cache\nP = 768', 'third\nP = 768', 'keys only\nP = 384'], fontsize=7)
    ax[0].set_ylabel('perplexity increase (%)'); ax[0].set_title('(a) no buffer'); ax[0].legend(frameon=False, fontsize=6)
    bm = [('offline', 'offline'), ('online', 'online'), ('attn-late', 'attn-late'), ('mix-late', 'mix-late')]
    s = S['KV 256/512 P=768']['excess']
    vals = [100 * s[f'{m} / buffer'] for m, _ in bm]
    bars = ax[1].bar(range(len(bm)), vals, 0.6, color=[cols[m] for m, _ in bm])
    for b_, v in zip(bars, vals):
        ax[1].text(b_.get_x() + b_.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center', fontsize=6)
    ax[1].set_xticks(range(len(bm))); ax[1].set_xticklabels([l_ for _, l_ in bm], fontsize=7)
    ax[1].set_ylabel('perplexity increase (%)'); ax[1].set_title('(b) half cache, buffer of 32')
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig19_kv_attnbasis.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
