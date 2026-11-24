"""Step (2026-11-19): separate the exact recent buffer from the basis update (follow-up to kv_update.py / report.pdf Section 11).

Run:  .venv\\Scripts\\python.exe experiments\\kv_buffer.py            (about 60-90 min, 16 windows)
      .venv\\Scripts\\python.exe experiments\\kv_buffer.py --replot
Writes experiments/results_kv_buffer.json and figures/fig18_kv_buffer.{pdf,png}.

Motivation. In Section 11 the updated basis reached +0.2% (half cache), but a post hoc comparison suggested that most of the
gain came from the exact recent buffer that all methods there used (the block being processed is read exactly), not from
the update. This run is a 2 x 2 design that separates the two, with the rules fixed before the run.

Setting: identical to kv_update.py (one basis per layer for all 12 heads, keys query-aware with the prompt's query weight,
values output-aware, centered with the prompt mean, first token exact, prompt of P tokens prefilled exactly then
compressed, continuation processed in blocks of B tokens, perplexity on the continuation only).
Two switches:
  buffer   on : while a block is processed, attention reads the block's own keys/values exactly (recent buffer of up
                to B tokens); they are compressed at the end of the block (as in kv_update.py).
           off: every new key/value is compressed as soon as it is made, before attention reads it (as in kv_online.py).
  update   on : at the end of each block, incremental SVD [S U^T ; A_blk] -> top R, older rows re-projected, the block
                compressed with the new basis (as kv_update.py 'update'). The exact block rows are needed for the SVD
                in both buffer modes (same memory); 'buffer off' only means attention never reads them.
           off: the prompt basis is kept (static).
Methods: dense; static/no buffer; static/buffer with B = 8, 16, 32, 64, 128; update/no buffer (B = 32);
update/buffer (B = 32); offline/no buffer; offline/buffer (B = 32) for reference.
Code checks: static/no buffer must reproduce kv_online.py 'online' (25.775 main, 29.314 at P = 384); static/buffer B=32
and update/buffer must reproduce kv_update.py 'static' 24.997 and 'update' 24.778 (main).

Rules fixed before the run, on the main setting (P = 768, keys R = 256, values R = 512, half cache), with
e(x) = ppl(x) / dense - 1:
  F1 update alone : e(update / no buffer)        <= 0.5 x e(static / no buffer).
  F2 buffer alone : e(static / buffer B=32)      <= 0.5 x e(static / no buffer).
  F3 complementary: e(update / buffer B=32)      <= 0.5 x min(e(update / no buffer), e(static / buffer B=32)).
  F4 buffer size  : e(static / buffer B) is non-increasing over B = 0 (no buffer), 8, 16, 32, 64, 128.
The keys-only short-prompt setting (P = 384, R = 256) is reported with F1-F3 as a secondary check (not deciding).

Options added 2026-11-22 (audit patches B1/B5; the defaults reproduce the reported run exactly):
  --window_offset k : use test windows k .. k+windows-1 (16 = fresh windows 16-31 for the confirmation run C1).
  --only s1 s2 ...  : run only methods whose label contains one of the strings ('dense' always runs).
  --settings s1 ... : run only settings whose name contains one of the strings.
  --calib_skip0     : offline calibration statistics without position 0 of every calibration window (the online
                      statistics never use position 0); C1b.
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
BSIZES = [8, 16, 32, 64, 128]


def top_right(A, R):
    _, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:R].T, s[:R]


def method_list(main):
    # (label, basis, update, buffer, B)
    m = [('dense', None, False, False, 32), ('static / no buffer', 'static', False, False, 32)]
    if main:
        m += [(f'static / buffer B={b}', 'static', False, True, b) for b in BSIZES]
    else:
        m += [('static / buffer B=32', 'static', False, True, 32)]
    m += [('update / no buffer', 'static', True, False, 32), ('update / buffer B=32', 'static', True, True, 32)]
    if main:
        m += [('offline / no buffer', 'offline', False, False, 32), ('offline / buffer B=32', 'offline', False, True, 32)]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_buffer.json'))
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
        cap[l] = out[0].detach()  # exact q, k, v of the tokens in this forward
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

    # ---- offline calibration (same 16 windows as kv_adaptive.py / kv_online.py / kv_update.py)
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
                ('K 256 P=384', 384, 256, None, method_list(False))]
    if a.settings:
        settings = [st_ for st_ in settings if any(x in st_[0] for x in a.settings)]
    if a.only:
        settings = [(n_, P_, rk_, rv_, [m_ for m_ in ms_ if m_[0] == 'dense' or any(x in m_[0] for x in a.only)])
                    for (n_, P_, rk_, rv_, ms_) in settings]
    res = {'config': vars(a), 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads},
           'settings': {}}
    nll = {(s[0], m[0]): 0.0 for s in settings for m in s[4]}
    nllw = {(s[0], m[0]): [] for s in settings for m in s[4]}  # per-window NLL (audit patch B1)
    cnt = {s[0]: 0 for s in settings}

    def Pi(s):
        return s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1]

    for wi, w in enumerate(wins):
        for (sname, P, RK, RV, methods) in settings:
            kinds = [('K', RK, 'keys', E)] + ([('V', RV, 'values', 2 * E)] if RV else [])
            mode['proj'] = None
            with torch.inference_mode():
                model(w[:P][None])
            pre = {l: cap[l].to(D64) for l in range(L)}
            for (m, basis, upd, buf, B) in methods:
                st = {}
                if basis is not None:
                    for kind, R, _, offs in kinds:
                        for l in range(L):
                            if basis == 'offline':
                                mu, T, U = off[(kind, l)]
                                st[(kind, l)] = {'mu': mu, 'T': T, 'U': U[:, :R].clone(), 'S': None}
                                continue
                            Xq = pre[l][1:, :E]
                            Xr = pre[l][1:, offs:offs + E]
                            if kind == 'K':
                                GK = qblock(Xq.T @ Xq / Xq.shape[0]); T = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
                            else:
                                T = TV[l]
                            mu = Xr.mean(0)
                            U, S = top_right((Xr - mu) @ T[0], R)
                            st[(kind, l)] = {'mu': mu, 'T': T, 'U': U, 'S': S}

                def hook_proj():
                    return {kind: [(st[(kind, l)]['mu'].float(), Pi(st[(kind, l)]).float()) for l in range(L)]
                            for kind, _, _, _ in kinds}

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
                    hp = hook_proj() if (basis is not None and not buf) else None
                    logits = []
                    t = P
                    while t < seq:
                        blk = w[t:t + B]; nb = blk.numel()
                        mode['proj'] = hp  # no buffer: new keys/values compressed before attention reads them
                        out = model(blk[None], past_key_values=past)
                        mode['proj'] = None
                        logits.append(out.logits[0]); past = out.past_key_values
                        newx = {l: cap[l].to(D64) for l in range(L)}  # exact rows of the block
                        if basis is not None and t + nb < seq and (buf or upd):
                            for kind, R, attr, offs in kinds:
                                for l in range(L):
                                    lay = past.layers[l]
                                    X = getattr(lay, attr)[0].transpose(0, 1).reshape(t + nb, E).to(D64)
                                    s = st[(kind, l)]
                                    Xnew = newx[l][:, offs:offs + E]
                                    if upd:
                                        Ablk = (Xnew - s['mu']) @ s['T'][0]
                                        M = torch.cat([s['S'][:, None] * s['U'].T, Ablk])
                                        s['U'], s['S'] = top_right(M, R)
                                        Pn = Pi(s)
                                        X[1:t] = s['mu'] + (X[1:t] - s['mu']) @ Pn
                                    else:
                                        Pn = Pi(s)
                                    X[t:] = s['mu'] + (Xnew - s['mu']) @ Pn
                                    setattr(lay, attr, X.float().reshape(t + nb, H, Dh).transpose(0, 1)[None].contiguous())
                            if upd and not buf:
                                hp = hook_proj()
                        t += nb
                    lg = torch.cat(logits)
                    lp = torch.log_softmax(lg[:-1].double(), -1)
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

    def rules(e):
        return {'F1_update_alone_halves_excess': e['update / no buffer'] <= 0.5 * e['static / no buffer'],
                'F2_buffer_alone_halves_excess': e['static / buffer B=32'] <= 0.5 * e['static / no buffer'],
                'F3_complementary': e['update / buffer B=32'] <= 0.5 * min(e['update / no buffer'], e['static / buffer B=32'])}
    try:
        em = res['settings']['KV 256/512 P=768']['excess']
        res['decision'] = rules(em)
        seqB = [em['static / no buffer']] + [em[f'static / buffer B={b}'] for b in BSIZES]
        res['decision']['F4_buffer_size_monotone'] = all(seqB[i + 1] <= seqB[i] for i in range(len(seqB) - 1))
    except KeyError as ex:  # --only / --settings runs lack some cells
        res.setdefault('decision', {})['incomplete'] = f'missing {ex}'
    try:
        res['secondary_P384'] = rules(res['settings']['K 256 P=384']['excess'])
    except KeyError as ex:
        res['secondary_P384'] = {'incomplete': f'missing {ex}'}
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decision:', res['decision'])
    print('secondary P=384:', res['secondary_P384'])
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
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'width_ratios': [1.2, 1]})
    names = ['KV 256/512 P=768', 'K 256 P=384']
    cells = [('static / no buffer', 'neither', dv.C['blue']), ('static / buffer B=32', 'buffer only', dv.C['yellow']),
             ('update / no buffer', 'update only', dv.C['magenta']), ('update / buffer B=32', 'both', dv.C['orange'])]
    x = np.arange(len(names)); w = 0.2
    for i, (m, lab, c) in enumerate(cells):
        vals = [100 * S[s]['excess'][m] for s in names]
        bars = ax[0].bar(x + (i - 1.5) * w, vals, w, color=c, label=lab)
        for b_, v in zip(bars, vals):
            ax[0].text(b_.get_x() + b_.get_width() / 2, v + 0.1, f'{v:.1f}', ha='center', fontsize=6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['half cache, P = 768\n(K 256, V 512)', 'keys only, P = 384\n(K 256)'], fontsize=7)
    ax[0].set_ylabel('perplexity increase (%)'); ax[0].set_title('(a) buffer and update, blocks of 32')
    ax[0].legend(frameon=False, fontsize=6)
    e = S['KV 256/512 P=768']['excess']
    bs = [0] + BSIZES
    ax[1].plot(bs, [100 * e['static / no buffer']] + [100 * e[f'static / buffer B={b}'] for b in BSIZES], 'o-',
               color=dv.C['yellow'], ms=3, label='static basis, buffer of B')
    ax[1].axhline(100 * e['update / buffer B=32'], color=dv.C['orange'], ls='--', lw=1, label='update + buffer (B = 32)')
    ax[1].axhline(100 * e['update / no buffer'], color=dv.C['magenta'], ls=':', lw=1, label='update, no buffer')
    ax[1].set_xscale('symlog', linthresh=8); ax[1].set_xticks(bs); ax[1].set_xticklabels([str(b) for b in bs])
    ax[1].set_xlabel('buffer / block size B (tokens)'); ax[1].set_ylabel('perplexity increase (%)')
    ax[1].set_ylim(bottom=0); ax[1].set_title('(b) half cache: buffer size'); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig18_kv_buffer.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
