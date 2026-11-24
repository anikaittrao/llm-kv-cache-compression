"""C2 (audit 2026-11-22): WHY do the exact buffer and the basis update help, and which exact tokens matter?

Run:  .venv\\Scripts\\python.exe experiments\\kv_synergy.py --out experiments\\rerun_2026-11-22\\c2_kv_synergy.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 65 min, 16 windows; smoke: 236 s per window)
      smoke test: add --windows 1 --smoke   (code checks on window 0 against kv_buffer.py, ~5 min)
      --replot redraws figures/c2... from the JSON.
Writes the JSON given by --out and <figdir>/fig_c2_kv_synergy.{pdf,png}.

Setting: kv_buffer.py main setting (one basis per layer, K 256 / V 512 = half the codes, P = 768 prompt prefilled
exactly then compressed, first token exact, 256 continuation tokens, perplexity on the continuation), static prompt
basis unless noted. The reported 2x2 (kv_buffer.py) compared "no buffer" (a new token's own K/V are compressed BEFORE
its own attention, which no real cache does) with a 32-token BLOCK buffer (exact window between 1 and 32 tokens).
Methods (all static prompt basis unless noted; exact token budgets in brackets):
  dense
  static / no buffer           [token 0]                        code check = kv_buffer.py 'static / no buffer'
  static / B=1                 [own token while it is processed]  token-by-token; realistic "no buffer"
  static / block 32            [1-32 most recent]               code check = kv_buffer.py 'static / buffer B=32'
  static / sliding 32          [own + 31 most recent, at every step] token-by-token
  static / random 31 + own     [own + 31 random prompt tokens 1..P-1, fixed per window (seed = window index)]
  static / top-attended 31 + own [own + the 31 prompt tokens that receive the most attention from the last 128 prompt
                                queries in prefill (head mean), i.e. the 'attn-late' weights of kv_attnbasis.py]
  update / no buffer           code check = kv_buffer.py 'update / no buffer'; with the age diagnostic below
  update / block 32            code check = kv_buffer.py 'update / buffer B=32'
  attn-late / no buffer        (kv_attnbasis.py basis) with the positive-d diagnostic below
Diagnostics (computed from the run's own queries and its compressed cache, every block, head-mean attention abar):
  age split (static/no buffer, update/no buffer, static/block 32, update/block 32): residual r_i = ||(x_hat_i - x_i)
     G^{1/2}||^2 against the run's own exact rows; D_young = sum over block queries t and keys i with t - i < 32 of
     abar_{t,i} r_i, D_old = same for t - i >= 32; share_young = D_young / (D_young + D_old) per window, layer, kind.
     Attention mass by age: share of abar on position 0, ages 0-15, 16-31, >= 32 (also for dense).
  positive-d share (static/no buffer = unweighted online basis, attn-late/no buffer): for each head and block query,
     delta_i = q.(k_hat_i - k_i)/8, d_i = delta_i - sum_j a_j delta_j (a = attention with the run's exact keys);
     share = sum over PROMPT tokens with a_i < 1/n_keys and d_i > 0 of |a^_i - a_i|  /  sum_i |a^_i - a_i|.
Rules fixed before the run (main setting, 16 windows), e(x) = ppl(x)/dense - 1:
  S1 own token : e(static / B=1) <= 0.7 x e(static / no buffer)        (B = 1 removes >= 30% of the no-buffer excess).
  S2 recency   : e(static / sliding 32) <= 0.5 x e(static / random 31 + own).
  S3 young error: median over windows and layers of share_young(update / no buffer) >= 0.5, for keys AND values.
  S4 sliding   : e(static / sliding 32) <= e(static / block 32).
Descriptive: e(top-attended 31 + own) vs sliding and random; positive-d share attn-late vs unweighted (A10 / gap 11).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, prompt_state, offline_state, kinds_of, run_block, run_tokens,  # noqa: E402
                   head_attention, get_rows, base_result, ppl_table, save_json, rss_mb)
from kv_attnbasis import attention, LATE  # noqa: E402

P, RK, RV, B = 768, 256, 512, 32
AGE = 32
METHODS = ['dense', 'static / no buffer', 'static / B=1', 'static / block 32', 'static / sliding 32',
           'static / random 31 + own', 'static / top-attended 31 + own', 'update / no buffer', 'update / block 32',
           'attn-late / no buffer']
AGE_DIAG = ('static / no buffer', 'update / no buffer', 'static / block 32', 'update / block 32')
POSD_DIAG = ('static / no buffer', 'attn-late / no buffer')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true', help='print the code checks against kv_buffer.py window 0')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c2_kv_synergy.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    ctx = Ctx(a.threads, need_train=False)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'windows': win_idx, 'tokens_per_window': 1024 - P - 1,
                          'age_diag': [], 'attn_mass': [], 'posd_diag': [], 'top_attended': {}})
    nllw = {m: [] for m in methods}

    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        # attn-late weights (kv_attnbasis.py) and the top-attended prompt tokens
        wlate, cl = {}, torch.zeros(P - 1, dtype=D64)
        for l in range(L):
            At = attention(pre[l])
            c = At[:, P - LATE:, :].sum(1).mean(0)[1:]
            wlate[l] = c / c.mean(); cl += c
        top = (torch.topk(cl, 31).indices + 1).tolist()
        g = torch.Generator().manual_seed(wi)
        rnd = (torch.randperm(P - 1, generator=g)[:31] + 1).tolist()
        res['top_attended'][str(wi)] = sorted(top)
        for m in methods:
            st = None if m == 'dense' else prompt_state(ctx, pre, kinds, row_weights=wlate if m.startswith('attn-late') else None)
            diag_age = m in AGE_DIAG or m == 'dense'
            diag_posd = m in POSD_DIAG
            acc = {}
            if diag_age or diag_posd:
                ex_store = {}
                for kind, R, attr, offs in kinds:
                    for l in range(L):
                        ex_store[(kind, l)] = pre[l][:, offs:offs + E].clone()  # exact prompt rows (prefill = run's own)

                def on_block(t, nb, past, newx, st_, ex_store=ex_store, acc=acc, m=m):
                    for kind, R, attr, offs in kinds:
                        for l in range(L):
                            ex_store[(kind, l)] = torch.cat([ex_store[(kind, l)], newx[l][:, offs:offs + E]])
                    n = t + nb
                    for l in range(L):
                        q = newx[l][:, :E]
                        Kc = get_rows(past, l, 'keys', n)
                        att = head_attention(q, Kc, t)                      # (H, nb, n), what the run used
                        abar = att.mean(0)
                        age = (torch.arange(t, n)[:, None] - torch.arange(n)[None, :]).to(D64)
                        valid = age >= 0
                        col = (torch.arange(n) >= 1)[None, :]
                        mass = acc.setdefault(('mass', l), np.zeros(4))
                        mass += [float(abar[:, 0].sum()), float(abar[valid & col & (age < 16)].sum()),
                                 float(abar[col & (age >= 16) & (age < 32)].sum()), float(abar[col & (age >= 32)].sum())]
                        if st_ is None:
                            continue
                        if diag_age:
                            for kind, R, attr, offs in kinds:
                                Xc = get_rows(past, l, attr, n)
                                Xe = ex_store[(kind, l)][:n]
                                r = ((Xc - Xe) @ st_[(kind, l)]['T'][0]).pow(2).sum(1); r[0] = 0
                                contrib = abar * r[None, :]
                                y = acc.setdefault(('age', kind, l), np.zeros(2))
                                y += [float(contrib[valid & (age < AGE)].sum()), float(contrib[age >= AGE].sum())]
                        if diag_posd:
                            Ke = ex_store[('K', l)][:n]
                            a_ = head_attention(q, Ke, t)                   # exact-key attention
                            qh = q.reshape(nb, H, Dh).transpose(0, 1)
                            dk = (Kc - Ke).reshape(n, H, Dh).transpose(0, 1)
                            delta = (qh @ dk.transpose(1, 2)) / math.sqrt(Dh)
                            delta = delta.masked_fill(~valid[None], 0.0)
                            dd = delta - (a_ * delta).sum(-1, keepdim=True)
                            nk = valid.sum(1).to(D64)[None, :, None]
                            low = (a_ < 1.0 / nk) & (dd > 0) & valid[None]
                            low[..., P:] = False                           # prompt tokens only
                            low[..., 0] = False
                            tv = (att - a_).abs()
                            y = acc.setdefault(('posd', l), np.zeros(2))
                            y += [float(tv[low].sum()), float(tv.sum())]
            else:
                on_block = None
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B, on_block=on_block)
            elif m == 'static / no buffer':
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=False, upd=False, on_block=on_block)
            elif m == 'static / block 32':
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=False, on_block=on_block)
            elif m == 'update / no buffer':
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=False, upd=True, on_block=on_block)
            elif m == 'update / block 32':
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=True, on_block=on_block)
            elif m == 'attn-late / no buffer':
                v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=False, upd=False, on_block=on_block)
            elif m == 'static / B=1':
                v, _ = run_tokens(ctx, w, P, kinds, st, window=1)
            elif m == 'static / sliding 32':
                v, _ = run_tokens(ctx, w, P, kinds, st, window=32)
            elif m == 'static / random 31 + own':
                v, _ = run_tokens(ctx, w, P, kinds, st, window=1, exact_set=rnd)
            elif m == 'static / top-attended 31 + own':
                v, _ = run_tokens(ctx, w, P, kinds, st, window=1, exact_set=top)
            else:
                raise ValueError(m)
            nllw[m].append(v)
            for key, val in acc.items():
                if key[0] == 'mass':
                    res['attn_mass'].append({'method': m, 'window': wi, 'layer': key[1],
                                             'pos0': val[0] / val.sum(), 'age0_15': val[1] / val.sum(),
                                             'age16_31': val[2] / val.sum(), 'age32plus': val[3] / val.sum()})
                elif key[0] == 'age':
                    res['age_diag'].append({'method': m, 'window': wi, 'kind': key[1], 'layer': key[2],
                                            'D_young': val[0], 'D_old': val[1], 'share_young': val[0] / max(val.sum(), 1e-300)})
                elif key[0] == 'posd':
                    res['posd_diag'].append({'method': m, 'window': wi, 'layer': key[1], 'share': val[0] / max(val[1], 1e-300)})
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / (1024 - P - 1)):.3f}' for m in methods), flush=True)

    tok = 1024 - P - 1
    ppl = ppl_table(nllw, tok)
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e})
    med = lambda rows: float(np.median(rows)) if rows else None
    summ = {}
    for m in AGE_DIAG:
        for kind in 'KV':
            summ[f'share_young {m} {kind}'] = med([x['share_young'] for x in res['age_diag'] if x['method'] == m and x['kind'] == kind])
    for m in ['dense'] + list(AGE_DIAG):
        for q in ('pos0', 'age0_15', 'age16_31', 'age32plus'):
            summ[f'attn_mass {m} {q}'] = med([x[q] for x in res['attn_mass'] if x['method'] == m])
    for m in POSD_DIAG:
        summ[f'posd_share {m}'] = med([x['share'] for x in res['posd_diag'] if x['method'] == m])
    res['summary'] = summ
    dec = {}
    try:
        dec['S1_own_token_removes_30pct'] = e['static / B=1'] <= 0.7 * e['static / no buffer']
        dec['S2_recent_beats_random'] = e['static / sliding 32'] <= 0.5 * e['static / random 31 + own']
        dec['S3_young_error_majority'] = (summ['share_young update / no buffer K'] >= 0.5 and summ['share_young update / no buffer V'] >= 0.5)
        dec['S4_sliding_no_worse_than_block'] = e['static / sliding 32'] <= e['static / block 32']
    except (KeyError, TypeError) as ex:
        dec['incomplete'] = f'missing {ex}'
    res['decision'] = dec
    if a.smoke:
        ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
        if os.path.exists(ref) and win_idx == [0]:
            kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
            res['code_checks_window0'] = {
                'static / no buffer': (ppl.get('static / no buffer'), kb['static / no buffer']),
                'static / block 32': (ppl.get('static / block 32'), kb['static / buffer B=32']),
                'update / no buffer': (ppl.get('update / no buffer'), kb['update / no buffer']),
                'update / block 32': (ppl.get('update / block 32'), kb['update / buffer B=32']),
                'dense': (ppl.get('dense'), kb['dense'])}
            print('code checks (this script, kv_buffer.py) on window 0:', res['code_checks_window0'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('ppl:', {k: round(v, 4) for k, v in ppl.items()})
    print('excess %:', {k: round(100 * v, 3) for k, v in e.items()})
    print('summary:', json.dumps(summ, indent=1))
    print('decision:', dec)
    try:
        make_figures(a, res)
    except Exception as ex:  # figures are optional for subset runs
        print('figure skipped:', ex)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import dataviz_style as dv
    dv.apply()
    os.makedirs(a.figdir, exist_ok=True)
    e = res['excess']
    order = [m for m in METHODS if m in e and m != 'dense']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.5, 1]})
    ax[0].barh(range(len(order)), [100 * e[m] for m in order], color=dv.C['blue'])
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=6); ax[0].invert_yaxis()
    ax[0].set_xlabel('perplexity increase (%)'); ax[0].set_title('(a) which exact tokens matter')
    s = res['summary']
    labs = [m for m in AGE_DIAG if s.get(f'share_young {m} K') is not None]
    x = np.arange(len(labs))
    ax[1].bar(x - 0.18, [s[f'share_young {m} K'] for m in labs], 0.36, label='keys')
    ax[1].bar(x + 0.18, [s[f'share_young {m} V'] for m in labs], 0.36, label='values')
    ax[1].axhline(0.5, color='gray', ls=':', lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([m.replace(' / ', '\n') for m in labs], fontsize=6)
    ax[1].set_ylabel('share of attention-read error\non tokens younger than 32'); ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title('(b) where the error is read')
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c2_kv_synergy.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
