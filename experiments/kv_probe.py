"""C6 (audit 2026-11-22): beyond teacher-forced perplexity: a copy (induction) probe and greedy-generation drift.

Run:  .venv\\Scripts\\python.exe experiments\\kv_probe.py --out experiments\\rerun_2026-11-22\\c6_kv_probe.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 75 min: calibration 75 s + ~270 s per window, smoke 2026-11-22)
      smoke test: --windows 1      --replot: redraw the figure from the JSON.

Setting: kv_buffer.py recipe, half the codes (K 256 / V 512), prompt = the first P = 768 tokens of each test window
(prefilled exactly, then compressed; first token exact), blocks of 32.
Methods: dense; offline / no buffer; offline / buffer 32; static / no buffer; static / buffer 32; update / buffer 32.
1. Copy probe. Continuation = r + r, with r = 128 tokens drawn at random positions of the training text (seed 10000 +
   window). The first copy is compressed like any other new text (it is new, and random text is far from the prompt's
   basis); on the second copy the model can only predict well by retrieving the first copy (induction heads).
   Measured on the second copy, targets r[1..127] (127 per window): mean NLL and top-1 accuracy.
2. Greedy drift. From the same prompt, 128 tokens are generated greedily by the dense model and by each method (the
   method's own compressed cache, token by token, with the same block semantics: tokens of the current block exact in
   the buffer methods, the basis update every 32 generated tokens). first_divergence = index of the first generated
   token that differs from the dense one (128 if none); match_rate = share of equal tokens position by position.
   Teacher-forced on the dense greedy text: mean KL(p_dense || p_method) per step and top-1 agreement.
Rules fixed before the run (16 windows):
  I1 copy   : top1(update / buffer 32) >= top1(dense) - 0.02 (2 percentage points).
  I2 drift  : median over windows of first_divergence(update / buffer 32) >= 32 tokens.
Descriptive: the same numbers for the other methods.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, Pi, top_right, prompt_state, offline_state, kinds_of, run_block,  # noqa: E402
                   get_rows, set_rows, hook_proj, base_result, save_json, rss_mb)

P, RK, RV, B = 768, 256, 512, 32
NR, G = 128, 128
METHODS = ['dense', 'offline / no buffer', 'offline / buffer 32', 'static / no buffer', 'static / buffer 32',
           'update / buffer 32']


def spec(m):
    return {'basis': None if m == 'dense' else m.split(' / ')[0], 'buf': 'buffer' in m and 'no buffer' not in m,
            'upd': m.startswith('update')}


def generate(ctx, prompt, kinds, st, buf, upd, n_new=G):
    """Greedy generation with the compressed cache (block semantics of run_block). st None = dense."""
    m = ctx.model
    out_tokens = []
    with torch.inference_mode():
        ctx.mode['proj'] = None
        out = m(prompt[None], use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if st is not None:
            for kind, R, attr, _ in kinds:
                for l in range(L):
                    X = get_rows(past, l, attr, P); s = st[(kind, l)]
                    Xh = s['mu'] + (X - s['mu']) @ Pi(s); Xh[0] = X[0]; set_rows(past, l, attr, Xh, P)
        hp = hook_proj(st, kinds) if (st is not None and not buf) else None
        blk_rows = {l: [] for l in range(L)}
        t = P
        for i in range(n_new):
            out_tokens.append(nxt)
            ctx.mode['proj'] = hp
            out = m(torch.tensor([[nxt]]), past_key_values=past)
            ctx.mode['proj'] = None
            past = out.past_key_values
            for l in range(L):
                blk_rows[l].append(ctx.cap[l].to(D64)[0:1] if ctx.cap[l].dim() == 2 else ctx.cap[l].to(D64).reshape(1, -1))
            nxt = int(out.logits[0, -1].argmax())
            t += 1
            if st is not None and (t - P) % B == 0 and (buf or upd) and i + 1 < n_new:
                t0 = t - B
                for kind, R, attr, offs in kinds:
                    for l in range(L):
                        X = get_rows(past, l, attr, t)
                        s = st[(kind, l)]
                        Xnew = torch.cat(blk_rows[l])[:, offs:offs + E]
                        if upd:
                            Ablk = (Xnew - s['mu']) @ s['T'][0]
                            M = torch.cat([s['S'][:, None] * s['U'].T, Ablk])
                            s['U'], s['S'] = top_right(M, R)
                            Pn = Pi(s)
                            X[1:t0] = s['mu'] + (X[1:t0] - s['mu']) @ Pn
                        else:
                            Pn = Pi(s)
                        X[t0:] = s['mu'] + (Xnew - s['mu']) @ Pn
                        set_rows(past, l, attr, X, t)
                if upd and not buf:
                    hp = hook_proj(st, kinds)
                blk_rows = {l: [] for l in range(L)}
    return out_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c6_kv_probe.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    off = ctx.offline()
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'windows': win_idx, 'copy_len': NR, 'gen_len': G, 'rows': []})

    def state(m, pre):
        sp = spec(m)
        if sp['basis'] is None:
            return None
        return offline_state(off, kinds) if sp['basis'] == 'offline' else prompt_state(ctx, pre, kinds)

    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        g = torch.Generator().manual_seed(10000 + wi)
        r = ctx.ids_train[torch.randint(0, ctx.ids_train.numel(), (NR,), generator=g)]
        wc = torch.cat([w[:P], r, r])
        gens = {}
        for m in methods:
            sp = spec(m)
            # 1. copy probe
            _, lg = run_block(ctx, wc, P, kinds, state(m, pre), B=B, buf=sp['buf'], upd=sp['upd'], keep_logits=True)
            pos = torch.arange(NR + 0, 2 * NR - 1)          # logits rows predicting r2[1..127] (row k predicts wc[P+k+1])
            tgt = wc[P + pos + 1]
            lp = torch.log_softmax(lg[pos].double(), -1)
            copy_nll = float(-lp[torch.arange(tgt.numel()), tgt].mean())
            copy_top1 = float((lg[pos].argmax(-1) == tgt).double().mean())
            # 2. greedy drift
            gens[m] = generate(ctx, w[:P], kinds, state(m, pre), sp['buf'], sp['upd'])
            res['rows'].append({'window': wi, 'method': m, 'copy_nll': copy_nll, 'copy_top1': copy_top1})
        dg = gens['dense']
        wt = torch.cat([w[:P], torch.tensor(dg)])
        _, lgd = run_block(ctx, wt, P, kinds, None, B=B, keep_logits=True)
        pd = torch.log_softmax(lgd.double(), -1)
        for row in res['rows']:
            if row['window'] != wi:
                continue
            m = row['method']
            gm = gens[m]
            div = next((i for i, (x, y) in enumerate(zip(gm, dg)) if x != y), G)
            row['first_divergence'] = div
            row['match_rate'] = float(np.mean([x == y for x, y in zip(gm, dg)]))
            if m == 'dense':
                row['tf_kl'] = 0.0; row['tf_top1_agree'] = 1.0
                continue
            sp = spec(m)
            _, lgm = run_block(ctx, wt, P, kinds, state(m, pre), B=B, buf=sp['buf'], upd=sp['upd'], keep_logits=True)
            pm = torch.log_softmax(lgm.double(), -1)
            row['tf_kl'] = float((pd.exp() * (pd - pm)).sum(-1).mean())
            row['tf_top1_agree'] = float((lgm.argmax(-1) == lgd.argmax(-1)).double().mean())
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f"{x['method']}: top1 {x['copy_top1']:.2f} div {x['first_divergence']}" for x in res['rows'] if x['window'] == wi),
              flush=True)
    summ = {}
    for m in methods:
        rr = [x for x in res['rows'] if x['method'] == m]
        summ[m] = {'copy_top1_mean': float(np.mean([x['copy_top1'] for x in rr])),
                   'copy_nll_mean': float(np.mean([x['copy_nll'] for x in rr])),
                   'first_divergence_median': float(np.median([x['first_divergence'] for x in rr])),
                   'match_rate_mean': float(np.mean([x['match_rate'] for x in rr])),
                   'tf_kl_mean': float(np.mean([x['tf_kl'] for x in rr])),
                   'tf_top1_agree_mean': float(np.mean([x['tf_top1_agree'] for x in rr]))}
    res['summary'] = summ
    d = {}
    try:
        d['I1_copy_top1_within_2pt'] = summ['update / buffer 32']['copy_top1_mean'] >= summ['dense']['copy_top1_mean'] - 0.02
        d['I2_first_divergence_ge_32'] = summ['update / buffer 32']['first_divergence_median'] >= 32
    except KeyError as ex:
        d['incomplete'] = f'missing {ex}'
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print(json.dumps(summ, indent=1))
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
    s = res['summary']
    ms = list(s)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].bar(range(len(ms)), [100 * s[m]['copy_top1_mean'] for m in ms], color=dv.C['blue'])
    ax[0].set_ylabel('copy top-1 (%)'); ax[0].set_title('(a) second copy of a random sequence')
    ax[1].bar(range(len(ms)), [s[m]['first_divergence_median'] for m in ms], color=dv.C['orange'])
    ax[1].set_ylabel('first divergence (tokens, median)'); ax[1].set_title('(b) greedy generation vs dense')
    for x in ax:
        x.set_xticks(range(len(ms))); x.set_xticklabels([m.replace(' / ', '\n') for m in ms], fontsize=5.5)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c6_kv_probe.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
