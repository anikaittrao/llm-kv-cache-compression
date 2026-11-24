"""C11 (audit 2026-11-22): port of the buffer/update 2x2 and the attention-weighting test to a small RoPE + GQA model.

Run:  .venv\\Scripts\\python.exe experiments\\kv_rope_port.py --out experiments\\rerun_2026-11-22\\c11_rope_smollm2.json \\
            --figdir experiments\\rerun_2026-11-22\\fig
      (SmolLM2-135M, 8 windows of P = 2048 + 1024 continuation; see 'Runtime' below)
      smoke test (short window): --windows 1 --P 512 --seq 768
      --replot redraws the figure from the JSON.

Model: HuggingFaceTB/SmolLM2-135M (Llama architecture: 30 layers, 9 query heads, 3 key/value heads (GQA, 3 query heads
per KV head), head dim 64, RoPE theta 1e5, context 8192), float32 on the CPU. Data: WikiText-2 raw test text with the
model's own tokenizer, consecutive windows of --seq tokens.
Recipe (the GPT-2 recipe of kv_buffer.py carried over):
  * one basis per layer shared by the 3 KV heads: keys and values are 192-dim rows per token; half the stored codes =
    K 64 / V 128 (the 1/3 : 2/3 split of the GPT-2 main setting); first token exact; centered with the prompt mean.
  * PRE-RoPE keys. The basis acts on the keys before the rotary embedding (rows from k_proj), because RoPE rotates
    every position differently and would destroy a shared low-rank structure. Cached keys are post-RoPE, so a stored
    row is un-rotated (inverse rotation at its position), projected, and re-rotated (un-rotate / project / re-rotate).
    New keys compressed before attention ('no buffer') are compressed inside k_proj, i.e. before RoPE.
  * key weight G_K = blockdiag over KV heads g of sum_{h in g} E[q_h q_h^T] (pre-RoPE prompt queries; with RoPE this
    weight is an approximation: the score error is q_t R_{t-s} (k_s - k^_s)); --kweight plain uses G_K = I.
    value weight G_V = blockdiag over KV heads g of sum_{h in g} W_O^h W_O^h^T (o_proj columns of the query heads that
    share the KV head); x_hat = mu + (x - mu) G^{1/2} U U^T G^{-1/2}.
  * blocks of B = 32 continuation tokens; 'buffer' = a block's own K/V exact while it is processed (kv_buffer.py);
    'update' = incremental SVD [S U^T ; A_blk] -> top R at every block end, old rows re-projected.
  * attn-late (kv_attnbasis.py): prompt rows scaled by sqrt(attention received from the last 128 prompt queries,
    post-RoPE, mean over the 9 query heads, / mean).
Methods: dense; static / no buffer; static / buffer 32; update / no buffer; update / buffer 32; attn-late / no buffer;
attn-late / buffer 32. No offline (calibration) basis in this port.
Sanity checks stored: blockwise dense = one pass (max logit diff); un-rotating the cached keys recovers the k_proj rows
(max abs diff).
Rules fixed before the run (the C1 rules carried over; e(x) = ppl(x)/dense - 1, all windows):
  Q1r both small     : e(update / buffer 32) <= 0.005.
  Q2r order          : e(update / buffer 32) < e(static / buffer 32) < e(update / no buffer) < e(static / no buffer).
  Q3r both vs buffer : the paired 95% bootstrap CI (experiments/stats_utils.py) of ppl(update / buffer 32) /
                       ppl(static / buffer 32) lies below 1.
  W1r                : e(attn-late / no buffer) <= 0.9 x e(static / no buffer).
  W3r                : e(attn-late / buffer 32) <= 0.9 x e(static / buffer 32).
Runtime / memory (measured 2026-11-22, 4 threads, idle CPU): one full-size window (P = 2048, 1024 continuation, all 7
methods) takes 228 s (dense 28 s, update methods ~45 s, others 20-28 s) -> 8 windows ~31 min, 16 windows ~62 min, plus
~15 s to load. Peak RSS 3.0 GB (model 0.55 GB fp32; the one-pass sanity check of 3072 tokens and the 49152-word logits
dominate). Smoke test (--windows 1 --P 512 --seq 768): 60 s, 1.6 GB. Download of SmolLM2-135M: ~260 MB (cached in
~/.cache/huggingface). Qwen2.5-0.5B (--model Qwen/Qwen2.5-0.5B, 24 layers, 14 query / 2 KV heads, 954 MB download)
would take roughly 4-5x longer and ~5-6 GB RSS; not downloaded.
Follow-up C13 (2026-11-23, run after C11 and its write-up): two more methods, 'attn-late+update / no buffer' and
'attn-late+update / buffer 32': the attn-late weighted prompt basis (S, U from sqrt-weighted prompt rows) combined with the
same block update [S U^T ; A_blk] (the new block rows are NOT weighted, a simplification: the weighted S mixes with
unweighted rows). Plus the C11 setting on 8 more windows (--window_offset 8) so that all 7 old methods have 16 windows.
Rules fixed before the C13 run (16 windows 0-15 pooled from c11 + c13 files, e(x) = ppl(x)/dense - 1):
  A1r update on top of weighting : e(attn-late+update / buffer 32) <= e(attn-late / buffer 32).
  A2r beats the plain update     : e(attn-late+update / buffer 32) <= 0.9 x e(update / buffer 32).
  Q1r-Q3r, W1r, W3r are re-evaluated on the 16 pooled windows (same definitions).
C21 (wave 4, 2026-11-24): '--scope head' = one basis PER KV HEAD (3 bases of 64 dims per layer, block-diagonal U, update done per
head) instead of one basis per layer. Equal stored numbers per token: per layer K64/V128 (192) vs per head K21/V43 (3 x 64 = 192;
run with --RK 63 --RV 129 --scope head). Windows 0-7, default P = 2048, methods 'static / no buffer', 'static / buffer 32',
'update / buffer 32' (--only static update). Rules fixed before the run (layer-scope numbers = c11_rope_smollm2.json, same windows,
same code path with --scope layer; e(x) = ppl(x)/dense - 1; ppl ratio = ppl(head) / ppl(layer), paired CI by stats_utils):
  PH1 per-layer basis better, static / buffer 32 : the CI of ppl(head)/ppl(layer) lies above 1.
  PH2 same for update / buffer 32.
  PH3 same for static / no buffer.
  (Per-head would 'win' a cell if the CI lies below 1; otherwise 'not resolved'.) Basis memory per layer is reported by the
  analysis: layer 192 x (64 + 128) numbers vs head 3 x 64 x (21 + 43).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_online import load_wikitext, psd_pow  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64
LATE = 128
METHODS = ['dense', 'static / no buffer', 'static / buffer 32', 'update / no buffer', 'update / buffer 32',
           'attn-late / no buffer', 'attn-late / buffer 32', 'attn-late+update / no buffer',
           'attn-late+update / buffer 32']


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return None


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Port:
    def __init__(self, a):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(a.threads)
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        c = self.model.config
        self.L, self.Hq, self.KVH = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.Dh = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
        self.rep = self.Hq // self.KVH
        self.dK = self.KVH * self.Dh
        self.layers = self.model.model.layers
        self.cap, self.mode = {}, {'proj': None}
        self.hooks = []
        for l, lay in enumerate(self.layers):
            at = lay.self_attn
            self.hooks.append(at.q_proj.register_forward_hook(lambda m, i, o, l=l: self._cap(l, 'q', o)))
            self.hooks.append(at.k_proj.register_forward_hook(lambda m, i, o, l=l: self._hook(l, 'K', o)))
            self.hooks.append(at.v_proj.register_forward_hook(lambda m, i, o, l=l: self._hook(l, 'V', o)))
        # output-aware value weight per layer (block diagonal over KV heads)
        self.TV = []
        for lay in self.layers:
            W = lay.self_attn.o_proj.weight.detach().to(D64)      # (hidden out, Hq*Dh in)
            blocks = []
            for g in range(self.KVH):
                G = torch.zeros(self.Dh, self.Dh, dtype=D64)
                for h in range(g * self.rep, (g + 1) * self.rep):
                    WOh = W[:, h * self.Dh:(h + 1) * self.Dh].T     # (Dh, hidden)
                    G += WOh @ WOh.T
                blocks.append(G)
            G = torch.block_diag(*blocks)
            self.TV.append((psd_pow(G, 0.5), psd_pow(G, -0.5)))

    def _cap(self, l, kind, o):
        self.cap[(l, kind)] = o[0].detach()
        return o

    def _hook(self, l, kind, o):
        self.cap[(l, kind)] = o[0].detach()
        P_ = self.mode['proj']
        if P_ is None or kind not in P_:
            return o
        mu, Pm = P_[kind][l]
        return mu + (o - mu) @ Pm

    def cos_sin(self, n):
        pos = torch.arange(n)[None]
        dummy = torch.zeros(1, n, self.Dh)
        cos, sin = self.model.model.rotary_emb(dummy, pos)
        return cos[0].to(D64), sin[0].to(D64)                       # (n, Dh)

    def rot(self, X, cs, inverse=False):
        cos, sin = cs
        Xh = X.reshape(X.shape[0], self.KVH, self.Dh)
        s = -sin if inverse else sin
        out = Xh * cos[:, None, :] + rotate_half(Xh) * s[:, None, :]
        return out.reshape(X.shape[0], self.dK)

    def get(self, past, l, attr, n):
        return getattr(past.layers[l], attr)[0].transpose(0, 1).reshape(n, self.dK).to(D64)

    def put(self, past, l, attr, X, n):
        setattr(past.layers[l], attr, X.float().reshape(n, self.KVH, self.Dh).transpose(0, 1)[None].contiguous())


def Pi(s):
    return s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1]


def top_right(A, R):
    _, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:R].T, s[:R]


def fit_basis(A, R, groups):
    """groups = None: one basis for all columns (the reported recipe, rank R). Otherwise a list of (slice, rank): one basis per
    column group (C21 per-KV-head scope), block diagonal U of shape (cols, sum of ranks), S concatenated."""
    if groups is None:
        return top_right(A, R)
    U = torch.zeros(A.shape[1], sum(r for _, r in groups), dtype=A.dtype); S = []; c = 0
    for sl, r in groups:
        Ug, Sg = top_right(A[:, sl], r)
        U[sl, c:c + r] = Ug; S.append(Sg); c += r
    return U, torch.cat(S)


def update_basis(s, Ablk, R, groups):
    """Incremental SVD step [S U^T ; A_blk] (reported recipe), per column group for the per-head scope."""
    if groups is None:
        return top_right(torch.cat([s['S'][:, None] * s['U'].T, Ablk]), R)
    U = torch.zeros_like(s['U']); S = []; c = 0
    for sl, r in groups:
        M = torch.cat([s['S'][c:c + r, None] * s['U'][sl, c:c + r].T, Ablk[:, sl]])
        Ug, Sg = top_right(M, r)
        U[sl, c:c + r] = Ug; S.append(Sg); c += r
    return U, torch.cat(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--P', type=int, default=2048)
    ap.add_argument('--seq', type=int, default=3072)
    ap.add_argument('--RK', type=int, default=64)
    ap.add_argument('--RV', type=int, default=128)
    ap.add_argument('--B', type=int, default=32)
    ap.add_argument('--kweight', choices=['query', 'plain'], default='query')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--scope', choices=['layer', 'head'], default='layer')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c11_rope_smollm2.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    pt = Port(a)
    print(f'model loaded ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB): L={pt.L} Hq={pt.Hq} KVH={pt.KVH} Dh={pt.Dh}', flush=True)
    ids = pt.tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    seq, P, B, L_ = a.seq, a.P, a.B, pt.L
    win_idx = list(range(a.window_offset, a.window_offset + a.windows))
    assert (win_idx[-1] + 1) * seq <= ids.numel(), 'not enough test tokens'
    wins = [ids[i * seq:(i + 1) * seq] for i in win_idx]
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    kinds = [('K', a.RK, 'keys'), ('V', a.RV, 'values')]
    # C21: --scope head = one basis per KV head, rank R // KVH each (use --RK 63 --RV 129 for the same 192 stored numbers as K64/V128)
    grp = {k: None for k in 'KV'}
    if a.scope == 'head':
        grp = {k: [(slice(g_ * pt.Dh, (g_ + 1) * pt.Dh), R_ // pt.KVH) for g_ in range(pt.KVH)] for k, R_ in (('K', a.RK), ('V', a.RV))}
    cs_all = pt.cos_sin(seq)
    res = {'config': vars(a), 'torch': torch.__version__, 'model_dims': {'L': pt.L, 'Hq': pt.Hq, 'KVH': pt.KVH, 'Dh': pt.Dh},
           'windows': win_idx, 'tokens_per_window': seq - P - 1, 'cache_frac_codes': (a.RK + a.RV) / (2 * pt.dK),
           'sanity': {}, 'timing_s': {}}
    nllw = {m: [] for m in methods}
    m_ = pt.model

    def nll_of(lg, w):
        lp = torch.log_softmax(lg[:-1].double(), -1)
        tgt = w[P + 1:]
        return float(-lp[torch.arange(tgt.numel()), tgt].sum())

    for wi, w in zip(win_idx, wins):
        # prompt statistics (exact prefill)
        pt.mode['proj'] = None
        with torch.inference_mode():
            m_(w[:P][None])
        pre = {(l, k): pt.cap[(l, k)].to(D64) for l in range(L_) for k in ('q', 'K', 'V')}
        # attn-late weights (post-RoPE scores of the last LATE prompt queries)
        cs_p = (cs_all[0][:P], cs_all[1][:P])
        wlate = {}
        for l in range(L_):
            q = pre[(l, 'q')].reshape(P, pt.Hq, pt.Dh)
            qr = q * cs_p[0][:, None, :] + rotate_half(q) * cs_p[1][:, None, :]
            kr = pt.rot(pre[(l, 'K')], cs_p).reshape(P, pt.KVH, pt.Dh)
            c = torch.zeros(P, dtype=D64)
            for h in range(pt.Hq):
                g = h // pt.rep
                s = (qr[P - LATE:, h] @ kr[:, g].T) / math.sqrt(pt.Dh)
                s = s.masked_fill(torch.arange(P)[None, :] > torch.arange(P - LATE, P)[:, None], float('-inf'))
                c += torch.softmax(s, -1).sum(0) / pt.Hq
            c = c[1:]
            wlate[l] = c / c.mean()

        def state(m):
            st = {}
            for kind, R, _ in kinds:
                for l in range(L_):
                    Xq = pre[(l, 'q')][1:]
                    Xr = pre[(l, kind)][1:]
                    if kind == 'K':
                        if a.kweight == 'plain':
                            T = (torch.eye(pt.dK, dtype=D64), torch.eye(pt.dK, dtype=D64))
                        else:
                            Qh = Xq.reshape(-1, pt.Hq, pt.Dh)
                            blocks = [sum(Qh[:, h].T @ Qh[:, h] for h in range(g * pt.rep, (g + 1) * pt.rep)) / Xq.shape[0]
                                      for g in range(pt.KVH)]
                            G = torch.block_diag(*blocks); T = (psd_pow(G, 0.5), psd_pow(G, -0.5))
                    else:
                        T = pt.TV[l]
                    mu = Xr.mean(0)
                    A = (Xr - mu) @ T[0]
                    if m.startswith('attn-late'):
                        A = wlate[l].sqrt()[:, None] * A
                    U, S = fit_basis(A, R, grp[kind])
                    st[(kind, l)] = {'mu': mu, 'T': T, 'U': U, 'S': S}
            return st

        def hook_proj(st):
            return {kind: [(st[(kind, l)]['mu'].float(), Pi(st[(kind, l)]).float()) for l in range(L_)] for kind, _, _ in kinds}

        for m in methods:
            tm = time.perf_counter()
            st = None if m == 'dense' else state(m)
            buf = m.endswith('buffer 32') and 'no buffer' not in m
            upd = m.startswith('update') or m.startswith('attn-late+update')
            with torch.inference_mode():
                pt.mode['proj'] = None
                out = m_(w[:P][None], use_cache=True)
                past = out.past_key_values
                if st is not None:
                    for kind, R, attr in kinds:
                        for l in range(L_):
                            X = pt.get(past, l, attr, P)
                            if kind == 'K':
                                X = pt.rot(X, cs_p, inverse=True)
                                if l == 0 and wi == win_idx[0] and 'unrotate_max_abs_diff' not in res['sanity']:
                                    res['sanity']['unrotate_max_abs_diff'] = float((X - pre[(0, 'K')]).abs().max())
                            s = st[(kind, l)]
                            Xh = s['mu'] + (X - s['mu']) @ Pi(s); Xh[0] = X[0]
                            pt.put(past, l, attr, pt.rot(Xh, cs_p) if kind == 'K' else Xh, P)
                hp = hook_proj(st) if (st is not None and not buf) else None
                logits = []
                t = P
                while t < seq:
                    blk = w[t:t + B]; nb = blk.numel()
                    pt.mode['proj'] = hp
                    out = m_(blk[None], past_key_values=past)
                    pt.mode['proj'] = None
                    logits.append(out.logits[0]); past = out.past_key_values
                    if st is not None and t + nb < seq and (buf or upd):
                        cs_t = (cs_all[0][:t + nb], cs_all[1][:t + nb])
                        for kind, R, attr in kinds:
                            for l in range(L_):
                                X = pt.get(past, l, attr, t + nb)
                                if kind == 'K':
                                    X = pt.rot(X, cs_t, inverse=True)
                                s = st[(kind, l)]
                                Xnew = pt.cap[(l, kind)].to(D64)               # exact pre-RoPE rows of the block
                                if upd:
                                    Ablk = (Xnew - s['mu']) @ s['T'][0]
                                    s['U'], s['S'] = update_basis(s, Ablk, R, grp[kind])
                                    Pn = Pi(s)
                                    X[1:t] = s['mu'] + (X[1:t] - s['mu']) @ Pn
                                else:
                                    Pn = Pi(s)
                                X[t:] = s['mu'] + (Xnew - s['mu']) @ Pn
                                pt.put(past, l, attr, pt.rot(X, cs_t) if kind == 'K' else X, t + nb)
                        if upd and not buf:
                            hp = hook_proj(st)
                    t += nb
                lg = torch.cat(logits)
                if m == 'dense' and 'blockwise_vs_one_pass_max_logit_diff' not in res['sanity']:
                    full = m_(w[None]).logits[0, P:]
                    res['sanity']['blockwise_vs_one_pass_max_logit_diff'] = float((full - lg).abs().max())
            nllw[m].append(nll_of(lg, w))
            res['timing_s'].setdefault(m, []).append(time.perf_counter() - tm)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / (seq - P - 1)):.3f}' for m in methods), flush=True)

    tok = seq - P - 1
    ppl = {m: math.exp(sum(v) / (tok * len(v))) for m, v in nllw.items()}
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e})
    d = {}
    try:
        d['Q1r_both_le_0.5pct'] = e['update / buffer 32'] <= 0.005
        d['Q2r_order'] = e['update / buffer 32'] < e['static / buffer 32'] < e['update / no buffer'] < e['static / no buffer']
        from stats_utils import ppl_ratio
        ci = ppl_ratio(nllw['update / buffer 32'], nllw['static / buffer 32'], tok)
        d['Q3r_both_over_buffer_CI'] = ci; d['Q3r_CI_below_1'] = ci['ci_hi'] < 1
        d['W1r'] = e['attn-late / no buffer'] <= 0.9 * e['static / no buffer']
        d['W3r'] = e['attn-late / buffer 32'] <= 0.9 * e['static / buffer 32']
    except KeyError as ex:
        d['incomplete'] = f'missing {ex}'
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('sanity:', res['sanity'])
    print('ppl:', {k: round(v, 4) for k, v in ppl.items()})
    print('excess %:', {k: round(100 * v, 3) for k, v in e.items()})
    print('seconds per window and method:', {k: round(float(np.mean(v)), 1) for k, v in res['timing_s'].items()})
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
    e = res['excess']
    ms = [m for m in METHODS if m in e and m != 'dense']
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.barh(range(len(ms)), [100 * e[m] for m in ms], color=dv.C['blue'])
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('perplexity increase (%)')
    c = res['config']
    ax.set_title(f"{c['model'].split('/')[-1]}, P = {c['P']}, K {c['RK']} / V {c['RV']} (pre-RoPE)", fontsize=8)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c11_rope_port.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
