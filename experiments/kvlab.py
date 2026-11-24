"""Shared helpers for the audit experiments C2-C9 (2026-11-22). Re-implements the kv_buffer.py / kv_update.py loop
operation by operation, so that its code checks reproduce the reported numbers (static / buffer B=32 = 24.997,
update / buffer B=32 = 24.778, static / no buffer = 25.775 on the 16 windows of the main setting).

Conventions (as in kv_buffer.py): one basis per layer for all 12 heads; keys query-aware (G_K = blockdiag of the prompt's
query second moments), values output-aware (G_V = W_O W_O^T); centered with the prompt mean; first token exact;
x_hat = mu + (x - mu) G^{1/2} U U^T G^{-1/2}; the prompt (P tokens) is prefilled exactly and then compressed; the
continuation runs in blocks of B tokens; perplexity on the continuation predictions only.
"""
import json, math, os, platform, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from kv_online import load_wikitext, psd_pow, qblock, rsvd_right  # noqa: E402,F401

D64 = torch.float64
L, H, E, Dh = 12, 12, 768, 64
SEQ = 1024


def top_right(A, R):
    _, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:R].T, s[:R]


def Pi(s):
    return s['T'][0] @ s['U'] @ s['U'].T @ s['T'][1]


def compress(X, s):
    return s['mu'] + (X - s['mu']) @ Pi(s)


class Ctx:
    """Model, data, hooks. mode['proj'] = {kind: [(mu, Pi) per layer]} compresses new K/V inside c_attn (no buffer)."""

    def __init__(self, threads=4, need_train=True, gv='full', attn_impl=None):
        torch.set_num_threads(threads)
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.tok = GPT2TokenizerFast.from_pretrained('gpt2')
        kw = {'attn_implementation': attn_impl} if attn_impl else {}
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', **kw).eval()
        self.ids_test = self.tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
        self.ids_train = self.tok(load_wikitext('train'), return_tensors='pt').input_ids[0] if need_train else None
        self.WO = [self.model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
        self.set_gv(gv)
        self.cap, self.mode = {}, {'proj': None}

        def hook(mod, args, out, l):
            self.cap[l] = out[0].detach()
            P_ = self.mode['proj']
            if P_ is None:
                if self.mode.get('on_hook') is not None:
                    self.mode['on_hook'](l, out, out)
                return out
            o = out.clone()
            for kind, offs in (('K', E), ('V', 2 * E)):
                if kind in P_:
                    mu, Pm = P_[kind][l]
                    o[..., offs:offs + E] = mu + (o[..., offs:offs + E] - mu) @ Pm
                    if kind in self.mode.get('keep0', ()):  # position 0 stored exactly (full-window passes only)
                        o[..., 0, offs:offs + E] = out[..., 0, offs:offs + E]
            if self.mode.get('on_hook') is not None:
                self.mode['on_hook'](l, out, o)
            return o
        self.hooks = [self.model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: hook(m, a_, o, l))
                      for l in range(L)]

    def set_gv(self, gv):
        """'full' = W_O W_O^T (the reported recipe); 'blockdiag' = only the per-head diagonal blocks (C8)."""
        self.gv = gv
        GV = []
        for l in range(L):
            G = self.WO[l] @ self.WO[l].T
            if gv == 'blockdiag':
                G = torch.block_diag(*[G[h * Dh:(h + 1) * Dh, h * Dh:(h + 1) * Dh] for h in range(H)])
            GV.append(G)
        self.TV = [(psd_pow(G, 0.5), psd_pow(G, -0.5)) for G in GV]

    def windows(self, n, offset=0, seq=SEQ):
        idx = list(range(offset, offset + n))
        return idx, [self.ids_test[i * seq:(i + 1) * seq] for i in idx]

    def offline(self, skip0=False, seq=SEQ):
        """Calibration bases as in kv_buffer.py (16 train windows, seed 0). Returns {(kind, l): (mu, T, U_full, lam)}."""
        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, self.ids_train.numel() - seq, (16,), generator=g).tolist()
        S1 = {k: [torch.zeros(E, dtype=D64) for _ in range(L)] for k in 'QKV'}
        S2 = {k: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for k in 'QKV'}
        sk = 1 if skip0 else 0
        self.mode['proj'] = None
        with torch.inference_mode():
            for s in starts:
                self.model(self.ids_train[s:s + seq][None])
                for l in range(L):
                    X = self.cap[l].to(D64)
                    for i, k in enumerate('QKV'):
                        Xi = X[sk:, i * E:(i + 1) * E]; S1[k][l] += Xi.sum(0); S2[k][l] += Xi.T @ Xi
        n = 16 * (seq - sk)
        off = {}
        for l in range(L):
            muK, muV = S1['K'][l] / n, S1['V'][l] / n
            CK = S2['K'][l] / n - torch.outer(muK, muK); CV = S2['V'][l] / n - torch.outer(muV, muV)
            GK = qblock(S2['Q'][l] / n); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
            wK, UK = torch.linalg.eigh(TK[0] @ CK @ TK[0])
            wV, UV = torch.linalg.eigh(self.TV[l][0] @ CV @ self.TV[l][0])
            off[('K', l)] = (muK, TK, UK.flip(1), wK.flip(0).clamp_min(0))
            off[('V', l)] = (muV, self.TV[l], UV.flip(1), wV.flip(0).clamp_min(0))
        self.calib = {'S1': S1, 'S2': S2, 'n': n}
        return off

    def prefill_stats(self, w, P):
        self.mode['proj'] = None
        with torch.inference_mode():
            self.model(w[:P][None])
        return {l: self.cap[l].to(D64) for l in range(L)}


def kinds_of(RK, RV):
    return [('K', RK, 'keys', E)] + ([('V', RV, 'values', 2 * E)] if RV else [])


def prompt_state(ctx, pre, kinds, row_weights=None, rank_cap=None, rows=None):
    """Online prompt basis exactly as kv_buffer.py: G_K from the prompt queries (positions 1..P-1), mean of the prompt rows,
    top-R right singular vectors of the weighted centered rows. row_weights[l]: optional (P-1,) weights (rows scaled by
    sqrt(w)); rank_cap: keep at most rank_cap directions; rows: slice of prompt positions to fit on (default 1..P-1)."""
    st = {}
    for kind, R, _, offs in kinds:
        for l in range(L):
            sl = rows if rows is not None else slice(1, None)
            Xq = pre[l][sl, :E]
            Xr = pre[l][sl, offs:offs + E]
            if kind == 'K':
                GK = qblock(Xq.T @ Xq / Xq.shape[0]); T = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
            else:
                T = ctx.TV[l]
            mu = Xr.mean(0)
            A = (Xr - mu) @ T[0]
            if row_weights is not None:
                A = row_weights[l].sqrt()[:, None] * A
            Rr = R if rank_cap is None else min(R, rank_cap)
            U, S = top_right(A, Rr)
            st[(kind, l)] = {'mu': mu, 'T': T, 'U': U, 'S': S}
    return st


def offline_state(off, kinds, n_rows_equiv=None):
    """Offline basis; S (for an update) = sqrt(lam * n_rows_equiv), i.e. the calibration covariance with the weight of
    n_rows_equiv rows (C3/C4 'offline + update' use n_rows_equiv = P - 1, the weight of the prompt)."""
    st = {}
    for kind, R, _, _ in kinds:
        for l in range(L):
            mu, T, U, lam = off[(kind, l)]
            S = (lam[:R] * n_rows_equiv).sqrt() if n_rows_equiv else None
            st[(kind, l)] = {'mu': mu, 'T': T, 'U': U[:, :R].clone(), 'S': S}
    return st


def hook_proj(st, kinds):
    return {kind: [(st[(kind, l)]['mu'].float(), Pi(st[(kind, l)]).float()) for l in range(L)] for kind, _, _, _ in kinds}


def get_rows(past, l, attr, n):
    return getattr(past.layers[l], attr)[0].transpose(0, 1).reshape(n, E).to(D64)


def set_rows(past, l, attr, X, n):
    setattr(past.layers[l], attr, X.float().reshape(n, H, Dh).transpose(0, 1)[None].contiguous())


def nll_of(logits, w, P):
    lp = torch.log_softmax(logits[:-1].double(), -1)
    tgt = w[P + 1:]
    return float(-lp[torch.arange(tgt.numel()), tgt].sum())


def run_block(ctx, w, P, kinds, st, B=32, buf=True, upd=False, compress_prompt=True, on_block=None, update_fn=None,
              block_compressor=None, prompt_compressor=None, post_reproj=None, seq=None, keep_logits=False):
    """kv_buffer.py loop. st None = dense. Returns (nll, logits or None).
    on_block(t, nb, past, newx, st): called after each block's forward, before the block is compressed.
    update_fn(s, Ablk, R) -> (U, S): replaces the incremental SVD (default = kv_buffer's top_right of [S U^T ; A_blk]).
    block_compressor(kind, l, Xnew, s) -> rows: replaces compress(Xnew, s) for the block rows (e.g. int8 codes).
    prompt_compressor(kind, l, X, s) -> rows: replaces compress(X, s) for the prompt rows.
    post_reproj(kind, l, Xold, s) -> rows: applied to the re-projected old rows 1..t-1 after an update (e.g. re-quantize)."""
    seq = seq or w.numel()
    m = ctx.model
    with torch.inference_mode():
        ctx.mode['proj'] = None
        out = m(w[:P][None], use_cache=True)
        past = out.past_key_values
        if st is not None and compress_prompt:
            for kind, R, attr, _ in kinds:
                for l in range(L):
                    X = get_rows(past, l, attr, P)
                    s = st[(kind, l)]
                    Xh = prompt_compressor(kind, l, X, s) if prompt_compressor else s['mu'] + (X - s['mu']) @ Pi(s)
                    Xh[0] = X[0]
                    set_rows(past, l, attr, Xh, P)
        hp = hook_proj(st, kinds) if (st is not None and not buf) else None
        logits = []
        t = P
        while t < seq:
            blk = w[t:t + B]; nb = blk.numel()
            ctx.mode['proj'] = hp
            out = m(blk[None], past_key_values=past)
            ctx.mode['proj'] = None
            logits.append(out.logits[0]); past = out.past_key_values
            newx = {l: ctx.cap[l].to(D64) for l in range(L)}
            if on_block is not None:
                on_block(t, nb, past, newx, st)
            if st is not None and t + nb < seq and (buf or upd):
                for kind, R, attr, offs in kinds:
                    for l in range(L):
                        X = get_rows(past, l, attr, t + nb)
                        s = st[(kind, l)]
                        Xnew = newx[l][:, offs:offs + E]
                        if upd:
                            Ablk = (Xnew - s['mu']) @ s['T'][0]
                            if update_fn is None:
                                M = torch.cat([s['S'][:, None] * s['U'].T, Ablk])
                                s['U'], s['S'] = top_right(M, R)
                            else:
                                s['U'], s['S'] = update_fn(s, Ablk, R)
                            Pn = Pi(s)
                            X[1:t] = s['mu'] + (X[1:t] - s['mu']) @ Pn
                            if post_reproj is not None:
                                X[1:t] = post_reproj(kind, l, X[1:t], s)
                        else:
                            Pn = Pi(s) if block_compressor is None else None
                        if block_compressor is not None:
                            X[t:] = block_compressor(kind, l, Xnew, s)
                        else:
                            X[t:] = s['mu'] + (Xnew - s['mu']) @ Pn
                        set_rows(past, l, attr, X, t + nb)
                if upd and not buf:
                    hp = hook_proj(st, kinds)
            t += nb
        lg = torch.cat(logits)
    return nll_of(lg, w, P), (lg if keep_logits else None)


def run_tokens(ctx, w, P, kinds, st, window=1, exact_set=(), seq=None, on_step=None, keep_logits=False):
    """Token-by-token continuation with a SLIDING exact window: when token t is processed, tokens t-window+1 .. t are
    exact (window = 1: only the token's own K/V, i.e. it is compressed right after its own step). Tokens in exact_set
    (and position 0) are never compressed. The prompt is compressed after prefill except its last window-1 tokens.
    Compression is done in place on the cache rows (float64 arithmetic as in run_block)."""
    seq = seq or w.numel()
    keep = set(exact_set) | {0}
    m = ctx.model
    Pm = {(kind, l): Pi(st[(kind, l)]) for kind, _, _, _ in kinds for l in range(L)}

    def comp_row(past, j):
        for kind, R, attr, _ in kinds:
            for l in range(L):
                ten = getattr(past.layers[l], attr)
                x = ten[0, :, j, :].reshape(E).to(D64)
                s = st[(kind, l)]
                xh = s['mu'] + (x - s['mu']) @ Pm[(kind, l)]
                ten[0, :, j, :] = xh.float().reshape(H, Dh)

    with torch.inference_mode():
        ctx.mode['proj'] = None
        out = m(w[:P][None], use_cache=True)
        past = out.past_key_values
        first_exact = P - window + 1  # prompt tokens >= this stay exact at the start (inside the window of query P)
        for kind, R, attr, _ in kinds:
            for l in range(L):
                X = get_rows(past, l, attr, P)
                s = st[(kind, l)]
                Xh = s['mu'] + (X - s['mu']) @ Pm[(kind, l)]
                for j in keep:
                    if j < P:
                        Xh[j] = X[j]
                if first_exact < P:
                    Xh[max(first_exact, 0):] = X[max(first_exact, 0):]
                set_rows(past, l, attr, Xh, P)
        logits = []
        for t in range(P, seq):
            out = m(w[t:t + 1][None], past_key_values=past)
            logits.append(out.logits[0]); past = out.past_key_values
            if on_step is not None:
                on_step(t, past)
            j = t - window + 1  # leaves the window before query t+1
            if j >= 1 and j not in keep and t + 1 < seq:
                comp_row(past, j)
        lg = torch.cat(logits)
    return nll_of(lg, w, P), (lg if keep_logits else None)


def head_attention(q, K, q_pos0):
    """Causal attention probabilities (H, nq, nk), float64. q (nq, E) queries at positions q_pos0.., K (nk, E)."""
    nq, nk = q.shape[0], K.shape[0]
    qh = q.reshape(nq, H, Dh).transpose(0, 1)
    kh = K.reshape(nk, H, Dh).transpose(0, 1)
    s = qh @ kh.transpose(1, 2) / math.sqrt(Dh)
    pos_q = torch.arange(q_pos0, q_pos0 + nq)[:, None]
    s = s.masked_fill(torch.arange(nk)[None, :] > pos_q, float('-inf'))
    return torch.softmax(s, -1)


def base_result(a, extra=None):
    r = {'config': vars(a), 'torch': torch.__version__,
         'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'threads': a.threads}}
    if extra:
        r.update(extra)
    return r


def ppl_table(nllw, tok_per_window):
    """{method: per-window NLL list} -> {method: ppl}."""
    return {m: math.exp(sum(v) / (tok_per_window * len(v))) for m, v in nllw.items() if v}


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return None
