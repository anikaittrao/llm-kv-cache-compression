"""C5 (audit 2026-11-22): equal-memory baselines for the low-rank KV cache (quantization, eviction, low-rank + int8).

Run:  .venv\\Scripts\\python.exe experiments\\kv_baselines.py --out experiments\\rerun_2026-11-22\\c5_kv_baselines.json \\
            --figdir experiments\\rerun_2026-11-22\\fig            (about 40 min: calibration 76 s + ~145 s per window, smoke 2026-11-22)
      smoke test (~10 min): --windows 1 --smoke       --replot: redraw the figure from the JSON.

Setting: kv_buffer.py main setting (P = 768 prompt prefilled exactly then compressed, 256 continuation tokens in blocks
of 32, perplexity on the continuation; first token exact for every method). Memory is the stored size at the end
(T = 1024 tokens) relative to an fp16 full cache, INCLUDING scales, the exact tokens (token 0 + the 32-token block
buffer) and, for per-sequence low-rank bases, the basis (experiments/kv_memory.py, Prop 2.2 of the audit plan).
Baselines (simulated, quantize-dequantize):
  int b (b = 8, 4, 2), KIVI-style: keys per channel over groups of 32 tokens (asymmetric min-max, an fp16 scale and
      zero per channel and group), values per token over each head's 64 channels; the 32-token block buffer stays fp16
      (as KIVI's residual window). Memory per number: keys b/8 + 4/32 bytes, values b/8 + 4/64 bytes (fp16 = 2).
  sink + window (StreamingLLM): keep tokens 0-3 and the most recent tokens; the rest are evicted (masked out).
  H2O-style: per head, keep token 0, the most recent half of the budget and the tokens with the largest accumulated
      attention (from every query so far, incl. the prompt queries) for the other half; evicted tokens are masked.
      Eviction budgets n_keep after each block with memory (n_keep + 32) / 1024 = f, f = 0.5, 0.75, 0.92.
      (Absolute positions are learned embeddings in GPT-2, so masking = evicting.)
Low-rank (this report's recipe, half the codes, K 256 / V 512): offline / buffer 32 (memory 0.516), static / buffer 32
  (0.924), update / buffer 32 (0.924; code checks 25.022 / 24.997 / 24.778 on the 16 windows), and low-rank + int8:
  the same with the codes quantized per token (int8, fp16 scale and zero per token and kind; re-projected old codes are
  re-quantized after every update).
Rules fixed before the run (16 windows), e(x) = ppl(x)/dense - 1:
  E1 int8 is near lossless  : e(int8) <= 0.003.
  E2 low-rank competitive   : e(update / buffer 32) <= min e(baseline) over the baselines whose memory at T = 1024 is
                              <= that of update / buffer 32 (0.924). Same test reported (not deciding) for offline /
                              buffer 32 (0.516) and low-rank + int8.
  E3 low-rank + int8        : e(update / buffer 32 + int8 codes) <= 0.01.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, prompt_state, offline_state, kinds_of, run_block, get_rows,  # noqa: E402
                   base_result, ppl_table, save_json, rss_mb, nll_of)
from kv_memory import M_numbers, rho as lr_rho  # noqa: E402

P, RK, RV, B, T_END = 768, 256, 512, 32, 1024
BUDGETS = (0.5, 0.75, 0.92)
EXTRA = {}      # layer -> bool keep mask (H or 1, k_len); None = plain sdpa
RECORD = {}     # layer -> accumulated attention (H, k_len) when recording is on
STATE = {'record': False}


def kvx_attention(module, query, key, value, attention_mask, **kw):
    """Plain sdpa unless a keep mask is set for this layer (or recording is on): then explicit causal attention."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    l = module.layer_idx
    km = EXTRA.get(l)
    if km is None and not STATE['record']:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    scaling = kw.get('scaling') or 1.0 / math.sqrt(query.shape[-1])
    q_len, k_len = query.shape[2], key.shape[2]
    s = (query @ key.transpose(-1, -2)) * scaling                       # (1, H, q, k)
    pos = torch.arange(k_len - q_len, k_len)[:, None]
    allowed = torch.arange(k_len)[None, :] <= pos                       # (q, k)
    if km is not None:
        allowed = allowed[None] & km[:, None, :]                        # (H, q, k)
    s = s.masked_fill(~allowed, float('-inf'))
    p = torch.softmax(s.float(), -1).to(query.dtype)
    if STATE['record']:
        acc = p[0].sum(1).double()                                     # (H, k): attention received
        old = RECORD.get(l)
        if old is None:
            RECORD[l] = acc
        else:
            RECORD[l] = torch.cat([old, torch.zeros(old.shape[0], k_len - old.shape[1], dtype=D64)], 1) + acc
    out = p @ value
    return out.transpose(1, 2).contiguous(), None


def quant_minmax(X, bits, dim):
    lo = X.amin(dim, keepdim=True); hi = X.amax(dim, keepdim=True)
    sc = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
    return torch.round((X - lo) / sc).clamp(0, 2 ** bits - 1) * sc + lo


def quant_keys(X, bits, group=32):
    """Per channel over groups of `group` consecutive tokens (rows)."""
    out = X.clone()
    for i in range(0, X.shape[0], group):
        out[i:i + group] = quant_minmax(X[i:i + group], bits, 0)
    return out


def quant_values(X, bits):
    """Per token over each head's 64 channels."""
    Xh = X.reshape(X.shape[0], H, Dh)
    return quant_minmax(Xh, bits, 2).reshape(X.shape[0], E)


def mem_quant(bits, T=T_END, n_ex=1 + B):
    bK, bV = bits / 8 + 4 / 32, bits / 8 + 4 / 64
    return ((T - n_ex) * E * (bK + bV) + n_ex * 2 * E * 2) / (2 * E * 2 * T)


def mem_lowrank(per_seq, n_ex=1 + B, int8=False, T=T_END):
    M = M_numbers(RK, RV, per_seq_G=per_seq, per_sequence_basis=per_seq)
    if not int8:
        return lr_rho(T, RK, RV, n_ex, M)
    # codes 1 byte + per token and kind an fp16 scale and zero (4 bytes); exact tokens and basis in fp16
    return ((T - n_ex) * ((RK + RV) * 1 + 8) + n_ex * 2 * E * 2 + 2 * M) / (2 * E * 2 * T)


def run_evict(ctx, w, policy, f):
    """Eviction baselines with masks; returns NLL. policy 'sink' (StreamingLLM) or 'h2o'."""
    n_keep = int(round(f * T_END)) - B
    m = ctx.model
    seq = w.numel()
    EXTRA.clear(); RECORD.clear()

    def masks(n, score, alive):
        """Keep mask (H, n) over the n cached tokens; score (H, n) = attention received so far (H2O); alive (H, n):
        tokens not evicted earlier (an evicted token can never come back)."""
        keep = torch.zeros(H, n, dtype=torch.bool)
        if n <= n_keep:
            keep[:] = True
            return keep & alive
        if policy == 'sink':
            keep[:, :4] = True
            keep[:, n - (n_keep - 4):] = True
            return keep & alive
        rec = n_keep // 2
        keep[:, 0] = True
        keep[:, n - rec:] = True
        for h in range(H):
            sc = score[h, :n].clone()
            sc[keep[h] | ~alive[h]] = -float('inf')
            k_ = min(n_keep - int(keep[h].sum()), int(torch.isfinite(sc).sum()))
            if k_ > 0:
                keep[h, torch.topk(sc, k_).indices] = True
        return keep & alive

    with torch.inference_mode():
        STATE['record'] = policy == 'h2o'
        out = m(w[:P][None], use_cache=True)
        past = out.past_key_values
        logits = []
        alive = {l: torch.ones(H, P, dtype=torch.bool) for l in range(L)}
        t = P
        while t < seq:
            km_layer = {}
            for l in range(L):
                kp = masks(t, RECORD[l][:, :t] if policy == 'h2o' else None, alive[l][:, :t])
                km_layer[l] = torch.cat([kp, torch.ones(H, min(B, seq - t), dtype=torch.bool)], 1)
                alive[l] = km_layer[l]  # evicted tokens stay evicted; the new block is alive
            blk = w[t:t + B]; nb = blk.numel()
            for l in range(L):
                EXTRA[l] = km_layer[l][:, :t + nb]
            out = m(blk[None], past_key_values=past)
            logits.append(out.logits[0]); past = out.past_key_values
            t += nb
        STATE['record'] = False
        EXTRA.clear(); RECORD.clear()
    return nll_of(torch.cat(logits), w, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c5_kv_baselines.json'))
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'fig'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    T0 = time.perf_counter()
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register('kvx', kvx_attention)
    AttentionMaskInterface.register('kvx', sdpa_mask)  # same masks as sdpa (without this, blocks with a cache break)
    ctx = Ctx(a.threads)
    # self-check: 'kvx' without a keep mask must give the sdpa logits, for a prefill and a block with a cache
    with torch.inference_mode():
        w0 = ctx.ids_test[:300]
        ref1 = ctx.model(w0[:200][None], use_cache=True); ref2 = ctx.model(w0[200:232][None], past_key_values=ref1.past_key_values)
        ctx.model.set_attn_implementation('kvx')
        new1 = ctx.model(w0[:200][None], use_cache=True); new2 = ctx.model(w0[200:232][None], past_key_values=new1.past_key_values)
        STATE['record'] = True  # explicit-attention path
        ex1 = ctx.model(w0[:200][None], use_cache=True); ex2 = ctx.model(w0[200:232][None], past_key_values=ex1.past_key_values)
        STATE['record'] = False; RECORD.clear()
    chk = {'kvx_vs_sdpa_max_logit_diff': float(max((ref1.logits - new1.logits).abs().max(), (ref2.logits - new2.logits).abs().max())),
           'explicit_vs_sdpa_max_logit_diff': float(max((ref1.logits - ex1.logits).abs().max(), (ref2.logits - ex2.logits).abs().max()))}
    print('attention self-check:', chk, flush=True)
    assert chk['kvx_vs_sdpa_max_logit_diff'] < 1e-4 and chk['explicit_vs_sdpa_max_logit_diff'] < 1e-2, chk
    off = ctx.offline()
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    kinds = kinds_of(RK, RV)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = ['dense'] + [f'int{b} + buffer 32' for b in (8, 4, 2)] + \
              [f'sink+window f={f}' for f in BUDGETS] + [f'H2O f={f}' for f in BUDGETS] + \
              ['offline / buffer 32', 'static / buffer 32', 'update / buffer 32',
               'offline / buffer 32 + int8 codes', 'update / buffer 32 + int8 codes']
    methods = [m for m in methods if m == 'dense' or not a.only or any(x in m for x in a.only)]
    memory = {'dense': 1.0}
    for m in methods:
        if m.startswith('int'):
            memory[m] = mem_quant(int(m[3]))
        elif m.startswith('sink') or m.startswith('H2O'):
            memory[m] = float(m.split('f=')[1])
        elif m.startswith('offline'):
            memory[m] = mem_lowrank(False, int8='int8' in m)
        elif m.startswith('static') or m.startswith('update'):
            memory[m] = mem_lowrank(True, int8='int8' in m)
    tok = T_END - P - 1
    res = base_result(a, {'P': P, 'RK': RK, 'RV': RV, 'B': B, 'windows': win_idx, 'tokens_per_window': tok,
                          'memory_T1024': memory,
                          'memory_T_inf': {m: (mem_lowrank(m.startswith(('static', 'update')), int8='int8' in m, T=10 ** 9)
                                               if ('/' in m) else memory[m]) for m in methods},
                          'attention_self_check': chk})

    def q_prompt(bits):
        def f(kind, l, X, s):
            if kind != 'K':
                return quant_values(X, bits)
            out = X.clone()                      # row 0 (sink key) stays exact and must not set group 0's scale
            out[1:] = quant_keys(X[1:], bits)
            return out
        return f

    def q_block(bits):
        return lambda kind, l, Xnew, s: (quant_keys(Xnew, bits) if kind == 'K' else quant_values(Xnew, bits))

    def codes8(kind, l, X, s):
        Tm, U = s['T'], s['U']
        c = (X - s['mu']) @ Tm[0] @ U
        c = quant_minmax(c, 8, 1)
        return s['mu'] + c @ U.T @ Tm[1]
    dummy = {(kind, l): {'mu': torch.zeros(E, dtype=D64), 'T': None, 'U': None, 'S': None} for kind in 'KV' for l in range(L)}
    nllw = {m: [] for m in methods}
    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        for m in methods:
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B)
            elif m.startswith('int'):
                bits = int(m[3])
                v, _ = run_block(ctx, w, P, kinds, dummy, B=B, buf=True, upd=False,
                                 prompt_compressor=q_prompt(bits), block_compressor=q_block(bits))
            elif m.startswith('sink'):
                v = run_evict(ctx, w, 'sink', float(m.split('f=')[1]))
            elif m.startswith('H2O'):
                v = run_evict(ctx, w, 'h2o', float(m.split('f=')[1]))
            else:
                st = offline_state(off, kinds) if m.startswith('offline') else prompt_state(ctx, pre, kinds)
                upd = m.startswith('update')
                if 'int8' in m:
                    v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd, prompt_compressor=codes8,
                                     block_compressor=codes8, post_reproj=codes8)
                else:
                    v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd)
            nllw[m].append(v)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / tok):.3f}' for m in methods), flush=True)
    ppl = ppl_table(nllw, tok)
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e})
    base = [m for m in methods if m.startswith(('int', 'sink', 'H2O'))]

    def best_at(mem):
        c = [m for m in base if memory[m] <= mem + 1e-12]
        return (min(c, key=lambda m: e[m]), min(e[m] for m in c)) if c else (None, None)
    d = {}
    try:
        d['E1_int8_near_lossless'] = e['int8 + buffer 32'] <= 0.003
        bm, be = best_at(memory['update / buffer 32'])
        d['E2_lowrank_update_competitive'] = e['update / buffer 32'] <= be
        d['E2_best_baseline_at_update_memory'] = [bm, be]
        d['E3_lowrank_int8_within_1pct'] = e['update / buffer 32 + int8 codes'] <= 0.01
    except (KeyError, TypeError) as ex:
        d['incomplete'] = f'missing {ex}'
    sec = {}
    for m in ('offline / buffer 32', 'offline / buffer 32 + int8 codes', 'update / buffer 32 + int8 codes', 'static / buffer 32'):
        if m in e:
            bm, be = best_at(memory[m])
            sec[m] = {'memory': memory[m], 'excess': e[m], 'best_baseline_at_or_below': bm, 'its_excess': be,
                      'competitive': (be is None) or e[m] <= be}
    d['E2_secondary'] = sec
    res['decision'] = d
    if a.smoke and win_idx == [0]:
        ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
        if os.path.exists(ref):
            kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
            res['code_checks_window0'] = {'static / buffer 32': (ppl.get('static / buffer 32'), kb['static / buffer B=32']),
                                          'update / buffer 32': (ppl.get('update / buffer 32'), kb['update / buffer B=32']),
                                          'offline / buffer 32': (ppl.get('offline / buffer 32'), kb['offline / buffer B=32']),
                                          'dense': (ppl['dense'], kb['dense'])}
            print('code checks (this script, kv_buffer.py) on window 0:', res['code_checks_window0'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print(f"{'method':36s} {'memory':>7s} {'ppl':>8s} {'excess%':>8s}")
    for m in methods:
        print(f"{m:36s} {memory[m]:7.3f} {ppl[m]:8.3f} {100 * e[m]:8.3f}")
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
    e, mem = res['excess'], res['memory_T1024']
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    groups = [('int', dv.C['aqua'], 'quantization'), ('sink', dv.C['yellow'], 'sink + window'), ('H2O', dv.C['magenta'], 'H2O-style'),
              ('offline', dv.C['black'], 'low-rank offline'), ('static', dv.C['violet'], 'low-rank static'), ('update', dv.C['orange'], 'low-rank update')]
    for pre, c, lab in groups:
        ms = [m for m in e if m.startswith(pre)]
        if ms:
            ax.scatter([mem[m] for m in ms], [100 * e[m] for m in ms], color=c, s=18, label=lab, zorder=3)
    ax.set_yscale('symlog', linthresh=1); ax.set_xlabel('memory at T = 1024 (fraction of fp16 cache, incl. basis)')
    ax.set_ylabel('perplexity increase (%)'); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig_c5_kv_baselines.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
