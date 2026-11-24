"""C15 (wave 3, 2026-11-24): SmolLM2-135M, fresh windows 16-31 - baselines (quantization, eviction, OFFLINE basis) next to the
low-rank methods, and confirmation of the post hoc attn-late+update cell.

Two scripts make up C15 (same windows, same P = 2048 + 1024 continuation, 8 threads):
  c15a  kv_rope_port.py --window_offset 16 --windows 16 --threads 8    all 9 existing methods (dense, static / update /
        attn-late / attn-late+update x no buffer / buffer 32), K 64 / V 128 of the 192-dim rows (half the codes), pre-RoPE.
  c15b  THIS script (reuses kv_rope_port.Port): dense, int8 + buffer 32, int4 + buffer 32, sink+window f=0.5 / 0.75,
        offline / buffer 32 (and, with --check, static / buffer 32 and update / buffer 32 for the code check against C11).
Methods of this script (the loop of kv_rope_port.py generalized; first token exact everywhere; post-RoPE cached keys):
  int{8,4} + buffer 32   KIVI-style fake quantization of the CACHED (post-RoPE) rows: keys per channel over groups of 32
        consecutive tokens (asymmetric min-max, kv_baselines.quant_keys; prompt rows 1..P-1 in groups of 32 from row 1, every
        continuation block is one group), values per token over each KV head's 64 channels; the current 32-token block stays
        fp16 while it is processed (the exact buffer) and is quantized at the end of the block; token 0 exact.
  sink+window f=0.5/0.75   StreamingLLM-style eviction with ONE sink token (token 0) and the most recent tokens; before every
        block the cache is cut to n_keep = round(f x seq) - 32 tokens (token 0 + the newest n_keep - 1), so that cache + block
        = f x seq; evicted tokens never return. The kept rows keep their ORIGINAL RoPE positions and the new block gets its
        true position ids (equivalent to masking the evicted tokens); the prompt prefill itself is exact (full attention).
  offline / buffer 32   one basis per layer from calibration: 16 random WikiText-2 TRAIN windows of 1024 tokens (SmolLM2
        tokenizer, seed 0, position 0 of every window skipped from the statistics because of the massive-activation first
        token), pre-RoPE statistics: key/value mean and covariance, G_K = blockdiag over KV heads of the summed query second
        moments, G_V = o_proj weighting as in kv_rope_port.py; top K64 / V128 eigen-directions of G^{1/2} C G^{1/2}; static
        (never updated), buffer 32, first token exact. No per-prompt statistics, so no per-sequence basis memory.
Memory (codes only, fraction of the fp16 full cache at T = 3072 incl. 33 exact tokens and the scales, NOT counting a
per-prompt basis): int8 / int4 from the stored bits plus 4 bytes per channel and 32 tokens (keys) or per token and KV head
(values); sink+window f; low-rank methods (RK + RV) / (2 dK) = 0.5 plus the exact tokens (offline needs no per-sequence
basis; the online methods would add their basis, ignored by this rule).
Sanity stored: dense NLL here vs kv_rope_port.py (merge check in c15_analyze.py); --check: static / buffer 32 and update /
buffer 32 vs c11_rope_smollm2.json window 0 (run with --window_offset 0 --windows 1); eviction plumbing: n_keep = inf gives the
dense NLL (--check); offline captured energy per layer; first window offline vs prompt fit are compared in the log.
Rules fixed before any full run (windows 16-31 = 16 windows, e(x) = ppl(x)/dense - 1; evaluated by c15_analyze.py from the
two JSONs; c15a supplies every low-rank method except the offline one):
  R1 confirms the post hoc cell: e(attn-late+update / buffer 32) <= 0.9 x e(update / buffer 32).
  R2 int4 is near lossless: e(int4 + buffer 32) <= 0.01.
  R3 offline versus online: e(offline / buffer 32) <= e(update / buffer 32) + 0.005 (does 'offline >= online' hold on SmolLM2?).
  R4 equal-memory analogue of E2: among all methods with codes-only memory <= 0.55 the one with the smallest excess is a
     low-rank method (static / update / attn-late / attn-late+update / offline). (int8 computes to 0.552, just outside the cut;
     int4 0.304, sink+window f=0.5 0.500, low-rank cells 0.505.) c15_analyze.py also reports the <= 0.60 list.
  Descriptive: the full table (memory, excess, paired CI against dense), the 9 + 6 cells, the paired CI of every low-rank
  cell against int8 and int4, and the C13 pooled-windows consistency of the update / static ratio.
Run:  .venv\\Scripts\\python.exe experiments\\kv_rope_baselines.py --threads 8 --out experiments\\rerun_2026-11-22\\c15b_rope_baselines_w16-31.json
      smoke: --window_offset 0 --windows 1 --check --out <scratch>.json  (compare with c11 window 0)
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_online import load_wikitext, psd_pow  # noqa: E402
from kv_rope_port import Port, Pi, top_right, D64, ROOT  # noqa: E402
from kv_baselines import quant_minmax, quant_keys  # noqa: E402

METHODS = ['dense', 'int8 + buffer 32', 'int4 + buffer 32', 'sink+window f=0.5', 'sink+window f=0.75', 'offline / buffer 32']
CHECK = ['static / buffer 32', 'update / buffer 32']


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return None


def quant_values_g(X, bits, kvh, dh):
    Xh = X.reshape(X.shape[0], kvh, dh)
    return quant_minmax(Xh, bits, 2).reshape(X.shape[0], kvh * dh)


def calibrate(pt, ids, n=16, seq=1024):
    """Offline pre-RoPE statistics; returns {(kind, l): (mu, T, U_full_flipped, lam)}."""
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids.numel() - seq, (n,), generator=g).tolist()
    S1 = {k: [None] * pt.L for k in 'qKV'}
    S2 = {k: [None] * pt.L for k in 'qKV'}
    pt.mode['proj'] = None
    with torch.inference_mode():
        for s in starts:
            pt.model(ids[s:s + seq][None])
            for l in range(pt.L):
                for k in 'qKV':
                    X = pt.cap[(l, k)].to(D64)[1:]
                    S1[k][l] = X.sum(0) if S1[k][l] is None else S1[k][l] + X.sum(0)
                    S2[k][l] = X.T @ X if S2[k][l] is None else S2[k][l] + X.T @ X
    nr = n * (seq - 1)
    off, energy = {}, {}
    for l in range(pt.L):
        Sq = S2['q'][l] / nr
        blocks = [sum(Sq[h * pt.Dh:(h + 1) * pt.Dh, h * pt.Dh:(h + 1) * pt.Dh] for h in range(g_ * pt.rep, (g_ + 1) * pt.rep))
                  for g_ in range(pt.KVH)]
        G = torch.block_diag(*blocks)
        TK = (psd_pow(G, 0.5), psd_pow(G, -0.5))
        for kind, T in (('K', TK), ('V', pt.TV[l])):
            mu = S1[kind][l] / nr
            C = S2[kind][l] / nr - torch.outer(mu, mu)
            w_, U = torch.linalg.eigh(T[0] @ C @ T[0])
            off[(kind, l)] = (mu, T, U.flip(1), w_.flip(0).clamp_min(0))
    return off


def mem_quant(bits, kvh_dh, dh, T, n_ex):
    bK, bV = bits / 8 + 4 / 32, bits / 8 + 4 / dh
    return ((T - n_ex) * kvh_dh * (bK + bV) + n_ex * 2 * kvh_dh * 2) / (2 * kvh_dh * 2 * T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=16)
    ap.add_argument('--P', type=int, default=2048)
    ap.add_argument('--seq', type=int, default=3072)
    ap.add_argument('--RK', type=int, default=64)
    ap.add_argument('--RV', type=int, default=128)
    ap.add_argument('--B', type=int, default=32)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--check', action='store_true', help='also run static / update buffer 32 and the eviction plumbing check')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c15b_rope_baselines_w16-31.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    pt = Port(a)
    P, B, seq, L_ = a.P, a.B, a.seq, pt.L
    ids = pt.tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    win_idx = list(range(a.window_offset, a.window_offset + a.windows))
    need = (win_idx[-1] + 1) * seq
    print(f'SmolLM2 test tokens: {ids.numel()}, windows {win_idx[0]}..{win_idx[-1]} need {need}', flush=True)
    assert need <= ids.numel(), f'not enough test tokens: need {need}, have {ids.numel()}'
    wins = [ids[i * seq:(i + 1) * seq] for i in win_idx]
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    if a.check:
        methods += CHECK
    kinds = [('K', a.RK, 'keys'), ('V', a.RV, 'values')]
    cs_all = pt.cos_sin(seq)
    off = None
    res = {'config': vars(a), 'torch': torch.__version__, 'model_dims': {'L': pt.L, 'Hq': pt.Hq, 'KVH': pt.KVH, 'Dh': pt.Dh},
           'windows': win_idx, 'tokens_per_window': seq - P - 1, 'test_tokens_available': int(ids.numel()), 'sanity': {}}
    if any(m.startswith('offline') for m in methods):
        ids_tr = pt.tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
        off = calibrate(pt, ids_tr)
        res['offline_energy_captured'] = {f'{kind}{l}': float(off[(kind, l)][3][:R].sum() / off[(kind, l)][3].sum())
                                          for kind, R, _ in kinds for l in (0, L_ // 2, L_ - 1)}
        print(f'offline bases ready ({time.perf_counter()-T0:.0f}s) energy captured:', res['offline_energy_captured'], flush=True)
    st_off = {(kind, l): {'mu': off[(kind, l)][0], 'T': off[(kind, l)][1], 'U': off[(kind, l)][2][:, :R].clone(), 'S': None}
              for kind, R, _ in kinds for l in range(L_)} if off else None
    m_ = pt.model
    tokw = seq - P - 1
    # memory (codes only)
    T_, n_ex = seq, 1 + B
    mem = {'dense': 1.0, 'int8 + buffer 32': mem_quant(8, pt.dK, pt.Dh, T_, n_ex), 'int4 + buffer 32': mem_quant(4, pt.dK, pt.Dh, T_, n_ex)}
    for f in (0.5, 0.75):
        mem[f'sink+window f={f}'] = f
    lr_mem = ((T_ - n_ex) * (a.RK + a.RV) * 2 + n_ex * 2 * pt.dK * 2) / (2 * pt.dK * 2 * T_)
    for m in methods:
        if m.startswith(('offline', 'static', 'update')):
            mem[m] = lr_mem
    res['memory_codes_only_T'] = mem

    def nll_of(lg, w):
        lp = torch.log_softmax(lg[:-1].double(), -1)
        tgt = w[P + 1:]
        return float(-lp[torch.arange(tgt.numel()), tgt].sum())

    def prompt_state(pre, w):
        st = {}
        for kind, R, _ in kinds:
            for l in range(L_):
                Xq, Xr = pre[(l, 'q')][1:], pre[(l, kind)][1:]
                if kind == 'K':
                    Qh = Xq.reshape(-1, pt.Hq, pt.Dh)
                    blocks = [sum(Qh[:, h].T @ Qh[:, h] for h in range(g * pt.rep, (g + 1) * pt.rep)) / Xq.shape[0]
                              for g in range(pt.KVH)]
                    G = torch.block_diag(*blocks); T = (psd_pow(G, 0.5), psd_pow(G, -0.5))
                else:
                    T = pt.TV[l]
                mu = Xr.mean(0)
                U, S = top_right((Xr - mu) @ T[0], R)
                st[(kind, l)] = {'mu': mu, 'T': T, 'U': U, 'S': S}
        return st

    def prefill(w):
        pt.mode['proj'] = None
        out = m_(w[:P][None], use_cache=True)
        return out.past_key_values

    def run_lowrank(w, st, upd):
        """kv_rope_port.py loop, buffer 32 (block exact while processed), static or update."""
        cs_p = (cs_all[0][:P], cs_all[1][:P])
        past = prefill(w)
        for kind, R, attr in kinds:
            for l in range(L_):
                X = pt.get(past, l, attr, P)
                if kind == 'K':
                    X = pt.rot(X, cs_p, inverse=True)
                s = st[(kind, l)]
                Xh = s['mu'] + (X - s['mu']) @ Pi(s); Xh[0] = X[0]
                pt.put(past, l, attr, pt.rot(Xh, cs_p) if kind == 'K' else Xh, P)
        logits, t = [], P
        while t < seq:
            blk = w[t:t + B]; nb = blk.numel()
            out = m_(blk[None], past_key_values=past)
            logits.append(out.logits[0]); past = out.past_key_values
            if t + nb < seq:
                cs_t = (cs_all[0][:t + nb], cs_all[1][:t + nb])
                for kind, R, attr in kinds:
                    for l in range(L_):
                        X = pt.get(past, l, attr, t + nb)
                        if kind == 'K':
                            X = pt.rot(X, cs_t, inverse=True)
                        s = st[(kind, l)]
                        Xnew = pt.cap[(l, kind)].to(D64)
                        if upd:
                            Ablk = (Xnew - s['mu']) @ s['T'][0]
                            M = torch.cat([s['S'][:, None] * s['U'].T, Ablk])
                            s['U'], s['S'] = top_right(M, R)
                            Pn = Pi(s)
                            X[1:t] = s['mu'] + (X[1:t] - s['mu']) @ Pn
                        else:
                            Pn = Pi(s)
                        X[t:] = s['mu'] + (Xnew - s['mu']) @ Pn
                        pt.put(past, l, attr, pt.rot(X, cs_t) if kind == 'K' else X, t + nb)
            t += nb
        return nll_of(torch.cat(logits), w)

    def run_quant(w, bits):
        past = prefill(w)
        for l in range(L_):
            for attr in ('keys', 'values'):
                X = pt.get(past, l, attr, P)
                out = X.clone()
                out[1:] = quant_keys(X[1:], bits) if attr == 'keys' else quant_values_g(X[1:], bits, pt.KVH, pt.Dh)
                pt.put(past, l, attr, out, P)
        logits, t = [], P
        while t < seq:
            blk = w[t:t + B]; nb = blk.numel()
            out = m_(blk[None], past_key_values=past)
            logits.append(out.logits[0]); past = out.past_key_values
            if t + nb < seq:
                for l in range(L_):
                    for attr in ('keys', 'values'):
                        X = pt.get(past, l, attr, t + nb)
                        X[t:] = quant_keys(X[t:], bits) if attr == 'keys' else quant_values_g(X[t:], bits, pt.KVH, pt.Dh)
                        pt.put(past, l, attr, X, t + nb)
            t += nb
        return nll_of(torch.cat(logits), w)

    def run_evict(w, n_keep):
        past = prefill(w)
        pos = torch.arange(P)
        logits, t = [], P
        while t < seq:
            blk = w[t:t + B]; nb = blk.numel()
            if pos.numel() > n_keep:
                idx = torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(pos.numel() - (n_keep - 1), pos.numel())])
                for lay in past.layers:
                    lay.keys = lay.keys[:, :, idx].contiguous(); lay.values = lay.values[:, :, idx].contiguous()
                pos = pos[idx]
            n_c = pos.numel()
            assert past.get_seq_length() == n_c, (past.get_seq_length(), n_c)
            out = m_(blk[None], past_key_values=past, position_ids=torch.arange(t, t + nb)[None],
                     cache_position=torch.arange(n_c, n_c + nb))
            logits.append(out.logits[0]); past = out.past_key_values
            pos = torch.cat([pos, torch.arange(t, t + nb)])
            t += nb
        return nll_of(torch.cat(logits), w)

    nllw = {m: [] for m in methods}
    tw = {m: [] for m in methods}
    with torch.inference_mode():
        for wi, w in zip(win_idx, wins):
            pre = None
            for m in methods:
                t1 = time.perf_counter()
                if m == 'dense':
                    pt.mode['proj'] = None
                    lg = []
                    past = prefill(w)
                    t = P
                    while t < seq:
                        blk = w[t:t + B]
                        o = m_(blk[None], past_key_values=past)
                        lg.append(o.logits[0]); past = o.past_key_values; t += blk.numel()
                    v = nll_of(torch.cat(lg), w)
                elif m.startswith('int'):
                    v = run_quant(w, int(m[3]))
                elif m.startswith('sink'):
                    f = float(m.split('f=')[1])
                    v = run_evict(w, int(round(f * seq)) - B)
                elif m.startswith('offline'):
                    st = {k: {**s_, 'U': s_['U'].clone()} for k, s_ in st_off.items()}
                    v = run_lowrank(w, st, False)
                else:  # static / update buffer 32 (check)
                    if pre is None:
                        pt.mode['proj'] = None
                        m_(w[:P][None])
                        pre = {(l, k): pt.cap[(l, k)].to(D64) for l in range(L_) for k in ('q', 'K', 'V')}
                    v = run_lowrank(w, prompt_state(pre, w), m.startswith('update'))
                nllw[m].append(v); tw[m].append(time.perf_counter() - t1)
            if a.check and wi == win_idx[0] and 'evict_plumbing_nll_diff' not in res['sanity']:
                res['sanity']['evict_plumbing_nll_diff'] = abs(run_evict(w, 10 ** 9) - nllw['dense'][0])
                print('eviction plumbing (n_keep = inf vs dense NLL):', res['sanity']['evict_plumbing_nll_diff'], flush=True)
            print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
                  ' '.join(f'{m}: {math.exp(nllw[m][-1] / tokw):.3f}' for m in methods), flush=True)

    ppl = {m: math.exp(sum(v) / (tokw * len(v))) for m, v in nllw.items()}
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e, 'timing_s': tw})
    if a.check and win_idx == [0]:
        c11 = json.load(open(os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c11_rope_smollm2.json')))['nll_per_window']
        res['code_checks_window0_vs_c11'] = {m: (nllw[m][0], c11[m][0]) for m in ('dense', 'static / buffer 32', 'update / buffer 32') if m in nllw}
        print('code checks vs c11 (window 0 NLL):', res['code_checks_window0_vs_c11'])
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"{'method':28s} {'memory':>7s} {'ppl':>8s} {'excess%':>8s}")
    for m in methods:
        print(f"{m:28s} {mem.get(m, float('nan')):7.3f} {ppl[m]:8.3f} {100 * e[m]:8.3f}")
    print('seconds per window and method:', {k: round(float(np.mean(v)), 1) for k, v in tw.items()})


if __name__ == '__main__':
    main()
