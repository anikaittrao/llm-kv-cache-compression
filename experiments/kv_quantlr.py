"""C16 (wave 3, 2026-11-24): low rank PLUS quantization on GPT-2. Why does 8-bit coding hurt the updated basis, and where is
the frontier?

Background (C5, kv_baselines.py, windows 0-15, main setting): int4 keys/values with a 32-token exact buffer cost +0.10% at
memory 0.32; offline/buffer 32 +1.19%; update/buffer 32 +0.20% but update + int8 codes (one scale per token and kind,
re-projected old codes re-quantized after every update) +2.54% (memory 0.68), offline + int8 codes +1.76% (0.28). Two
hypotheses for the +2.3 point jump of the updated basis:
  A (compounding): the old rows are re-projected AND re-quantized at every update (7 updates in 256 tokens), so the code
     error accumulates with the number of re-projections.
  B (scale): one min-max scale per token over all R codes: the first PCA coordinates are large, so the scale zeroes the
     low-energy coordinates; the updated basis rotates the coordinates at every update and the per-token scale mismatch
     compounds. Per-coordinate scales over groups of 32 tokens (as for the keys in KIVI) avoid this.
Setting: kv_buffer.py main setting (K256/V512 = half the codes, P = 768, B = 32, 256 continuation tokens, windows 0-15,
first token exact, 32-token exact block buffer, prompt compressed after prefill). Codes c = (x - mu) G^{1/2} U are quantized
asymmetric min-max, 'per token' = one fp16 scale+zero per token and kind over its R codes (kv_baselines.py codes8),
'grouped' = one scale+zero per coordinate over groups of 32 consecutive tokens (prompt rows 1..767 in groups of 32 from row
1, last group 31 rows; every continuation block is one group; when the basis is updated the old rows are re-projected,
then re-quantized with the same group edges).
Methods (names):
  dense; static / buffer 32; update / buffer 32 (code checks 24.997 / 24.778 on 16 windows);
  (i)   static / buffer 32 + int8 codes             no re-projection ever
  (ii)  update / buffer 32 + int8 codes, master     float master rows are re-projected at every update; the cache the
        attention reads is quantize(master) in the current basis (so no quantization is ever re-quantized). DIAGNOSTIC:
        the float master rows are not counted in the memory.
  (iii) update / buffer 32 + int8 grouped codes     compounding as in (iv) but with grouped per-coordinate scales
  (iv)  update / buffer 32 + int8 codes             reference = C5 'update / buffer 32 + int8 codes' (+2.54%; code check)
  (iii-m) update / buffer 32 + int8 grouped codes, master   (extra: both fixes)
  int4 + buffer 32 (C5 baseline: KIVI-style keys per channel over groups of 32 tokens, values per token and head, +0.10%)
  offline K256/V512 / buffer 32 (+ unquantized reference, check 25.022); offline K256/V512 + int4 codes (per token);
  offline K384/V768 / buffer 32 and + int4 codes (3/4 of the codes); extra: offline K256/V512 and K384/V768 + int4 grouped
  codes (descriptive; in the 1-window smoke test per-token int4 codes were catastrophic because of the min-max scale).
Memory (fraction of the fp16 full cache at T = 1024, incl. 33 exact tokens, per-prompt basis and weights for online
bases, scales; formula of kv_baselines.mem_lowrank generalized to b bits; offline: no per-sequence basis) is stored per method
under 'memory_T1024'. NOTE the orchestrator's brief quoted ~0.28 (offline K256/V512 + int4) and ~0.37 (K384/V768 + int4);
those are the INT8 numbers (0.277 = C5; 0.75 x 0.5 = 0.375). With 4-bit codes the formula gives about 0.16 and 0.21.
Diagnostic (iv) and (ii): relative code error versus the number of re-projections each row went through, at the LAST update
(after block 6 of 8; rows of the prompt: 7 re-projections, continuation block b: 6 - b, the newest block: 0). For each layer
and kind: sqrt(sum_rows ||(cache - master) G^{1/2}||^2 / sum_rows ||(master - mu) G^{1/2}||^2) per count, where master =
the float re-projected rows of the same run (not the unquantized update run, which differs by the changed attention). Stored
as reproj_err[method][window][kind][layer][count 0..7].
Rules fixed before any full run (e(x) = ppl(x)/dense - 1, 16 windows 0-15; evaluated by c16_analyze.py):
  Q1 hypothesis A: e(ii) <= e(i) + 0.003   (quantizing at read removes the compounding: the updated basis with float master
     codes is no worse than STATIC + int8 by more than 0.3 point).
  Q2 hypothesis B: e(iii) <= e(iv) - 0.005 (grouped per-coordinate scales recover at least 0.5 point of update + int8).
  Q3 frontier: e(offline K256/V512 + int4 codes) <= 0.01.  Q3b (descriptive but also reported pass/fail): the same for K384/V768.
  Verdict (computed, not a rule on a threshold): A is supported if Q1 passes AND e(iv) - e(ii) >= 0.005; B is supported if Q2
  passes AND e(iv) - e(iii) >= 0.005; the diagnostic supports A if the median relative code error of (iv) at 7 re-projections
  is >= 1.5 x that at 0 re-projections while (ii) is flat (ratio <= 1.2).
Run:  .venv\\Scripts\\python.exe experiments\\kv_quantlr.py --threads 8 --out experiments\\rerun_2026-11-22\\c16_kv_quantlr.json
      smoke: --windows 1 --smoke --out <scratch>.json   (code checks against C5 window 0 and kv_buffer.py window 0)
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, H, E, Dh, D64, ROOT, Pi, prompt_state, offline_state, kinds_of, run_block, base_result,  # noqa: E402
                   save_json, rss_mb)
from kv_baselines import quant_minmax, quant_keys, quant_values  # noqa: E402
from kv_memory import M_numbers  # noqa: E402

P, B, SEQ, T_END = 768, 32, 1024, 1024
NCNT = 8  # re-projection counts 0..7
METHODS = ['dense', 'static / buffer 32', 'update / buffer 32',
           'static / buffer 32 + int8 codes',                              # (i)
           'update / buffer 32 + int8 codes, master',                     # (ii)
           'update / buffer 32 + int8 grouped codes',                     # (iii)
           'update / buffer 32 + int8 codes',                             # (iv)
           'update / buffer 32 + int8 grouped codes, master',             # (iii-m)
           'int4 + buffer 32',
           'offline K256/V512 / buffer 32', 'offline K256/V512 / buffer 32 + int4 codes',
           'offline K256/V512 / buffer 32 + int4 grouped codes',
           'offline K384/V768 / buffer 32', 'offline K384/V768 / buffer 32 + int4 codes',
           'offline K384/V768 / buffer 32 + int4 grouped codes']


def quant_rows(X, s, bits, grouped, edges=None):
    """Quantize the codes of rows X in the basis s (per token over the R codes, or grouped per coordinate), reconstruct."""
    c = (X - s['mu']) @ s['T'][0] @ s['U']
    if grouped:
        cq = torch.empty_like(c)
        for lo, hi in zip(edges[:-1], edges[1:]):
            cq[lo:hi] = quant_minmax(c[lo:hi], bits, 0)
    else:
        cq = quant_minmax(c, bits, 1)
    return s['mu'] + cq @ s['U'].T @ s['T'][1]


def edges_n(n):
    return list(range(0, n, 32)) + [n]


def edges_cont(t):
    """Group edges (relative to row 1) of rows 1..t-1: prompt groups of 32 from row 1 (last 31), then one group per block."""
    e = list(range(0, P - 1, 32)) + [P - 1]
    e += list(range(P - 1 + 32, t - 1, 32))
    if e[-1] != t - 1:
        e.append(t - 1)
    return e


class QRun:
    """Compressor hooks for kvlab.run_block. master: re-quantize from a float master instead of the already quantized rows."""

    def __init__(self, bits, grouped=False, master=False, diag=False):
        self.bits, self.grouped, self.master, self.diag = bits, grouped, master, diag
        self.track = master or diag
        self.M, self.rc, self.last, self.rel = {}, {}, {}, {}

    def prompt(self, kind, l, X, s):
        out = X.clone()
        out[1:] = quant_rows(X[1:], s, self.bits, self.grouped, edges_n(X.shape[0] - 1) if self.grouped else None)
        if self.track:
            Xm = s['mu'] + (X - s['mu']) @ Pi(s); Xm[0] = X[0]
            self.M[(kind, l)] = Xm
            self.rc[(kind, l)] = torch.zeros(X.shape[0], dtype=torch.long)
        return out

    def reproj(self, kind, l, Xold, s):
        key = (kind, l)
        t = Xold.shape[0] + 1
        if self.track:
            M = self.M[key]
            assert M.shape[0] == t, (M.shape, t)
            M[1:] = s['mu'] + (M[1:] - s['mu']) @ Pi(s)
            self.rc[key][1:] += 1
        src = self.M[key][1:] if self.master else Xold
        out = quant_rows(src, s, self.bits, self.grouped, edges_cont(t) if self.grouped else None)
        self.last[key] = out
        return out

    def block(self, kind, l, Xnew, s):
        key = (kind, l)
        q = quant_rows(Xnew, s, self.bits, self.grouped, [0, Xnew.shape[0]] if self.grouped else None)
        if self.track:
            Xm = s['mu'] + (Xnew - s['mu']) @ Pi(s)
            if self.diag and key in self.last:
                M, T0 = self.M[key][1:], s['T'][0]
                D = torch.cat([((self.last[key] - M) @ T0).pow(2).sum(1), ((q - Xm) @ T0).pow(2).sum(1)])
                Rf = torch.cat([((M - s['mu']) @ T0).pow(2).sum(1), ((Xm - s['mu']) @ T0).pow(2).sum(1)])
                cnt = torch.cat([self.rc[key][1:], torch.zeros(Xnew.shape[0], dtype=torch.long)])
                rel = []
                for c in range(NCNT):
                    mk = cnt == c
                    rel.append(float((D[mk].sum() / Rf[mk].sum()).sqrt()) if bool(mk.any()) else None)
                self.rel[key] = rel                               # overwritten at every update: the last one stays
            self.M[key] = torch.cat([self.M[key], Xm])
            self.rc[key] = torch.cat([self.rc[key], torch.zeros(Xnew.shape[0], dtype=torch.long)])
        return q


def mem_lr(bits, RK, RV, per_seq, grouped=False, T=T_END, n_ex=1 + B):
    M = M_numbers(RK, RV, per_seq_G=per_seq, per_sequence_basis=per_seq)
    R = RK + RV
    code_bytes = R * bits / 8 + (R * 4 / 32 if grouped else 8)
    return ((T - n_ex) * code_bytes + n_ex * 2 * E * 2 + 2 * M) / (2 * E * 2 * T)


def mem_int(bits, T=T_END, n_ex=1 + B):
    bK, bV = bits / 8 + 4 / 32, bits / 8 + 4 / 64
    return ((T - n_ex) * E * (bK + bV) + n_ex * 2 * E * 2) / (2 * E * 2 * T)


def memory_of(m):
    if m in ('dense',):
        return 1.0
    if m.startswith('int4'):
        return mem_int(4)
    off = m.startswith('offline')
    RK, RV = (384, 768) if 'K384' in m else (256, 512)
    bits = 4 if 'int4' in m else (8 if 'int8' in m else 16)
    grouped = 'grouped' in m
    if bits == 16:
        M = M_numbers(RK, RV, per_seq_G=not off, per_sequence_basis=not off)
        return ((T_END - 33) * (RK + RV) * 2 + 33 * 3072 + 2 * M) / (3072 * T_END)   # fp16 codes
    return mem_lr(bits, RK, RV, not off, grouped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--window_offset', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c16_kv_quantlr.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    off = ctx.offline()
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    kinds = kinds_of(256, 512)
    kinds_big = kinds_of(384, 768)
    win_idx, wins = ctx.windows(a.windows, a.window_offset)
    methods = [m for m in METHODS if m == 'dense' or not a.only or any(x in m for x in a.only)]
    if a.smoke:
        methods.append('update / buffer 32 + int8 codes (C5 codes8, check)')
    tok = SEQ - P - 1
    res = base_result(a, {'P': P, 'B': B, 'windows': win_idx, 'tokens_per_window': tok,
                          'memory_T1024': {m: memory_of(m) for m in methods}, 'reproj_err': {}})
    nllw = {m: [] for m in methods}
    tw = {m: [] for m in methods}
    dummy = {(kind, l): {'mu': torch.zeros(E, dtype=D64), 'T': None, 'U': None, 'S': None} for kind in 'KV' for l in range(L)}

    def q_prompt4(kind, l, X, s):
        if kind != 'K':
            return quant_values(X, 4)
        out = X.clone(); out[1:] = quant_keys(X[1:], 4)
        return out

    for wi, w in zip(win_idx, wins):
        pre = ctx.prefill_stats(w, P)
        for m in methods:
            t1 = time.perf_counter()
            if m == 'dense':
                v, _ = run_block(ctx, w, P, kinds, None, B=B)
            elif m == 'int4 + buffer 32':
                v, _ = run_block(ctx, w, P, kinds, dummy, B=B, buf=True, upd=False, prompt_compressor=q_prompt4,
                                 block_compressor=lambda kind, l, Xn, s: quant_keys(Xn, 4) if kind == 'K' else quant_values(Xn, 4))
            elif m.startswith('offline'):
                kd = kinds_big if 'K384' in m else kinds
                st = offline_state(off, kd)
                if 'int4' in m:
                    q = QRun(4, grouped='grouped' in m)
                    v, _ = run_block(ctx, w, P, kd, st, B=B, buf=True, upd=False, prompt_compressor=q.prompt, block_compressor=q.block)
                else:
                    v, _ = run_block(ctx, w, P, kd, st, B=B, buf=True, upd=False)
            else:
                st = prompt_state(ctx, pre, kinds)
                upd = m.startswith('update')
                if 'C5 codes8' in m:                    # --smoke only: kv_baselines.py codes8 verbatim, same process
                    def codes8(kind, l, X, s):
                        c = quant_minmax((X - s['mu']) @ s['T'][0] @ s['U'], 8, 1)
                        return s['mu'] + c @ s['U'].T @ s['T'][1]
                    v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd, prompt_compressor=codes8,
                                     block_compressor=codes8, post_reproj=codes8)
                elif 'int8' not in m:
                    v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd)
                else:
                    q = QRun(8, grouped='grouped' in m, master='master' in m, diag=upd and 'grouped' not in m)
                    v, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd, prompt_compressor=q.prompt,
                                     block_compressor=q.block, post_reproj=q.reproj if upd else None)
                    if q.diag:
                        res['reproj_err'].setdefault(m, []).append(
                            {kind: [q.rel.get((kind, l)) for l in range(L)] for kind in 'KV'})
            nllw[m].append(v); tw[m].append(time.perf_counter() - t1)
        print(f'window {wi} done ({time.perf_counter()-T0:.0f}s, rss {rss_mb():.0f} MB) ' +
              ' '.join(f'{m}: {math.exp(nllw[m][-1] / tok):.3f}' for m in methods), flush=True)
    ppl = {m: math.exp(sum(v) / (tok * len(v))) for m, v in nllw.items()}
    e = {m: ppl[m] / ppl['dense'] - 1 for m in ppl}
    res.update({'nll_per_window': nllw, 'ppl': ppl, 'excess': e, 'timing_s': tw})
    if a.smoke and win_idx == [0]:
        chk = {}
        ref = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'smoke', 'patched_kv_buffer_w1.json')
        if os.path.exists(ref):
            kb = json.load(open(ref))['settings']['KV 256/512 P=768']['ppl']
            for m, k in (('static / buffer 32', 'static / buffer B=32'), ('update / buffer 32', 'update / buffer B=32'),
                         ('offline K256/V512 / buffer 32', 'offline / buffer B=32'), ('dense', 'dense')):
                if m in ppl:
                    chk[m] = (ppl[m], kb[k])
        c5p = os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c5_kv_baselines.json')
        if os.path.exists(c5p):
            c5 = json.load(open(c5p))['nll_per_window']
            for m, k in (('update / buffer 32 + int8 codes', 'update / buffer 32 + int8 codes'), ('int4 + buffer 32', 'int4 + buffer 32')):
                if m in nllw:
                    chk['NLL ' + m] = (nllw[m][0], c5[k][0])
        res['code_checks_window0'] = chk
        print('code checks:', json.dumps(chk, indent=1))
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print(f"{'method':58s} {'memory':>7s} {'ppl':>8s} {'excess%':>8s}")
    for m in methods:
        print(f"{m:58s} {res['memory_T1024'][m]:7.3f} {ppl[m]:8.3f} {100 * e[m]:8.3f}")
    print('seconds per window and method:', {k: round(float(np.mean(v)), 1) for k, v in tw.items()})
    for m, rows in res['reproj_err'].items():
        for kind in 'KV':
            med = [float(np.median([r[kind][l][c] for r in rows for l in range(L) if r[kind][l] is not None and r[kind][l][c] is not None] or [np.nan]))
                   for c in range(NCNT)]
            print(f'reproj error median by count 0..7 [{m}] {kind}:', [round(x, 4) for x in med])


if __name__ == '__main__':
    main()
