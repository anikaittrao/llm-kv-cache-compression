"""C19 (wave 4, 2026-11-24): the attention-sink / centering / keep0 check of Section 5.7 repeated on SmolLM2-135M (RoPE, GQA).

Model and pre-RoPE convention as kv_rope_port.py (30 layers, 9 query heads, 3 KV heads of dim 64; keys are compressed BEFORE
the rotary embedding, inside k_proj). Per-head key bases like kv_sink2.py (C7): one basis per KV head g (64 dims), rank r of 64,
  plain    : mu = 0, top r eigenvectors of E[k k^T]
  centered : mu = mean key, top r eigenvectors of the covariance
calibrated on 16 train windows of 1024 tokens (seed 0, all positions incl. position 0); keep0 = position 0's key stored exactly.
Evaluation as kv_adaptive.py / kv_sink2.py: full 1024-token passes of test windows 0-7, keys of every position compressed,
perplexity over all predictions; ranks r = 16, 32, 48. Per-window NLL stored; paired bootstrap CIs by stats_utils.ppl_ratio.
Mechanism (dense passes of test windows 0-3, per layer, median over the 3 KV heads, mean over windows):
  k0_coord_mu = k_0 . mu / ||mu||^2 (pre-RoPE; position 0 is not rotated, so this is also the post-RoPE key),
  attn0 = mean attention weight on position 0 (post-RoPE scores, head mean over the 9 query heads, queries t >= 1),
  err0 / errrest (r = 32, plain and centered): mean |q_t . (rot(khat_s) - rot(k_s))| / 8 for s = 0 and for 1 <= s <= t
  (post-RoPE score error of the pre-RoPE-compressed key; query heads of the KV group).
Rules fixed before the run:
  S1 geometry      : the layer median of k0_coord_mu is < 0 in at least 20 of the 30 layers.
  S2 centering hurts: ppl(centered r) > ppl(plain r) at r = 16, 32 and 48.
  S3 keep0 rescues centering: ppl(centered r keep0) < ppl(centered r) at all three ranks.
  S4 keep0 hurts plain (the GPT-2 puzzle replicated): ppl(plain r keep0) > ppl(plain r) at >= 2 of the 3 ranks.
  S5 the GPT-2 'explained' rule: ppl(centered r keep0) <= ppl(plain r keep0) at all three ranks.
  S6 descriptive: attn0 >= 0.3 in how many layers (no threshold on the count).
Run:  .venv\\Scripts\\python.exe experiments\\kv_rope_sink.py --threads 4 --out experiments\\rerun_2026-11-22\\c19_rope_sink.json
      smoke: --windows 1 --mech_windows 1 --ranks 32
"""
import argparse, json, math, os, sys, time, types
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_rope_port import Port, rotate_half, D64, ROOT, rss_mb  # noqa: E402
from kv_online import load_wikitext  # noqa: E402
from stats_utils import ppl_ratio  # noqa: E402


class SinkPort(Port):
    def _hook(self, l, kind, o):
        self.cap[(l, kind)] = o[0].detach()
        P_ = self.mode['proj']
        if P_ is None or kind not in P_:
            return o
        mu, Pm = P_[kind][l]
        out = mu + (o - mu) @ Pm
        if self.mode.get('keep0'):
            out[:, 0] = o[:, 0]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--mech_windows', type=int, default=4)
    ap.add_argument('--ranks', type=int, nargs='*', default=[16, 32, 48])
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c19_rope_sink.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    pt = SinkPort(a)
    L_, KVH, Dh, dK, Hq, rep = pt.L, pt.KVH, pt.Dh, pt.dK, pt.Hq, pt.rep
    seq = a.seq
    ids_te = pt.tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_tr = pt.tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids_tr.numel() - seq, (16,), generator=g).tolist()
    S1 = [torch.zeros(dK, dtype=D64) for _ in range(L_)]; S2 = [torch.zeros(dK, dK, dtype=D64) for _ in range(L_)]
    with torch.inference_mode():
        for s in starts:
            pt.model(ids_tr[s:s + seq][None])
            for l in range(L_):
                X = pt.cap[(l, 'K')].to(D64); S1[l] += X.sum(0); S2[l] += X.T @ X
    n = 16 * seq
    mu = [S1[l] / n for l in range(L_)]; M2 = [S2[l] / n for l in range(L_)]
    C = [M2[l] - torch.outer(mu[l], mu[l]) for l in range(L_)]
    wins = [ids_te[i * seq:(i + 1) * seq] for i in range(a.windows)]
    tok = seq - 1
    print(f'calibration ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    def projector(kind, r):
        out = []
        for l in range(L_):
            Pm = torch.zeros(dK, dK, dtype=D64); m_ = torch.zeros(dK, dtype=D64)
            for gi in range(KVH):
                s = slice(gi * Dh, (gi + 1) * Dh)
                Mat = M2[l][s, s] if kind == 'plain' else C[l][s, s]
                U = torch.linalg.eigh(Mat)[1][:, -r:]
                Pm[s, s] = U @ U.T
                if kind == 'centered':
                    m_[s] = mu[l][s]
            out.append((m_.float(), Pm.float(), m_, Pm))
        return out

    def evaluate(proj, keep0):
        pt.mode['proj'] = {'K': [(p[0], p[1]) for p in proj]} if proj else None
        pt.mode['keep0'] = keep0
        nll = []
        with torch.inference_mode():
            for w in wins:
                nll.append(float(pt.model(w[None], labels=w[None]).loss) * tok)
        pt.mode['proj'] = None; pt.mode['keep0'] = False
        return nll

    res = {'config': vars(a), 'torch': torch.__version__, 'tokens_per_window': tok, 'runs': {}}
    dense = evaluate(None, False)
    res['dense_nll'] = dense; res['dense_ppl'] = math.exp(sum(dense) / (tok * len(dense)))
    print(f"dense ppl {res['dense_ppl']:.3f} ({time.perf_counter()-T0:.0f}s)", flush=True)
    nllw = {}
    for r in a.ranks:
        for kind in ('plain', 'centered'):
            proj = projector(kind, r)
            for k0 in (False, True):
                lab = f'{kind} r={r}' + (' keep0' if k0 else '')
                nllw[lab] = evaluate(proj, k0)
                res['runs'][lab] = {'ppl': math.exp(sum(nllw[lab]) / (tok * len(wins))), 'nll_per_window': nllw[lab]}
                print(f"{lab:24s} {res['runs'][lab]['ppl']:9.3f} ({time.perf_counter()-T0:.0f}s)", flush=True)

    # ---- mechanism
    cs = pt.cos_sin(seq)
    cosq, sinq = cs[0][:, None, :], cs[1][:, None, :]
    geo = np.zeros((L_, KVH)); att0 = np.zeros(L_)
    err = {(kind, which): np.zeros(L_) for kind in ('plain', 'centered') for which in ('err0', 'errrest')}
    P32 = {kind: projector(kind, 32) for kind in ('plain', 'centered')}
    nm = a.mech_windows
    pt.mode['proj'] = None
    causal_rest = torch.tril(torch.ones(seq, seq, dtype=D64), -1)[1:, 1:]        # s in 1..t-1 (t >= 2)
    with torch.inference_mode():
        for w in wins[:nm]:
            pt.model(w[None])
            for l in range(L_):
                Kp = pt.cap[(l, 'K')].to(D64)
                q = pt.cap[(l, 'q')].to(D64).reshape(seq, Hq, Dh)
                qr = q * cosq + rotate_half(q) * sinq
                kr = pt.rot(Kp, cs).reshape(seq, KVH, Dh)
                k0 = Kp[0].reshape(KVH, Dh); m_ = mu[l].reshape(KVH, Dh)
                geo[l] += ((k0 * m_).sum(1) / (m_ * m_).sum(1)).numpy() / nm
                a0 = 0.0
                scs = {}
                for h in range(Hq):
                    gi = h // rep
                    sc = qr[:, h] @ kr[:, gi].T / math.sqrt(Dh)
                    sc = sc.masked_fill(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), float('-inf'))
                    a0 += float(torch.softmax(sc, -1)[1:, 0].mean()) / Hq
                att0[l] += a0 / nm
                for kind in ('plain', 'centered'):
                    _, _, m2, Pm = P32[kind][l]
                    Kh = m2 + (Kp - m2) @ Pm
                    krh = pt.rot(Kh, cs).reshape(seq, KVH, Dh)
                    e0s, ers = [], []
                    for h in range(Hq):
                        gi = h // rep
                        e = qr[:, h] @ (krh[:, gi] - kr[:, gi]).T / math.sqrt(Dh)
                        e0s.append(float(e[1:, 0].abs().mean()))
                        ers.append(float((e[1:, 1:].abs() * causal_rest).sum() / causal_rest.sum()))
                    err[(kind, 'err0')][l] += float(np.median(e0s)) / nm
                    err[(kind, 'errrest')][l] += float(np.median(ers)) / nm
            print(f'mechanism window done ({time.perf_counter()-T0:.0f}s)', flush=True)
    geo_med = np.median(geo, 1)
    res['mechanism'] = {'k0_coord_mu_per_layer_median': geo_med.tolist(), 'k0_coord_mu_per_layer_head': geo.tolist(),
                        'attn0_per_layer': att0.tolist(),
                        **{f'{k[0]}_{k[1]}_per_layer': v.tolist() for k, v in err.items()}}
    d = {}
    d['S1_layers_negative'] = int((geo_med < 0).sum()); d['S1'] = d['S1_layers_negative'] >= 20
    ppl = {k: v['ppl'] for k, v in res['runs'].items()}
    cmp_ = {}

    def pr(A, B):
        r = ppl_ratio(nllw[A], nllw[B], tok)
        return {k: r[k] for k in ('ratio', 'ci_lo', 'ci_hi', 'windows_won', 'windows', 'sign_p')}
    d['S2'] = d['S3'] = d['S5'] = True; s4 = 0
    for r in a.ranks:
        cmp_[f'centered_over_plain r={r}'] = pr(f'centered r={r}', f'plain r={r}')
        cmp_[f'centered_keep0_over_centered r={r}'] = pr(f'centered r={r} keep0', f'centered r={r}')
        cmp_[f'plain_keep0_over_plain r={r}'] = pr(f'plain r={r} keep0', f'plain r={r}')
        cmp_[f'centered_keep0_over_plain_keep0 r={r}'] = pr(f'centered r={r} keep0', f'plain r={r} keep0')
        d['S2'] &= ppl[f'centered r={r}'] > ppl[f'plain r={r}']
        d['S3'] &= ppl[f'centered r={r} keep0'] < ppl[f'centered r={r}']
        d['S5'] &= ppl[f'centered r={r} keep0'] <= ppl[f'plain r={r} keep0']
        s4 += ppl[f'plain r={r} keep0'] > ppl[f'plain r={r}']
    d['S4_ranks_plain_keep0_worse'] = int(s4); d['S4'] = s4 >= 2
    d['S6_layers_attn0_ge_0.3'] = int((att0 >= 0.3).sum())
    d['S6_attn0_mean'] = float(att0.mean())
    res['comparisons'] = cmp_
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('ppl:', {k: round(v, 3) for k, v in ppl.items()})
    print('decision:', d)


if __name__ == '__main__':
    main()
