"""C20 (wave 4, 2026-11-24): why does keeping position 0's key exact (keep0) make the PLAIN per-head key basis WORSE?

Background (C7, kv_sink2.py): per-head keys, plain basis (mu = 0, top r eigenvectors of E[k k^T], all calibration positions), eval =
full 1024-token passes of test windows 0-7, perplexity: plain r=16 310.7 -> 317.3 with keep0, r=32 70.58 -> 73.25. keep0 only
changes how position 0 is STORED (the basis is identical), so the basis-fit explanation is excluded ('no0' is a different run).
Candidate explanation tested here, "common-mode cancellation": compression shifts the score of the sink key AND of the other
keys by correlated amounts. A softmax only sees score DIFFERENCES, so an error e0 on the sink partly cancels the error of the
other keys; storing k_0 exactly removes that cancellation. Per query t, head h (dense pass, plain basis rank r):
  e_s = q_t . (khat_s - k_s) / 8 ; a = dense attention; a0 = a_t0; ebar = sum_{s>=1} a_ts e_s / (1 - a0)
  without keep0 the sink-vs-rest shift is ebar - e0, with keep0 it is ebar; the first-order mass moved to/from the sink is
  proportional to a0 (1 - a0) times it, so  cancel = mean_{t>=1,h} a0 (1 - a0) (|ebar| - |ebar - e0|)  (> 0: keep0 hurts).
Also stored: attention-weighted mean |e_0| vs mean |e_s| (s>=1) and the correlation of e0 and ebar over queries.
Single-layer ablation: for every layer l, plain (r) compression of the keys of layer l only, with and without keep0 (other
layers dense), per-window NLL. Delta_l = NLL(keep0, layer l) - NLL(no keep0, layer l) in nats/token.
Rules fixed before the run (r = 16 is the primary rank, windows 0-7):
  H0 replication  : all-layer plain keep0 is worse than plain at r=16 AND r=32 (paired CI of ppl ratio keep0/plain, lower bound > 1).
  H1 cancellation : Spearman(Delta_l, cancel_l) over the 12 layers >= 0.6 AND sign(Delta_l) = sign(cancel_l) in >= 9 of 12 layers.
  H2 smaller sink error (descriptive threshold): attention-weighted mean |e_0| < attention-weighted mean |e_s| (s>=1), median over
     heads, in >= 9 of 12 layers (then keep0 buys little, so a harmful side effect is not outweighed).
  Verdict 'cancellation explains keep0 harm' needs H0 and H1.
Run:  .venv\\Scripts\\python.exe experiments\\kv_keep0_diag.py --threads 8 --out experiments\\rerun_2026-11-22\\c20_kv_keep0_diag.json
      smoke: --windows 1 --layers 2
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import Ctx, L, H, E, Dh, D64, ROOT, base_result, save_json, rss_mb, head_attention  # noqa: E402
from stats_utils import ppl_ratio  # noqa: E402


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--layers', type=int, default=L, help='smoke: only the first n layers in the single-layer part')
    ap.add_argument('--ranks', type=int, nargs='*', default=[16, 32])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c20_kv_keep0_diag.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    ctx = Ctx(a.threads)
    ctx.offline()
    n = ctx.calib['n']
    M2 = [ctx.calib['S2']['K'][l] / n for l in range(L)]
    _, wins = ctx.windows(a.windows)
    tok = wins[0].numel() - 1
    print(f'calibration ready ({time.perf_counter()-T0:.0f}s)', flush=True)

    def plain_P(r):
        out = []
        for l in range(L):
            Pm = torch.zeros(E, E, dtype=D64)
            for h in range(H):
                s = slice(h * Dh, (h + 1) * Dh)
                U = torch.linalg.eigh(M2[l][s, s])[1][:, -r:]
                Pm[s, s] = U @ U.T
            out.append(Pm)
        return out
    ident = (torch.zeros(E), torch.eye(E))

    def run(Pm, layers, keep0):
        ctx.mode['proj'] = {'K': [(torch.zeros(E), Pm[l].float()) if l in layers else ident for l in range(L)]} if layers else None
        ctx.mode['keep0'] = ('K',) if keep0 else ()
        nll = []
        with torch.inference_mode():
            for w in wins:
                nll.append(float(ctx.model(w[None], labels=w[None]).loss) * tok)
        ctx.mode.update({'proj': None, 'keep0': ()})
        return nll

    res = base_result(a, {'tokens_per_window': tok, 'runs': {}})
    dense = run(None, set(), False)
    res['dense_nll'] = dense
    res['dense_ppl'] = math.exp(sum(dense) / (tok * len(dense)))
    d = {}
    for r in a.ranks:
        Pm = plain_P(r)
        al = run(Pm, set(range(L)), False); ak = run(Pm, set(range(L)), True)
        res['runs'][f'all r={r}'] = {'plain': al, 'plain_keep0': ak,
                                     'ppl_plain': math.exp(sum(al) / (tok * len(al))), 'ppl_plain_keep0': math.exp(sum(ak) / (tok * len(ak))),
                                     'ratio_keep0_over_plain': ppl_ratio(ak, al, tok)}
        print(f"r={r}: plain {res['runs'][f'all r={r}']['ppl_plain']:.3f} keep0 {res['runs'][f'all r={r}']['ppl_plain_keep0']:.3f} ({time.perf_counter()-T0:.0f}s)", flush=True)
        d[f'H0_r{r}_keep0_worse'] = bool(res['runs'][f'all r={r}']['ratio_keep0_over_plain']['ci_lo'] > 1)
        if r == a.ranks[0]:
            Dl, nl, nk = [], [], []
            for l in range(a.layers):
                x = run(Pm, {l}, False); y = run(Pm, {l}, True)
                nl.append(x); nk.append(y)
                Dl.append((np.mean(y) - np.mean(x)) / tok)
                print(f'  layer {l}: delta keep0 - plain = {Dl[-1]:+.5f} nats/token ({time.perf_counter()-T0:.0f}s)', flush=True)
            res['single_layer'] = {'r': r, 'delta_nats_per_token': Dl, 'nll_plain': nl, 'nll_keep0': nk}
            # mechanism
            acc = {k: np.zeros((L, H)) for k in ('cancel', 'e0', 'erest', 'corr_n', 'a0')}
            corr_lists = {(l, h): ([], []) for l in range(L) for h in range(H)}
            ctx.mode['proj'] = None
            with torch.inference_mode():
                for w in wins:
                    ctx.model(w[None])
                    for l in range(L):
                        Xl = ctx.cap[l].to(D64)
                        q, K = Xl[:, :E], Xl[:, E:2 * E]
                        Kh = K @ Pm[l]
                        att = head_attention(q, K, 0)                        # (H, T, T)
                        T_ = q.shape[0]
                        qh = q.reshape(T_, H, Dh).transpose(0, 1)
                        dk = (Kh - K).reshape(T_, H, Dh).transpose(0, 1)
                        e = qh @ dk.transpose(1, 2) / 8.0                      # (H, T, T)
                        a0 = att[:, 1:, 0]
                        rest = att[:, 1:, 1:] * torch.tril(torch.ones(T_ - 1, T_ - 1, dtype=D64))[None]
                        wr = rest.sum(-1).clamp_min(1e-12)
                        ebar = (rest * e[:, 1:, 1:]).sum(-1) / wr
                        e0 = e[:, 1:, 0]
                        c = (a0 * (1 - a0) * (ebar.abs() - (ebar - e0).abs())).mean(1)
                        acc['cancel'][l] += c.numpy() / len(wins)
                        acc['e0'][l] += (a0 * 0 + e0.abs()).mean(1).numpy() / len(wins) * 0 + ((att[:, 1:, 0] * e0.abs()).sum(1) / a0.sum(1)).numpy() / len(wins)
                        acc['erest'][l] += ((rest * e[:, 1:, 1:].abs()).sum((1, 2)) / rest.sum((1, 2))).numpy() / len(wins)
                        acc['a0'][l] += a0.mean(1).numpy() / len(wins)
                        for h in range(H):
                            corr_lists[(l, h)][0].append(e0[h].numpy()); corr_lists[(l, h)][1].append(ebar[h].numpy())
            corr = np.zeros((L, H))
            for (l, h), (u, v) in corr_lists.items():
                corr[l, h] = np.corrcoef(np.concatenate(u), np.concatenate(v))[0, 1]
            res['mechanism_r'] = r
            res['mechanism_per_layer_head'] = {k: v.tolist() for k, v in acc.items()}
            res['mechanism_per_layer_head']['corr_e0_ebar'] = corr.tolist()
            cancel_l = acc['cancel'].mean(1)[:a.layers]
            res['mechanism_per_layer'] = {'cancel': cancel_l.tolist(), 'e0_weighted_abs_median_head': np.median(acc['e0'], 1).tolist(),
                                          'erest_weighted_abs_median_head': np.median(acc['erest'], 1).tolist(),
                                          'corr_e0_ebar_median_head': np.median(corr, 1).tolist(), 'attn_pos0_mean_head': acc['a0'].mean(1).tolist()}
            d['H1_spearman'] = spearman(np.array(Dl), cancel_l)
            d['H1_sign_agree_layers'] = int(sum(np.sign(Dl[i]) == np.sign(cancel_l[i]) for i in range(a.layers)))
            d['H1'] = bool(d['H1_spearman'] >= 0.6 and d['H1_sign_agree_layers'] >= 9)
            d['H1_layers_keep0_harmful'] = int(sum(x > 0 for x in Dl))
            d['H2_layers_e0_smaller'] = int(sum(np.median(acc['e0'], 1)[i] < np.median(acc['erest'], 1)[i] for i in range(a.layers)))
            d['H2'] = d['H2_layers_e0_smaller'] >= 9
            print('mechanism done', {k: v for k, v in d.items()}, flush=True)
    d['verdict_cancellation_explains'] = bool(all(d.get(f'H0_r{r}_keep0_worse', False) for r in a.ranks) and d.get('H1', False))
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    print('decision:', d)


if __name__ == '__main__':
    main()
