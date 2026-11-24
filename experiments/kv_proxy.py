"""C9 (audit 2026-11-22): do the attention-output bounds rank the bases better than a trivial error proxy?

Run:  .venv\\Scripts\\python.exe experiments\\kv_proxy.py --out experiments\\rerun_2026-11-22\\c9_kv_proxy.json
            (about 20 min, 16 windows, every 8th continuation query; smoke 2 layers x 1 window: 11 s + 80 s calibration)   smoke test: --windows 1 --layers 2

Setting: identical to kv_bound.py / kv_bound2.py (online setting of kv_online.py, P = 768, bases offline / online /
rsvd q = 0, 1, 2, half and third of the cache, dense pass of each window, every layer compressed one at a time, first
token exact). The 10 settings are ranked by perplexity (results_kv_online.json). The reported bounds ranked them with
Spearman 0.92 (levels 2, 2') and 0.99 (level V, = the actual one-layer error).
Trivial proxies (summed over layers and, where per head, heads):
  res_all   : mean over all tokens of ||e^K_i|| + ||e^V_i||, e = weighted residual rows (metric G^{1/2}, as Cor 7.3).
  res_cont  : the same over the continuation tokens only.
  res_sq    : mean squared weighted residual (K + V), all tokens.
  attn_val  : sum_i a_i ||(v^_i - v_i) W_O^h|| with the exact attention a (the value part of level 1').
  attn_score: sum_i a_i |delta_i|, delta = score error (the key part, first order).
and, as a check, err = the actual one-layer error ||(o^ - o) W_O^h|| (reported Spearman 0.988).
Rule (descriptive, fixed before the run): a proxy with Spearman >= 0.98 means the bounds add no ranking information
beyond it; report all Spearman values and whether the offline-vs-online order at half cache is right.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_online import load_wikitext, psd_pow, qblock, rsvd_right, ROOT, D64, L, H, E, Dh  # noqa: E402
from kv_bound import SETTINGS, METHODS, PPL_KEY, spearman  # noqa: E402

PROXIES = ('err', 'res_all', 'res_cont', 'res_sq', 'attn_val', 'attn_score')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--layers', type=int, default=L)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--P', type=int, default=768)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c9_kv_proxy.json'))
    ap.add_argument('--online_json', default=os.path.join(ROOT, 'experiments', 'results_kv_online.json'))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    T0 = time.perf_counter()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    seq, P = 1024, a.P
    WO = [model.transformer.h[l].attn.c_proj.weight.detach().to(D64) for l in range(L)]
    TV = [(psd_pow(WO[l] @ WO[l].T, 0.5), psd_pow(WO[l] @ WO[l].T, -0.5)) for l in range(L)]
    WOh = [[WO[l][h * Dh:(h + 1) * Dh] for h in range(H)] for l in range(L)]
    cap = {}
    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(lambda m, a_, o, l=l: cap.__setitem__(l, o[0].detach()))
             for l in range(L)]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids_train.numel() - seq, (16,), generator=g).tolist()
    S1 = {k: [torch.zeros(E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    S2 = {k: [torch.zeros(E, E, dtype=D64) for _ in range(L)] for k in 'QKV'}
    with torch.inference_mode():
        for s in starts:
            model(ids_train[s:s + seq][None])
            for l in range(L):
                X = cap[l].to(D64)
                for i, k in enumerate('QKV'):
                    Xi = X[:, i * E:(i + 1) * E]; S1[k][l] += Xi.sum(0); S2[k][l] += Xi.T @ Xi
    n = 16 * seq
    off = {}
    for l in range(L):
        muK, muV = S1['K'][l] / n, S1['V'][l] / n
        CK = S2['K'][l] / n - torch.outer(muK, muK); CV = S2['V'][l] / n - torch.outer(muV, muV)
        GK = qblock(S2['Q'][l] / n); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
        off[l] = {'K': (muK, TK, torch.linalg.eigh(TK[0] @ CK @ TK[0])[1].flip(1)),
                  'V': (muV, TV[l], torch.linalg.eigh(TV[l][0] @ CV @ TV[l][0])[1].flip(1))}
    print(f'offline bases ready ({time.perf_counter()-T0:.0f}s)', flush=True)
    tri = torch.ones(seq, seq, dtype=torch.bool).tril()
    qidx = torch.arange(P, seq, a.stride)
    acc = {}
    for wi in range(a.windows):
        w = ids_test[wi * seq:(wi + 1) * seq]
        with torch.inference_mode():
            model(w[None])
        for l in range(a.layers):
            X = cap[l].to(D64)
            Xq, Xk, Xv = X[:, :E], X[:, E:2 * E], X[:, 2 * E:]
            q = Xq[qidx].reshape(len(qidx), H, Dh).transpose(0, 1)
            k = Xk.reshape(seq, H, Dh).transpose(0, 1)
            v = Xv.reshape(seq, H, Dh).transpose(0, 1)
            mask = tri[qidx]
            s = ((q @ k.transpose(1, 2)) / 8.0).masked_fill(~mask, float('-inf'))
            att = torch.softmax(s, -1)
            vW = torch.stack([v[h] @ WOh[l][h] for h in range(H)])
            o = att @ vW
            pq, pk, pv = Xq[1:P], Xk[1:P], Xv[1:P]
            GKon = qblock(pq.T @ pq / pq.shape[0]); TKon = (psd_pow(GKon, 0.5), psd_pow(GKon, -0.5))
            onl = {}
            for kind, Xr, T in (('K', pk, TKon), ('V', pv, TV[l])):
                mu = Xr.mean(0); A = (Xr - mu) @ T[0]
                onl[kind] = (mu, T, torch.linalg.svd(A, full_matrices=False)[2].T, A)
            for sname, RK, RV in SETTINGS:
                for m in METHODS:
                    xh, en = {}, {}
                    for kind, R, Xfull in (('K', RK, Xk), ('V', RV, Xv)):
                        if m == 'offline':
                            mu, T, U = off[l][kind]; Ur = U[:, :R]
                        else:
                            mu, T, U, A = onl[kind]
                            if m == 'online':
                                Ur = U[:, :R]
                            else:
                                qq = int(m[-1])
                                gen = torch.Generator().manual_seed(1000 * wi + 10 * l + (0 if kind == 'K' else 1) + 100000 * qq)
                                Ur = rsvd_right(A, R, a.p, qq, gen)[0]
                        Ew = (Xfull - mu) @ T[0]
                        e = Ew - (Ew @ Ur) @ Ur.T
                        e[0] = 0
                        xh[kind] = Xfull - e @ T[1]
                        en[kind] = e.norm(dim=1)
                    kh = xh['K'].reshape(seq, H, Dh).transpose(0, 1)
                    vh = xh['V'].reshape(seq, H, Dh).transpose(0, 1)
                    sh = ((q @ kh.transpose(1, 2)) / 8.0).masked_fill(~mask, float('-inf'))
                    ah = torch.softmax(sh, -1)
                    vWh = torch.stack([vh[h] @ WOh[l][h] for h in range(H)])
                    err = (ah @ vWh - o).norm(dim=-1)
                    eps = (vWh - vW).norm(dim=-1)
                    delta = (sh - s).masked_fill(~mask, 0.0)
                    r = acc.setdefault((sname, m), {p_: 0.0 for p_ in PROXIES})
                    r['err'] += float(err.mean(1).sum())
                    r['attn_val'] += float((att * eps[:, None, :]).sum(-1).mean(1).sum())
                    r['attn_score'] += float((att * delta.abs()).sum(-1).mean(1).sum())
                    r['res_all'] += float(en['K'].mean() + en['V'].mean())
                    r['res_cont'] += float(en['K'][P:].mean() + en['V'][P:].mean())
                    r['res_sq'] += float(en['K'].pow(2).mean() + en['V'].pow(2).mean())
        print(f'window {wi+1}/{a.windows} done ({time.perf_counter()-T0:.0f}s)', flush=True)
    for h_ in hooks:
        h_.remove()
    with open(a.online_json) as f:
        ppl = json.load(f)['settings']
    rows = [{'setting': sn, 'method': m, 'ppl': ppl[PPL_KEY[sn]]['ppl'][m], **{p_: v_ / a.windows for p_, v_ in r.items()}}
            for (sn, m), r in acc.items()]
    y = [r_['ppl'] for r_ in rows]
    sp = {p_: spearman([r_[p_] for r_ in rows], y) for p_ in PROXIES}
    half = {r_['method']: r_ for r_ in rows if r_['setting'] == 'half'}
    order_ok = {p_: (half['offline'][p_] < half['online'][p_]) == (half['offline']['ppl'] < half['online']['ppl']) for p_ in PROXIES}
    res = {'config': vars(a), 'rows': rows, 'spearman': sp, 'offline_vs_online_half_order_right': order_ok,
           'reported_bound_spearman': {'level 2 (kv_bound)': 0.9151515151515152, "level 2' (kv_bound2)": 0.9151515151515152,
                                       'level V (kv_bound3)': 0.9878787878787878, 'actual err': 0.9878787878787878},
           'decision': {'proxies_with_spearman_ge_0.98': [p_ for p_ in PROXIES if p_ != 'err' and sp[p_] >= 0.98]},
           'total_runtime_s': time.perf_counter() - T0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('spearman:', {k: round(v, 3) for k, v in sp.items()})
    print('offline vs online order right:', order_ok)
    print('decision:', res['decision'])


if __name__ == '__main__':
    main()
