"""Step 5 mechanism check (2026-11-18): why does centering hurt per-head keys, and why does keeping the first key exact fix it?

Run:  .venv\\Scripts\\python.exe experiments\\kv_sink_check.py      (about 3 min)
Writes experiments/results_kv_sink_mech.json.

Same calibration as kv_adaptive.py (16 train windows, seed 0); per-head bases at rank r (default 32) of the keys:
plain (top eigenvectors of E[k k^T], mu = 0) and centered (top eigenvectors of the covariance, mu = mean key).
On the first --windows test windows (dense model) it measures per layer, median over heads:
  * attn_to_pos0  : average attention weight that queries put on position 0 (the "attention sink").
  * logit_err     : mean |q_t . (k_hat_s - k_s)| / sqrt(64), the error in the attention score (logit), for s = 0 and for s > 0
                    (all causal pairs s <= t, t >= 1).
  * k0 geometry   : ||k_0|| / RMS ||k_s||, and the component of (k_0 - mu) along the mean direction, relative to ||mu||
                    (-1 means k_0 has none of the mean). The centered basis need not contain the mean direction, so a
                    key far from mu along it is reconstructed badly; the plain basis contains it whenever the mean is large.
Added 2026-11-22 (audit patch B7; the old fields are unchanged):
  * k0_coord_mu        : k_0 . mu / (mu . mu) = k0_mean_comp + 1, the coordinate of k_0 ITSELF along the mean direction,
                         in units of ||mu|| (negative = k_0 lies on the other side of the origin from mu).
  * muhat_resid_<b>    : ||mu_hat (I - P_b)||, the part of the unit mean direction that basis b (plain / centered) misses.
  * Remark 5.1 check (centered basis): with k_0 - mu = c mu_hat + w (w orthogonal to mu_hat),
        ||k_0 - k_hat_0|| >= |c| ||mu_hat (I - P)|| - ||w (I - P)||;
    stored per head: rem51_c_abs = |c|, rem51_w_resid = ||w (I - P)||, rem51_err = ||k_0 - k_hat_0||,
    rem51_lower = |c| ||mu_hat(I-P)|| - ||w(I-P)|| (averaged over windows like every other field), and the
    per-window count of violations (must be 0) in 'rem51_violations'.
The default stays 2 test windows (the reported run); the audit's C7 uses --windows 16 with a new --out.
"""
import argparse, json, math, os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D64 = torch.float64


def load_wikitext(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download('Salesforce/wikitext', f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset')
    return '\n\n'.join(pq.read_table(f).column('text').to_pylist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_sink_mech.json'))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').eval()
    L, H, E = 12, 12, 768; d, seq = 64, 1024
    ids_test = tok(load_wikitext('test'), return_tensors='pt').input_ids[0]
    ids_train = tok(load_wikitext('train'), return_tensors='pt').input_ids[0]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, ids_train.numel() - seq, (16,), generator=g).tolist()

    cap = {}
    hooks = [model.transformer.h[l].attn.c_attn.register_forward_hook(
        lambda m, a_, o, l=l: cap.__setitem__(l, o[0].detach().to(D64))) for l in range(L)]
    S1 = [torch.zeros(E, dtype=D64) for _ in range(L)]; S2 = [torch.zeros(E, E, dtype=D64) for _ in range(L)]; n = 0
    with torch.inference_mode():
        for s in starts:
            model(ids_train[s:s + seq][None]); n += seq
            for l in range(L):
                K = cap[l][:, E:2 * E]; S1[l] += K.sum(0); S2[l] += K.T @ K
    mu = [S1[l] / n for l in range(L)]; M2 = [S2[l] / n for l in range(L)]
    C = [M2[l] - torch.outer(mu[l], mu[l]) for l in range(L)]

    def basis(Mat):
        w, U = torch.linalg.eigh(Mat); return U[:, -a.r:]

    P = {'plain': [[None] * H for _ in range(L)], 'centered': [[None] * H for _ in range(L)]}
    for l in range(L):
        for h in range(H):
            s = slice(h * d, (h + 1) * d)
            Up, Uc = basis(M2[l][s, s]), basis(C[l][s, s])
            P['plain'][l][h] = (torch.zeros(d, dtype=D64), Up @ Up.T)
            P['centered'][l][h] = (mu[l][s], Uc @ Uc.T)

    acc = {k: np.zeros((L, H)) for k in ('attn0', 'k0_norm_ratio', 'k0_mean_comp',
                                          'err0_plain', 'errrest_plain', 'err0_centered', 'errrest_centered',
                                          'k0_coord_mu', 'muhat_resid_plain', 'muhat_resid_centered',
                                          'rem51_c_abs', 'rem51_w_resid', 'rem51_err', 'rem51_lower')}
    rem51_viol = 0
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool), -1)  # pairs s < t (t >= 1); position 0 is column 0
    with torch.inference_mode():
        for wi in range(a.windows):
            w = ids_test[wi * seq:(wi + 1) * seq]
            out = model(w[None], output_attentions=True)
            for l in range(L):
                att = out.attentions[l][0].to(D64)  # (H, T, T)
                Q, K = cap[l][:, :E], cap[l][:, E:2 * E]
                for h in range(H):
                    s = slice(h * d, (h + 1) * d)
                    q, k = Q[:, s], K[:, s]
                    acc['attn0'][l, h] += float(att[h, 1:, 0].mean()) / a.windows
                    acc['k0_norm_ratio'][l, h] += float(k[0].norm() / k[1:].pow(2).sum(1).mean().sqrt()) / a.windows
                    m = mu[l][s]
                    acc['k0_mean_comp'][l, h] += float((k[0] - m) @ m / (m @ m)) / a.windows
                    acc['k0_coord_mu'][l, h] += float(k[0] @ m / (m @ m)) / a.windows
                    mh = m / m.norm()
                    for name in ('plain', 'centered'):
                        Pi_ = P[name][l][h][1]
                        acc[f'muhat_resid_{name}'][l, h] += float((mh - mh @ Pi_).norm()) / a.windows
                    Pc = P['centered'][l][h][1]
                    dk = k[0] - m
                    c_ = float(dk @ mh); wv = dk - c_ * mh
                    wres = float((wv - wv @ Pc).norm()); err_ = float((dk - dk @ Pc).norm())
                    lower = abs(c_) * float((mh - mh @ Pc).norm()) - wres
                    if err_ < lower - 1e-9 * max(1.0, abs(lower)):
                        rem51_viol += 1
                    acc['rem51_c_abs'][l, h] += abs(c_) / a.windows
                    acc['rem51_w_resid'][l, h] += wres / a.windows
                    acc['rem51_err'][l, h] += err_ / a.windows
                    acc['rem51_lower'][l, h] += lower / a.windows
                    for name in ('plain', 'centered'):
                        c, Pi = P[name][l][h]
                        kh = c + (k - c) @ Pi
                        E_ = ((q @ (kh - k).T).abs() / math.sqrt(d))  # (T queries, T keys)
                        acc[f'err0_{name}'][l, h] += float(E_[1:, 0].mean()) / a.windows
                        acc[f'errrest_{name}'][l, h] += float(E_[causal].mean()) / a.windows  # includes s=0 for t>=1, negligible share
    for hk in hooks:
        hk.remove()
    res = {'r': a.r, 'windows': a.windows, 'per_layer_median': {k: np.median(v, 1).tolist() for k, v in acc.items()},
           'per_layer_head': {k: v.tolist() for k, v in acc.items()},
           'overall_median': {k: float(np.median(v)) for k, v in acc.items()},
           'rem51_violations': rem51_viol}
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    for k, v in res['per_layer_median'].items():
        print(f'{k:18s}', ' '.join(f'{x:7.3f}' for x in v))
    print('overall median:', {k: round(v, 3) for k, v in res['overall_median'].items()})


if __name__ == '__main__':
    main()
