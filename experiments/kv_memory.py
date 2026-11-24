"""C0 (audit 2026-11-22): memory of a factored KV cache including the per-sequence basis (Prop 2.2 of the audit plan).

Run:  .venv\\Scripts\\python.exe experiments\\kv_memory.py        (seconds; no model, no data)
Writes experiments/results_kv_memory.json. Pure arithmetic, plus a cross-check that counts the arrays the experiment
code actually builds (kv_update.py / kv_buffer.py state: mu, G^{1/2}, G^{-1/2}, U, S) for one layer.

Counting (numbers stored per layer and per sequence; every number has the same width, so ratios are byte ratios):
  full cache            : 2 d T                                  (keys + values, d = 768)
  factored cache        : T (R_K + R_V)  codes                    (R = d for a kind that is not compressed)
                        + n_ex (2 d - R_K - R_V)                  (n_ex tokens stored exactly instead of as codes:
                                                                   the first token, plus a buffer / exact-row stash)
                        + M                                       (the basis and its side information)
  M = d (R_K + R_V)  [U_K, U_V]  +  d n_c  [means, n_c = number of compressed kinds]  +  (R_K + R_V)  [singular values S,
      needed by the update]  +  n_G  [n_G = H d_h^2 for the block-diagonal G_K^{1/2} if the key weight is per sequence
      (the online methods use the prompt's own queries); 0 if G_K is shared. G_V = W_O W_O^T is a model constant.]
  rho(T) = [T (R_K + R_V) + n_ex (2d - R_K - R_V) + M] / (2 d T)  =  rho_0 + [M + n_ex (2d - R_K - R_V)] / (2 d T),
  rho_0 = (R_K + R_V) / (2d);  break-even length T* (rho = 1):  T* = [M + n_ex (2d - R_K - R_V)] / (2d - R_K - R_V).
  Offline (calibration) bases are shared by all sequences: M is a one-time model constant, not per sequence (reported
  separately as 'model_constant_numbers', for all 12 layers).
Assumptions stated in the JSON:
  * Only one of G_K^{1/2}, G_K^{-1/2} is counted (the other follows per head from a 64 x 64 inverse); the variant
    'n_G both halves' counts both. Folding G^{1/2} into U (storing G^{1/2} U and U^T G^{-1/2}) needs 2 d R numbers
    instead of d R + n_G, which is more for every R > H d_h^2 / d = 64: it does not save memory ('folded' variant).
  * 'update / no buffer' still keeps the block's exact rows for the SVD (kv_buffer.py docstring), so its memory equals
    the buffer-32 case; 'static / no buffer' keeps no exact rows besides token 0.
  * The experiment code stores G^{1/2} and G^{-1/2} as dense 768 x 768 matrices (simulation convenience); the minimal
    count above uses the block-diagonal nonzeros. 'code_count' reports both.
  * Brand every U with all rows since the last update ('brand every 64/128', kv_cheapupdate.py) keeps up to U exact rows.
"""
import json, math, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from kv_online import psd_pow, qblock  # noqa: E402

d, H, Dh, L = 768, 12, 64, 12
TS = [1024, 2048, 8192, None]  # None = T -> infinity


def M_numbers(RK, RV, per_seq_G=True, nG_both=False, folded=False, per_sequence_basis=True):
    """Side information per layer and sequence (numbers)."""
    if not per_sequence_basis:
        return 0
    kinds = [R for R in (RK, RV) if R < d]
    nG = (2 if nG_both else 1) * H * Dh * Dh if (per_seq_G and RK < d) else 0
    if folded:  # store G^{1/2}U and U^T G^{-1/2} for keys (2 d R_K), U_V for values (G_V model constant)
        return (2 * d * RK if RK < d else 0) + (d * RV if RV < d else 0) + d * len(kinds) + sum(kinds)
    return d * sum(kinds) + d * len(kinds) + sum(kinds) + nG


def rho(T, RK, RV, n_ex, M):
    R = RK + RV
    if T is None:
        return R / (2 * d)
    return (T * R + n_ex * (2 * d - R) + M) / (2 * d * T)


def t_star(RK, RV, n_ex, M):
    gap = 2 * d - RK - RV
    return (M + n_ex * gap) / gap if gap > 0 else math.inf


def code_count(P=768, RK=256, RV=512, seed=0):
    """Build the per-layer state exactly as kv_update.py / kv_buffer.py do (random rows of the right shape) and count."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(P, 3 * d, generator=g, dtype=torch.float64)
    WO = torch.randn(d, d, generator=g, dtype=torch.float64)
    TV = (psd_pow(WO @ WO.T, 0.5), psd_pow(WO @ WO.T, -0.5))
    Xq = X[1:, :d]
    GK = qblock(Xq.T @ Xq / Xq.shape[0]); TK = (psd_pow(GK, 0.5), psd_pow(GK, -0.5))
    out = {}
    for kind, R, offs, T in (('K', RK, d, TK), ('V', RV, 2 * d, TV)):
        Xr = X[1:, offs:offs + d]
        mu = Xr.mean(0)
        _, s, Vh = torch.linalg.svd((Xr - mu) @ T[0], full_matrices=False)
        U, S = Vh[:R].T, s[:R]
        codes = (X[1:, offs:offs + d] - mu) @ T[0] @ U  # what a real factored cache would store per token
        out[kind] = {'mu': mu.numel(), 'U': U.numel(), 'S': S.numel(), 'T_dense_both': T[0].numel() + T[1].numel(),
                     'T_half_nonzeros': int((T[0].abs() > 0).sum()), 'codes_per_token': codes.shape[1]}
    minimal = (out['K']['mu'] + out['K']['U'] + out['K']['S'] + out['K']['T_half_nonzeros']
               + out['V']['mu'] + out['V']['U'] + out['V']['S'])  # G_V^{1/2} is a model constant
    as_implemented = minimal - out['K']['T_half_nonzeros'] + out['K']['T_dense_both'] + out['V']['T_dense_both']
    return {'arrays': out, 'minimal_side_info': minimal, 'as_implemented_side_info_incl_dense_G_matrices': as_implemented,
            'formula_M': M_numbers(RK, RV, per_seq_G=True), 'formula_matches_count': minimal == M_numbers(RK, RV, True)}


def main():
    settings = [  # (name, R_K, R_V, n_ex, per-sequence basis?, per-sequence G_K?, note)
        ('offline per-layer K256/V512 (Sec 5.5), no exact token', 256, 512, 0, False, False, 'shared calibration basis'),
        ('offline per-layer K256/V512, first token exact', 256, 512, 1, False, False, 'keep0 variant'),
        ('offline K256/V512 + buffer 32 (Sec 8 offline rows)', 256, 512, 33, False, False, ''),
        ('online/static K256/V512, no buffer (Sec 6)', 256, 512, 1, True, True, 'prompt basis, prompt G_K'),
        ('online/static K256/V512, buffer 32', 256, 512, 33, True, True, ''),
        ('update K256/V512, buffer 32 (Sec 8 main)', 256, 512, 33, True, True, 'update/no buffer has the same memory'),
        ('update K256/V512, buffer 32, G_K shared', 256, 512, 33, True, False, 'variant: calibration G_K'),
        ('online/static K256/V512, no buffer, G_K shared', 256, 512, 1, True, False, 'variant'),
        ('brand every 64 (all rows) K256/V512', 256, 512, 1 + 64, True, True, 'exact-row stash up to 64'),
        ('brand every 128 (all rows) K256/V512', 256, 512, 1 + 128, True, True, 'exact-row stash up to 128'),
        ('online third K128/V384, no buffer', 128, 384, 1, True, True, ''),
        ('online third K128/V384, buffer 32', 128, 384, 33, True, True, ''),
        ('offline third K128/V384, no buffer', 128, 384, 1, False, False, ''),
        ('online keys only K256, no buffer (P=384/512/768)', 256, d, 1, True, True, 'values full'),
        ('online keys only K256, buffer 32', 256, d, 33, True, True, ''),
        ('offline keys only K256, no buffer', 256, d, 1, False, False, ''),
        ('offline per-head keys r=32 (Option 1), keep0', 12 * 32, d, 1, False, False, 'per-head codes 32/64'),
        ('offline groups g (any g) half cache', 256, 512, 0, False, False, 'arithmetic differs, memory does not'),
    ]
    res = {'description': __doc__.split('Counting')[0].strip(), 'd': d, 'H': H, 'd_h': Dh, 'layers': L,
           'T_values': ['1024', '2048', '8192', 'inf'], 'settings': {}}
    for name, RK, RV, nex, ps, pg, note in settings:
        M = M_numbers(RK, RV, per_seq_G=pg, per_sequence_basis=ps)
        row = {'R_K': RK, 'R_V': RV, 'n_exact': nex, 'per_sequence_basis': ps, 'per_sequence_G_K': pg, 'note': note,
               'M_numbers_per_layer': M, 'rho0': (RK + RV) / (2 * d),
               'rho': {('inf' if T is None else str(T)): rho(T, RK, RV, nex, M) for T in TS},
               'T_star': t_star(RK, RV, nex, M)}
        if ps:
            Mb = M_numbers(RK, RV, per_seq_G=pg, nG_both=True)
            Mf = M_numbers(RK, RV, per_seq_G=pg, folded=True)
            row['variant_nG_both_halves'] = {'M': Mb, 'rho_1024': rho(1024, RK, RV, nex, Mb), 'T_star': t_star(RK, RV, nex, Mb)}
            row['variant_folded'] = {'M': Mf, 'rho_1024': rho(1024, RK, RV, nex, Mf), 'T_star': t_star(RK, RV, nex, Mf)}
        else:
            kinds = [R for R in (RK, RV) if R < d]
            row['model_constant_numbers'] = L * (d * sum(kinds) + d * len(kinds) + (H * Dh * Dh if RK < d else 0))
        # memory per sequence for all 12 layers in fp16 (2 bytes), at T = 1024
        row['MB_fp16_12_layers_T1024'] = {'full': 2 * d * 1024 * L * 2 / 2 ** 20,
                                          'factored': rho(1024, RK, RV, nex, M) * 2 * d * 1024 * L * 2 / 2 ** 20}
        res['settings'][name] = row
    res['code_count_main_K256_V512_P768'] = code_count()
    res['code_count_third_K128_V384_P768'] = code_count(RK=128, RV=384)
    # headline numbers for the text
    S = res['settings']
    res['headline'] = {
        'half_cache_online_T1024': S['online/static K256/V512, no buffer (Sec 6)']['rho']['1024'],
        'half_cache_online_buffer32_T1024': S['online/static K256/V512, buffer 32']['rho']['1024'],
        'half_cache_T_star': S['online/static K256/V512, no buffer (Sec 6)']['T_star'],
        'half_cache_buffer32_T_star': S['online/static K256/V512, buffer 32']['T_star'],
        'half_cache_G_K_shared_T1024': S['online/static K256/V512, no buffer, G_K shared']['rho']['1024'],
        'third_online_T1024': S['online third K128/V384, no buffer']['rho']['1024'],
        'keys_only_online_T1024': S['online keys only K256, no buffer (P=384/512/768)']['rho']['1024'],
        'offline_half_buffer32_T1024': S['offline K256/V512 + buffer 32 (Sec 8 offline rows)']['rho']['1024']}
    out = os.path.join(ROOT, 'experiments', 'results_kv_memory.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"{'setting':58s} {'M/layer':>8s} {'T=1024':>7s} {'2048':>7s} {'8192':>7s} {'inf':>6s} {'T*':>7s}")
    for name, r in S.items():
        print(f"{name:58s} {r['M_numbers_per_layer']:8d} " + ' '.join(f"{r['rho'][k]:7.3f}" for k in ('1024', '2048', '8192')) +
              f" {r['rho']['inf']:6.3f} {r['T_star']:7.0f}")
    print('code count (main):', json.dumps(res['code_count_main_K256_V512_P768'], indent=1))
    print('headline:', {k: round(v, 4) for k, v in res['headline'].items()})


if __name__ == '__main__':
    main()
