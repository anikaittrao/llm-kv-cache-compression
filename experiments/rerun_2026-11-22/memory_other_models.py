"""Proposition 2.2 of final_report evaluated for other model shapes (2026-11-23, gap item G1). Pure arithmetic, no runs.

Run:  .venv\\Scripts\\python.exe experiments\\rerun_2026-11-22\\memory_other_models.py   -> memory_other_models.json here.
Same formula as experiments/kv_memory.py (which hard-codes GPT-2's d = 768), per layer and sequence:
  M = sum over compressed kinds k of (d R_k + d + R_k) + n_G,  n_G = H_kv d_h^2 if keys are compressed and G_K is per sequence
  rho(T) = (R_K + R_V)/(2d) + [M + n_ex (2d - R_K - R_V)] / (2 d T),   T* = [M + n_ex (2d - R_K - R_V)] / (2d - R_K - R_V)
d = H_kv d_h (the key or value width of one token in one layer). n_ex = 1 (first token) + 32 (block buffer) = 33.
The GPT-2 row reproduces results_kv_memory.json (0.924 at T = 1024, T* = 868) as a check. These are computed values, not
measured runs. The 8B shape is Llama-3-8B-like (8 KV heads of size 128, half the numbers per token: R_K + R_V = 1024,
split 1/3 : 2/3 as for GPT-2).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def memory(H_kv, d_h, RK, RV, n_ex, Ts, per_seq_GK=True):
    d = H_kv * d_h
    M = sum(d * R + d + R for R in (RK, RV) if R < d) + (H_kv * d_h * d_h if (per_seq_GK and RK < d) else 0)
    gap = 2 * d - RK - RV
    out = {'d': d, 'R_K': RK, 'R_V': RV, 'n_ex': n_ex, 'M_numbers_per_layer': M, 'rho0': (RK + RV) / (2 * d),
           'T_star': (M + n_ex * gap) / gap, 'rho': {}}
    for T in Ts:
        out['rho'][str(T)] = (RK + RV) / (2 * d) + (M + n_ex * gap) / (2 * d * T)
    return out


res = {
    'GPT-2 small (check vs results_kv_memory.json)': memory(12, 64, 256, 512, 33, [1024, 2048, 8192]),
    'SmolLM2-135M, K64/V128, buffer 32': memory(3, 64, 64, 128, 33, [3072, 8192]),
    'SmolLM2-135M, K64/V128, no buffer (first token only)': memory(3, 64, 64, 128, 1, [3072, 8192]),
    '8B shape (8 KV heads x 128), K 1024/3 + V 2048/3 ~ K342/V682, buffer 32': memory(8, 128, 342, 682, 33, [4096, 8192, 32768]),
}
# offline keys-only basis with the 32-token buffer (no per-sequence side information), GPT-2, T = 1024
d = 768
res['GPT-2 offline keys only K256, buffer 32 (M = 0)'] = {'rho_1024': (256 + 768) / (2 * d) + 33 * (2 * d - 1024) / (2 * d * 1024)}
with open(os.path.join(HERE, 'memory_other_models.json'), 'w') as f:
    json.dump(res, f, indent=1)
for k, v in res.items():
    print(k, json.dumps(v))
