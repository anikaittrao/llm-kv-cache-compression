"""C21 analysis (wave 4): per-KV-head basis (c21_rope_scope_head.json, K21/V43 per head) versus the per-layer basis (c11_rope_smollm2.json,
K64/V128 per layer), SmolLM2-135M, windows 0-7, P = 2048, both 192 stored numbers per token and layer. Rules PH1-PH3 are in the
docstring of kv_rope_port.py (C21 paragraph); writes c21_summary.json. Code check: dense NLL of both runs identical."""
import numpy as np
import w3_stats as W

CELLS = {'PH1': 'static / buffer 32', 'PH2': 'update / buffer 32', 'PH3': 'static / no buffer'}


def main():
    lay, head = W.load('c11_rope_smollm2.json'), W.load('c21_rope_scope_head.json')
    tok = lay['tokens_per_window']
    assert head['windows'] == lay['windows'] and head['tokens_per_window'] == tok
    out = {'code_check_dense_max_abs_nll_diff': float(np.max(np.abs(np.array(lay['nll_per_window']['dense']) - np.array(head['nll_per_window']['dense'])))),
           'stored_numbers_per_token_layer_head': {'layer': lay['config']['RK'] + lay['config']['RV'],
                                                    'head': 3 * (head['config']['RK'] // 3 + head['config']['RV'] // 3)},
           'basis_numbers_per_layer': {'layer': 192 * (64 + 128), 'head': 3 * 64 * (21 + 43)}, 'cells': {}}
    for rule, m in CELLS.items():
        a, b = head['nll_per_window'][m], lay['nll_per_window'][m]
        d = lay['nll_per_window']['dense']
        r = W.ppl_ratio(a, b, tok)
        out['cells'][m] = {'rule': rule, 'e_layer': W.excess(b, d, tok), 'e_head': W.excess(a, d, tok),
                           'e_layer_ci': W.boot(lambda s: W.e_of({'x': b, 'dense': d}, 'x', 'dense', s, tok), len(b)),
                           'e_head_ci': W.boot(lambda s: W.e_of({'x': a, 'dense': d}, 'x', 'dense', s, tok), len(a)),
                           'ppl_ratio_head_over_layer': {k: r[k] for k in ('ratio', 'ci_lo', 'ci_hi', 'windows_won', 'windows', 'sign_p')},
                           'per_layer_better': bool(r['ci_lo'] > 1), 'per_head_better': bool(r['ci_hi'] < 1)}
    out['verdict'] = {k: ('per-layer better' if c['per_layer_better'] else 'per-head better' if c['per_head_better'] else 'not resolved')
                      for k, c in out['cells'].items()}
    W.save('c21_summary.json', out)
    print(out)


if __name__ == '__main__':
    main()
