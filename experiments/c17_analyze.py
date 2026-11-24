"""C17 analysis (wave 3): AttSVD accumulating mode on GPT-2. Reads c17_kv_accum.json (kv_accum.py), writes c17_summary.json.
Rule fixed before any full run (e(x) = ppl(x)/dense - 1, windows 0-15):
  A1 weighting helps when new tokens are exact (main setting KV256/512 P=768):
     e(attn-late accumulating) <= 0.9 x e(unweighted accumulating).
Reported with paired bootstrap CIs: the excess of both, the ratio of excesses, and ppl(attn-late) / ppl(unweighted); the same
for the keys-only P = 384 setting (descriptive; the excess there is ~0 so the ratio of excesses is unstable: look at the ppl
ratio).
"""
import w3_stats as W


def main():
    r = W.load('c17_kv_accum.json')
    out = {}
    for name, s in r['settings'].items():
        nllw, tok = s['nll_per_window'], s['tokens_per_window']
        A, U = 'attn-late accumulating', 'unweighted accumulating'
        d = {'excess': {m: s['excess'][m] for m in s['excess'] if m != 'dense'},
             'excess_ci': {m: W.ex_ci(nllw, m, 'dense', tok) for m in (A, U)},
             'ratio_of_excess': W.ex_ratio_ci(nllw, A, U, 'dense', tok),
             'ppl_ratio_attnlate_over_unweighted': W.ppl_ratio_ci(nllw, A, U, tok)}
        d['A1_pass'] = bool(s['excess'][A] <= 0.9 * s['excess'][U])
        out[name] = d
        print(name, 'e unweighted %.4f%%  e attn-late %.4f%%  ratio %.3f  A1 %s' %
              (100 * s['excess'][U], 100 * s['excess'][A], s['excess'][A] / s['excess'][U], d['A1_pass']))
    out['A1_main_setting'] = out.get('KV256/512 P=768', {}).get('A1_pass')
    W.save('c17_summary.json', out)


if __name__ == '__main__':
    main()
