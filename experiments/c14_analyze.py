"""C14 analysis (wave 3): why SmolLM2 differs from GPT-2. Reads the three C14 runs and writes
experiments/rerun_2026-11-22/c14_summary.json. Rules fixed before any full run (e(x) = ppl(x)/dense - 1, all windows of the run):

Inputs (written by run_wave3.ps1):
  c14a_smollm2_gpt2geom.json   kv_rope_port.py --P 768 --seq 1024 --RK 64 --RV 128, 16 windows, 9 methods ("GPT-2 geometry":
                               same P and continuation as GPT-2, R/n = 0.083 / 0.167)
  c14b_smollm2_plainK.json     kv_rope_port.py --kweight plain (G_K = I), P 2048, seq 3072, 8 windows (offset 0), 9 methods
  c14c_kv_lowrn.json           kv_lowrn.py: GPT-2 K64/V128, P 768, B 32, 16 windows
  references: c13_summary.json (SmolLM2 main geometry P 2048, query-weighted G_K, 16 windows 0-15 pooled) and
  c11_rope_smollm2.json (the same 7 old methods on windows 0-7, query-weighted G_K).

  L1 (GPT-2 at low R/n; c14c): e(attn-late / buffer 32) <= 0.9 x e(static / buffer 32)   (weighting starts helping).
  G1 (SmolLM2 at GPT-2 geometry; c14a vs the main geometry c13): EITHER e(attn-late / buffer 32) > 0.9 x e(static / buffer 32)
     (the weighting gain disappears) OR the update's relative gain over static shrinks: r_geo = e(update / buffer 32) /
     e(static / buffer 32) <= 0.75 x r_main, with r_main = the same ratio at the main geometry (c13_summary.json, 16 windows).
     G1 is true if either part holds ("the SmolLM2 differences are geometry, not architecture").
  K1 (plain key weight; c14b vs c11 windows 0-7): W3r changes verdict, i.e. [e(attn-late / buffer 32) <= 0.9 x
     e(static / buffer 32)] differs between plain G_K (c14b) and query-weighted G_K (c11, same 8 windows), OR the attn-late
     gain g = 1 - e(attn-late / buffer 32) / e(static / buffer 32) with plain G_K is at most half of that with the query-weighted
     G_K (g_plain <= 0.5 g_query). K1 true means 'the weighting gain depends on the approximate query weight'. (Brief wording:
     'weighting compensates for the approximate query weight'; if plain G_K gave a LARGER gain, K1 is false and the
     descriptive field 'gain_plain_over_query' > 1 says so.)
Descriptive: every excess with a paired bootstrap CI, ratios e(.)/e(.) with CIs, the C14 geometry contrast
(c14a vs c14c: same model-independent R/n and P).
"""
import numpy as np
import w3_stats as W


def summarize(res, label, pairs):
    nllw, tok = res['nll_per_window'], res['tokens_per_window']
    out = {'windows': res['windows'], 'excess': {}, 'excess_ci': {}, 'ratios': {}}
    for m in nllw:
        if m == 'dense':
            continue
        out['excess'][m] = res['excess'][m]
        out['excess_ci'][m] = W.ex_ci(nllw, m, 'dense', tok)
    for A, Bm in pairs:
        if A in nllw and Bm in nllw:
            out['ratios'][f'e({A}) / e({Bm})'] = W.ex_ratio_ci(nllw, A, Bm, 'dense', tok)
            out['ratios'][f'ppl({A}) / ppl({Bm})'] = W.ppl_ratio_ci(nllw, A, Bm, tok)
    return out


PAIRS = [('attn-late / buffer 32', 'static / buffer 32'), ('update / buffer 32', 'static / buffer 32'),
         ('attn-late / no buffer', 'static / no buffer'), ('update / no buffer', 'static / no buffer'),
         ('attn-late+update / buffer 32', 'update / buffer 32'), ('attn-late+update / buffer 32', 'attn-late / buffer 32')]


def main():
    out = {}
    # ---- L1
    try:
        c = W.load('c14c_kv_lowrn.json')
        e = c['excess']
        s = summarize(c, 'c14c', PAIRS)
        s['L1_attnlate_buf_le_0.9_static_buf'] = e['attn-late / buffer 32'] <= 0.9 * e['static / buffer 32']
        s['L1_ratio'] = e['attn-late / buffer 32'] / e['static / buffer 32']
        s['update_over_static_buf'] = e['update / buffer 32'] / e['static / buffer 32']
        out['c14c_gpt2_lowrn'] = s
        print('L1', s['L1_attnlate_buf_le_0.9_static_buf'], 'ratio', round(s['L1_ratio'], 4), {k: round(100 * v, 2) for k, v in e.items()})
    except FileNotFoundError as ex:
        print('missing', ex)
    # ---- G1
    try:
        a = W.load('c14a_smollm2_gpt2geom.json')
        ref = W.load('c13_summary.json')
        e = a['excess']
        s = summarize(a, 'c14a', PAIRS)
        r_main = ref['excess']['update / buffer 32'] / ref['excess']['static / buffer 32']
        r_geo = e['update / buffer 32'] / e['static / buffer 32']
        w_geo = e['attn-late / buffer 32'] / e['static / buffer 32']
        w_main = ref['excess']['attn-late / buffer 32'] / ref['excess']['static / buffer 32']
        s.update({'r_geo_update_over_static_buf': r_geo, 'r_main_update_over_static_buf': r_main,
                  'w_geo_attnlate_over_static_buf': w_geo, 'w_main_attnlate_over_static_buf': w_main,
                  'G1_part1_attnlate_gain_gone': e['attn-late / buffer 32'] > 0.9 * e['static / buffer 32'],
                  'G1_part2_update_gain_grows': r_geo <= 0.75 * r_main})
        s['G1'] = bool(s['G1_part1_attnlate_gain_gone'] or s['G1_part2_update_gain_grows'])
        # bootstrap CI of r_geo / r_main (main geometry fixed at its point value)
        nllw, tok = a['nll_per_window'], a['tokens_per_window']
        s['r_geo_ci'] = W.ex_ratio_ci(nllw, 'update / buffer 32', 'static / buffer 32', 'dense', tok)
        s['w_geo_ci'] = W.ex_ratio_ci(nllw, 'attn-late / buffer 32', 'static / buffer 32', 'dense', tok)
        s['main_geometry_excess_c13'] = {k: ref['excess'][k] for k in ref['excess']}
        out['c14a_smollm2_gpt2geom'] = s
        print('G1', s['G1'], 'part1', s['G1_part1_attnlate_gain_gone'], 'w_geo', round(w_geo, 3), 'w_main', round(w_main, 3),
              '| part2', s['G1_part2_update_gain_grows'], 'r_geo', round(r_geo, 3), 'r_main', round(r_main, 3))
    except FileNotFoundError as ex:
        print('missing', ex)
    # ---- K1
    try:
        b = W.load('c14b_smollm2_plainK.json')
        c11 = W.load('c11_rope_smollm2.json')
        assert b['windows'] == c11['windows'], (b['windows'], c11['windows'])
        s = summarize(b, 'c14b', PAIRS)
        eb = b['excess']
        nq = c11['nll_per_window']; tok = c11['tokens_per_window']
        eq = {m: W.excess(nq[m], nq['dense'], tok) for m in nq if m != 'dense'}
        gp = 1 - eb['attn-late / buffer 32'] / eb['static / buffer 32']
        gq = 1 - eq['attn-late / buffer 32'] / eq['static / buffer 32']
        w3_plain = eb['attn-late / buffer 32'] <= 0.9 * eb['static / buffer 32']
        w3_query = eq['attn-late / buffer 32'] <= 0.9 * eq['static / buffer 32']
        s.update({'query_weight_excess_windows0_7_c11': eq, 'gain_plain': gp, 'gain_query': gq,
                  'gain_plain_over_query': gp / gq if gq else None, 'W3r_plain': bool(w3_plain), 'W3r_query_windows0_7': bool(w3_query),
                  'K1_W3r_changes_verdict': bool(w3_plain != w3_query), 'K1_gain_at_most_half': bool(gp <= 0.5 * gq)})
        s['K1'] = bool(s['K1_W3r_changes_verdict'] or s['K1_gain_at_most_half'])
        # paired comparison on the same 8 windows: plain vs query excess of static/buffer and of attn-late/buffer
        nb = b['nll_per_window']
        s['ppl_ratio_plain_over_query'] = {m: W.ppl_ratio_ci({'a': nb[m], 'b': nq[m]}, 'a', 'b', tok) for m in
                                           ('static / buffer 32', 'attn-late / buffer 32', 'update / buffer 32') if m in nb and m in nq}
        out['c14b_smollm2_plainK'] = s
        print('K1', s['K1'], 'W3r plain/query', w3_plain, w3_query, 'gain plain/query', round(gp, 3), round(gq, 3))
    except FileNotFoundError as ex:
        print('missing', ex)
    W.save('c14_summary.json', out)


if __name__ == '__main__':
    main()
