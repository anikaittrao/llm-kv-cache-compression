"""C22 (wave 4, 2026-11-24): domain shift of the OFFLINE (calibration) basis versus the ONLINE prompt basis, GPT-2.

Domains: 'wiki' = WikiText-2 (calibration = the 16 train windows, seed 0, as everywhere; targets = test windows 0-7) and
'story' = TinyStories-valid (roneneldan/TinyStories, TinyStories-valid.txt, downloaded from the HF hub without credentials; GPT-2
tokens of the text, raw, '<|endoftext|>' separators kept). Story data are split by character position: calibration windows are drawn
(seed 0, 16 windows of 1024 tokens) from the FIRST 6,000,000 characters, target windows are the first 8 consecutive
1024-token windows of the characters 6,000,000-8,000,000. Children's stories are a clearly different register from encyclopaedic
Wikipedia text; they are not code and not a different language (stated limitation).
Setting as kv_buffer.py main: P = 768 prompt tokens prefilled exactly then compressed, 256 continuation tokens in blocks of B = 32
(buffer on), one basis per layer, K256 / V512, G_K from queries, G_V = W_O W_O^T, first token exact (kvlab.run_block).
Methods per target domain (windows 0-7 of the target):
  dense; off(wiki) / buffer; off(story) / buffer; online static / buffer; online update / buffer;
  off(wiki)+update / buffer; off(story)+update / buffer   (offline basis as the starting basis of the incremental update with the
  weight of P-1 rows, as C3).
e(x) = ppl(x)/dense - 1 on the target's continuation tokens; paired bootstrap CI over the 8 windows (w3_stats).
'matched' = the offline basis of the target's own domain, 'mismatched' = the other one.
Rules fixed before the run (both targets):
  D1 shift costs the offline basis : e(mismatched offline) > e(matched offline) with the paired CI of the ppl ratio
                                     (mismatched / matched) above 1, on BOTH targets.
  D2 online does not care          : e(online update / buffer) <= e(mismatched offline / buffer) on both targets.
  D3 update repairs the shift      : e(mismatched offline + update) <= e(mismatched offline) on both targets.
Run:  .venv\\Scripts\\python.exe experiments\\kv_domainshift.py --threads 8 --out experiments\\rerun_2026-11-22\\c22_kv_domainshift.json
      smoke: --windows 1
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlab import (Ctx, L, E, ROOT, base_result, save_json, rss_mb, offline_state, prompt_state, run_block, kinds_of)  # noqa: E402
import w3_stats as W  # noqa: E402

P, SEQ, B = 768, 1024, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'rerun_2026-11-22', 'c22_kv_domainshift.json'))
    a = ap.parse_args()
    T0 = time.perf_counter()
    from huggingface_hub import hf_hub_download
    txt = open(hf_hub_download('roneneldan/TinyStories', 'TinyStories-valid.txt', repo_type='dataset'), encoding='utf-8').read()
    ctx = Ctx(a.threads)
    ids_story_cal = ctx.tok(txt[:6_000_000], return_tensors='pt').input_ids[0]
    ids_story_eval = ctx.tok(txt[6_000_000:8_000_000], return_tensors='pt').input_ids[0]
    assert ids_story_eval.numel() >= a.windows * SEQ
    kinds = kinds_of(256, 512)
    wiki_train = ctx.ids_train
    off_w = ctx.offline()
    ctx.ids_train = ids_story_cal
    off_s = ctx.offline()
    ctx.ids_train = wiki_train
    print(f'calibrations ready ({time.perf_counter()-T0:.0f}s); story eval tokens {ids_story_eval.numel()}', flush=True)
    targets = {'wiki': [ctx.ids_test[i * SEQ:(i + 1) * SEQ] for i in range(a.windows)],
               'story': [ids_story_eval[i * SEQ:(i + 1) * SEQ] for i in range(a.windows)]}
    methods = ['dense', 'off(wiki) / buffer', 'off(story) / buffer', 'online static / buffer', 'online update / buffer',
               'off(wiki)+update / buffer', 'off(story)+update / buffer']
    res = base_result(a, {'P': P, 'B': B, 'story_cal_tokens': int(ids_story_cal.numel()), 'story_eval_tokens': int(ids_story_eval.numel()),
                          'tokens_per_window': SEQ - P - 1, 'targets': {}})
    for tname, wins in targets.items():
        nllw = {m: [] for m in methods}
        for wi, w in enumerate(wins):
            pre = ctx.prefill_stats(w, P)
            for m in methods:
                if m == 'dense':
                    st, upd = None, False
                elif m.startswith('off('):
                    off = off_w if m.startswith('off(wiki)') else off_s
                    upd = '+update' in m
                    st = offline_state(off, kinds, P - 1 if upd else None)
                else:
                    st = prompt_state(ctx, pre, kinds); upd = 'update' in m
                nll, _ = run_block(ctx, w, P, kinds, st, B=B, buf=True, upd=upd)
                nllw[m].append(nll)
            print(f"{tname} window {wi} ({time.perf_counter()-T0:.0f}s): " +
                  ' '.join(f'{m}={math.exp(nllw[m][-1]/(SEQ-P-1)):.2f}' for m in methods), flush=True)
        tok = SEQ - P - 1
        e = {m: W.excess(nllw[m], nllw['dense'], tok) for m in methods if m != 'dense'}
        res['targets'][tname] = {'nll_per_window': nllw, 'excess': e, 'excess_ci': {m: W.ex_ci(nllw, m, 'dense', tok) for m in e},
                                 'dense_ppl': math.exp(sum(nllw['dense']) / (tok * len(wins)))}
    d = {}
    for tname, matched, mism in (('wiki', 'off(wiki)', 'off(story)'), ('story', 'off(story)', 'off(wiki)')):
        t = res['targets'][tname]; nl = t['nll_per_window']; e = t['excess']; tok = SEQ - P - 1
        r = W.ppl_ratio_ci(nl, f'{mism} / buffer', f'{matched} / buffer', tok)
        d[f'D1_{tname}'] = {'ratio_mismatched_over_matched': r, 'e_matched': e[f'{matched} / buffer'], 'e_mismatched': e[f'{mism} / buffer'],
                            'pass': bool(r['ci_lo'] > 1)}
        d[f'D2_{tname}'] = {'e_online_update': e['online update / buffer'], 'e_mismatched': e[f'{mism} / buffer'],
                            'pass': bool(e['online update / buffer'] <= e[f'{mism} / buffer'])}
        d[f'D3_{tname}'] = {'e_mismatched_plus_update': e[f'{mism}+update / buffer'], 'e_mismatched': e[f'{mism} / buffer'],
                            'pass': bool(e[f'{mism}+update / buffer'] <= e[f'{mism} / buffer'])}
        d[f'online_vs_matched_offline_{tname}'] = W.ppl_ratio_ci(nl, 'online update / buffer', f'{matched} / buffer', tok)
    d['D1'] = bool(d['D1_wiki']['pass'] and d['D1_story']['pass'])
    d['D2'] = bool(d['D2_wiki']['pass'] and d['D2_story']['pass'])
    d['D3'] = bool(d['D3_wiki']['pass'] and d['D3_story']['pass'])
    res['decision'] = d
    res['total_runtime_s'] = time.perf_counter() - T0
    res['peak_rss_mb'] = rss_mb()
    save_json(a.out, res)
    for tname, t in res['targets'].items():
        print(tname, 'dense ppl', round(t['dense_ppl'], 3), {k: round(100 * v, 3) for k, v in t['excess'].items()})
    print('decision:', {k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in d.items()})


if __name__ == '__main__':
    main()
