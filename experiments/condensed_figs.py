"""Condensed charts that replace dense tables of papers/final_report.tex.

Every plotted number is read from the saved JSONs (experiments/*.json, experiments/rerun_2026-11-22/*.json).
Confidence intervals come from the JSONs where stored; where none is stored they are recomputed with the project's own
paired bootstrap (stats_utils.ppl_ratio, 10,000 resamples, seed 0) from the per-window NLLs in the JSON. Each such case
is marked "recomputed" below. The script also prints a check of the plotted values against the table values typed in
CHECKS (rounded as in final_report.tex) and lists mismatches.

Run from the project root:   .venv\\Scripts\\python.exe -I experiments\\condensed_figs.py
Outputs: figures/short_*.pdf and .png (new names only; existing figures are never touched).

  short_frontier    chart 1  memory vs excess (GPT-2 and SmolLM2)        tables fu-baselines, fu-r3base, fu-r3quant
  short_kvfollow    chart 2  dot plot of the 12 kvfollow methods          table kvfollow
  short_lowr        chart 3  GPT-2 excess vs fraction of numbers kept     table fu-r3lowr
  short_smollm      chart 4  SmolLM2 recipe, grouped bars                 tables fu-rope, fu-r3rope
  short_equalbudget chart 5  equal-budget recency controls                table fu-r3equal (top block)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dataviz_style as dv
import stats_utils as su
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

dv.apply()
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8})
R = os.path.join(HERE, 'rerun_2026-11-22')
FIG = os.path.join(ROOT, 'figures')
W = 6.3


def J(name, base=R):
    with open(os.path.join(base, name)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(FIG, f'{name}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print('saved', name)


MISMATCH = []


def check(label, value, table_value, nd):
    """value in the unit of the table (percent, ppl ...); table_value as printed."""
    up = lambda v: np.floor(abs(v) * 10 ** nd + 0.5 + 1e-9) * np.sign(v)   # round half up, as in the tables
    if up(value) != up(table_value * 1.0):
        MISMATCH.append((label, value, table_value))
        print(f'  MISMATCH {label}: JSON {value:.4f} -> {round(value, nd)} vs table {table_value}')


def boot_excess(nll_a, nll_d, tok):
    """percent excess and 95% CI of method over dense (recomputed with stats_utils)."""
    r = su.ppl_ratio(nll_a, nll_d, tok)
    return 100 * (r['ratio'] - 1), 100 * (r['ci_lo'] - 1), 100 * (r['ci_hi'] - 1)


# ---------------------------------------------------------------- chart 1
def fig_frontier():
    c5 = J('c5_kv_baselines.json')
    c16 = J('c16_summary.json')
    c16full = J('c16_kv_quantlr.json')
    c15 = J('c15_summary.json')
    tok = c5['tokens_per_window']
    dense = c5['nll_per_window']['dense']
    mem5 = c5['memory_T1024']

    def g(key):                     # GPT-2 C5 point: (mem, excess%, lo, hi)  [CI recomputed]
        e, lo, hi = boot_excess(c5['nll_per_window'][key], dense, tok)
        assert abs(e - 100 * c5['excess'][key]) < 1e-6, key
        return mem5[key], e, lo, hi

    def g16(key):                   # GPT-2 C16 point: CI stored in c16_summary.json
        d = c16['excess_ci'][key]
        assert abs(d['point'] - c16['excess'][key]) < 1e-12
        return c16['memory_T1024'][key], 100 * d['point'], 100 * d['ci_lo'], 100 * d['ci_hi']

    P = {}
    JIT = {}
    for k in ['int8 + buffer 32', 'int4 + buffer 32', 'int2 + buffer 32', 'sink+window f=0.5', 'sink+window f=0.75',
              'sink+window f=0.92', 'H2O f=0.5', 'H2O f=0.75', 'H2O f=0.92', 'offline / buffer 32',
              'static / buffer 32', 'update / buffer 32']:
        P[k] = g(k)
    # C16 (8 threads): grouped-scale and per-token-scale cells; C5 compounded cells (4 threads) are NOT plotted
    for k in ['update / buffer 32 + int8 grouped codes', 'update / buffer 32 + int8 codes',
              'static / buffer 32 + int8 codes', 'offline K256/V512 / buffer 32 + int4 grouped codes',
              'offline K384/V768 / buffer 32', 'offline K384/V768 / buffer 32 + int4 grouped codes']:
        P[k] = g16(k)
    # checks against table fu-baselines / fu-r3quant
    for k, t in [('int8 + buffer 32', 0.002), ('int4 + buffer 32', 0.10), ('int2 + buffer 32', 1.80),
                 ('sink+window f=0.5', 18.47), ('sink+window f=0.75', 3.92), ('sink+window f=0.92', 0.07),
                 ('H2O f=0.5', 8.35), ('H2O f=0.75', 2.50), ('H2O f=0.92', 0.16), ('offline / buffer 32', 1.19),
                 ('static / buffer 32', 1.09), ('update / buffer 32', 0.20)]:
        check('frontier GPT-2 ' + k, P[k][1], t, 3 if k.startswith('int8') else 2)
    for k, t in [('update / buffer 32 + int8 grouped codes', 0.203), ('update / buffer 32 + int8 codes', 2.18),
                 ('static / buffer 32 + int8 codes', 1.97), ('offline K384/V768 / buffer 32', 0.31),
                 ('offline K384/V768 / buffer 32 + int4 grouped codes', 0.31),
                 ('offline K256/V512 / buffer 32 + int4 grouped codes', 1.26)]:
        check('frontier C16 ' + k, P[k][1], t, 3 if t == 0.203 else 2)
    # memory column of tables (2 decimals)
    for k, t in [('int8 + buffer 32', .56), ('int4 + buffer 32', .32), ('int2 + buffer 32', .20),
                 ('offline / buffer 32', .52), ('static / buffer 32', .92), ('update / buffer 32', .92)]:
        check('frontier mem ' + k, P[k][0], t, 2)
    for k, t in [('update / buffer 32 + int8 grouped codes', .71), ('update / buffer 32 + int8 codes', .68),
                 ('offline K384/V768 / buffer 32', .76), ('offline K384/V768 / buffer 32 + int4 grouped codes', .26),
                 ('offline K256/V512 / buffer 32 + int4 grouped codes', .18)]:
        check('frontier C16 mem ' + k, P[k][0], t, 2)
    # catastrophic per-token int4 cells (off axis)
    cat = {k: (c16full['memory_T1024'][k], c16full['ppl'][k]) for k in
           ['offline K256/V512 / buffer 32 + int4 codes', 'offline K384/V768 / buffer 32 + int4 codes']}
    check('frontier ppl offline K256 int4 per-token', cat['offline K256/V512 / buffer 32 + int4 codes'][1], 18357, 0)
    check('frontier ppl offline K384 int4 per-token', cat['offline K384/V768 / buffer 32 + int4 codes'][1], 5159065, 0)

    # SmolLM2 (c15, windows 16-31); CI stored
    S = {}
    for k in ['int8 + buffer 32', 'int4 + buffer 32', 'sink+window f=0.5', 'sink+window f=0.75', 'offline / buffer 32',
              'update / buffer 32', 'attn-late+update / buffer 32']:
        d = c15['excess_ci'][k]
        assert abs(d['point'] - c15['excess'][k]) < 1e-12
        S[k] = (c15['memory_codes_only'][k], 100 * d['point'], 100 * d['ci_lo'], 100 * d['ci_hi'])
    for k, t, tl, th in [('int8 + buffer 32', 0.009, 0.001, 0.017), ('int4 + buffer 32', 0.60, 0.38, 0.83),
                         ('sink+window f=0.5', 3.58, 1.95, 5.68), ('sink+window f=0.75', 0.53, 0.21, 0.88),
                         ('offline / buffer 32', 9.68, 8.66, 10.69), ('update / buffer 32', 5.83, 4.72, 6.89),
                         ('attn-late+update / buffer 32', 1.15, 0.74, 1.57)]:
        check('frontier SmolLM2 ' + k, S[k][1], t, 3 if t == 0.009 else 2)
        check('frontier SmolLM2 lo ' + k, S[k][2], tl, 3 if t == 0.009 else 2)
        check('frontier SmolLM2 hi ' + k, S[k][3], th, 3 if t == 0.009 else 2)

    QC, EC, LC, QL = dv.C['blue'], dv.C['orange'], dv.C['green'], dv.C['violet']
    fig, axs = plt.subplots(2, 1, figsize=(W, 5.3), gridspec_kw={'hspace': 0.5, 'height_ratios': [1.45, 1]})

    def pt(ax, p, color, marker, filled=True, ms=5.5, z=3):
        x, y, lo, hi = p
        ax.errorbar([x], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt='none', ecolor=color, elinewidth=1.0,
                    capsize=2, alpha=0.8, zorder=z - 1)
        ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc=color if filled else dv.SURFACE, mew=1.3,
                ls='none', zorder=z)

    def line(ax, ps, color):
        ax.plot([p[0] for p in ps], [p[1] for p in ps], color=color, lw=0.9, alpha=0.5, zorder=1)

    def lab(ax, p, text, dx, dy, color, ha='left', va='center'):
        ax.annotate(text, (p[0], p[1]), xytext=(dx, dy), textcoords='offset points', color=color, ha=ha, va=va,
                    fontsize=8)

    def fmt_axes(ax, xlim, ylim):
        ax.set_yscale('symlog', linthresh=0.1, linscale=0.35)
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        ax.set_yticks([0, 0.1, 0.3, 1, 3, 10, 30])
        ax.set_yticklabels(['0', '0.1', '0.3', '1', '3', '10', '30'])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, axis='x', color=dv.GRID, lw=0.4)
        ax.axhline(0, color=dv.AXIS, lw=0.8)
        ax.set_xlabel('memory, fraction of the full 16-bit KV cache (1024 tokens)')
        ax.set_ylabel('excess perplexity (%)')
        ax.axhline(0.1, color=dv.MUTED, lw=0.7, ls=':', zorder=1)
        ax.text(1.0, 0.1, 'linear | log ', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7,
                color=dv.MUTED)

    # ---- GPT-2
    ax = axs[0]
    ymin_lo = min(v[2] for v in list(P.values()))
    fmt_axes(ax, (0.0, 1.14), (min(-0.12, 1.4 * ymin_lo), 75))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    q = [P['int8 + buffer 32'], P['int4 + buffer 32'], P['int2 + buffer 32']]
    line(ax, q, QC)
    for p in q:
        pt(ax, p, QC, 'o')
    lab(ax, q[0], 'int8', 0, 11, QC, ha='center')
    lab(ax, q[1], 'int4', 8, 6, QC)
    lab(ax, q[2], 'int2', 8, 5, QC)
    jit = lambda p, d: (p[0] + d,) + tuple(p[1:])
    sw = [P['sink+window f=0.5'], P['sink+window f=0.75'], jit(P['sink+window f=0.92'], -0.022)]
    h2 = [P['H2O f=0.5'], P['H2O f=0.75'], P['H2O f=0.92']]
    line(ax, sw, EC)
    line(ax, h2, EC)
    for p in sw:
        pt(ax, p, EC, 's', ms=5)
    for p in h2:
        pt(ax, p, EC, '^', ms=6)
    lab(ax, sw[0], 'sink + window', 9, 0, EC)
    lab(ax, h2[0], 'H2O-style', 9, 0, EC)
    lr = {'off': P['offline / buffer 32'], 'sta': jit(P['static / buffer 32'], 0.022), 'upd': jit(P['update / buffer 32'], 0.022)}
    pt(ax, lr['off'], LC, 'D', ms=5)
    pt(ax, lr['sta'], LC, 'D', ms=5)
    pt(ax, lr['upd'], LC, 'D', ms=5)
    lab(ax, lr['off'], 'offline', 0, -22, LC, ha='center')
    lab(ax, lr['sta'], 'static', 8, 2, LC)
    lab(ax, lr['upd'], 'update', 8, -2, LC)
    lq = {k: P[k] for k in ['update / buffer 32 + int8 grouped codes', 'update / buffer 32 + int8 codes',
                            'static / buffer 32 + int8 codes', 'offline K256/V512 / buffer 32 + int4 grouped codes',
                            'offline K384/V768 / buffer 32', 'offline K384/V768 / buffer 32 + int4 grouped codes']}
    pt(ax, lq['update / buffer 32 + int8 grouped codes'], QL, 'D', ms=5)
    pt(ax, lq['update / buffer 32 + int8 codes'], QL, 'D', filled=False, ms=5)
    pt(ax, lq['static / buffer 32 + int8 codes'], QL, 'D', filled=False, ms=5)
    pt(ax, lq['offline K256/V512 / buffer 32 + int4 grouped codes'], QL, 'D', ms=5)
    pt(ax, lq['offline K384/V768 / buffer 32 + int4 grouped codes'], QL, 'D', ms=5)
    pt(ax, lq['offline K384/V768 / buffer 32'], LC, 'D', filled=False, ms=5)
    lab(ax, lq['update / buffer 32 + int8 grouped codes'], 'update, int8,\ngrouped scales', 0, -22, QL, ha='center')
    ap = dict(arrowstyle='-', color=QL, lw=0.7, alpha=0.7, shrinkA=2, shrinkB=4)
    ax.annotate('static / update, int8,\nper-token scale (open)', (lq['update / buffer 32 + int8 codes'][0], 2.0),
                xytext=(0.80, 8.0), textcoords='data', color=QL, fontsize=8, ha='left', va='center', arrowprops=ap)
    ax.annotate('offline K256/V512,\nint4 grouped scales', (lq['offline K256/V512 / buffer 32 + int4 grouped codes'][0], 1.26),
                xytext=(0.03, 6.0), textcoords='data', color=QL, fontsize=8, ha='left', va='center', arrowprops=ap)
    lab(ax, lq['offline K384/V768 / buffer 32'], 'offline K384/V768\n(open)', 9, 8, LC)
    lab(ax, lq['offline K384/V768 / buffer 32 + int4 grouped codes'], 'K384/V768,\nint4 grouped', -9, -8, QL, ha='right')
    ytop = 40
    for k, (m, ppl) in cat.items():
        ax.plot([m], [ytop], marker='X', ms=7, color=dv.C['red'], ls='none', zorder=4)
    ax.annotate('perplexity 18,357 and 5.2e6 (off scale)', (0.216, ytop),
                xytext=(10, 0), textcoords='offset points', color=dv.C['red'], fontsize=7.5, va='center')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker='o', color=QC, ls='none', ms=5, label='quantization + buffer 32'),
          Line2D([], [], marker='s', color=EC, ls='none', ms=5, label='eviction: sink + window'),
          Line2D([], [], marker='^', color=EC, ls='none', ms=5.5, label='eviction: H2O-style'),
          Line2D([], [], marker='D', color=LC, ls='none', ms=5, label='low rank + buffer 32'),
          Line2D([], [], marker='D', color=QL, ls='none', ms=5, label='low rank + quantized codes'),
          Line2D([], [], marker='D', color=dv.INK2, mfc=dv.SURFACE, mew=1.3, ls='none', ms=5,
                 label='open: per-token scale, 8 threads (or plain K384/V768)'),
          Line2D([], [], marker='X', color=dv.C['red'], ls='none', ms=6, label='off scale: offline + int4, per-token')]
    fig.legend(handles=hs, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3,
               columnspacing=1.2, labelspacing=0.3)
    ax.set_title('(a) GPT-2, 16 windows, half the codes at prompt 768 (full cache 24.73)', pad=4, fontsize=7.5)

    # ---- SmolLM2
    ax = axs[1]
    fmt_axes(ax, (0.2, 1.0), (-0.12, 30))
    q = [S['int8 + buffer 32'], S['int4 + buffer 32']]
    line(ax, q, QC)
    for p in q:
        pt(ax, p, QC, 'o')
    lab(ax, q[0], 'int8', 8, 6, QC)
    lab(ax, q[1], 'int4', 8, 6, QC)
    sw = [S['sink+window f=0.5'], S['sink+window f=0.75']]
    line(ax, sw, EC)
    for p in sw:
        pt(ax, p, EC, 's', ms=5)
    lab(ax, sw[0], 'sink + window', -9, -8, EC, ha='right')
    for k, name in [('offline / buffer 32', 'offline'), ('update / buffer 32', 'update'),
                    ('attn-late+update / buffer 32', 'attention-weighted + update (+~0.04 basis)')]:
        pt(ax, S[k], LC, 'D', ms=5)
        lab(ax, S[k], name, 9, 0, LC)
    ax.set_xlabel('memory, fraction of the full 16-bit KV cache (3072 tokens), codes only')
    ax.set_title('(b) SmolLM2-135M, fresh windows 16-31, half the codes at prompt 2048 (full cache 12.73)', pad=4, fontsize=7.5)
    fig.subplots_adjust(top=0.83)
    save(fig, 'short_frontier')


# ---------------------------------------------------------------- chart 2
def fig_kvfollow():
    d = J('results_kv_adaptive.json', HERE)
    runs = {r['label']: r for r in d['runs']}
    dense = d['dense_ppl']
    check('kvfollow dense', dense, 33.64, 2)
    rows = [  # (section, text, label in JSON, table ppl, table whole-cache fraction, post hoc)
        ('Option 1: keys only, one basis per head (rank 32)', None),
        ('plain, same rank', 'K head plain uniform r=32', 70.58, 0.75, False),
        ('query-aware, greedy', 'K head query-aware greedy r=32', 56.40, 0.75, False),
        ('query-aware, in-layer', 'K head query-aware in-layer r=32', 45.78, 0.75, False),
        ('Option 2: one basis per layer (rank R of 768)', None),
        ('keys R=128', 'K joint query-aware uniform R=128', 39.64, 0.58, False),
        ('keys R=192', 'K joint query-aware uniform R=192', 36.05, 0.63, False),
        ('keys R=256', 'K joint query-aware uniform R=256', 34.39, 0.67, False),
        ('keys R=384', 'K joint query-aware uniform R=384', 33.73, 0.75, False),
        ('values R=384', 'V joint output-aware uniform R=384', 36.72, 0.75, False),
        ('values R=512', 'V joint output-aware uniform R=512', 34.29, 0.83, False),
        ('keys 192 + values 384', 'KV joint aware uniform R=192/384', 39.56, 0.38, True),
        ('keys 256 + values 384', 'KV joint aware uniform R=256/384', 37.71, 0.42, True),
        ('keys 256 + values 512', 'KV joint aware uniform R=256/512', 34.98, 0.50, True),
        ('same, first token exact', 'sink KV joint aware uniform R=256/512 keep0', 34.49, 0.50, True),
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W, 4.6))
    ypos, y = [], 0
    for r in rows:
        ypos.append(y)
        y += 1
    ymax = max(ypos)
    passline = 1.05 * dense
    check('kvfollow pass line', passline, 35.33, 2)
    ax.axvline(dense, color=dv.MUTED, lw=1, ls=':', zorder=1)
    ax.axvline(passline, color=dv.C['red'], lw=1, ls='--', zorder=1)
    ax.text(dense - 0.15, -1.15, 'full cache\n33.64', color=dv.INK2, fontsize=7.5, ha='right', va='bottom')
    ax.text(passline + 0.12, -1.15, 'pass line\n35.33', color=dv.C['red'], fontsize=7.5, ha='left', va='bottom')
    for r, yy in zip(rows, ypos):
        if r[1] is None:
            ax.text(31.9, yy, r[0], fontsize=8, fontweight='bold', color=dv.INK, va='center', ha='left',
                    bbox=dict(fc=dv.SURFACE, ec='none', pad=1.5), zorder=4)
            continue
        text, lab, tp, tf, post = r
        run = runs[lab]
        ppl, fr = run['ppl'], run['cache_frac']['total']
        check('kvfollow ppl ' + lab, ppl, tp, 2)
        check('kvfollow cache ' + lab, fr, tf, 2)
        ok = ppl <= passline
        col = dv.C['blue'] if ok else dv.C['orange']
        ax.plot([dense, ppl], [yy, yy], color=dv.GRID, lw=1.2, zorder=1)
        ax.plot([ppl], [yy], 'o', ms=6.5, color=col, mfc=col if not post else dv.SURFACE, mew=1.5, zorder=3)
        ax.text(ppl * 1.025, yy, f'{ppl:.2f}', fontsize=8, color=col, va='center', ha='left',
                bbox=dict(fc=dv.SURFACE, ec='none', pad=0.8), zorder=4)
        ax.text(1.03, yy, f'{fr + 1e-9:.2f}', transform=ax.get_yaxis_transform(), fontsize=8, color=dv.INK2, va='center',
                ha='left')
        ax.text(0.0, yy, text, transform=ax.get_yaxis_transform(), fontsize=8, color=dv.INK, va='center', ha='right')
    ax.text(1.03, -2.0, 'cache size,\nfraction of full', transform=ax.get_yaxis_transform(), fontsize=7.5, color=dv.INK2,
            ha='left', va='center')
    ax.set_xscale('log')
    ax.set_xlim(31.5, 82)
    ax.set_ylim(ymax + 0.7, -2.9)
    ax.set_yticks([])
    ax.grid(False)
    ax.grid(True, axis='x', color=dv.GRID, lw=0.5)
    ax.set_xticks([34, 40, 50, 60, 70])
    ax.set_xticklabels(['34', '40', '50', '60', '70'])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('perplexity on WikiText-2 (8192 test tokens, log scale; lower is better)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', color=dv.C['blue'], ls='none', label='within 5% of full cache'),
                       Line2D([], [], marker='o', color=dv.C['orange'], ls='none', label='fails'),
                       Line2D([], [], marker='o', color=dv.INK2, mfc=dv.SURFACE, mew=1.5, ls='none',
                              label='open: chosen after seeing results (post hoc)')],
              loc='upper center', bbox_to_anchor=(0.42, -0.13), ncol=3, columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.30, right=0.84)
    save(fig, 'short_kvfollow')


# ---------------------------------------------------------------- chart 3
def fig_lowr():
    a = J('c14_summary.json')
    lo = a['c14c_gpt2_lowrn']              # 1/8, K64/V128
    mi = a['c14c_gpt2_lowrn_K128V256']     # 1/4
    kb = J('results_kv_buffer.json', HERE)['settings']['KV 256/512 P=768']['ppl']
    ka = J('results_kv_attnbasis.json', HERE)['settings']['KV 256/512 P=768']['ppl']
    d = kb['dense']
    assert abs(ka['dense'] - d) < 1e-9
    half = {'static / no buffer': kb['static / no buffer'] / d - 1, 'static / buffer 32': kb['static / buffer B=32'] / d - 1,
            'update / no buffer': kb['update / no buffer'] / d - 1, 'update / buffer 32': kb['update / buffer B=32'] / d - 1,
            'attn-late / no buffer': ka['attn-late / no buffer'] / d - 1,
            'attn-late / buffer 32': ka['attn-late / buffer'] / d - 1}
    for k, t in [('static / no buffer', 4.24), ('static / buffer 32', 1.09), ('update / no buffer', 2.58),
                 ('update / buffer 32', 0.20), ('attn-late / no buffer', 4.15), ('attn-late / buffer 32', 1.16)]:
        check('lowr 1/2 ' + k, 100 * half[k], t, 2)
    tab = {'static / no buffer': (89.78, 23.18), 'static / buffer 32': (30.75, 6.84),
           'update / no buffer': (81.13, 17.19), 'update / buffer 32': (22.29, 3.25),
           'attn-late / no buffer': (114.15, 26.77), 'attn-late / buffer 32': (36.05, 8.75),
           'attn-late+update / buffer 32': (25.88, 4.39)}
    for k, (t1, t2) in tab.items():
        check('lowr 1/8 ' + k, 100 * lo['excess'][k], t1, 2)
        check('lowr 1/4 ' + k, 100 * mi['excess'][k], t2, 2)
    series = [('static', 'static', dv.C['blue']), ('update', 'update', dv.C['orange']),
              ('attn-late', 'attention-weighted', dv.C['aqua']),
              ('attn-late+update', 'attention-weighted + update', dv.C['violet'])]
    xs = [0, 1, 2]
    c16 = J('c16_summary.json')['excess_ci']
    cidict = {0: lo['excess_ci'], 1: mi['excess_ci']}
    for k16 in ('static / buffer 32', 'update / buffer 32'):
        assert abs(c16[k16]['point'] - half[k16]) < 1e-6, k16     # c16 point equals the buffer/attnbasis point
    fig, ax = plt.subplots(figsize=(W, 3.8))
    sdx = {'static': -0.045, 'update': -0.015, 'attn-late': 0.015, 'attn-late+update': 0.045}
    for key, name, col in series:
        for buf, ls, mk in (('buffer 32', '-', 'o'), ('no buffer', '--', 'o')):
            k = f'{key} / {buf}'
            if k not in lo['excess']:
                continue
            ys = [100 * lo['excess'][k], 100 * mi['excess'][k]]
            xx = [0, 1]
            if k in half:
                ys.append(100 * half[k]); xx.append(2)
            ax.plot(xx, ys, ls=ls, color=col, marker=mk, ms=4.5, mfc=col if buf == 'buffer 32' else dv.SURFACE,
                    mew=1.2, lw=1.5, label=None)
            for xi in xx:
                if xi < 2:
                    c = cidict[xi][k]
                elif k in c16:
                    c = c16[k]
                else:
                    continue
                y0 = 100 * c['point']
                ax.errorbar([xi + sdx[key]], [y0], yerr=[[y0 - 100 * c['ci_lo']], [100 * c['ci_hi'] - y0]], fmt='none',
                            ecolor=col, elinewidth=0.9, capsize=1.8, alpha=0.75, zorder=1)
    ax.set_yscale('log')
    ax.set_ylim(0.005, 220)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(['1/8\n(keys 64, values 128)\nR/n = 0.08 / 0.17', '1/4\n(keys 128, values 256)\nR/n = 0.17 / 0.33',
                        '1/2\n(keys 256, values 512)\nR/n = 0.33 / 0.67'])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('fraction of the numbers per token that are kept')
    ax.set_ylabel('excess perplexity over\nfull cache (%, log scale)')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=2, label=n) for _, n, c in series]
    hs += [Line2D([], [], color=dv.INK2, ls='-', marker='o', ms=4.5, label='buffer 32 (filled)'),
           Line2D([], [], color=dv.INK2, ls='--', marker='o', ms=4.5, mfc=dv.SURFACE, mew=1.2,
                  label='no buffer (open)')]
    ax.legend(handles=hs, loc='upper right', ncol=1, handletextpad=0.4, labelspacing=0.3)
    # end labels at 1/2 column (values)
    for k, dy in [('static / buffer 32', 0.0), ('update / buffer 32', 0.0)]:
        pass
    ax.text(2.05, 100 * half['update / buffer 32'], f"{100 * half['update / buffer 32']:.2f}", fontsize=7.5,
            color=dv.C['orange'], va='center')
    ax.text(2.05, 100 * half['static / buffer 32'] * 0.88, f"{100 * half['static / buffer 32']:.2f}", fontsize=7.5,
            color=dv.C['blue'], va='center')
    ax.text(2.05, 100 * half['attn-late / buffer 32'] * 1.14, f"{100 * half['attn-late / buffer 32']:.2f}", fontsize=7.5,
            color=dv.C['aqua'], va='center')
    ax.text(-0.15, 0.007, 'Bars: 95% intervals where stored (all cells at 1/8 and 1/4; static and update with buffer at 1/2).\nNo interval stored: the other cells at 1/2. Attention-weighted + update has no no-buffer or 1/2 cell.',
            fontsize=7, color=dv.INK2, va='bottom')
    ax.set_title('GPT-2, prompt 768, 16 windows (full cache perplexity 24.73), 255 continuation tokens')
    save(fig, 'short_lowr')


# ---------------------------------------------------------------- chart 4
def fig_smollm():
    c13 = J('c13_summary.json')
    c14 = J('c14_summary.json')
    c15 = J('c15_summary.json')
    meth = [('static', 'static'), ('update', 'update'), ('attn-late', 'attention-\nweighted'),
            ('attn-late+update', 'attention-\nweighted +\nupdate')]
    c11 = J('c11_rope_smollm2.json'); c13a = J('c13a_rope_w0-7_attnupdate.json'); c13b = J('c13b_rope_w8-15_all.json')
    tok13 = c13b['tokens_per_window']
    nll13 = {m: list(c11['nll_per_window'].get(m) or c13a['nll_per_window'][m]) + list(c13b['nll_per_window'][m])
             for m in c13b['nll_per_window']}
    ci13 = {}
    for m in nll13:
        if m == 'dense':
            continue
        e_, lo_, hi_ = boot_excess(nll13[m], nll13['dense'], tok13)
        assert abs(e_ / 100 - c13['excess'][m]) < 1e-9, m
        ci13[m] = {'point': e_ / 100, 'ci_lo': lo_ / 100, 'ci_hi': hi_ / 100}
    panels = [
        ('(a) windows 0-15, prompt 2048\n(full cache perplexity 13.18)', {k: v for k, v in c13['excess'].items()}, ci13),
        ('(b) fresh windows 16-31, prompt 2048\n(full cache perplexity 12.73)', c15['excess'], c15['excess_ci']),
        ('(c) GPT-2 geometry: prompt 768, windows 0-15\n(full cache perplexity 14.41)', c14['c14a_smollm2_gpt2geom']['excess'],
         c14['c14a_smollm2_gpt2geom']['excess_ci']),
        ('(d) plain key weight, windows 0-7, prompt 2048\n(full cache perplexity 12.56)', c14['c14b_smollm2_plainK']['excess'],
         c14['c14b_smollm2_plainK']['excess_ci']),
    ]
    # table checks (fu-rope = c13; fu-r3rope)
    t13 = {'static': (15.09, 5.26), 'update': (13.59, 4.61), 'attn-late': (11.41, 1.75), 'attn-late+update': (8.67, 0.98)}
    t15 = {'static': (18.02, 6.72), 'update': (15.67, 5.83), 'attn-late': (12.82, 2.17), 'attn-late+update': (8.98, 1.15)}
    tg = {'static': (16.03, 3.95), 'update': (13.35, 2.83), 'attn-late': (15.06, 2.66), 'attn-late+update': (10.49, 1.09)}
    tp = {'static': (16.38, 5.28), 'update': (15.25, 5.09), 'attn-late': (14.82, 2.83), 'attn-late+update': (10.99, 1.76)}
    for (title, ex, ci), tt in zip(panels, [t13, t15, tg, tp]):
        for m, _ in meth:
            check(f'smollm {title[:3]} {m} nobuf', 100 * ex[f'{m} / no buffer'], tt[m][0], 2)
            check(f'smollm {title[:3]} {m} buf', 100 * ex[f'{m} / buffer 32'], tt[m][1], 2)
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.6), sharey=True, gridspec_kw={'hspace': 0.72, 'wspace': 0.08})
    cn, cb = '#9ec5f4', dv.C['blue']
    for ax, (title, ex, ci) in zip(axs.ravel(), panels):
        x = np.arange(len(meth))
        for j, (buf, col, off) in enumerate((('no buffer', cn, -0.2), ('buffer 32', cb, 0.2))):
            ys = np.array([100 * ex[f'{m} / {buf}'] for m, _ in meth])
            bars = ax.bar(x + off, ys, 0.38, color=col, label=buf, zorder=2)
            if ci is not None:
                lo = np.array([100 * ci[f'{m} / {buf}']['ci_lo'] for m, _ in meth])
                hi = np.array([100 * ci[f'{m} / {buf}']['ci_hi'] for m, _ in meth])
                ax.errorbar(x + off, ys, yerr=[ys - lo, hi - ys], fmt='none', ecolor=dv.INK, elinewidth=0.9,
                            capsize=1.8, zorder=3)
            tops = hi if ci is not None else ys
            for xi, yv, tp in zip(x + off, ys, tops):
                ax.text(xi, tp + 0.4, f'{yv:.1f}' if yv >= 10 else f'{yv:.2f}',
                        ha='center', va='bottom', fontsize=7, color=dv.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in meth], fontsize=7.5)
        ax.set_ylim(0, 24)
        ax.set_title(title, fontsize=8, pad=5)
        ax.tick_params(axis='x', length=0)
    for ax in axs[:, 0]:
        ax.set_ylabel('excess perplexity over\nfull cache (%)')
    axs[0, 0].legend(loc='upper right', handletextpad=0.4)
    save(fig, 'short_smollm')


# ---------------------------------------------------------------- chart 5
def fig_equal():
    s = J('c18_summary.json')
    a = J('c18a_kv_equalcount.json')
    ctl = J('c18a_kv_control_perstep.json')
    z3 = s['c18a_equalcount']
    ex, ci = z3['from63']['excess'], z3['from63']['excess_ci']
    avg = a['avg_exact_tokens']
    # checks against table fu-r3equal
    for k, t, ta in [('static / own only', 3.78, 3.47), ('static / sliding 16', 1.12, 0.95),
                     ('static / sliding 32', 0.91, 0.77), ('static / block 16', 1.68, 1.55),
                     ('static / block 32', 1.21, 1.09), ('static / block 64', 0.93, 0.79)]:
        check('equal from63 ' + k, 100 * ex[k], t, 2)
        check('equal all ' + k, 100 * z3['all']['excess'][k], ta, 2)
    for k, (tl, th) in {'static / own only': (2.59, 5.05), 'static / sliding 16': (0.59, 1.67),
                        'static / sliding 32': (0.41, 1.44), 'static / block 16': (1.14, 2.24),
                        'static / block 32': (0.67, 1.75), 'static / block 64': (0.51, 1.38)}.items():
        check('equal ci lo ' + k, 100 * ci[k]['ci_lo'], tl, 2)
        check('equal ci hi ' + k, 100 * ci[k]['ci_hi'], th, 2)
    # lagged / random old: point from c18_summary Z3 (from step 63); CI recomputed from per-step NLL
    zz = s['Z3']['excess_from63']
    dense_step = np.array(a['nll_per_step']['dense'])
    ctl_step = {k: np.array(v) for k, v in ctl['nll_per_step'].items()}
    out = {}
    for k in ['lagged 31 + own', 'random old 31 + own']:
        na = ctl_step[k][:, 63:].sum(1)
        nd = dense_step[:, 63:].sum(1)
        e, lo, hi = boot_excess(na, nd, dense_step[:, 63:].shape[1])
        assert abs(e / 100 - zz[k]) < 1e-9, (k, e, zz[k])
        out[k] = (e, lo, hi)
    check('equal lagged', out['lagged 31 + own'][0], 3.37, 2)
    check('equal random', out['random old 31 + own'][0], 3.69, 2)
    rows = [  # label, budget text, (pt, lo, hi), colour, marker
        ('own token only', 1, 'static / own only', dv.C['black'], 'o'),
        ('block buffer 16', 8.5, 'static / block 16', dv.C['orange'], 's'),
        ('sliding window 16', 16, 'static / sliding 16', dv.C['blue'], 'o'),
        ('block buffer 32', 16.5, 'static / block 32', dv.C['orange'], 's'),
        ('sliding window 32', 32, 'static / sliding 32', dv.C['blue'], 'o'),
        ('block buffer 64', 32.5, 'static / block 64', dv.C['orange'], 's'),
        ('lagged: 31 old tokens + own', 32, 'lagged 31 + own', dv.C['red'], 'D'),
        ('random: 31 old tokens + own', 32, 'random old 31 + own', dv.C['red'], '^'),
    ]
    fig, ax = plt.subplots(figsize=(W, 3.5))
    n = len(rows)
    for i, (name, bud, k, col, mk) in enumerate(rows):
        if k in ex:
            p, lo, hi = 100 * ex[k], 100 * ci[k]['ci_lo'], 100 * ci[k]['ci_hi']
        else:
            p, lo, hi = out[k]
        ax.errorbar([p], [i], xerr=[[p - lo], [hi - p]], fmt='none', ecolor=col, elinewidth=1.2, capsize=2.5, alpha=0.8,
                    zorder=2)
        ax.plot([p], [i], marker=mk, ms=6.5, color=col, ls='none', zorder=3)
        ax.text(-0.145, i, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8,
                color=dv.INK)
        ax.text(-0.015, i, f'{bud:g}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8,
                color=dv.INK2)
        ax.text(hi + 0.12, i, f'{p:.2f}', ha='left', va='center', fontsize=8, color=col)
    ax.text(-0.015, -0.95, 'avg. exact\ntokens', transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=7.5, color=dv.INK2)
    ax.axvline(0, color=dv.AXIS, lw=0.8)
    ax.set_ylim(n - 0.4, -1.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    ax.grid(True, axis='x', color=dv.GRID, lw=0.5)
    ax.set_xlim(0, 6.1)
    ax.set_xlabel('excess perplexity over full cache, steps 63-254 (%, with 95% bootstrap interval)')
    fig.subplots_adjust(left=0.36, right=0.97)
    save(fig, 'short_equalbudget')


if __name__ == '__main__':
    fig_frontier()
    fig_kvfollow()
    fig_lowr()
    fig_smollm()
    fig_equal()
    print('\nMISMATCHES between plotted JSON values and final_report.tex table values:', len(MISMATCH))
    for m in MISMATCH:
        print('  ', m)



