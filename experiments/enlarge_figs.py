"""Re-draw six existing figures with larger text (and two colour/bar fixes) as NEW files with suffix _s.

Run from the project root:  .venv\\Scripts\\python.exe -I experiments\\enlarge_figs.py
Each job calls the plotting script's own --replot into a scratch folder (reads only the saved results JSONs), and, just before
each save, (1) scales every text element by a factor, (2) re-runs tight_layout, (3) applies the _fr theorem relabelling of
papers/relabel_figures_fr.py, (4) swaps the 'yellow' category colour (adjacent to orange) for violet in fig14/fig16 and
(5) replaces the bars of fig13 panel b (log axis, baseline cut) by dots. Writes figures/<stem>_s.{pdf,png}; no existing
figure file is touched, no plotting script is modified.
"""
import os, runpy, shutil, sys, tempfile, warnings
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import dataviz_style as dv

MAP = [('Thm 5.5', 'Thm 3.22'), ('Lemma 10.4', 'Lemma 7.4'), ('Prop 10.2', 'Prop 7.2'),
       ('Prop 10.5', 'Prop 7.5'), ('Prop 10.6', 'Prop 7.6')]
# script, stem written by the script, new stem, scale factor, recolour yellow, bars->dots on log axis (fig13)
JOBS = [('kv_online.py', 'fig14_kv_online', 'fig14_kv_online_fr_s', 1.22, True, False),
        ('kv_bound2.py', 'fig17_kv_bound2', 'fig17_kv_bound2_fr_s', 1.3, False, False),
        ('kv_bound3.py', 'fig21_kv_bound3', 'fig21_kv_bound3_fr_s', 1.5, False, False),
        ('llm_compression.py', 'fig3_als_convergence', 'fig3_als_convergence_s', 1.2, False, False),
        ('kv_update.py', 'fig16_kv_update', 'fig16_kv_update_s', 1.35, True, False),
        ('kv_adaptive.py', 'fig13_kv_sink', 'fig13_kv_sink_s', 1.2, False, True)]
YEL = to_hex(dv.C['yellow']).lower()
VIO = dv.C['violet']
state = {}
_orig_save = Figure.savefig


def _fix(s):
    if not isinstance(s, str):
        return s
    for u, v in state.get('wrap', []):
        s = s.replace(u, v)
    for a, b in MAP:
        s = s.replace(a, b)
    return s


def _save(self, *args, **kw):
    if getattr(self, '_enl_done', False):
        return _orig_save(self, *args, **kw)
    self._enl_done = True
    k, recolor, dots = state['k'], state['recolor'], state['dots']
    self.canvas.draw()
    for ax in self.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            if labs:
                axis.set_tick_params(labelsize=labs[0].get_fontsize() * k)
            fmt = axis.get_major_formatter()
            if not getattr(fmt, '_fr_wrapped', False):
                new = FuncFormatter(lambda x, pos=None, f=fmt: _fix(f(x, pos)))
                new._fr_wrapped = True
                axis.set_major_formatter(new)
    seen = set(id(l) for ax in self.axes for l in ax.get_xticklabels() + ax.get_yticklabels())
    for t in self.findobj(Text):
        if id(t) in seen:
            continue
        seen.add(id(t))
        t.set_text(_fix(t.get_text()))
        t.set_fontsize(t.get_fontsize() * k)
    if recolor:
        for a in self.findobj(Rectangle):
            for getter, setter in ((a.get_facecolor, a.set_facecolor), (a.get_edgecolor, a.set_edgecolor)):
                c = getter()
                if to_hex(c).lower() == YEL:
                    setter(VIO)
        for a in self.findobj(Line2D):
            if to_hex(a.get_color()).lower() == YEL:
                a.set_color(VIO)
            try:
                if to_hex(a.get_markerfacecolor()).lower() == YEL:
                    a.set_markerfacecolor(VIO)
            except Exception:
                pass
    if dots:   # fig13 panel b: bars on a log axis -> dots at the bar heights
        ax = self.axes[1] if len(self.axes) > 1 else self.axes[0]
        bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 0 and ax.get_yscale() == 'log']
        for p in bars:
            ax.plot([p.get_x() + p.get_width() / 2], [p.get_y() + p.get_height()], 'o', ms=6,
                    color=p.get_facecolor()[:3], mfc=p.get_facecolor()[:3] if p.get_alpha() in (None, 1.0) or
                    p.get_facecolor()[3] > 0.9 else 'none', mew=1.4, zorder=3)
            p.set_visible(False)
        if bars:
            ax.grid(True, axis='y')
        ax.tick_params(axis='x', labelsize=7)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tight_layout()
    except Exception as e:
        print('  tight_layout failed:', e)
    return _orig_save(self, *args, **kw)


Figure.savefig = _save
for script, stem, new, k, recolor, dots in JOBS:
    state.update(k=k, recolor=recolor, dots=dots)
    state['wrap'] = ([('half-sweep (U-step', 'half-sweep\n(U-step'), (': random mask', ':\nrandom mask'),
                      (': magnitude mask', ':\nmagnitude mask')] if stem.startswith('fig3_') else [])
    tmp = tempfile.mkdtemp(prefix='enl_fig_')
    sys.argv = [script, '--replot', '--figdir', tmp]
    try:
        runpy.run_path(os.path.join(ROOT, 'experiments', script), run_name='__main__')
    except SystemExit:
        pass
    for ext in ('pdf', 'png'):
        src = os.path.join(tmp, f'{stem}.{ext}')
        dst = os.path.join(ROOT, 'figures', f'{new}.{ext}')
        assert not os.path.exists(dst) or new.endswith('_s'), dst
        shutil.copyfile(src, dst)
        print('wrote', dst)
    shutil.rmtree(tmp, ignore_errors=True)




