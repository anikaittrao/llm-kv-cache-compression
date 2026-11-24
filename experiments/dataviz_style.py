"""Shared matplotlib style for the project's figures (dataviz skill, light mode).

Palette = the validated 8-slot categorical order (adjacent pairs pass the CVD and
normal-vision gates; slots 3/4/5 are below 3:1 contrast on the surface, so always
use a legend or direct labels). Use at most 3 slots for scatter-like plots.

Usage inside a make_figures():   import dataviz_style as dv; dv.apply()
Then colour by role:  dv.C['blue'], dv.C['orange'], ...  or dv.SERIES[i].
"""
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'magenta': '#e87ba4', 'green': '#008300', 'violet': '#6250d6', 'red': '#e34948',
     'gray': MUTED, 'black': INK}
SERIES = [C[k] for k in ('blue', 'orange', 'aqua', 'yellow', 'magenta', 'green', 'violet', 'red')]
# sequential blue ramp, light to dark (steps 100..700)
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def apply():
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'axes.titleweight': 'regular',
        'axes.titlelocation': 'left', 'legend.fontsize': 7.5, 'legend.frameon': False,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'axes.prop_cycle': cycler(color=SERIES),
        'lines.linewidth': 1.8, 'lines.markersize': 4, 'patch.linewidth': 0,
        'pdf.fonttype': 42,
    })
