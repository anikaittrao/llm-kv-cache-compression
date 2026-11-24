"""High-precision recheck of the worst Lemma 10.4 (upper side) case found by kv_bound2.py --dump_worst.
Run: .venv\\Scripts\\python.exe experiments\\bound2_hp_check.py <worst.pt>
Takes the float64 scores s and s^ as exact inputs and recomputes ||a^ - a||_1 and the bound with 50 digits (mpmath)."""
import sys
import mpmath as mp
import torch

mp.mp.dps = 50
w = torch.load(sys.argv[1])
s = [mp.mpf(float(x)) for x in w['s']]; sh = [mp.mpf(float(x)) for x in w['sh']]
def softmax(v):
    M = max(v); e = [mp.e ** (x - M) for x in v]; Z = mp.fsum(e); return [x / Z for x in e]
a, ah = softmax(s), softmax(sh)
delta = [y - x for x, y in zip(s, sh)]
db = mp.fsum(ai * di for ai, di in zip(a, delta))
d = [di - db for di in delta]
R = mp.fsum(ai * (mp.expm1(di) - di) for ai, di in zip(a, d))
m1 = mp.fsum(ai * abs(di) for ai, di in zip(a, d))
tv = mp.fsum(abs(x - y) for x, y in zip(ah, a))
up = (m1 + 2 * R) / (1 + R)
print('where', w['where'], 'n tokens', len(s))
print('float64: tv = %.6e  up = %.6e  rel. violation = %.3e' % (w['tv'], w['up'], w['v']))
print('50 digits: tv = %s  up = %s  tv/up = %s' % (mp.nstr(tv, 12), mp.nstr(up, 12), mp.nstr(tv / up, 15)))
print('float64 absolute error in tv: %.2e' % abs(w['tv'] - float(tv)))
