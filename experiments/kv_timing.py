"""Step 4 (2026-11-18): does the extra arithmetic of a shared KV basis cost real time? CPU timing of one decode step.

Run:  .venv\\Scripts\\python.exe experiments\\kv_timing.py           (about 4 min; needs an otherwise idle CPU)
      .venv\\Scripts\\python.exe experiments\\kv_timing.py --replot
Writes experiments/results_kv_timing.json and figures/fig12_kv_timing.{pdf,png}.

Setting: generation of one new token (query length 1, batch 1) with T tokens already cached, float32, GPT-2 small shapes
(12 heads x 64). Methods, all storing half the cache except the first:
  full     : standard attention, per head scores q_h K_h^T (T x 64 keys) and output a_h V_h. Timed both as plain batched
             matmuls and with torch's scaled_dot_product_attention (SDPA); the faster one is the baseline.
  g groups : one basis per group of 12/g consecutive heads (section 8 of report.pdf), keys rank Rk = 256/g, values
             Rv = 512/g (exactly half the cache). The cache holds codes Z_K (T x Rk) and Z_V (T x Rv) per group. Per step:
               q~_h = q_h B_{K,h}^T  (64 -> Rk, per head, cost independent of T)
               scores of all heads of a group = Q~ (h x Rk) Z_K^T          (Rk multiply-adds per cached token and head)
               softmax, then c_h = a_h Z_V (Rv per token and head), out_h = mu_V,h + c_h B_{V,h}   (per head, indep. of T)
             The key mean adds q_h.mu_K,h to every score of a head; softmax ignores it, so it is dropped.
             Writing the new token's codes (z = (x - mu) T U, 768 x R) is included in the layer timing.
Correctness: the factored computation is checked against attention with the reconstructed k_hat, v_hat (random bases).
Two timings: (1) attention over the cache only, (2) one whole GPT-2 layer step (ln, c_attn, attention, c_proj, MLP) with
real GPT-2 weights (layer 0; all layers have the same shapes). Time per token ~ 12 x layer + final ln + lm_head.
Methods are run round-robin, several rounds, and the median is reported, so background load hits all methods alike.

Question fixed before the run: is the per-layer shared basis (g = 1, 6x arithmetic) no slower per generated token than
the full cache for T >= 4096 on this CPU? Synthetic T beyond 1024 (GPT-2's limit) only measures the kernels.
"""
import argparse, json, math, os, platform, statistics, subprocess, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, D, E = 12, 64, 768


def make_compressed(g, T, gen):
    """Random codes and bases with the right shapes for g groups at half the cache."""
    hp = H // g
    Rk, Rv = 256 // g, 512 // g
    return {'g': g, 'hp': hp, 'Rk': Rk, 'Rv': Rv,
            'ZK': torch.randn(g, T, Rk, generator=gen), 'ZV': torch.randn(g, T, Rv, generator=gen),
            'BK': torch.randn(g, hp, D, Rk, generator=gen) / math.sqrt(D),   # q_h -> q~_h
            'BV': torch.randn(g, hp, Rv, D, generator=gen) / math.sqrt(Rv),  # c_h -> out_h
            'muV': torch.randn(g, hp, D, generator=gen),
            'EK': torch.randn(E, 256, generator=gen) / math.sqrt(E), 'EV': torch.randn(E, 512, generator=gen) / math.sqrt(E)}


def cpu_load():
    """System CPU load in percent (Windows), or None. Timings are only meaningful on an otherwise idle machine."""
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).LoadPercentage'],
                           capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def attn_full_bmm(q, K, V):  # q (H,1,D), K,V (H,T,D)
    s = torch.bmm(q, K.transpose(1, 2)) * (1 / math.sqrt(D))
    return torch.bmm(torch.softmax(s, -1), V)  # (H,1,D)


def attn_full_sdpa(q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(q[None], K[None], V[None])[0]


def attn_comp(q, C):  # q (H,1,D)
    g, hp = C['g'], C['hp']
    qg = q.view(g, hp, D)
    qt = torch.einsum('ghd,ghdr->ghr', qg, C['BK'])                         # (g,hp,Rk), independent of T
    s = torch.bmm(qt, C['ZK'].transpose(1, 2)) * (1 / math.sqrt(D))          # (g,hp,T)
    c = torch.bmm(torch.softmax(s, -1), C['ZV'])                             # (g,hp,Rv)
    return (C['muV'] + torch.einsum('ghr,ghrd->ghd', c, C['BV'])).view(H, 1, D)


def check_correctness(gen):
    """Factored attention == attention with reconstructed k_hat, v_hat (float64, random weighted bases)."""
    T, out = 300, {}
    dt = torch.float64
    for g in (1, 2, 4, 12):
        W, hp, Rk, Rv = E // g, H // g, 256 // g, 512 // g
        X = {k: torch.randn(T, E, generator=gen, dtype=dt) for k in 'KV'}
        q = torch.randn(H, 1, D, generator=gen, dtype=dt)
        Khat, Vhat = torch.zeros(T, E, dtype=dt), torch.zeros(T, E, dtype=dt)
        C = {'g': g, 'hp': hp, 'ZK': [], 'ZV': [], 'BK': [], 'BV': [], 'muV': []}
        for j in range(g):
            sl = slice(j * W, (j + 1) * W)
            for kind, R, hat in (('K', Rk, Khat), ('V', Rv, Vhat)):
                A = torch.randn(W, W, generator=gen, dtype=dt); G = A @ A.T / W + torch.eye(W, dtype=dt)
                w, Q = torch.linalg.eigh(G); Th, Ti = (Q * w.sqrt()) @ Q.T, (Q / w.sqrt()) @ Q.T
                U = torch.linalg.qr(torch.randn(W, R, generator=gen, dtype=dt))[0]
                mu = torch.randn(W, generator=gen, dtype=dt)
                Z = (X[kind][:, sl] - mu) @ Th @ U                 # stored codes
                B = U.T @ Ti                                       # R x W, x_hat = mu + Z B
                hat[:, sl] = mu + Z @ B
                Bh = torch.stack([B[:, h * D:(h + 1) * D] for h in range(hp)])  # (hp, R, 64)
                if kind == 'K':
                    C['ZK'].append(Z); C['BK'].append(Bh.transpose(1, 2))
                else:
                    C['ZV'].append(Z); C['BV'].append(Bh); C['muV'].append(mu.view(hp, D))
        C = {k: (torch.stack(v) if isinstance(v, list) else v) for k, v in C.items()}
        ref = attn_full_bmm(q, Khat.view(T, H, D).transpose(0, 1), Vhat.view(T, H, D).transpose(0, 1))
        got = attn_comp(q, C)
        out[g] = float((got - ref).abs().max() / ref.abs().max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=4)
    # defaults = the reported run (an earlier, shorter run used rounds 5, reps 20, T up to 32768)
    ap.add_argument('--T', type=int, nargs='*', default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument('--groups', type=int, nargs='*', default=[1, 2, 4, 12])
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--figdir', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'experiments', 'results_kv_timing.json'))
    ap.add_argument('--replot', action='store_true')
    a = ap.parse_args()
    if a.replot:
        with open(a.out) as f:
            make_figures(a, json.load(f))
        print('figures regenerated'); return
    torch.set_num_threads(a.threads)
    gen = torch.Generator().manual_seed(0)
    T0 = time.perf_counter()
    corr = check_correctness(gen)
    print('correctness (max rel. diff, float64):', corr, flush=True)

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    blk = model.transformer.h[0]
    ln_f, lm_head = model.transformer.ln_f, model.lm_head

    def timeit(fn):
        with torch.inference_mode():
            for _ in range(3):
                fn()
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    res = {'config': vars(a), 'torch': torch.__version__,
           'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                        'torch_threads': torch.get_num_threads()},
           'correctness_max_rel_diff': corr, 'cpu_load_start_pct': cpu_load(), 'points': []}
    print('CPU load at start (%):', res['cpu_load_start_pct'], flush=True)
    x1 = torch.randn(1, 1, E, generator=gen)
    with torch.inference_mode():
        res['lm_head_ms'] = 1e3 * statistics.median([timeit(lambda: lm_head(ln_f(x1))) for _ in range(a.rounds)])
    print(f"final ln + lm_head: {res['lm_head_ms']:.3f} ms", flush=True)

    for T in a.T:
        K = torch.randn(H, T, D, generator=gen); V = torch.randn(H, T, D, generator=gen)
        Cs = {g: make_compressed(g, T, gen) for g in a.groups}
        q = torch.randn(H, 1, D, generator=gen)
        x = torch.randn(1, 1, E, generator=gen)

        def layer(attn_fn, C=None):
            def f():
                h = blk.ln_1(x)
                qkv = blk.attn.c_attn(h)[0, 0]
                qq, kk, vv = qkv.split(E)
                if C is not None:  # encode the new token's key and value codes (cost independent of T)
                    kk @ C['EK']; vv @ C['EV']
                o = attn_fn(qq.view(H, 1, D))
                y = x + blk.attn.c_proj(o.reshape(1, 1, E))
                return y + blk.mlp(blk.ln_2(y))
            return f

        fns = {'full_bmm': (lambda: attn_full_bmm(q, K, V), layer(lambda qq: attn_full_bmm(qq, K, V))),
               'full_sdpa': (lambda: attn_full_sdpa(q, K, V), layer(lambda qq: attn_full_sdpa(qq, K, V)))}
        for g, C in Cs.items():
            fns[f'g{g}'] = (lambda C=C: attn_comp(q, C), layer(lambda qq, C=C: attn_comp(qq, C), C))
        samples = {k: {'attn': [], 'layer': []} for k in fns}
        for _ in range(a.rounds):  # round-robin so that background load affects all methods alike
            for k, (fa, fl) in fns.items():
                samples[k]['attn'].append(timeit(fa)); samples[k]['layer'].append(timeit(fl))
        pt = {'T': T, 'cpu_load_pct': cpu_load(), 'methods': {}}
        for k, s in samples.items():
            at, lt = 1e3 * statistics.median(s['attn']), 1e3 * statistics.median(s['layer'])
            if k.startswith('g'):
                g = int(k[1:]); Rk, Rv = 256 // g, 512 // g
                cache_bytes, macs = 4 * T * (Rk + Rv) * g, T * H * (Rk + Rv)
            else:
                cache_bytes, macs = 4 * T * 2 * E, T * H * 2 * D
            pt['methods'][k] = {'attn_ms': at, 'layer_ms': lt, 'token_ms_est': 12 * lt + res['lm_head_ms'],
                                'attn_ms_rounds': [1e3 * v for v in s['attn']], 'layer_ms_rounds': [1e3 * v for v in s['layer']],
                                'cache_bytes_per_layer': cache_bytes, 'attn_macs': macs,
                                'attn_read_GBps': cache_bytes / (at * 1e-3) / 1e9}
        base = min(('full_bmm', 'full_sdpa'), key=lambda k: pt['methods'][k]['token_ms_est'])
        pt['baseline'] = base
        for k, m in pt['methods'].items():
            m['attn_ratio'] = m['attn_ms'] / pt['methods'][base]['attn_ms']
            m['token_ratio'] = m['token_ms_est'] / pt['methods'][base]['token_ms_est']
        res['points'].append(pt)
        print(f'T={T:6d} base={base:9s} ' + '  '.join(f"{k}: attn {m['attn_ms']:.3f}ms ({m['attn_ratio']:.2f}x) tok {m['token_ratio']:.2f}x"
                                                    for k, m in pt['methods'].items()) + f'  [{time.perf_counter()-T0:.0f}s]', flush=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        del K, V, Cs
    long = [p for p in res['points'] if p['T'] >= 4096]
    res['decision'] = {'question': 'g1 per-token time <= full cache for all T >= 4096',
                       'g1_token_ratio_by_T': {p['T']: p['methods']['g1']['token_ratio'] for p in res['points']},
                       'answer_yes': bool(long) and all(p['methods']['g1']['token_ratio'] <= 1.0 for p in long)}
    res['total_runtime_s'] = time.perf_counter() - T0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('decision:', json.dumps(res['decision'], indent=1))
    make_figures(a, res)


def make_figures(a, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator
    import dataviz_style as dv
    dv.apply()
    cols = {'full_bmm': dv.C['black'], 'full_sdpa': dv.C['gray'], 'g1': dv.C['blue'], 'g2': dv.C['orange'], 'g4': dv.C['aqua'], 'g12': dv.C['magenta']}
    labs = {'full_bmm': 'full cache (matmul)', 'full_sdpa': 'full cache (SDPA)', 'g1': 'half, 1 basis/layer (6x arith.)',
            'g2': 'half, 2 groups (3x)', 'g4': 'half, 4 groups (1.5x)', 'g12': 'half, per head (0.5x)'}
    P = res['points']; Ts = [p['T'] for p in P]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for k in P[0]['methods']:
        ls = '--' if k == 'full_sdpa' else '-'
        ax[0].loglog(Ts, [p['methods'][k]['attn_ms'] for p in P], marker='o', ms=3, color=cols.get(k), label=labs.get(k, k), ls=ls)
        ax[1].semilogx(Ts, [p['methods'][k]['token_ratio'] for p in P], marker='o', ms=3, color=cols.get(k), label=labs.get(k, k), ls=ls)
    ax[1].axhline(1, color='gray', lw=0.8)
    for x in ax:
        x.xaxis.set_minor_locator(NullLocator()); x.xaxis.set_minor_formatter(NullFormatter())
        x.set_xticks(Ts); x.set_xticklabels([f'{t // 1024}k' if t >= 1024 else str(t) for t in Ts])
        x.set_xlabel('cached tokens T')
    ax[0].set_ylabel('time (ms)'); ax[0].set_title('(a) attention over the cache, one layer')
    ax[1].set_ylabel('time per token / full cache'); ax[1].set_title('(b) whole decode step (12 layers + output)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig.savefig(os.path.join(a.figdir, f'fig12_kv_timing.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)


if __name__ == '__main__':
    main()
