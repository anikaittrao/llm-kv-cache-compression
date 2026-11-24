import json
r = json.load(open('experiments/results_kv_bound.json'))
for k, v in r['frob'].items():
    print(f"{k:22s} max/frob(prompt) {v['max_over_frob_prompt_median']:.3f}  cont/prompt rms {v['cont_over_prompt_rms_median']:.3f}")
print('runtime', r['total_runtime_s'])
import numpy as np
b = np.array(r['beta']); print('beta range', b.min(), np.median(b), b.max())
