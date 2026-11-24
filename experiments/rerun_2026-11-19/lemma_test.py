import torch
torch.set_num_threads(1); torch.manual_seed(0)
worst_l, worst_p = 0.0, 0.0
for _ in range(20000):
    n = int(torch.randint(2, 40, (1,)))
    s = torch.randn(n, dtype=torch.float64) * float(torch.rand(1) * 6)
    d = torch.randn(n, dtype=torch.float64) * float(torch.rand(1) * 3)
    a, ah = torch.softmax(s, 0), torch.softmax(s + d, 0)
    tv = (ah - a).abs().sum(); osc = d.max() - d.min()
    worst_l = max(worst_l, float(tv / (0.5 * osc)))
    V = torch.randn(n, 5, dtype=torch.float64); Vh = V + 0.1 * torch.randn(n, 5, dtype=torch.float64)
    err = (ah @ Vh - a @ V).norm()
    diam = torch.cdist(V, V).max()
    b = 0.5 * tv * diam + (ah * (Vh - V).norm(dim=1)).sum()
    worst_p = max(worst_p, float(err / b))
t = torch.tensor(1.0, dtype=torch.float64)
print('max tv/(osc/2):', worst_l, ' max err/bound:', worst_p, ' sharpness (t=1e-3, 2 tokens):',
      float((torch.softmax(torch.tensor([1e-3, 0.]), 0) - 0.5).abs().sum() / 5e-4))
