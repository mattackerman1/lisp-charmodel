# numpy (float64) reference of Keller Jordan's Muon, compared against the Lisp muon-tensor dump (/tmp/muontest.txt)
import numpy as np
def ns5(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(5):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if tr else X
L = open('/tmp/muontest.txt').read().split('\n'); i = 0; worst = 0
while i < len(L) and L[i].startswith('shape'):
    m, n = map(int, L[i].split()[1:]); rd = lambda s: np.array([float(x) for x in s.split()[1:]]).reshape(m, n)
    p, g1, g2, l1, l2 = (rd(L[i + k]) for k in range(1, 6)); i += 6
    mom = np.zeros((m, n)); scale = np.sqrt(max(1, n / m)); out = []
    for g in (g1, g2):
        g = 0.5 * g; mom = 0.95 * mom + g; u = g + 0.95 * mom
        p = 0.999 * p - 0.02 * scale * ns5(u); out.append(p.copy())
    e1 = np.abs(out[0] - l1).max(); e2 = np.abs(out[1] - l2).max(); worst = max(worst, e1, e2)
    sv = np.linalg.svd(ns5(0.5 * g1), compute_uv=False)
    print(f'din x dout = {m}x{n}: max|lisp-numpy| step1 {e1:.2e} step2 {e2:.2e}; NS5 singular values {sv.min():.3f}..{sv.max():.3f}')
print('PASS' if worst < 1e-5 else 'FAIL', f'worst {worst:.2e}')
