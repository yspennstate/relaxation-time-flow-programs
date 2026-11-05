"""Checks for "The relaxation time is not the value of a flow program".

1. Exact relaxation times tau_2 = 1/(1 - lambda_2) of the simple random walk on P5, P6, C5, K4 and
   the Petersen graph, from the characteristic polynomial of the transition matrix in exact
   arithmetic, compared with closed forms; irrationality is read off the minimal polynomial.
2. Exact optima of the two flow programs on P5 and P6 (minimum congestion of the uniform
   multicommodity flow with demands pi_a pi_b, and the minimum-energy unit flow between the ends):
   rational, as Lemma 2.1 requires.
3. The variational identity of Proposition 3.1: for each chain, the squared-Euclidean
   configuration given by an eigenvector of lambda_2 attains 2 tau_2, and 4,000 random
   configurations per chain (dimension 1 to 4) never exceed it.
4. Random weighted graphs on 5 to 7 vertices with integer edge weights: how often tau_2 is
   irrational.

Writes checks.json next to this file. Needs sympy, numpy and scipy.
"""
import json
import pathlib
import random
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog

ROOT = pathlib.Path(__file__).resolve().parent
out = {}


def walk(n, edges, w=None):
    """Reversible chain from a weighted graph: p_ij = w_ij / w_i, pi_i = w_i / W."""
    w = w or {e: 1 for e in edges}
    W = [[sp.Integer(0)] * n for _ in range(n)]
    for (i, j) in edges:
        W[i][j] += w[(i, j)]
        W[j][i] += w[(i, j)]
    deg = [sum(r) for r in W]
    tot = sum(deg)
    P = sp.Matrix(n, n, lambda i, j: sp.Rational(W[i][j], deg[i]))
    pi = [sp.Rational(d, tot) for d in deg]
    return P, pi, W, tot


def tau2_exact(P):
    lam = sp.Symbol("lam")
    cp = sp.Poly(P.charpoly(lam).as_expr(), lam)
    roots = sorted(sp.real_roots(cp), key=lambda r: float(r.evalf(60)), reverse=True)
    one = roots.pop(0)
    assert abs(float(one.evalf(60)) - 1) < 1e-40
    l2 = roots[0]
    tau = sp.nsimplify(1 / (1 - l2)) if isinstance(l2, sp.Rational) else 1 / (1 - l2)
    mp = sp.minimal_polynomial(1 / (1 - l2), sp.Symbol("x"))
    return l2, sp.simplify(sp.radsimp(tau)), mp


path = lambda n: [(i, i + 1) for i in range(n - 1)]
cycle = lambda n: [(i, (i + 1) % n) for i in range(n)]
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
petersen = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] + \
           [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
graphs = {
    "P5": (5, path(5), 2 + sp.sqrt(2)),
    "P6": (6, path(6), 3 + sp.sqrt(5)),
    "C5": (5, cycle(5), 1 + sp.sqrt(5) / 5),
    "K4": (4, K4, sp.Rational(3, 4)),
    "Petersen": (10, petersen, sp.Rational(3, 2)),
}
ex = {}
chains = {}
for name, (n, E, closed) in graphs.items():
    P, pi, W, tot = walk(n, E)
    chains[name] = (n, E, P, pi, W, tot)
    l2, tau, mp = tau2_exact(P)
    assert sp.simplify(tau - closed) == 0, (name, tau, closed)
    ex[name] = {"lambda_2": str(sp.radsimp(sp.nsimplify(l2.evalf(80), [sp.sqrt(2), sp.sqrt(5)]))),
                "tau_2": str(closed), "tau_2_float": float(closed),
                "minimal_polynomial_of_tau_2": str(mp), "rational": mp.as_poly().degree() == 1}
    print(name, ex[name], flush=True)
out["exact_tau2"] = ex

# 2. flow programs, exactly, on the paths (a tree carries one flow per demand)
fl = {}
for name in ("P5", "P6"):
    n, E, P, pi, W, tot = chains[name]
    pif = [Fr(int(sp.numer(x)), int(sp.denom(x))) for x in pi]
    c = [pif[i] * Fr(int(W[i][i + 1]), int(sum(W[i]))) for i in range(n - 1)]  # c_e = pi_i p_ij
    loads = []
    for k in range(n - 1):  # edge (k, k+1): ordered pairs (a,b) with a<=k<b or b<=k<a
        L = sum(pif[: k + 1])
        loads.append(2 * L * (1 - L) / c[k])
    cong = max(loads)
    energy = sum(1 / ce for ce in c)  # unit flow end to end: sum f_e^2 / c_e with f_e = 1
    fl[name] = {"min_congestion": str(cong), "end_to_end_energy": str(energy)}
    print(name, "flow programs:", fl[name], flush=True)
# the congestion LP solved as an LP on C5 (two routes per pair), floating point, for comparison
n, E, P, pi, W, tot = chains["C5"]
pif = [float(x) for x in pi]
ce = 1.0 / 10  # pi_i p_ij = (2/10)(1/2)
pairs = [(a, b) for a in range(5) for b in range(5) if a != b]
nv = 2 * len(pairs) + 1  # x_cw, x_ccw per ordered pair, then t
A, bvec, Aeq, beq = [], [], [], []
for e in range(5):
    row = [0.0] * nv
    for k, (a, b) in enumerate(pairs):
        cw = [(a + s) % 5 for s in range((b - a) % 5)]
        ccw = [(b + s) % 5 for s in range((a - b) % 5)]
        if e in cw:
            row[2 * k] = 1.0
        if e in ccw:
            row[2 * k + 1] = 1.0
    row[-1] = -ce
    A.append(row)
    bvec.append(0.0)
for k, (a, b) in enumerate(pairs):
    row = [0.0] * nv
    row[2 * k] = row[2 * k + 1] = 1.0
    Aeq.append(row)
    beq.append(pif[a] * pif[b])
res = linprog([0.0] * (nv - 1) + [1.0], A_ub=A, b_ub=bvec, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nv)
assert res.status == 0
fl["C5_min_congestion_float"] = res.fun
print("C5 min congestion (LP, float):", res.fun, " = 12/5 exactly:", abs(res.fun - 2.4) < 1e-9, flush=True)
assert abs(res.fun - 2.4) < 1e-9
out["flow_programs"] = fl

# 3. variational identity: 2 tau_2 = max sum_{a,b} pi_a pi_b ||x_a - x_b||^2 / sum_e c_e ||x_i - x_j||^2
rng = np.random.default_rng(20261105)
var = {}
for name, (n, E, P, pi, W, tot) in chains.items():
    Pn = np.array(P.evalf(30).tolist(), dtype=float)
    pin = np.array([float(x) for x in pi])
    tau = float(graphs[name][2])
    edges = [(i, j, pin[i] * Pn[i, j]) for (i, j) in E]

    def ratio(X):
        num = sum(pin[a] * pin[b] * np.sum((X[a] - X[b]) ** 2) for a in range(n) for b in range(n))
        den = sum(cij * np.sum((X[i] - X[j]) ** 2) for (i, j, cij) in edges)
        return num / den

    D = np.diag(np.sqrt(pin))
    S = D @ Pn @ np.linalg.inv(D)
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    f = np.linalg.inv(D) @ vecs[:, -2]
    best = ratio(f.reshape(-1, 1))
    worst_gap = min(2 * tau - ratio(rng.standard_normal((n, rng.integers(1, 5)))) for _ in range(4000))
    assert abs(best - 2 * tau) < 1e-9 and worst_gap > -1e-9
    var[name] = {"eigenvector_ratio": best, "two_tau_2": 2 * tau, "min_slack_random": worst_gap}
    print(name, "variational:", var[name], flush=True)
out["variational"] = var

# 4. random integer-weighted graphs
random.seed(5)
cnt = {"graphs": 0, "tau2_irrational": 0}
for _ in range(60):
    n = random.randint(5, 7)
    while True:
        E = [(i, j) for i in range(n) for j in range(i + 1, n) if random.random() < 0.45]
        adj = {i: set() for i in range(n)}
        for i, j in E:
            adj[i].add(j)
            adj[j].add(i)
        seen, st = {0}, [0]
        while st:
            u = st.pop()
            for v in adj[u] - seen:
                seen.add(v)
                st.append(v)
        if len(seen) == n:
            break
    w = {e: random.randint(1, 4) for e in E}
    P, pi, W, tot = walk(n, E, w)
    l2, tau, mp = tau2_exact(P)
    cnt["graphs"] += 1
    cnt["tau2_irrational"] += int(mp.as_poly().degree() > 1)
print("random weighted graphs:", cnt, flush=True)
out["random_graphs"] = cnt

(ROOT / "checks.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf8")
print("all checks passed", flush=True)
