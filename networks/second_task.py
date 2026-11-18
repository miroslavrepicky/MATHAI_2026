"""
Úloha 5: ručné prepočítanie dvoch epoch na sieti 4-2-1 (regresia, MSE).

Skript vypíše celý postup v poradí zo zadania, so vzorcami a dosadenými číslami,
takže ho môžeš prepisovať na papier. Na konci výsledky skontroluje voči BaseModel.
"""
import numpy as np

from base_model import BaseModel
from layers import InputLayer, HiddenLayer, OutputLayer
from activations import *
from loss_funtions import *

D = 6   # počet desatinných miest vo výpise (na papier radšej aspoň 5-6)


def f(v):
    """Číslo na D desatinných miest (pri zápornom bez medzery navyše)."""
    return f"{v:.{D}f}"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(x, W, b, v, c):
    z = W @ x + b                  # z_j = sum_i w_ji x_i + b_j
    h = sigmoid(z)
    y_hat = v @ h + c
    return z, h, y_hat


def print_forward(x, W, b, v, c, y):
    """Dopredný prechod so vzorcami a dosadenými číslami."""
    z, h, y_hat = forward(x, W, b, v, c)
    xs = ["x1", "x2", "x3", "x4"]
    print("  DOPREDNÝ PRECHOD")
    for j in range(2):
        terms = " + ".join(f"({f(W[j, i])})({f(x[i])})" for i in range(4))
        print(f"  z{j+1} = w{j+1}1*x1 + w{j+1}2*x2 + w{j+1}3*x3 + w{j+1}4*x4 + b{j+1}")
        print(f"     = {terms} + ({f(b[j])})")
        print(f"     = {f(z[j])}")
    for j in range(2):
        print(f"  h{j+1} = sigmoid(z{j+1}) = 1/(1+e^(-({f(z[j])}))) = {f(h[j])}")
    print(f"  y^ = v1*h1 + v2*h2 + c")
    print(f"     = ({f(v[0])})({f(h[0])}) + ({f(v[1])})({f(h[1])}) + ({f(c)})")
    print(f"     = {f(y_hat)}")
    L = 0.5 * (y - y_hat) ** 2
    print(f"  L  = 1/2 (y - y^)^2 = 1/2 ({f(y)} - {f(y_hat)})^2 = {f(L)}")
    return z, h, y_hat, L


def epoch(n, x, y, W, b, v, c, alpha):
    print("=" * 70)
    print(f"EPOCHA {n}")
    print("=" * 70)
    print("  Parametre na začiatku epochy:")
    print(f"  w1* = {np.round(W[0], D)}   w2* = {np.round(W[1], D)}")
    print(f"  b = {np.round(b, D)}   v = {np.round(v, D)}   c = {f(c)}")
    print()
    z, h, y_hat, L = print_forward(x, W, b, v, c, y)

    # ---- chybové signály ----
    print("\n  CHYBOVÉ SIGNÁLY")
    delta = y_hat - y
    print(f"  delta = dL/dy^ = y^ - y = {f(y_hat)} - {f(y)} = {f(delta)}")
    sig_d = h * (1 - h)
    delta_h = delta * v * sig_d      # POZOR: so STARÝMI v (pred aktualizáciou)
    for j in range(2):
        print(f"  d{j+1} = dL/dz{j+1} = delta * v{j+1} * h{j+1}(1-h{j+1})")
        print(f"     = ({f(delta)})({f(v[j])})({f(h[j])})(1-{f(h[j])})")
        print(f"     = {f(delta_h[j])}")

    # ---- gradienty ----
    print("\n  GRADIENTY")
    dv = delta * h
    dc = delta
    for j in range(2):
        print(f"  dL/dv{j+1} = delta*h{j+1} = ({f(delta)})({f(h[j])}) = {f(dv[j])}")
    print(f"  dL/dc  = delta = {f(dc)}")
    dW = np.outer(delta_h, x)
    db = delta_h
    for j in range(2):
        for i in range(4):
            print(f"  dL/dw{j+1}{i+1} = d{j+1}*x{i+1} = ({f(delta_h[j])})({f(x[i])}) = {f(dW[j, i])}")
    for j in range(2):
        print(f"  dL/db{j+1} = d{j+1} = {f(db[j])}")

    # ---- aktualizácia ----
    print(f"\n  AKTUALIZÁCIA (theta_nové = theta - alpha * dL/dtheta, alpha = {alpha})")
    W_new = W - alpha * dW
    b_new = b - alpha * db
    v_new = v - alpha * dv
    c_new = c - alpha * dc
    for j in range(2):
        for i in range(4):
            print(f"  w{j+1}{i+1} = {f(W[j, i])} - {alpha}*({f(dW[j, i])}) = {f(W_new[j, i])}")
    for j in range(2):
        print(f"  b{j+1}  = {f(b[j])} - {alpha}*({f(db[j])}) = {f(b_new[j])}")
    for j in range(2):
        print(f"  v{j+1}  = {f(v[j])} - {alpha}*({f(dv[j])}) = {f(v_new[j])}")
    print(f"  c   = {f(c)} - {alpha}*({f(dc)}) = {f(c_new)}")
    print()
    return W_new, b_new, v_new, c_new, L


if __name__ == "__main__":
    # ---------------- zadanie (podľa tabuľky) ----------------
    x = np.array([0.30, 0.80, 0.70, 0.90])
    W = np.array([(0.20, -0.20, -0.40, -0.10),     # riadok 1: x1..x4 -> h1
                  (-0.30, -0.60, 0.70, 0.40)])     # riadok 2: x1..x4 -> h2
    b = np.array([-0.14, 0.03])
    v = np.array([0.20, 0.60])
    c = 0.07
    alpha = 0.03
    y = 1.0

    print("ZADANIE: sieť 4-2-1, skrytá vrstva sigmoid, výstup lineárny, L = 1/2 (y - y^)^2")
    print(f"x = {x}\ny = {y}   alpha = {alpha}\n")

    W1, b1, v1, c1, L1 = epoch(1, x, y, W, b, v, c, alpha)
    W2, b2, v2, c2, L2 = epoch(2, x, y, W1, b1, v1, c1, alpha)

    # ---- bod 3: výstup a strata po druhej aktualizácii ----
    print("=" * 70)
    print("PO DRUHEJ AKTUALIZÁCII")
    print("=" * 70)
    _, _, _, L3 = print_forward(x, W2, b2, v2, c2, y)
    print("\n  ZHRNUTIE STRATY")
    print(f"  pred tréningom : L0 = {f(L1)}")
    print(f"  po 1. epoche   : L1 = {f(L2)}")
    print(f"  po 2. epoche   : L2 = {f(L3)}")
    ok = L1 > L2 > L3
    print(f"  L0 > L1 > L2 ?  {'ÁNO, strata klesá' if ok else 'NIE'}")

    # ---------------- kontrola voči BaseModel ----------------
    model = BaseModel()
    model.add_layer(InputLayer(4))
    model.add_layer(HiddenLayer(2, Sigmoid()))
    model.add_layer(OutputLayer(1, Linear()))
    model.build(mode="zadanie")
    model.loss = MSE()
    for _ in range(2):
        model.forward(x)
        model.backward(x, np.array([y]), alpha)
    same = (np.allclose(model.layers[1].weights.T, W2)
            and np.allclose(model.layers[1].b, b2)
            and np.allclose(model.layers[2].weights.ravel(), v2)
            and np.allclose(model.layers[2].b, c2))
    print(f"\nKontrola voči BaseModel (2 epochy): {'zhoda' if same else 'NEZHODA'}")