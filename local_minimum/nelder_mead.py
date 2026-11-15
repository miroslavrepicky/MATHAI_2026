import random
import math
from sympy import symbols, lambdify
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def fmt_point(p):
    # Zaokrúhľujeme LEN pri výpise, nie vo výpočtoch
    return f"({p[0]:.4f}, {p[1]:.4f})"

def nelder_mead(function, points: list, alpha=1, gamma=2, rho=0.5, omega=0.5, k_max=20000):
    triangle_history = []
    x, y = symbols('x y', real=True)
    z_raw = lambdify((x, y), function, 'math')
    z = lambda a, b: z_raw(a, b)
    def key(p):
        return (z(p[0], p[1]), p[0])
    sorted_points = sorted(np.array(points), key=key)
    b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
    triangle_history.append(sorted_points.copy())
    triangle_history[-1].append(sorted_points[0].copy())


    # ***
    # f(b) f(g) f(w)
    # Do the cycle while the simplex is larger than epsilon (max distance from best vertex)
    # ***
    table = PrettyTable()
    table.field_names = ["Iter", "BEST b", "f(b)", "GOOD g", "f(g)", "WORST w", "f(w)"]
    table.align = "r"

    def add_row(iteration, b, g, w):
        table.add_row([iteration,
                       fmt_point(b), f"{z(b[0], b[1]):.6f}",
                       fmt_point(g), f"{z(g[0], g[1]):.6f}",
                       fmt_point(w), f"{z(w[0], w[1]):.6f}"])

    iteration = 0
    add_row(iteration, b, g, w)
    epsilon = 10e-8

    def converged(b, g, w):
        # Zastavenie: d = max vzdialenosť vrcholov od najlepšieho vrcholu B, d < epsilon
        d = max(np.linalg.norm(g - b), np.linalg.norm(w - b))
        return d < epsilon

    found = converged(b, g, w)
    counter = 0
    while not found:
        if counter >= k_max:
            print("MAX ITERATION REACHED")
            break
        counter += 1
        iteration += 1
        center = ((b + g) / 2)

        r = reflection(center, w, alpha)
        if z(b[0], b[1]) <= z(r[0], r[1]) < z(g[0], g[1]):
            sorted_points[2] = r
            sorted_points = sorted(np.array(sorted_points), key=key)
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]

        elif z(r[0], r[1]) < z(b[0], b[1]):
            e = expansion_r(center, r, gamma)
            if z(e[0], e[1]) < z(r[0], r[1]):
                sorted_points[2] = e
            else:
                sorted_points[2] = r
            sorted_points = sorted(np.array(sorted_points), key=key)
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
        else:
            method = None
            if z(g[0],g[1]) <= z(r[0], r[1]) < z(w[0], w[1]):
                k = contraction_out(center, r, rho)
                method = "Out"
                # prijimame aj je lepsia alebo rovna ako r
            else:
                k = contraction(center, w, rho)
                method = "In"
                # prijimame ak je lepsia ako w
            if method == "In" and z(k[0], k[1]) < z(w[0], w[1]):
                sorted_points[2] = k
            elif method == "Out" and z(k[0], k[1]) <= z(r[0], r[1]):
                sorted_points[2] = k
            else:
                g, w = shrinkage(b, g, w, omega)
                sorted_points[2] = w
                sorted_points[1] = g
            sorted_points = sorted(np.array(sorted_points), key=key)
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
        triangle_history.append(sorted_points.copy())
        triangle_history[-1].append(sorted_points[0].copy())
        add_row(iteration, b, g, w)
        found = converged(b, g, w)
    print(table)
    return np.array(triangle_history)



def reflection(center , worst, alpha=1):
    # center moze byt oznacovany ako M alebo ako X0
    r = center + alpha * (center - worst)
    return r

def expansion_r(center, r, gamma=2):
    e = center + gamma * (r - center)
    return e

def contraction(center, worst, rho=0.5):
    k = center + rho * (worst - center)
    return k

def contraction_out(center, r, rho=0.5):
    k = center + rho * (r - center)
    return k

def shrinkage(best, good, worst, omega=0.5):
    g = best + omega * (good - best)
    w = best + omega * (worst - best)
    return g, w

if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    # f = x**2 - (4 * x) + y**2 - y - x*y
    f = ((x + 1) ** 2) + 4 * (y - (x**3)) ** 2

    # f = x ** 4 - 2 * x ** 2 + (y - x) ** 2
    f = math.e**(x+1) + math.e**(-y) + (x - y)**2
    # hx, hy, hz = None
    triangle_history = nelder_mead(f, [(1,-2), (2, -2), (1, -1)])

