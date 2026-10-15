import random

from sympy import symbols, lambdify
import matplotlib.pyplot as plt
import numpy as np


def nelder_mead(function, points: list, alpha=1, gamma=2, rho=0.5, omega=0.5):
    triangle_history = []
    x, y = symbols('x y', real=True)
    z = lambdify((x, y), function, 'math')
    sorted_points = sorted(np.array(points), key=lambda p: z(p[0], p[1]))
    b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
    triangle_history.append(sorted_points.copy())

    center = (b + g) / 2
    # ***
    # f(b) f(g) f(w)
    # Do the cycle while the difference between functions values at given point is not significant
    # ***

    while abs(z(b[0], b[1]) - z(g[0], g[1])) > 1e-5 or abs(z(b[0], b[1]) - z(w[0], w[1])) > 1e-5 or abs(z(w[0], w[1]) - z(g[0], g[1])) > 1e-5:
        # print(f"BEST:{b}\nGOOD:{g}\nWORST:{w}")
        r = reflection(center, w, alpha)
        if z(b[0], b[1]) <= z(r[0], r[1]) < z(g[0], g[1]):
            sorted_points[2] = r
            sorted_points = sorted(np.array(sorted_points), key=lambda p: z(p[0], p[1]))
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]

        elif z(r[0], r[1]) < z(b[0], b[1]):
            e = expansion_r(center, r, gamma)
            if z(e[0], e[1]) < z(r[0], r[1]):
                sorted_points[2] = e
            else:
                sorted_points[2] = r
            sorted_points = sorted(np.array(sorted_points), key=lambda p: z(p[0], p[1]))
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
        else:
            k = contraction(center, w, rho)
            if z(k[0], k[1]) < z(w[0], w[1]):
                sorted_points[2] = k
            else:
                g, w = shrinkage(b, g, w, omega)
                sorted_points[2] = w
                sorted_points[1] = g
            sorted_points = sorted(np.array(sorted_points), key=lambda p: z(p[0], p[1]))
            b, g, w = sorted_points[0], sorted_points[1], sorted_points[2]
        triangle_history.append(sorted_points.copy())
    return np.array(triangle_history)



def reflection(center , worst, alpha=1):
    # center moze byt oznacovany ako M alebo ako X0
    r = center + alpha * (center - worst)
    return r


def expansion(center, worst, gamma=2):
    e = (gamma + 1) * center - gamma * worst
    return e

def expansion_r(center, r, gamma=2):
    e = center + gamma * (r - center)
    return e

def contraction(center, worst, rho=0.5):
    k = center + rho * (worst - center)
    return k

def shrinkage(best, good, worst, omega=0.5):
    g = best + omega * (good - best)
    w = best + omega * (worst - best)
    return g, w

if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    f = x**2 + 2 * y**2
    # hx, hy, hz = None
    triangle_history = nelder_mead(f, [(2, 2), (2, 1), (1, 2)])
    z = lambdify((x, y), f, 'math')

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    fig = plt.figure(figsize=(12, 6))

    # # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-2.5, 2.5, 500)
    Y = np.linspace(-2.5, 2.5, 500)
    X, Y = np.meshgrid(X, Y)
    # Z = 0.25 * (X**2 + Y**2) + 2 * m_sin(2.5*X) + 1.5 * m_sin(2*Y)
    Z = X**2 + 2 * Y**2

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    # Vykreslenie cesty zostupu červenou čiarou s bodmi
    for triangle in triangle_history:
        print("Triangle: ", triangle)
        print(triangle[:,0])
        # [[b, g, w], [b, g, w]]
        c = random.choice(['red', 'blue', 'green', 'yellow', 'black', 'brown', 'purple', 'grey', 'pink'])
        ax1.plot(triangle[:,0], triangle[:,1], [z(triangle[i][0], triangle[i][1]) for i in range(len(triangle))], color=c, label='Trojuholnik')
    # ax1.plot(hx, hy, hz, color='red', marker='o', markersize=4, label='Cesta zostupu')
    ax1.set_title('3D pohľad na gradientný sestup')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.legend()
    #
    # 2. Vrstevnicový graf (Contour plot) – často prehľadnejší na sledovanie optimalizácie zhora
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    # ax2.plot(hx, hy, color='red', marker='o', markersize=4, label='Optimalizačné kroky')
    for triangle in triangle_history:
        # [[b, g, w], [b, g, w]]
        c = random.choice(['red', 'blue', 'green', 'yellow', 'black', 'brown', 'purple', 'grey', 'pink'])
        ax2.plot(triangle[:,0], triangle[:,1], color=c, label='Trojuholnik')
    ax2.set_title('Vrstevnice (Top-down pohľad)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()

    # plt.tight_layout()
    plt.show()