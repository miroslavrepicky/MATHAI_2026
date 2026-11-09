from sympy import symbols, diff, lambdify, solve, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin as m_sin
from math import sqrt

from scipy.optimize import minimize_scalar



def powell_method(function ,start_point, u1, u2):
    x, y, g = symbols('x y g', real=True)
    z = lambdify((x, y), function, 'math')
    x_val, y_val = start_point
    z_val = z(start_point[0], start_point[1])
    history_x = [x_val]
    history_y = [y_val]
    history_z = [z_val]
    z_prev = z_val + 1
    x_n1 = x_val
    y_n1 = y_val
    P0_x = x_val
    P0_y = y_val
    i = 0
    print("ITERACIA")
    while abs(z_val - z_prev) > 0.000001:
        i += 1
        print(30*'=')
        print("ITERACIA", i)
        z_prev = z_val

        # P1
        # f_p1 = function.subs({x: P0_x + g * u1[0], y: P0_y + g * u1[1]})
        # roots = solve(diff(f_p1, g), g)
        phi = lambda g: z(P0_x + g * u1[0], P0_y + g * u1[1])
        gamma = minimize_scalar(phi).x
        P1_x = P0_x + gamma * u1[0]
        P1_y = P0_y + gamma * u1[1]
        print(f"P1: {P1_x:.4f}; {P1_y:.4f}")

        # P2
        # f_p2 = function.subs({x: P1_x + g * u2[0], y: P1_y + g * u2[1]})
        # roots = solve(diff(f_p2, g), g)
        phi = lambda g: z(P1_x + g * u2[0], P1_y + g * u2[1])
        gamma = minimize_scalar(phi).x
        P2_x = P1_x + gamma * u2[0]
        P2_y = P1_y + gamma * u2[1]
        print(f"P2: {P2_x:.4f}; {P2_y:.4f}")

        # X1
        u1 = u2.copy()
        u2 = [P2_x - P0_x, P2_y - P0_y]

        d_x, d_y = P2_x - P0_x, P2_y - P0_y
        phi = lambda g: z(x_n1 + g * d_x, y_n1 + g * d_y)
        gamma = minimize_scalar(phi).x
        x_n_x = x_n1 + gamma * d_x
        x_n_y = y_n1 + gamma * d_y
        z_val = z(x_n_x, x_n_y)

        print(f"x_{i}: {x_n_x:.4f}; {x_n_y:.4f}")

        x_n1 = x_n_x
        y_n1 = x_n_y
        P0_x = x_n_x
        P0_y = x_n_y

        history_x.append(x_n_x)
        history_y.append(x_n_y)
        history_z.append(z_val)
        print(f"V bode {x_n_x:.4f}; {x_n_y:.4f} ma funckcia hodnotu: {z_val}")

    return history_x, history_y, history_z


if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    f = (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
    # f = (x - y)/(x**2 + y**2 + 2)
    u2 = [1, 0]
    u1 = [0, 1]
    hx, hy, hz = powell_method(f, (1, 1), u1, u2)

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    fig = plt.figure(figsize=(24, 12))

    # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-2.5, 2.5, 50)
    Y = np.linspace(-2.5, 2.5, 50)
    X, Y = np.meshgrid(X, Y)
    Z = (1.5 - X + X * Y)**2 + (2.25 - X + X * Y**2)**2 + (2.625 - X + X * Y**3)**2
    # Z = (X - Y) / (X**2 + Y**2 + 2)

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    # Vykreslenie cesty zostupu červenou čiarou s bodmi
    ax1.plot(hx, hy, hz, color='red', marker='o', markersize=4, label='Cesta sestupu')
    ax1.set_title('3D pohľad na gradientný sestup')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.legend()

    # 2. Vrstevnicový graf (Contour plot) – často prehľadnejší na sledovanie optimalizácie zhora
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    ax2.plot(hx, hy, color='red', marker='o', markersize=4, label='Optimalizačné kroky')
    ax2.set_title('Vrstevnice (Top-down pohľad)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()

    plt.tight_layout()
    plt.show()