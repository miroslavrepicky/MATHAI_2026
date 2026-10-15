from mpmath.functions.zeta import secondzeta_prime_term
from sympy import symbols, diff, lambdify, Matrix, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin as m_sin


def nelder_mead(function, points: list, alpha=1, gamma=2, rho=0.5, omega=0.5):
    b, g, w = sorted(points, key=lambda p: lambdify((x, y), function, 'math')(p[0], p[1]))
    print(f"BEST:{b}\nGOOD:{g}\nWORST:{w}")


if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    #f = x**2 + 2*y**2
    f = x**2 + 2 * y**2
    # hx, hy, hz = None
    nelder_mead(f, [(2, 2), (2, 1), (1, 2)])

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    fig = plt.figure(figsize=(12, 6))

    # # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    # ax1 = fig.add_subplot(121, projection='3d')
    # X = np.linspace(-5.5, 5.5, 500)
    # Y = np.linspace(-5.5, 5.5, 500)
    # X, Y = np.meshgrid(X, Y)
    # Z = 0.25 * (X**2 + Y**2) + 2 * m_sin(2.5*X) + 1.5 * m_sin(2*Y)
    #
    # ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    # # Vykreslenie cesty zostupu červenou čiarou s bodmi
    # ax1.plot(hx, hy, hz, color='red', marker='o', markersize=4, label='Cesta sestupu')
    # ax1.set_title('3D pohľad na gradientný sestup')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('f(x, y)')
    # ax1.legend()
    #
    # # 2. Vrstevnicový graf (Contour plot) – často prehľadnejší na sledovanie optimalizácie zhora
    # ax2 = fig.add_subplot(122)
    # contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    # fig.colorbar(contour, ax=ax2)
    # ax2.plot(hx, hy, color='red', marker='o', markersize=4, label='Optimalizačné kroky')
    # ax2.set_title('Vrstevnice (Top-down pohľad)')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.legend()
    #
    # plt.tight_layout()
    # plt.show()