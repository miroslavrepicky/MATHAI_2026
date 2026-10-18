from sympy import symbols, diff, lambdify, solve
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt



def gradient_descent_normalized(function ,start_point):
    x, y, g = symbols('x y g', real=True)
    z = lambdify((x, y), function, 'math')
    x_val, y_val = start_point
    z_val = z(start_point[0], start_point[1])
    history_x = [x_val]
    history_y = [y_val]
    history_z = [z_val]
    z_prev = z_val + 1


    d_x = lambdify((x, y), diff(function, x), 'math')
    d_y = lambdify((x, y), diff(function, y), 'math')
    while abs(z_val - z_prev) > 0.000001:
        z_prev = z_val
        grad_x = d_x(x_val, y_val)
        grad_y = d_y(x_val, y_val)
        norm_grad = sqrt(grad_x**2 + grad_y**2)
        Sx = - grad_x / norm_grad
        Sy = - grad_y / norm_grad
        f_p1 = function.subs({x: x_val + g * Sx, y: y_val + g * Sy})
        roots = solve(diff(f_p1, g), g)
        # len. kladné reálne korene, z nich ten s najnižšou hodnotou funkcie
        candidates = [float(r) for r in roots if r.is_real and r > 0]
        if not candidates:
            break
        gamma = min(candidates, key=lambda c: float(f_p1.subs(g, c)))
        x_val = x_val + gamma * Sx
        y_val = y_val + gamma * Sy
        z_val = z(x_val, y_val)
        history_x.append(x_val)
        history_y.append(y_val)
        history_z.append(z_val)
        print(f"V bode {x_val} {y_val} ma funckcia hodnotu: {z_val}")

    return history_x, history_y, history_z


if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    f = x**2 - 4*x + y**2 - y - x*y
    # f = (x - y)/(x**2 + y**2 + 2)
    hx, hy, hz = gradient_descent_normalized(f, (5, 5))

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    fig = plt.figure(figsize=(24, 12))

    # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-5.5, 5.5, 50)
    Y = np.linspace(-5.5, 5.5, 50)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 - 4*X + Y**2 - Y - X*Y
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