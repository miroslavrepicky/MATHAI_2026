import math

from sympy import symbols, diff, lambdify, solve
import matplotlib.pyplot as plt
import numpy as np

def steepest_gradient(function, x_0, epsilon=10e-6, k_max=100):

    x, y, alpha = symbols('x y alpha', real=True)
    z = lambdify((x, y), function, 'math')

    d_x = diff(function, x)
    d_y = diff(function, y)
    grad_x, grad_y = lambdify((x, y), d_x, 'math'), lambdify((x, y), d_y, 'math')
    x_val, y_val = x_0
    f_x0 = z(x_val, y_val)
    history_x = [x_val]
    history_y = [y_val]
    history_z = [f_x0]
    counter = 0
    while True:
        counter += 1
        alpha_0 = -lambdify((x, y), d_x, 'math')(x_val, y_val), -lambdify((x, y), d_y, 'math')(x_val, y_val)
        x_1 = x_val + alpha * alpha_0[0]
        y_1 = y_val + alpha * alpha_0[1]
        # print(x_1, "; " ,y_1)
        f_alpha = function.subs({x: x_1, y: y_1})
        # print(f_alpha)
        f1_alpha = diff(f_alpha, alpha)
        a = solve(f1_alpha, alpha)
        # print(a)
        f2_alpha = diff(f1_alpha, alpha)
        f2 = lambdify(alpha, f2_alpha, 'math')
        # print(f2_alpha)
        v_a = a[0]
        for value in a:
            if f2(value) > 0 and value > 0:
                v_a = value
        # print(v_a)
        new_x_val, new_y_val = lambdify(alpha, x_1, 'math')(v_a), lambdify(alpha, y_1, 'math')(v_a)
        f_x1 = z(new_x_val, new_y_val)
        # print(new_x_val)
        # print(new_y_val)
        # print(f_x1)
        history_x.append(new_x_val)
        history_y.append(new_y_val)
        history_z.append(f_x1)

        # Kontrola
        # 1. norma gradientu < epsilon
        if math.sqrt(grad_x(new_x_val, new_y_val) ** 2 + grad_y(new_x_val, new_y_val) ** 2) < epsilon:
            break
        if math.sqrt((new_x_val - x_val) ** 2 + (new_y_val - y_val) ** 2) < epsilon:
            break
        if abs(f_x1 - f_x0) < epsilon:
            break
        if counter >= k_max:
            break
        x_val = new_x_val
        y_val = new_y_val
        f_x0 = f_x1
    return history_x, history_y, history_z




if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    f = 2 * x**2 + x*y + 3*y**2 - 4*x + 2*y
    hx, hy, hz = steepest_gradient(f, (0, 0))

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    fig = plt.figure(figsize=(12, 6))

    # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-2.5, 2.5, 50)
    Y = np.linspace(-2.5, 2.5, 50)
    X, Y = np.meshgrid(X, Y)
    Z = 2 * X ** 2 + X*Y + 3 * Y ** 2 - 4*X + 2*Y

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