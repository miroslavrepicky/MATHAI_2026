import math

from sympy import symbols, diff, lambdify
import matplotlib.pyplot as plt
import numpy as np
import math



def gradient_descent(function ,start_point, alpha):
    x, y = symbols('x y', real=True)
    z = lambdify((x, y), function, 'math')
    x_val, y_val = start_point
    x_prev = x_val
    y_prev = y_val
    z_val = z(start_point[0], start_point[1])
    history_x = [x_val]
    history_y = [y_val]
    history_z = [z_val]
    z_prev = z_val + 1
    d_x = diff(function, x)
    d_y = diff(function, y)
    iteracia = 0
    while abs(x_prev - x_val) + abs(y_prev - y_val) >= 10e-8 or iteracia == 0:
        iteracia += 1
        z_prev = z_val
        x_prev = x_val
        y_prev = y_val
        x_val = x_prev - alpha * lambdify((x, y), d_x, 'math')(x_prev, y_prev)
        y_val = y_prev - alpha * lambdify((x, y), d_y, 'math')(x_prev, y_prev)
        z = lambdify((x, y), function, 'math')
        try:
            z_val = z(x_val, y_val)
        except:
            print("-")
        print(f"ITERACIA: {iteracia} | V bode {x_val:.4f} {y_val:.4f} ma funckcia hodnotu: {z_val:.4f}")
        if iteracia == 20000:
            break

    return history_x, history_y, history_z


if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    # f = x**2 + 2*y**2
    # f = (x + 1)** 2 + 4 * (y - x**3)**2
    # f = x ** 4 - 2 * x ** 2 + (y - x)**2
    f = (x ** 6 / 6) - 1.25 * x ** 4 + 2 * x ** 2 + (y - (x / 2)) ** 2
    f = 20 * (x ** 2) + y ** 2 + (x - y + 1) ** 4
    # f = math.e**(x+1) + math.e**(-y) + (x - y)**2
    hx, hy, hz = gradient_descent(f, (2, 3), 0.005)



    # # --- VYKRESLENIE POMOCOU MATPLOTLIB ---
    # fig = plt.figure(figsize=(12, 6))
    #
    # # 1. 3D graf povrchu funkcie + čiarová trajektória zostupu
    # ax1 = fig.add_subplot(121, projection='3d')
    # X = np.linspace(-2.5, 2.5, 50)
    # Y = np.linspace(-2.5, 2.5, 50)
    # X, Y = np.meshgrid(X, Y)
    # Z = X ** 2 + 2 * Y ** 2
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