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
    x_prev = x_val
    y_prev = y_val
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
    while abs(x_prev - x_val) + abs(y_prev - y_val) >= 10e-8 or i == 0:
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
        x_prev = x_val
        y_prev = y_val
        x_val = x_n_x
        y_val = x_n_y

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
    import math
    x, y = symbols('x y', real=True)
    f = ((x + 1) ** 2) + 4 * (y - (x ** 3)) ** 2
    f = x ** 4 - 2 * x ** 2 + (y - x) ** 2
    f = math.e ** (x + 1) + math.e ** (-y) + (x - y) ** 2
    f = 20 * (x**2) + y**2  + (x - y + 1)**4
    # f = (x - y)/(x**2 + y**2 + 2)
    u2 = [0, 1]
    u1 = [1, 0]
    hx, hy, hz = powell_method(f, (2, 3), u1, u2)
