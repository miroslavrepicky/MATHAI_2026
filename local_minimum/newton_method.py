from sympy import symbols, diff, lambdify, Matrix, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin as m_sin
import math

from math import sqrt



def newton_method(function ,start_point):
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
    d_xx = diff(d_x, x)
    d_yy = diff(d_y, y)
    d_xy = diff(d_x, y)
    d_yx = diff(d_y, x)
    h = Matrix(2, 2, [d_xx, d_xy, d_yx, d_yy])
    h1 = h.inv()
    iteracia = 0
    while abs(x_prev - x_val) + abs(y_prev - y_val) >= 10e-8 or iteracia == 0:
        iteracia += 1
        try:
            h1_val = h1.subs({x: x_val, y: y_val})
            gx = d_x.subs({x: x_val, y: y_val})
            gy = d_y.subs({x: x_val, y: y_val})

            new_x = float(x_val - (h1_val[0, 0] * gx + h1_val[0, 1] * gy))
            new_y = float(y_val - (h1_val[1, 0] * gx + h1_val[1, 1] * gy))
        except (ZeroDivisionError, TypeError):
            print(f"Hessián je v bode ({x_val}, {y_val}) singulárny, metóda sa zastavuje.")
            break

        z_prev = z_val
        x_prev = x_val
        y_prev = y_val
        x_val, y_val = new_x, new_y
        try:
            z_val = z(x_val, y_val)
        except:
            print("NEVYSLO")
            pass



        print(f"ITERACIA: {iteracia} | V bode {x_val} {y_val} ma funckcia hodnotu: {z_val}")

    return history_x, history_y, history_z


if __name__ == '__main__':
    x, y = symbols('x y', real=True)
    #f = x**2 + 2*y**2
    # f = ((x + 1) ** 2) + 4 * (y - (x**3)) ** 2

    # f = x ** 4 - 2 * x ** 2 + (y - x) ** 2
    # f = math.e**(x+1) + math.e**(-y) + (x - y)**2
    f = (x**6 / 6) - 1.25 * x**4 + 2 * x ** 2 + (y - (x/2))**2
    hx, hy, hz = newton_method(f, (3, 1))

    # --- VYKRESLENIE POMOCOU MATPLOTLIB ---