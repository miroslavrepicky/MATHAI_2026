from gradient_descent import gradient_descent
from gradient_descent_normalized import gradient_descent_normalized
from newton_method import newton_method
from nelder_mead import nelder_mead
from powell_method import powell_method

from sympy import symbols, lambdify


if __name__ == "__main__":
    x, y = symbols('x y', real=True)
    functions = [
        x**2 + y**2,
        x**2 + 25*y**2,
        (x + 2*y - 7)**2 + (2*x + y - 5)**2,
        0.26*(x**2 + y**2) - 0.48*x*y,
        (1 - x)**2 + 100*(y - x**2)**2,
        (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
        (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    ]

    starting_points = [
        [3, 2],
        [3, 1],
        [0, 0],
        [5, -3],
        [-1.2, 1],
        [0, 0],
        [1, 1]
    ]

    minimums = [
        [0, 0],
        [0, 0],
        [1, 3],
        [0, 0],
        [1, 1],
        [3, 2],
        [3, 0.5]
    ]




