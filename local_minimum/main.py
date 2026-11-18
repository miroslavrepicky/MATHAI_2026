"""
Porovnanie optimalizačných algoritmov na testovacích funkciách.

Súbory musia byť v jednom priečinku:
    steepest_gradient.py, newton_method.py, powell_method.py, nelder_mead.py

Spustenie:  python main.py
"""
import io
import math
import time
import queue
import multiprocessing as mp
from contextlib import redirect_stdout, nullcontext

from sympy import symbols, lambdify
from prettytable import PrettyTable

from steepest_gradient import steepest_gradient
from newton_method import newton_method
from powell_method import powell_method
from nelder_mead import nelder_mead

# ----------------------------------------------------------------------------
# NASTAVENIA
# ----------------------------------------------------------------------------
TIMEOUT = 60000        # max. čas (s) na jeden beh jedného algoritmu
VERBOSE = False     # True = vypisovať aj priebeh iterácií z jednotlivých algoritmov
NM_STEP = 1.0       # veľkosť počiatočného simplexu pre Nelder-Mead
TOL = 1e-2          # vzdialenosť od známeho minima, do ktorej beh považujeme za "OK"

# Ktoré algoritmy / funkcie spustiť (None = všetky, inak zoznam názvov)
ONLY_ALGORITHMS = None   # napr. ["Newton", "Powell"]
ONLY_FUNCTIONS = None    # napr. ["Booth", "Beale"]

x, y = symbols('x y', real=True)

# ----------------------------------------------------------------------------
# TESTOVACIE FUNKCIE: názov, výraz, štartovací bod, známe minimá
# ----------------------------------------------------------------------------
FUNCTIONS = [
    dict(name="Sféra",
         f=x**2 + y**2,
         start=(3, 2),
         minima=[(0, 0)]),
    dict(name="Pretiahnutá kvadratická",
         f=x**2 + 25 * y**2,
         start=(3, 1),
         minima=[(0, 0)]),
    dict(name="Booth",
         f=(x + 2*y - 7)**2 + (2*x + y - 5)**2,
         start=(0, 0),
         minima=[(1, 3)]),
    dict(name="Matyas",
         f=0.26 * (x**2 + y**2) - 0.48 * x * y,
         start=(5, -3),
         minima=[(0, 0)]),
    dict(name="Rosenbrock",
         f=(1 - x)**2 + 100 * (y - x**2)**2,
         start=(-1.2, 1),
         minima=[(1, 1)]),
    dict(name="Himmelblau",
         f=(x**2 + y - 11)**2 + (x + y**2 - 7)**2,
         start=(0, 0),
         minima=[(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]),
    dict(name="Beale",
         f=(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
         start=(1, 1),
         minima=[(3, 0.5)]),
]


# ----------------------------------------------------------------------------
# RUNNERY: každý vráti (x, y, f(x, y), počet_iterácií)
# ----------------------------------------------------------------------------
def _f_value(f, px, py):
    return lambdify((x, y), f, 'math')(px, py)


def run_steepest(f, start):
    hx, hy, hz = steepest_gradient(f, start)
    return hx[-1], hy[-1], hz[-1], len(hx) - 1


def run_newton(f, start):
    hx, hy, hz = newton_method(f, start)
    return hx[-1], hy[-1], hz[-1], len(hx) - 1


def run_powell(f, start):
    hx, hy, hz = powell_method(f, start, [0, 1], [1, 0])
    return hx[-1], hy[-1], hz[-1], len(hx) - 1


def run_nelder_mead(f, start):
    # Nelder-Mead potrebuje 3 body -> z jedného štartu spravíme pravouhlý trojuholník
    sx, sy = start
    pts = [(sx, sy), (sx + NM_STEP, sy), (sx, sy + NM_STEP)]
    hist = nelder_mead(f, pts)          # tvar (iterácie, 4, 2); [0] = najlepší vrchol
    best = hist[-1][0]
    return float(best[0]), float(best[1]), _f_value(f, best[0], best[1]), len(hist) - 1


RUNNERS = {
    "Steepest descent": run_steepest,
    "Newton": run_newton,
    "Powell": run_powell,
    "Nelder-Mead": run_nelder_mead,
}


# ----------------------------------------------------------------------------
# SPÚŠŤANIE S TIMEOUTOM (každý beh v samostatnom procese)
# ----------------------------------------------------------------------------
def _worker(q, algo_name, f, start):
    try:
        out = nullcontext() if VERBOSE else redirect_stdout(io.StringIO())
        t0 = time.perf_counter()
        with out:
            res = RUNNERS[algo_name](f, start)
        q.put(("ok", res, time.perf_counter() - t0))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}", 0.0))


def run_with_timeout(algo_name, f, start):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q, algo_name, f, start))
    p.start()
    try:
        status, payload, elapsed = q.get(timeout=TIMEOUT)
    except queue.Empty:
        p.terminate()
        p.join()
        return "timeout", None, TIMEOUT
    p.join()
    return status, payload, elapsed


# ----------------------------------------------------------------------------
# VYHODNOTENIE
# ----------------------------------------------------------------------------
def nearest_minimum_distance(px, py, minima):
    return min(math.hypot(px - mx, py - my) for mx, my in minima)


def main():
    algos = [a for a in RUNNERS if ONLY_ALGORITHMS is None or a in ONLY_ALGORITHMS]
    funcs = [fn for fn in FUNCTIONS if ONLY_FUNCTIONS is None or fn["name"] in ONLY_FUNCTIONS]

    summary = PrettyTable(["Funkcia"] + algos)
    summary.align = "l"

    for fn in funcs:
        print(f"\n=== {fn['name']} ===")
        print(f"f(x, y) = {fn['f']}")
        print(f"štart: {fn['start']}, známe minimá: {fn['minima']}")

        table = PrettyTable(["Algoritmus", "Výsledný bod", "f(bod)",
                             "Vzd. od minima", "Iterácie", "Čas [s]", "Stav"])
        table.align = "r"
        table.align["Algoritmus"] = "l"
        table.align["Stav"] = "l"
        summary_row = [fn["name"]]

        for algo in algos:
            status, payload, elapsed = run_with_timeout(algo, fn["f"], fn["start"])

            if status == "ok":
                px, py, fv, iters = payload
                dist = nearest_minimum_distance(px, py, fn["minima"])
                ok = dist < TOL
                label = "OK" if ok else "mimo minima"
                table.add_row([algo, f"({px:.5f}, {py:.5f})", f"{fv:.3e}",
                               f"{dist:.2e}", iters, f"{elapsed:.3f}", label])
                summary_row.append(f"{label} ({iters} it., {elapsed:.2f}s)")
            elif status == "timeout":
                table.add_row([algo, "-", "-", "-", "-", f">{TIMEOUT}", "TIMEOUT"])
                summary_row.append("TIMEOUT")
            else:
                table.add_row([algo, "-", "-", "-", "-", "-", f"CHYBA: {payload}"[:60]])
                summary_row.append("CHYBA")

        print(table)
        summary.add_row(summary_row)

    print("\n=== SÚHRN ===")
    print(summary)


if __name__ == '__main__':
    main()
