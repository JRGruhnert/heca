import numpy as np
from scipy.optimize import minimize
import cvxpy as cp


def _scales(var: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(var, float), 1e-15))


def _slsqp(m1, s1, m2, s2, chi: float, cap: float, starts) -> tuple[float, np.ndarray]:
    """Solve the convex programme with SLSQP when no conic solver is installed."""

    def constraints(z: np.ndarray) -> np.ndarray:
        t = z[-1]
        x = z[:-1]
        return np.concatenate(
            [
                [
                    t * chi - np.linalg.norm((x - m1) / s1),
                    t * chi - np.linalg.norm((x - m2) / s2),
                ],
                t * cap - np.abs(x - m1) / s1,
                t * cap - np.abs(x - m2) / s2,
            ]
        )

    best = None
    for start in starts:
        result = minimize(
            lambda z: z[-1],
            np.concatenate([np.asarray(start, float), [1.0]]),
            method="SLSQP",
            bounds=[(None, None)] * (len(m1) + 1),
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"maxiter": 120, "ftol": 1e-9},
        )
        if result.success and np.all(constraints(result.x) >= -1e-9):
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        return np.inf, None
    return float(best.fun), np.asarray(best.x[:-1], float)


def _cvxpy(m1, s1, m2, s2, chi: float, cap: float) -> tuple[float, np.ndarray] | None:
    """The same programme handed to a conic solver, when one is available."""
    d = len(m1)
    z = cp.Variable(d)
    t = cp.Variable(nonneg=True)
    constraints = []
    for m, s in ((m1, s1), (m2, s2)):
        constraints += [
            cp.norm((z - m) / s, 2) <= t * chi,
            cp.abs((z - m) / s) <= t * cap,
        ]
    problem = cp.Problem(cp.Minimize(t), constraints)
    for solver in ("CLARABEL", "SCS", "ECOS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except Exception:  # a solver that trips over the cone is not the answer
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and z.value is not None:
            return float(t.value), np.asarray(z.value, float)
    return None


def violation(pose, mu1, var1, mu2, var2, chi: float, cap: float) -> float:
    """The gate's own measure at ``pose``: > 1 means outside a region."""
    worst = 0.0
    for mu, var in ((mu1, var1), (mu2, var2)):
        z = np.abs(pose - mu) / _scales(var)
        worst = max(worst, float(z.max()) / cap, float(np.linalg.norm(z)) / chi)
    return worst


def _line_grid(mu1, mu2, steps: int = 21) -> list[np.ndarray]:
    return [mu1 + f * (mu2 - mu1) for f in np.linspace(0.0, 1.0, steps)]


def minimax_point(
    mu1: np.ndarray,
    var1: np.ndarray,
    mu2: np.ndarray,
    var2: np.ndarray,
    chi: float,
    cap: float,
    steps: int = 21,
) -> tuple[float, np.ndarray | None]:
    """``(t, z)``: the smallest shared violation and the pose reaching it.

    The conic solver answers exactly when it is installed. Without one, SLSQP is
    warm-started from a cheap line grid and its answer is only kept when it beats
    that grid, so the fallback is never worse than sampling the line -- and ``t``
    is always the measured violation, never a solver-reported number.
    """
    m1 = np.asarray(mu1, float)
    m2 = np.asarray(mu2, float)
    s1 = _scales(var1)
    s2 = _scales(var2)
    exact = _cvxpy(m1, s1, m2, s2, chi, cap)
    if exact is not None:
        return exact

    seeds = _line_grid(m1, m2, steps) + [0.5 * (m1 + m2), m1, m2]
    best = min(
        ((violation(z, m1, var1, m2, var2, chi, cap), z) for z in seeds),
        key=lambda item: item[0],
    )
    # only a couple of warm starts: the grid already bounds the answer
    t, pose = _slsqp(m1, s1, m2, s2, chi, cap, [best[1], 0.5 * (m1 + m2)])
    if pose is None:
        return best
    measured = violation(pose, m1, var1, m2, var2, chi, cap)
    if measured < best[0]:
        return measured, pose
    return best
