from jax import numpy as jnp
import numpy as np


def convergence_rates(E_values, eps_values, show=True):
    from numpy import log

    r = [
        log(E_values[i] / E_values[i - 1]) / log(eps_values[i] / eps_values[i - 1])
        for i in range(1, len(eps_values))
    ]

    if show:
        print("Computed convergence rates: {}".format(r))
    return r


def taylor_test(eval_f, grad, x):
    v = jnp.zeros_like(x)

    for j, v_j in enumerate(v):
        v = v.at[:].set(0.0)
        v = v.at[j].set(1.0)
        for eps_s in [1e-3, 1e-4, 1e-5]:
            delta_y_approx = (eval_f(x + eps_s * v) - eval_f(x - eps_s * v)) / (
                2.0 * (eps_s * v[j])
            )
        print(f"{delta_y_approx} for {eps_s}")
    print(grad)

    print("Running Taylor test")
    J = eval_f(x)
    residuals = []
    epsilons = [0.01 / 2 ** i for i in range(4)]
    v = np.random.rand(len(x))
    for eps in epsilons:
        Jp = eval_f(x + eps * v)
        print(Jp)
        res = abs(Jp - J - eps * grad.dot(v))
        residuals.append(res)

    print("Computed residuals: {}".format(residuals))
    convergence_rates(residuals, epsilons)
