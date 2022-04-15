from jax import numpy as jnp
from jax.experimental import host_callback as hcb
import jax
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from experimental_helpers import (
    interpolate_experimental_data,
    obtain_experiment_cv_data,
)
import argparse
from jax.experimental import stax
from jax import flatten_util
from simulation_parameters import n_cycles, scan_rates, current_units_change, N, n_steps
from jax.scipy import optimize
import pickle
from taylor_test import taylor_test
from jax.config import config
import optax


config.update("jax_enable_x64", True)

h = 1.0 / N
tlimit = 1.0
Dt = tlimit / n_steps
alpha = 1.0
gamma = 0.0

epsi = 0.5


t_array = jnp.linspace(0, tlimit * n_cycles, n_steps * n_cycles + 1)


def supercap_evolution(solution, N, filename="movie.mp4"):
    """Plot evolution of the potentials along the 1D electrode

    Args:
        solution (_type_): _description_
        N (_type_): _description_
        filename (str, optional): _description_. Defaults to "movie.mp4".
    """
    # Plot solutions
    fig, ax = plt.subplots()
    ims = []
    h_arr = np.linspace(0, 1.0, N)
    for solution_i in solution[:-1, :]:
        (plot_c,) = ax.plot(
            h_arr,
            solution_i[:N],
            "g",
            label="concentration",
        )
        (plot_phi1,) = ax.plot(h_arr, solution_i[N : 2 * N], "r", label="Phi 2")
        (plot_phi2,) = ax.plot(h_arr, solution_i[2 * N :], "b", label="Phi 2")
        ims.append([plot_c, plot_phi1, plot_phi2])

    from matplotlib import animation

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=False)
    ani.save(filename)


def hcb_print(string_from_args, *args, **kwargs):
    """Helper for printer optimizer messages via host callbacks. No-op if `verbose`
    is set to `False`."""

    hcb.id_tap(
        lambda args_kwargs, _unused_transforms: print(
            string_from_args(*args_kwargs[0], **args_kwargs[1]),
        ),
        (args, kwargs),
    )


def cv(time):
    """Cyclic voltammetry loading

    Args:
        time ([type]): Time in simulation

    Returns:
        [type]: Potential applied
    """
    return (
        1.0
        / jnp.pi
        * jnp.arcsin(jnp.sin(2 * jnp.pi / (2.0 * tlimit) * (time + 1.5 * tlimit)))
    ) + 1.0 / 2


def create_nn(nodes):
    # Use stax to set up network initialization and evaluation functions
    # Define R^2 -> R^1 function
    def bias_initializer(key, shape, dtype=np.float32):
        return jax.random.uniform(key, shape, dtype, minval=1.0, maxval=1.0)

    net_init, net_apply = stax.serial(
        stax.Dense(nodes, b_init=bias_initializer),
        stax.Gelu,
        stax.Dense(1, b_init=bias_initializer),
    )
    return net_init, net_apply


def get_experimental_current(ud_array, scan_rates):
    """Return the experimental data for the given scan rates
    and interpolated for the values of the potential in ud_array


    Args:
        ud_array (_type_): _description_
        scan_rates (_type_): _description_

    Returns:
        _type_: _description_
    """
    ud_array_one_cyle = ud_array[: n_steps * 2 + 1]
    current_exp_collec = []
    for scan_rate in scan_rates:
        voltage_exp, current_exp = obtain_experiment_cv_data(scan_rate)
        current_exp = interpolate_experimental_data(
            ud_array_one_cyle, voltage_exp, current_exp
        )
        current_exp_collec.append(current_exp)

    return jnp.stack(current_exp_collec)


def main(nodes, output_dir):

    un = jnp.zeros(3 * N)
    un = un.at[:N].set(1.0)  # Set a nonzero concentration to avoid zero jacobian

    t_array = np.linspace(0, tlimit * n_cycles, n_steps * n_cycles + 1)
    ud_array = cv(t_array)

    current_exp_set = get_experimental_current(ud_array, scan_rates)
    max_current_exp_set = jnp.max(jnp.abs(current_exp_set), 1)

    net_init, net_apply = create_nn(nodes)
    # Initialize parameters, not committing to a batch shape
    rng = jax.random.PRNGKey(1)
    in_shape = (N - 2, 2)
    _, net_params = net_init(rng, in_shape)
    _, unravel = flatten_util.ravel_pytree(net_params)

    @jax.jit
    def cp_func(u, net_params):
        """Evaluate the neural network

        Args:
            u (_type_): _description_
            net_params (_type_): _description_

        Returns:
            _type_: _description_
        """
        return jax.vmap(partial(net_apply, net_params))(u).flatten()

    @jax.jit
    def residual(u, un, t, x, scan_rate):
        """Residual of the coupled equations

        Args:
            u (_type_): _description_
            un (_type_): _description_
            t (_type_): _description_
            x (_type_): _description_
            scan_rate (_type_): _description_

        Returns:
            _type_: _description_
        """
        net_params = unravel(x[:-3])
        tc = x[-1] * x[-1]
        kappa = x[-2] * x[-2]
        sigma = x[-3] * x[-3]
        # Solution field
        # c: u[0:N]
        # phi 1: u[N:(2*N)]
        # phi 2: u[(2*N):(3*N)]
        zeta = (sigma) / (1.0 / scan_rate)
        delta = (kappa) / (1.0 / scan_rate)
        # zeta = 0.1
        # delta = 0.01

        # Concentration equations
        def kernel_c(u):
            """Concentration diffusion stencil

            Args:
                u (_type_): _description_

            Returns:
                _type_: _description_
            """
            return u[2:N] - 2.0 * u[1 : N - 1] + u[: N - 2]

        # Phi 1 equations
        def kernel_phi_1(u):
            """Electronic potential diffusion stencil

            Args:
                u (_type_): _description_

            Returns:
                _type_: _description_
            """
            return u[(N + 2) : 2 * N] - 2.0 * u[(N + 1) : 2 * N - 1] + u[N : 2 * N - 2]

        # Phi 2 equations. The inverse comes from the harmonic average.
        def inv_c_i(u):
            return 2.0 / (1.0 / u[2:N] + 1.0 / u[1 : N - 1])

        def inv_c_i_minus(u):
            return 2.0 / (1.0 / u[1 : N - 1] + 1.0 / u[: N - 2])

        def kernel_phi_2(u):
            """Ionic potential diffusion stencil

            Args:
                u (_type_): _description_

            Returns:
                _type_: _description_
            """
            return inv_c_i(u) * (
                u[(2 * N + 2) : 3 * N] - u[(2 * N + 1) : 3 * N - 1]
            ) - inv_c_i_minus(u) * (u[(2 * N + 1) : 3 * N - 1] - u[2 * N : 3 * N - 2])

        # Grab the potentials
        newun = jnp.reshape(un[N:], [N, 2], order="F")
        cp_NN = cp_func(newun, net_params)
        cp_NN = cp_NN * cp_NN

        def cp_current(u):
            """Electronic potential minus ionic potential

            Args:
                u (_type_): _description_

            Returns:
                _type_: _description_
            """
            return u[(N + 1) : 2 * N - 1] - u[2 * N + 1 : 3 * N - 1]

        res = jnp.zeros_like(u)
        # Concentration residual
        res = res.at[1 : (N - 1)].set(
            -alpha * (u[1 : (N - 1)] - un[1 : (N - 1)]) / Dt
            + 1.0 / h ** 2 * ((kernel_c(u) + kernel_c(un)) / 2.0)
            + gamma * cp_NN[1 : N - 1] * (cp_current(u) - cp_current(un)) / Dt
        )
        # Phi 1 residual
        res = res.at[(N + 1) : (2 * N - 1)].set(
            (1.0 - epsi) ** (1.5)
            * 1.0
            / h ** 2
            * ((kernel_phi_1(u) + kernel_phi_1(un)) / 2.0)
            - zeta * cp_NN[1 : N - 1] * (cp_current(u) - cp_current(un)) / Dt
        )
        # Phi 2 residual
        res = res.at[(2 * N + 1) : (3 * N - 1)].set(
            epsi ** (1.5) * 1.0 / h ** 2 * ((kernel_phi_2(u) + kernel_phi_2(un)) / 2.0)
            + delta * cp_NN[1 : N - 1] * (cp_current(u) - cp_current(un)) / Dt
        )

        # Potential 1 boundary conditions
        res = res.at[2 * N - 1].set(u[2 * N - 1] - 0.0)  # Grounded
        res = res.at[N].set(u[N + 1] - u[N])  # Zero flux

        # Potential 2 boundary conditions
        u_bc = cv(t)
        res = res.at[2 * N].set(u[2 * N] - u_bc)  # Applied potential
        res = res.at[3 * N - 1].set(u[3 * N - 1] - u[3 * N - 2])  # Zero flux

        # Concentration boundary conditions
        res = res.at[0].set(u[0] - u[1])  # Zero flux
        res = res.at[N - 1].set(u[N - 1] - u[N - 2])  # Zero flux

        return res

    @jax.jit
    def newton_step(u, un, t, x, scan_rate):
        jacobian = jax.jacfwd(residual, argnums=(0,))
        K = jacobian(u, un, t, x, scan_rate)[0]
        delta_u = jnp.linalg.solve(K, -residual(u, un, t, x, scan_rate))
        u += delta_u
        # hcb_print(
        #    lambda res, t, tc, kappa, sigma: f"Current residual norm at {t}: {res}. tc: {tc}, kappa: {kappa}, sigma: {sigma}",
        #    res=jnp.linalg.norm(residual(u, un, t, net_params, tc, kappa, sigma)),
        #    t=t,
        #    tc=tc,
        #    kappa=kappa,
        #    sigma=sigma,
        # )

        return u

    @jax.custom_vjp
    @jax.jit
    def newton_solver(un, t, x, scan_rate):
        u = jnp.zeros_like(un)
        u = un.at[:N].set(1.0)  # Set a nonzero concentration
        tolerance = 1e-3

        return jax.lax.while_loop(
            lambda u: jnp.linalg.norm(residual(u, un, t, x, scan_rate)) > tolerance,
            lambda u: newton_step(u, un, t, x, scan_rate),
            u,
        )

    def newton_solver_fwd(un, t, x, scan_rate):
        u = newton_solver(un, t, x, scan_rate)
        return u, (un, u, t, x, scan_rate)

    def newton_solver_bwd(res, lmbda):
        un, u, t, x, scan_rate = res
        jacobian = jax.jacfwd(residual, argnums=(0,))
        K = jacobian(u, un, t, x, scan_rate)[0]
        # Boundary conditions
        adjoint = jnp.linalg.solve(K.transpose(), -lmbda)

        _, vjp_un = jax.vjp(lambda un: residual(u, un, t, x, scan_rate), un)
        _, vjp_t = jax.vjp(lambda t: residual(u, un, t, x, scan_rate), t)
        _, vjp_x = jax.vjp(
            lambda x: residual(u, un, t, x, scan_rate),
            x,
        )
        _, vjp_scan_rate = jax.vjp(
            lambda scan_rate: residual(u, un, t, x, scan_rate),
            scan_rate,
        )
        return (
            vjp_un(adjoint)[0],
            vjp_t(adjoint)[0],
            vjp_x(adjoint)[0],
            vjp_scan_rate(adjoint)[0],
        )

    newton_solver.defvjp(newton_solver_fwd, newton_solver_bwd)

    @jax.jit
    def newton_solver_time_step(x, scan_rate, un, t):
        u = newton_solver(un, t, x, scan_rate)
        return (u, u)

    @jax.jit
    def transient_solver(x, scan_rate, un):

        ts_solver = partial(newton_solver_time_step, x, scan_rate)
        _, un = jax.lax.scan(ts_solver, un, t_array)

        return un

    @jax.jit
    def current(un, kappa):
        return -(
            (epsi ** 1.5)
            / (2.0 * h)
            * (un[(n_steps * 2) :, 2 * N + 2] - un[(n_steps * 2) :, 2 * N])
        )

    @jax.jit
    def loss(x, un, current_exp, max_current_exp, scan_rate):
        net_params = unravel(x[:-3])
        tc = x[-1] * x[-1]
        kappa = x[-2] * x[-2]
        sigma = x[-3] * x[-3]
        hcb_print(
            lambda tc, kappa, sigma: f"tc: {tc}, kappa: {kappa}, sigma: {sigma}",
            tc=tc,
            kappa=kappa,
            sigma=sigma,
        )
        return jnp.sum(
            (
                current(transient_solver(x, scan_rate, un), kappa)
                - current_exp * current_units_change
            )
            ** 2
            / (max_current_exp * current_units_change)
        )

    vloss = jax.vmap(loss, in_axes=(None, None, 0, 0, 0))

    @jax.jit
    def lump_loss(x):
        total_loss = jnp.sum(
            vloss(x, un, current_exp_set, max_current_exp_set, scan_rates)
        )
        hcb_print(
            lambda total_loss: f"Current loss {total_loss}", total_loss=total_loss
        )
        return total_loss

    x, unravel = jax.flatten_util.ravel_pytree(net_params)
    # Great values for 500
    tc = 0.001
    kappa = 0.01
    sigma = 0.2
    x0 = jnp.append(x, sigma)
    x0 = jnp.append(x0, kappa)
    x0 = jnp.append(x0, tc)

    # total_loss = loss(x0, un, current_exp_set[0], max_current_exp_set[0], scan_rates[0])
    # loss_grad = jax.grad(loss)(
    #    x0, un, current_exp_set[0], max_current_exp_set[0], scan_rates[0]
    # )
    # eval_f = partial(
    #    loss,
    #    un=un,
    #    current_exp=current_exp_set[0],
    #    max_current_exp=max_current_exp_set[0],
    #    scan_rate=scan_rates[0],
    # )
    # taylor_test(eval_f, loss_grad, x0)

    for scan_rate in scan_rates:
        solution = transient_solver(x0, scan_rate, un)
        current_sim = current(solution, kappa)
        plt.plot(ud_array[n_steps * 2 :], current_sim, "r--")
    plt.plot(ud_array[n_steps * 2 :], current_exp_set.T * current_units_change, "b")
    plt.show()
    # supercap_evolution(solution, N)

    # total_loss = loss(x0, un, current_exp)
    # loss_grad = jax.grad(loss)(x0, un, current_exp)
    # eval_f = partial(loss, un=un, current_exp=current_exp)
    # taylor_test(eval_f, loss_grad, x0)

    opt_algorithm = "notadam"
    print("Start optimization")
    if opt_algorithm == "adam":
        start_learning_rate = 5e-2
        optimizer = optax.adam(start_learning_rate)
        opt_state = optimizer.init((x0,))
        # A simple update loop.

        for i in range(100):
            loss_val, grads = jax.value_and_grad(lump_loss, argnums=(0,))(x0)
            # print(f"Iteration {i}, Loss: {loss_val}")
            updates, opt_state = optimizer.update(grads, opt_state)
            x0 = optax.apply_updates(x0, updates[0])
        x = x0
    else:
        opt_sol = optimize.minimize(
            lump_loss,
            x0,
            method="BFGS",
            options=dict(maxiter=100),
        )
        x = opt_sol.x

    print(f"Final iteration Loss: {lump_loss(x)}")
    net_params = unravel(x[:-3])
    tc = x[-1] * x[-1]
    kappa = x[-2] * x[-2]
    sigma = x[-3] * x[-3]
    print(f"Optimal values: tc:{tc}, kappa:{kappa}, sigma:{sigma}")
    # tc = opt_sol.x[-1]

    for scan_rate in scan_rates:
        current_sim = current(transient_solver(x, scan_rate, un), kappa)
        plt.plot(ud_array[n_steps * 2 :], current_sim)
        with open(f"{output_dir}/current_sim_{scan_rate}.pickle", "wb") as file:
            pickle.dump(current_sim, file)
    plt.plot(ud_array[n_steps * 2 :], current_exp_set.T * current_units_change)
    plt.show()

    with open(f"{output_dir}/nn_trained.pickle", "wb") as file:
        pickle.dump(net_params, file)

    # with open(f"{output_dir}/tc.txt", "w") as file:
    #    file.write(tc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Neural Networks for Supercapacitor modeling")
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        type=str,
        help="Output directory",
        default="./",
    )
    parser.add_argument(
        "--nodes",
        action="store",
        dest="nodes",
        type=int,
        help="Number of nodes in the first layer",
        default=8,
    )
    parser.add_argument(
        "--model",
        action="store",
        dest="model",
        type=str,
        help="Model",
        default="test",
    )

    nodes = 8
    args, unknown = parser.parse_known_args()
    main(nodes, args.output_dir)
