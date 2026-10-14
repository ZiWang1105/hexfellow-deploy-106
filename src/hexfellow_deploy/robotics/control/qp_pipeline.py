"""QP assembly shared by preparation and tracking; no state integration or dispatch."""

import numpy as np
import pink
import pinocchio as pin
import qpsolvers


def reduced_problem(problem, active):
    """Eliminate inactive displacements, exactly zero, from the optimization."""
    active = np.asarray(active, dtype=int)
    size = len(problem.q)
    inactive = np.setdiff1d(np.arange(size), active)
    for bounds, sign in ((problem.lb, 1), (problem.ub, -1)):
        if bounds is not None and np.any(sign * bounds[inactive] > 0):
            raise ValueError("inactive coordinate excludes zero displacement")
    a, b = problem.A, problem.b
    g, h = problem.G, problem.h
    if g is not None:
        g = np.asarray(g)[:, active]
        nonzero = np.any(g != 0.0, axis=1)
        if np.any(np.asarray(h)[~nonzero] < -1e-12):
            raise ValueError("inactive coordinate violates a hard inequality")
        g, h = g[nonzero], np.asarray(h)[nonzero]
    if a is not None:
        a = np.asarray(a)[:, active]
        nonzero = np.any(a != 0.0, axis=1)
        if np.any(np.asarray(b)[~nonzero] != 0.0):
            raise ValueError("inactive coordinate has nonzero equality target")
        a, b = a[nonzero], np.asarray(b)[nonzero]
        if len(b):
            _, unique = np.unique(np.column_stack((a, b)), axis=0, return_index=True)
            a, b = a[unique], b[unique]
        else:
            a = b = None
    return qpsolvers.Problem(
        P=np.asarray(problem.P)[np.ix_(active, active)],
        q=problem.q[active],
        G=g,
        h=h,
        A=a,
        b=b,
        lb=None if problem.lb is None else problem.lb[active],
        ub=None if problem.ub is None else problem.ub[active],
    )


def assemble_problem(
    configuration,
    tasks,
    dt,
    regularization,
    limits,
    barriers,
    lock_rows,
    boundary_limits,
    posture_target=None,
    posture_velocity=None,
    task_configuration=None,
    task_objectives=None,
):
    model = configuration.model
    warnings = []
    problem = pink.build_ik(
        configuration,
        tasks if posture_target is None and task_configuration is None else [],
        dt,
        damping=regularization,
        limits=limits,
        barriers=barriers,
    )
    if posture_target is None and task_configuration is not None:
        # Task geometry uses actual feedback. Limits still use their original
        # reference history/configuration, so command continuity is unchanged.
        for task in tasks:
            objective = (task_objectives or {}).get(task)
            hessian, linear = (task.compute_qp_objective(task_configuration)
                               if objective is None else objective)
            problem.P = problem.P + hessian
            problem.q = problem.q + linear
    if posture_target is not None:
        # Unit-gain posture error (1/s) plus the continuous waypoint
        # velocity. Pink's optimization variable is displacement.
        desired = dt * (posture_velocity + pin.difference(model, configuration.q, posture_target))
        problem = qpsolvers.Problem(
            P=np.eye(model.nv),
            q=-desired,
            G=problem.G,
            h=problem.h,
            A=problem.A,
            b=problem.b,
            lb=problem.lb,
            ub=problem.ub,
        )
    if len(lock_rows):
        problem = qpsolvers.Problem(
            P=problem.P,
            q=problem.q,
            G=problem.G,
            h=problem.h,
            A=(lock_rows if problem.A is None else np.vstack((problem.A, lock_rows))),
            b=(
                np.zeros(len(lock_rows)) if problem.b is None else np.concatenate((problem.b, np.zeros(len(lock_rows))))
            ),
            lb=problem.lb,
            ub=problem.ub,
        )
    hard_problem = problem
    if boundary_limits:
        warnings = [limit.compute_qp_inequalities(configuration, dt) for limit in boundary_limits]
        warnings = [item for item in warnings if item is not None]
        if warnings:
            problem = qpsolvers.Problem(
                P=problem.P,
                q=problem.q,
                G=np.vstack((problem.G, *(item[0] for item in warnings))),
                h=np.concatenate((problem.h, *(item[1] for item in warnings))),
                A=problem.A,
                b=problem.b,
                lb=problem.lb,
                ub=problem.ub,
            )
    return problem, hard_problem, warnings
