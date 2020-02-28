"""
compute_trim
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from chap5.compute_models import euler_state, quaternion_state

def compute_trim(mav, Va, gamma, R, display=False):
    # define initial state and input
    e = Euler2Quaternion(0.0, gamma, 0.0)
    state0 = np.array([[mav._state.item(0)],  # (0) Position North
                       [mav._state.item(1)],   # (1) Position East
                       [mav._state.item(2)],   # (2) Position Down
                       [Va],    # (3) Velocity body x
                       [0.0],    # (4) Velocity body y
                       [0.0],    # (5) Velocity body z
                       [e.item(0)],    # (6) Quaternion e0
                       [e.item(1)],    # (7) Quaternion e1
                       [e.item(2)],    # (8) Quaternion e2
                       [e.item(3)],    # (9) Quaternion e3
                       [0.0],    # (10) Angular velocity body x
                       [0.0],    # (11) Angular velocity body y
                       [0.0]])   # (12) Angular velocity body z
    delta0 = np.array([[0.0],   # Elevator
                       [0.0],   # Aeleron
                       [0.0],   # Rudder
                       [0.5]])  # Throttle
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma, R),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T

    if display:
        print("Optimized Trim Output")
        print("trim_states: \n", trim_state, "\n")
        print("trim_inputs: \n", trim_input, "\n")

    return trim_state, trim_input

# objective function to be minimized
def trim_objective(x, mav, Va, gamma, R):
    # TODO include turning radius into quaternion trim state
    # Define desired derivatives
    x_dot_star = np.array([[0.0],  # (0) Position North
                       [0.0],   # (1) Position East
                       [-Va*np.sin(gamma)],   # (2) Position Down
                       [0.0],    # (3) Velocity body x
                       [0.0],    # (4) Velocity body y
                       [0.0],    # (5) Velocity body z
                       [0.0],    # (6) Quaternion e0
                       [0.0],    # (7) Quaternion e1
                       [0.0],    # (8) Quaternion e2
                       [0.0],    # (9) Quaternion e3
                       [0.0],    # (10) Angular velocity body x
                       [0.0],    # (11) Angular velocity body y
                       [0.0]])   # (12) Angular velocity body z

    # Calculate current derivatives
    x_star = x[0:13]
    delta_star = x[13:17]
    mav._state = x_star
    mav._update_velocity_data()
    forces_moments_ = mav._forces_moments(delta_star)
    x_dot_current = mav._derivatives(x_star, forces_moments_)
    # Penalty Function
    J = np.linalg.norm(x_dot_star[2:13] - x_dot_current[2:13])**2
    return J

