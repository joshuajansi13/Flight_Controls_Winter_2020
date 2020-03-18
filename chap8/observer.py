"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        2/27/2020 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.tools import RotationBody2Vehicle
from tools.wrap import wrap
import parameters.aerosonde_parameters as MAV
from scipy import stats

from message_types.msg_state import msg_state

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements, sim_time):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.static_pressure) / (MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(2.0*self.lpf_diff.update(measurements.diff_pressure) / MAV.rho)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements, sim_time)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha*self.y + (1.0-self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        self.Q = 1e-10 * np.eye(2)                         # tuning param
        self.Q_gyro = np.eye(4) * SENSOR.gyro_sigma**2
        self.R_accel = np.eye(3) * SENSOR.accel_sigma**2
        self.N = 5  # number of prediction step per sample
        self.xhat = np.array([[MAV.phi0], [MAV.theta0]]) # initial state: phi, theta
        self.P = np.eye(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement, sim_time):
        self.propagate_model(state)
        # if sim_time > 1:
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        p = state.p
        q = state.q
        r = state.r
        phi = x.item(0)
        theta = x.item(1)
        _f = np.array([[p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],  # phidot
                      [q*np.cos(phi) - r*np.sin(phi)]])                                  # thetadot
        return _f

    def h(self, x, state):
        # measurement model y
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        phi = x.item(0)
        theta = x.item(1)

        _h = np.array([[q*Va*np.sin(theta) + MAV.gravity*np.sin(theta)],                                   # x_accel
                      [r*Va*np.cos(theta) - p*Va*np.sin(theta) - MAV.gravity*np.cos(theta)*np.sin(phi)],  # y_accel
                      [-q*Va*np.cos(theta) - MAV.gravity*np.cos(theta)*np.cos(phi)]])                     # z_accel
        return _h

    def propagate_model(self, state):
        # model propagation
        phi = state.phi
        theta = state.theta
        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + self.Ts * self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G = np.array([[1.0, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta), 0.0],
                         [0.0, np.cos(phi), -np.sin(phi), 0.0]])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(2) + A*self.Ts  + A @ A * self.Ts**2 / 2.0
            # G_d =
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2 * (self.Q + G @ self.Q_gyro @ G.T)

    def measurement_update(self, state, measurement):
        # measurement updates
        # self.accel_threshold = stats.chi2.isf(q=0.01, df=3)
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T
        # for i in range(0, 3):
        # if np.abs(y[i]-h[i,0]) < threshold:
        S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        if stats.chi2.sf((y-h).T @ S_inv @ (y-h), df=3) < 0.01:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(2) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_accel @ L.T
            self.xhat = self.xhat + L @ (y-h)

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        self.Q = 1e-2 * np.eye(7)
        self.R = np.diag([SENSOR.gps_n_sigma**2, SENSOR.gps_e_sigma**2,
                          SENSOR.gps_course_sigma**2, SENSOR.gps_Vg_sigma**2])
        self.R_pseudo = 0.01 * np.eye(2)
        self.N = 10  # number of prediction step per sample
        self.Ts = SIM.ts_control / self.N
        self.xhat = np.array([[MAV.pn0], [MAV.pe0], [MAV.Va0], [MAV.psi0], [0.0],
                             [0.0], [MAV.psi0]])  # initialize pn, pe, Vg, chi, wn, we, psi
        self.P = np.eye(7)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = state.Va
        theta = state.theta
        phi = state.phi
        q = state.q
        r = state.r
        psidot = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
        _f = np.array([[Vg*np.cos(chi)],  # pndot
                       [Vg*np.sin(chi)],  # pedot
                       [((Va*np.cos(psi)+wn)*(-Va*psidot*np.sin(psi)) + (Va*np.sin(psi)+we)*(Va*psidot*np.cos(psi))) / Vg],  # Vgdot
                       [MAV.gravity*np.tan(phi)*np.cos(chi-psi) / Vg],  # chidot
                       [0],  # wndot
                       [0],  # wedot
                       [psidot]])  # psidot
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        _h = np.array([[x.item(0)],   # pn
                       [x.item(1)],   # pe
                       [x.item(2)],   # Vg
                       [x.item(3)]])  # chi
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangle pseudo measurement
        Va = state.Va
        psi = state.psi
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        _h = np.array([[Va*np.cos(psi) + wn - Vg*np.cos(chi)],
                       [Va*np.sin(psi) + we - Vg*np.sin(chi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + self.Ts * self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # convert to discrete time models
            A_d = np.eye(7) + A*self.Ts + A @ A * self.Ts**2 / 2.0
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2 * self.Q

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([[0, 0]]).T
        # for i in range(0, 2):
        S_inv = np.linalg.inv(self.R_pseudo + C @ self.P @ C.T)
        L = self.P @ C.T @ S_inv
        self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + L @ self.R_pseudo @ L.T
        self.xhat = self.xhat + L @ (y-h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([[measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course]]).T
            y[3, 0] = wrap(y[3, 0], h[3, 0])

            # for i in range(0, 4):
            S_inv = np.linalg.inv(self.R + C @ self.P @ C.T)
            L = self.P @ C.T @ S_inv
            tmp = np.eye(7) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R @ L.T
            self.xhat = self.xhat + L @ (y-h)

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J
