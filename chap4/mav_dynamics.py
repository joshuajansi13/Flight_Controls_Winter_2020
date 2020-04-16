"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

"""
import sys

sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Rotation, Quaternion2Euler, \
    RotationVehicle2Body


class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                                [MAV.pe0],  # (1)
                                [MAV.pd0],  # (2)
                                [MAV.u0],  # (3)
                                [MAV.v0],  # (4)
                                [MAV.w0],  # (5)
                                [MAV.e0],  # (6)
                                [MAV.e1],  # (7)
                                [MAV.e2],  # (8)
                                [MAV.e3],  # (9)
                                [MAV.p0],  # (10)
                                [MAV.q0],  # (11)
                                [MAV.r0]])  # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array(
            [[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        self.thrust = 0
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''

        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1,
                               forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2,
                               forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6][0] = self._state.item(6) / normE
        self._state[7][0] = self._state.item(7) / normE
        self._state[8][0] = self._state.item(8) / normE
        self._state[9][0] = self._state.item(9) / normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)

        u = state.item(3)
        v = state.item(4)
        w = state.item(5)

        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)

        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pn_dot = (e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2) * u + 2 * (
                    e1 * e2 - e3 * e0) * v + 2 * (e1 * e3 + e2 * e0) * w
        pe_dot = 2 * (e1 * e2 + e3 * e0) * u + (
                    e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2) * v + 2 * (
                             e2 * e3 - e1 * e0) * w
        pd_dot = 2 * (e1 * e3 - e2 * e0) * u + 2 * (e2 * e3 + e1 * e0) * v + (
                    e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2) * w

        # position dynamics
        u_dot = (r * v - q * w) + fx / MAV.mass
        v_dot = (p * w - r * u) + fy / MAV.mass
        w_dot = (q * u - p * v) + fz / MAV.mass

        # rotatonal dynamics
        p_dot = (MAV.gamma1 * p * q - MAV.gamma2 * q * r) + (
                    MAV.gamma3 * l + MAV.gamma4 * n)
        q_dot = (MAV.gamma5 * p * r - MAV.gamma6 * (p ** 2 - r ** 2)) + (
                    m / MAV.Jy)
        r_dot = (MAV.gamma7 * p * q - MAV.gamma1 * q * r) + (
                    MAV.gamma4 * l + MAV.gamma8 * n)

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot,
                           r_dot]]).T

        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6, 1))):
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)

        phi, theta, psi = Quaternion2Euler(np.array([e0, e1, e2, e3]))
        Rvb = RotationVehicle2Body(phi, theta, psi)

        # calculate wind components
        self._wind = wind[0:3]
        wind_combined = Rvb @ wind[0:3] + wind[3:6]
        # compute relative airspeed components
        self._ur = self._state.item(3) - wind_combined.item(0)
        self._vr = self._state.item(4) - wind_combined.item(1)
        self._wr = self._state.item(5) - wind_combined.item(2)

        self.Vg = Rvb.transpose() @ self._state[3:6]  # In vehicle frame
        self._Vg = np.linalg.norm(self.Vg)

        # compute airspeed
        self._Va = np.sqrt(self._ur ** 2 + self._vr ** 2 + self._wr ** 2)
        # compute angle of attack
        self._alpha = np.arctan2(self._wr, self._ur)
        # compute sideslip angle
        if self._Va == 0.0:
            self._beta = 0.0
        else:
            self._beta = np.arcsin(self._vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        mass = MAV.mass
        g = MAV.gravity
        rho = MAV.rho
        S = MAV.S_wing
        a = self._alpha
        beta = self._beta
        b = MAV.b
        c = MAV.c
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
        Va = self._Va

        # control surface offsets
        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        # min and max values for delta_t
        if delta_t < 0:
            delta_t = 0.0
        elif delta_t > 1:
            delta_t = 1.0
        else:
            delta_t = delta_t

        # drag coefficients
        C_D_q = MAV.C_D_q
        C_L_q = MAV.C_L_q
        C_D_de = MAV.C_D_delta_e
        C_L_de = MAV.C_L_delta_e
        C_Y_0 = MAV.C_Y_0
        C_Y_b = MAV.C_Y_beta
        C_Y_p = MAV.C_Y_p
        C_Y_r = MAV.C_Y_r
        C_Y_da = MAV.C_Y_delta_a
        C_Y_dr = MAV.C_Y_delta_r
        C_l_0 = MAV.C_ell_0
        C_l_b = MAV.C_ell_beta
        C_l_p = MAV.C_ell_p
        C_l_r = MAV.C_ell_r
        C_l_da = MAV.C_ell_delta_a
        C_l_dr = MAV.C_ell_delta_r
        C_m_0 = MAV.C_m_0
        C_m_a = MAV.C_m_alpha
        C_m_q = MAV.C_m_q
        C_m_de = MAV.C_m_delta_e
        C_n_0 = MAV.C_n_0
        C_n_b = MAV.C_n_beta
        C_n_p = MAV.C_n_p
        C_n_r = MAV.C_n_r
        C_n_da = MAV.C_n_delta_a
        C_n_dr = MAV.C_n_delta_r

        f_g = Quaternion2Rotation(self._state[6:10]).transpose() @ np.array(
            [[0], [0], [mass * g]])  # gravitational force
        f_n = mass * g

        # Propeller Thrust Calculations
        # map delta throttle command (0 to 1) into motor input voltage
        V_in = MAV.V_max * delta_t

        # Quadratic formula to solve for motor speed
        a1 = MAV.rho * MAV.D_prop ** 5 / ((2.0 * np.pi) ** 2) * MAV.C_Q0
        b1 = MAV.rho * MAV.D_prop ** 4 / (2.0 * np.pi) * MAV.C_Q1 * self._Va + (
                    MAV.KQ ** 2) / MAV.R_motor
        c1 = MAV.rho * MAV.D_prop ** 3 * MAV.C_Q2 * self._Va ** 2 - MAV.KQ / MAV.R_motor * V_in + MAV.KQ * MAV.i0

        # Consider only positive root
        Omega_op = (-b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / (2.0 * a1)

        # compute advance ratio
        J_op = 2 * np.pi * self._Va / (Omega_op * MAV.D_prop)

        # compute non-dimensionalized coefficients of thrust and torque
        C_T = MAV.C_T2 * J_op ** 2 + MAV.C_T1 * J_op + MAV.C_T0
        C_Q = MAV.C_Q2 * J_op ** 2 + MAV.C_Q1 * J_op + MAV.C_Q0

        # add thrust and torque due to propeller
        n = Omega_op / (2.0 * np.pi)
        f_p = np.array([[MAV.rho * n ** 2 * MAV.D_prop ** 4 * C_T], [0], [0]])
        m_p = np.array([[-MAV.rho * n ** 2 * MAV.D_prop ** 5 * C_Q], [0], [0]])

        # # Alternative Simplified Thrust model
        # f_p = (0.5*MAV.rho*MAV.S_prop*MAV.C_prop) * np.array([[(MAV.k_motor*delta_t)**2 - self.msg_true_state.Va**2], [0], [0]])  # prop thrust
        # m_p = np.array([[-MAV.kTp*(MAV.kOmega*delta_t)**2], [0], [0]])  # prop torque

        C_D = lambda alpha: MAV.C_D_p + (
                    MAV.C_L_0 + MAV.C_L_alpha * alpha) ** 2 / (
                                        np.pi * MAV.e * MAV.AR)
        sig = lambda alpha: (1 + np.exp(-MAV.M * (alpha - MAV.alpha0)) + np.exp(
            MAV.M * (alpha + MAV.alpha0))) / ((1 + np.exp(
            -MAV.M * (alpha - MAV.alpha0))) * (1 + np.exp(
            MAV.M * (alpha + MAV.alpha0))))
        C_L = lambda alpha: (1 - sig(alpha)) * (
                    MAV.C_L_0 + MAV.C_L_alpha * alpha) + sig(alpha) * (
                                        2 * np.sign(alpha) * np.sin(
                                    alpha) ** 2 * np.cos(alpha))
        C_X = lambda alpha: -C_D(alpha) * np.cos(alpha) + C_L(alpha) * np.sin(
            alpha)
        C_X_q = lambda alpha: -C_D_q * np.cos(alpha) + C_L_q * np.sin(alpha)
        C_X_de = lambda alpha: -C_D_de * np.cos(alpha) + C_L_de * np.sin(alpha)
        C_Z = lambda alpha: -C_D(alpha) * np.sin(alpha) - C_L(alpha) * np.cos(
            alpha)
        C_Z_q = lambda alpha: -C_D_q * np.sin(alpha) - C_L_q * np.cos(alpha)
        C_Z_de = lambda alpha: -C_D_de * np.sin(alpha) - C_L_de * np.cos(alpha)

        if Va == 0.0:
            f_a = np.array([[0.], [0.], [0.]])
            # f_p = np.array([[0.], [0.], [0.]])
        else:
            f_a = (0.5 * rho * Va ** 2 * S) * np.array(
                [[C_X(a) + C_X_q(a) * (c / (2 * Va)) * q + C_X_de(a) * delta_e],
                 [C_Y_0 + C_Y_b * beta + C_Y_p * (b / (2 * Va)) * p + C_Y_r * (b / (
                             2 * Va)) * r + C_Y_da * delta_a + C_Y_dr * delta_r],
                 [C_Z(a) + C_Z_q(a) * (c / (2 * Va)) * q + C_Z_de(a) * delta_e]])

        f_total = f_g + f_a + f_p

        self.thrust = f_p.item(0)
        self._forces = np.array(
            [[f_total.item(0)], [f_total.item(1)], [f_total.item(2)]])

        if Va == 0.0:
            m_a = np.array([[0.], [0.], [0.]])
        else:
            m_a = (0.5 * rho * Va ** 2 * S) * \
                  np.array([[b * (C_l_0 + C_l_b * beta + C_l_p * (b / (2 * Va)) * p + C_l_r *
                                  (b / (2 * Va)) * r + C_l_da * delta_a + C_l_dr * delta_r)],
                            [c * (C_m_0 + C_m_a * a + C_m_q * (c / (2 * Va)) * q + C_m_de * delta_e)],
                            [b * (C_n_0 + C_n_b * beta + C_n_p * (b / (2 * Va)) * p + C_n_r *
                                  (b / (2 * Va)) * r + C_n_da * delta_a + C_n_dr * delta_r)]])

        m_total = m_a + m_p

        if self._state.item(2) < 1:
            # mx_ground = m_total.item(0)
            # mz_ground = m_total.item(2)
            return np.array(
                [[f_total.item(0), f_total.item(1), f_total.item(2) - f_n,
                  0., m_total.item(1), 0.]]).T
        else:
            return np.array(
                [[f_total.item(0), f_total.item(1), f_total.item(2) - f_n,
                  m_total.item(0), m_total.item(1), m_total.item(2)]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = np.sqrt(
            self.Vg.item(0) ** 2 + self.Vg.item(1) ** 2 + self.Vg.item(2) ** 2)
        self.msg_true_state.gamma = np.arctan2(-self.Vg[2], np.sqrt(
            self.Vg[0] ** 2 + self.Vg[1] ** 2))
        self.msg_true_state.chi = np.arctan2(self.Vg[1], self.Vg[0])[0]
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)