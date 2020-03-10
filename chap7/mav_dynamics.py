"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/16/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state
from message_types.msg_sensors import msg_sensors

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.tools import Quaternion2Rotation, Quaternion2Euler, RotationVehicle2Body

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
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()

        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize the sensors message
        self.sensors = msg_sensors()

        # initialize true_state message
        self.msg_true_state = msg_state()

        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=np.array([[0.0], [0.0], [0.0], [0.5]]))

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
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_msg_true_state()
        # update sensors
        self.update_sensors()

    def update_sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        # State variables
        fx = self._forces.item(0)
        fy = self._forces.item(1)
        fz = self._forces.item(2)
        phi = self.msg_true_state.phi
        theta = self.msg_true_state.theta
        psi = self.msg_true_state.psi
        p = self.msg_true_state.p
        q = self.msg_true_state.q
        r = self.msg_true_state.r
        h = self.msg_true_state.h
        Va = self.msg_true_state.Va
        pn = self.msg_true_state.pn
        pe = self.msg_true_state.pe
        pd = -self.msg_true_state.h
        wn = self.msg_true_state.wn
        we = self.msg_true_state.we

        # Noise terms
        n_gyro_x = np.random.randn()*SENSOR.gyro_sigma
        n_gyro_y = np.random.randn()*SENSOR.gyro_sigma
        n_gyro_z = np.random.randn()*SENSOR.gyro_sigma
        n_accel_x = np.random.randn()*SENSOR.accel_sigma
        n_accel_y = np.random.randn()*SENSOR.accel_sigma
        n_accel_z = np.random.randn()*SENSOR.accel_sigma
        n_abs_pres = np.random.randn()*SENSOR.static_pres_sigma
        n_diff_pres = np.random.randn()*SENSOR.diff_pres_sigma
        # n_mag =
        n_gps_n = np.random.randn()*SENSOR.gps_n_sigma
        n_gps_e = np.random.randn()*SENSOR.gps_e_sigma
        n_gps_h = np.random.randn()*SENSOR.gps_h_sigma
        n_gps_course = np.random.randn()*SENSOR.gps_course_sigma
        n_gps_Vg = np.random.randn()*SENSOR.gps_Vg_sigma

        # simulate rate gyros(units are rad / sec)
        self.sensors.gyro_x = p + n_gyro_x + SENSOR.gyro_x_bias
        self.sensors.gyro_y = q + n_gyro_y + SENSOR.gyro_y_bias
        self.sensors.gyro_z = r + n_gyro_z + SENSOR.gyro_z_bias

        # simulate accelerometers(units of g)
        self.sensors.accel_x = fx / MAV.mass + MAV.gravity*np.sin(theta) + n_accel_x
        self.sensors.accel_y = fy / MAV.mass - MAV.gravity*np.cos(theta)*np.sin(phi) + n_accel_y
        self.sensors.accel_z = fz / MAV.mass - MAV.gravity*np.cos(theta)*np.cos(phi) + n_accel_z

        # # simulate magnetometers
        # # magnetic field in provo has magnetic declination of 12.5 degrees
        # # and magnetic inclination of 66 degrees
        # R_mag = RotationVehicle2Body(0.0, np.radians(-66), np.radians(12.5))
        # # magnetic field in inertial frame: unit vector
        # mag_inertial =
        # R = Quaternion2Rotation(self._state[6:10]) # body to inertial
        # # magnetic field in body frame: unit vector
        # mag_body =
        # self.sensors.mag_x =
        # self.sensors.mag_y =
        # self.sensors.mag_z =

        # simulate pressure sensors
        beta_abs_pres = 0.0  # temperature related bias drift
        beta_diff_pres = 0.0  # temperature-related bias drift
        self.sensors.static_pressure = MAV.rho*MAV.gravity*h + beta_abs_pres + n_abs_pres
        self.sensors.diff_pressure = MAV.rho*Va**2 / 2 + beta_diff_pres + n_diff_pres

        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            k = SENSOR.gps_beta
            Ts = SENSOR.ts_gps
            self._gps_eta_n = np.exp(-k*Ts)*self._gps_eta_n + n_gps_n
            self._gps_eta_e = np.exp(-k*Ts)*self._gps_eta_e + n_gps_e
            self._gps_eta_h = np.exp(-k*Ts)* self._gps_eta_h + n_gps_h
            self.sensors.gps_n = pn + self._gps_eta_n
            self.sensors.gps_e = pe + self._gps_eta_e
            self.sensors.gps_h = h + self._gps_eta_h
            self.sensors.gps_Vg = np.sqrt((Va*np.cos(psi) + wn)**2 + (Va*np.sin(psi) + we)**2) + n_gps_Vg
            self.sensors.gps_course = np.arctan2(Va*np.sin(psi) + we, Va*np.cos(psi) + wn) + n_gps_course
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        # return self.sensors


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

        #   extract forces/moments
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

        # rotational dynamics
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
        if delta_t < 0:
            delta_t = 0.0

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

        f_a = (0.5 * rho * Va ** 2 * S) * np.array(
            [[C_X(a) + C_X_q(a) * (c / (2 * Va)) * q + C_X_de(a) * delta_e],
             [C_Y_0 + C_Y_b * beta + C_Y_p * (b / (2 * Va)) * p + C_Y_r * (b / (
                         2 * Va)) * r + C_Y_da * delta_a + C_Y_dr * delta_r],
             [C_Z(a) + C_Z_q(a) * (c / (2 * Va)) * q + C_Z_de(a) * delta_e]])
        f_total = f_g + f_a + f_p
        self.thrust = f_p.item(0)
        self._forces = np.array(
            [[f_total.item(0)], [f_total.item(1)], [f_total.item(2)]])

        m_a = (0.5 * rho * Va ** 2 * S) * \
              np.array([[b * (C_l_0 + C_l_b * beta + C_l_p * (b / (2 * Va)) * p + C_l_r *
                              (b / (2 * Va)) * r + C_l_da * delta_a + C_l_dr * delta_r)],
                        [c * (C_m_0 + C_m_a * a + C_m_q * (c / (2 * Va)) * q + C_m_de * delta_e)],
                        [b * (C_n_0 + C_n_b * beta + C_n_p * (b / (2 * Va)) * p + C_n_r *
                              (b / (2 * Va)) * r + C_n_da * delta_a + C_n_dr * delta_r)]])
        m_total = m_a + m_p

        return np.array([[f_total.item(0), f_total.item(1), f_total.item(2),
                          m_total.item(0), m_total.item(1), m_total.item(2)]]).T

    # def _motor_thrust_torque(self, Va, delta_t):
    #     return T_p, Q_p

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = np.linalg.norm(pdot)
        self.msg_true_state.gamma = np.arcsin(pdot.item(2) / self.msg_true_state.Vg)
        self.msg_true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
        # self.msg_true_state.bx = SENSOR.gyro_x_bias
        # self.msg_true_state.by = SENSOR.gyro_y_bias
        # self.msg_true_state.bz = SENSOR.gyro_z_bias

