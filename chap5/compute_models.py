"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import transfer_function
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts

def compute_tf_model(mav, trim_state, trim_input, display=False):

    # euler states
    e_state = euler_state(trim_state)
    theta = e_state.item(7)


    # control surface inputs
    delta_e_star = trim_input[0, 0]
    delta_t_star = trim_input[3, 0]

    # additional mav variables
    Va = mav._Va
    Va_star = mav._Va
    Vg = mav._Vg
    alpha_star = mav._alpha
    b = MAV.b
    rho = MAV.rho
    S = MAV.S_wing
    mass = MAV.mass
    c = MAV.c
    Jy = MAV.Jy
    g = MAV.gravity

    # -------------- Transfer Functions --------------
    # phi to delta_a
    a_phi_1 = -0.5*rho*Va**2*S*b*MAV.C_p_p*b/(2.0*Va)
    a_phi_2 = 0.5*rho*Va**2*S*b*MAV.C_p_delta_a
    T_phi_delta_a = transfer_function(num=np.array([[a_phi_2]]),
                                     den=np.array([[1, a_phi_1, 0]]),
                                     Ts=Ts)

    # chi to phi
    T_chi_phi = transfer_function(num=np.array([[g/Vg]]),
                                  den=np.array([[1, 0]]),
                                  Ts=Ts)

    # beta to delta_r
    a_b_1 = -rho*Va*S/(2*mass)*MAV.C_Y_beta
    a_b_2 = rho*Va*S/(2*mass)*MAV.C_Y_delta_r
    T_beta_delta_r = transfer_function(num=np.array([[a_b_2]]),
                                       den=np.array([[1, a_b_1]]),
                                       Ts=Ts)

    #theta to delta_e
    a_t_1 = -rho*Va**2*c*S/(2*Jy)*MAV.C_m_q*c/(2.0*Va)
    a_t_2 = -rho*Va**2*c*S/(2*Jy)*MAV.C_m_alpha
    a_t_3 = rho*Va**2*c*S/(2*Jy)*MAV.C_m_delta_e
    T_theta_delta_e = transfer_function(num=np.array([[a_t_3]]),
                                        den=np.array([[1, a_t_1, a_t_2]]),
                                        Ts=Ts)

    # h to theta
    T_h_theta = transfer_function(num=np.array([[Va]]),
                                  den=np.array([[1, 0]]),
                                  Ts=Ts)

    # h to Va
    T_h_Va = transfer_function(num=np.array([[theta]]),
                               den=np.array([[1, 0]]),
                               Ts=Ts)

    # Va to delta_t

    # book approach (non-finite difference)
    # a_v_1 = rho*Va_star*S/mass*(MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e*delta_e_star) + rho*MAV.S_prop/mass*MAV.C_prop*Va_star
    # a_v_2 = rho*MAV.S_prop/mass*MAV.C_prop*MAV.k_motor**2*delta_t_star
    # a_v_3 = g

    # finite difference approach
    a_v_1 = rho*Va_star*S/mass*(MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e*delta_e_star) - 1.0/mass*dT_dVa(mav, Va, delta_t_star)
    a_v_2 = 1.0/mass*dT_ddelta_t(mav, Va, delta_t_star)
    a_v_3 = g * np.cos(theta - alpha_star)
    T_Va_delta_t = transfer_function(num=np.array([[a_v_2]]),
                                     den=np.array([[1, a_v_1]]),
                                     Ts=Ts)

    # Va to theta
    T_Va_theta = transfer_function(num=np.array([[-a_v_3]]),
                                   den=np.array([[1, a_v_1]]),
                                   Ts=Ts)

    if display:
        print("Calculated Transfer Function Forms")
        print("T_phi_delta_a: ")
        T_phi_delta_a.print()
        print("T_chi_phi: ")
        T_chi_phi.print()
        print("T_beta_delta_r: ")
        T_beta_delta_r.print()
        print("T_theta_delta_e: ")
        T_theta_delta_e.print()
        print("T_h_theta: ")
        T_h_theta.print()
        print("T_h_Va: ")
        T_h_Va.print()
        print("T_Va_delta_t: ")
        T_Va_delta_t.print()
        print("T_Va_theta: ")
        T_Va_theta.print()

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input, display=False, euler=False):
    if euler:
        # euler states
        e_state = euler_state(trim_state)
        u_star = e_state.item(3)
        v_star = e_state.item(4)
        w_star = e_state.item(5)
        phi_star = e_state.item(6)
        theta_star = e_state.item(7)
        p_star = e_state.item(9)
        q_star = e_state.item(10)
        r_star = e_state.item(11)

        # control surface inputs
        delta_a_star = trim_input[0][0]
        delta_e_star = trim_input[1][0]
        delta_r_star = trim_input[2][0]
        delta_t_star = trim_input[3][0]

        Va_star = mav._Va
        b = MAV.b
        rho = MAV.rho
        S = MAV.S_wing
        S_prop = MAV.S_prop
        C_prop = MAV.C_prop
        mass = MAV.mass
        alpha = mav._alpha

        c = MAV.c
        Jy = MAV.Jy
        g = MAV.gravity
        k = MAV.k_motor

        # Aerodynamic Coefficients
        C_D = lambda alpha: MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * alpha) ** 2 / (np.pi * MAV.e * MAV.AR)
        sig = lambda alpha: (1 + np.exp(-MAV.M * (alpha - MAV.alpha0)) + np.exp(MAV.M * (alpha + MAV.alpha0))) / ((1 + np.exp(-MAV.M * (alpha - MAV.alpha0))) * (1 + np.exp(MAV.M * (alpha + MAV.alpha0))))
        C_L = lambda alpha: (1 - sig(alpha)) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + sig(alpha) * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))
        C_X_0 = lambda alpha: -C_D(alpha) * np.cos(alpha) + C_L(alpha) * np.sin(alpha)
        C_X_q = lambda alpha: -MAV.C_D_q * np.cos(alpha) + MAV.C_L_q * np.sin(alpha)
        C_X_a = lambda alpha: -MAV.C_D_alpha * np.cos(alpha) + MAV.C_L_alpha * np.sin(alpha)
        C_X_de = lambda alpha: -MAV.C_D_delta_e * np.cos(alpha) + MAV.C_L_delta_e * np.sin(alpha)
        C_Z_0 = lambda alpha: -C_D(alpha) * np.sin(alpha) - C_L(alpha) * np.cos(alpha)
        C_Z_q = lambda alpha: -MAV.C_D_q * np.sin(alpha) - MAV.C_L_q * np.cos(alpha)
        C_Z_a = lambda alpha: -MAV.C_D_alpha * np.sin(alpha) - MAV.C_L_alpha * np.cos(alpha)
        C_Z_de = lambda alpha: -MAV.C_D_delta_e * np.sin(alpha) - MAV.C_L_delta_e * np.cos(alpha)

        C_Y_p = MAV.C_Y_p
        C_Y_r = MAV.C_Y_r
        C_Y_0 = MAV.C_Y_0
        C_Y_b = MAV.C_Y_beta
        C_Y_dr = MAV.C_Y_delta_r
        C_Y_da = MAV.C_Y_delta_a
        C_p_p = MAV.C_p_p
        C_p_r = MAV.C_p_r
        C_p_0 = MAV.C_p_0
        C_p_b = MAV.C_p_beta
        C_p_dr = MAV.C_p_delta_r
        C_p_da = MAV.C_p_delta_a
        C_r_p = MAV.C_r_p
        C_r_r = MAV.C_r_r
        C_r_0 = MAV.C_r_0
        C_r_b = MAV.C_r_beta
        C_r_dr = MAV.C_r_delta_r
        C_r_da = MAV.C_r_delta_a
        C_X_0 = C_X_0(alpha)
        C_X_a = C_X_a(alpha)
        C_X_q  = C_X_q(alpha)
        C_X_de = C_X_de(alpha)
        C_Z_0 = C_Z_0(alpha)
        C_Z_a = C_Z_a(alpha)
        C_Z_q = C_Z_q(alpha)
        C_Z_de = C_Z_de(alpha)
        C_m_0 = MAV.C_m_0
        C_m_a = MAV.C_m_alpha
        C_m_q = MAV.C_m_q
        C_m_de = MAV.C_m_delta_e

        # Trim Variables
        alpha_star = mav._alpha
        beta_star = mav._beta

        # Lateral State Space Model Coefficients (Table 5.1)
        Y_v = rho*S*b*v_star/(4.0*mass*Va_star)*(C_Y_p*p_star + C_Y_r*r_star) + rho*S*v_star/mass*(C_Y_0 + C_Y_b*beta_star + C_Y_da*delta_a_star + C_Y_dr*delta_r_star) + rho*S*C_Y_b/(2.0*mass)*np.sqrt(u_star**2 + w_star**2)
        Y_p = w_star + rho*Va_star*S*b/(4.0*mass)*C_Y_p
        Y_r = -u_star + rho*Va_star*S*b/(4.0*mass)*C_Y_r
        Y_da = rho*Va_star**2*S/(4.0*mass)*C_Y_da
        Y_dr = rho*Va_star**2*S/(4.0*mass)*C_Y_dr
        L_v = rho*S*b**2*v_star/(4.0*Va_star)*(C_p_p*p_star + C_p_r*r_star) + rho*S*b*v_star*(C_p_0 + C_p_b*beta_star + C_p_da*delta_a_star + C_p_dr*delta_r_star) + rho*S*b*C_p_b/2.0*np.sqrt(u_star**2 + w_star**2)
        L_p = MAV.gamma1*q_star + rho*Va_star*S*b**2/4.0*C_p_p
        L_r = -MAV.gamma2*q_star + rho*Va_star*S*b**2/4.0*C_p_r
        L_da = rho*Va_star**2*S*b/2.0*C_p_da
        L_dr = rho*Va_star**2*S*b/2.0*C_p_dr
        N_v = rho*S*b**2*v_star/(4.0*Va_star)*(C_r_p*p_star + C_r_r*r_star) + rho*S*b*v_star*(C_r_0 + C_r_b*beta_star + C_r_da*delta_a_star + C_r_dr*delta_r_star) + rho*S*b*C_r_b/2.0*np.sqrt(u_star**2 + w_star**2)
        N_p = MAV.gamma7*q_star + rho*Va_star*S*b**2/4.0*C_r_p
        N_r = -MAV.gamma1*q_star + rho*Va_star*S*b**2/4.0*C_r_r
        N_da = rho*Va_star**2*S*b/2.0*C_r_da
        N_dr = rho*Va_star**2*S*b/2.0*C_r_dr

        # Longitudinal State Space Model Coefficients (Table 5.2)
        X_u = u_star*rho*S/mass*(C_X_0 + C_X_a*alpha_star + C_X_de*delta_e_star) - rho*S*w_star*C_X_a/(2.0*mass) + rho*S*c*C_X_q*u_star*q_star/(4.0*mass*Va_star) - rho*S_prop*C_prop*u_star/mass
        X_w = -q_star + w_star*rho*S/mass*(C_X_0 + C_X_a*alpha_star + C_X_de*delta_e_star) + rho*S*c*C_X_q*w_star*q_star/(4.0*mass*Va_star) + rho*S*C_X_a*u_star/(2.0*mass) - rho*S_prop*C_prop*w_star/mass
        X_q = -w_star + rho*Va_star**2*S*C_X_q*c/(4.0*mass)
        X_de = rho*Va_star**2*S*C_X_de/(2.0*mass)
        X_dt = rho*S_prop*C_prop*k**2*delta_t_star/mass
        Z_u = q_star + u_star*rho*S/mass*(C_Z_0 + C_Z_a*alpha_star + C_Z_de*delta_e_star) - rho*S*C_Z_a*w_star/(2.0*mass) + u_star*rho*S*C_Z_q*c*q_star/(4.0*mass*Va_star)
        Z_w = w_star*rho*S/mass*(C_Z_0 + C_Z_a*alpha_star + C_Z_de*delta_e_star) + rho*S*C_Z_a*u_star/(2.0*mass) + rho*w_star*S*c*C_Z_q*q_star/(4.0*mass*Va_star)
        Z_q = u_star + rho*Va_star*S*C_Z_q*c/(4.0*mass)
        Z_de = rho*Va_star**2*S*C_Z_de/(2.0*mass)
        M_u = u_star*rho*S*c/Jy*(C_m_0 + C_m_a*alpha_star + C_m_de*delta_e_star) - rho*S*c*C_m_a*w_star/(2.0*Jy) + rho*S*c**2*C_m_q*q_star*u_star/(4.0*Jy*Va_star)
        M_w = w_star*rho*S*c/Jy*(C_m_0 + C_m_a*alpha_star + C_m_de*delta_e_star) + rho*S*c*C_m_a*u_star/(2.0*Jy) + rho*S*c**2*C_m_q*q_star*w_star/(4.0*Jy*Va_star)
        M_q = rho*Va_star*S*c**2*C_m_q/(4.0*Jy)
        M_de = rho*Va_star**2*S*c*C_m_de/(2.0*Jy)

        A_lat = np.array([[Y_v, Y_p, Y_r, g*np.cos(theta_star)*np.cos(phi_star), 0],
                          [L_v, L_p, L_r, 0, 0],
                          [N_v, N_p, N_r, 0, 0],
                          [0, 1, np.cos(phi_star)*np.tan(theta_star), q_star*np.cos(phi_star)*np.tan(theta_star)-r_star*np.sin(phi_star)*np.tan(theta_star), 0],
                          [0, 0, np.cos(phi_star)*1/np.cos(theta_star), q_star*np.cos(phi_star)*1/np.cos(theta_star)-r_star*np.sin(phi_star)*np.tan(theta_star), 0]])
        B_lat = np.array([[Y_da, Y_dr],
                          [L_da, L_dr],
                          [N_da, N_dr],
                          [0, 0],
                          [0, 0]])
        A_lon = np.array([[X_u, X_w, X_q, -g*np.cos(theta_star), 0],
                          [Z_u, Z_w, Z_q, -g*np.sin(theta_star), 0],
                          [M_u, M_w, M_q, 0, 0],
                          [0, 0, 1, 0, 0],
                          [np.sin(theta_star), -np.cos(theta_star), 0, u_star*np.cos(theta_star) + w_star*np.sin(theta_star), 0]])
        B_lon = np.array([[X_de, X_dt],
                          [Z_de, 0],
                          [M_de, 0],
                          [0, 0],
                          [0, 0]])
    else:
        A_maj = df_dx(mav, trim_state, trim_input)
        B_maj = df_du(mav, trim_state, trim_input)
        lat_row_ind = [[4], [10], [12], [6], [7], [9]]
        lat_col_ind = [4, 10, 12, 6, 7, 9]
        latB_col_ind = [0, 2]
        lon_row_ind = [[3], [5], [11], [8], [2]]
        lon_col_ind = [3, 5, 11, 8, 2]
        lonB_col_ind = [1, 3]
        A_lat = A_maj[lat_row_ind, lat_col_ind]
        A_lon = A_maj[lon_row_ind, lon_col_ind]
        B_lat = B_maj[lat_row_ind, latB_col_ind]
        B_lon = B_maj[lon_row_ind, lonB_col_ind]
        A_lon[4] = -A_lon[4]
        B_lon[4] = -B_lon[4]

    if display:
        print("Calculated State Space Forms")
        print("A_lat: \n", A_lat, "\n")
        print("B_lat: \n", B_lat, "\n")
        print("A_lon: \n", A_lon, "\n")
        print("B_lon: \n", B_lon, "\n")

    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    e = np.array([[x_quat[6], x_quat[7], x_quat[8], x_quat[9]]])
    [phi, theta, psi] = Quaternion2Euler(e)
    x_euler = np.array([[x_quat[0][0]],
                            [x_quat[1][0]],
                            [x_quat[2][0]],
                            [x_quat[3][0]],
                            [x_quat[4][0]],
                            [x_quat[5][0]],
                            [phi],
                            [theta],
                            [psi],
                            [x_quat[10][0]],
                            [x_quat[11][0]],
                            [x_quat[12][0]]])
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    e = Euler2Quaternion(x_euler[6], x_euler[7], x_euler[8])
    x_quat = np.array([[x_euler[0][0]],  # (0)
                            [x_euler[1][0]],  # (1)
                            [x_euler[2][0]],  # (2)
                            [x_euler[3][0]],  # (3)
                            [x_euler[4][0]],  # (4)
                            [x_euler[5][0]],  # (5)
                            [e[0][0]],  # (6)
                            [e[1][0]],  # (7)
                            [e[2][0]],  # (8)
                            [e[3][0]],  # (9)
                            [x_euler[9][0]],  # (10)
                            [x_euler[10][0]],  # (11)
                            [x_euler[11][0]]])  # (12)
    return x_quat


def finite_diff(f1, f2, dx):
    partial = (f2-f1)/(2.0*dx)
    return partial

def dxe_dxq(xq, dq=0.0001):
    e = xq[6:10]
    dxe_dxq_ = np.eye(12, 13)
    phi = lambda e: np.arctan2(2 * (e.item(0) * e.item(1) + e.item(2) * e.item(3)), (e.item(0) ** 2 + e.item(3) ** 2 - e.item(1) ** 2 - e.item(2) ** 2))
    theta = lambda e: np.arcsin(2 * (e.item(0) * e.item(2) - e.item(1) * e.item(3)))
    psi = lambda e: np.arctan2(2 * (e.item(0) * e.item(3) + e.item(1) * e.item(2)), (e.item(0) ** 2 + e.item(1) ** 2 - e.item(2) ** 2 - e.item(3) ** 2))
    E = np.array([phi, theta, psi])
    for i in range(6, 9):
        for j in range(6, 10):
            dxe_dxq_[i, j] = finite_diff(E[i-6](e-dq*np.eye(4)[j-6]), E[i-6](e+dq*np.eye(4)[j-6]), dq)
    return dxe_dxq_

def f_quat(mav, x_quat, input):
    # return 13x1 dynamics (as if state were Quaternion state)
    # compute f at quaternion_state
    old_state = mav._state
    mav._state = x_quat
    mav._update_velocity_data()
    forces_moments_ = mav._forces_moments(input)
    f_quat_ = mav._derivatives(x_quat, forces_moments_)

    # revert to original state
    mav._state = old_state
    mav._update_velocity_data()
    return f_quat_

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as f state were Quaternion state)
    # compute f at quaternion_state
    x_quat_ = quaternion_state(x_euler)
    f_quat_ = f_quat(mav, x_quat_, input)
    dxe_dxq_ = dxe_dxq(x_quat_)
    f_euler_ = dxe_dxq_ @ f_quat_
    return f_euler_

def df_dx(mav, x_quat_, input):
    # take partial of f_quat with respect to x_quat
    dx = 0.0001
    f = lambda x: f_quat(mav, x, input)
    A = np.zeros((13, 13))
    for j in range(13):
        A[:, j] = finite_diff(f(x_quat_-dx*np.eye(13)[j].reshape(-1, 1)), f(x_quat_+dx*np.eye(13)[j].reshape(-1, 1)), dx).flatten()
    return A

def df_du(mav, x_quat_, delta):
    # take partial of f_quat with respect to delta
    dx = 0.0001
    f = lambda d: f_quat(mav, x_quat_, d)
    B = np.zeros((13, 4))
    for j in range(4):
        B[:, j] = finite_diff(f(delta - dx * np.eye(4)[j].reshape(-1, 1)), f(delta + dx * np.eye(4)[j].reshape(-1, 1)), dx).flatten()
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    dx = 0.0001
    Va_old = mav._Va
    mav._Va = Va - dx
    mav._forces_moments(np.array([0.0, 0.0, 0.0, delta_t]))
    f_low = mav.thrust
    mav._Va = Va + dx
    mav._forces_moments(np.array([0.0, 0.0, 0.0, delta_t]))
    f_up = mav.thrust
    dThrust = finite_diff(f_low, f_up, dx)
    mav._Va = Va_old
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    dx = 0.0001
    Va_old = mav._Va
    mav._Va = Va
    mav._forces_moments(np.array([0.0, 0.0, 0.0, delta_t-dx]))
    f_low = mav.thrust
    mav._forces_moments(np.array([0.0, 0.0, 0.0, delta_t+dx]))
    f_up = mav.thrust
    dThrust = finite_diff(f_low, f_up, dx)
    mav._Va = Va_old
    return dThrust