import sys
sys.path.append('..')
import numpy as np
# import chap5.transfer_function_coef as TF
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity
# sigma = ??
Va0 = MAV.Va0
Vg = MAV.Va0

#----------roll loop-------------
wn_phi = 10.0      # tuning parameter
zeta_phi = 0.707   # tuning parameter

a_phi_1 = -0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2.0*Va0)
a_phi_2 = 0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a

e_phi_max = np.radians(45)
roll_kp = MAV.delta_a_max / e_phi_max
roll_kd = (2*zeta_phi*wn_phi - a_phi_1) / a_phi_2

#----------course loop-------------
W_chi = 10.0   # tuning parameter / bandwidth separation
wn_chi = wn_phi/W_chi
zeta_chi = 0.707  # tuning parameter

course_kp = 2*zeta_chi*wn_chi*Vg / gravity
course_ki = wn_chi**2 * Vg / gravity

#----------sideslip loop------------- (not used)
wn_beta = 0.5
zeta_beta = 0.707

a_b_1 = -MAV.rho*Va0*MAV.S_wing/(2*MAV.mass)*MAV.C_Y_beta
a_b_2 = MAV.rho*Va0*MAV.S_wing/(2*MAV.mass)*MAV.C_Y_delta_r

sideslip_ki = -wn_beta**2 / a_b_2
sideslip_kp = -(2*zeta_beta*wn_beta - a_b_1) / a_b_2

#----------yaw damper-------------
yaw_damper_tau_r = 0.01  # tune
yaw_damper_kp = 0.5

#----------pitch loop-------------
wn_theta = 20.0
zeta_theta = 0.707

a_t_1 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_q*MAV.c/(2.0*Va0)
a_t_2 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_alpha
a_t_3 = MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2*MAV.Jy)*MAV.C_m_delta_e

pitch_kp = (wn_theta**2 - a_t_2) / a_t_3
pitch_kd = (2*zeta_theta*wn_theta - a_t_1) / a_t_3
K_theta_DC = pitch_kp * a_t_3 / (a_t_2 + pitch_kp*a_t_3)

#----------altitude loop-------------
W_h = 15
wn_h = wn_theta/W_h
zeta_h = 1.2

altitude_kp = 2*zeta_h*wn_h / (K_theta_DC * Va0)
altitude_ki = wn_h**2 / (K_theta_DC*Va0)
altitude_zone = 500

#---------airspeed hold using throttle---------------
wn_V = 2.5
zeta_V = 0.9

a_v_1 = MAV.rho*MAV.Va_star*MAV.S_wing/MAV.mass*(MAV.C_D_0 + MAV.C_D_alpha*MAV.alpha_star + MAV.C_D_delta_e*MAV.delta_e_star) + MAV.rho*MAV.S_prop/MAV.mass*MAV.C_prop*MAV.Va_star
a_v_2 = MAV.rho*MAV.S_prop/MAV.mass*MAV.C_prop*MAV.k_motor**2*MAV.delta_t_star

airspeed_throttle_kp = (2*zeta_V*wn_V - a_v_1) / a_v_2
airspeed_throttle_ki = wn_V**2 / a_v_2
