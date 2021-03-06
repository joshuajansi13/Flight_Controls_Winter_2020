"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.aerosonde_parameters as MAV
import parameters.control_parameters as AP
from tools.transfer_function import transfer_function
from tools.wrap import wrap
from chap6.pid_control import pid_control, pi_control, pd_control_with_rate
from message_types.msg_state import msg_state


class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pd_control_with_rate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = pi_control(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = transfer_function(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pd_control_with_rate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = pi_control(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(15))
        self.airspeed_from_throttle = pi_control(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = msg_state()
        self.state_regime = 1  # for state machine
        self.i = 0

    def update(self, cmd, state, at_rest):
        # state regime:
        # 1 = at rest
        # 2 = take off (full throttle; regulate pitch to a fixed theta_c
        # 3 = climb zone (full throttle; regulate airspeed by commanding pitch attitude)
        # 4 = altitude hold zone (regulate altitude by commanding pitch attitude; regulate airspeed by commanding throttle)
        # 5 = descend zone (zero throttle; regulate airspeed by commanding pitch attitude)

        # TODO implement state machine
        h_c = cmd.altitude_command
        if self.state_regime == 1:  # at rest
            theta_c = 0.
            if not at_rest:
                self.state_regime = 2
        elif self.state_regime == 2:  # going down the runway
            theta_c = 0.
            if state.h >= 1:
                self.state_regime = 3
        elif self.state_regime == 3:  # pitch up and take off
            theta_c = np.radians(10)
            if state.h > AP.h_takeoff:
                self.state_regime = 4
        elif self.state_regime == 4:  # climb zone up to commanded altitude and steady level flight and descent
            theta_c = self.altitude_from_pitch.update(h_c, state.h)
            if state.h <= 1:
                self.state_regime = 5
        elif self.state_regime == 5:  # flare
            theta_c = np.radians(5)
            if state.Va <= 15:
                self.state_regime = 6
        elif self.state_regime == 6:  # landed
            theta_c = self.altitude_from_pitch.update(h_c, state.h)

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        # h_c = cmd.altitude_command
        # theta_c = self.altitude_from_pitch.update(h_c, state.h)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        if at_rest:
            delta_t = 0.
        else:
            delta_t = MAV.delta_t_star + self.airspeed_from_throttle.update(
                cmd.airspeed_command, state.Va)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_a], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
