import numpy as np
# from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot
import parameters.aerosonde_parameters as MAV

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(60)  # approach angle for large distance from straight-line path (0 - 90) degrees
        self.k_path = 1.0 / 30.0  # proportional gain for straight-line path following
        self.k_orbit = 5  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        chi_q = np.arctan2(path.line_direction.item(1), path.line_direction.item(0))
        chi = state.chi
        chi_q = self._wrap(chi_q, chi)
        R_ip = np.array([[np.cos(chi_q), np.sin(chi_q), 0],
                         [-np.sin(chi_q), np.cos(chi_q), 0],
                         [0, 0, 1]])
        p_i = np.array([[state.pn], [state.pe], [-state.h]])
        r_i = path.line_origin
        e_p = R_ip @ (p_i - r_i)
        e_ip = p_i - r_i
        q = path.line_direction
        k_i = np.array([[0], [0], [1]])
        n = np.cross(q.T, k_i.T) / (np.linalg.norm(np.cross(q.T, k_i.T)))
        s_i = e_ip - (np.dot(e_ip, n)) * n.T

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_q - self.chi_inf * (2.0 / np.pi) * np.arctan(self.k_path * e_p.item(1))
        self.autopilot_commands.altitude_command = -r_i.item(2) + np.sqrt(s_i.item(0)**2 + s_i.item(1)**2) * (q.item(2) / np.sqrt(q.item(0)**2 + q.item(1)**2))
        self.autopilot_commands.phi_feedforward = 0.0

    def _follow_orbit(self, path, state):
        c = path.orbit_center
        rho = path.orbit_radius
        if path.orbit_direction == "CW":
            lbda = 1
        else:
            lbda = -1
        chi = state.chi

        d = np.sqrt((state.pn - c.item(0))**2 + (state.pe - c.item(1))**2)
        vpsi = np.arctan2((state.pe - c.item(1)), (state.pn - c.item(0)))
        vpsi = self._wrap(vpsi, chi)
        Va = state.Va
        g = MAV.gravity

        self.autopilot_commands.course_command = vpsi + lbda * ((np.pi/2.0) + np.arctan(self.k_orbit*((d-rho)/rho)))
        self.autopilot_commands.altitude_command = -path.orbit_center.item(2)
        self.autopilot_commands.phi_feedforward = lbda * np.arctan2(Va**2, (g*rho))
        self.autopilot_commands.airspeed_command = path.airspeed

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

