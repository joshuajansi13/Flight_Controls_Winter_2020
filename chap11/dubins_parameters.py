# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab
#     - Beard & McLain, PUP, 2012
#     - Update history:
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')


class dubins_parameters:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, ps, chis, pe, chie, R):
        ell = np.linalg.norm(ps - pe)
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            crs = ps + R * rotz(np.pi/2.0) @ np.array(
                [[np.cos(chis)], [np.sin(chis)], [0.0]])
            cls = ps + R * rotz(-np.pi/2.0) @ np.array(
                [[np.cos(chis)], [np.sin(chis)], [0.0]])
            cre = pe + R * rotz(np.pi / 2.0) @ np.array(
                [[np.cos(chie)], [np.sin(chie)], [0.0]])
            cle = pe + R * rotz(-np.pi / 2.0) @ np.array(
                [[np.cos(chie)], [np.sin(chie)], [0.0]])

            # compute L1, L2, L3, L4 using eqs. 11.9-11.12
            theta = np.arctan2((cre-crs).item(1), (cre-crs).item(0))
            L1 = np.linalg.norm(crs - cre) + R * mod(
                2.0 * np.pi + mod(theta - np.pi / 2.0) - mod(chis - np.pi / 2.0)) + \
                R * mod(2.0 * np.pi + mod(chie - np.pi / 2.0) - mod(theta - np.pi / 2.0))

            theta = np.arctan2((cle-crs).item(1), (cle-crs).item(0))
            ell = np.linalg.norm(cle-crs)
            theta2 = theta - np.pi/2.0 + np.arcsin(2.0*R/ell)
            L2 = np.sqrt(ell ** 2 - 4 * R ** 2) + R * mod(
                2.0 * np.pi + mod(theta2) - mod(chis - np.pi / 2.0)) + \
                R * mod(2.0 * np.pi + mod(theta2 + np.pi) - mod(chie + np.pi / 2.0))

            theta = np.arctan2((cre-cls).item(1), (cre-cls).item(0))
            ell = np.linalg.norm(cre-cls)
            theta2 = np.arccos(2.0*R/ell)
            L3 = np.sqrt(ell ** 2 - 4 * R ** 2) + R * mod(
                2.0 * np.pi + mod(chis + np.pi/2.0) - mod(theta + theta2)) + \
                R * mod(2.0 * np.pi + mod(chie - np.pi/2.0) - mod(theta + theta2 - np.pi))

            theta = np.arctan2((cle-cls).item(1), (cle-cls).item(0))
            L4 = np.linalg.norm(cls - cle) + R * mod(
                2.0 * np.pi + mod(chis + np.pi / 2.0) - mod(theta + np.pi / 2.0)) + \
                R * mod(2.0 * np.pi + mod(theta + np.pi / 2.0) - mod(chie + np.pi / 2.0))

            L = np.array([L1, L2, L3, L4])
            self.length = np.min(L)
            e1 = np.array([[1], [0], [0]])

            if np.argmin(L) == 0:
                self.center_s = crs
                self.dir_s = 1  # +1 is right -1 is left
                self.center_e = cre
                self.dir_e = 1
                self.n1 = (self.center_e-self.center_s) / np.linalg.norm(self.center_e-self.center_s)
                self.r1 = self.center_s + R * rotz(-np.pi/2.0) @ self.n1
                self.r2 = self.center_e + R * rotz(-np.pi/2.0) @ self.n1
            elif np.argmin(L) == 1:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                ell = np.linalg.norm(self.center_e - self.center_s)
                theta = np.arctan2((self.center_e - self.center_s).item(1), (self.center_e - self.center_s).item(0))
                theta2 = theta - np.pi/2.0 + np.arcsin(2.0*R/ell)
                self.n1 = rotz(theta2 + np.pi/2.0) @ e1
                self.r1 = self.center_s + R * rotz(theta2) @ e1
                self.r2 = self.center_e + R * rotz(theta2 + np.pi) @ e1
            elif np.argmin(L) == 2:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                ell = np.linalg.norm(self.center_e - self.center_s)
                theta = np.arctan2((self.center_e - self.center_s).item(1),
                                   (self.center_e - self.center_s).item(0))
                theta2 = np.arccos(2.0 * R / ell)
                self.n1 = rotz(theta + theta2 - np.pi / 2.0) @ e1
                self.r1 = self.center_s + R * rotz(theta + theta2) @ e1
                self.r2 = self.center_e + R * rotz(theta + theta2 - np.pi) @ e1
            else:
                self.center_s = cls
                self.dir_s = -1  # +1 is right -1 is left
                self.center_e = cle
                self.dir_e = -1
                self.n1 = (self.center_e - self.center_s) / np.linalg.norm(
                    self.center_e - self.center_s)
                self.r1 = self.center_s + R * rotz(np.pi / 2.0) @ self.n1
                self.r2 = self.center_e + R * rotz(np.pi / 2.0) @ self.n1

            self.r3 = pe
            self.n3 = rotz(chie) @ e1

            self.p_s = ps
            self.chi_s = chis
            self.p_e = pe
            self.chi_e = chie
            self.radius = R

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

def mod(x):
    # make x between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


