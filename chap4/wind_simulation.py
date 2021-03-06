"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import transfer_function
import numpy as np

class wind_simulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([0, 0, 0])  # NED

        sig_u = 1.06
        sig_v = sig_u
        sig_w = 0.7
        Va = 25
        Lu = 200
        Lv = Lu
        Lw = 50
        a1 = sig_u*np.sqrt((2*Va)/Lu)
        a2 = sig_v*np.sqrt(3*Va/Lv)
        a3 = sig_v*np.sqrt(3*Va/Lv)*Va/(np.sqrt(3)*Lv)
        a4 = sig_w*np.sqrt(3*Va/Lw)
        a5 = sig_w*np.sqrt(3*Va/Lw)*Va/(np.sqrt(3)*Lw)
        b1 = Va/Lu
        b2 = Va/Lv
        b3 = Va/Lw

        self.u_w = transfer_function(num=np.array([[a1]]),
                                     den=np.array([[1, b1]]),
                                     Ts=Ts)
        self.v_w = transfer_function(num=np.array([[a2, a3]]),
                                     den=np.array([[1, 2*b2, b2**2.0]]),
                                     Ts=Ts)
        self.w_w = transfer_function(num=np.array([[a4, a5]]),
                                     den=np.array([[1, 2*b3, b3**2.0]]),
                                     Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        gust = np.array([[0.],[0.],[0.]]).flatten()
        return np.concatenate((self._steady_state, gust))

