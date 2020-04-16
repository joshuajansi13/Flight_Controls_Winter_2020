"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap6.autopilot import autopilot
from tools.signals import signals

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)

# autopilot commands
from message_types.msg_autopilot import msg_autopilot
commands = msg_autopilot()
brakes_on = True
at_rest = True
# Va_command = signals(dc_offset=0.0, amplitude=1.0, start_time=10.0, frequency = 0.01)
# h_command = 0.0 #signals(dc_offset=100.0, amplitude=10.0, start_time=0.0, frequency=0.02)
chi_command = 0.0 #signals(dc_offset=np.radians(180), amplitude=np.radians(10), start_time=5.0, frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    if sim_time < 5:
        brakes_on = True
        at_rest = True
        commands.airspeed_command = 0
        commands.altitude_command = 0
    elif 5 < sim_time < 70:
        at_rest = False
        if commands.airspeed_command < 20:
            Va_command = signals(dc_offset=0.0, amplitude=1.0, start_time=5.0,
                                 frequency=0.01)
            commands.airspeed_command = Va_command.sawtooth(sim_time)
        else:
            commands.airspeed_command = 20
        commands.altitude_command = 100
    else:
        commands.airspeed_command = 0
        # commands.altitude_command = 0
        if commands.altitude_command > 0:
            h_command = signals(dc_offset=100.0, amplitude=-1.0, start_time=70.0,
                                 frequency=0.01)
            commands.altitude_command = h_command.sawtooth(sim_time)
        else:
            commands.altitude_command = 0

    #-------controller-------------
    estimated_state = mav.msg_true_state  # uses true states in the control
    # commands.airspeed_command = Va_command #.sawtooth(sim_time)
    commands.course_command = chi_command #.square(sim_time)
    # commands.altitude_command = h_command #.square(sim_time)
    delta, commanded_state = ctrl.update(commands, estimated_state, at_rest)

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation
