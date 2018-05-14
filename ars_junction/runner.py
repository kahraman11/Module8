#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

from traci.constants import VAR_SPEED, VAR_POSITION, VAR_DISTANCE
from constants import *

import os
import sys
import optparse

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
import logging

logger = logging.getLogger(__name__)


# The 490 is a magic variable which should  be changed when the road get's longer.
def detectCollision(traci_data, veh_travelled_distance):
    return VEH_ID not in traci_data and veh_travelled_distance <= 490


def run():
    traci.vehicle.subscribe(VEH_ID, [VAR_SPEED, VAR_POSITION, VAR_DISTANCE])
    # traci.vehicle.subscribe('0', [VAR_SPEED, VAR_POSITION])

    traci_data = traci.vehicle.getSubscriptionResults()
    traci.simulationStep()

    while traci.simulation.getMinExpectedNumber() > 0:
        pos = traci_data[VEH_ID][VAR_DISTANCE]
        traci.simulationStep()

        # Collision check test
        collision = detectCollision(traci_data, pos)

        if VEH_ID in traci.vehicle.getIDList():
            # traci.vehicle.setSpeedMode('0', 0)
            # traci.vehicle.setSpeedMode(VEH_ID, 0)    # disable all safety checks
            print(traci.vehicle.getSpeed(VEH_ID))

        # to create a collision:
        # if traci.simulation.getCurrentTime() > 11000:
        #     traci.vehicle.setSpeed(VEH_ID, 0)
        #     car_to_colide = 'left_0'
        #     traci.vehicle.setSpeedMode(car_to_colide, 0)    # disable all safety checks
        #     traci.vehicle.setSpeed(car_to_colide, 10)

    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/{}.sumocfg".format(PREFIX)])
    run()
