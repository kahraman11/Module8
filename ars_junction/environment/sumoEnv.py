from __future__ import absolute_import
from __future__ import print_function

import constants
import gym
import numpy as np
from gym import spaces

import createRoute
from constants import *
import random as rn

import os
import sys

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
from traci.constants import *

# Setting the seeds to get reproducible results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


# The 90 is a magic variable which should  be changed when the road get's longer.
def detectCollision(traci_data, veh_travelled_distance):
    return VEH_ID not in traci_data and veh_travelled_distance <= 90


def speedReward(speed):
    return -1 * (1 / (MAX_LANE_SPEED / (4 / 5))) * speed ** 5 + speed ** 4


def getReward(traci_data):
    speed = traci_data[VEH_ID][VAR_SPEED]
    reward = speedReward(speed)
    reward *= 10/speedReward(MAX_LANE_SPEED)
    return reward


class SumoEnv(gym.Env):

    def __init__(self):
        # Speeds in meter per second
        self.maxSpeed = 20
        self.minSpeed = 0

        high = np.append([
                self.maxSpeed
            ],
            np.ones(shape=(13, 6))
        )
        low = np.append([
                self.minSpeed
            ],
            np.zeros(shape=(13, 6))
        )

        # Observation space of the environment
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(5)

        self.viewer = None
        self.state = None
        self.log = False
        self.result = []
        self.run = []
        self.test = False

        self._seed = 12345
        self.config_path = "data/{}.sumocfg".format(PREFIX)

        # This variable automatically get's updated after traci.simulationStep()
        # self.traci_data

    # subcribe all possible vehicles
    def subscribe_vehicles(self):
        for veh in traci.vehicle.getIDList():
            if veh not in self.traci_data.keys():
                traci.vehicle.subscribe(veh, [VAR_SPEED, VAR_DISTANCE, VAR_POSITION, VAR_ANGLE])
                if "up" in veh:
                    traci.vehicle.setSpeedMode(veh, 23)

    # Sets the state to the currently known values
    def set_state(self):
        speed = self.traci_data[VEH_ID][VAR_SPEED]

        position_grid = np.zeros(shape=(13, 6))
        car_position = self.traci_data[VEH_ID][VAR_POSITION]

        # filter out VEH_ID
        data = [self.traci_data[a] for a in self.traci_data if a != VEH_ID]
        for pos, angle in [(x[VAR_POSITION], x[VAR_ANGLE]) for x in data]:
            relative_x = pos[0] - car_position[0]
            relative_y = pos[1] - car_position[1]
            x_index = int(relative_x/5)
            y_index = 6 - int(relative_y/5)

            # Make sure that the index doesn't go out of bounds
            if 0 <= x_index <= 5 and 0 <= y_index <= 12:
                if (angle == 180 and y_index > 6) or (angle == 0 and y_index < 6):
                    # Filter out the cars that have passed the junction.
                    pass
                else:
                    position_grid[y_index][x_index] = 1

        self.state = np.reshape(np.append([speed], position_grid), (1, self.observation_space.shape[0]))

    def step(self, action):
        self.subscribe_vehicles()

        pos = self.traci_data[VEH_ID][VAR_DISTANCE]
        if VEH_ID in self.traci_data:
            # apply the given action
            if action == 0:
                traci.vehicle.setSpeed(VEH_ID, self.traci_data[VEH_ID][VAR_SPEED] + 0.082)
            if action == 2:
                traci.vehicle.setSpeed(VEH_ID, self.traci_data[VEH_ID][VAR_SPEED] - 0.372)
            if action == 3:
                traci.vehicle.setSpeed(VEH_ID, self.traci_data[VEH_ID][VAR_SPEED] + 0.287)
            if action == 4:
                traci.vehicle.setSpeed(VEH_ID, self.traci_data[VEH_ID][VAR_SPEED] - 0.5)

        # Run a step of the simulation
        traci.simulationStep()

        if detectCollision(self.traci_data, pos):
            print("collision detected")
            return np.array(self.state), - 1000, True, {}

        # Check the result of this step and assign a reward
        if VEH_ID in self.traci_data:
            reward = getReward(self.traci_data)
            self.set_state()

            # if 1 in self.state.position_grid:
            #     print("Test")

            if self.log:
                print("{:.2f} {:d} {:.2f}".format(self.traci_data[VEH_ID][VAR_SPEED], action, reward))
            if self.test:
                self.run.append(self.traci_data[VEH_ID][VAR_SPEED])
            return np.array(self.state), reward, False, {}
        return np.array(self.state), 0, True, {}

    def reset(self):
        if self.test and len(self.run) != 0:
            self.result.append(list(self.run))
            self.run.clear()

        traci.load(["-c", self.config_path])

        # Go to the simulation step where our autonomous car get's in action
        for _ in range(DEPART_TIME*10):
            traci.simulationStep()

        # Setup environment
        traci.vehicle.setSpeedMode(VEH_ID, 0)
        traci.vehicle.setSpeed(VEH_ID, rn.randint(1, 8))
        traci.simulationStep()

        traci.vehicle.subscribe(VEH_ID, [VAR_SPEED, VAR_DISTANCE, VAR_POSITION, VAR_ANGLE])

        self.set_state()
        return self.state

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", self.config_path])
        self.traci_data = traci.vehicle.getSubscriptionResults()

    def close(self):
        traci.close()
