from __future__ import absolute_import
from __future__ import print_function

import fileinput

import constants
import gym
import numpy as np
import time
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
counter = 0

collisions = []

redAccident = False
#CARS = [VEH_ID]


# The 90 is a magic variable which should  be changed when the road get's longer.
# Returns (Bool, Bool) first bool is accident? second is penalty?
def detectCollision(traci_data, veh_travelled_distance, car):
    # print("Car: ", car, " has travelled ", veh_travelled_distance, " meters")
    try:
        if veh_travelled_distance >= 0:
                # print("eerste: ", abs(traci_data[vehicle][VAR_POSITION][0] - traci_data[VEH_ID][VAR_POSITION][0]))
                # print("tweede: ", abs(traci_data[vehicle][VAR_POSITION][1] - traci_data[VEH_ID][VAR_POSITION][1]))

                if car == VEH_ID:
                    for vehicle in CARS:
                        if vehicle not in traci_data:
                            continue
                        if abs(traci_data[vehicle][VAR_POSITION][0] - traci_data[VEH_ID][VAR_POSITION][0]) <= 3.0 \
                         and abs(traci_data[vehicle][VAR_POSITION][1] - traci_data[VEH_ID][VAR_POSITION][1]) <= 3.0:
                            if traci_data[vehicle][VAR_POSITION][0] > traci_data[VEH_ID][VAR_POSITION][0]:  # Fout van rode auto
                                for col in collisions:
                                    if col == (vehicle, VEH_ID):
                                        return False, False
                                print("Collision between: ", VEH_ID, " and ", vehicle)
                                print("Fout van de rode auto@@@@@@@@@@@@@@")
                                collisions.append((vehicle, VEH_ID))
                                global redAccident
                                redAccident = True
                                return True, True

                if car != VEH_ID and car in traci_data and VEH_ID in traci_data and \
                        abs(traci_data[car][VAR_POSITION][0] - traci_data[VEH_ID][VAR_POSITION][0]) <= 3.0\
                        and abs(traci_data[car][VAR_POSITION][1] - traci_data[VEH_ID][VAR_POSITION][1]) <= 3.0:
                        # and (abs(traci_data[car][VAR_POSITION][0] - traci_data[VEH_ID][VAR_POSITION][0]) > 0 or abs(traci_data[vehicle][VAR_POSITION][1] - traci_data[VEH_ID][VAR_POSITION][1]) > 0): # haal dit weg waarschijnlijk
                    # print("eerste: ", abs(traci_data[vehicle][VAR_POSITION][0] - traci_data[VEH_ID][VAR_POSITION][0]))
                    # print("tweede: ", abs(traci_data[vehicle][VAR_POSITION][1] - traci_data[VEH_ID][VAR_POSITION][1]))
                    print("Collision between: ", car, " and ", VEH_ID)
                    alreadyRegistered = False
                    for col in collisions:
                        if col == (car, VEH_ID):
                            alreadyRegistered = True
                            # return (False, False)
                    if not alreadyRegistered:
                        collisions.append((car, VEH_ID))
                        if traci_data[car][VAR_POSITION][0] > traci_data[VEH_ID][VAR_POSITION][0]:  # Fout van rode auto
                            print("Fout van de rode auto")
                            redAccident = True
                            return True, False
                        return True, True

                for vehicle in CARS:
                    # if car == "AUTO4" and vehicle == "AUTO2":
                    #     print("FOCK DIE SHIT")
                    #     print(traci_data[car][VAR_POSITION][0] == traci_data[vehicle][VAR_POSITION][0])
                    #     print(traci_data[car][VAR_POSITION][1] - traci_data[vehicle][VAR_POSITION][1])
                    if vehicle not in traci_data:
                        # print(vehicle)
                        continue
                    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # print(car, traci_data[car][VAR_POSITION][1])
                    # print(vehicle, traci_data[vehicle][VAR_POSITION][1])
                    # print(car, traci_data[car][VAR_POSITION][0])
                    # print(vehicle, traci_data[vehicle][VAR_POSITION][0])
                    # print(car, vehicle, traci_data[vehicle][VAR_POSITION][1] - traci_data[car][VAR_POSITION][1])
                    # print(car, vehicle, traci_data[car][VAR_POSITION][1] - traci_data[vehicle][VAR_POSITION][1])
                    if car != VEH_ID and car != vehicle and vehicle != VEH_ID and \
                            traci_data[car][VAR_POSITION][0] == traci_data[vehicle][VAR_POSITION][0] >= 50:    # van onder naar boven
                        if 7.60 >= (traci_data[vehicle][VAR_POSITION][1] - traci_data[car][VAR_POSITION][1]) > 0:   # car / vehicle
                            acc = False
                            for col in collisions:
                                if col == (car, vehicle) or col == (vehicle, car):
                                    acc = True
                            if acc:
                                continue
                            print("Collision between: ", car, " and ", vehicle, " Is a crash on the car in front ONDER NAAR BOVEN")
                            print("@@@Y van vehicle: ", traci_data[vehicle][VAR_POSITION][1])
                            print("Y van car: ", traci_data[car][VAR_POSITION][1])
                            print("Verschil: ", traci_data[car][VAR_POSITION][1] - traci_data[vehicle][VAR_POSITION][1])
                            collisions.append((car, vehicle))
                            return True, True
                    elif car != VEH_ID and car != vehicle and vehicle != VEH_ID and \
                            traci_data[car][VAR_POSITION][0] == traci_data[vehicle][VAR_POSITION][0]:  # van boven naar beneden
                        if 7.60 >= (traci_data[car][VAR_POSITION][1] - traci_data[vehicle][VAR_POSITION][1]) > 0:   # vehicle / car
                            acc = False
                            for col in collisions:
                                if col == (car, vehicle) or col == (vehicle, car):
                                    acc = True
                            if acc:
                                continue
                            print("Collision between: ", car, " and ", vehicle, " Is a crash on the car in front BOVEN NAAR BENEDEN")
                            print("Y van vehicle: ", traci_data[car][VAR_POSITION][1])
                            print("Y van car: ", traci_data[vehicle][VAR_POSITION][1])
                            print("Verschil: ", traci_data[vehicle][VAR_POSITION][1] - traci_data[car][VAR_POSITION][1])
                            collisions.append((car, vehicle))
                            return True, True

                # for vehicle in CARS:
                #     if car != vehicle and vehicle != VEH_ID and car != VEH_ID and abs(traci_data[car][VAR_POSITION][1] - traci_data[vehicle][VAR_POSITION][1]) < 2.5:   # Kijk ook naar X as
                #         # print("Collisions: ", collisions)
                #         # check to make sure the accident has not already been reported
                #         for col in collisions:
                #             if col == (vehicle, car) or col == (car, vehicle):
                #                 return False
                #         print("Collision between: ", vehicle, " and ", car, " Is a crash on the car in front")
                #         collisions.append((vehicle, car))
                #         return True

        else:
            return False, False
        return False, False
    except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(car, exc_type, fname, exc_tb.tb_lineno)
                pass



def speedReward(speed):
    # print("speed: ", speed)
    # print("eerste gedeelte: ", -1 * (1 / (MAX_LANE_SPEED / (4 / 5))) * speed ** 5)
    # print("tweede gedeelte: ", speed ** 4)
    # print("totaal: ", -1 * (1 / (MAX_LANE_SPEED / (4 / 5))) * speed ** 5 + speed ** 4)
    if speed <= 0:
        return 0
    else:
        # return -1 * (1 / (MAX_LANE_SPEED / (4 / 5))) * speed ** 5 + speed ** 4
        if speed > 10:
            print("speed is te hoog", speed, (MAX_LANE_SPEED - (2 * abs(MAX_LANE_SPEED - speed))) * 100)
            return (MAX_LANE_SPEED - (2 * abs(MAX_LANE_SPEED - speed))) * 100
        if speed < 5:
            return (MAX_LANE_SPEED - (abs(MAX_LANE_SPEED - speed))) * -5
        return (MAX_LANE_SPEED - abs(MAX_LANE_SPEED - speed)) * 100

def getReward(traci_data, car):
    # print(CARS)
    # for vehicle in CARS:
        # print(car)
        speed = traci_data[car][VAR_SPEED]
        reward = speedReward(speed)
        reward *= 10/speedReward(MAX_LANE_SPEED)
        # print(car, ": ", reward, " en nu de speed: ", speed)
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

    # subcribe all possible vehicles, not the selfdriving             Kijken of mijn autos bestaan
    def subscribe_vehicles(self):
        for veh in traci.vehicle.getIDList():
            # print("veh: ", veh)
            if veh not in self.traci_data.keys():
                print("Deze print is er alleen wanneer er SUMO auto's gegenereerd zijn")
                traci.vehicle.subscribe(veh, [VAR_SPEED, VAR_DISTANCE, VAR_POSITION, VAR_ANGLE])
                if "up" in veh:
                    traci.vehicle.setSpeedMode(veh, 23)

    # Sets the state to the currently known values
    def set_state(self, car):
            try:
                if car in self.traci_data:
                    speed = self.traci_data[car][VAR_SPEED]

                    position_grid = np.zeros(shape=(13, 6))
                    car_position = self.traci_data[car][VAR_POSITION]

                    # filter out VEH_ID
                    data = [self.traci_data[a] for a in self.traci_data if a != car]
                    if car == VEH_ID:
                        for pos, angle in [(x[VAR_POSITION], x[VAR_ANGLE]) for x in data]:
                            relative_x = pos[0] - car_position[0]
                            relative_y = pos[1] - car_position[1]
                            x_index = int(relative_x/2.5)
                            y_index = 6 - int(relative_y/5)

                            # Make sure that the index doesn't go out of bounds
                            if 0 <= x_index <= 5 and 0 <= y_index <= 12:
                                position_grid[y_index][x_index] = 1

                    elif self.traci_data[car][VAR_ANGLE] == 0:      # Van beneden naar boven GEEL
                        for pos, angle in [(x[VAR_POSITION], x[VAR_ANGLE]) for x in data]:
                            relative_x = pos[0] - car_position[0]
                            relative_y = pos[1] - car_position[1]
                            x_index = int(relative_x/2.5) + 5
                            y_index = 12 - int(relative_y/5)

                            # Make sure that the index doesn't go out of bounds
                            if 0 <= x_index <= 5 and 0 <= y_index <= 12:
                                position_grid[y_index][x_index] = 1

                    elif self.traci_data[car][VAR_ANGLE] == 180:      # Van boven naar beneden PAARS
                        for pos, angle in [(x[VAR_POSITION], x[VAR_ANGLE]) for x in data]:
                            relative_x = abs(pos[0] - car_position[0])
                            relative_y = pos[1] - car_position[1]
                            x_index = 5 - int(relative_x/2.5)
                            y_index = int(relative_y/5) + 12       # start is onderaan

                            # Make sure that the index doesn't go out of bounds
                            if 0 <= x_index <= 5 and 0 <= y_index <= 12:
                                position_grid[y_index][x_index] = 1

                    # print(car, speed, position_grid)
                    self.state = np.reshape(np.append([speed], position_grid), (1, self.observation_space.shape[0]))
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(car, exc_type, fname, exc_tb.tb_lineno)
                print(e)
                pass

    def step(self, action, car):
        self.subscribe_vehicles()
        #for veh in traci.vehicle.getIDList():
            #print("jooh: ", veh)

        try:
            if car in self.traci_data:
                # apply the given action
                # print("Car: ", car, ", Action: ", action)
                if action == 0:
                    traci.vehicle.setSpeed(car, self.traci_data[car][VAR_SPEED] + 0.082)
                if action == 2:
                    # print("Normaal remmen: ", car)
                    traci.vehicle.setSpeed(car, self.traci_data[car][VAR_SPEED] - 0.372)
                if action == 3:
                    traci.vehicle.setSpeed(car, self.traci_data[car][VAR_SPEED] + 0.287)
                if action == 4:
                    # print("Hard remmen: ", car)
                    traci.vehicle.setSpeed(car, self.traci_data[car][VAR_SPEED] - 0.5)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(car, exc_type, fname, exc_tb.tb_lineno)
            pass

    def secondStep(self, car):
        try:
            if car in self.traci_data:
                pos = self.traci_data[car][VAR_DISTANCE]
                accident, penalty = detectCollision(self.traci_data, pos, car)
                global redAccident
                if car == VEH_ID and redAccident:
                    redAccident = False
                    return np.array(self.state), - 500, True, {}
                if accident:
                    print("############################### Collision detected ################################", accident, penalty)
                    if penalty:
                        return np.array(self.state), - 500, True, {}
                    else:
                        reward = getReward(self.traci_data, car)
                        self.set_state(car)
                        return np.array(self.state), reward, True, {}
            else:
                return np.array(self.state), 0, True, {}
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(car, exc_type, fname, exc_tb.tb_lineno)
            print(Exception, e)
            pass

        try:
            # Check the result of this step and assign a reward
            if car in self.traci_data:
                # if accident:  # check to make sure it gets - 1000 points
                #     return np.array(self.state), - 1000, True, {}

                reward = getReward(self.traci_data, car)
                self.set_state(car)

                # if 1 in self.state.position_grid:
                #     print("Test")
                if self.log:
                    print("{:.2f} {:d} {:.2f}".format(self.traci_data[car][VAR_SPEED], action, reward))
                if self.test:
                    self.run.append(self.traci_data[car][VAR_SPEED])
                    # print(car, reward)
                return np.array(self.state), reward, False, {}
        except KeyError as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(car, exc_type, fname, exc_tb.tb_lineno)
            # CARS.remove(car)
            pass
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(car, exc_type, fname, exc_tb.tb_lineno)
            pass
        # CARS.remove(car)
        return np.array(self.state), 0, True, {}

    def simulationStep(self):
        traci.simulationStep()

    def reset(self):
        del CARS[:]
        del collisions[:]
        if self.test and len(self.run) != 0:
            self.result.append(list(self.run))
            self.run.clear()

        global counter
        with fileinput.FileInput("data/junction.sumocfg", inplace=True) as file:
            for line in file:
                print(line.replace("junction" + str(counter) + ".rou.xml", "junction" + str(counter + 1) + ".rou.xml"), end='')

        counter += 1
        createRoute.generate_random_routefile(counter)
        traci.load(["-c", self.config_path])

        # Go to the simulation step where our autonomous car get's in action
        for _ in range(DEPART_TIME*10):
            traci.simulationStep()

        # Setup environment
        #print("CARS: ", CARS)
        for car in CARS:
            try:
                traci.vehicle.setSpeedMode(car, 0)
                traci.vehicle.setSpeed(car, rn.randint(1, 8))

                traci.vehicle.subscribe(car, [VAR_SPEED, VAR_DISTANCE, VAR_POSITION, VAR_ANGLE])
            except traci.exceptions.TraCIException as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(car, exc_type, fname, exc_tb.tb_lineno)
                pass
        traci.simulationStep()
        self.set_state(VEH_ID)
        return self.state

    def start(self, gui=False):
        for line in fileinput.input("data/junction.sumocfg", inplace=True):
            if line.strip().startswith('<route-files value='):
                line = '<route-files value="junction0.rou.xml"/>\n'
            sys.stdout.write(line)
        global counter
        counter = 0
        createRoute.generate_random_routefile(counter)
        #print("Cars: ", CARS)
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", self.config_path])
        self.traci_data = traci.vehicle.getSubscriptionResults()

    def close(self):
        traci.close()
