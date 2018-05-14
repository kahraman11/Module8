#!/usr/bin/env python
"""
Script to generate the rou.xml file in the data folder.
"""
from __future__ import absolute_import
from __future__ import print_function
import model.VehicleType as VehicleType
import model.Vehicle as Vehicle

from constants import *
import random as rn
rn.seed(12345)

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


# method to generate routes file.
def generate_routefile(file_location="data/junction.rou.xml"):
    created_cars = [VEH_ID]

    vtype_left = VehicleType.VehicleType("right_car")
    vtype_right = VehicleType.VehicleType("left_car")
    vtype_right.impatience = 1.0
    vtype_right.jmIgnoreFoeProb = 1.0
    route = rn.choice(['4to2', '4to3', '4to5'])
    vehicle = Vehicle.Vehicle(VEH_ID, depart=DEPART_TIME, route='4to2', color="1,0,0")

    with open(file_location, "w") as route_file:
        print("""<routes>
        """, vtype_left.printXML(), """"
        """, vtype_right.printXML(), """"
        
        <route id="4to2" edges="4to1 1to2" />
        <route id="4to3" edges="4to1 1to3" />
        <route id="4to5" edges="4to1 1to5" />
        
        <route id="5to3" edges="5to1 1to3" />
        <route id="3to5" edges="3to1 1to5" />
        
        <flow id="up" color="1,1,0"  begin="0" end= "200" probability="0.15" type="right_car">
            <route edges="5to1 1to3"/>
        </flow>
        <flow id="down" color="1,1,0"  begin="0" end= "200" probability="0.15" type="left_car">
            <route edges="3to1 1to5"/>
        </flow>
        """, vehicle.printXML(), """
    
        """, file=route_file)

        """"
        Route generation without flows
        """
        # route = rn.choice(['5to3', '3to5'])
        # depart_time = float(rn.randint(0, 15))/10
        # for i in range(1, rn.randint(2, 12)):
        #     vehicle = Vehicle.Vehicle(i, route=route, depart=depart_time)
        #     depart_time += 0.1
        #     print(vehicle.printXML(), file=route_file)
        #     created_cars.append(str(i))

        print("</routes>", file=route_file)
    return created_cars


# this is the main entry point of this script
if __name__ == "__main__":
    generate_routefile()
