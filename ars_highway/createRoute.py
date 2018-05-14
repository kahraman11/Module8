#!/usr/bin/env python
"""
Script to generate the rou.xml file in the data folder.
"""
from __future__ import absolute_import
from __future__ import print_function
import model.VehicleType as VehicleType
import model.Vehicle as Vehicle

import random
from constants import *

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
def generate_routefile():
    vtype = VehicleType.VehicleType("self_car")
    vehicle = Vehicle.Vehicle("AUTO", vtype.id, "1to3")
    with open("data/%s.rou.xml" % PREFIX, "w") as routes:
        print("""<routes>
        """,vtype.printXML(),""""
        

        <route id="1to3" edges="1to2 2to3" />
        """,vehicle.printXML(),"""
    
        """, file=routes)

        #vehNr = 0
        #for i in range(0, 3):
        #     # from 4 to 3 cars every step
        #     print('    <vehicle id="car_%i" type="car" route="1to3" depart="%i" />' % (
        #         vehNr, i), file=routes)
        #     vehNr += 1

        print("</routes>", file=routes)


# this is the main entry point of this script
if __name__ == "__main__":
    generate_routefile()
