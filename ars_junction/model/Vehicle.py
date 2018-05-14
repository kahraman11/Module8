"""
A class which simulates a SUMO car
An object created contains most variables a SUMO car can have. For documentation:
http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes#Vehicles_and_Routes
"""
from model.VehicleType import VehicleType


class Vehicle:

    def __init__(self, id, route, type=VehicleType("left_car").id, depart=0.0, color="1,1,0"):
        self.id = id
        self.type = type
        self.route = route
        self.depart = depart
        self.color = color

    def printXML(self):
        xmlString = "<vehicle "
        for attr in dir(self):
            if hasattr(self, attr) and "_" not in attr and not callable(getattr(self, attr)):
                xmlString += (" {}=\"{}\"".format(attr, getattr(self, attr)))
        xmlString += "/>"
        return xmlString
