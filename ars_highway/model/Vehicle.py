"""
A class which simulates a SUMO car
An object created contains most variables a SUMO car can have. For documentation:
http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes#Vehicles_and_Routes
"""
class Vehicle:

    def __init__(self, id, type, route):
        self.id = id
        self.type = type
        self.route = route
        self.depart = 0.0
        self.color = "1,0,0"


    def printXML(obj):
        xmlString = "<vehicle "
        for attr in dir(obj):
            if hasattr(obj, attr) and "_" not in attr and not callable(getattr(obj,attr)):
                xmlString += (" %s=\"%s\"" % (attr, getattr(obj, attr)))
        xmlString += "/>"
        return xmlString