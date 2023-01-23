#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Test with user-defined load function and user-defined spring-damper function (Duffing oscillator)
#
# Author:   Johannes Gerstmayr
# Date:     2019-11-15
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.itemInterface import *
import numpy as np

useGraphics = True #without test
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#you can erase the following lines and all exudynTestGlobals related operations if this is not intended to be used as TestModel:
try: #only if called from test suite
    from modelUnitTests import exudynTestGlobals #for globally storing test results
    useGraphics = exudynTestGlobals.useGraphics
except:
    class ExudynTestGlobals:
        pass
    exudynTestGlobals = ExudynTestGlobals()
    exudynTestGlobals.isPerformanceTest = False
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SC = exu.SystemContainer()
mbs = SC.AddSystem()

L=0.5
mass = 1.6          #mass in kg
spring = 4000       #stiffness of spring-damper in N/m
damper = 4          #damping constant in N/(m/s)
load0 = 80

omega0=np.sqrt(spring/mass)
f0 = 0.*omega0/(2*np.pi)
f1 = 1.*omega0/(2*np.pi)

tEnd = 50     #end time of simulation
h = 1e-4

#tEnd = 0.05     #end time of simulation
#steps = 5  #number of steps


#user function for spring force
def springForce(mbs, t, itemIndex, u, v, k, d, offset): #changed 2023-01-21:, mu, muPropZone):
    return 0.1*k*u+k*u**3+v*d

#linear frequency sweep in time interval [0, t1] and frequency interval [f0,f1];
def Sweep(t, t1, f0, f1):
    k = (f1-f0)/t1
    return np.sin(2*np.pi*(f0+k*0.5*t)*t) #take care of factor 0.5 in k*0.5*t, in order to obtain correct frequencies!!!

#user function for load
def userLoad(mbs, t, load):
    #return load*np.sin(0.5*omega0*t) #gives resonance
    #exu.Print(t)
    return load*Sweep(t, tEnd, f0, f1)
    #return load*Sweep(t, tEnd, f1, f0) #backward sweep

#node for 3D mass point:
n1=mbs.AddNode(Point(referenceCoordinates = [L,0,0]))

#ground node
nGround=mbs.AddNode(NodePointGround(referenceCoordinates = [0,0,0]))

#add mass point (this is a 3D object with 3 coordinates):
massPoint = mbs.AddObject(MassPoint(physicsMass = mass, nodeNumber = n1))

#marker for ground (=fixed):
groundMarker=mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= nGround, coordinate = 0))
#marker for springDamper for first (x-)coordinate:
nodeMarker  =mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= n1, coordinate = 0))

#Spring-Damper between two marker coordinates
mbs.AddObject(CoordinateSpringDamper(markerNumbers = [groundMarker, nodeMarker], 
                                     stiffness = spring, damping = damper, 
                                     springForceUserFunction = springForce)) 

loadC = mbs.AddLoad(LoadCoordinate(markerNumber = nodeMarker, 
                           load = load0, loadUserFunction=userLoad))

mbs.Assemble()

simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.numberOfSteps = int(tEnd/h)
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.displayStatistics = True
simulationSettings.displayComputationTime = True

simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 1

simulationSettings.displayStatistics = True
simulationSettings.timeIntegration.verboseMode = 1

#start solver:
exu.SolveDynamic(mbs, simulationSettings)

#evaluate final (=current) output values
u = mbs.GetNodeOutput(n1, exu.OutputVariableType.Position)
exu.Print('result perfSpringDamperUserFunction=',u[0])

exudynTestGlobals.testResult = u[0]

