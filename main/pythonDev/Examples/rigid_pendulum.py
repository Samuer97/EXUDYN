#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Example 2D rigid pendulum
#
# Author:   Johannes Gerstmayr
# Date:     2019-08-15
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.itemInterface import *

SC = exu.SystemContainer()
mbs = SC.AddSystem()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#rigid pendulum:
rect = [-2,-2,2,2] #xmin,ymin,xmax,ymax
background = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[rect[0],rect[1],0, rect[2],rect[1],0, rect[2],rect[3],0, rect[0],rect[3],0, rect[0],rect[1],0]} #background
oGround=mbs.AddObject(ObjectGround(referencePosition= [0,0,0], visualization=VObjectGround(graphicsData= [background])))
a = 0.5     #x-dim of pendulum
b = 0.05    #y-dim of pendulum
massRigid = 12
inertiaRigid = massRigid/12*(2*a)**2
g = 9.81    # gravity

graphics2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-a,-b,0, a,-b,0, a,b,0, -a,b,0, -a,-b,0]} #background
nRigid = mbs.AddNode(Rigid2D(referenceCoordinates=[-0.5,1,0], initialVelocities=[0,0,2]));
oRigid = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid,visualization=VObjectRigidBody2D(graphicsData= [graphics2])))

mR1 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid, localPosition=[-0.5,0.,0.])) #support point
mR2 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid, localPosition=[ 0.,0.,0.])) #end point

mG0 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oGround, localPosition=[-1,1.,0.]))
mbs.AddObject(RevoluteJoint2D(markerNumbers=[mG0,mR1]))

mbs.AddLoad(Force(markerNumber = mR2, loadVector = [0, -massRigid*g, 0]))

mbs.Assemble()
print(mbs)

simulationSettings = exu.SimulationSettings() #takes currently set values or default values

simulationSettings.timeIntegration.numberOfSteps = 1000000
simulationSettings.timeIntegration.endTime = 4000
simulationSettings.timeIntegration.startTime = 2000
simulationSettings.timeIntegration.newton.relativeTolerance = 1e-8*100 #10000
simulationSettings.timeIntegration.newton.absoluteTolerance = 1e-10
simulationSettings.timeIntegration.verboseMode = 1

simulationSettings.timeIntegration.newton.useModifiedNewton = False
simulationSettings.timeIntegration.newton.numericalDifferentiation.minimumCoordinateSize = 1
simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.5
simulationSettings.displayStatistics = True

#SC.visualizationSettings.nodes.defaultSize = 0.05

simulationSettings.solutionSettings.solutionInformation = "Rigid pendulum"

exu.StartRenderer()


exu.SolveDynamic(mbs, simulationSettings)

SC.WaitForRenderEngineStopFlag()
exu.StopRenderer() #safely close rendering window!

