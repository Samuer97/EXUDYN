#+++++++++++++++++++++++++++++++++++++++++++
# Mini example for class ObjectConnectorSpringDamper
#+++++++++++++++++++++++++++++++++++++++++++

import sys
sys.path.append('../TestModels')

import exudyn as exu
from exudyn.utilities import *

from modelUnitTests import ExudynTestStructure, exudynTestGlobals
import numpy as np

#create an environment for mini example
SC = exu.SystemContainer()
mbs = SC.AddSystem()

oGround=mbs.AddObject(ObjectGround(referencePosition= [0,0,0]))
nGround = mbs.AddNode(NodePointGround(referenceCoordinates=[0,0,0]))

exu.Print("start mini example for class ObjectConnectorSpringDamper")
try: #puts example in safe environment
    node = mbs.AddNode(NodePoint(referenceCoordinates = [1.05,0,0]))
    oMassPoint = mbs.AddObject(MassPoint(nodeNumber = node, physicsMass=1))
    
    m0 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oGround, localPosition=[0,0,0]))
    m1 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oMassPoint, localPosition=[0,0,0]))
    
    mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers=[m0,m1],
                                              referenceLength = 1, #shorter than initial distance
                                              stiffness = 100,
                                              damping = 1))

    #assemble and solve system for default parameters
    mbs.Assemble()
    exu.SolveDynamic(mbs)

    #check result at default integration time
    exudynTestGlobals.testResult = mbs.GetNodeOutput(node, exu.OutputVariableType.Position)[0]

except BaseException as e:
    exu.Print("An error occured in test example for ObjectConnectorSpringDamper:", e)
else:
    exu.Print("example for ObjectConnectorSpringDamper completed, test result =", exudynTestGlobals.testResult)

