# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:53:30 2018

@author: Johannes Gerstmayr

automatically generate pybindings for specific classes and functions AND latex documentation for these functions
"""

from autoGenerateHelper import DefPyFunctionAccess, DefPyStartClass, DefPyFinishClass, DefLatexStartClass, DefLatexFinishClass, GetDateStr, AddEnumValue


s = ''  #C++ pybind local includes
sL = '' #Latex documentation

#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#structures and enums:

sLenum = '\section{Type definitions}\nThis section defines a couple of structures, which are used to select, e.g., a configuration type or a variable type. In the background, these types are integer numbers, but for safety, the types should be used as type variables. \n\n'
sLenum+= 'Conversion to integer is possible: \n \\bi \n \\item[] \\texttt{x = int(exu.OutputVariableType.Displacement)} \n\\ei and also conversion from integer: \n \\bi \n \\item[] \\texttt{varType = exu.OutputVariableType(8)}\n \\ei\n'
s += '\n//        pybinding to enum classes:\n'

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'OutputVariableType'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for selecting output values, e.g. for GetObjectOutput(...) or for selecting variables for contour plot.\n\n'
descriptionStr += 'Available output variables and the interpreation of the output variable can be found at the object definitions. \n The OutputVariableType does not provide information about the size of the output variable, which can be either scalar or a list (vector). For vector output quantities, the contour plot option offers an additional parameter for selection of the component of the OutputVariableType. The components are usually out of \{0,1,2\}, representing \{x,y,z\} components (e.g., of displacements, velocities, ...), or \{0,1,2,3,4,5\} representing \{xx,yy,zz,yz,xz,xy\} components (e.g., of strain or stress). In order to compute a norm, chose component=-1, which will result in the quadratic norm for other vectors and to a norm specified for stresses (if no norm is defined for an outputVariable, it does not compute anything)\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, '_None', 'no value; used, e.g., to select no output variable in contour plot'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Distance', 'e.g., measure distance in spring damper connector'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Position', 'measure 3D position, e.g., of node or body'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Displacement', 'measure displacement; usually difference between current position and reference position'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'DisplacementLocal', 'measure local displacement, e.g. in local joint coordinates'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Velocity', 'measure (translational) velocity of node or object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'VelocityLocal', 'measure local (translational) velocity, e.g. in local body or joint coordinates'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Acceleration', 'measure (translational) acceleration of node or object'); s+=s1; sLenum+=sL1
#[s1,sL1] = AddEnumValue(pyClass, 'AccelerationLocal', 'measure (translational) acceleration of node or object in local coordinates'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'RotationMatrix', 'measure rotation matrix of rigid body node or object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Rotation', 'measure, e.g., scalar rotation of 2D body, Euler angles of a 3D object or rotation within a joint'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'AngularVelocity', 'measure angular velocity of node or object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'AngularVelocityLocal', 'measure local (body-fixed) angular velocity of node or object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'AngularAcceleration', 'measure angular acceleration of node or object'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'Coordinates', 'measure the coordinates of a node or object; coordinates usually just contain displacements, but not the position values'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Coordinates_t', 'measure the time derivative of coordinates (= velocity coordinates) of a node or object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Coordinates_tt', 'measure the second time derivative of coordinates (= acceleration coordinates) of a node or object'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'SlidingCoordinate', 'measure sliding coordinate in sliding joint'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Director1', 'measure a director (e.g. of a rigid body frame), or a slope vector in local 1 or x-direction'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Director2', 'measure a director (e.g. of a rigid body frame), or a slope vector in local 2 or y-direction'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Director3', 'measure a director (e.g. of a rigid body frame), or a slope vector in local 3 or z-direction'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'Force', 'measure global force, e.g., in joint or beam (resultant force), or generalized forces; see description of according object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'ForceLocal', 'measure local force, e.g., in joint or beam (resultant force)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Torque', 'measure torque, e.g., in joint or beam (resultant couple/moment)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'TorqueLocal', 'measure local torque, e.g., in joint or beam (resultant couple/moment)'); s+=s1; sLenum+=sL1
# unused for now, maybe later on in finite elements, fluid, etc.
# [s1,sL1] = AddEnumValue(pyClass, 'Strain', 'measure strain, e.g., axial strain in beam'); s+=s1; sLenum+=sL1
# [s1,sL1] = AddEnumValue(pyClass, 'Stress', 'measure stress, e.g., axial stress in beam'); s+=s1; sLenum+=sL1
# [s1,sL1] = AddEnumValue(pyClass, 'Curvature', 'measure curvature; may be scalar or vectorial: twist and curvature'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'StrainLocal', 'measure local strain, e.g., axial strain in cross section frame of beam or Green-Lagrange strain'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'StressLocal', 'measure local stress, e.g., axial stress in cross section frame of beam or Second Piola-Kirchoff stress; choosing component==-1 will result in the computation of the Mises stress'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'CurvatureLocal', 'measure local curvature; may be scalar or vectorial: twist and curvature of beam in cross section frame'); s+=s1; sLenum+=sL1

[s1,sL1] = AddEnumValue(pyClass, 'ConstraintEquation', 'evaluates constraint equation (=current deviation or drift of constraint equation)'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'ConfigurationType'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for selecting a configuration for reading or writing information to the module. Specifically, the ConfigurationType.Current configuration is usually used at the end of a solution process, to obtain result values, or the ConfigurationType.Initial is used to set initial values for a solution process.\n\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, '_None', 'no configuration; usually not valid, but may be used, e.g., if no configurationType is required'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Initial', 'initial configuration prior to static or dynamic solver; is computed during mbs.Assemble() or AssembleInitializeSystemCoordinates()'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Current', 'current configuration during and at the end of the computation of a step (static or dynamic)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Reference', 'configuration used to define deformable bodies (reference configuration for finite elements) or joints (configuration for which some joints are defined)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'StartOfStep', 'during computation, this refers to the solution at the start of the step = end of last step, to which the solver falls back if convergence fails'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Visualization', 'this is a state completely de-coupled from computation, used for visualization'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'EndOfEnumList', 'this marks the end of the list, usually not important to the user'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'ItemType'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for defining types of indices, e.g., in render window and will be also used in item dictionaries in future.\n\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, '_None', 'item has no type'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Node', 'item or index is of type Node'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Object', 'item or index is of type Object'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Marker', 'item or index is of type Marker'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Load', 'item or index is of type Load'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'Sensor', 'item or index is of type Sensor'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'NodeType'
cClass = 'Node'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for defining node types for 3D rigid bodies.\n\n'

s +=	'  py::enum_<' + cClass + '::Type' + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1, sL1] = AddEnumValue(cClass, '_None', 'node has no type'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Ground', 'ground node'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Position2D', '2D position node '); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Orientation2D', 'node with 2D rotation'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Point2DSlope1', '2D node with 1 slope vector'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Position', '3D position node'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Orientation', '3D orientation node'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RigidBody', 'node that can be used for rigid bodies'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RotationEulerParameters', 'node with 3D orientations that are modelled with Euler parameters (unit quaternions)'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RotationRxyz', 'node with 3D orientations that are modelled with Tait-Bryan angles'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RotationRotationVector', 'node with 3D orientations that are modelled with the rotation vector'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RotationLieGroup', 'node intended to be solved with Lie group methods'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'GenericODE2', 'node with general ODE2 variables'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'GenericODE1', 'node with general ODE1 variables'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'GenericAE', 'node with general algebraic variables'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'GenericData', 'node with general data variables'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Point3DSlope1', 'node with 1 slope vector'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'Point3DSlope23', 'node with 2 slope vectors in y and z direction'); s += s1; sLenum += sL1


s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'JointType'
cClass = 'Joint'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for defining joint types, used in KinematicTree.\n\n'

s +=	'  py::enum_<' + cClass + '::Type' + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1, sL1] = AddEnumValue(cClass, '_None', 'node has no type'); s += s1; sLenum += sL1

[s1, sL1] = AddEnumValue(cClass, 'RevoluteX', 'revolute joint type with rotation around local X axis'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RevoluteY', 'revolute joint type with rotation around local Y axis'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'RevoluteZ', 'revolute joint type with rotation around local Z axis'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'PrismaticX', 'prismatic joint type with translation along local X axis'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'PrismaticY', 'prismatic joint type with translation along local Y axis'); s += s1; sLenum += sL1
[s1, sL1] = AddEnumValue(cClass, 'PrismaticZ', 'prismatic joint type with translation along local Z axis'); s += s1; sLenum += sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'DynamicSolverType'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for selecting dynamic solvers for simulation.\n\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, 'GeneralizedAlpha', 'an implicit solver for index 3 problems; intended to be used for solving directly the index 3 constraints using the spectralRadius sufficiently small (usually 0.5 .. 1)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'TrapezoidalIndex2', 'an implicit solver for index 3 problems with index2 reduction; uses generalized alpha solver with settings for Newmark with index2 reduction'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'ExplicitEuler',    'an explicit 1st order solver (generally not compatible with constraints)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'ExplicitMidpoint', 'an explicit 2nd order solver (generally not compatible with constraints)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'RK33',     'an explicit 3 stage 3rd order Runge-Kutta method, aka "Heun"; (generally not compatible with constraints)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'RK44',     'an explicit 4 stage 4th order Runge-Kutta method, aka "classical Runge Kutta" (generally not compatible with constraints), compatible with Lie group integration and elimination of CoordinateConstraints'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'RK67',     "an explicit 7 stage 6th order Runge-Kutta method, see 'On Runge-Kutta Processes of High Order', J. C. Butcher, J. Austr Math Soc 4, (1964); can be used for very accurate (reference) solutions, but without step size control!"); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'ODE23',    'an explicit Runge Kutta method with automatic step size selection with 3rd order of accuracy and 2nd order error estimation, see Bogacki and Shampine, 1989; also known as ODE23 in MATLAB'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'DOPRI5',   "an explicit Runge Kutta method with automatic step size selection with 5th order of accuracy and 4th order error estimation, see  Dormand and Prince, 'A Family of Embedded Runge-Kutta Formulae.', J. Comp. Appl. Math. 6, 1980"); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'DVERK6', '[NOT IMPLEMENTED YET] an explicit Runge Kutta solver of 6th order with 5th order error estimation; includes adaptive step selection'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'KeyCode'

descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for special key codes in keyPressUserFunction.\n\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, 'SPACE', 'space key'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'ENTER', 'enter (return) key'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'TAB',   ''); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'BACKSPACE', ''); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'RIGHT', 'cursor right'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'LEFT', 'cursor left'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'DOWN', 'cursor down'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'UP', 'cursor up'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F1', 'function key F1'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F2', 'function key F2'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F3', 'function key F3'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F4', 'function key F4'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F5', 'function key F5'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F6', 'function key F6'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F7', 'function key F7'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F8', 'function key F8'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F9', 'function key F9'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'F10', 'function key F10'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++++++++++++++++++++
pyClass = 'LinearSolverType'


descriptionStr = 'This section shows the ' + pyClass + ' structure, which is used for selecting linear solver types, which are dense or sparse solvers.\n\n'

s +=	'  py::enum_<' + pyClass + '>(m, "' + pyClass + '")\n'
sLenum += DefLatexStartClass(sectionName = pyClass, 
                            description=descriptionStr, 
                            subSection=True, labelName=pyClass)
#keep this list synchronized with the accoring enum structure in C++!!!
[s1,sL1] = AddEnumValue(pyClass, '_None', 'no value; used, e.g., if no solver is selected'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'EXUdense', 'use dense matrices and according solvers for densly populated matrices (usually the CPU time grows cubically with the number of unknowns)'); s+=s1; sLenum+=sL1
[s1,sL1] = AddEnumValue(pyClass, 'EigenSparse', 'use sparse matrices and according solvers; additional overhead for very small systems; specifically, memory allocation is performed during a factorization process'); s+=s1; sLenum+=sL1

s +=	'		.export_values();\n\n'
sLenum += DefLatexFinishClass()

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Access functions to EXUDYN
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[s1,sL1] = DefPyStartClass('','', 'These are the access functions to the \\codeName\\ module.'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess('', 'GetVersionString', 'PyGetVersionString', 
                               argList=['addDetails'],
                               defaultArgs=['false'],
                               description='Get Exudyn built version as string (if addDetails=True, adds more information on Python version, platform, etc.)',
                               ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess('', 'RequireVersion', '', 
                               argList=['requiredVersionString'],
                               description = 'Checks if the installed version is according to the required version. Major, micro and minor version must agree the required level. Example: RequireVersion("1.0.31")')
sL+=sL1; #s+=s1;  #this function is defined in __init__.py ==> do not add to cpp bindings

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='StartRenderer', cName='PyStartOpenGLRenderer', 
                                defaultArgs=['false'],
                                argList=['verbose'],
                                description="Start OpenGL rendering engine (in separate thread); use verbose=True to output information during OpenGL window creation; some of the information will only be seen in windows command (powershell) windows or linux shell, but not inside iPython of Spyder"); s+=s1; sL+=sL1

#old, without [s1,sL1] = DefPyFunctionAccess('', 'StopRenderer', 'PyStopOpenGLRenderer', "Stop OpenGL rendering engine"); s+=s1; sL+=sL1

#new, defined in C++ as lambda function:
[s1,sL1] = DefPyFunctionAccess('', 'StopRenderer', 'no direct link to C++ here', "Stop OpenGL rendering engine"); sL+=sL1; #s+=s1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='DoRendererIdleTasks', cName='PyDoRendererIdleTasks', 
                                defaultArgs=['0'],
                                argList=['waitSeconds'],
                                description="Call this function in order to interact with Renderer window; use waitSeconds in order to run this idle tasks while animating a model (e.g. waitSeconds=0.04), use waitSeconds=0 without waiting, or use waitSeconds=-1 to wait until window is closed"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SolveStatic', cName='SolveDynamic', 
                               description='Static solver function, mapped from module \\texttt{solver}; for details on the Python interface see \\refSection{sec:solver:SolveStatic}; for background on solvers, see \\refSection{sec:solvers}',
                               argList=['mbs', 'simulationSettings', 'updateInitialValues', 'storeSolver'],
                               defaultArgs=['','exudyn.SimulationSettings()','False','True']
                               ); sL+=sL1
                
[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SolveDynamic', cName='SolveDynamic', 
                               description='Dynamic solver function, mapped from module \\texttt{solver}; for details on the Python interface see \\refSection{sec:solver:SolveDynamic}; for background on solvers, see \\refSection{sec:solvers}',
                               argList=['mbs', 'simulationSettings', 'solverType', 'updateInitialValues', 'storeSolver'],
                               defaultArgs=['','exudyn.SimulationSettings()','exudyn.DynamicSolverType.GeneralizedAlpha','False','True']
                               ); sL+=sL1
                
[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='ComputeODE2Eigenvalues', cName='ComputeODE2Eigenvalues', 
                               description='Simple interface to scipy eigenvalue solver for eigenvalue analysis of the second order differential equations part in mbs, mapped from module \\texttt{solver}; for details on the Python interface see \\refSection{sec:solver:ComputeODE2Eigenvalues}',
                               argList=['mbs', 'simulationSettings', 'useSparseSolver', 'numberOfEigenvalues', 'setInitialValues', 'convert2Frequencies'],
                               defaultArgs=['','exudyn.SimulationSettings()','False','-1','True','False']); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SetOutputPrecision', cName='PySetOutputPrecision', 
                                description="Set the precision (integer) for floating point numbers written to console (reset when simulation is started!); NOTE: this affects only floats converted to strings inside C++ exudyn; if you print a float from Python, it is usually printed with 16 digits; if printing numpy arrays, 8 digits are used as standard, to be changed with numpy.set_printoptions(precision=16); alternatively convert into a list",
                                argList=['numberOfDigits']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SetLinalgOutputFormatPython', cName='PySetLinalgOutputFormatPython', 
                                description="True: use Python format for output of vectors and matrices; False: use matlab format",
                                argList=['flagPythonFormat']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SetWriteToConsole', cName='PySetWriteToConsole', 
                            description="set flag to write (True) or not write to console; default = True",
                            argList=['flag']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SetWriteToFile', cName='PySetWriteToFile', 
                            description="set flag to write (True) or not write to console; default value of flagWriteToFile = False; flagAppend appends output to file, if set True; in order to finalize the file, write exu.SetWriteToFile('', False) to close the output file",
                            argList=['filename', 'flagWriteToFile', 'flagAppend'],
                            defaultArgs=['', 'true', 'false'],
                            example="exu.SetWriteToConsole(False) \\#no output to console\\\\exu.SetWriteToFile(filename='testOutput.log', flagWriteToFile=True, flagAppend=False)\\\\exu.Print('print this to file')\\\\exu.SetWriteToFile('', False) \\#terminate writing to file which closes the file"
                            ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SetPrintDelayMilliSeconds', cName='PySetPrintDelayMilliSeconds', 
                            description="add some delay (in milliSeconds) to printing to console, in order to let Spyder process the output; default = 0",
                            argList=['delayMilliSeconds']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='Print', cName='PyPrint', 
                            description="this allows printing via exudyn with similar syntax as in Python print(args) except for keyword arguments: print('test=',42); allows to redirect all output to file given by SetWriteToFile(...); does not output in case that SetWriteToConsole is set to False",
                            #argList=['pyObject'] #not compatible with py::args
                            ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='SuppressWarnings', cName='PySuppressWarnings', 
                            description="set flag to suppress (=True) or enable (=False) warnings",
                            argList=['flag']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass='', pyName='InfoStat', cName='PythonInfoStat', 
                               description='Retrieve list of global information on memory allocation and other counts as list:[array_new_counts, array_delete_counts, vector_new_counts, vector_delete_counts, matrix_new_counts, matrix_delete_counts, linkedDataVectorCast_counts]; May be extended in future; if writeOutput==True, it additionally prints the statistics; counts for new vectors and matrices should not depend on numberOfSteps, except for some objects such as ObjectGenericODE2 and for (sensor) output to files; Not available if code is compiled with __FAST_EXUDYN_LINALG flag',
                               argList=['writeOutput'],
                               defaultArgs=['true']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess('', 'Go', 'PythonGo', 'Creates a SystemContainer SC and a main system mbs'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess('', 'InvalidIndex', 'GetInvalidIndex', 
                            "This function provides the invalid index, which may depend on the kind of 32-bit, 64-bit signed or unsigned integer; e.g. node index or item index in list; currently, the InvalidIndex() gives -1, but it may be changed in future versions, therefore you should use this function"); s+=s1; sL+=sL1

#s += '        m.def_readwrite("variables", &exudynVariables, py::return_value_policy::reference)\n' 
#variables in the module itself are exported with "m.attr(...)" !
s += '        m.attr("variables") = exudynVariables;\n' 
sL += '  variables & this dictionary may be used by the user to store exudyn-wide data in order to avoid global Python variables; usage: exu.variables["myvar"] = 42 \\\\ \\hline  \n'

s += '        m.attr("sys") = exudynSystemVariables;\n' 
sL += "  sys & this dictionary is used and reserved by the system, e.g. for testsuite, graphics or system function to store module-wide data in order to avoid global Python variables; the variable exu.sys['renderState'] contains the last render state after exu.StopRenderer() and can be used for subsequent simulations \\\\ \\hline  \n"

[s1,sL1] = DefPyFinishClass('')
s+=s1; sL+=sL1





#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#currently, only latex binding:
pyClassStr = 'SystemContainer'
classStr = 'Main'+pyClassStr
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, 'The SystemContainer is the top level of structures in \\codeName. The container holds all systems, solvers and all other data structures for computation. Currently, only one container shall be used. In future, multiple containers might be usable at the same time.' +
        ' \\\\ Example: \\\\ \\texttt{import exudyn as exu \\\\ SC = exu.SystemContainer() \\\\ mbs = SC.AddSystem()}')
sL+=sL1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#GENERAL FUNCTIONS

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Reset', cName='Reset', 
                                description="delete all systems and reset SystemContainer (including graphics); this also releases SystemContainer from the renderer, which requires SC.AttachToRenderEngine() to be called in order to reconnect to rendering; a safer way is to delete the current SystemContainer and create a new one (SC=SystemContainer() )"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddSystem', cName='AddMainSystem', 
                                description="add a new computational system", options='py::return_value_policy::reference'); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfSystems', cName='NumberOfSystems', 
                                description="obtain number of systems available in system container"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSystem', cName='GetMainSystem', 
                                description="obtain systems with index from system container",
                                argList=['systemNumber']); sL+=sL1


#s += '        .def_property("visualizationSettings", &MainSystemContainer::PyGetVisualizationSettings, &MainSystemContainer::PySetVisualizationSettings)\n' 
sL += '  visualizationSettings & this structure is read/writeable and contains visualization settings, which are immediately applied to the rendering window. \\tabnewline\n    EXAMPLE:\\tabnewline\n    SC = exu.SystemContainer()\\tabnewline\n    SC.visualizationSettings.autoFitScene=False  \\\\ \\hline  \n'

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetRenderState', cName='PyGetRenderState', 
                                description="Get dictionary with current render state (openGL zoom, modelview, etc.); will have no effect if GLFW_GRAPHICS is deactivated",
                                example = "SC = exu.SystemContainer()\\\\renderState = SC.GetRenderState() \\\\print(renderState['zoom'])"
                                ); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetRenderState', cName='PySetRenderState', 
                                description="Set current render state (openGL zoom, modelview, etc.) with given dictionary; usually, this dictionary has been obtained with GetRenderState; will have no effect if GLFW_GRAPHICS is deactivated",
                                example = "SC = exu.SystemContainer()\\\\SC.SetRenderState(renderState)",
                                argList=['renderState'],
                                ); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='RedrawAndSaveImage', cName='RedrawAndSaveImage', 
                                description="Redraw openGL scene and save image (command waits until process is finished)"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='WaitForRenderEngineStopFlag', cName='WaitForRenderEngineStopFlag', 
                                description="Wait for user to stop render engine (Press 'Q' or Escape-key)"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='RenderEngineZoomAll', cName='PyZoomAll', 
                                description="Send zoom all signal, which will perform zoom all at next redraw request"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='RedrawAndSaveImage', cName='RedrawAndSaveImage', 
                                description="Redraw openGL scene and save image (command waits until process is finished)"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AttachToRenderEngine', cName='AttachToRenderEngine', 
                                description="Links the SystemContainer to the render engine, such that the changes in the graphics structure drawn upon updates, etc.; done automatically on creation of SystemContainer; return False, if no renderer exists (e.g., compiled without GLFW) or cannot be linked (if other SystemContainer already linked)"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='DetachFromRenderEngine', cName='DetachFromRenderEngine', 
                                description="Releases the SystemContainer from the render engine; return True if successfully released, False if no GLFW available or detaching failed"); sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetCurrentMouseCoordinates', cName='PyGetCurrentMouseCoordinates', 
                                description="Get current mouse coordinates as list [x, y]; x and y being floats, as returned by GLFW, measured from top left corner of window; use GetCurrentMouseCoordinates(useOpenGLcoordinates=True) to obtain OpenGLcoordinates of projected plane",
                                argList=['useOpenGLcoordinates'],
                                defaultArgs=['False'],
                                ); sL+=sL1

#removed 2021-07-12 as deprecated:
# [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='TimeIntegrationSolve', cName="""[](MainSystemContainer& msc, MainSystem& ms, HString solverName, const SimulationSettings& simulationSettings) {
#                             		pout.precision(simulationSettings.outputPrecision);
#                             		if (solverName == "RungeKutta1")
#                             			msc.GetSolvers().GetSolverRK1().SolveSystem(simulationSettings, *(ms.GetCSystem()));
#                             		else if (solverName == "GeneralizedAlpha")
#                             			msc.GetSolvers().GetSolverGeneralizedAlpha().SolveSystem(simulationSettings, *(ms.GetCSystem()));
#                             		else
#                             			PyError(HString("SystemContainer::TimeIntegrationSolve: invalid solverName '")+solverName+"'; options are: RungeKutta1 or GeneralizedAlpha");
#                             		}""", 
#                                 argList=['mainSystem','solverName','simulationSettings'],
#                                 description="DEPRECATED, use exu.SolveDynamic(...) instead, see \\refSection{sec:solver:SolveDynamic}! Call time integration solver for given system with solverName ('RungeKutta1'...explicit solver, 'GeneralizedAlpha'...implicit solver); use simulationSettings to individually configure the solver",
#                                 example = "simSettings = exu.SimulationSettings()\\\\simSettings.timeIntegration.numberOfSteps = 1000\\\\simSettings.timeIntegration.endTime = 2\\\\simSettings.timeIntegration.verboseMode = 1\\\\SC.TimeIntegrationSolve(mbs, 'GeneralizedAlpha', simSettings)",
#                                 isLambdaFunction = True
#                                 ); sL+=sL1

# [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='StaticSolve', cName="""[](MainSystemContainer& msc, MainSystem& ms, const SimulationSettings& simulationSettings) {
#                                 pout.precision(simulationSettings.outputPrecision);
#                                 msc.GetSolvers().GetSolverStatic().SolveSystem(simulationSettings, *(ms.GetCSystem()));
#                                 }""", 
#                                 argList=['mainSystem','simulationSettings'],
#                                 description="DEPRECATED, use exu.SolveStatic(...) instead, see \\refSection{sec:solver:SolveStatic}! Call solver to compute a static solution of the system, considering acceleration and velocity coordinates to be zero (initial velocities may be considered by certain objects)",
#                                 example = "simSettings = exu.SimulationSettings()\\\\simSettings.staticSolver.newton.relativeTolerance = 1e-6\\\\SC.StaticSolve(mbs, simSettings)",
#                                 isLambdaFunction = True
#                                 ); sL+=sL1


sL += DefLatexFinishClass()#only finalize latex table



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
classStr = 'MainSystem'
[s1,sL1] = DefPyStartClass(classStr, classStr, "This is the structure which defines a (multibody) system. In C++, there is a MainSystem (links to Python) and a System (computational part). For that reason, the name is MainSystem on the Python side, but it is often just called 'system'. It can be created, visualized and computed. " + "Use the following functions for system manipulation." +
        ' \\\\ \\\\ Usage: \\\\ \\\\ \\texttt{import exudyn as exu \\\\ SC = exu.SystemContainer() \\\\ mbs = SC.AddSystem()}')
s+=s1; sL+=sL1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#GENERAL FUNCTIONS

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Assemble', cName='Assemble', 
                                description="assemble items (nodes, bodies, markers, loads, ...); Calls CheckSystemIntegrity(...), AssembleCoordinates(), AssembleLTGLists(), AssembleInitializeSystemCoordinates(), and AssembleSystemInitialize()"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AssembleCoordinates', cName='AssembleCoordinates', 
                                description="assemble coordinates: assign computational coordinates to nodes and constraints (algebraic variables)"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AssembleLTGLists', cName='AssembleLTGLists', 
                                description="build \\ac{LTG} coordinate lists for objects (used to build global ODE2RHS, MassMatrix, etc. vectors and matrices) and store special object lists (body, connector, constraint, ...)"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AssembleInitializeSystemCoordinates', cName='AssembleInitializeSystemCoordinates', 
                                description="initialize all system-wide coordinates based on initial values given in nodes"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AssembleSystemInitialize', cName='AssembleSystemInitialize', 
                                description="initialize some system data, e.g., generalContact objects (searchTree, etc.)"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Reset', cName='Reset', 
                                description="reset all lists of items (nodes, bodies, markers, loads, ...) and temporary vectors; deallocate memory"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSystemContainer', cName='GetMainSystemContainer', 
                                description="return the systemContainer where the mainSystem (mbs) was created"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='WaitForUserToContinue', cName='WaitForUserToContinue', 
                                description="interrupt further computation until user input --> 'pause' function"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SendRedrawSignal', cName='SendRedrawSignal', 
                                description="this function is used to send a signal to the renderer that the scene shall be redrawn because the visualization state has been updated"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetRenderEngineStopFlag', cName='GetRenderEngineStopFlag', 
                                description="get the current stop simulation flag; True=user wants to stop simulation"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetRenderEngineStopFlag', cName='SetRenderEngineStopFlag', 
                                description="set the current stop simulation flag; set to False, in order to continue a previously user-interrupted simulation"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ActivateRendering', cName='ActivateRendering', 
                                argList=['flag'],
                                defaultArgs=['true'],
                                description="activate (flag=True) or deactivate (flag=False) rendering for this system"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetPreStepUserFunction', cName='PySetPreStepUserFunction', 
                                description="Sets a user function PreStepUserFunction(mbs, t) executed at beginning of every computation step; in normal case return True; return False to stop simulation after current step",
                                example = 'def PreStepUserFunction(mbs, t):\\\\ \\TAB print(mbs.systemData.NumberOfNodes())\\\\ \\TAB if(t>1): \\\\ \\TAB \\TAB return False \\\\ \\TAB return True \\\\ mbs.SetPreStepUserFunction(PreStepUserFunction)'); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetPostNewtonUserFunction', cName='PySetPostNewtonUserFunction', 
                                description="Sets a user function PostNewtonUserFunction(mbs, t) executed after successful Newton iteration in implicit or static solvers and after step update of explicit solvers, but BEFORE PostNewton functions are called by the solver; function returns list [discontinuousError, recommendedStepSize], containing a error of the PostNewtonStep, which is compared to [solver].discontinuous.iterationTolerance. The recommendedStepSize shall be negative, if no recommendation is given, 0 in order to enforce minimum step size or a specific value to which the current step size will be reduced and the step will be repeated; use this function, e.g., to reduce step size after impact or change of data variables",
                                example = 'def PostNewtonUserFunction(mbs, t):\\\\ \\TAB if(t>1): \\\\ \\TAB \\TAB return [0, 1e-6] \\\\ \\TAB return [0,0] \\\\ mbs.SetPostNewtonUserFunction(PostNewtonUserFunction)'); s+=s1; sL+=sL1

#contact:                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddGeneralContact', cName='AddGeneralContact', 
                                description="add a new general contact, used to enable efficient contact computation between objects (nodes or markers)", options='py::return_value_policy::reference'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetGeneralContact', cName='GetGeneralContact', 
                                description="get read/write access to GeneralContact with index 'generalContactNumber' stored in mbs",
                                argList=['generalContactNumber']); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='DeleteGeneralContact', cName='DeleteGeneralContact', 
                                description="delete GeneralContact with index 'generalContactNumber' in mbs; other general contacts are resorted (index changes!)",
                                argList=['generalContactNumber']); s+=s1; sL+=sL1

#++++++++++++++++

#old version, with variables: [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', cName='[](const MainSystem &ms) {\n            return "<systemData: \\n" + ms.GetMainSystemData().PyInfoSummary() + "\\nmainSystem:\\n  variables = " + EXUstd::ToString(ms.variables) + "\\n  sys = " + EXUstd::ToString(ms.systemVariables) + "\\n>\\n"; }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', cName='[](const MainSystem &ms) {\n            return "<systemData: \\n" + ms.GetMainSystemData().PyInfoSummary() + "\\nFor details see mbs.systemData, mbs.sys and mbs.variables\\n>\\n"; }', 
                                description="return the representation of the system, which can be, e.g., printed",
                                isLambdaFunction = True,
                                example = 'print(mbs)'); s+=s1; sL+=sL1

s += '        .def_property("systemIsConsistent", &MainSystem::GetFlagSystemIsConsistent, &MainSystem::SetFlagSystemIsConsistent)\n' 
sL += '  systemIsConsistent & this flag is used by solvers to decide, whether the system is in a solvable state; this flag is set to False as long as Assemble() has not been called; any modification to the system, such as Add...(), Modify...(), etc. will set the flag to False again; this flag can be modified (set to True), if a change of e.g.~an object (change of stiffness) or load (change of force) keeps the system consistent, but would normally lead to systemIsConsistent=False  \\\\ \\hline  \n'

s += '        .def_property("interactiveMode", &MainSystem::GetInteractiveMode, &MainSystem::SetInteractiveMode)\n' 
sL += '  interactiveMode & set this flag to True in order to invoke a Assemble() command in every system modification, e.g. AddNode, AddObject, ModifyNode, ...; this helps that the system can be visualized in interactive mode. \\\\ \\hline  \n'

s += '        .def_readwrite("variables", &MainSystem::variables, py::return_value_policy::reference)\n' 
sL += '  variables & this dictionary may be used by the user to store model-specific data, in order to avoid global Python variables in complex models; mbs.variables["myvar"] = 42 \\\\ \\hline  \n'

s += '        .def_readwrite("sys", &MainSystem::systemVariables, py::return_value_policy::reference)\n' 
sL += '  sys & this dictionary is used by exudyn Python libraries, e.g., solvers, to avoid global Python variables \\\\ \\hline \n'

s += '        .def_property("solverSignalJacobianUpdate", &MainSystem::GetFlagSolverSignalJacobianUpdate, &MainSystem::SetFlagSolverSignalJacobianUpdate)\n' 
sL += '  solverSignalJacobianUpdate & this flag is used by solvers to decide, whether the jacobian should be updated; at beginning of simulation and after jacobian computation, this flag is set automatically to False; use this flag to indicate system changes, e.g. during time integration  \\\\ \\hline  \n'

s += '        .def_readwrite("systemData", &MainSystem::mainSystemData, py::return_value_policy::reference)\n' 
sL += '  systemData & Access to SystemData structure; enables access to number of nodes, objects, ... and to (current, initial, reference, ...) state variables (ODE2, AE, Data,...)\\\\ \\hline  \n'


sL += DefLatexFinishClass()#only finalize latex table

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#NODE
s += "\n//        NODES:\n"
sL+=DefLatexStartClass(classStr+': Node', '\label{sec:mainsystem:node}\n This section provides functions for adding, reading and modifying nodes. Nodes are used to define coordinates (unknowns to the static system and degrees of freedom if constraints are not present). Nodes can provide various types of coordinates for second/first order differential equations (ODE2/ODE1), algebraic equations (AE) and for data (history) variables -- which are not providing unknowns in the nonlinear solver but will be solved in an additional nonlinear iteration for e.g. contact, friction or plasticity.', subSection=True)
#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddNode', cName='[](MainSystem& mainSystem, py::dict itemDict) {return mainSystem.AddMainNode(itemDict); }', 
#                                description="add a node with nodeDefinition in dictionary format; returns (global) node number of newly added node",
#                                argList=['itemDict'],
#                                example="nodeDict = {'nodeType': 'Point', \\\\'referenceCoordinates': [1.0, 0.0, 0.0], \\\\'initialCoordinates': [0.0, 2.0, 0.0], \\\\'name': 'example node'} \\\\ mbs.AddNode(nodeDict)",
#                                isLambdaFunction = True
#                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddNode', cName='[](MainSystem& mainSystem, py::object pyObject) {return mainSystem.AddMainNodePyClass(pyObject); }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddNode', cName='AddMainNodePyClass', 
                                description="add a node with nodeDefinition from Python node class; returns (global) node index (type NodeIndex) of newly added node; use int(nodeIndex) to convert to int, if needed (but not recommended in order not to mix up index types of nodes, objects, markers, ...)",
                                argList=['pyObject'],
                                example = "item = Rigid2D( referenceCoordinates= [1,0.5,0], initialVelocities= [10,0,0]) \\\\mbs.AddNode(item) \\\\" + "nodeDict = {'nodeType': 'Point', \\\\'referenceCoordinates': [1.0, 0.0, 0.0], \\\\'initialCoordinates': [0.0, 2.0, 0.0], \\\\'name': 'example node'} \\\\ mbs.AddNode(nodeDict)"
#                                isLambdaFunction = True
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeNumber', cName='PyGetNodeNumber', 
                                description="get node's number by name (string)",
                                argList=['nodeName'],
                                example = "n = mbs.GetNodeNumber('example node')"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNode', cName='PyGetNode', 
                                description="get node's dictionary by node number (type NodeIndex)",
                                argList=['nodeNumber'],
                                example = "nodeDict = mbs.GetNode(0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ModifyNode', cName='PyModifyNode', 
                                description="modify node's dictionary by node number (type NodeIndex)",
                                argList=['nodeNumber','nodeDict'],
                                example = "mbs.ModifyNode(nodeNumber, nodeDict)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeDefaults', cName='PyGetNodeDefaults', 
                                description="get node's default values for a certain nodeType as (dictionary)",
                                argList=['typeName'],
                                example = "nodeType = 'Point'\\\\nodeDict = mbs.GetNodeDefaults(nodeType)"
                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='CallNodeFunction', cName='PyCallNodeFunction', 
#                                description="call specific node function",
#                                argList=['nodeNumber', 'functionName', 'args'],
#                                defaultArgs=['', '', 'py::dict()']
#                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeOutput', cName='PyGetNodeOutputVariable', 
                                description="get the ouput of the node specified with the OutputVariableType; default configuration = 'current'; output may be scalar or array (e.g. displacement vector)",
                                argList=['nodeNumber','variableType','configuration'],
                                defaultArgs=['','','ConfigurationType::Current'],
                                example = "mbs.GetNodeOutput(nodeNumber=0, variableType=exu.OutputVariableType.Displacement)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeODE2Index', cName='PyGetNodeODE2Index', 
                                description="get index in the global ODE2 coordinate vector for the first node coordinate of the specified node",
                                argList=['nodeNumber'],
                                example = "mbs.GetNodeODE2Index(nodeNumber=0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeODE1Index', cName='PyGetNodeODE1Index', 
                                description="get index in the global ODE1 coordinate vector for the first node coordinate of the specified node",
                                argList=['nodeNumber'],
                                example = "mbs.GetNodeODE1Index(nodeNumber=0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeAEIndex', cName='PyGetNodeAEIndex', 
                                description="get index in the global AE coordinate vector for the first node coordinate of the specified node",
                                argList=['nodeNumber'],
                                example = "mbs.GetNodeAEIndex(nodeNumber=0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetNodeParameter', cName='PyGetNodeParameter', 
                                description="get nodes's parameter from node number (type NodeIndex) and parameterName; parameter names can be found for the specific items in the reference manual; for visualization parameters, use a 'V' as a prefix",
                                argList=['nodeNumber', 'parameterName'],
                                example = "mbs.GetNodeParameter(0, 'referenceCoordinates')",
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetNodeParameter', cName='PySetNodeParameter', 
                                description="set parameter 'parameterName' of node with node number (type NodeIndex) to value; parameter names can be found for the specific items in the reference manual; for visualization parameters, use a 'V' as a prefix",
                                argList=['nodeNumber', 'parameterName', 'value'],
                                example = "mbs.SetNodeParameter(0, 'Vshow', True)",
                                ); s+=s1; sL+=sL1

sL += DefLatexFinishClass()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#OBJECT
s += "\n//        OBJECTS:\n"
sL += DefLatexStartClass(classStr+': Object', '\label{sec:mainsystem:object}\n This section provides functions for adding, reading and modifying objects, which can be bodies (mass point, rigid body, finite element, ...), connectors (spring-damper or joint) or general objects. Objects provided terms to the residual of equations resulting from every coordinate given by the nodes. Single-noded objects (e.g.~mass point) provides exactly residual terms for its nodal coordinates. Connectors constrain or penalize two markers, which can be, e.g., position, rigid or coordinate markers. Thus, the dependence of objects is either on the coordinates of the marker-objects/nodes or on nodes which the objects possess themselves.', subSection=True)
#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddObject', cName='[](MainSystem& mainSystem, py::dict itemDict) {return mainSystem.AddMainObject(itemDict); }', 
#                                description="add a object with objectDefinition in dictionary format; returns (global) object number of newly added object",
#                                argList=['itemDict'],
#                                example="objectDict = {'objectType': 'MassPoint', \\\\'physicsMass': 10, \\\\'nodeNumber': 0, \\\\'name': 'example object'} \\\\ mbs.AddObject(objectDict)",
#                                isLambdaFunction = True
#                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddObject', cName='[](MainSystem& mainSystem, py::object pyObject) {return mainSystem.AddMainObjectPyClass(pyObject); }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddObject', cName='AddMainObjectPyClass', 
                                description="add an object with objectDefinition from Python object class; returns (global) object number (type ObjectIndex) of newly added object",
                                argList=['pyObject'],
                                example = "item = MassPoint(name='heavy object', nodeNumber=0, physicsMass=100) \\\\mbs.AddObject(item) \\\\" + "objectDict = {'objectType': 'MassPoint', \\\\'physicsMass': 10, \\\\'nodeNumber': 0, \\\\'name': 'example object'} \\\\ mbs.AddObject(objectDict)"
#                                isLambdaFunction = True
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectNumber', cName='PyGetObjectNumber', 
                                description="get object's number by name (string)",
                                argList=['objectName'],
                                example = "n = mbs.GetObjectNumber('heavy object')"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObject', cName='PyGetObject', 
                                description="get object's dictionary by object number (type ObjectIndex); NOTE: visualization parameters have a prefix 'V'",
                                argList=['objectNumber'],
                                example = "objectDict = mbs.GetObject(0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ModifyObject', cName='PyModifyObject', 
                                description="modify object's dictionary by object number (type ObjectIndex); NOTE: visualization parameters have a prefix 'V'",
                                argList=['objectNumber','objectDict'],
                                example = "mbs.ModifyObject(objectNumber, objectDict)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectDefaults', cName='PyGetObjectDefaults', 
                                description="get object's default values for a certain objectType as (dictionary)",
                                argList=['typeName'],
                                example = "objectType = 'MassPoint'\\\\objectDict = mbs.GetObjectDefaults(objectType)"
                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='CallObjectFunction', cName='PyCallObjectFunction', 
#                                description="call specific object function",
#                                argList=['objectNumber', 'functionName', 'args'],
#                                defaultArgs=['', '', 'py::dict()']
#                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectOutput', cName='PyGetObjectOutputVariable', 
                                description="get object's current output variable from object number (type ObjectIndex) and OutputVariableType; can only be computed for exu.ConfigurationType.Current configuration!",
                                argList=['objectNumber', 'variableType']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectOutputBody', cName='PyGetObjectOutputVariableBody', 
                                description="get body's output variable from object number (type ObjectIndex) and OutputVariableType, using the localPosition as defined in the body, and as used in MarkerBody and SensorBody",
                                argList=['objectNumber', 'variableType', 'localPosition', 'configuration'],
                                defaultArgs=['','','','ConfigurationType::Current'],
                                example = "u = mbs.GetObjectOutputBody(objectNumber = 1, variableType = exu.OutputVariableType.Position, localPosition=[1,0,0], configuration = exu.ConfigurationType.Initial)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectOutputSuperElement', cName='PyGetObjectOutputVariableSuperElement', 
                                description="get output variable from mesh node number of object with type SuperElement (GenericODE2, FFRF, FFRFreduced - CMS) with specific OutputVariableType; the meshNodeNumber is the object's local node number, not the global node number!",
                                argList=['objectNumber', 'variableType', 'meshNodeNumber', 'configuration'],
                                defaultArgs=['','','','ConfigurationType::Current'],
                                example = "u = mbs.GetObjectOutputSuperElement(objectNumber = 1, variableType = exu.OutputVariableType.Position, meshNodeNumber = 12, configuration = exu.ConfigurationType.Initial)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectParameter', cName='PyGetObjectParameter', 
                                description="get objects's parameter from object number (type ObjectIndex) and parameterName; parameter names can be found for the specific items in the reference manual; for visualization parameters, use a 'V' as a prefix",
                                argList=['objectNumber', 'parameterName'],
                                example = "mbs.GetObjectParameter(objectNumber = 0, parameterName = 'nodeNumber')",
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetObjectParameter', cName='PySetObjectParameter', 
                                description="set parameter 'parameterName' of object with object number (type ObjectIndex) to value;; parameter names can be found for the specific items in the reference manual; for visualization parameters, use a 'V' as a prefix",
                                argList=['objectNumber', 'parameterName', 'value'],
                                example = "mbs.SetObjectParameter(objectNumber = 0, parameterName = 'Vshow', value=True)",
                                ); s+=s1; sL+=sL1

sL += DefLatexFinishClass()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#MARKER
s += "\n//        MARKER:\n"
sL += DefLatexStartClass(classStr+': Marker', '\label{sec:mainsystem:marker}\n This section provides functions for adding, reading and modifying markers. Markers define how to measure primal kinematical quantities on objects or nodes (e.g., position, orientation or coordinates themselves), and how to act on the quantities which are dual to the kinematical quantities (e.g., force, torque and generalized forces). Markers provide unique interfaces for loads, sensors and constraints in order to address these quantities independently of the structure of the object or node (e.g., rigid or flexible body).', subSection=True)
#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddMarker', cName='[](MainSystem& mainSystem, py::dict itemDict) {return mainSystem.AddMainMarker(itemDict); }', 
#                                description="add a marker with markerDefinition in dictionary format; returns (global) marker number of newly added marker",
#                                argList=['itemDict'],
#                                example="markerDict = {'markerType': 'NodePosition', \\\\ 'nodeNumber': 0, \\\\ 'name': 'position0'}\\\\ mbs.AddMarker(markerDict)",
#                                isLambdaFunction = True
#                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddMarker', cName='[](MainSystem& mainSystem, py::object pyObject) {return mainSystem.AddMainMarkerPyClass(pyObject); }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddMarker', cName='AddMainMarkerPyClass', 
                                description="add a marker with markerDefinition from Python marker class; returns (global) marker number (type MarkerIndex) of newly added marker",
                                argList=['pyObject'],
                                example = "item = MarkerNodePosition(name='my marker',nodeNumber=1) \\\\mbs.AddMarker(item)\\\\" + "markerDict = {'markerType': 'NodePosition', \\\\ 'nodeNumber': 0, \\\\ 'name': 'position0'}\\\\ mbs.AddMarker(markerDict)"
#                                isLambdaFunction = True
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetMarkerNumber', cName='PyGetMarkerNumber', 
                                description="get marker's number by name (string)",
                                argList=['markerName'],
                                example = "n = mbs.GetMarkerNumber('my marker')"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetMarker', cName='PyGetMarker', 
                                description="get marker's dictionary by index",
                                argList=['markerNumber'],
                                example = "markerDict = mbs.GetMarker(0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ModifyMarker', cName='PyModifyMarker', 
                                description="modify marker's dictionary by index",
                                argList=['markerNumber','markerDict'],
                                example = "mbs.ModifyMarker(markerNumber, markerDict)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetMarkerDefaults', cName='PyGetMarkerDefaults', 
                                description="get marker's default values for a certain markerType as (dictionary)",
                                argList=['typeName'],
                                example = "markerType = 'NodePosition'\\\\markerDict = mbs.GetMarkerDefaults(markerType)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetMarkerParameter', cName='PyGetMarkerParameter', 
                                description="get markers's parameter from markerNumber and parameterName; parameter names can be found for the specific items in the reference manual",
                                argList=['markerNumber', 'parameterName']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetMarkerParameter', cName='PySetMarkerParameter', 
                                description="set parameter 'parameterName' of marker with markerNumber to value; parameter names can be found for the specific items in the reference manual",
                                argList=['markerNumber', 'parameterName', 'value']
                                ); s+=s1; sL+=sL1


sL += DefLatexFinishClass()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#LOAD
s += "\n//        LOADS:\n"
sL += DefLatexStartClass(classStr+': Load', '\label{sec:mainsystem:load}\n This section provides functions for adding, reading and modifying operating loads. Loads are used to act on the quantities which are dual to the primal kinematic quantities, such as displacement and rotation. Loads represent, e.g., forces, torques or generalized forces.', subSection=True)
#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddLoad', cName='[](MainSystem& mainSystem, py::dict itemDict) {return mainSystem.AddMainLoad(itemDict); }', 
#                                description="add a load with loadDefinition in dictionary format; returns (global) load number of newly added load",
#                                argList=['itemDict'],
#                                example="loadDict = {'loadType': 'ForceVector',\\\\ 'markerNumber': 0,\\\\ 'loadVector': [1.0, 0.0, 0.0],\\\\ 'name': 'heavy load'} \\\\ mbs.AddLoad(loadDict)",
#                                isLambdaFunction = True
#                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddLoad', cName='[](MainSystem& mainSystem, py::object pyObject) {return mainSystem.AddMainLoadPyClass(pyObject); }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddLoad', cName='AddMainLoadPyClass', 
                                description="add a load with loadDefinition from Python load class; returns (global) load number (type LoadIndex) of newly added load",
                                argList=['pyObject'],
                                example = "item = mbs.AddLoad(LoadForceVector(loadVector=[1,0,0], markerNumber=0, name='heavy load')) \\\\mbs.AddLoad(item)\\\\" + "loadDict = {'loadType': 'ForceVector',\\\\ 'markerNumber': 0,\\\\ 'loadVector': [1.0, 0.0, 0.0],\\\\ 'name': 'heavy load'} \\\\ mbs.AddLoad(loadDict)"
#                                isLambdaFunction = True
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetLoadNumber', cName='PyGetLoadNumber', 
                                description="get load's number by name (string)",
                                argList=['loadName'],
                                example = "n = mbs.GetLoadNumber('heavy load')"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetLoad', cName='PyGetLoad', 
                                description="get load's dictionary by index",
                                argList=['loadNumber'],
                                example = "loadDict = mbs.GetLoad(0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ModifyLoad', cName='PyModifyLoad', 
                                description="modify load's dictionary by index",
                                argList=['loadNumber','loadDict'],
                                example = "mbs.ModifyLoad(loadNumber, loadDict)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetLoadDefaults', cName='PyGetLoadDefaults', 
                                description="get load's default values for a certain loadType as (dictionary)",
                                argList=['typeName'],
                                example = "loadType = 'ForceVector'\\\\loadDict = mbs.GetLoadDefaults(loadType)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetLoadValues', cName='PyGetLoadValues', 
                                description="Get current load values, specifically if user-defined loads are used; can be scalar or vector-valued return value",
                                argList=['loadNumber']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetLoadParameter', cName='PyGetLoadParameter', 
                                description="get loads's parameter from loadNumber and parameterName; parameter names can be found for the specific items in the reference manual",
                                argList=['loadNumber', 'parameterName']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetLoadParameter', cName='PySetLoadParameter', 
                                description="set parameter 'parameterName' of load with loadNumber to value; parameter names can be found for the specific items in the reference manual",
                                argList=['loadNumber', 'parameterName', 'value']
                                ); s+=s1; sL+=sL1

sL += DefLatexFinishClass()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SENSORS
s += "\n//        SENSORS:\n"
sL += DefLatexStartClass(classStr+': Sensor', '\label{sec:mainsystem:sensor}\n This section provides functions for adding, reading and modifying operating sensors. Sensors are used to measure information in nodes, objects, markers, and loads for output in a file.', subSection=True)
#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddSensor', cName='[](MainSystem& mainSystem, py::dict itemDict) {return mainSystem.AddMainSensor(itemDict); }', 
#                                description="add a sensor with sensor definition in dictionary format; returns (global) sensor number of newly added sensor",
#                                argList=['itemDict'],
#                                example="sensorDict = {'sensorType': 'Node',\\\\ 'nodeNumber': 0,\\\\ 'fileName': 'sensor.txt',\\\\ 'name': 'test sensor'} \\\\ mbs.AddSensor(sensorDict)",
#                                isLambdaFunction = True
#                                ); s+=s1; sL+=sL1

#[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddSensor', cName='[](MainSystem& mainSystem, py::object pyObject) {return mainSystem.AddMainSensorPyClass(pyObject); }', 
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddSensor', cName='AddMainSensorPyClass',
                                description="add a sensor with sensor definition from Python sensor class; returns (global) sensor number (type SensorIndex) of newly added sensor",
                                argList=['pyObject'],
                                example = "item = mbs.AddSensor(SensorNode(sensorType= exu.SensorType.Node, nodeNumber=0, name='test sensor')) \\\\mbs.AddSensor(item)\\\\" + "sensorDict = {'sensorType': 'Node',\\\\ 'nodeNumber': 0,\\\\ 'fileName': 'sensor.txt',\\\\ 'name': 'test sensor'} \\\\ mbs.AddSensor(sensorDict)"
#                                isLambdaFunction = True
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensorNumber', cName='PyGetSensorNumber', 
                                description="get sensor's number by name (string)",
                                argList=['sensorName'],
                                example = "n = mbs.GetSensorNumber('test sensor')"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensor', cName='PyGetSensor', 
                                description="get sensor's dictionary by index",
                                argList=['sensorNumber'],
                                example = "sensorDict = mbs.GetSensor(0)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ModifySensor', cName='PyModifySensor', 
                                description="modify sensor's dictionary by index",
                                argList=['sensorNumber','sensorDict'],
                                example = "mbs.ModifySensor(sensorNumber, sensorDict)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensorDefaults', cName='PyGetSensorDefaults', 
                                description="get sensor's default values for a certain sensorType as (dictionary)",
                                argList=['typeName'],
                                example = "sensorType = 'Node'\\\\sensorDict = mbs.GetSensorDefaults(sensorType)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensorValues', cName='PyGetSensorValues', 
                                description="get sensors's values for configuration; can be a scalar or vector-valued return value!",
                                defaultArgs=['','ConfigurationType::Current'],
                                argList=['sensorNumber', 'configuration']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensorStoredData', cName='PyGetSensorStoredData',
                                description="get sensors's internally stored data as matrix (all time points stored); rows are containing time and sensor values as obtained by sensor (e.g., time, and x, y, and z value of position)",
                                defaultArgs=[''],
                                argList=['sensorNumber']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSensorParameter', cName='PyGetSensorParameter', 
                                description="get sensors's parameter from sensorNumber and parameterName; parameter names can be found for the specific items in the reference manual",
                                argList=['sensorNumber', 'parameterName']
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetSensorParameter', cName='PySetSensorParameter', 
                                description="set parameter 'parameterName' of sensor with sensorNumber to value; parameter names can be found for the specific items in the reference manual",
                                argList=['sensorNumber', 'parameterName', 'value']
                                ); s+=s1; sL+=sL1

sL += DefLatexFinishClass() #Sensors

#now finalize pybind class, but do nothing on latex side (sL1 ignored)
[s1,sL1] = DefPyFinishClass('MainSystem'); s+=s1 #; sL+=sL1



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pyClassStr = 'SystemData'
classStr = 'Main'+pyClassStr
[s1,sL1] = DefPyStartClass(classStr,pyClassStr, 'This is the data structure of a system which contains Objects (bodies/constraints/...), Nodes, Markers and Loads. The SystemData structure allows advanced access to this data, which HAS TO BE USED WITH CARE, as unexpected results and system crash might happen.' +
        ' \\\\ \n Usage: \\\\ \\small \n\\texttt{\\#obtain current ODE2 system vector (e.g. after static simulation finished): \\\\ u = mbs.systemData.GetODE2Coordinates() \\\\ \\#set initial ODE2 vector for next simulation:\\\\ \nmbs.systemData.SetODE2Coordinates(coordinates=u,configurationType=exu.ConfigurationType.Initial)}\n')
s+=s1; sL+=sL1


s += "\n//        General functions:\n"
#sL += '\\\\ \n'+classStr+': General functions', 'These functions allow to obtain system information (e.g. for debug purposes)', subSection=True)

#+++++++++++++++++++++++++++++++++
#General functions:
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfLoads', cName='[](const MainSystemData& msd) {return msd.GetMainLoads().NumberOfItems(); }', 
                                description="return number of loads in system",
                                isLambdaFunction = True,
                                example = 'print(mbs.systemData.NumberOfLoads())'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfMarkers', cName='[](const MainSystemData& msd) {return msd.GetMainMarkers().NumberOfItems(); }', 
                                description="return number of markers in system",
                                isLambdaFunction = True,
                                example = 'print(mbs.systemData.NumberOfMarkers())'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfNodes', cName='[](const MainSystemData& msd) {return msd.GetMainNodes().NumberOfItems(); }', 
                                description="return number of nodes in system",
                                isLambdaFunction = True,
                                example = 'print(mbs.systemData.NumberOfNodes())'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfObjects', cName='[](const MainSystemData& msd) {return msd.GetMainObjects().NumberOfItems(); }', 
                                description="return number of objects in system",
                                isLambdaFunction = True,
                                example = 'print(mbs.systemData.NumberOfObjects())'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='NumberOfSensors', cName='[](const MainSystemData& msd) {return msd.GetMainSensors().NumberOfItems(); }', 
                                description="return number of sensors in system",
                                isLambdaFunction = True,
                                example = 'print(mbs.systemData.NumberOfSensors())'); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ODE2Size', cName='PyODE2Size', 
                                description="get size of ODE2 coordinate vector for given configuration (only works correctly after mbs.Assemble() )",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "print('ODE2 size=',mbs.systemData.ODE2Size())"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='ODE1Size', cName='PyODE1Size', 
                                description="get size of ODE1 coordinate vector for given configuration (only works correctly after mbs.Assemble() )",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "print('ODE1 size=',mbs.systemData.ODE1Size())"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AEsize', cName='PyAEsize', 
                                description="get size of AE coordinate vector for given configuration (only works correctly after mbs.Assemble() )",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "print('AE size=',mbs.systemData.AEsize())"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='DataSize', cName='PyDataSize', 
                                description="get size of Data coordinate vector for given configuration (only works correctly after mbs.Assemble() )",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "print('Data size=',mbs.systemData.DataSize())"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SystemSize', cName='PySystemSize', 
                                description="get size of System coordinate vector for given configuration (only works correctly after mbs.Assemble() )",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "print('System size=',mbs.systemData.SystemSize())"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetTime', cName='PyGetStateTime', 
                                description="get configuration dependent time.",
                                argList=['configurationType'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "mbs.systemData.GetTime(exu.ConfigurationType.Initial)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetTime', cName='PySetStateTime', 
                                description="set configuration dependent time; use this access with care, e.g. in user-defined solvers.",
                                argList=['newTime','configurationType'],
                                defaultArgs=['', 'exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "mbs.systemData.SetTime(10., exu.ConfigurationType.Initial)"
                                ); s+=s1; sL+=sL1


#removed2021-05-01
# [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetCurrentTime', cName='PyGetCurrentTime', 
#                                 description="DEPRECATED; get current (simulation) time; time is updated in time integration solvers and in static solver; use this function e.g. during simulation to define time-dependent loads",
#                                 example = "mbs.systemData.GetCurrentTime()"
#                                 ); s+=s1; sL+=sL1

# [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetVisualizationTime', cName='PySetVisualizationTime', 
#                                 description="DEPRECATED; set time for render window (visualization)",
#                                 example = "mbs.systemData.SetVisualizationTime(1.3)"
#                                 ); s+=s1; sL+=sL1


[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Info', cName='[](const MainSystemData& msd) {pout << msd.PyInfoDetailed(); }', 
                                description="print detailed system information for every item; for short information use print(mbs)",
                                isLambdaFunction = True,
                                example = 'mbs.systemData.Info()'); s+=s1; sL+=sL1



sL += DefLatexFinishClass()

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
s += "\n//        Coordinate access:\n"
sL += DefLatexStartClass(pyClassStr+': Access coordinates', '\label{sec:mbs:systemData}This section provides access functions to global coordinate vectors. Assigning invalid values or using wrong vector size might lead to system crash and unexpected results.', subSection=True)
#+++++++++++++++++++++++++++++++++
#coordinate access functions:

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetODE2Coordinates', cName='GetODE2Coords', 
                                description="get ODE2 system coordinates (displacements) for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "uCurrent = mbs.systemData.GetODE2Coordinates()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetODE2Coordinates', cName='SetODE2Coords', 
                                description="set ODE2 system coordinates (displacements) for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetODE2Coordinates(uCurrent)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetODE2Coordinates_t', cName='GetODE2Coords_t', 
                                description="get ODE2 system coordinates (velocities) for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "vCurrent = mbs.systemData.GetODE2Coordinates_t()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetODE2Coordinates_t', cName='SetODE2Coords_t', 
                                description="set ODE2 system coordinates (velocities) for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetODE2Coordinates_t(vCurrent)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetODE2Coordinates_tt', cName='GetODE2Coords_tt', 
                                description="get ODE2 system coordinates (accelerations) for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "vCurrent = mbs.systemData.GetODE2Coordinates_tt()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetODE2Coordinates_tt', cName='SetODE2Coords_tt', 
                                description="set ODE2 system coordinates (accelerations) for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetODE2Coordinates_tt(aCurrent)"
                                ); s+=s1; sL+=sL1


[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetODE1Coordinates', cName='GetODE1Coords', 
                                description="get ODE1 system coordinates (displacements) for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "qCurrent = mbs.systemData.GetODE1Coordinates()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetODE1Coordinates', cName='SetODE1Coords', 
                                description="set ODE1 system coordinates (displacements) for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetODE1Coordinates(qCurrent)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetAECoordinates', cName='GetAECoords', 
                                description="get algebraic equations (AE) system coordinates for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "lambdaCurrent = mbs.systemData.GetAECoordinates()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetAECoordinates', cName='SetAECoords', 
                                description="set algebraic equations (AE) system coordinates for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetAECoordinates(lambdaCurrent)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetDataCoordinates', cName='GetDataCoords', 
                                description="get system data coordinates for given configuration (default: exu.Configuration.Current)",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'],
                                example = "dataCurrent = mbs.systemData.GetDataCoordinates()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetDataCoordinates', cName='SetDataCoords', 
                                description="set system data coordinates for given configuration (default: exu.Configuration.Current); invalid vector size may lead to system crash!",
                                argList=['coordinates','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'],
                                example = "mbs.systemData.SetDataCoordinates(dataCurrent)"
                                ); s+=s1; sL+=sL1



[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetSystemState', cName='PyGetSystemState', 
                                description="get system state for given configuration (default: exu.Configuration.Current); state vectors do not include the non-state derivatives ODE1_t and ODE2_tt and the time; function is copying data - not highly efficient; format of pyList: [ODE2Coords, ODE2Coords_t, ODE1Coords, AEcoords, dataCoords]",
                                argList=['configuration'],
                                defaultArgs=['exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "sysStateList = mbs.systemData.GetSystemState()"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetSystemState', cName='PySetSystemState', 
                                description="set system data coordinates for given configuration (default: exu.Configuration.Current); invalid list of vectors / vector size may lead to system crash; write access to state vectors (but not the non-state derivatives ODE1_t and ODE2_tt and the time); function is copying data - not highly efficient; format of pyList: [ODE2Coords, ODE2Coords_t, ODE1Coords, AEcoords, dataCoords]",
                                argList=['systemStateList','configuration'],
                                defaultArgs=['','exu.ConfigurationType::Current'], #exu will be removed for binding
                                example = "mbs.systemData.SetSystemState(sysStateList, configuration = exu.ConfigurationType.Initial)"
                                ); s+=s1; sL+=sL1


sL += DefLatexFinishClass()

#+++++++++++++++++++++++++++++++++
#LTG-functions:
s += "\n//        LTG readout functions:\n"
sL += DefLatexStartClass(pyClassStr+': Get object LTG coordinate mappings', '\\label{sec:systemData:ObjectLTG}This section provides access functions the \\ac{LTG}-lists for every object (body, constraint, ...) in the system. For details on the \\ac{LTG} mapping, see \\refSection{sec:systemData:LTG}', subSection=True)

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectLTGODE2', cName='PyGetObjectLocalToGlobalODE2', 
                                description="get local-to-global coordinate mapping (list of global coordinate indices) for ODE2 coordinates; only available after Assemble()",
                                argList=['objectNumber'],
                                example = "ltgObject4 = mbs.systemData.GetObjectLTGODE2(4)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectLTGODE1', cName='PyGetObjectLocalToGlobalODE1', 
                                description="get local-to-global coordinate mapping (list of global coordinate indices) for ODE1 coordinates; only available after Assemble()",
                                argList=['objectNumber'],
                                example = "ltgObject4 = mbs.systemData.GetObjectLTGODE1(4)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectLTGAE', cName='PyGetObjectLocalToGlobalAE', 
                                description="get local-to-global coordinate mapping (list of global coordinate indices) for algebraic equations (AE) coordinates; only available after Assemble()",
                                argList=['objectNumber'],
                                example = "ltgObject4 = mbs.systemData.GetObjectLTGODE2(4)"
                                ); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetObjectLTGData', cName='PyGetObjectLocalToGlobalData', 
                                description="get local-to-global coordinate mapping (list of global coordinate indices) for data coordinates; only available after Assemble()",
                                argList=['objectNumber'],
                                example = "ltgObject4 = mbs.systemData.GetObjectLTGData(4)"
                                ); s+=s1; sL+=sL1

sL += DefLatexFinishClass()

#now finalize pybind class, but do nothing on latex side (sL1 ignored)
[s1,sL1] = DefPyFinishClass('SystemData'); s+=s1 #; sL+=sL1


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for MatrixContainer
classStr = 'PyMatrixContainer'
pyClassStr = 'MatrixContainer'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The MatrixContainer is a versatile representation for dense and sparse matrices." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{MatrixContainer} with \\texttt{mc = MatrixContainer()} \n'+
        '  \\item Create \\texttt{MatrixContainer} with dense matrix \\texttt{mc = MatrixContainer(matrix)}, where matrix can be a list of lists of a numpy array \n'+
        '  \\item Set with dense \\text{pyArray} (a numpy array): \\texttt{mc.SetWithDenseMatrix(pyArray, bool useDenseMatrix = True)}\n'+
        '  \\item Set with sparse \\text{pyArray} (a numpy array), which has 3 colums and according rows containing the sparse triplets \\texttt{(row, col, value)} describing the sparse matrix\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("matrix"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetWithDenseMatrix', cName='SetWithDenseMatrix', 
                                argList=['pyArray','useDenseMatrix'],
                                defaultArgs=['','false'],
                                description="set MatrixContainer with dense numpy array; array (=matrix) contains values and matrix size information; if useDenseMatrix=True, matrix will be stored internally as dense matrix, otherwise it will be converted and stored as sparse matrix (which may speed up computations for larger problems)"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetWithSparseMatrixCSR', cName='SetWithSparseMatrixCSR', 
                                argList=['numberOfRowsInit', 'numberOfColumnsInit', 'pyArrayCSR','useDenseMatrix'],
                                defaultArgs=['','','','true'],
                                description="set with sparse CSR matrix format: numpy array 'pyArrayCSR' contains sparse triplet (row, col, value) per row; numberOfRows and numberOfColumns given extra; if useDenseMatrix=True, matrix will be converted and stored internally as dense matrix, otherwise it will be stored as sparse matrix"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                                description="convert MatrixContainer to numpy array (dense) or dictionary (sparse): containing nr. of rows, nr. of columns, numpy matrix with sparse triplets"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Convert2DenseMatrix', cName='Convert2DenseMatrix', 
                                description="convert MatrixContainer to dense numpy array (SLOW and may fail for too large sparse matrices)"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='UseDenseMatrix', cName='UseDenseMatrix', 
                                description="returns True if dense matrix is used, otherwise False"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', cName='[](const PyMatrixContainer &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                                description="return the string representation of the MatrixContainer",
                                isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('MatrixContainer'); s+=s1; sL+=sL1


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for PyVector3DList
classStr = 'PyVector3DList'
pyClassStr = 'Vector3DList'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The Vector3DList is used to represent lists of 3D vectors. This is used to transfer such lists from Python to C++." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{Vector3DList} with \\texttt{x = Vector3DList()} \n'+
        '  \\item Create \\texttt{Vector3DList} with list of numpy arrays:\\\\ \\texttt{x = Vector3DList([ numpy.array([1.,2.,3.]), numpy.array([4.,5.,6.]) ])}\n'+
        '  \\item Create \\texttt{Vector3DList} with list of lists \\texttt{x = Vector3DList([[1.,2.,3.], [4.,5.,6.]])}\n'+
        '  \\item Append item: \\texttt{x.Append([0.,2.,4.])}\n'+
        '  \\item Convert into list of numpy arrays: \\texttt{x.GetPythonObject()}\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("listOfArrays"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Append', cName='PyAppend', 
                               argList=['pyArray'],
                               description="add single array or list to Vector3DList; array or list must have appropriate dimension!"); s+=s1; sL+=sL1
                                                                                                            
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                               description="convert Vector3DList into (copied) list of numpy arrays"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__len__', 
                               cName='[](const PyVector3DList &item) {\n            return item.NumberOfItems(); }', 
                               description="return length of the Vector3DList, using len(data) where data is the Vector3DList",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__setitem__', 
                               cName='[](PyVector3DList &item, Index index, const py::object& vector) {\n            item.PySetItem(index, vector); }', 
                               description="set list item 'index' with data, write: data[index] = ...",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__getitem__', 
                               cName='[](const PyVector3DList &item, Index index) {\n            return py::array_t<Real>(item[index].NumberOfItems(), item[index].GetDataPointer()); }', 
                               description="get copy of list item with 'index' as vector",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', 
                               cName='[](const PyVector3DList &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                               description="return the string representation of the Vector3DList data, e.g.: print(data)",
                               isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('PyVector3DList'); s+=s1; sL+=sL1

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for PyVector2DList
classStr = 'PyVector2DList'
pyClassStr = 'Vector2DList'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The Vector2DList is used to represent lists of 2D vectors. This is used to transfer such lists from Python to C++." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{Vector2DList} with \\texttt{x = Vector2DList()} \n'+
        '  \\item Create \\texttt{Vector2DList} with list of numpy arrays:\\\\ \\texttt{x = Vector2DList([ numpy.array([1.,2.]), numpy.array([4.,5.]) ])}\n'+
        '  \\item Create \\texttt{Vector2DList} with list of lists \\texttt{x = Vector2DList([[1.,2.], [4.,5.]])}\n'+
        '  \\item Append item: \\texttt{x.Append([0.,2.])}\n'+
        '  \\item Convert into list of numpy arrays: \\texttt{x.GetPythonObject()}\n'+
        '  \\item similar to Vector3DList !\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("listOfArrays"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Append', cName='PyAppend', 
                               argList=['pyArray'],
                               description="add single array or list to Vector2DList; array or list must have appropriate dimension!"); s+=s1; sL+=sL1
                                                                                                            
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                               description="convert Vector2DList into (copied) list of numpy arrays"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__len__', 
                               cName='[](const PyVector2DList &item) {\n            return item.NumberOfItems(); }', 
                               description="return length of the Vector2DList, using len(data) where data is the Vector2DList",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__setitem__', 
                               cName='[](PyVector2DList &item, Index index, const py::object& vector) {\n            item.PySetItem(index, vector); }', 
                               description="set list item 'index' with data, write: data[index] = ...",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__getitem__', 
                               cName='[](const PyVector2DList &item, Index index) {\n            return py::array_t<Real>(item[index].NumberOfItems(), item[index].GetDataPointer()); }', 
                               description="get copy of list item with 'index' as vector",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', 
                               cName='[](const PyVector2DList &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                               description="return the string representation of the Vector2DList data, e.g.: print(data)",
                               isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('PyVector2DList'); s+=s1; sL+=sL1

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for PyVector6DList
classStr = 'PyVector6DList'
pyClassStr = 'Vector6DList'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The Vector6DList is used to represent lists of 6D vectors. This is used to transfer such lists from Python to C++." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{Vector6DList} with \\texttt{x = Vector6DList()} \n'+
        '  \\item Convert into list of numpy arrays: \\texttt{x.GetPythonObject()}\n'+
        '  \\item similar to Vector3DList !\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("listOfArrays"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Append', cName='PyAppend', 
                               argList=['pyArray'],
                               description="add single array or list to Vector6DList; array or list must have appropriate dimension!"); s+=s1; sL+=sL1
                                                                                                            
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                               description="convert Vector6DList into (copied) list of numpy arrays"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__len__', 
                               cName='[](const PyVector6DList &item) {\n            return item.NumberOfItems(); }', 
                               description="return length of the Vector6DList, using len(data) where data is the Vector6DList",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__setitem__', 
                               cName='[](PyVector6DList &item, Index index, const py::object& vector) {\n            item.PySetItem(index, vector); }', 
                               description="set list item 'index' with data, write: data[index] = ...",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__getitem__', 
                               cName='[](const PyVector6DList &item, Index index) {\n            return py::array_t<Real>(item[index].NumberOfItems(), item[index].GetDataPointer()); }', 
                               description="get copy of list item with 'index' as vector",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', 
                               cName='[](const PyVector6DList &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                               description="return the string representation of the Vector6DList data, e.g.: print(data)",
                               isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('PyVector6DList'); s+=s1; sL+=sL1

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for PyMatrix3DList
classStr = 'PyMatrix3DList'
pyClassStr = 'Matrix3DList'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The Matrix3DList is used to represent lists of 3D Matrices. . This is used to transfer such lists from Python to C++." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{Matrix3DList} with \\texttt{x = Matrix3DList()} \n'+
        '  \\item Create \\texttt{Matrix3DList} with list of numpy arrays:\\\\  \\texttt{x = Matrix3DList([ numpy.eye(3), numpy.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]) ])}\n'+
        # '  \\item Create \\texttt{Matrix3DList} with list of lists \\texttt{x = Matrix3DList([[1.,2.,3.], [4.,5.,6.]])}\n'+
        '  \\item Append item: \\texttt{x.Append(numpy.eye(3))}\n'+
        '  \\item Convert into list of numpy arrays: \\texttt{x.GetPythonObject()}\n'+
        '  \\item similar to Vector3DList !\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("listOfArrays"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Append', cName='PyAppend', 
                               argList=['pyArray'],
                               description="add single 3D array or list of lists to Matrix3DList; array or lists must have appropriate dimension!"); s+=s1; sL+=sL1
                                                                                                            
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                               description="convert Matrix3DList into (copied) list of 2D numpy arrays"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__len__', 
                               cName='[](const PyMatrix3DList &item) {\n            return item.NumberOfItems(); }', 
                               description="return length of the Matrix3DList, using len(data) where data is the Matrix3DList",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__setitem__', 
                               cName='[](PyMatrix3DList &item, Index index, const py::object& matrix) {\n            item.PySetItem(index, matrix); }', 
                               description="set list item 'index' with matrix, write: data[index] = ...",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__getitem__', 
                               cName='[](const PyMatrix3DList &item, Index index) {\n            return item.PyGetItem(index); }', 
                               description="get copy of list item with 'index' as matrix",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', 
                               cName='[](const PyMatrix3DList &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                               description="return the string representation of the Matrix3DList data, e.g.: print(data)",
                               isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('PyMatrix3DList'); s+=s1; sL+=sL1

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for PyMatrix6DList
classStr = 'PyMatrix6DList'
pyClassStr = 'Matrix6DList'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "The Matrix6DList is used to represent lists of 6D Matrices. . This is used to transfer such lists from Python to C++." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Create empty \\texttt{Matrix6DList} with \\texttt{x = Matrix6DList()} \n'+
        '  \\item Create \\texttt{Matrix6DList} with list of numpy arrays:\\\\  \\texttt{x = Matrix6DList([ numpy.eye(6), 2*numpy.eye(6) ])}\n'+
        '  \\item Append item: \\texttt{x.Append(numpy.eye(6))}\n'+
        '  \\item Convert into list of numpy arrays: \\texttt{x.GetPythonObject()}\n'+
        '  \\item similar to Matrix3DList !\n'+
        '\\ei\n')
s+=s1; sL+=sL1

s+= '        .def(py::init<const py::object&>(), py::arg("listOfArrays"))\n' #constructor with numpy array or list of lists

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Append', cName='PyAppend', 
                               argList=['pyArray'],
                               description="add single 6D array or list of lists to Matrix6DList; array or lists must have appropriate dimension!"); s+=s1; sL+=sL1
                                                                                                            
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                               description="convert Matrix6DList into (copied) list of 2D numpy arrays"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__len__', 
                               cName='[](const PyMatrix6DList &item) {\n            return item.NumberOfItems(); }', 
                               description="return length of the Matrix6DList, using len(data) where data is the Matrix6DList",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__setitem__', 
                               cName='[](PyMatrix6DList &item, Index index, const py::object& matrix) {\n            item.PySetItem(index, matrix); }', 
                               description="set list item 'index' with matrix, write: data[index] = ...",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__getitem__', 
                               cName='[](const PyMatrix6DList &item, Index index) {\n            return item.PyGetItem(index); }', 
                               description="get copy of list item with 'index' as matrix",
                               isLambdaFunction = True); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', 
                               cName='[](const PyMatrix6DList &item) {\n            return EXUstd::ToString(item.GetPythonObject()); }', 
                               description="return the string representation of the Matrix6DList data, e.g.: print(data)",
                               isLambdaFunction = True); s+=s1; sL+=sL1

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('PyMatrix6DList'); s+=s1; sL+=sL1

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for MatrixContainer
classStr = 'PyGeneralContact'
pyClassStr = 'GeneralContact'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "\\label{sec:GeneralContact}Structure to define general and highly efficient contact functionality in multibody systems\\footnote{Note that GeneralContact is still developed, use with care.}. For further explanations and theoretical backgrounds, see \\refSection{secContactTheory}." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item Add \\texttt{GeneralContact} to mbs \\texttt{gContact = mbs.AddGeneralContact()} \n'+
        '  \\item Add contact elements, e.g., \\texttt{gContact.AddSphereWithMarker(...)}, using appropriate arguments \n'+
        '  \\item Call SetFrictionPairings(...) to set friction pairings and adjust searchTree if needed.\n'+
        '\\ei\n')
#{\\bf NOTE: For internal use only! GeneralContact is currently developed and must be used with care; interfaces may change significantly in upcoming versions}]
s+=s1; sL+=sL1

#already included: s+= '        .def(py::init<>())\n' #empty constructor 

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='GetPythonObject', cName='GetPythonObject', 
                                description="convert member variables of GeneralContact into dictionary; use this for debug only!"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Reset', cName='Reset', 
                                argList=['freeMemory'],
                                defaultArgs=['true'],
                                description="remove all contact objects and reset contact parameters"); s+=s1; sL+=sL1

s +=  '        .def_readwrite("isActive", &PyGeneralContact::isActive, py::return_value_policy::reference)\n' 
sL += '  isActive & default = True (compute contact); if isActive=False, no contact computation is performed for this contact set \\\\ \\hline  \n'

s +=  '        .def_readwrite("verboseMode", &PyGeneralContact::verboseMode, py::return_value_policy::reference)\n' 
sL += '  verboseMode & default = 0; verboseMode = 1 or higher outputs useful information on the contact creation and computation \\\\ \\hline  \n'

s +=  '        .def_readwrite("visualization", &PyGeneralContact::visualization, py::return_value_policy::reference)\n' 
sL += '  visualization & access visualization data structure \\\\ \\hline  \n'

#s +=  '        .def_readwrite("intraSpheresContact", &PyGeneralContact::settings.intraSpheresContact, py::return_value_policy::reference)\n' 
s +=  '        .def_property("sphereSphereContact", &PyGeneralContact::GetSphereSphereContact, &PyGeneralContact::SetSphereSphereContact)\n' 
sL += '  sphereSphereContact & activate/deactivate contact between spheres \\\\ \\hline  \n'

s +=  '        .def_property("sphereSphereFrictionRecycle", &PyGeneralContact::GetSphereSphereFrictionRecycle, &PyGeneralContact::SetSphereSphereFrictionRecycle)\n' 
sL += '  sphereSphereFrictionRecycle & False: compute static friction force based on tangential velocity; True: recycle friction from previous PostNewton step, which greatly improves convergence, but may lead to unphysical artifacts; will be solved in future by step reduction \\\\ \\hline  \n'

s +=  '        .def_property("minRelDistanceSpheresTriangles", &PyGeneralContact::GetMinRelDistanceSpheresTriangles, &PyGeneralContact::SetMinRelDistanceSpheresTriangles)\n' 
sL += '  minRelDistanceSpheresTriangles & (default=1e-10) tolerance (relative to sphere radiues) below which the contact between triangles and spheres is ignored; used for spheres directly attached to triangles \\\\ \\hline  \n'

s +=  '        .def_property("frictionProportionalZone", &PyGeneralContact::GetFrictionProportionalZone, &PyGeneralContact::SetFrictionProportionalZone)\n' 
sL += '  frictionProportionalZone & (default=0.001) velocity $v_{\mu,reg}$ upon which the dry friction coefficient is interpolated linearly (regularized friction model); must be greater 0; very small values cause oscillations in friction force \\\\ \\hline  \n'

s +=  '        .def_property("frictionVelocityPenalty", &PyGeneralContact::GetFrictionVelocityPenalty, &PyGeneralContact::SetFrictionVelocityPenalty)\n' 
sL += '  frictionVelocityPenalty & (default=1e3) regularization factor for friction [N/(m$^2 \cdot$m/s) ];$k_{\mu,reg}$, multiplied with tangential velocity to compute friciton force as long as it is smaller than $\mu$ times contact force; large values cause oscillations in friction force \\\\ \\hline  \n'

s +=  '        .def_property("excludeOverlappingTrigSphereContacts", &PyGeneralContact::GetExcludeOverlappingTrigSphereContacts, &PyGeneralContact::SetExcludeOverlappingTrigSphereContacts)\n' 
sL += '  		excludeOverlappingTrigSphereContacts & (default=True) for consistent, closed meshes, we can exclude overlapping contact triangles (which would cause holes if mesh is overlapping and not consistent!!!) \\\\ \\hline  \n'

s +=  '        .def_property("excludeDuplicatedTrigSphereContactPoints", &PyGeneralContact::GetExcludeDuplicatedTrigSphereContactPoints, &PyGeneralContact::SetExcludeDuplicatedTrigSphereContactPoints)\n' 
sL += '  		excludeDuplicatedTrigSphereContactPoints & (default=False) run additional checks for double contacts at edges or vertices, being more accurate but can cause additional costs if many contacts \\\\ \\hline  \n'

s +=  '        .def_property("ancfCableUseExactMethod", &PyGeneralContact::GetAncfCableUseExactMethod, &PyGeneralContact::SetAncfCableUseExactMethod)\n' 
sL += '  		ancfCableUseExactMethod & (default=True) if true, uses exact computation of intersection of 3rd order polynomials and contacting circles \\\\ \\hline  \n'

s +=  '        .def_property("ancfCableNumberOfContactSegments", &PyGeneralContact::GetAncfCableNumberOfContactSegments, &PyGeneralContact::SetAncfCableNumberOfContactSegments)\n' 
sL += '  		ancfCableNumberOfContactSegments & (default=1) number of segments to be used in case that ancfCableUseExactMethod=False; maximum number of segments=3 \\\\ \\hline  \n'


# [s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='FinalizeContact', cName='PyFinalizeContact', 
#                                 argList=['mainSystem','searchTreeSize','frictionPairingsInit','searchTreeBoxMin','searchTreeBoxMax'],
#                                 defaultArgs=['','','', '(std::vector<Real>)Vector3D( EXUstd::MAXREAL )','(std::vector<Real>)Vector3D( EXUstd::LOWESTREAL )'],
#                                 description="WILL CHANGE IN FUTURE: Call this function after mbs.Assemble(); precompute some contact arrays (mainSystem needed) and set up necessary parameters for contact: friction, SearchTree, etc.; done after all contacts have been added; function performs checks; empty box will autocompute size!"); s+=s1; sL+=sL1
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetFrictionPairings', cName='SetFrictionPairings', 
                               argList=['frictionPairings'],
                               example='\\#set 3 surface friction types, all being 0.1:\\\\gContact.SetFrictionPairings(0.1*np.ones((3,3)));',
                               description="set Coulomb friction coefficients for pairings of materials (e.g., use material 0,1, then the entries (0,1) and (1,0) define the friction coefficients for this pairing); matrix should be symmetric!"); s+=s1; sL+=sL1
                
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetFrictionProportionalZone', cName='SetFrictionProportionalZone', 
                               argList=['frictionProportionalZone'],
                               description="regularization for friction (m/s); used for all contacts"); s+=s1; sL+=sL1
                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetSearchTreeCellSize', cName='SetSearchTreeCellSize', 
                               argList=['numberOfCells'],
                               example='gContact.SetSearchTreeInitSize([10,10,10])',
                               description="set number of cells of search tree (boxed search) in x, y and z direction"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='SetSearchTreeBox', cName='SetSearchTreeBox', 
                               argList=['pMin','pMax'],
                               example='gContact.SetSearchTreeBox(pMin=[-1,-1,-1],\\\\   \\phantom   pMax=[1,1,1])',
                               description="set geometric dimensions of searchTreeBox (point with minimum coordinates and point with maximum coordinates); if this box becomes smaller than the effective contact objects, contact computations may slow down significantly"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddSphereWithMarker', cName='AddSphereWithMarker', 
                                argList=['markerIndex','radius','contactStiffness','contactDamping','frictionMaterialIndex'],
                                description="add contact object using a marker (Position or Rigid), radius and contact/friction parameters; frictionMaterialIndex refers to frictionPairings in GeneralContact; contact is possible between spheres (circles in 2D) (if intraSphereContact = True), spheres and triangles and between sphere (=circle) and ANCFCable2D; contactStiffness is computed as serial spring between contacting objects, while damping is computed as a parallel damper (otherwise the smaller damper would always dominate)!"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddANCFCable', cName='AddANCFCable', 
                                argList=['objectIndex','halfHeight','contactStiffness','contactDamping','frictionMaterialIndex'],
                                description="add contact object for an ANCF cable element, using the objectIndex of the cable element and the cable's half height as an additional distance to contacting objects (currently not causing additional torque in case of friction); currently only contact with spheres (circles in 2D) possible; contact computed using exact geometry of elements, finding max 3 intersecting contact regions"); s+=s1; sL+=sL1

[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='AddTrianglesRigidBodyBased', cName='PyAddTrianglesRigidBodyBased', 
                                argList=['rigidBodyMarkerIndex','contactStiffness','contactDamping','frictionMaterialIndex','pointList','triangleList'],
                                description="add contact object using a rigidBodyMarker (of a body), contact/friction parameters, a list of points (as 3D numpy arrays or lists; coordinates relative to rigidBodyMarker) and a list of triangles (3 indices as numpy array or list) according to a mesh attached to the rigidBodyMarker; mesh can be produced with GraphicsData2TrigsAndPoints(...); contact is possible between sphere (circle) and Triangle but yet not between triangle and triangle; frictionMaterialIndex refers to frictionPairings in GeneralContact; contactStiffness is computed as serial spring between contacting objects, while damping is computed as a parallel damper (otherwise the smaller damper would always dominate); the triangle normal must point outwards, with the normal of a triangle given with local points (p0,p1,p2) defined as n=(p1-p0) x (p2-p0), see function ComputeTriangleNormal(...)"); s+=s1; sL+=sL1
                                                      
[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='__repr__', cName='[](const PyGeneralContact &item) {\n            return EXUstd::ToString(item); }', 
                                description="return the string representation of the GeneralContact, containing basic information and statistics",
                                isLambdaFunction = True); s+=s1; sL+=sL1


#++++++++++++++++
[s1,sL1] = DefPyFinishClass('GeneralContact'); s+=s1; sL+=sL1


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#documentation and pybindings for VisuGeneralContact
classStr = 'VisuGeneralContact'
pyClassStr = 'VisuGeneralContact'
[s1,sL1] = DefPyStartClass(classStr, pyClassStr, "Data structure for visualization inside GeneralContact." +
        ' \\\\ \\\\ Usage: \\bi\n'+
        '  \\item \\texttt{gContact.visualization.drawSpheres = True} \n'+
        '\\ei\n')
s+=s1; sL+=sL1

#already included; s+= '        .def(py::init<>())\n' #empty constructor 


[s1,sL1] = DefPyFunctionAccess(cClass=classStr, pyName='Reset', cName='Reset', 
                                description="reset visualization parameters to default values"); s+=s1; sL+=sL1

# s +=  '        .def_readwrite("spheresMarkerBasedDraw", &VisuGeneralContact::spheresMarkerBasedDraw, py::return_value_policy::reference)\n' 
# sL += '  spheresMarkerBasedDraw & default = False; if True, markerBased spheres are drawn with given resolution and color \\\\ \\hline  \n'

# s +=  '        .def_readwrite("spheresMarkerBasedResolution", &VisuGeneralContact::spheresMarkerBasedResolution, py::return_value_policy::reference)\n' 
# sL += '  spheresMarkerBasedResolution & default = 4; integer value for number of triangles per circumference of markerBased spheres; higher values leading to smoother spheres but higher graphics costs \\\\ \\hline  \n'

# s +=  '        .def_readwrite("spheresMarkerBasedColor", &VisuGeneralContact::spheresMarkerBasedColor, py::return_value_policy::reference)\n' 
# sL += '  spheresMarkerBasedColor & vector with 4 floats (Float4) for color of markerBased spheres \\\\ \\hline  \n'

#++++++++++++++++
[s1,sL1] = DefPyFinishClass('GeneralContact'); s+=s1; sL+=sL1


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#now finalize files:

sL += sLenum #put latex description of enums after the systemData section

directoryString = '../Autogenerated/'
pybindFile = directoryString + 'pybind_manual_classes.h'
latexFile = '../../../docs/theDoc/manual_interfaces.tex'

file=open(pybindFile,'w')  #clear file by one write access
file.write('// AUTO:  ++++++++++++++++++++++\n')
file.write('// AUTO:  pybind11 manual module includes; generated by Johannes Gerstmayr\n')
file.write('// AUTO:  last modified = '+ GetDateStr() + '\n')
file.write('// AUTO:  ++++++++++++++++++++++\n')
file.write(s)
file.close()

file=open(latexFile,'w')  #clear file by one write access
file.write('% ++++++++++++++++++++++')
file.write('% description of manual pybind interfaces; generated by Johannes Gerstmayr')
file.write('% ++++++++++++++++++++++')
file.write(sL)
file.close()

