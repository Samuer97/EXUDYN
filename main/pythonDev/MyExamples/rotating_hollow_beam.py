
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Simulation of a rotating hollow beam with prescribed motion,
#           based on user specifications.
#
# Author:   GitHub Copilot (Generated)
# Date:     2025-08-12
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
import exudyn.graphics as graphics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. System and Environment Setup
# =============================================================================
SC = exu.SystemContainer()
mbs = SC.AddSystem()

# Graphics for visualization
background = graphics.CheckerBoard(point=[0,0,-0.5], size=20)
oGround = mbs.AddObject(ObjectGround(referencePosition=[0,0,0], visualization=VObjectGround(graphicsData=[background])))

# =============================================================================
# 2. Physical and Simulation Parameters
# =============================================================================
# Material and Geometry
L = 8.0                   # Length of the beam in m
A = 0.7299e-4             # Cross-sectional area in m^2
I = 0.8215e-8             # Second moment of area in m^4
rho = 2766.67             # Density in kg/m^3
E = 68.95e9               # Young's modulus in Pa (GPa to Pa)
# nu = 0.3                # Poisson's ratio (not used in 2D beam)

# Motion Parameters
Ts = 15.0                 # Transition time in s
omega_s = 100.0             # Steady state angular velocity in rad/s

# Simulation Settings
tEnd = 20.0               # Total simulation time in s
h = 1e-3                  # Time step in s
numberOfElements = 10     # Number of ANCF elements for the beam

# =============================================================================
# 3. Prescribed Motion User Function
# =============================================================================
def UF_prescribed_rotation(mbs, t, itemNumber, lOffset):
    """
    User function to define the prescribed rotation angle theta(t) for the hub.
    """
    if t <= Ts:
        term1 = t**2 / 2
        term2 = (Ts / (2 * np.pi))**2 * (np.cos(2 * np.pi * t / Ts) - 1)
        return (omega_s / Ts) * (term1 + term2)
    else:
        return omega_s * (t - Ts / 2)

# =============================================================================
# 4. Multibody System Modeling
# =============================================================================
# Create a template for the ANCF beam elements
beam_template = Cable2D(
    physicsMassPerLength=rho * A,
    physicsBendingStiffness=E * I,
    physicsAxialStiffness=E * A,
    physicsBendingDamping=0.001 * E * I, # Add small damping for stability
    visualization=VCable2D(drawHeight=0.1, color=graphics.color.blue)
)

# Generate the straight ANCF beam along the x-axis
positionOfNode0 = [0, 0, 0]
positionOfNode1 = [L, 0, 0]
ancf_beam = GenerateStraightLineANCFCable2D(
    mbs,
    positionOfNode0,
    positionOfNode1,
    numberOfElements,
    beam_template
)

ancfNodes = ancf_beam[0]
firstNode = ancfNodes[0]
lastNode = ancfNodes[-1]

# Create a rigid hub
hub_size = 0.2
gBody = graphics.Cylinder(axis=[0,0,1], radius=hub_size, length=hub_size, color=graphics.color.red)
hub_dict = mbs.CreateRigidBody(
    referencePosition=[0,0,0],
    inertia=InertiaCylinder(1000, hub_size, hub_size, 2),
    graphicsDataList=[gBody],
    create2D=True,
    returnDict=True
)
hub_body_number = hub_dict['bodyNumber']
hub_node_number = hub_dict['nodeNumber']

# Connect the beam's first node to the hub (fixed connection)
mBeamFirst = mbs.AddMarker(MarkerNodeRigid(nodeNumber=firstNode))
mHub = mbs.AddMarker(MarkerBodyRigid(bodyNumber=hub_body_number, localPosition=[0,0,0]))
mbs.AddObject(GenericJoint(
    markerNumbers=[mBeamFirst, mHub],
    constrainedAxes=[1,1,0, 0,0,1], # Weld joint
    visualization=VGenericJoint(axesRadius=0.05, axesLength=0.2)
))

# Connect the hub to the ground with a revolute joint
mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0,0,0]))
mbs.AddObject(RevoluteJoint2D(
    markerNumbers=[mHub, mGround],
    visualization=VRevoluteJoint2D(drawSize=hub_size)
))

# Apply the prescribed rotation to the hub
nGroundRef = mbs.AddNode(NodePointGround(referenceCoordinates=[0,0,0]))
mcGroundRef = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGroundRef, coordinate=0))
mHubRotation = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=hub_node_number, coordinate=2)) # coordinate 2 is rotation for 2D rigid body

mbs.AddObject(CoordinateConstraint(
    markerNumbers=[mcGroundRef, mHubRotation],
    offset=0.0,
    offsetUserFunction=UF_prescribed_rotation
))

# Add sensor to track the free end position
sensorTip = mbs.AddSensor(SensorNode(nodeNumber=lastNode, 
                                     outputVariableType=exu.OutputVariableType.Position,
                                     storeInternal=True))

# =============================================================================
# 5. Solve the System
# =============================================================================
mbs.Assemble()

simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.timeIntegration.numberOfSteps = int(tEnd / h)
simulationSettings.timeIntegration.newton.relativeTolerance = 1e-5
simulationSettings.timeIntegration.newton.absoluteTolerance = 1e-7
simulationSettings.timeIntegration.verboseMode = 1 # Show progress
simulationSettings.displayComputationTime = True
simulationSettings.pauseAfterEachStep = False  # Prevent pausing during simulation
simulationSettings.solutionSettings.writeSolutionToFile = True # Enable sensor data collection
simulationSettings.solutionSettings.solutionWritePeriod = h * 10  # Write every 10 time steps

# Use a robust sparse solver
simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse

print("Starting dynamic simulation...")
exu.SolveDynamic(mbs, simulationSettings)
print("Simulation finished.")

# =============================================================================
# 6. Post-processing and Plotting
# =============================================================================
print("Post-processing results...")

# Retrieve simulation data from sensor using correct method
sensorData = mbs.GetSensorStoredData(sensorTip)
time_points = sensorData[:, 0]  # First column is time
tip_positions = sensorData[:, 1:4]  # Columns 1-3 are x, y, z positions

# Extract x, y, z coordinates
tip_x = tip_positions[:, 0]
tip_y = tip_positions[:, 1] 
tip_z = tip_positions[:, 2]  # Should be mostly zero for 2D simulation

print(f"Data shape: time_points {time_points.shape}, tip_positions {tip_positions.shape}")
print(f"Time range: {time_points[0]:.3f} to {time_points[-1]:.3f} seconds")
print(f"Position range: X=[{tip_x.min():.3f}, {tip_x.max():.3f}], Y=[{tip_y.min():.3f}, {tip_y.max():.3f}]")

# Calculate deformations (relative to rigid body motion)
axial_deformation = []
transverse_deformation = []

for i, t in enumerate(time_points):
    # Get actual position of the beam tip
    actual_pos = tip_positions[i, 0:2]  # x, y coordinates

    # Get the current angle of the hub
    theta = UF_prescribed_rotation(mbs, t, 0, 0)

    # Calculate the ideal rigid position of the tip
    ideal_pos = np.array([L * np.cos(theta), L * np.sin(theta)])

    # Calculate the total displacement vector
    displacement_vec = actual_pos - ideal_pos

    # Define axial and transverse direction vectors
    axial_unit_vec = np.array([np.cos(theta), np.sin(theta)])
    transverse_unit_vec = np.array([-np.sin(theta), np.cos(theta)])

    # Project displacement vector onto local axes
    axial_def = np.dot(displacement_vec, axial_unit_vec)
    transverse_def = np.dot(displacement_vec, transverse_unit_vec)

    axial_deformation.append(axial_def)
    transverse_deformation.append(transverse_def)

axial_deformation = np.array(axial_deformation)
transverse_deformation = np.array(transverse_deformation)

# Create comprehensive plots
plt.style.use('default')  # Use default style for better compatibility

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Absolute position in space (X-Y plot)
ax1 = plt.subplot(2, 3, 1)
plt.plot(tip_x, tip_y, 'b-', linewidth=2, label='Tip trajectory')
plt.plot(tip_x[0], tip_y[0], 'go', markersize=8, label='Start')
plt.plot(tip_x[-1], tip_y[-1], 'ro', markersize=8, label='End')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Beam Tip Trajectory in Space')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')

# 2. X position vs time
ax2 = plt.subplot(2, 3, 2)
plt.plot(time_points, tip_x, 'r-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('X position (m)')
plt.title('X Position vs Time')
plt.grid(True, alpha=0.3)

# 3. Y position vs time
ax3 = plt.subplot(2, 3, 3)
plt.plot(time_points, tip_y, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Y position (m)')
plt.title('Y Position vs Time')
plt.grid(True, alpha=0.3)

# 4. Axial deformation vs time
ax4 = plt.subplot(2, 3, 4)
plt.plot(time_points, axial_deformation, 'r-', linewidth=2, label='Axial Deformation')
plt.xlabel('Time (s)')
plt.ylabel('Axial Deformation (m)')
plt.title('Axial Deformation vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

# 5. Transverse deformation vs time
ax5 = plt.subplot(2, 3, 5)
plt.plot(time_points, transverse_deformation, 'b-', linewidth=2, label='Transverse Deformation')
plt.xlabel('Time (s)')
plt.ylabel('Transverse Deformation (m)')
plt.title('Transverse Deformation vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

# 6. 3D trajectory plot
ax6 = plt.subplot(2, 3, 6, projection='3d')
ax6.plot(tip_x, tip_y, time_points, 'b-', linewidth=2, label='3D trajectory')
ax6.scatter(tip_x[0], tip_y[0], time_points[0], color='green', s=50, label='Start')
ax6.scatter(tip_x[-1], tip_y[-1], time_points[-1], color='red', s=50, label='End')
ax6.set_xlabel('X position (m)')
ax6.set_ylabel('Y position (m)')
ax6.set_zlabel('Time (s)')
ax6.set_title('3D Space-Time Trajectory')
ax6.legend()

plt.tight_layout()
plt.show()

# Additional 3D visualization - trajectory in space with time color coding
fig2 = plt.figure(figsize=(12, 10))

# Create 3D plot for spatial trajectory with time color mapping
ax_3d = fig2.add_subplot(111, projection='3d')

# Create a color map based on time
colors = plt.cm.viridis(time_points / time_points[-1])

# Plot trajectory with color coding
for i in range(len(time_points)-1):
    ax_3d.plot([tip_x[i], tip_x[i+1]], [tip_y[i], tip_y[i+1]], [0, 0], 
               color=colors[i], linewidth=2)

# Add start and end markers
ax_3d.scatter(tip_x[0], tip_y[0], 0, color='green', s=100, label='Start')
ax_3d.scatter(tip_x[-1], tip_y[-1], 0, color='red', s=100, label='End')

# Add hub position
ax_3d.scatter(0, 0, 0, color='black', s=150, marker='s', label='Hub')

ax_3d.set_xlabel('X position (m)')
ax_3d.set_ylabel('Y position (m)')
ax_3d.set_zlabel('Z position (m)')
ax_3d.set_title('3D Beam Tip Trajectory (Color = Time)')
ax_3d.legend()

# Add color bar
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=time_points[-1]))
mappable.set_array(time_points)
plt.colorbar(mappable, ax=ax_3d, label='Time (s)', shrink=0.8)

plt.show()

print("Analysis Summary:")
print(f"Maximum axial deformation: {np.max(np.abs(axial_deformation)):.6f} m")
print(f"Maximum transverse deformation: {np.max(np.abs(transverse_deformation)):.6f} m")
print(f"Final tip position: ({tip_x[-1]:.3f}, {tip_y[-1]:.3f}) m")
print(f"Total tip displacement: {np.sqrt(tip_x[-1]**2 + tip_y[-1]**2):.3f} m")

# =============================================================================
# 7. Enhanced 3D Visualization
# =============================================================================
print("Starting enhanced 3D visualization...")

# Configure visualization settings for better 3D display
SC.visualizationSettings.nodes.defaultSize = 0.02
SC.visualizationSettings.bodies.defaultSize = [0.05, 0.05, 0.05]
SC.visualizationSettings.connectors.defaultSize = 0.02
SC.visualizationSettings.loads.defaultSize = 0.1

# Set up better camera and rendering
SC.visualizationSettings.openGL.multiSampling = 4
SC.visualizationSettings.general.renderWindowString = "Rotating Hollow Beam Simulation"
SC.visualizationSettings.general.autoFitScene = False

# Set initial camera position for better view
SC.visualizationSettings.general.graphicsUpdateInterval = 0.02  # Update every 20ms for smooth animation

# Configure animation settings
SC.visualizationSettings.general.useMultiThreadedRendering = True

# Add coordinate system display
SC.visualizationSettings.general.showSolutionInformation = True

print("Starting EXUDYN 3D viewer...")
print("Controls:")
print("  - Mouse: Rotate view")
print("  - Mouse wheel: Zoom")
print("  - Space: Play/Pause animation")
print("  - Left/Right arrows: Step through time")
print("  - R: Reset view")

mbs.SolutionViewer()
