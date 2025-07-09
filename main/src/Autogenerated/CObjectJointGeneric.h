/** ***********************************************************************************************
* @class        CObjectJointGenericParameters
* @brief        Parameter class for CObjectJointGeneric
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
* @date         2025-05-08  11:59:28 (last modified)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#ifndef COBJECTJOINTGENERICPARAMETERS__H
#define COBJECTJOINTGENERICPARAMETERS__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"

#include <functional> //! AUTO: needed for std::function
#include "Pymodules/PythonUserFunctions.h" //! AUTO: needed for user functions, without pybind11
class MainSystem; //AUTO; for std::function / userFunction; avoid including MainSystem.h

//! AUTO: Parameters for class CObjectJointGenericParameters
class CObjectJointGenericParameters // AUTO: 
{
public: // AUTO: 
    ArrayIndex markerNumbers;                     //!< AUTO: list of markers used in connector
    ArrayIndex constrainedAxes;                   //!< AUTO: flag, which determines which translation (0,1,2) and rotation (3,4,5) axes are constrained; for \f$j_i\f$, two values are possible: 0=free axis, 1=constrained axis
    Matrix3D rotationMarker0;                     //!< AUTO: local rotation matrix for marker \f$m0\f$; translation and rotation axes for marker \f$m0\f$ are defined in the local body coordinate system and additionally transformed by rotationMarker0
    Matrix3D rotationMarker1;                     //!< AUTO: local rotation matrix for marker \f$m1\f$; translation and rotation axes for marker \f$m1\f$ are defined in the local body coordinate system and additionally transformed by rotationMarker1
    bool activeConnector;                         //!< AUTO: flag, which determines, if the connector is active; used to deactivate (temporarily) a connector or constraint
    Vector6D offsetUserFunctionParameters;        //!< AUTO: vector of 6 parameters for joint's offsetUserFunction
    PythonUserFunctionBase< std::function<StdVector6D(const MainSystem&,Real,Index,StdVector6D)> > offsetUserFunction;//!< AUTO: A Python function which defines the time-dependent (fixed) offset of translation (indices 0,1,2) and rotation (indices 3,4,5) joint coordinates with parameters (mbs, t, offsetUserFunctionParameters)
    PythonUserFunctionBase< std::function<StdVector6D(const MainSystem&,Real,Index,StdVector6D)> > offsetUserFunction_t;//!< AUTO: (NOT IMPLEMENTED YET)time derivative of offsetUserFunction using the same parameters
    bool alternativeConstraints;                  //!< AUTO: this is an experimental flag, may change in future: if uses alternative contraint equations for rotations, currently in case of 3 locked rotations: \f$\LU{0}{\tv}_{x0}\tp (\LU{0}{\tv}_{y1} \times \LU{0}{\tv}_{z0})\f$, \f$\LU{0}{\tv}_{y0}\tp (\LU{0}{\tv}_{z1} \times \LU{0}{\tv}_{x0})\f$, \f$\LU{0}{\tv}_{z0}\tp (\LU{0}{\tv}_{x1} \times \LU{0}{\tv}_{y0})\f$; this avoids 180\textdegree flips of the standard configuration in static computations, but leads to different values in Lagrange multipliers
    //! AUTO: default constructor with parameter initialization
    CObjectJointGenericParameters()
    {
        markerNumbers = ArrayIndex({ EXUstd::InvalidIndex, EXUstd::InvalidIndex });
        constrainedAxes = ArrayIndex({1,1,1,1,1,1});
        rotationMarker0 = EXUmath::unitMatrix3D;
        rotationMarker1 = EXUmath::unitMatrix3D;
        activeConnector = true;
        offsetUserFunctionParameters = Vector6D({0.,0.,0.,0.,0.,0.});
        offsetUserFunction = 0;
        offsetUserFunction_t = 0;
        alternativeConstraints = false;
    };
};


/** ***********************************************************************************************
* @class        CObjectJointGeneric
* @brief        A generic joint in 3D; constrains components of the absolute position and rotations of two points given by PointMarkers or RigidMarkers. An additional local rotation (rotationMarker) can be used to adjust the three rotation axes and/or sliding axes.
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"

//! AUTO: CObjectJointGeneric
class CObjectJointGeneric: public CObjectConstraint // AUTO: 
{
protected: // AUTO: 
    static constexpr Index nConstraints = 6;
    CObjectJointGenericParameters parameters; //! AUTO: contains all parameters for CObjectJointGeneric

public: // AUTO: 

    // AUTO: access functions
    //! AUTO: Write (Reference) access to parameters
    virtual CObjectJointGenericParameters& GetParameters() { return parameters; }
    //! AUTO: Read access to parameters
    virtual const CObjectJointGenericParameters& GetParameters() const { return parameters; }

    //! AUTO:  return true, if object has a computation user function
    virtual bool HasUserFunction() const override
    {
        return (parameters.offsetUserFunction!=0) || (parameters.offsetUserFunction_t!=0);
    }

    //! AUTO:  default (read) function to return Marker numbers
    virtual const ArrayIndex& GetMarkerNumbers() const override
    {
        return parameters.markerNumbers;
    }

    //! AUTO:  default (write) function to return Marker numbers
    virtual ArrayIndex& GetMarkerNumbers() override
    {
        return parameters.markerNumbers;
    }

    //! AUTO:  constraints uses Lagrance multiplier formulation
    virtual bool IsPenaltyConnector() const override
    {
        return false;
    }

    //! AUTO:  Computational function: compute algebraic equations and write residual into 'algebraicEquations'; velocityLevel: equation provided at velocity level
    virtual void ComputeAlgebraicEquations(Vector& algebraicEquations, const MarkerDataStructure& markerData, Real t, Index itemIndex, bool velocityLevel = false) const override;

    //! AUTO:  compute derivative of algebraic equations w.r.t. \hac{ODE2}, \hac{ODE2t}, \hac{ODE1} and \hac{AE} coordinates in jacobian [flags ODE2_t_AE_function, AE_AE_function, etc. need to be set in GetAvailableJacobians()]; jacobianODE2[_t] has dimension GetAlgebraicEquationsSize() x GetODE2Size() ; q are the system coordinates; markerData provides according marker information to compute jacobians
    virtual void ComputeJacobianAE(ResizableMatrix& jacobian_ODE2, ResizableMatrix& jacobian_ODE2_t, ResizableMatrix& jacobian_ODE1, ResizableMatrix& jacobian_AE, const MarkerDataStructure& markerData, Real t, Index itemIndex) const override;

    //! AUTO:  return the available jacobian dependencies and the jacobians which are available as a function; if jacobian dependencies exist but are not available as a function, it is computed numerically; can be combined with 2^i enum flags; available jacobians is switched depending on velocity level and on activeConnector condition
    virtual JacobianType::Type GetAvailableJacobians() const override;

    //! AUTO:  provide according output variable in 'value'
    virtual void GetOutputVariableConnector(OutputVariableType variableType, const MarkerDataStructure& markerData, Index itemIndex, Vector& value) const override;

    //! AUTO:  provide requested markerType for connector
    virtual Marker::Type GetRequestedMarkerType() const override
    {
        return (Marker::Type)((Index)Marker::Position + (Index)Marker::Orientation);
    }

    //! AUTO:  return object type (for node treatment in computation)
    virtual CObjectType GetType() const override
    {
        return (CObjectType)((Index)CObjectType::Connector + (Index)CObjectType::Constraint);
    }

    //! AUTO:  number of algebraic equations; independent of node/body coordinates
    virtual Index GetAlgebraicEquationsSize() const override
    {
        return 6;
    }

    //! AUTO:  return if connector is active-->speeds up computation
    virtual bool IsActive() const override
    {
        return parameters.activeConnector;
    }

    //! AUTO:  call to user function implemented in separate file to avoid including pybind and MainSystem.h at too many places
    void EvaluateUserFunctionOffset(Vector6D& offset, const MainSystemBase& mainSystem, Real t, Index itemIndex) const;

    //! AUTO:  call to user function implemented in separate file to avoid including pybind and MainSystem.h at too many places
    void EvaluateUserFunctionOffset_t(Vector6D& offset, const MainSystemBase& mainSystem, Real t, Index itemIndex) const;

    virtual OutputVariableType GetOutputVariableTypes() const override
    {
        return (OutputVariableType)(
            (Index)OutputVariableType::Position +
            (Index)OutputVariableType::Velocity +
            (Index)OutputVariableType::DisplacementLocal +
            (Index)OutputVariableType::VelocityLocal +
            (Index)OutputVariableType::Rotation +
            (Index)OutputVariableType::AngularVelocityLocal +
            (Index)OutputVariableType::ForceLocal +
            (Index)OutputVariableType::TorqueLocal );
    }

};



#endif //#ifdef include once...
