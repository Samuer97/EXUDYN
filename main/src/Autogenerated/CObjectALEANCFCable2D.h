/** ***********************************************************************************************
* @class        CObjectALEANCFCable2DParameters
* @brief        Parameter class for CObjectALEANCFCable2D
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
* @date         2022-11-17  15:43:14 (last modified)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#ifndef COBJECTALEANCFCABLE2DPARAMETERS__H
#define COBJECTALEANCFCABLE2DPARAMETERS__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"

#include "Objects/CObjectANCFCable2DBase.h"

//! AUTO: Parameters for class CObjectALEANCFCable2DParameters
class CObjectALEANCFCable2DParameters // AUTO: 
{
public: // AUTO: 
    Real physicsLength;                           //!< AUTO:  [SI:m] reference length of beam; such that the total volume (e.g. for volume load) gives \f$\rho A L\f$; must be positive
    Real physicsMassPerLength;                    //!< AUTO:  [SI:kg/m] total mass per length of beam (including axially moving parts / fluid)
    Real physicsMovingMassFactor;                 //!< AUTO: this factor denotes the amount of \f$\rho A\f$ which is moving; physicsMovingMassFactor=1 means, that all mass is moving; physicsMovingMassFactor=0 means, that no mass is moving; factor can be used to simulate e.g. pipe conveying fluid, in which \f$\rho A\f$ is the mass of the pipe+fluid, while \f$physicsMovingMassFactor \cdot \rho A\f$ is the mass per unit length of the fluid
    Real physicsBendingStiffness;                 //!< AUTO:  [SI:Nm\f$^2\f$] bending stiffness of beam; the bending moment is \f$m = EI (\kappa - \kappa_0)\f$, in which \f$\kappa\f$ is the material measure of curvature
    Real physicsAxialStiffness;                   //!< AUTO:  [SI:N] axial stiffness of beam; the axial force is \f$f_{ax} = EA (\varepsilon -\varepsilon_0)\f$, in which \f$\varepsilon = |\rv^\prime|-1\f$ is the axial strain
    Real physicsBendingDamping;                   //!< AUTO:  [SI:Nm\f$^2\f$/s] bending damping of beam ; the additional virtual work due to damping is \f$\delta W_{\dot \kappa} = \int_0^L \dot \kappa \delta \kappa dx\f$
    Real physicsAxialDamping;                     //!< AUTO:  [SI:N/s] axial damping of beam; the additional virtual work due to damping is \f$\delta W_{\dot\varepsilon} = \int_0^L \dot \varepsilon \delta \varepsilon dx\f$
    Real physicsReferenceAxialStrain;             //!< AUTO:  [SI:1] reference axial strain of beam (pre-deformation) of beam; without external loading the beam will statically keep the reference axial strain value
    Real physicsReferenceCurvature;               //!< AUTO:  [SI:1/m] reference curvature of beam (pre-deformation) of beam; without external loading the beam will statically keep the reference curvature value
    bool physicsUseCouplingTerms;                 //!< AUTO: true: correct case, where all coupling terms due to moving mass are respected; false: only include constant mass for ALE node coordinate, but deactivate other coupling terms (behaves like ANCFCable2D then)
    bool physicsAddALEvariation;                  //!< AUTO: true: correct case, where additional terms related to variation of strain and curvature are added
    Index3 nodeNumbers;                           //!< AUTO: two node numbers ANCF cable element, third node=ALE GenericODE2 node
    Index useReducedOrderIntegration;             //!< AUTO: 0/false: use Gauss order 9 integration for virtual work of axial forces, order 5 for virtual work of bending moments; 1/true: use Gauss order 7 integration for virtual work of axial forces, order 3 for virtual work of bending moments
    Real strainIsRelativeToReference;             //!< AUTO:  if set to 1., a pre-deformed reference configuration is considered as the stressless state; if set to 0., the straight configuration plus the values of \f$\varepsilon_0\f$ and \f$\kappa_0\f$ serve as a reference geometry; allows also values between 0. and 1.
    //! AUTO: default constructor with parameter initialization
    CObjectALEANCFCable2DParameters()
    {
        physicsLength = 0.;
        physicsMassPerLength = 0.;
        physicsMovingMassFactor = 1.;
        physicsBendingStiffness = 0.;
        physicsAxialStiffness = 0.;
        physicsBendingDamping = 0.;
        physicsAxialDamping = 0.;
        physicsReferenceAxialStrain = 0.;
        physicsReferenceCurvature = 0.;
        physicsUseCouplingTerms = true;
        physicsAddALEvariation = true;
        nodeNumbers = Index3({EXUstd::InvalidIndex, EXUstd::InvalidIndex, EXUstd::InvalidIndex});
        useReducedOrderIntegration = 0;
        strainIsRelativeToReference = 0.;
    };
};


/** ***********************************************************************************************
* @class        CObjectALEANCFCable2D
* @brief        A 2D cable finite element using 2 nodes of type NodePoint2DSlope1 and a axially moving coordinate of type NodeGenericODE2, which adds additional (redundant) motion in axial direction of the beam. This allows modeling pipes but also axially moving beams. The localPosition of the beam with length \f$L\f$=physicsLength and height \f$h\f$ ranges in \f$X\f$-direction in range \f$[0, L]\f$ and in \f$Y\f$-direction in range \f$[-h/2,h/2]\f$ (which is in fact not needed in the \hac{EOM}).
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

//! AUTO: CObjectALEANCFCable2D
class CObjectALEANCFCable2D: public CObjectANCFCable2DBase // AUTO: 
{
protected: // AUTO: 
    mutable bool massTermsALEComputed; //!< flag which shows that ALE mass terms have been computed; will be set to false at time when parameters are set
    mutable ConstSizeMatrix<nODE2coordinates*nODE2coordinates> preComputedM1, preComputedM2, preComputedB1, preComputedB2; //!< if massTermsALEComputed=true, this contains the constant mass terms for faster computation
    CObjectALEANCFCable2DParameters parameters; //! AUTO: contains all parameters for CObjectALEANCFCable2D

public: // AUTO: 

    // AUTO: access functions
    //! AUTO: Write (Reference) access to parameters
    virtual CObjectALEANCFCable2DParameters& GetParameters() { return parameters; }
    //! AUTO: Read access to parameters
    virtual const CObjectALEANCFCable2DParameters& GetParameters() const { return parameters; }

    //! AUTO:  access to individual element paramters for base class functions
    virtual Real GetLength() const override
    {
        return parameters.physicsLength;
    }

    //! AUTO:  access to individual element paramters for base class functions
    virtual Real GetMassPerLength() const override
    {
        return parameters.physicsMassPerLength;
    }

    //! AUTO:  access to individual element paramters for base class functions
    virtual void GetMaterialParameters(Real& physicsBendingStiffness, Real& physicsAxialStiffness, Real& physicsBendingDamping, Real& physicsAxialDamping, Real& physicsReferenceAxialStrain, Real& physicsReferenceCurvature, Real& physicsMovingMassFactor) const override
    {
        physicsBendingStiffness = parameters.physicsBendingStiffness; physicsAxialStiffness = parameters.physicsAxialStiffness; physicsBendingDamping = parameters.physicsBendingDamping; physicsAxialDamping = parameters.physicsAxialDamping; physicsReferenceAxialStrain = parameters.physicsReferenceAxialStrain; physicsReferenceCurvature = parameters.physicsReferenceCurvature; physicsMovingMassFactor = parameters.physicsMovingMassFactor;
    }

    //! AUTO:  access to useReducedOrderIntegration from derived class
    virtual Index UseReducedOrderIntegration() const override
    {
        return parameters.useReducedOrderIntegration;
    }

    //! AUTO:  access to strainIsRelativeToReference from derived class
    virtual Real StrainIsRelativeToReference() const override
    {
        return parameters.strainIsRelativeToReference;
    }

    //! AUTO:  access to physicsAddALEvariation
    virtual bool AddALEvariation() const override
    {
        return parameters.physicsAddALEvariation;
    }

    //! AUTO:  Computational function: compute mass matrix
    virtual void ComputeMassMatrix(EXUmath::MatrixContainer& massMatrixC, const ArrayIndex& ltg, Index objectNumber) const override;

    //! AUTO:  Computational function: compute left-hand-side (LHS) of second order ordinary differential equations (ODE) to 'ode2Lhs'
    virtual void ComputeODE2LHS(Vector& ode2Lhs, Index objectNumber) const override;

    //! AUTO:  return the available jacobian dependencies and the jacobians which are available as a function; if jacobian dependencies exist but are not available as a function, it is computed numerically; can be combined with 2^i enum flags
    virtual JacobianType::Type GetAvailableJacobians() const override
    {
        return (JacobianType::Type)(JacobianType::ODE2_ODE2 + JacobianType::ODE2_ODE2_t);
    }

    //! AUTO:  provide Jacobian at localPosition in 'value' according to object access
    virtual void GetAccessFunctionBody(AccessFunctionType accessType, const Vector3D& localPosition, Matrix& value) const override;

    //! AUTO:  return the (global) velocity of 'localPosition' according to configuration type
    virtual Vector3D GetVelocity(const Vector3D& localPosition, ConfigurationType configuration = ConfigurationType::Current) const override;

    //! AUTO:  Get global node number (with local node index); needed for every object ==> does local mapping
    virtual Index GetNodeNumber(Index localIndex) const override
    {
        CHECKandTHROW(localIndex <= 2, __EXUDYN_invalid_local_node2);
        return parameters.nodeNumbers[localIndex];
    }

    //! AUTO:  number of nodes; needed for every object
    virtual Index GetNumberOfNodes() const override
    {
        return 3;
    }

    //! AUTO:  number of \hac{ODE2} coordinates; needed for object?
    virtual Index GetODE2Size() const override
    {
        return nODE2coordinates+1;
    }

    //! AUTO:  return true if object has time and coordinate independent (=constant) mass matrix
    virtual bool HasConstantMassMatrix() const override
    {
        return false;
    }

    //! AUTO:  This flag is reset upon change of parameters; says that mass matrix (future: other pre-computed values) need to be recomputed
    virtual void ParametersHaveChanged() override
    {
        massTermsALEComputed = false; massMatrixComputed = false;
    }

    //! AUTO:  precompute mass terms if it has not been done yet
    virtual void PreComputeMassTerms() const override;

    virtual OutputVariableType GetOutputVariableTypes() const override
    {
        return (OutputVariableType)(
            (Index)OutputVariableType::Position +
            (Index)OutputVariableType::Displacement +
            (Index)OutputVariableType::Velocity +
            (Index)OutputVariableType::VelocityLocal +
            (Index)OutputVariableType::Rotation +
            (Index)OutputVariableType::Director1 +
            (Index)OutputVariableType::StrainLocal +
            (Index)OutputVariableType::CurvatureLocal +
            (Index)OutputVariableType::ForceLocal +
            (Index)OutputVariableType::TorqueLocal );
    }

};



#endif //#ifdef include once...
