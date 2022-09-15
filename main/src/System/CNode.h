/** ***********************************************************************************************
* @class	    CNode
* @brief		Class for computational nodes, which define coordinates
* @details		Details:
				- nodes define coordinates of computational objects (CObjects)
				- nodes can be of one category: ODE1coordinates, ODE2coordinates, AEvariables, DataVariables
*
* @author		Gerstmayr Johannes
* @date			2018-05-17 (generated)
* @pre			...
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
* @note			Bug reports, support and further information:
* 				- email: johannes.gerstmayr@uibk.ac.at
* 				- weblink: https://github.com/jgerstmayr/EXUDYN
* 				
*
* *** Example code ***
*
************************************************************************************************ */
#ifndef CNODE__H
#define CNODE__H

#include "Utilities/ReleaseAssert.h"
#include <initializer_list>
#include "Utilities/BasicDefinitions.h" //defines Real
#include "Utilities/ResizableArray.h" 
#include "Linalg/LinkedDataVector.h"	//includes Vector.h and SlimVector.h

//#include "System/CSystem.h" 

class CData;

////! nodetype is used to know, which quantities can be measured (position, rotation) and which actions are possible (force, moment)
//enum class CNodeType {
//    _None, Point, RigidBody, Temperature, General
//};

//namespace instead of class enum CNodeType ==> allows to write logics easier (e.g. adding Node::Type::Position + Node::Type::Orientation)
//! node types are used for integrity checks to verify that a node is suitable for an object
namespace Node {
	enum Type {
		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//keep these lists synchronized with PybindModule.cpp lists
		_None = 0, //marks that no type is used
		Ground = 1 << 0,					//!< used for ground nodes

		//2D
		Position2D = 1 << 1,					//!< used for: 2D point nodes, rigid nodes, nodes with position and slopes, ...; must provide position + translational displacement, velocity
		Orientation2D = 1 << 2,					//!< used for: 2D rigid nodes (independent of parameterization); node must provide rotation matrix, dAngularVelocity/dq
		Point2DSlope1 = 1 << 3,					//!< used for: nodes which provide a 2D position and a 2D slope vector in 1-direction
		//3D
		Position = 1 << 4,						//!< used for: point nodes, rigid nodes, nodes with position and slopes, ...; must provide position + translational displacement, velocity
		Orientation = 1 << 5,					//!< nodes, which can measure rotation, can apply torque (not only rigid nodes); node must provide rotation matrix, dAngularVelocity/dq
		RigidBody = 1 << 6,						//!< nodes which are derived from NodeRigidBody; used for ObjectRigidBody or for beams; node must provide rotation matrix, dAngularVelocity/dq, and G, Glocal, ...
		//special Rotation:
		RotationEulerParameters = 1 << 7,		//!< used if orientation is described with euler parameters
		RotationRxyz = 1 << 8,					//!< used if orientation is described with euler angles
		RotationRotationVector = 1 << 9,		//!< used if orientation is described with rotation vector parameters
		LieGroupWithDirectUpdate = 1 << 10,		//!< used if a lie group formulation is used; this means, that equations are written vor angular acc (omega_t), not for rotationParameters_tt
		LieGroupWithDataCoordinates = 1 << 11,	//!< used if a lie group formulation is used with data variables for start-of-step state and local frame approach
		//General
		GenericODE2 = 1 << 12,					//!< used for node with ODE2 coordinates (no specific access functions, except on coordinate level)
		GenericODE1 = 1 << 13,					//!< used for node with ODE1 coordinates (no specific access functions, except on coordinate level)
		GenericAE = 1 << 14,					//!< (CURRENTLY UNUSED!) used for node with AE coordinates (no specific access functions, except on coordinate level)
		GenericData = 1 << 15,					//!< used for node with data coordinates
		//ANCF:
		Point3DSlope1 = 1 << 16,				//!< used for: nodes which provide a position and a slope vector in 1-direction
		Point3DSlope23 = 1 << 17				//!< used for: nodes which provide a position and slope vectors in 2 and 3-direction
		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//keep these lists synchronized with autoGeneratePybindings.py lists
	};
	//! transform type into string (e.g. for error messages); this is slow and cannot be used during computation!
	inline STDstring GetTypeString(Type var)
	{
		STDstring t; //empty string
		if (var == Node::_None) { t = "_None/Undefined"; }
		if (var & Ground) { t += "Ground"; }

		if (var & Position2D) { t += "Position2D"; }
		if (var & Orientation2D) { t += "Orientation2D"; }
		if (var & Point2DSlope1) { t += "Point2DSlope1"; }

		if (var & Position) { t += "Position"; }
		if (var & Orientation) { t += "Orientation"; }
		if (var & RigidBody) { t += "RigidBody"; }

		if (var & RotationEulerParameters) { t += "RotationEulerParameters"; }
		if (var & RotationRxyz) { t += "RotationRxyz"; }
		if (var & RotationRotationVector) { t += "RotationRotationVector"; }
		if (var & LieGroupWithDirectUpdate) { t += "LieGroupWithDirectUpdate"; }
		if (var & LieGroupWithDataCoordinates) { t += "LieGroupWithDataCoordinates"; }

		if (var & GenericODE2) { t += "GenericODE2"; }
		if (var & GenericData) { t += "GenericData"; }
		if (t.length() == 0) { CHECKandTHROWstring("Node::GetTypeString(...) called for invalid type!"); }

		return t;
	}
};

// if nodes should have several groups, use namespace enum and 2^i values
enum class CNodeGroup {
    _None=0, ODE1variables=1, ODE2variables=2, AEvariables=4, DataVariables=8
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//keep these lists synchronized with PybindModule.cpp lists
};

// allow only ODE1 or ODE2 or AE coordinates 
//open Question: where to put CData* (into node or object, or none of them?)

class CNode
{
protected:

    CData* computationalData; //! this is the only thing which the node needs to know to perform evaluation -> absolutely needed?
public:
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // CONSTRUCTOR, DESTRUCTOR, INITIALIZATION
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	CNode();
	virtual ~CNode() {} //added for correct deletion of derived classes
	virtual CNode* GetClone() const { return new CNode(*this); }

    virtual void Print(std::ostream& os) const {
        os << "CNode";
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // FUNCTIONS
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
	CData* GetCData() const;
	CData*& GetCData();

    virtual void SetGlobalODE2CoordinateIndex(Index globalIndex) { CHECKandTHROWstring("CNode::SetGlobalODE2CoordinateIndex(): call illegal"); }
    virtual void SetGlobalODE1CoordinateIndex(Index globalIndex) { CHECKandTHROWstring("CNode::SetGlobalODE1CoordinateIndex(): call illegal"); }
	virtual void SetGlobalAECoordinateIndex(Index globalIndex) { CHECKandTHROWstring("CNode::SetGlobalAECoordinateIndex(): call illegal"); }
	virtual void SetGlobalDataCoordinateIndex(Index globalIndex) { CHECKandTHROWstring("CNode::SetGlobalDataCoordinateIndex(): call illegal"); }

    //! number of state variables; includes position AND velocity coordinates
	virtual Index GetNumberOfStateCoordinates() const final { return 2 * GetNumberOfODE2Coordinates() + GetNumberOfODE1Coordinates() + GetNumberOfAECoordinates(); }

	//! number of coordinates which lead to unknowns and which are accessible (e.g. by generalized force)
	virtual Index GetNumberOfAccessibleCoordinates() const final { return GetNumberOfODE2Coordinates() + GetNumberOfODE1Coordinates() + GetNumberOfAECoordinates(); }

	//! workaround for ground nodes (could be changed to a separate flag in derived nodes in future ...)
	virtual bool IsGroundNode() const final { return GetNumberOfAccessibleCoordinates() == 0; }

	//! read single coordinate in current configuration
	virtual const Real& GetCurrentCoordinate(Index i) const {CHECKandTHROWstring("CNode::GetCurrentCoordinate"); return computationalData->GetCurrent().GetTime(); /*dummy variable time...*/}

	//! read globally stored initial coordinates
	virtual LinkedDataVector GetInitialCoordinateVector() const {CHECKandTHROWstring("CNode::GetInitialCoordinateVector"); return LinkedDataVector(); }

    //! read current nodal coordinates out of global coordinate vector
	virtual LinkedDataVector GetCurrentCoordinateVector() const {CHECKandTHROWstring("CNode::GetCurrentCoordinateVector"); return LinkedDataVector(); }

	//! read configuration dependent nodal coordinates from global coordinate vector
	virtual LinkedDataVector GetCoordinateVector(ConfigurationType configuration) const {CHECKandTHROWstring("CNode::GetCoordinateVector"); return LinkedDataVector(); }

	//!read (internally stored) reference coordinate vector (implemented in specialized class)
	virtual LinkedDataVector GetReferenceCoordinateVector() const { CHECKandTHROWstring("CNode::GetReferenceCoordinateVector: call illegal");  return LinkedDataVector(); }

	//!read (internally stored) reference coordinate vector (implemented in specialized class)
	virtual LinkedDataVector GetStartOfStepCoordinateVector() const { CHECKandTHROWstring("CNode::GetStartOfStepCoordinateVector: call illegal");  return LinkedDataVector(); }

	//! read visualization coordinates
	virtual LinkedDataVector GetVisualizationCoordinateVector() const { CHECKandTHROWstring("CNode::GetReferenceCoordinateVector: call illegal");  return LinkedDataVector(); }

	virtual Index GetNumberOfODE1Coordinates() const { return 0; }
    virtual Index GetNumberOfODE2Coordinates() const { return 0; }
	virtual Index GetNumberOfAECoordinates() const { return 0; }
	virtual Index GetNumberOfDataCoordinates() const { return 0; }

	virtual Index GetAlgebraicEquationsSize() const { return 0; } //only relevant for Euler Parameter nodes ...

    virtual Index GetGlobalODE2CoordinateIndex() const {
        CHECKandTHROWstring("CNode::GetGlobalODE2CoordinateIndex"); return 0; }
    virtual Index GetGlobalODE1CoordinateIndex() const { 
        CHECKandTHROWstring("CNode::GetGlobalODE1CoordinateIndex"); return 0; }
    virtual Index GetGlobalAECoordinateIndex() const { 
        CHECKandTHROWstring("CNode::GetGlobalAECoordinateIndex"); return 0; }
	virtual Index GetGlobalDataCoordinateIndex() const {
		CHECKandTHROWstring("CNode::GetGlobalDataCoordinateIndex"); return 0;	}

	virtual OutputVariableType GetOutputVariableTypes() const { CHECKandTHROWstring("ERROR: illegal call to CNode::GetOutputVariableTypes"); return OutputVariableType::_None; }
	virtual void GetOutputVariable(OutputVariableType variableType, ConfigurationType configuration, Vector& value) const { CHECKandTHROWstring("ERROR: illegal call to CNode::GetOutputVariable"); }


    virtual CNodeGroup GetNodeGroup() const { CHECKandTHROWstring("CNode::GetNodeGroup"); return CNodeGroup::_None; }
    virtual Node::Type GetType() const { CHECKandTHROWstring("CNode::GetType"); return Node::_None; }

    friend std::ostream& operator<<(std::ostream& os, const CNode& object) {
        object.Print(os);
        return os;
    }

};

//! node with ODE2 variables: for mass points, finite elements, etc.
class CNodeODE2: public CNode
{
protected:
    Index globalODE2CoordinateIndex;                //!< refers to the place in the global ODE2 coordinate vector, either position or velocity level (must be the same!)
public:
    CNodeODE2() 
    {
        globalODE2CoordinateIndex = EXUstd::InvalidIndex; //mark that globalODE2CoordinateIndex cannot be accessed
    }
    //! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
    virtual CNodeODE2* GetClone() const { return new CNodeODE2(*this); }

    virtual void Print(std::ostream& os) const {
        os << "CNodeODE2(ODE2Index=" << globalODE2CoordinateIndex << ", size=" << GetNumberOfODE2Coordinates() << "):";
        CNode::Print(os);
    }

    virtual CNodeGroup GetNodeGroup() const { return CNodeGroup::ODE2variables; }
    virtual void SetGlobalODE2CoordinateIndex(Index globalIndex) { globalODE2CoordinateIndex = globalIndex; }

    virtual Index GetGlobalODE2CoordinateIndex() const {
        return globalODE2CoordinateIndex;
    }

	//! read single coordinate in current configuration
	virtual const Real& GetCurrentCoordinate(Index i) const override;

	//! read single velocity coordinate in current configuration
	virtual const Real& GetCurrentCoordinate_t(Index i) const;

	//! read single acceleration coordinate in current configuration
	virtual const Real& GetCurrentCoordinate_tt(Index i) const;

	//! read globally stored current coordinates (displacements)
	inline virtual LinkedDataVector GetCurrentCoordinateVector() const override
	{
		return LinkedDataVector(computationalData->currentState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
	}

	//! get current + reference coordinates written into a TVector (assumed to be e.g. a ConstVector
	template<class TVector>
	void GetCurrentAndReferenceODE2CoordinateVector(TVector& vector) const
	{
		vector.CopyFrom(GetReferenceCoordinateVector());
		vector += LinkedDataVector(computationalData->currentState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
	}

	//! read globally stored current coordinates (velocities)
	inline virtual LinkedDataVector GetCurrentCoordinateVector_t() const
	{
		return LinkedDataVector(computationalData->currentState.ODE2Coords_t, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
	}

	//! read globally stored current coordinates (accelerations)
	inline virtual LinkedDataVector GetCurrentCoordinateVector_tt() const
	{
		return LinkedDataVector(computationalData->currentState.ODE2Coords_tt, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
	}

	//! read globally stored initial coordinates (displacements)
	virtual LinkedDataVector GetInitialCoordinateVector() const override;

	//! read globally stored initial coordinates (velocities)
	virtual LinkedDataVector GetInitialCoordinateVector_t() const;

	//! read globally stored initial coordinates (accelerations)
	virtual LinkedDataVector GetInitialCoordinateVector_tt() const;

	//! read globally stored start of step coordinates (displacements)
	virtual LinkedDataVector GetStartOfStepCoordinateVector() const override;

	//! read globally stored start of step coordinates (velocities)
	virtual LinkedDataVector GetStartOfStepCoordinateVector_t() const;

	//! read globally stored start of step coordinates (accelerations)
	virtual LinkedDataVector GetStartOfStepCoordinateVector_tt() const;

	//! read visualization coordinates (displacements)
	virtual LinkedDataVector GetVisualizationCoordinateVector() const override;

	//! read visualization coordinates (velocities)
	virtual LinkedDataVector GetVisualizationCoordinateVector_t() const;

	//! read visualization coordinates (accelerations)
	virtual LinkedDataVector GetVisualizationCoordinateVector_tt() const;

	template<class TVector>
	void GetODE2CoordinateVectorWithReference(TVector& vector, ConfigurationType configuration = ConfigurationType::Current) const
	{
		vector.CopyFrom(GetReferenceCoordinateVector());
		
		switch (configuration)
		{
		case ConfigurationType::Current:
			vector += LinkedDataVector(computationalData->currentState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
			break;
		case ConfigurationType::Initial: 
			vector += LinkedDataVector(computationalData->initialState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
			break;
		case ConfigurationType::Reference:
			break;
		case ConfigurationType::StartOfStep:
			vector += LinkedDataVector(computationalData->startOfStepState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
			break;
		case ConfigurationType::Visualization:
			vector += LinkedDataVector(computationalData->visualizationState.ODE2Coords, globalODE2CoordinateIndex, GetNumberOfODE2Coordinates());
			break;
		default: CHECKandTHROWstring("CNodeODE2::GetCoordinateVectorWithReference: invalid ConfigurationType");
		}
	}



	virtual LinkedDataVector GetCoordinateVector(ConfigurationType configuration) const override
    {
        switch (configuration)
        {
            case ConfigurationType::Current: return GetCurrentCoordinateVector();
            case ConfigurationType::Initial: return GetInitialCoordinateVector();
			case ConfigurationType::Reference: return GetReferenceCoordinateVector();
			case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector();
			case ConfigurationType::Visualization:
			{
				return GetVisualizationCoordinateVector();
			}
			default: CHECKandTHROWstring("CNodeODE2::GetCoordinateVector: invalid ConfigurationType"); return LinkedDataVector();
        }
    }

	virtual LinkedDataVector GetCoordinateVector_t(ConfigurationType configuration) const
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector_t();
		case ConfigurationType::Initial: return GetInitialCoordinateVector_t();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector_t();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector_t();
		default: CHECKandTHROWstring("CNodeODE2::GetCoordinateVector_t: invalid ConfigurationType"); return LinkedDataVector();
		}
	}

	virtual LinkedDataVector GetCoordinateVector_tt(ConfigurationType configuration) const
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector_tt();
		case ConfigurationType::Initial: return GetInitialCoordinateVector_tt();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector_tt();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector_tt();

		default: CHECKandTHROWstring("CNodeODE2::GetCoordinateVector_tt: invalid ConfigurationType"); return LinkedDataVector();
		}
	}

	//! return configuration-dependent position (for drawing, marker)
	virtual Vector3D GetPosition(ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeODE2::GetPosition: call illegal"); return Vector3D(); }
	
	//! return configuration-dependent velocity (for marker)
	virtual Vector3D GetVelocity(ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeODE2::GetVelocity: call illegal"); return Vector3D(); }

	//! return configuration-dependent velocity (for marker)
	virtual Vector3D GetAcceleration(ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeODE2::GetAcceleration: call illegal"); return Vector3D(); }

	//! return configuration dependent position of node; returns always a 3D Vector
	virtual Matrix3D GetRotationMatrix(ConfigurationType configuration = ConfigurationType::Current) const 
		{ CHECKandTHROWstring("CNodeODE2::GetRotationMatrix: call illegal"); return Matrix3D(); }

	//! return configuration dependent angular velocity of node; returns always a 3D Vector
	virtual Vector3D GetAngularVelocity(ConfigurationType configuration = ConfigurationType::Current) const
	{
		CHECKandTHROWstring("CNodeODE2::GetAngularVelocity: call illegal"); return Vector3D();
	}

	//! return configuration dependent angular acceleration of node; returns always a 3D Vector
	virtual Vector3D GetAngularAcceleration(ConfigurationType configuration = ConfigurationType::Current) const
	{
		CHECKandTHROWstring("CNodeODE2::GetAngularAcceleration: call illegal"); return Vector3D();
	}

	//! return configuration dependent local (body fixed) angular velocity of node; returns always a 3D Vector
	virtual Vector3D GetAngularVelocityLocal(ConfigurationType configuration = ConfigurationType::Current) const
	{
		CHECKandTHROWstring("CNodeODE2::GetAngularVelocityLocal: call illegal"); return Vector3D();
	}

	//! provide position Jacobian in matrix value (for marker) for current configuration
	virtual void GetPositionJacobian(Matrix& value) const { CHECKandTHROWstring("CNodeODE2::GetPositionJacobian: call illegal"); }

	//! provide rotation Jacobian in matrix value (for marker) for current configuration
	virtual void GetRotationJacobian(Matrix& value) const { CHECKandTHROWstring("CNodeODE2::GetRotationJacobian: call illegal"); }

	//! provide derivative w.r.t. coordinates of rotation Jacobian times vector; for current configuration
	virtual void GetRotationJacobianTTimesVector_q(const Vector3D& vector, Matrix& jacobian_q) const { CHECKandTHROWstring("CNodeODE2::GetJacobianTimesVector_q: call illegal"); }

	//! ONLY for nodes with AE / Euler parameters: compute algebraic equations to 'algebraicEquations', which has dimension GetNumberOfAECoordinates();
	virtual void ComputeAlgebraicEquations(Vector& algebraicEquations, bool useIndex2 = false) const { CHECKandTHROWstring("ERROR: illegal call to CNodeODE2::ComputeAlgebraicEquations"); }

	//! ONLY for nodes with AE / Euler parameters: compute algebraic equations to 'algebraicEquations', which has dimension GetNumberOfAECoordinates();
	virtual void ComputeJacobianAE(ResizableMatrix& jacobian_ODE2, ResizableMatrix& jacobian_ODE2_t, ResizableMatrix& jacobian_ODE1, ResizableMatrix& jacobian_AE,
		JacobianType::Type& filledJacobians) const { CHECKandTHROWstring("ERROR: illegal call to CNodeODE2::ComputeJacobianAE"); }
};

//! node with ODE1 variables: for first order differential equations, e.g., linear state space systems
class CNodeODE1 : public CNode
{
protected:
	Index globalODE1CoordinateIndex; //!< refers to the place in the global ODE1 coordinate vector
public:
	CNodeODE1()
	{
		globalODE1CoordinateIndex = EXUstd::InvalidIndex; //mark that globalODE1CoordinateIndex cannot be accessed
	}
	//! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
	virtual CNodeODE1* GetClone() const { return new CNodeODE1(*this); }

	virtual void Print(std::ostream& os) const {
		os << "CNodeODE1(ODE1Index=" << globalODE1CoordinateIndex << ", size=" << GetNumberOfODE1Coordinates() << "):";
		CNode::Print(os);
	}

	virtual CNodeGroup GetNodeGroup() const { return CNodeGroup::ODE1variables; }
	virtual void SetGlobalODE1CoordinateIndex(Index globalIndex) { globalODE1CoordinateIndex = globalIndex; }

	virtual Index GetGlobalODE1CoordinateIndex() const {
		return globalODE1CoordinateIndex;
	}

	//! read single coordinate in current configuration
	virtual const Real& GetCurrentCoordinate(Index i) const override;

	//! read single velocity coordinate in current configuration
	virtual const Real& GetCurrentCoordinate_t(Index i) const;

	//! read globally stored current coordinates (displacements)
	virtual LinkedDataVector GetCurrentCoordinateVector() const override;

	//! read globally stored current coordinates (velocities)
	virtual LinkedDataVector GetCurrentCoordinateVector_t() const;

	//! read globally stored initial coordinates (displacements)
	virtual LinkedDataVector GetInitialCoordinateVector() const override;

	//! read globally stored initial coordinates (velocities)
	virtual LinkedDataVector GetInitialCoordinateVector_t() const;

	//! read globally stored start of step coordinates (displacements)
	virtual LinkedDataVector GetStartOfStepCoordinateVector() const override;

	//! read globally stored start of step coordinates (velocities)
	virtual LinkedDataVector GetStartOfStepCoordinateVector_t() const;

	//! read visualization coordinates (displacements)
	virtual LinkedDataVector GetVisualizationCoordinateVector() const override;

	//! read visualization coordinates (velocities)
	virtual LinkedDataVector GetVisualizationCoordinateVector_t() const;

	virtual LinkedDataVector GetCoordinateVector(ConfigurationType configuration) const override
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector();
		case ConfigurationType::Initial: return GetInitialCoordinateVector();
		case ConfigurationType::Reference: return GetReferenceCoordinateVector();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector();
		default: CHECKandTHROWstring("CNodeODE1::GetCoordinateVector: invalid ConfigurationType"); return LinkedDataVector();
		}
	}

	virtual LinkedDataVector GetCoordinateVector_t(ConfigurationType configuration) const
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector_t();
		case ConfigurationType::Initial: return GetInitialCoordinateVector_t();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector_t();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector_t();
		default: CHECKandTHROWstring("CNodeODE1::GetCoordinateVector_t: invalid ConfigurationType"); return LinkedDataVector();
		}
	}
};


//! node with AE variables: for purely algebraic equations
class CNodeAE : public CNode
{
protected:
	Index globalAEcoordinateIndex; //!< refers to the place in the global AE coordinate vector
public:
	CNodeAE()
	{
		globalAEcoordinateIndex = EXUstd::InvalidIndex; //mark that globalAEcoordinateIndex cannot be accessed
	}
	//! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
	virtual CNodeAE* GetClone() const { return new CNodeAE(*this); }

	virtual void Print(std::ostream& os) const {
		os << "CNodeAE(AEIndex=" << globalAEcoordinateIndex << ", size=" << GetNumberOfAECoordinates() << "):";
		CNode::Print(os);
	}

	virtual CNodeGroup GetNodeGroup() const { return CNodeGroup::AEvariables; }
	virtual void SetGlobalAECoordinateIndex(Index globalIndex) { globalAEcoordinateIndex = globalIndex; }

	virtual Index GetGlobalAECoordinateIndex() const {
		return globalAEcoordinateIndex;
	}

	//! read single coordinate in current configuration
	virtual const Real& GetCurrentCoordinate(Index i) const override;

	//! read globally stored current coordinates (displacements)
	virtual LinkedDataVector GetCurrentCoordinateVector() const override;

	//! read globally stored initial coordinates (displacements)
	virtual LinkedDataVector GetInitialCoordinateVector() const override;

	//! read globally stored start of step coordinates (displacements)
	virtual LinkedDataVector GetStartOfStepCoordinateVector() const override;

	//! read visualization coordinates (displacements)
	virtual LinkedDataVector GetVisualizationCoordinateVector() const override;

	virtual LinkedDataVector GetCoordinateVector(ConfigurationType configuration) const override
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector();
		case ConfigurationType::Initial: return GetInitialCoordinateVector();
		case ConfigurationType::Reference: return GetReferenceCoordinateVector();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector();
		default: CHECKandTHROWstring("CNodeAE::GetCoordinateVector: invalid ConfigurationType"); return LinkedDataVector();
		}
	}

};


////! node with mixed ODE2 and algebraic equations coordinates
//class CNodeODE2AE : public CNodeODE2
//{
//protected:
//	Index globalAEcoordinateIndex;                //!< refers to the place in the global ODE2 coordinate vector, either position or velocity level (must be the same!)
//public:
//	CNodeODE2AE(): CNodeODE2()
//	{
//		globalAEcoordinateIndex = EXUstd::InvalidIndex;
//	}
//	//! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
//	virtual CNodeODE2AE* GetClone() const { return new CNodeODE2AE(*this); }
//
//	virtual void Print(std::ostream& os) const {
//		os << "CNodeODE2AE(ODE2Index=" << globalODE2CoordinateIndex << ", size=" << GetNumberOfODE2Coordinates() << ", ";
//		os << "AEIndex=" << globalAEcoordinateIndex << ", size=" << GetNumberOfAECoordinates() << "):";
//		CNode::Print(os);
//	}
//
//	virtual CNodeGroup GetNodeGroup() const { return (CNodeGroup)((Index)CNodeGroup::ODE2variables + (Index)CNodeGroup::AEvariables); }
//	virtual void SetGlobalAECoordinateIndex(Index globalIndex) override { globalAEcoordinateIndex = globalIndex; }
//
//	virtual Index GetGlobalAECoordinateIndex() const override { return globalAEcoordinateIndex; }
//};

//! rigid body node for rigid bodies, beams, etc.
class CNodeRigidBody : public CNodeODE2
{
public:
	static const Index maxRotationCoordinates = 4; //this is used to define the return value of ConstSizeVectors for rotation coordinates
	static const Index maxDisplacementCoordinates = 3; //this is used to define the return value of ConstSizeVectors for displacement coordinates
	static const Index nDim3D = 3; //dimensionality of 3D body (in order to avoid pure number "3" in implementation)
	//CNodeRigidBody(): CNodeODE2()
	//{
	//}

	//! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
	virtual CNodeRigidBody* GetClone() const { return new CNodeRigidBody(*this); }

	virtual void Print(std::ostream& os) const {
		os << "CNodeRigidBody(ODE2Index=" << globalODE2CoordinateIndex << ", size=" << GetNumberOfODE2Coordinates() << ", ";
		CNode::Print(os);
	}

	virtual Index GetNumberOfRotationCoordinates() const { CHECKandTHROWstring("CNodeRigidBody::GetNumberOfRotationCoordinates(): invalid call");  return 0; }
	virtual Index GetNumberOfDisplacementCoordinates() const { CHECKandTHROWstring("CNodeRigidBody::GetNumberOfDisplacementCoordinates(): invalid call");  return 0; }

	//! return current rotation parameters (= current+reference coordinates!)
	virtual ConstSizeVector<maxRotationCoordinates> GetRotationParameters(ConfigurationType configuration = ConfigurationType::Current) const
	{ 
		CHECKandTHROWstring("CNodeRigidBody::GetRotationParameters(...): invalid call");  
		return ConstSizeVector<maxRotationCoordinates>(); 
	}

	//! return current rotation parameters_t (=time derivative)
	virtual LinkedDataVector GetRotationParameters_t(ConfigurationType configuration = ConfigurationType::Current) const
	{ 
		CHECKandTHROWstring("CNodeRigidBody::GetRotationParameters_t(...): invalid call");  
		return LinkedDataVector();
	}

	virtual void GetG(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetG(...): invalid call"); }
	virtual void GetGlocal(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetGlocal(...): invalid call"); }
	virtual void GetG_t(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetG_t(...): invalid call"); }
	virtual void GetGlocal_t(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetGlocal_t(...): invalid call"); }

	//! compute d(G^T*v)/dq for any set of parameters; needed for jacobians
	virtual void GetGTv_q(const Vector3D& v, ConstSizeMatrix<maxRotationCoordinates * maxRotationCoordinates>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetGTv_q(...): invalid call"); }

	//! compute d(Glocal^T*v)/dq for any set of parameters; needed for jacobians
	virtual void GetGlocalTv_q(const Vector3D& v, ConstSizeMatrix<maxRotationCoordinates * maxRotationCoordinates>& matrix, ConfigurationType configuration = ConfigurationType::Current) const { CHECKandTHROWstring("CNodeRigidBody::GetGlocalTv_q(...): invalid call"); }

	//! obtain Glocal and local angular velocity
	virtual void CollectCurrentNodeData1(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& Glocal, Vector3D& angularVelocityLocal) const { CHECKandTHROWstring("CNodeRigidBody::CollectCurrentNodeData1(...): invalid call"); }

	//! obtain G matrices, position, velocity, rotation matrix A (local to global), local angular velocity 
	virtual void CollectCurrentNodeMarkerData(ConstSizeMatrix<maxRotationCoordinates * nDim3D>& Glocal, ConstSizeMatrix<maxRotationCoordinates * nDim3D>& G,
		Vector3D& pos, Vector3D& vel, Matrix3D& A, Vector3D& angularVelocityLocal) const { CHECKandTHROWstring("CNodeRigidBody::CollectCurrentNodeData1(...): invalid call"); }

	//! apply "SE3" composition rule for current position and orientation coordinates (depending on node type) with incremental motion, compute new coordinates
	virtual void CompositionRule(const LinkedDataVector& currentPosition, const LinkedDataVector& currentOrientation, 
		const Vector6D& incrementalMotion,
		LinkedDataVector& newPosition, LinkedDataVector& newOrientation) const { CHECKandTHROWstring("CNodeRigidBody::CompositionRule(...): invalid call");	};
};

//! node with data variables
class CNodeData : public CNode
{
protected:
	Index globalDataCoordinateIndex;                //!< refers to the place in the global data coordinate vector, either position or velocity level (must be the same!)
public:
	CNodeData()
	{
		globalDataCoordinateIndex = EXUstd::InvalidIndex; //mark that globalDataCoordinateIndex cannot be accessed
	}
	//! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
	virtual CNodeData* GetClone() const { return new CNodeData(*this); }

	virtual void Print(std::ostream& os) const {
		os << "CNodeData(DataIndex=" << globalDataCoordinateIndex << ", size=" << GetNumberOfDataCoordinates() << "):";
		CNode::Print(os);
	}

	virtual CNodeGroup GetNodeGroup() const { return CNodeGroup::DataVariables; }
	virtual void SetGlobalDataCoordinateIndex(Index globalIndex) { globalDataCoordinateIndex = globalIndex; }

	virtual Index GetGlobalDataCoordinateIndex() const {
		return globalDataCoordinateIndex;
	}

	//! read single coordinate in current configuration
	virtual const Real& GetCurrentCoordinate(Index i) const override;

	//! read globally stored current coordinates (displacements)
	virtual LinkedDataVector GetCurrentCoordinateVector() const override;

	//! read globally stored initial coordinates (displacements)
	virtual LinkedDataVector GetInitialCoordinateVector() const override;

	//! read globally stored start of step coordinates (displacements)
	virtual LinkedDataVector GetStartOfStepCoordinateVector() const override;

	//! read visualization coordinates (displacements)
	virtual LinkedDataVector GetVisualizationCoordinateVector() const override;

	virtual LinkedDataVector GetCoordinateVector(ConfigurationType configuration) const override
	{
		switch (configuration)
		{
		case ConfigurationType::Current: return GetCurrentCoordinateVector();
		case ConfigurationType::Initial: return GetInitialCoordinateVector();
		case ConfigurationType::StartOfStep: return GetStartOfStepCoordinateVector();
		case ConfigurationType::Visualization: return GetVisualizationCoordinateVector();
		default: CHECKandTHROWstring("CODE2Node::GetCoordinateVector: invalid ConfigurationType (Reference not possible)!"); return LinkedDataVector();
		}
	}

};


//class CAEvariablesNode: public CNode
//{
//protected:
//    Index globalCoordinateIndex; //start of global coordinates index
//    Index numberOfAEvariables;   //the number of coordinates = Lagrange multipliers is variable in this node type
//public:
//    //! get an exact clone of *this, must be implemented in all derived classes! Necessary for better handling in ObjectContainer
//    virtual CAEvariablesNode* GetClone() const { return new CAEvariablesNode(*this); }
//    virtual void Print(std::ostream& os) const {
//        os << "CAEvariablesNode(AEIndex=" << globalCoordinateIndex << ", size=" << numberOfAEvariables << "):";
//        CNode::Print(os);
//    }
//
//    Index GetNumberOfAEvariables() const { return numberOfAEvariables; }
//    CNodeGroup GetNodeGroup() const { return CNodeGroup::AEvariables; }
//
//
//};

#endif
