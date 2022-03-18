/** ***********************************************************************************************
* @class        CMarker
* @brief        A marker is the interface between forces and their application (body, node, etc.) as well as the interface between the Sensor and the sensor position; this means, forces are applied to markers and they know how to apply the force to the body or node
*
* @author       Gerstmayr Johannes
* @date         2018-05-18 (generated)
* @date         2019-05-07 (last modfied)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
				- email: johannes.gerstmayr@uibk.ac.at
				- weblink: https://github.com/jgerstmayr/EXUDYN
				
************************************************************************************************ */
#ifndef CMARKER__H
#define CMARKER__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "Main/MarkerData.h" 


class CMarker // 
{
protected: // 

public: // 

	virtual ~CMarker() {} //added for correct deletion of derived classes
	// access functions
  //! clone object; specifically for copying instances of derived class, for automatic memory management e.g. in ObjectContainer
	virtual CMarker* GetClone() const { return new CMarker(*this); }

	//! determine type of marker in order to decide according action in assembly; to be filled in derived class
	virtual Marker::Type GetType() const {
		CHECKandTHROWstring("Invalid call to CMarker::Type");
		return Marker::_None;
	}

	//! if body marker: get object number (otherwise assertion)
	virtual Index GetObjectNumber() const {
		CHECKandTHROWstring("Invalid call to CMarker::GetBodyNumber");
		return EXUstd::InvalidIndex;
	}

	//! if node marker: get node number (otherwise assertion)
	virtual Index GetNodeNumber() const {
		CHECKandTHROWstring("Invalid call to CMarker::GetNodeNumber");
		return EXUstd::InvalidIndex;
	}

	//! if marker is of coordinate type: return coordinate of node or body
	virtual Index GetCoordinateNumber() const {
		CHECKandTHROWstring("Invalid call to CMarker::GetCoordinateNumber");
		return EXUstd::InvalidIndex;
	}

	//! compute current position of marker in global frame; some position should always be available for marker (also temperatur ...)
	virtual void GetPosition(const CSystemData& cSystemData, Vector3D& position, ConfigurationType configuration = ConfigurationType::Current) const {
		CHECKandTHROWstring("Invalid call to CMarker::GetPosition");
	}

	//! compute current velocity of marker in global frame; this function could be erased from CMarker (CAUTION: derived class CMarkerPosition uses override)
	virtual void GetVelocity(const CSystemData& cSystemData, Vector3D& velocity, ConfigurationType configuration = ConfigurationType::Current) const {
		CHECKandTHROWstring("Invalid call to CMarker::GetVelocity");
	}

	//! return configuration dependent rotation matrix of node; returns always a 3D Matrix
	virtual void GetRotationMatrix(const CSystemData& cSystemData, Matrix3D& rotationMatrix, ConfigurationType configuration = ConfigurationType::Current) const {
		CHECKandTHROWstring("Invalid call to CMarker::GetRotationMatrix");
	}

	//! return configuration dependent angular velocity of node; returns always a 3D Vector
	virtual void GetAngularVelocity(const CSystemData& cSystemData, Vector3D& angularVelocity, ConfigurationType configuration = ConfigurationType::Current) const {
		CHECKandTHROWstring("Invalid call to CMarker::GetAngularVelocity");
	}

	//! return configuration dependent angular velocity of node; returns always a 3D Vector
	virtual void GetAngularVelocityLocal(const CSystemData& cSystemData, Vector3D& angularVelocity, ConfigurationType configuration = ConfigurationType::Current) const {
		CHECKandTHROWstring("Invalid call to CMarker::GetAngularVelocity");
	}

	//should be filled directly into markerdata
	////! Jacobian of position with respect to all body/node DOF
	//virtual void GetPositionJacobian(const CSystemData& cSystemData, Matrix& jacobian) const {
	//	CHECKandTHROWstring("Invalid call to CMarker::GetPositionJacobian");
	//}

	//! compute markerdata: fill in according data, which is used by force or constraint; some computations could be saved, if constraint::GetRequestedMarkerType would be used
	virtual void ComputeMarkerData(const CSystemData& cSystemData, bool computeJacobian, MarkerData& markerData) const {
		CHECKandTHROWstring("Invalid call to CMarker::ComputeMarkerData");
	}

	//! compute markerdata: fill in according data for derivative of jacobian times 6D vector v6D, e.g.: d(Jpos.T @ v6D[0:3])/dq
	virtual void ComputeMarkerDataJacobianDerivative(const CSystemData& cSystemData, const Vector6D& v6D, MarkerData& markerData) const {
		CHECKandTHROWstring("Invalid call to CMarker::ComputeMarkerDataJacobianDerivative");
	}

	//DELETE (replaced by Marker::JacobianDerivativeAvailable)
	////! true = signal that analytical jacobian derivative is available
	//virtual bool ProvidesJacobianDerivative() const { return false; } 

	//! dimension, which an according connector would have
	virtual Index GetDimension(const CSystemData& cSystemData) const {
		CHECKandTHROWstring("Invalid call to CMarker::Dimension");
		return EXUstd::InvalidIndex;
	}

	virtual void Print(std::ostream& os) const
	{
		os << "CMarker";
		os << "\n";
	}

	friend std::ostream& operator<<(std::ostream& os, const CMarker& object)
	{
		object.Print(os);
		return os;
	}

};


#endif
