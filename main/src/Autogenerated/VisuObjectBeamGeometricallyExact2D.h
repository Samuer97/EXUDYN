/** ***********************************************************************************************
* @class        VisualizationObjectBeamGeometricallyExact2D
* @brief        A 2D geometrically exact beam finite element, currently using 2 nodes of type NodeRigidBody2D; FURTHER TESTS REQUIRED. Note that the orientation of the nodes need to follow the cross section orientation in case that includeReferenceRotations=True; e.g., an angle 0 represents the cross section aligned with the \f$y\f$-axis, while and angle \f$\pi/2\f$ means that the cross section points in negative \f$x\f$-direction. Pre-curvature can be included with physicsReferenceCurvature and axial pre-stress can be considered by using a physicsLength different from the reference configuration of the nodes. The localPosition of the beam with length \f$L\f$=physicsLength and height \f$h\f$ ranges in \f$X\f$-direction in range \f$[-L/2, L/2]\f$ and in \f$Y\f$-direction in range \f$[-h/2,h/2]\f$ (which is in fact not needed in the \hac{EOM}).
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
* @date         2024-02-03  15:27:06 (last modified)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#ifndef VISUALIZATIONOBJECTBEAMGEOMETRICALLYEXACT2D__H
#define VISUALIZATIONOBJECTBEAMGEOMETRICALLYEXACT2D__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"

class VisualizationObjectBeamGeometricallyExact2D: public VisualizationObject // AUTO: 
{
protected: // AUTO: 
    float drawHeight;                             //!< AUTO: if beam is drawn with rectangular shape, this is the drawing height
    Float4 color;                                 //!< AUTO: RGBA color of the object; if R==-1, use default color

public: // AUTO: 
    //! AUTO: default constructor with parameter initialization
    VisualizationObjectBeamGeometricallyExact2D()
    {
        show = true;
        drawHeight = 0.f;
        color = Float4({-1.f,-1.f,-1.f,-1.f});
    };

    // AUTO: access functions
    //! AUTO:  Update visualizationSystem -> graphicsData for item; index shows item Number in CData
    virtual void UpdateGraphics(const VisualizationSettings& visualizationSettings, VisualizationSystem* vSystem, Index itemNumber) override;

    //! AUTO:  Write (Reference) access to:if beam is drawn with rectangular shape, this is the drawing height
    void SetDrawHeight(const float& value) { drawHeight = value; }
    //! AUTO:  Read (Reference) access to:if beam is drawn with rectangular shape, this is the drawing height
    const float& GetDrawHeight() const { return drawHeight; }
    //! AUTO:  Read (Reference) access to:if beam is drawn with rectangular shape, this is the drawing height
    float& GetDrawHeight() { return drawHeight; }

    //! AUTO:  Write (Reference) access to:RGBA color of the object; if R==-1, use default color
    void SetColor(const Float4& value) { color = value; }
    //! AUTO:  Read (Reference) access to:RGBA color of the object; if R==-1, use default color
    const Float4& GetColor() const { return color; }
    //! AUTO:  Read (Reference) access to:RGBA color of the object; if R==-1, use default color
    Float4& GetColor() { return color; }

};



#endif //#ifdef include once...
