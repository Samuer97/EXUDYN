/** ***********************************************************************************************
* @class        VisualizationMarkerSuperElementPosition
* @brief        A position marker attached to a SuperElement, such as ObjectFFRF, ObjectGenericODE2 and ObjectFFRFreducedOrder (for which it is in its current implementation inefficient for large number of meshNodeNumbers). The marker acts on the mesh (interface) nodes, not on the underlying nodes of the object.
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
* @date         2024-02-03  15:27:08 (last modified)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#ifndef VISUALIZATIONMARKERSUPERELEMENTPOSITION__H
#define VISUALIZATIONMARKERSUPERELEMENTPOSITION__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"

class VisualizationMarkerSuperElementPosition: public VisualizationMarker // AUTO: 
{
protected: // AUTO: 
    bool showMarkerNodes;                         //!< AUTO: set true, if all nodes are shown (similar to marker, but with less intensity)

public: // AUTO: 
    //! AUTO: default constructor with parameter initialization
    VisualizationMarkerSuperElementPosition()
    {
        show = true;
        showMarkerNodes = true;
    };

    // AUTO: access functions
    //! AUTO:  Write (Reference) access to:set true, if all nodes are shown (similar to marker, but with less intensity)
    void SetShowMarkerNodes(const bool& value) { showMarkerNodes = value; }
    //! AUTO:  Read (Reference) access to:set true, if all nodes are shown (similar to marker, but with less intensity)
    const bool& GetShowMarkerNodes() const { return showMarkerNodes; }
    //! AUTO:  Read (Reference) access to:set true, if all nodes are shown (similar to marker, but with less intensity)
    bool& GetShowMarkerNodes() { return showMarkerNodes; }

    //! AUTO:  Update visualizationSystem -> graphicsData for item; index shows item Number in CData
    virtual void UpdateGraphics(const VisualizationSettings& visualizationSettings, VisualizationSystem* vSystem, Index itemNumber) override;

};



#endif //#ifdef include once...
