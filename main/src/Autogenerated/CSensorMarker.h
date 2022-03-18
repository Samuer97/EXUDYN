/** ***********************************************************************************************
* @class        CSensorMarkerParameters
* @brief        Parameter class for CSensorMarker
*
* @author       Gerstmayr Johannes
* @date         2019-07-01 (generated)
* @date         2022-02-19  00:16:29 (last modified)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#ifndef CSENSORMARKERPARAMETERS__H
#define CSENSORMARKERPARAMETERS__H

#include <ostream>

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h"
#include "System/ItemIndices.h"


//! AUTO: Parameters for class CSensorMarkerParameters
class CSensorMarkerParameters // AUTO: 
{
public: // AUTO: 
    Index markerNumber;                           //!< AUTO: marker number to which sensor is attached to
    bool writeToFile;                             //!< AUTO: True: write sensor output to file; flag is ignored (interpreted as False), if fileName=''
    std::string fileName;                         //!< AUTO: directory and file name for sensor file output; default: empty string generates sensor + sensorNumber + outputVariableType; directory will be created if it does not exist
    OutputVariableType outputVariableType;        //!< AUTO: OutputVariableType for sensor; output variables are only possible according to markertype, see general description of SensorMarker
    bool storeInternal;                           //!< AUTO: true: store sensor data in memory (faster, but may consume large amounts of memory); false: internal storage not available
    //! AUTO: default constructor with parameter initialization
    CSensorMarkerParameters()
    {
        markerNumber = EXUstd::InvalidIndex;
        writeToFile = true;
        fileName = "";
        outputVariableType = OutputVariableType::_None;
        storeInternal = false;
    };
};


/** ***********************************************************************************************
* @class        CSensorMarker
* @brief        A sensor attached to a marker. The sensor measures the selected marker values and outputs values into a file, showing per line [time, sensorValue[0], sensorValue[1], ...]. Depending on markers, it can measure Coordinates (MarkerNodeCoordinate), Position and Velocity (MarkerXXXPosition), Position, Velocity, Rotation and AngularVelocityLocal (MarkerXXXRigid). Note that marker values are only available for the current configuration. Use SensorUserFunction to modify sensor results (e.g., transforming to other coordinates) and writing to file
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

//! AUTO: CSensorMarker
class CSensorMarker: public CSensor // AUTO: 
{
protected: // AUTO: 
    CSensorMarkerParameters parameters; //! AUTO: contains all parameters for CSensorMarker

public: // AUTO: 

    // AUTO: access functions
    //! AUTO: Write (Reference) access to parameters
    virtual CSensorMarkerParameters& GetParameters() { return parameters; }
    //! AUTO: Read access to parameters
    virtual const CSensorMarkerParameters& GetParameters() const { return parameters; }

    //! AUTO:  general access to marker number
    virtual Index GetMarkerNumber() const override
    {
        return parameters.markerNumber;
    }

    //! AUTO:  return sensor type
    virtual SensorType GetType() const override
    {
        return SensorType::Marker;
    }

    //! AUTO:  get writeToFile flag
    virtual bool GetWriteToFileFlag() const override
    {
        return parameters.writeToFile;
    }

    //! AUTO:  get storeInternal flag
    virtual bool GetStoreInternalFlag() const override
    {
        return parameters.storeInternal;
    }

    //! AUTO:  get file name
    virtual STDstring GetFileName() const override
    {
        return parameters.fileName;
    }

    //! AUTO:  get OutputVariableType
    virtual OutputVariableType GetOutputVariableType() const override
    {
        return parameters.outputVariableType;
    }

    //! AUTO:  main function to generate sensor output values
    virtual void GetSensorValues(const CSystemData& cSystemData, Vector& values, ConfigurationType configuration = ConfigurationType::Current) const override;

};



#endif //#ifdef include once...
