/** ***********************************************************************************************
* @class        VisualizationBasics
* @brief		
* @details		Details:
 				- helper classes mainly for colors
*
* @author		Gerstmayr Johannes
* @date			2020-05-13 (generated)
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
#ifndef VISUALIZATIONBASICS__H
#define VISUALIZATIONBASICS__H


namespace EXUvis {
	//default colors used in some elements, etc.
	const Float4 defaultColorFloat4 = { 0.f,0.f,0.f,1.f }; //default color black, if color is not defined
	const Float4 defaultColorBlue4 = { 0.4f,0.4f,0.9f,1.f }; //default color blue, if color is not defined for triangles

	//colors in exudyn
	const Float4 red = { 0.9f,0.1f,0.1f,1.f };
	const Float4 green = { 0.1f,0.9f,0.1f,1.f };
	const Float4 blue = { 0.1f,0.1f,0.9f,1.f };
	const Float4 cyan = { 0.f,1.f,1.f,1.f };
	const Float4 magenta = { 1.f,0.f,1.f,1.f };
	const Float4 yellow = { 1.f,1.f,0.f,1.f };
	const Float4 black = { 0.01f,0.01f,0.01f,1.f };
	const Float4 white = { 1.f,1.f,1.f,1.f };
	const Float4 grey1 = { 0.3f,0.3f,0.3f,1.f };
	const Float4 grey2 = { 0.55f,0.55f,0.55f,1.f };
	const Float4 grey3 = { 0.8f,0.8f,0.8f,1.f };
	const Float4 orange = { 1.f,0.5f,0.25f,1.f };
	const Float4 lila = { 0.5f,0.5f,1.f,1.f };
	const Float4 rose = { 1.f,0.5f,0.5f,1.f };
	const Float4 brown = { 0.5f,0.25f,0.25f,1.f };

	//! get color from index; used e.g. for axes numeration
	inline const Float4& GetColor(Index i);

	const float modifyColorFactor = 0.25f; //!< standard value to modify color with ModifyColor
	//! modify a color by a factor: 1=original color, 0=grey; this can be used to visualize axes, joints, ... using one color definition represented in two colors
	inline Float4 ModifyColor(const Float4& color, float colorFactor = 1.f)
	{
		if (colorFactor == 1.f) { return color; }
		else
		{
			Float4 grey({ 0.5f,0.5f,0.5f,color[3] });
			return Float4((1.f - colorFactor)*grey + colorFactor * color);
		}
	}
    //! get alternated color (e.g. for items, to have different text from item color)
    inline Float4 AltColor(const Float4& color)
    {
        float avColor2 = (0.5f / 3.f)*(color[0] + color[1] + color[2]);
        if (fabs(color[0] - color[1]) < 0.3f && fabs(color[0] - color[2]) < 0.3f)
        {
            //a rather grey color should be altered in brightness
            if (avColor2 < 0.6)
            { //make brighter
                return Float4({
                    EXUstd::Minimum(0.9f, color[0] + 0.4f),
                    EXUstd::Minimum(0.9f, color[1] + 0.4f),
                    EXUstd::Minimum(0.9f, color[2] + 0.4f),
                    color[3] });
            }
            else
            { //make darker, but not black
                return Float4({
                    EXUstd::Maximum(0.1f, color[0] - 0.3f),
                    EXUstd::Maximum(0.1f, color[1] - 0.3f),
                    EXUstd::Maximum(0.1f, color[2] - 0.3f),
                    color[3] });
            }
        }
        return Float4({ 
            0.5f*color[0] + avColor2,
            0.5f*color[1] + avColor2,
            0.5f*color[2] + avColor2, color[3] });
    }

} //namespace EXUvis

#endif
