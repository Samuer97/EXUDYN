/** ***********************************************************************************************
* @class        VisualizationPrimitives implementation
* @brief		
*
* @author		Gerstmayr Johannes
* @date			2020-02-08 (generated)
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

#include "Utilities/ReleaseAssert.h"
#include "Utilities/BasicDefinitions.h" //includes stdoutput.h
#include "Utilities/BasicFunctions.h"	//includes stdoutput.h
#include "Linalg/BasicLinalg.h"		//includes Vector.h

#include "Graphics/VisualizationSystemContainer.h"
#include "Graphics/VisualizationPrimitives.h"

#include <stdio.h>

//! class used to define standard colors
namespace EXUvis {

	//! get color from index; used e.g. for axes numeration
	const Float4& GetColor(Index i)
	{
		switch (i)
		{
		case 0: return red;
		case 1: return green;
		case 2: return blue;
		case 3: return cyan;
		case 4: return magenta;
		case 5: return yellow;
		case 6: return black;
		case 7: return orange;
		case 8: return lila;
		case 9: return grey2;
		default: return rose; //any other number gives this color
		}
	}

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//! copy bodyGraphicsData (of body) into global graphicsData (of system)
	void AddBodyGraphicsDataColored(const BodyGraphicsData& bodyGraphicsData, GraphicsData& graphicsData, 
		const Float3& position, const Matrix3DF& rotation, const Float3& refPosition, const Matrix3DF& refRotation, const Float3& velocity, const Float3& angularVelocity,
		Index itemID, const VisualizationSettings& visualizationSettings, bool contourColor)
	{
		bool applyRotation = true;
		if (rotation(0, 0) == 1.f && rotation(1, 1) == 1.f && rotation(2, 2) == 1.f) { applyRotation = false; }

		for (GLLine item : bodyGraphicsData.glLines) //copy objects, because we also need the transformed objects
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point1, item.point1);
				EXUmath::RigidBodyTransformation(rotation, position, item.point2, item.point2);
			}
			else
			{
				item.point1 += position;
				item.point2 += position;
			}
			graphicsData.glLines.Append(item);
		}

		for (GLCircleXY item : bodyGraphicsData.glCirclesXY) //copy objects
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point, item.point);
			}
			else
			{
				item.point += position;
			}
			graphicsData.glCirclesXY.Append(item);
		}

		for (GLText item : bodyGraphicsData.glTexts) //copy objects, but string pointers are just assigned!
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point, item.point);
			}
			else
			{
				item.point += position;
			}

			UnsignedIndex len = strlen(item.text);
			//int i = (int)strlen("x");
			char* temp = new char[len + 1]; //needs to be copied, because string is destroyed everytime it is updated! ==> SLOW for large number of texts (node numbers ...)
			//strcpy_s(temp, len + 1, item.text); //not working with gcc
			strcpy(temp, item.text); //item.text will be destroyed upon deletion of BodyGraphicsData!
			item.text = temp;
			graphicsData.glTexts.Append(item);
		}

		if (!contourColor)
		{
			for (GLTriangle item : bodyGraphicsData.glTriangles) //copy objects
			{
				item.itemID = itemID;
				if (applyRotation)
				{
					for (Index i = 0; i < 3; i++)
					{
						EXUmath::RigidBodyTransformation(rotation, position, item.points[i], item.points[i]);
						item.normals[i] = rotation * item.normals[i];
					}
				}
				else
				{
					for (Index i = 0; i < 3; i++)
					{
						item.points[i] += position;
					}
				}
				graphicsData.glTriangles.Append(item);
			}
		}
		else
		{
			OutputVariableType outputVariable = visualizationSettings.contour.outputVariable;
			Index outputVariableComponent = visualizationSettings.contour.outputVariableComponent;
			//Float4 currentColor(defaultColorBlue4);
			Float3 value; //this is the computed value per triangle vertex (node)
			Float3 pRef;

			for (Index iItem = 0; iItem < bodyGraphicsData.glTriangles.NumberOfItems(); iItem++)
			{
				GLTriangle  item = bodyGraphicsData.glTriangles[iItem]; //copy objects, then modify some data
				item.itemID = itemID;
				if (applyRotation)
				{
					for (Index i = 0; i < 3; i++)
					{
						Float3 locPoint = item.points[i];
						EXUmath::RigidBodyTransformation(rotation, position, locPoint, item.points[i]);
						item.normals[i] = rotation * item.normals[i];

						switch (outputVariable)
						{
						case OutputVariableType::Position:
							value = item.points[i];
							break;
						case OutputVariableType::Displacement:
						{
							//reference rotation may be different from zero!
							EXUmath::RigidBodyTransformation(refRotation, refPosition, locPoint, pRef);
							value = item.points[i] - pRef;
							break;
						}
						case OutputVariableType::DisplacementLocal:
						{
							value.SetAll(0);
							break;
						}
						case OutputVariableType::Velocity:
						{
							//reference rotation may be different from zero!
							value = velocity + angularVelocity.CrossProduct(rotation*locPoint);
							break;
						}
						case OutputVariableType::VelocityLocal:
						{
							//reference rotation may be different from zero!
							value = (velocity + (angularVelocity).CrossProduct(rotation*locPoint))*rotation; //this is (A^T * v)
							break;
						}
						case OutputVariableType::AngularVelocity:
						{
							value = angularVelocity;
							break;
						}
						case OutputVariableType::AngularVelocityLocal:
						{
							value = angularVelocity * rotation;
							break;
						}
						default:
							value.SetAll(0);
							break;
						}
						EXUvis::ComputeContourColor(value, outputVariable, outputVariableComponent, item.colors[i]);
					}
				}
				else //without rotation
				{
					for (Index i = 0; i < 3; i++)
					{
						item.points[i] += position;
						switch (outputVariable)
						{
						case OutputVariableType::Position:
							value = item.points[i];
							break;
						case OutputVariableType::Displacement:
							value = item.points[i] - refPosition;
							break;
						case OutputVariableType::DisplacementLocal:
							value.SetAll(0);
							break;
						case OutputVariableType::Velocity:
						{
							value = velocity; //no rotation
							break;
						}
						case OutputVariableType::VelocityLocal:
						{
							value = velocity; //no rotation
							break;
						}
						case OutputVariableType::AngularVelocity:
							value.SetAll(0);
							break;
						case OutputVariableType::AngularVelocityLocal:
							value.SetAll(0);
							break;
						default:
							value.SetAll(0);
							break;
						}
						EXUvis::ComputeContourColor(value, outputVariable, outputVariableComponent, item.colors[i]);
					}
				}
				graphicsData.glTriangles.Append(item);
			}
		}

	}

	//! copy bodyGraphicsData (of body) into global graphicsData (of system)
	void AddBodyGraphicsData(const BodyGraphicsData& bodyGraphicsData, GraphicsData& graphicsData, const Float3& position,
		const Matrix3DF& rotation, Index itemID)
	{
		bool applyRotation = true;
		if (rotation(0, 0) == 1.f && rotation(1, 1) == 1.f && rotation(2, 2) == 1.f) { applyRotation = false; }

		for (GLLine item : bodyGraphicsData.glLines) //copy objects, because we also need the transformed objects
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point1, item.point1);
				EXUmath::RigidBodyTransformation(rotation, position, item.point2, item.point2);
			}
			else
			{
				item.point1 += position;
				item.point2 += position;
			}
			graphicsData.glLines.Append(item);
		}

		for (GLCircleXY item : bodyGraphicsData.glCirclesXY) //copy objects
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point, item.point);
			}
			else
			{
				item.point += position;
			}
			graphicsData.glCirclesXY.Append(item);
		}

		for (GLText item : bodyGraphicsData.glTexts) //copy objects, but string pointers are just assigned!
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				EXUmath::RigidBodyTransformation(rotation, position, item.point, item.point);
			}
			else
			{
				item.point += position;
			}

			UnsignedIndex len = strlen(item.text);
			//int i = (int)strlen("x");
			char* temp = new char[len + 1]; //needs to be copied, because string is destroyed everytime it is updated! ==> SLOW for large number of texts (node numbers ...)
			//strcpy_s(temp, len + 1, item.text); //not working with gcc
			strcpy(temp, item.text); //item.text will be destroyed upon deletion of BodyGraphicsData!
			item.text = temp;
			graphicsData.glTexts.Append(item);
		}

		for (GLTriangle item : bodyGraphicsData.glTriangles) //copy objects
		{
			item.itemID = itemID;
			if (applyRotation)
			{
				for (Index i = 0; i < 3; i++)
				{
					EXUmath::RigidBodyTransformation(rotation, position, item.points[i], item.points[i]);
					item.normals[i] = rotation * item.normals[i];
				}
			}
			else
			{
				for (Index i = 0; i < 3; i++)
				{
					item.points[i] += position;
				}
			}
			graphicsData.glTriangles.Append(item);
		}
	}

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//! draw a simple spring in 2D with given endpoints p0,p1 a width, a (normalized) normal vector for the width drawing and number of spring points numberOfPoints
	void DrawSpring2D(const Vector3D& p0, const Vector3D& p1, const Vector3D& vN, Index numberOfPoints, Real halfWidth, const Float4& color, 
		GraphicsData& graphicsData, Index itemID)
	{
		//2D drawing in XY plane
		Vector3D v0 = p1 - p0;

		Real L = v0.GetL2Norm(); //length of spring
		Real d = L / (Real)numberOfPoints; //split spring into pieces: shaft, (n-2) parts, end
		if (L != 0.f) { v0 /= L; }

		Vector3D pLast; //for last drawn point
		for (Index i = 0; i <= numberOfPoints; i++)
		{
			Vector3D pAct = p0 + v0 * (float)i*d;
			Real sign = (Real)(i % 2); //sign
			if (i > 1 && i < numberOfPoints - 1) { pAct += halfWidth * (sign*2.f - 1.f)* vN; }

			if (i > 0)
			{
				graphicsData.AddLine(pLast, pAct, color, color, itemID);
			}

			pLast = pAct;
		}

	}

	//! draw a spring in 3D with given endpoints p0,p1, a width, windings and tiling
	void DrawSpring(const Vector3D& p0, const Vector3D& p1, Index numberOfWindings, Index nTilesPerWinding,
		Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID, bool draw3D)
	{
		Vector3D v0 = p1 - p0;

		Real L = v0.GetL2Norm(); //length of spring
		Real d = L / (Real)numberOfWindings; //split spring into pieces: shaft, (n-2) parts, end
		if (L != 0.f) 
		{ 
			v0 /= L;
			Vector3D n1, n2;
			EXUmath::ComputeOrthogonalBasisVectors(v0, n1, n2, true);

			Vector3D pLast = p0;

			for (Index i = 0; i < numberOfWindings; i++)
			{
				for (Index j = 0; j < nTilesPerWinding; j++)
				{
					Real phi = 2 * EXUstd::pi * j / (Real)nTilesPerWinding;
					Vector3D p = p0 + d * ((Real)i + j / (Real)nTilesPerWinding)*v0 + radius*sin(phi)*n1 + radius*cos(phi)*n2;

					graphicsData.AddLine(pLast, p, color, color, itemID);
					pLast = p;
				}
			}
			graphicsData.AddLine(pLast, p1, color, color, itemID);

		}

	}

	////! draw number for item at selected position and with label, such as 'N' for nodes, etc.
	//void DrawItemNumberWithoutID(const Float3& pos, VisualizationSystem* vSystem, Index itemNumber, const char* label, const Float4& color)
	//{
	//	float offx = 0.25f; //in text coordinates, relative to textsize
	//	float offy = 0.25f; //in text coordinates, relative to textsize
	//	float textSize = 0.f; //use default value
	//	vSystem->graphicsData.AddText(pos, color, label + EXUstd::ToString(itemNumber), textSize, offx, offy, Index2ItemID(-1, ItemType::_None, 0));
	//}

	//! draw number for item at selected position and with label, such as 'N' for nodes, etc.
	void DrawItemNumber(const Float3& pos, VisualizationSystem* vSystem, Index itemID, const char* label, const Float4& color)
	{
		float offx = 0.25f; //in text coordinates, relative to textsize
		float offy = 0.25f; //in text coordinates, relative to textsize
		float textSize = 0.f; //use default value
		Index itemNumber;
		ItemType itemType;
		Index mbsNumber;
		ItemID2IndexType(itemID, itemNumber, itemType, mbsNumber);
        //Float4 colorAlt = color;
        //if (colorAlt[0] + colorAlt[1] + colorAlt[2] > 0.7)
        //{
        //    colorAlt[0] *= 0.5f; //make this color slightly darker
        //    colorAlt[1] *= 0.5f;
        //    colorAlt[2] *= 0.5f;
        //}
        //else
        //{
        //    colorAlt[0] += 0.3f; //make lighter (this will always work
        //    colorAlt[1] += 0.3f;
        //    colorAlt[2] += 0.3f;
        //}
        vSystem->graphicsData.AddText(pos, EXUvis::AltColor(color), label + EXUstd::ToString(itemNumber), textSize, offx, offy, itemID);
	}


	//! add a 3D circle to graphicsData with reference point (pMid) and rotation matrix rot, which serves as a transformation;
	//! points for circles are computed in x/y plane and hereafter rotated by rotation matrix rot
	void DrawCircle(const Vector3D& pMid, const Matrix3D& rot, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID, Index nTiles)
	{
		if (nTiles < 2) { nTiles = 2; } //less than 2 tiles makes no sense
		if (radius <= 0.) { return; } //just a line

		Real fact = (Real)(nTiles - 1) / (2.*EXUstd::pi);
		for (Index i = 0; i < nTiles+1; i++)
		{
			Real phi0 = i / fact;
			Real phi1 = (i + 1) / fact;

			Real x0 = radius*sin(phi0);
			Real y0 = radius * cos(phi0);
			Real x1 = radius * sin(phi1);
			Real y1 = radius * cos(phi1);

			graphicsData.AddLine(pMid + rot * Vector3D({ x0,y0,0. }), pMid + rot * Vector3D({ x1,y1,0. }), color, color, itemID);
		}
	}

	//! add a cylinder to graphicsData with reference point (pAxis0), axis vector (vAxis) and radius using triangle representation
	//! angleRange is used to draw only part of the cylinder; 
	//! if lastFace=true, a closing face is drawn in case of limited angle; 
	//! cutPlain=true: a plain cut through cylinder is made; false: draw the cake shape ...
	//! innerRadius: if > 0, then this is a cylinder with a hole
	void DrawCylinder(const Vector3D& pAxis0, const Vector3D& vAxis, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID,
		Index nTiles, Real innerRadius, Vector2D angleRange, bool lastFace, bool cutPlain, bool drawSmooth)
	{
		if (nTiles < 2) { nTiles = 2; } //less than 2 tiles makes no sense
		if (radius <= 0.) { return; } //just a line
		if (vAxis.GetL2NormSquared() == 0.) { return; } //too short

		//create points at left and right face
		//points0 = copy.deepcopy(pAxis) #[pAxis[0], pAxis[1], pAxis[2]] #avoid change of pAxis
		Vector3D pAxis1 = pAxis0 + vAxis;

		Vector3D basisN1, basisN2;
		EXUmath::ComputeOrthogonalBasisVectors(vAxis, basisN1, basisN2, false);

		//#create normals at left and right face(pointing inwards)
		Real alpha = angleRange[1] - angleRange[0]; //angular range
		Real alpha0 = angleRange[0];

		Real fact = (Real)nTiles; //#create correct part of cylinder (closed/not closed
		if (alpha < 2.*EXUstd::pi) { fact = (Real)(nTiles - 1); }

		std::array<Vector3D, 3> points;
		std::array<Vector3D, 3> normals;
		std::array<Float4, 3> colors = { {color,color,color} }; //std::array has no real initializer list==>use {{}}; all triangles have same color

		Vector3D nF1 = vAxis;
		nF1.Normalize();

#ifdef FLIP_NORMALS
		nF1 = -nF1;
#endif

		std::array<Vector3D, 3> normalsFace1 = { { nF1,nF1,nF1 } };
		nF1 = -nF1;
		std::array<Vector3D, 3> normalsFace0 = { { nF1,nF1,nF1 } };
		Vector3D n0(0);
		Vector3D n1(0);


		for (Index i = 0; i < nTiles; i++)
		{
			Real phi0 = alpha0 + i * alpha / fact;
			Real phi1 = alpha0 + (i+1) * alpha / fact;

			Real x0 = sin(phi0);
			Real y0 = cos(phi0);
			Real x1 = sin(phi1);
			Real y1 = cos(phi1);
			Vector3D vv0 = x0 * basisN1 + y0 * basisN2;
			Vector3D vv1 = x1 * basisN1 + y1 * basisN2;
			Vector3D pzL0 = pAxis0 + radius * vv0;
			Vector3D pzL1 = pAxis0 + radius * vv1;
			Vector3D pzR0 = pAxis1 + radius * vv0;
			Vector3D pzR1 = pAxis1 + radius * vv1;
			if (drawSmooth)
			{
#ifdef FLIP_NORMALS
				n0 = -vv0;
				n1 = -vv1;
#else
				n0 = vv0;
				n1 = vv1;
#endif
				n0.Normalize();
				n1.Normalize();
			}

			//+++++++++++++++++++++++++++++++
			//circumference:
			normals[0] = n0;
			normals[1] = n1;
			normals[2] = n0;
			points[0] = pzL0;
			points[1] = pzR1;
			points[2] = pzR0;
			graphicsData.AddTriangle(points, normals, colors, itemID);

			//normals[0] = n0;
			//normals[1] = n1;
			normals[2] = n1;
			//points[0] = pzL0;
			points[1] = pzL1;
			points[2] = pzR1;
			graphicsData.AddTriangle(points, normals, colors, itemID);

			if (innerRadius > 0.)
			{
				Vector3D pzL0i = pAxis0 + innerRadius * vv0;
				Vector3D pzL1i = pAxis0 + innerRadius * vv1;
				Vector3D pzR0i = pAxis1 + innerRadius * vv0;
				Vector3D pzR1i = pAxis1 + innerRadius * vv1;

				//+++++++++++++++++++++++++++++++
				//circumference:
				normals[0] = -n0;
				normals[1] = -n1;
				normals[2] = -n0;
				points[0] = pzL0i;
				points[1] = pzR0i;
				points[2] = pzR1i;
				graphicsData.AddTriangle(points, normals, colors, itemID);

				//normals[0] = -n0;
				//normals[1] = -n1;
				normals[2] = -n1;
				//points[0] = pzL0i;
				points[1] = pzR1i;
				points[2] = pzL1i;
				graphicsData.AddTriangle(points, normals, colors, itemID);

				//+++++++++++++++++++++++++++++++
				//side faces:
				points[0] = pzL0i;
				points[1] = pzL1;
				points[2] = pzL0;
				graphicsData.AddTriangle(points, normalsFace0, colors, itemID);
				points[0] = pzL0i;
				points[1] = pzL1i;
				points[2] = pzL1;
				graphicsData.AddTriangle(points, normalsFace0, colors, itemID);

				points[0] = pzR0i;
				points[1] = pzR0;
				points[2] = pzR1;
				graphicsData.AddTriangle(points, normalsFace1, colors, itemID);

				points[0] = pzR1i;
				points[1] = pzR0i;
				points[2] = pzR1;
				graphicsData.AddTriangle(points, normalsFace1, colors, itemID);
			}
			else
			{
				//+++++++++++++++++++++++++++++++
				//side faces:
				points[0] = pAxis0;
				points[1] = pzL1;
				points[2] = pzL0;
				graphicsData.AddTriangle(points, normalsFace0, colors, itemID);

				points[0] = pAxis1;
				points[1] = pzR0;
				points[2] = pzR1;
				graphicsData.AddTriangle(points, normalsFace1, colors, itemID);
			}
		}
	}

	//! draw a sphere with center at p, radius and color; nTiles are in 2 dimensions (8 tiles gives 8x8 x 2 faces)
	void DrawSphere(const Vector3D& p, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID,
		Index nTiles, bool drawSmooth)
	{
		if (nTiles < 2) { nTiles = 2; } //less than 2 tiles makes no sense
		if (radius <= 0.) { return; } //not visible

		//drawSmooth = false;
#ifdef FLIP_TRIANGLES  
		const Index trigOff = 1;
#else
		const Index trigOff = 0;
#endif
#ifdef FLIP_NORMALS
		const Real flipFact = -1.;
#else
		const Real flipFact = 1.;
#endif

		std::array<Vector3D, 3> points;
		std::array<Vector3D, 3> normals; // = { Vector3D(0), Vector3D(0), Vector3D(0) };
		std::array<Float4, 3> colors = { { color,color,color } }; //all triangles have same color

		Index nTiles2 = 2 * nTiles;
		//create points for circles around z - axis with tiling
		for (Index i0 = 0; i0 < nTiles; i0++) //nTiles+1 in python, when generating points
		{
			for (Index iphi = 0; iphi < nTiles2; iphi++)
			{
				Real z0 = -radius * cos(EXUstd::pi * (Real)i0 / (Real)nTiles);    //runs from - r ..r(this is the coordinate of the axis of circles)
				Real fact0 = sin(EXUstd::pi*(Real)i0 / (Real)nTiles);
				Real z1 = -radius * cos(EXUstd::pi * (Real)(i0+1) / (Real)nTiles);    //runs from - r ..r(this is the coordinate of the axis of circles)
				Real fact1 = sin(EXUstd::pi*(Real)(i0 + 1) / (Real)nTiles);

				Real phiA = 2. * EXUstd::pi * (Real)iphi / (Real)nTiles2; //angle
				Real phiB = 2. * EXUstd::pi * (Real)(iphi+1) / (Real)nTiles2; //angle

				Real x0A = fact0 * radius * sin(phiA);
				Real y0A = fact0 * radius * cos(phiA);
				Real x1A = fact1 * radius * sin(phiA);
				Real y1A = fact1 * radius * cos(phiA);
				Real x0B = fact0 * radius * sin(phiB);
				Real y0B = fact0 * radius * cos(phiB);
				Real x1B = fact1 * radius * sin(phiB);
				Real y1B = fact1 * radius * cos(phiB);

				Vector3D v0A({ x0A, y0A, z0 });
				Vector3D v1A({ x1A, y1A, z1 });
				Vector3D v0B({ x0B, y0B, z0 });
				Vector3D v1B({ x1B, y1B, z1 });

				points[0] = p + v0A;
				points[1+trigOff] = p + v1A;
				points[2-trigOff] = p + v1B;

				//triangle1: 0A, 1B, 1A
				if (drawSmooth)
				{
					normals[0] = flipFact * v0A;
					normals[1+ trigOff] = flipFact * v1A;
					normals[2- trigOff] = flipFact * v1B;
					normals[0].Normalize();
					normals[1].Normalize();
					normals[2].Normalize();
				}
				else
				{
					ComputeTriangleNormals(points, normals);
				}
				graphicsData.AddTriangle(points, normals, colors, itemID);

				points[0] = p + v0A;
				points[2 - trigOff] = p + v0B;
				points[1 + trigOff] = p + v1B;
				//triangle1: 0A, 0B, 1B
				if (drawSmooth)
				{
					normals[0] = flipFact * v0A;
					normals[2- trigOff] = flipFact * v0B;
					normals[1+ trigOff] = flipFact * v1B;

					normals[0].Normalize();
					normals[1].Normalize();
					normals[2].Normalize();
				}
				else
				{
					ComputeTriangleNormals(points, normals);
				}
				graphicsData.AddTriangle(points, normals, colors, itemID);
			}
		}
	}

	//! draw cube with midpoint and size in x,y and z direction
	void DrawOrthoCube(const Vector3D& midPoint, const Vector3D& size, const Float4& color, GraphicsData& graphicsData, 
		Index itemID, bool showFaces, bool showEdges)
	{
		Real x = 0.5*size[0];
		Real y = 0.5*size[1];
		Real z = 0.5*size[2];

		SlimVectorBase<Vector3D, 8> pc = { Vector3D({-x,-y,-z}), Vector3D({ x,-y,-z}), Vector3D({ x, y,-z}), Vector3D({-x, y,-z}),
										   Vector3D({-x,-y, z}), Vector3D({ x,-y, z}), Vector3D({ x, y, z}), Vector3D({-x, y, z}) }; //cube corner points

		for (Vector3D& point : pc)
		{
			point += midPoint;
		}

		if (showEdges)
		{
			graphicsData.AddLine(pc[0], pc[1], color, color, itemID);
			graphicsData.AddLine(pc[1], pc[2], color, color, itemID);
			graphicsData.AddLine(pc[2], pc[3], color, color, itemID);
			graphicsData.AddLine(pc[3], pc[0], color, color, itemID);
			graphicsData.AddLine(pc[4], pc[5], color, color, itemID);
			graphicsData.AddLine(pc[5], pc[6], color, color, itemID);
			graphicsData.AddLine(pc[6], pc[7], color, color, itemID);
			graphicsData.AddLine(pc[7], pc[4], color, color, itemID);
			graphicsData.AddLine(pc[0], pc[4], color, color, itemID);
			graphicsData.AddLine(pc[1], pc[5], color, color, itemID);
			graphicsData.AddLine(pc[2], pc[6], color, color, itemID);
			graphicsData.AddLine(pc[3], pc[7], color, color, itemID);
		}

		if (showFaces)
		{
			//sketch of cube: (z goes upwards from node 1 to node 5)
			// bottom :         top:
			// ^ y				^ y
			// |				|
			// 3---2			7---6
			// |   |			|   |
			// |   |			|   |
			// 0---1--> x		4---5--> x

			//std::array<SlimArray<Index,3>, 12> //does not work with recursive initializer list
			const Index nTrigs = 12;
			//Index trigList[nTrigs][3] = { {0, 1, 2}, {0, 2, 3},  {6, 5, 4}, {6, 4, 7},  {0, 4, 1}, {1, 4, 5},  {1, 5, 2}, {2, 5, 6},  {2, 6, 3}, {3, 6, 7},  {3, 7, 0}, {0, 7, 4} };

			SlimVectorBase<Index, 12 * 3> trigList = { 0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 6, 2, 6, 3, 3, 6, 7, 3, 7, 0, 0, 7, 4 };

			std::array<Vector3D, 3> points;
			std::array<Vector3D, 3> normals = { Vector3D(0), Vector3D(0), Vector3D(0) };
			SlimVectorBase<Float4, 3> colors({ color,color,color }); //all triangles have same color
			//std::array<Vector3D, 8> pc = { Vector3D({-x,-y,-z}), Vector3D({ x,-y,-z}), Vector3D({ x, y,-z}), Vector3D({-x, y,-z}),
			//							   Vector3D({-x,-y, z}), Vector3D({ x,-y, z}), Vector3D({ x, y, z}), Vector3D({-x, y, z}) }; //cube corner points

			//std::array<Vector3D, 3> points;
			//std::array<Vector3D, 3> normals = { Vector3D(0), Vector3D(0), Vector3D(0) };
			//std::array<Float4, 3> colors({ color,color,color }); //all triangles have same color

			for (Index i = 0; i < nTrigs; i++)
			{
				points[0] = pc[trigList[i * 3 + 0]];
				points[1] = pc[trigList[i * 3 + 1]];
				points[2] = pc[trigList[i * 3 + 2]];
				//points[0] = pc[trigList[i][0]];
				//points[1] = pc[trigList[i][1]];
				//points[2] = pc[trigList[i][2]];
				ComputeTriangleNormals(points, normals);
				graphicsData.AddTriangle(points, normals, colors, itemID);
			}
		}

	}


	//! add a cone to graphicsData with reference point (pAxis0), axis vector (vAxis) and radius using triangle representation
	//! cone starts at pAxis0, tip is at pAxis0+vAxis0
	void DrawCone(const Vector3D& pAxis0, const Vector3D& vAxis, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID,
		Index nTiles, bool drawSmooth)
	{
		if (nTiles < 2) { nTiles = 2; } //less than 2 tiles makes no sense
		if (radius <= 0.) { return; } //just a line
		Real axisLength = vAxis.GetL2Norm();
		if (axisLength == 0.) { return; } //too short

		//create points at left and right face
		//points0 = copy.deepcopy(pAxis) #[pAxis[0], pAxis[1], pAxis[2]] #avoid change of pAxis
		Vector3D pAxis1 = pAxis0 + vAxis;

		Vector3D basisN1, basisN2;
		EXUmath::ComputeOrthogonalBasisVectors(vAxis, basisN1, basisN2, false);

		//#create normals at left and right face(pointing inwards)
		Real alpha = 2.*EXUstd::pi;

		Real fact = (Real)nTiles; //#create correct part of cylinder (closed/not closed

		std::array<Vector3D, 3> points;
		std::array<Vector3D, 3> normals = { Vector3D(0), Vector3D(0), Vector3D(0) };
		std::array<Float4, 3> colors = { color,color,color }; //all triangles have same color

		Vector3D nF0 = vAxis;
		nF0.Normalize();

#ifdef FLIP_NORMALS
		std::array<Vector3D, 3> normalsFace0 = { nF0,nF0,nF0 };
#else
		std::array<Vector3D, 3> normalsFace0 = { -nF0,-nF0,-nF0 };
#endif
		for (Index i = 0; i < nTiles; i++)
		{
			Real phi0 = i * alpha / fact;
			Real phi1 = (i + 1) * alpha / fact;

			Real x0 = radius * sin(phi0);
			Real y0 = radius * cos(phi0);
			Real x1 = radius * sin(phi1);
			Real y1 = radius * cos(phi1);
			Vector3D vv0 = x0 * basisN1 + y0 * basisN2;
			Vector3D vv1 = x1 * basisN1 + y1 * basisN2;
			Vector3D pzL0 = pAxis0 + vv0;
			Vector3D pzL1 = pAxis0 + vv1;

			//+++++++++++++++++++++++++++++++
			//circumference:
			if (drawSmooth)
			{
				//normal to cone surface:
				Vector3D n0 = (axisLength / radius)*vv0 + radius * nF0;
				Vector3D n1 = (axisLength / radius)*vv1 + radius * nF0;
				n0.Normalize();
				n1.Normalize();
#ifdef FLIP_NORMALS
				normals[0] = -n0;
				normals[1] = -n1;
				normals[2] = -n1;
#else
				normals[0] = n0;
				normals[1] = n1;
				normals[2] = n1;
#endif
			}
			points[0] = pzL0;
			points[1] = pzL1;
			points[2] = pAxis1;
			graphicsData.AddTriangle(points, normals, colors, itemID);

			//+++++++++++++++++++++++++++++++
			//side faces:
			points[0] = pAxis0;
			points[1] = pzL1;
			points[2] = pzL0;
			graphicsData.AddTriangle(points, normalsFace0, colors, itemID);
		}
	}

	//! draw orthonormal basis at point p using a rotation matrix, which transforms local to global coordinates
	//! red=axisX, green=axisY, blue=axisZ
	//! length defines the length of each axis; radius is the radius of the shaft; arrowSize is diameter relative to radius
	//! colorfactor: 1=rgb color, 0=grey color (and any value between)
	void DrawOrthonormalBasis(const Vector3D& p, const Matrix3D& rot, Real length, Real radius, 
		GraphicsData& graphicsData, Index itemID, float colorFactor, bool draw3D, Index nTiles, Real arrowSizeRelative, Index showNumber, const char* preText)
	{

		for (Index i = 0; i < 3; i++)
		{
			Vector3D v = rot.GetColumnVector<3>(i);
			Float4 color(ModifyColor(GetColor(i), colorFactor));
			if (draw3D)
			{
				DrawCylinder(p, length*v, radius, color, graphicsData, itemID, nTiles);
				DrawCone(p + length * v, (radius*arrowSizeRelative * 3)*v, arrowSizeRelative*radius, color, graphicsData, itemID, nTiles);
			} else //draw as simple line
			{
				graphicsData.AddLine(p, p + length * v, color, color, itemID);
			}
			if (showNumber != EXUstd::InvalidIndex || preText != nullptr)
			{
				STDstring textStr;
				if (preText != nullptr) { textStr = preText; }
				if (showNumber != EXUstd::InvalidIndex)
				{ 
					textStr += EXUstd::ToString(showNumber);
				}
				graphicsData.AddText(p + (length + radius * arrowSizeRelative * 3) * v, color, textStr, 0.f, 0.25f, 0.25f, itemID);
			}
		}
	}
	void DrawArrow(const Vector3D& p, const Vector3D& v, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID,
		Index nTiles, bool doubleArrow, bool draw3D)
	{
		Real arrowSizeRelative = 2.5;
		Real len = v.GetL2Norm();

		if (len != 0)
		{
			Vector3D v0 = (1. / len)*v;

			if (!draw3D) //draw simplified vector
			{
				Vector3D v1 = (len - 3 * radius * arrowSizeRelative)*v0;
				Vector3D n1, n2;
				EXUmath::ComputeOrthogonalBasisVectors(v0, n1, n2, true);

				graphicsData.AddLine(p, p + v, color, color, itemID);
				graphicsData.AddLine(p + v, p + v1 + radius * n1, color, color, itemID);
				graphicsData.AddLine(p + v, p + v1 - radius * n1, color, color, itemID);
				graphicsData.AddLine(p + v, p + v1 + radius * n2, color, color, itemID);
				graphicsData.AddLine(p + v, p + v1 - radius * n2, color, color, itemID);

				if (doubleArrow)
				{
					Vector3D v2 = (len - 2*3 * radius * arrowSizeRelative)*v0;

					graphicsData.AddLine(p + v1, p + v2 + radius * n1, color, color, itemID);
					graphicsData.AddLine(p + v1, p + v2 - radius * n1, color, color, itemID);
					graphicsData.AddLine(p + v1, p + v2 + radius * n2, color, color, itemID);
					graphicsData.AddLine(p + v1, p + v2 - radius * n2, color, color, itemID);
				}
			}
			else
			{
				if (!doubleArrow)
				{
					Vector3D v1 = (len - 3 * radius * arrowSizeRelative)*v0;
					DrawCylinder(p, v1, radius, color, graphicsData, itemID, nTiles);
					DrawCone(p + v1, (3 * radius * arrowSizeRelative) * v0, arrowSizeRelative*radius, color, graphicsData, itemID, nTiles);
				}
				else
				{
					Vector3D v1 = (len - 2 * 3 * radius * arrowSizeRelative)*v0;
					DrawCylinder(p, v1, radius, color, graphicsData, itemID, nTiles);
					DrawCone(p + v1, (3 * radius * arrowSizeRelative) * v0, arrowSizeRelative*radius, color, graphicsData, itemID, nTiles);
					DrawCone(p + v1 + (3 * radius * arrowSizeRelative) * v0, (3 * radius * arrowSizeRelative) * v0, arrowSizeRelative*radius, 
						color, graphicsData, itemID, nTiles);
				}
			}
		}
	}

	//! draw node either with 3 circles or with sphere at given point and with given radius
	void DrawNode(const Vector3D& p, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID, bool draw3D, Index nTiles)
	{
		if (radius == 0) { return; } //this is for the special case where the node is not shown, but the basis vectors shall be drawn!
		if (nTiles == 0)
		{
			graphicsData.AddSphere(p, color, itemID);
		}
		else if (draw3D)
		{
			Index bitTiling = TilingToBitResolution(nTiles);
			graphicsData.AddSphere(p, color, itemID, (float)radius, bitTiling);
			//DrawSphere(p, radius, color, graphicsData, itemID, nTiles); //slow!
		}
		else
		{
			Vector3D pPrevious[3]; //end points of previous segment
			Vector3D pAct[3];
			for (Index i = 0; i <= nTiles; i++)
			{
				Real phi = (Real)i / (Real)nTiles * 2. * EXUstd::pi;
				Real x = radius * sin(phi);
				Real y = radius * cos(phi);

				pAct[0] = p + Vector3D({ 0,x,y });
				pAct[1] = p + Vector3D({ x,0,y });
				pAct[2] = p + Vector3D({ x,y,0 });

				if (i > 0)
				{
					for (Index j = 0; j < 3; j++)
					{
						graphicsData.AddLine(pAct[j],pPrevious[j],color,color, itemID);
					}
				}
				for (Index j = 0; j < 3; j++)
				{
					pPrevious[j] = pAct[j];
				}
			}
		}
	}

	//! draw marker either with 3 crosses or with cube at given point and with given size
	void DrawMarker(const Vector3D& p, Real size, const Float4& color, GraphicsData& graphicsData, Index itemID, bool draw3D)
	{
		if (draw3D)
		{
			DrawOrthoCube(p, Vector3D({ size,size,size }), color, graphicsData, itemID);
			//DrawSphere(p, size, color, graphicsData, 2, false); //draw coarse and with flat shading
		}
		else
		{
			Real s = 0.5*size;
			graphicsData.AddLine(p + Vector3D({ s,s,0 }), p - Vector3D({ s,s,0 }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ -s,s,0 }), p - Vector3D({ -s,s,0 }), color, color, itemID);

			graphicsData.AddLine(p + Vector3D({ s,0,s }), p - Vector3D({ s,0,s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ -s,0,s }), p - Vector3D({ -s,0,s }), color, color, itemID);

			graphicsData.AddLine(p + Vector3D({ 0,s,s }), p - Vector3D({ 0,s,s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ 0,-s,s }), p - Vector3D({ 0,-s,s }), color, color, itemID);

		}
	}

	//! draw sensor as diamond
	void DrawSensor(const Vector3D& p, Real radius, const Float4& color, GraphicsData& graphicsData, Index itemID, bool draw3D)
	{
		if (draw3D)
		{
			DrawSphere(p, radius, color, graphicsData, itemID, 2, false); //draw coarse and with flat shading
		}
		else
		{
			Real s = radius;
			graphicsData.AddLine(p + Vector3D({ s,0,0 }),  p - Vector3D({ 0, s,0 }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ s,0,0 }),  p - Vector3D({ 0,-s,0 }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ -s,0,0 }), p - Vector3D({ 0, s,0 }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ -s,0,0 }), p - Vector3D({ 0,-s,0 }), color, color, itemID);

			graphicsData.AddLine(p + Vector3D({ s,0,0 }), p - Vector3D({ 0,0, s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ s,0,0 }), p - Vector3D({ 0,0,-s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({-s,0,0 }), p - Vector3D({ 0,0, s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({-s,0,0 }), p - Vector3D({ 0,0,-s }), color, color, itemID);

			graphicsData.AddLine(p + Vector3D({ 0, s,0 }), p - Vector3D({ 0,0, s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ 0, s,0 }), p - Vector3D({ 0,0,-s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ 0,-s,0 }), p - Vector3D({ 0,0, s }), color, color, itemID);
			graphicsData.AddLine(p + Vector3D({ 0,-s,0 }), p - Vector3D({ 0,0,-s }), color, color, itemID);
		}
	}

};
