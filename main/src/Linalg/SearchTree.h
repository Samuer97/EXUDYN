/** ***********************************************************************************************
* @class	    SearchTree
* @brief		Class for boxed search
* @details		Details:
*               allows to find objects in space efficiently
*
* @author		Gerstmayr Johannes
* @date			2021-10-23 (generated)
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
#ifndef SEARCHTREE__H
#define SEARCHTREE__H

#include "Linalg/BasicLinalg.h"		//includes Vector.h


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++    BOX3D     +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


template<typename T>
class Box3DBase
{
public:
	using Vector3T = SlimVectorBase<T, 3>;

	Box3DBase(bool clear = true)
	{
		if (clear)
		{
			Clear();
		}
	}

	Box3DBase(const Vector3T& p1, const Vector3T& p2)
	{
		pmin[0] = EXUstd::Minimum(p1[0], p2[0]);
		pmin[1] = EXUstd::Minimum(p1[1], p2[1]);
		pmin[2] = EXUstd::Minimum(p1[2], p2[2]);

		pmax[0] = EXUstd::Maximum(p1[0], p2[0]);
		pmax[1] = EXUstd::Maximum(p1[1], p2[1]);
		pmax[2] = EXUstd::Maximum(p1[2], p2[2]);
	}

	Box3DBase(const Box3DBase& b)
	{
		for (int i = 0; i < 3; ++i)
		{
			pmin[i] = b.pmin[i];
			pmax[i] = b.pmax[i];
		}
	}

	Box3DBase(const Vector3T& c, T r)
	{
		for (int i = 0; i < 3; ++i)
		{
			pmin[i] = pmax[i] = c[i];
		}
		Increase(r);
	}

	//! check if box is empty, only based on x-value!
	bool Empty() const
	{
		return pmin[0] == std::numeric_limits<T>::max();
	}

	void Clear()
	{
		for (int i = 0; i < 3; ++i)
		{
			pmin[i] = std::numeric_limits<T>::max();
			pmax[i] = std::numeric_limits<T>::lowest();
		}
	}

	void Add(const Vector3T& p)
	{
		if (Empty())
		{
			for (int i = 0; i < 3; ++i) { pmin[i] = pmax[i] = p[i]; }
		}
		else
		{
			for (int i = 0; i < 3; ++i)
			{
				pmin[i] = EXUstd::Minimum(pmin[i], p[i]);
				pmax[i] = EXUstd::Maximum(pmax[i], p[i]);
			}
		}
	}

	void Add(const Box3DBase& b)
	{
		if (b.Empty()) return;

		if (Empty())
		{
			for (int i = 0; i < 3; ++i)
			{
				pmin[i] = b.pmin[i];
				pmax[i] = b.pmax[i];
			}
		}
		else
		{
			for (int i = 0; i < 3; ++i)
			{
				pmin[i] = EXUstd::Minimum(pmin[i], b.pmin[i]);
				pmax[i] = EXUstd::Maximum(pmax[i], b.pmax[i]);
			}
		}
	}

	//! get c-arrays
	const T* PMinC() const { return pmin; }
	const T* PMaxC() const { return pmax; }
	
	//! convert to Vector3D; const added to raise error if assignment happens e.g. box.PMin() = ...
	const Vector3T PMin() const { return Vector3T({ pmin[0], pmin[1], pmin[2] }); }
	const Vector3T PMax() const { return Vector3T({ pmax[0], pmax[1], pmax[2] }); }

	//! direct read access to Reals:
	EXUINLINE const T& PMinX() const { return pmin[0]; }
	EXUINLINE const T& PMinY() const { return pmin[1]; }
	EXUINLINE const T& PMinZ() const { return pmin[2]; }
	EXUINLINE const T& PMaxX() const { return pmax[0]; }
	EXUINLINE const T& PMaxY() const { return pmax[1]; }
	EXUINLINE const T& PMaxZ() const { return pmax[2]; }

	//! direct write access to Reals:
	EXUINLINE T& PMinX() { return pmin[0]; }
	EXUINLINE T& PMinY() { return pmin[1]; }
	EXUINLINE T& PMinZ() { return pmin[2]; }
	EXUINLINE T& PMaxX() { return pmax[0]; }
	EXUINLINE T& PMaxY() { return pmax[1]; }
	EXUINLINE T& PMaxZ() { return pmax[2]; }

	//! set with Vector3D
	EXUINLINE void SetPMin(const Vector3T& p) { for (int i = 0; i < 3; ++i) pmin[i] = p[i]; }
	EXUINLINE void SetPMax(const Vector3T& p) { for (int i = 0; i < 3; ++i) pmax[i] = p[i]; }

	T SizeX() const { return pmax[0] - pmin[0]; }
	T SizeY() const { return pmax[1] - pmin[1]; }
	T SizeZ() const { return pmax[2] - pmin[2]; }

	Vector3T Center() const { return Vector3T({ (pmin[0] + pmax[0]) * (T)0.5, (pmin[1] + pmax[1]) * (T)0.5, (pmin[2] + pmax[2]) * (T)0.5 }); }
	T Radius() const { return (PMax() - PMin()).GetL2Norm() * (T)0.5; }

	void Increase(T x, T y, T z)
	{
		pmin[0] -= x; pmin[1] -= y; pmin[2] -= z;
		pmax[0] += x; pmax[1] += y; pmax[2] += z;
	}
	void Increase(T x) { Increase(x, x, x); }

	void InflateFactor(T x)
	{
		Vector3T pc = Center();
		for (int i = 0; i < 3; ++i)
		{
			pmin[i] = pc[i] + (pmin[i] - pc[i]) * x;
			pmax[i] = pc[i] + (pmax[i] - pc[i]) * x;
		}
	}

	//! check if box b intersects with this (boundary included)
	bool Intersect(const Box3DBase& b) const
	{
		//> and < changed to >= and <= in order to simplify problems with points on boundaries
		if (pmin[0] >= b.pmax[0] || pmax[0] <= b.pmin[0]
			|| pmin[1] >= b.pmax[1] || pmax[1] <= b.pmin[1]
			|| pmin[2] >= b.pmax[2] || pmax[2] <= b.pmin[2])
		{
			return false;
		}
		return true;
	}

	//! return true if point p in closure or on boundary
	bool IsInside(const Vector3T& p) const
	{
		for (int i = 0; i < 3; ++i)
		{
			if (p[i] < pmin[i] || p[i] > pmax[i]) return false;
		}
		return true;
	}

	//! Function to check intersection of Box3D and triangle given by vertices v0, v1 and v2
	//! returns true, if box and triangle intersect
	bool BoxTriangleIntersect(const Vector3T& v0, const Vector3T& v1, const Vector3T& v2)
	{
		Vector3T boxCenter = this->Center();
		Vector3T boxHalfSize = (this->PMax() - this->PMin()) * (T)0.5;

		// Move triangle relative to AABB center
		Vector3T v0rel = v0 - boxCenter;
		Vector3T v1rel = v1 - boxCenter;
		Vector3T v2rel = v2 - boxCenter;

		// Triangle edges
		Vector3T edge1 = v1rel - v0rel;
		Vector3T edge2 = v2rel - v1rel;
		Vector3T edge3 = v0rel - v2rel;

		// AABB face normals (x, y, z axes)
		for (int i = 0; i < 3; i++)
		{
			if (std::max(v0rel[i], std::max(v1rel[i], v2rel[i])) < -boxHalfSize[i] ||
				std::min(v0rel[i], std::min(v1rel[i], v2rel[i])) > boxHalfSize[i]) {
				return false; // Separating axis found
			}
		}

		// Triangle normal
		Vector3T triNormal = edge1.CrossProduct(edge2);
		if (!(triNormal == 0))
		{
			triNormal.NormalizeSafe();
			if (!EXUmath::TriangleOverlapOnAxis(triNormal, v0rel, v1rel, v2rel, boxHalfSize)) { return false; }
		}

		// Test cross products of box axes and triangle edges
		std::array<Vector3T, 3> boxAxes = { Vector3T({1, 0, 0}), Vector3T({0, 1, 0}), Vector3T({0, 0, 1}) };
		std::array<Vector3T, 3> edges = { edge1, edge2, edge3 };

		for (const auto& edge : edges)
		{
			for (const auto& axis : boxAxes)
			{
				Vector3T testAxis = axis.CrossProduct(edge);
				T axisLength = testAxis.GetL2Norm();
				if (axisLength > 1e-12)
				{
					if (!EXUmath::TriangleOverlapOnAxis(((T)1. / axisLength) * testAxis, v0rel, v1rel, v2rel, boxHalfSize))
					{
						return false;
					}
				}
			}
		}

		return true; // No separating axis found, intersection exists
	}


	//! @brief Output operator for Box3D
	friend std::ostream& operator<<(std::ostream& os, const Box3DBase& v)
	{
		os << "{" << v.PMin() << ", " << v.PMax() << "}";
		return os;
	}

private:
	T pmin[3];
	T pmax[3];
};

typedef Box3DBase<Real> Box3D; 


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//Zero-based!!!!
//generate a searchtree with a physical size of Box3D b
//all items must fit into this box and should be equally distributed
//then use AddItems to add items with a bounding box and an identifier
//GetItemsInBox() gives you all identifiers which have a bounding box within the specified box
template<typename T>
class SearchTreeBase
{
private:
	using Box3T = Box3DBase<T>;
	using Vector3T = SlimVectorBase<T, 3>;

	Index sx, sy, sz;
	ArrayIndex* data;
	Box3T box;

public:
	SearchTreeBase(): sx(0), sy(0), sz(0), data(nullptr) {};

	bool isEmpty()
	{
		return TotalSize() == 0;
	}

	~SearchTreeBase()
	{
		Flush();
	}

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//! empty all data and reset cells
	void Flush()
	{
		if (data)
		{
			FlushCells();
			delete[] data;
			data = nullptr;
		}
	}

	//! erase memory of cells, but keep search tree
	void FlushCells()
	{
		if (data)
		{
			for (Index i = 0; i < TotalSize(); i++)
			{
				data[i].Flush();
			}
		}
	}

	//! get searchtree box
	const Box3T& GetBox() const { return box; }
	Box3T& GetBox() { return box; }

	void GetCellLengths(T& Lx, T& Ly, T& Lz) const
	{ 
		CHECKandTHROW(sx+sy+sz > 0, "SearchTree/GeneralContact: number of cells may not be zero for any dimension");
		Lx = box.SizeX() / sx;
		Ly = box.SizeY() / sy;
		Lz = box.SizeZ() / sz;
	}

	//! get box for a specific bin
	Box3T GetBinBox(Index ix, Index iy, Index iz) const
	{
		T Lx, Ly, Lz;
		GetCellLengths(Lx, Ly, Lz);
		return Box3T(Vector3T({
			box.PMinX() + ix * Lx,
			box.PMinY() + iy * Ly,
			box.PMinZ() + iz * Lz }),
			Vector3T({
			box.PMinX() + (ix + 1) * Lx,
			box.PMinY() + (iy + 1) * Ly,
			box.PMinZ() + (iz + 1) * Lz }));
	}

	Index SizeX() const { return sx; }
	Index SizeY() const { return sy; }
	Index SizeZ() const { return sz; }
	Index TotalSize() const { return sx*sy*sz; }

	Index3 NumberOfCellsXYZ() const { return Index3({ sx, sy, sz }); }
	Index NumberOfCells() const { return sx*sy*sz; }

	void ResetSearchTree(Index sizex, Index sizey, Index sizez, Box3T b)
	{
		ClearItems(); //empty all items
		box = b;
		CHECKandTHROW(box.SizeX()*box.SizeY()*box.SizeZ() > 0, "SearchTree: size of box must be not equal 0");

		if (sx != sizex || sy != sizey || sz != sizez)
		{
			if (data)
			{
				for (Index i = 0; i < TotalSize(); i++)
				{
					data[i].Flush();
				}
				delete [] data;
			}

			sx = sizex;
			sy = sizey;
			sz = sizez;

			data = new ArrayIndex[TotalSize()]();
		}
	}

	//return 6 indices for box: minx, maxx, miny, maxy, minz, maxz
	void GetBoxIndizes(const Box3T& b, Index6& ind) const
	{
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
	}

	//return items in box defined by 6 indices: minx, maxx, miny, maxy, minz, maxz
	void GetItemsInBox(Index6& ind, ArrayIndex& items, bool resetItems = true) const
	{
		CHECKandTHROW(data != 0, "GetItemsInBox: data=0");

		if (resetItems) { items.SetNumberOfItems(0); }
		Index ld;
		for (Index ix = ind[0]; ix <= ind[1]; ix++) //<= ind is correct, as it contains at most (sx-1)
		{
			for (Index iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (Index iz = ind[4]; iz <= ind[5]; iz++)
				{
					ld = data[GlobalIndex(ix, iy, iz)].NumberOfItems();
					for (Index i = 0; i < ld; i++)
					{
						items.AppendPure((data[GlobalIndex(ix, iy, iz)])[i]);
					}
				}
			}
		}
	}

	//add items in box defined by 6 indices: minx, maxx, miny, maxy, minz, maxz
	//does not reset items list
	void AddItemsInBox(Index6& ind, ArrayIndex& items) const
	{
		CHECKandTHROW(data != 0, "AddItemsInBox: data=0");

		Index ld;
		for (Index ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (Index iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (Index iz = ind[4]; iz <= ind[5]; iz++)
				{
					ld = data[GlobalIndex(ix, iy, iz)].NumberOfItems();
					for (Index i = 0; i < ld; i++)
					{
						items.AppendPure((data[GlobalIndex(ix, iy, iz)])[i]);
					}
				}
			}
		}
	}

	//get only items in box
	void GetItemsInBox(const Box3T& b, ArrayIndex& items, bool resetItems = true) const
	{
		if (resetItems) { items.SetNumberOfItems(0); }
		Index ind[6];
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
		Index ix, iy, iz, i;
		ArrayIndex* id;

		for (ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (iz = ind[4]; iz <= ind[5]; iz++)
				{
					id = &data[GlobalIndex(ix, iy, iz)];
					for (i = 0; i < id->NumberOfItems(); i++)
					{
						items.AppendPure(id->GetItemUnsafe(i));
					}
				}
			}
		}
	}

	//get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	//leave out items with index >= maxIndex or index < minIndex
	void GetSingleItemsInBoxMaxMinIndex(const Box3T& b, ArrayIndex& items, ResizableArray<bool>& indexFlags, 
		Index maxIndex, Index minIndex = 0, bool clearIndexFlags = true, bool resetItems = true) const
	{
		if (resetItems) { items.SetNumberOfItems(0); }
		Index ind[6];
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
		Index ix, iy, iz, i;
		ArrayIndex* id;

		for (ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (iz = ind[4]; iz <= ind[5]; iz++)
				{
					id = &data[GlobalIndex(ix, iy, iz)];
					for (i = 0; i < id->NumberOfItems(); i++)
					{
						Index newItem = id->GetItemUnsafe(i);
						if (!indexFlags[newItem] && (newItem < maxIndex) && (newItem >= minIndex))
						{
							items.AppendPure(newItem);
							indexFlags[newItem] = true;
						}
					}
				}
			}
		}
		//reset indexFlags back to original state, containing only false
		if (clearIndexFlags)
		{
			for (Index i : items)
			{
				indexFlags[i] = false;
			}
		}
	}
	
	//!get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	//!leave out items with index >= maxIndex or index < minIndex
	//!by supplying the pre-computed bounding boxes of items, only items are considered, which really intersect with b
	void GetSingleItemsInBoxMaxMinIndex(const Box3T& b, ArrayIndex& items, ResizableArray<bool>& indexFlags,
		const ResizableArray<Box3T>& allBoundingBoxes, Index maxIndex, Index minIndex = 0, bool clearIndexFlags = true, 
		bool resetItems = true) const
	{
		if (resetItems) { items.SetNumberOfItems(0); }
		Index ind[6];
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
		Index ix, iy, iz, i;
		ArrayIndex* id;

		for (ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (iz = ind[4]; iz <= ind[5]; iz++)
				{
					id = &data[GlobalIndex(ix, iy, iz)];
					for (i = 0; i < id->NumberOfItems(); i++)
					{
						Index newItem = id->GetItemUnsafe(i);
						if (!indexFlags[newItem] && 
							(newItem < maxIndex) && (newItem >= minIndex) &&
							b.Intersect(allBoundingBoxes[newItem]))
						{
							items.AppendPure(newItem);
							indexFlags[newItem] = true;
						}
					}
				}
			}
		}
		//reset indexFlags back to original state, containing only false
		if (clearIndexFlags)
		{
			for (Index i : items)
			{
				indexFlags[i] = false;
			}
		}
	}

	//get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	void GetSingleItemsInBox(const Box3T& b, ArrayIndex& items, ResizableArray<bool>& indexFlags, 
		bool clearIndexFlags=true, bool resetItems = true) const
	{
		if (resetItems) { items.SetNumberOfItems(0); }
		Index ind[6];
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
		Index ix, iy, iz, i;
		ArrayIndex* id;

		for (ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (iz = ind[4]; iz <= ind[5]; iz++)
				{
					id = &data[GlobalIndex(ix, iy, iz)];
					for (i = 0; i < id->NumberOfItems(); i++)
					{
						Index newItem = id->GetItemUnsafe(i);
						if (!indexFlags[newItem])
						{
							items.AppendPure(newItem);
							indexFlags[newItem] = true;
						}
					}
				}
			}
		}
		//reset indexFlags back to original state, containing only false
		if (clearIndexFlags)
		{
			for (Index i : items)
			{
				indexFlags[i] = false;
			}
		}
	}

	//get items in box, do not reset items list
	void AddItemsInBox(const Box3T& b, ArrayIndex& items) const
	{
		Index ind[6];
		ind[0] = IndX(b.PMinX());
		ind[1] = IndX(b.PMaxX());
		ind[2] = IndY(b.PMinY());
		ind[3] = IndY(b.PMaxY());
		ind[4] = IndZ(b.PMinZ());
		ind[5] = IndZ(b.PMaxZ());
		Index ix, iy, iz, i;
		ArrayIndex* id;

		for (ix = ind[0]; ix <= ind[1]; ix++)
		{
			for (iy = ind[2]; iy <= ind[3]; iy++)
			{
				for (iz = ind[4]; iz <= ind[5]; iz++)
				{
					id = &data[GlobalIndex(ix, iy, iz)];
					for (i = 0; i < id->NumberOfItems(); i++)
					{
						items.AppendPure(id->GetItem(i));
					}
				}
			}
		}
	}

	//! get number of items in box with (global) index
	Index NumberOfItemsInBox(Index globalBinIndex) const
	{
		return data[globalBinIndex].NumberOfItems();
	}

	////! get all items of the box with global index ind
	//void GetItemsOfBox(Index ind, ArrayIndex& items) const
	//{
	//	items = data[ind];
	//}

	////! get all items of the box in which the point 'pos' lies
	//void GetItemsOfBox(const Vector3T& pos, ArrayIndex& items) const
	//{
	//	GetItemsOfBox(GlobalIndex(IndX(pos.X()), IndY(pos.Y()), IndZ(pos.Z())), items);
	//}

	//add all items of the box with index ind
	void AddItemsOfBox(Index globalBinIndex, ArrayIndex& items) const
	{
		items.AppendArray(data[globalBinIndex]);
	}

	//! add all items of the box in which the point lies
	void AddItemsOfPoint(const Vector3T& pos, ArrayIndex& items) const
	{
		AddItemsOfBox(GlobalIndex(IndX(pos.X()), IndY(pos.Y()), IndZ(pos.Z())), items);
	}

	//! add item to search tree
	void AddItem(const Box3T& b, Index identifier)
	{
		Index pMinX = IndX(b.PMinX());
		Index pMaxX = IndX(b.PMaxX());
		Index pMinY = IndY(b.PMinY());
		Index pMaxY = IndY(b.PMaxY());
		Index pMinZ = IndZ(b.PMinZ());
		Index pMaxZ = IndZ(b.PMaxZ());
		//Index gi = iX + iY * sx + iZ * (sx*sy);
		Index sxsy = sx * sy;
		Index gi;
		for (Index iz = pMinZ; iz <= pMaxZ; iz++)
		{
			gi = pMinY * sx + iz * sxsy;
			for (Index iy = pMinY; iy <= pMaxY; iy++)
			{
				for (Index ix = pMinX; ix <= pMaxX; ix++)
				{
					//data[GlobalIndex(ix, iy, iz)].AppendPure(identifier);
					data[gi+ix].AppendPure(identifier);
				}
				gi += sx;
			}
		}
	}

	//! add triangle item only if triangle really intersects with bins
	void AddItemTriangle(const Box3T& b, Index identifier, const Vector3T& v0, const Vector3T& v1, const Vector3T& v2)
	{
		Index pMinX = IndX(b.PMinX());
		Index pMaxX = IndX(b.PMaxX());
		Index pMinY = IndY(b.PMinY());
		Index pMaxY = IndY(b.PMaxY());
		Index pMinZ = IndZ(b.PMinZ());
		Index pMaxZ = IndZ(b.PMaxZ());
		//Index gi = iX + iY * sx + iZ * (sx*sy);
		Index sxsy = sx * sy;
		Index gi;
		for (Index iz = pMinZ; iz <= pMaxZ; iz++)
		{
			gi = pMinY * sx + iz * sxsy;
			for (Index iy = pMinY; iy <= pMaxY; iy++)
			{
				for (Index ix = pMinX; ix <= pMaxX; ix++)
				{
					Box3T binBox = GetBinBox(ix, iy, iz);
					if (binBox.BoxTriangleIntersect(v0, v1, v2))
					{
						data[gi + ix].AppendPure(identifier);
					}
				}
				gi += sx;
			}
		}

	}

	//! return x-index for a T x-value in Box
	Index IndX(T x) const
	{
		//(Index) would behave wrong between -1 to +1: (int)0.8 => 0, (int)(-0.8) => 0
		//        floor(...) corrects this: floor(0.8) => 0, fllor(-0.8) => -1
		Index i = (Index)floor(((x - box.PMinX())*(T)sx) / box.SizeX());
		if (i < 0) { i = 0; }
		if (i >= sx) { i = sx - 1; }
		return i;
	}

	//! return y-index for a T y-value in Box
	Index IndY(T y) const
	{
		Index i = (Index)floor(((y - box.PMinY())*(T)sy) / box.SizeY());
		if (i < 0) { i = 0; }
		if (i >= sy) { i = sy - 1; }
		return i;
	}

	//! return z-index for a T z-value in Box
	Index IndZ(T z) const
	{
		Index i = (Index)floor(((z - box.PMinZ())*(T)sz) / box.SizeZ());
		if (i < 0) { i = 0; }
		if (i >= sz) { i = sz - 1; }
		return i;
	}

	//! return x/y/z index for T value and given axis in [0,1,2] corresponding to [x,y,z]
	Index IndexOfReal(T value, Index axis)
	{
		if (axis == 0) { return IndX(value); }
		else if (axis == 1) { return IndY(value); }
		else if (axis == 2) { return IndZ(value); }
		else { CHECKandTHROWstring("SearchTree::IndexOfReal called with invalid axis"); return 0; }
	}

	//! get index in 3D list from 3 indices
	//! could be improved, especially by storing sx*sy
	Index GlobalIndex(Index iX, Index iY, Index iZ) const
	{
		Index gi = iX + iY * sx + iZ * (sx*sy);
		//not necessary lateron, as IndX, etc. add checks!
		//CHECKandTHROW((gi < TotalSize()) && (gi >= 0), "GlobalIndex: computation of global index failed");
		return gi;
	}

	//! empty searchtree, but keep memory
	void ClearItems()
	{
		for (Index i = 0; i < TotalSize(); i++)
		{
			data[i].SetNumberOfItems(0);
		}
	}

	void GetStatistics(Index& numberOfTreeItems, T& averageFill, Index& numberOfZeros, Index& maxFill, Index& numberOf10average) const
	{
		numberOfTreeItems = 0;
		averageFill = -1;
		numberOf10average = -1;
		maxFill = 0;
		numberOfZeros = 0;
		for (Index i = 0; i < TotalSize(); i++)
		{
			Index value = data[i].NumberOfItems();
			numberOfTreeItems += value;
			maxFill = EXUstd::Maximum(value, maxFill);
			if (value == 0) { numberOfZeros++; }
		}
		if (TotalSize() != 0)
		{
			averageFill = (T)numberOfTreeItems / (T)TotalSize();
			numberOf10average = 0;
			for (Index i = 0; i < TotalSize(); i++)
			{
				if (data[i].NumberOfItems() > (Index)(10. * averageFill))
				{
					numberOf10average++;
				}
			}
		}

	}

	//! get all bin indices that are crossed by a line with startPoint and normalized direction
	void GetBinsCrossedByLine(const Vector3T& startPoint, const Vector3T& direction, ArrayIndex& globalBinIndices) const
	{
		globalBinIndices.SetNumberOfItems(0);

		const Box3T& b = this->box;

		// Ray-box intersection using slab method
		T tmin = 0;
		T tmax = std::numeric_limits<T>::max();

		for (int i = 0; i < 3; ++i)
		{
			if (std::abs(direction[i]) < (T)1e-15)
			{
				if (startPoint[i] < b.PMinC()[i] || startPoint[i] > b.PMaxC()[i])
					return; // Ray parallel and outside slab
			}
			else
			{
				T invD = (T)1. / direction[i];
				T t0 = (b.PMinC()[i] - startPoint[i]) * invD;
				T t1 = (b.PMaxC()[i] - startPoint[i]) * invD;
				if (t0 > t1) std::swap(t0, t1);
				tmin = std::max(tmin, t0);
				tmax = std::min(tmax, t1);
				if (tmin > tmax) return; // No intersection
			}
		}

		// Start position inside box
		Vector3T entryPoint = startPoint + direction * tmin;

		// Cell sizes
		T cellSizeX = b.SizeX() / (T)sx;
		T cellSizeY = b.SizeY() / (T)sy;
		T cellSizeZ = b.SizeZ() / (T)sz;

		// Starting voxel
		Index ix = IndX(entryPoint[0]);
		Index iy = IndY(entryPoint[1]);
		Index iz = IndZ(entryPoint[2]);

		Index stepX = (direction[0] > 0) ? 1 : (direction[0] < 0) ? -1 : 0;
		Index stepY = (direction[1] > 0) ? 1 : (direction[1] < 0) ? -1 : 0;
		Index stepZ = (direction[2] > 0) ? 1 : (direction[2] < 0) ? -1 : 0;

		auto computeInitial = [](T coord, T dir, T minCoord, T cellSize, Index i, Index step) -> std::pair<T, T> {
			if (dir == 0) return { std::numeric_limits<T>::max(), std::numeric_limits<T>::max() };
			T nextBoundary = minCoord + ((step > 0) ? (i + 1) * cellSize : i * cellSize);
			T tMax = (nextBoundary - coord) / dir;
			T tDelta = cellSize / std::abs(dir);
			return { tMax, tDelta };
			};

		auto [tMaxX, tDeltaX] = computeInitial(entryPoint[0], direction[0], b.PMinX(), cellSizeX, ix, stepX);
		auto [tMaxY, tDeltaY] = computeInitial(entryPoint[1], direction[1], b.PMinY(), cellSizeY, iy, stepY);
		auto [tMaxZ, tDeltaZ] = computeInitial(entryPoint[2], direction[2], b.PMinZ(), cellSizeZ, iz, stepZ);

		// Traverse until the ray exits the box
		while (ix >= 0 && ix < sx && iy >= 0 && iy < sy && iz >= 0 && iz < sz)
		{
			globalBinIndices.AppendPure(GlobalIndex(ix, iy, iz));

			if (tMaxX < tMaxY)
			{
				if (tMaxX < tMaxZ) {
					if (tMaxX > tmax) break;
					ix += stepX;
					tMaxX += tDeltaX;
				}
				else {
					if (tMaxZ > tmax) break;
					iz += stepZ;
					tMaxZ += tDeltaZ;
				}
			}
			else
			{
				if (tMaxY < tMaxZ) {
					if (tMaxY > tmax) break;
					iy += stepY;
					tMaxY += tDeltaY;
				}
				else {
					if (tMaxZ > tmax) break;
					iz += stepZ;
					tMaxZ += tDeltaZ;
				}
			}
		}
	}

	//! collect items 
	void GetUniqueItemsFromBins(const ArrayIndex& globalBinIndices, ArrayIndex& items,
		ResizableArray<bool>& indexFlags,
		bool clearIndexFlags = true, bool resetItems = true) const
	{
		if (resetItems) {
			items.SetNumberOfItems(0);
		}

		for (Index binIndex : globalBinIndices)
		{
			const ArrayIndex& binItems = data[binIndex];
			for (Index i = 0; i < binItems.NumberOfItems(); ++i)
			{
				Index item = binItems.GetItemUnsafe(i);
				if (!indexFlags[item])
				{
					items.AppendPure(item);
					indexFlags[item] = true;
				}
			}
		}

		// Reset flags if requested
		if (clearIndexFlags)
		{
			for (Index i : items)
			{
				indexFlags[i] = false;
			}
		}
	}

	//! collect items WITH DUPLICATES
	void GetItemsFromBins(const ArrayIndex& globalBinIndices, ArrayIndex& items, bool resetItems = true) const
	{
		if (resetItems) {
			items.SetNumberOfItems(0);
		}

		for (Index binIndex : globalBinIndices)
		{
			const ArrayIndex& binItems = data[binIndex];
			for (Index i = 0; i < binItems.NumberOfItems(); ++i)
			{
				Index item = binItems.GetItemUnsafe(i);
				items.AppendPure(item);
			}
		}
	}

	//! helper function to well estimate searchtree size
	static void ComputeSearchTreeBinCounts(Index nItems, const Box3T& box,
		Index& sx, Index& sy, Index& sz, Index maxBins = 100, Index minBins=2)
	{
		T sizeX = box.SizeX();
		T sizeY = box.SizeY();
		T sizeZ = box.SizeZ();
	
		// Normalize size ratios
		T volume = sizeX * sizeY * sizeZ;
		if (volume <= (T)0.0) {
			sx = sy = sz = minBins;
			return;
		}

		// Target total bin count
		Index targetBins = EXUstd::Maximum(1, nItems);

		// Compute scaling factors
		T scaleX = sizeX / std::cbrt(volume);
		T scaleY = sizeY / std::cbrt(volume);
		T scaleZ = sizeZ / std::cbrt(volume);

		// Compute unbounded bin sizes based on relative volume
		T totalScale = scaleX * scaleY * scaleZ;
		T scaleFactor = std::pow((T)targetBins / totalScale, (T)(1.0 / 3.0));

		// Initial floating bin counts
		T binX = scaleX * scaleFactor;
		T binY = scaleY * scaleFactor;
		T binZ = scaleZ * scaleFactor;

		// Clamp to 1...100
		sx = EXUstd::Maximum(minBins, EXUstd::Minimum(maxBins, (Index)(binX + (T)0.5)));
		sy = EXUstd::Maximum(minBins, EXUstd::Minimum(maxBins, (Index)(binY + (T)0.5)));
		sz = EXUstd::Maximum(minBins, EXUstd::Minimum(maxBins, (Index)(binZ + (T)0.5)));

	}};

typedef SearchTreeBase<Real> SearchTree;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//! a search tree containing a static and a dynamic search tree for higher efficiency in contact search;
//! it is assumed that both trees have same box size and cells, otherwise it would not work!
//! specific functions are referring to staticSearchTree (indices, box size, etc.)
class StaticDynamicSearchTree
{
private:
	SearchTree staticSearchTree;
	SearchTree dynamicSearchTree;

public:
	//StaticDynamicSearchTree {};

	bool isEmpty()
	{
		return staticSearchTree.TotalSize() == 0 && dynamicSearchTree.TotalSize() == 0;
	}

	const SearchTree& GetStaticSearchTree() const { return staticSearchTree; }
	SearchTree& GetStaticSearchTree() { return staticSearchTree; }

	const SearchTree& GetDynamicSearchTree() const { return dynamicSearchTree; }
	SearchTree& GetDynamicSearchTree() { return dynamicSearchTree; }

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//! empty all data and reset cells
	void Flush(bool flushStatic = false)
	{
		if (flushStatic) { staticSearchTree.Flush(); };
		dynamicSearchTree.Flush();
	}

	//! erase memory of cells, but keep search tree
	void FlushCells(bool flushStatic = false)
	{
		if (flushStatic) { staticSearchTree.FlushCells(); }
		dynamicSearchTree.FlushCells();
	}

	const Box3D& GetBox() const { return staticSearchTree.GetBox(); }
	Box3D& GetBox() { return staticSearchTree.GetBox(); }

	Index SizeX() const { return staticSearchTree.SizeX(); }
	Index SizeY() const { return staticSearchTree.SizeY(); }
	Index SizeZ() const { return staticSearchTree.SizeZ(); }
	Index TotalSize() const { return staticSearchTree.TotalSize(); }

	Index3 NumberOfCellsXYZ() const { return staticSearchTree.NumberOfCellsXYZ(); }

	void ResetSearchTree(Index sizex, Index sizey, Index sizez, Box3D b)
	{
		staticSearchTree.ResetSearchTree(sizex, sizey, sizez, b);
		dynamicSearchTree.ResetSearchTree(sizex, sizey, sizez, b);
	}

	//return 6 indices for box: minx, maxx, miny, maxy, minz, maxz
	void GetBoxIndizes(const Box3D& b, Index6& ind) const
	{
		staticSearchTree.GetBoxIndizes(b, ind);
	}

	//return items in box defined by 6 indices: minx, maxx, miny, maxy, minz, maxz
	void GetItemsInBox(Index6& ind, ArrayIndex& items) const
	{
		staticSearchTree.GetItemsInBox(ind, items, true);
		dynamicSearchTree.GetItemsInBox(ind, items, false);
	}

	//add items in box defined by 6 indices: minx, maxx, miny, maxy, minz, maxz
	//does not reset items list
	void AddItemsInBox(Index6& ind, ArrayIndex& items) const
	{
		staticSearchTree.AddItemsInBox(ind, items);
		dynamicSearchTree.AddItemsInBox(ind, items);
	}

	//get only items in box
	void GetItemsInBox(const Box3D& b, ArrayIndex& items) const
	{
		staticSearchTree.GetItemsInBox(b, items);
		dynamicSearchTree.GetItemsInBox(b, items);
	}

	//get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	//leave out items with index >= maxIndex or index < minIndex
	void GetSingleItemsInBoxMaxMinIndex(const Box3D& b, ArrayIndex& items, ResizableArray<bool>& indexFlags,
		Index maxIndex, Index minIndex = 0, bool clearIndexFlags = true) const
	{
		staticSearchTree.GetSingleItemsInBoxMaxMinIndex(b, items, indexFlags, maxIndex, minIndex, clearIndexFlags, true);
		dynamicSearchTree.GetSingleItemsInBoxMaxMinIndex(b, items, indexFlags, maxIndex, minIndex, clearIndexFlags, false);
	}

	//!get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	//!leave out items with index >= maxIndex or index < minIndex
	//!by supplying the pre-computed bounding boxes of items, only items are considered, which really intersect with b
	void GetSingleItemsInBoxMaxMinIndex(const Box3D& b, ArrayIndex& items, ResizableArray<bool>& indexFlags,
		const ResizableArray<Box3D>& allBoundingBoxes, Index maxIndex, Index minIndex = 0, bool clearIndexFlags = true) const
	{
		staticSearchTree.GetSingleItemsInBoxMaxMinIndex(b, items, indexFlags, allBoundingBoxes, maxIndex, minIndex, clearIndexFlags, true);
		dynamicSearchTree.GetSingleItemsInBoxMaxMinIndex(b, items, indexFlags, allBoundingBoxes, maxIndex, minIndex, clearIndexFlags, false);
	}

	//get items in box b; do not add duplicates by using indexFlags array, having one bool per index, all initialized with false
	void GetSingleItemsInBox(const Box3D& b, ArrayIndex& items, ResizableArray<bool>& indexFlags, bool clearIndexFlags = true) const
	{
		staticSearchTree.GetSingleItemsInBox(b, items, indexFlags, clearIndexFlags, true);
		dynamicSearchTree.GetSingleItemsInBox(b, items, indexFlags, clearIndexFlags, false);
	}

	//get items in box, do not reset items list
	void AddItemsInBox(const Box3D& b, ArrayIndex& items) const
	{
		staticSearchTree.AddItemsInBox(b, items);
		dynamicSearchTree.AddItemsInBox(b, items);
	}

	//! get number of items in box with (global) index
	Index NumberOfItemsInBox(Index ind) const
	{
		return staticSearchTree.NumberOfItemsInBox(ind) + dynamicSearchTree.NumberOfItemsInBox(ind);
	}


	//add all items of the box with index ind
	void AddItemsOfBox(Index ind, ArrayIndex& items) const
	{
		staticSearchTree.AddItemsOfBox(ind, items);
		dynamicSearchTree.AddItemsOfBox(ind, items);
	}

	//! add all items of the box in which the point lies
	void AddItemsOfPoint(const Vector3D& pos, ArrayIndex& items) const
	{
		staticSearchTree.AddItemsOfPoint(pos, items);
		dynamicSearchTree.AddItemsOfPoint(pos, items);
	}

	//! add item to search tree
	void AddItem(const Box3D& b, Index identifier, bool useDynamicSearchTree = true)
	{
		if (useDynamicSearchTree) { dynamicSearchTree.AddItem(b, identifier); }
		else { staticSearchTree.AddItem(b, identifier); }
	}
	void AddItemTriangle(const Box3D& b, Index identifier, const Vector3D& v0, const Vector3D& v1, const Vector3D& v2, bool useDynamicSearchTree = true)
	{
		if (useDynamicSearchTree) { dynamicSearchTree.AddItemTriangle(b, identifier, v0, v1, v2); }
		else { staticSearchTree.AddItemTriangle(b, identifier, v0, v1, v2); }
	}

	//! return x-index for a Real x-value in Box
	Index IndX(Real x) const
	{
		return staticSearchTree.IndX(x);
	}

	//! return y-index for a Real y-value in Box
	Index IndY(Real y) const
	{
		return staticSearchTree.IndY(y);
	}

	//! return z-index for a Real z-value in Box
	Index IndZ(Real z) const
	{
		return staticSearchTree.IndZ(z);
	}

	//! return x/y/z index for Real value and given axis in [0,1,2] corresponding to [x,y,z]
	Index IndexOfReal(Real value, Index axis)
	{
		return staticSearchTree.IndexOfReal(value, axis);
	}

	//! get index in 3D list from 3 indices
	//! could be improved, especially by storing sx*sy
	Index GlobalIndex(Index iX, Index iY, Index iZ) const
	{
		return staticSearchTree.GlobalIndex(iX, iY, iZ);
	}

	//! empty searchtree, but keep memory
	void ClearItems(bool clearStatic = false)
	{
		if (clearStatic) { staticSearchTree.ClearItems(); };
		dynamicSearchTree.ClearItems();
	}

	void GetStatistics(Index& numberOfTreeItems, Real& averageFill, Index& numberOfZeros, Index& maxFill, Index& numberOf10average) const
	{
		Index numberOfTreeItemsStatic;
		Real averageFillStatic;
		Index numberOfZerosStatic;
		Index maxFillStatic;
		Index numberOf10averageStatic;
		staticSearchTree.GetStatistics(numberOfTreeItemsStatic, averageFillStatic, numberOfZerosStatic, maxFillStatic, numberOf10averageStatic);

		dynamicSearchTree.GetStatistics(numberOfTreeItems, averageFill, numberOfZeros, maxFill, numberOf10average);
		
		//compute sum, max or average ...
		numberOfTreeItems += numberOfTreeItemsStatic;
		averageFill += averageFillStatic; averageFill /= 2.;
		numberOfZeros += numberOfZerosStatic;
		maxFill = EXUstd::Maximum(maxFillStatic, maxFill);
		numberOf10average += numberOf10averageStatic;
	}

};



#endif
