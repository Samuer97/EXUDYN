/** ***********************************************************************************************
* @brief		file including test routines, used in Python
* @details		Details:
				- implementation of some test functions of PybindTests.h
*
* @author		Gerstmayr Johannes
* @date			2021-05-02 
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

//++++++++++++++++++++++++++
//#include <iostream>
//#include <cstdlib>
//#include <array>
//#include <vector>
//#include <signal.h>
////++++++++++++++++++++++++++

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
//#include <pybind11/operators.h>
#include <pybind11/numpy.h>       //interface to numpy
//#include <pybind11/buffer_info.h> //passing reference to matrix to numpy
#include <pybind11/embed.h>       //scoped interpreter
////does not work globally: #include <pybind11/iostream.h> //used to redirect cout:  py::scoped_ostream_redirect output;
//#include <pybind11/cast.h> //for arguments
//#include <pybind11/functional.h> //for function handling ... otherwise gives a python error (no compilation error in C++ !)
namespace py = pybind11;
//using namespace pybind11::literals; //brings in the '_a' literals; e.g. for short arguments definition


#include "Pymodules/PybindTests.h"

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//test functions, will be eliminated from time to time
#include "Main/OutputVariable.h"

#include "Linalg/BasicLinalg.h"
#include <initializer_list>
#include <vector>

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//for Eigen tests:
//#include "../Eigen/Sparse"
//#include "../Eigen/Dense"
//#include "../Eigen/Eigen"
//#include "../Eigen/src/Eigenvalues/ComplexEigenSolver.h"
//#include "ngs-core-master/ngs_core.hpp"

//#include "Utilities/Parallel.h" //include after 

void PyTest()
{
	if (1)
	{
		Vector4D c({ 3., 7.5, 2., 0.3 });
		auto xList = { -4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4. };

		for (Real x : xList)
		{
			pout << "x=" << x
				<< ", P(x) =" << EXUmath::EvaluatePolynomial(c, x)
				<< ", P'(x)=" << EXUmath::EvaluatePolynomial_x(c, x) << "\n";
		}
	}
	////contact testing
	//if (1)
	//{
	//	//using namespace Eigen;
	//	STDstring endl = "\n";
	//	SlimVector<7> p;
	//	//p.SetVector({ 0.1, -0.1, 0.01, -0.01, 0.9225, -0.36, -0.1025 });
	//	p.SetVector({5.1200000000000045, -8.0, 6.489999999999995, -1.2500000000000004, -0.057500000000000134, -0.15, -0.04250000000000009});
	//	pout << p << "\n";

	//	if (0)
	//	{
	//		Eigen::MatrixXcf A = Eigen::MatrixXcf::Random(4, 4);
	//		pout << "Here is a random 4x4 matrix, A:" << endl << A << endl << endl;

	//		Eigen::ComplexEigenSolver<Eigen::MatrixXcf> ces;
	//		ces.compute(A);
	//		pout << "The eigenvalues of A are:" << endl << ces.eigenvalues() << endl;
	//		pout << "The matrix of eigenvectors, V, is:" << endl << ces.eigenvectors() << endl << endl;

	//		std::complex<float> lambda = ces.eigenvalues()[0];
	//		pout << "Consider the first eigenvalue, lambda = " << lambda << endl;
	//		Eigen::VectorXcf v = ces.eigenvectors().col(0);
	//		pout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
	//		pout << "... and A * v = " << endl << A * v << endl << endl;
	//	}
	//	if (1)
	//	{
	//		Real c0 = p[6];
	//		Real c1 = p[5];
	//		Real c2 = p[4];
	//		Real c3 = p[3];
	//		Real c4 = p[2];
	//		Real c5 = p[1];
	//		Real c6 = p[0];
	//		typedef Eigen::Matrix<Eigen::dcomplex, 6, 6> EigMatrix66c;
	//		typedef std::complex<Real> RealC;
	//		EigMatrix66c A;
	//		A.setZero();
	//		A(0, 5) = RealC(-c0 / c6, 0);
	//		A(1, 5) = RealC(-c1 / c6, 0);
	//		A(2, 5) = RealC(-c2 / c6, 0);
	//		A(3, 5) = RealC(-c3 / c6, 0);
	//		A(4, 5) = RealC(-c4 / c6, 0);
	//		A(5, 5) = RealC(-c5 / c6, 0);
	//		A(1, 0) = RealC(1., 0.);
	//		A(2, 1) = RealC(1., 0.);
	//		A(3, 2) = RealC(1., 0.);
	//		A(4, 3) = RealC(1., 0.);
	//		A(5, 4) = RealC(1., 0.);
	//		//A = [[0, 0, 0, 0, 0, -c0 / c6], 
	//		//	[1, 0, 0, 0, 0, -c1 / c6], 
	//		//	[0, 1, 0, 0, 0, -c2 / c6], 
	//		//	[0, 0, 1, 0, 0, -c3 / c6], 
	//		//	[0, 0, 0, 1, 0, -c4 / c6], 
	//		//	[0, 0, 0, 0, 1, -c5 / c6]]
	//		Real t = -EXUstd::GetTimeInSeconds();
	//		Index n = 1000000;
	//		Eigen::ComplexEigenSolver<EigMatrix66c> ces;
	//		ces.compute(A, false); //27microsecs !
	//		//for (Index i = 0; i < n; i++)
	//		//{
	//		//	ces.compute(A); //27microsecs on surface!
	//		//}
	//		ngstd::TaskManager::SetNumThreads(1);
	//		//i9:
	//		//1 thread:   21micros
	//		//12 threads: 2micros
	//		//14 threads: 1.85micros
	//		//26 threads: 1.62micros
	//		//surface:
	//		//1 thread : 20.7micros (computeEigenvectors=true)
	//		//1 thread : 16.6micros (computeEigenvectors=false)
	//		//8 threads: 4.69micros (computeEigenvectors=true)
	//		//8 threads: 3.79micros (computeEigenvectors=false)
	//		Index taskmanagerNumthreads = ngstd::EnterTaskManager(); //this is needed in order that any ParallelFor is executed in parallel during solving
	//		NGSsizeType nItems = (NGSsizeType)n;
	//		ngstd::ParallelFor(nItems, [&A, &nItems](NGSsizeType j)
	//		{
	//			Eigen::ComplexEigenSolver<EigMatrix66c> ces;
	//			ces.compute(A, false); //27microsecs !
	//		}, 8*20);
	//		t += EXUstd::GetTimeInSeconds();
	//		ngstd::ExitTaskManager(taskmanagerNumthreads);
	//		pout << "Eigenvalues needed " << t / (Real)n * 1e6 << " microsecs\n\n";
	//		pout << "The eigenvalues of A are:" << endl << ces.eigenvalues() << endl;

	//	}
	//}
	//add testing here
	/*
	if (0)
	{
		pout << "test vectorized Vector add\n";
		Index n = 20;
		Vector v1(n);
		Vector v2(n);
		Vector result(n);
		LinkedDataVector r(result, 0, std::min(20,n));
		for (Index i = 0; i < n; i++)
		{
			v1[i] = i * 1;
			v2[i] = i * 2;
		}

		Index runs = 500000000;
		//AVX vs. single threaded:
		//speedup for n=502: 3.1  (varies !)
		//speedup for n=1002: 3.5 (varies !)
		//speedup for n=5002: 1.7
		//speedup for n=10002: 1.33
		//speedup for n=20002: ~0.9 (no speedup!)
		//speedup for n=40002: ~0.9 (no speedup!)
		//++++++++++++++++++++++++++++++++++++
		//SurfacePro (4 core), 4 threads:
		//16*1e8:      AVX=0.2, MT=136.3, MTAVX=149.3, serial=1.35
		//1002*1e7:    AVX=0.95, MT=11.57, MTAVX=10.62, serial=8.3
		//2002*5e6:    AVX=2.66, MT=7.53, MTAVX=7.12, serial=8.3
		//4002*25e5:   AVX=2.19, MT=5.40, MTAVX=4.50, serial=8.25
		//10002*1e6:   AVX=2.40, MT=3.84, MTAVX=2.88, serial=8.3
		//20002*5e5:   AVX=4.33, MT=3.38, MTAVX=2.61, serial=8.28

		//50002*2e5:   AVX=4.07, MT=3.08, MTAVX=2.54, serial=8.23
		//200002*5e4:  AVX=4.88, MT=3.02, MTAVX=2.46, serial=16.9
		//1000002*1e4: AVX=15.01 MT=13.13, MTAVX=13.51, serial=17.27
		//SurfacePro (4 core), 8 threads:
		//100002*1e5:  AVX=4.13, MT=2.06, MTAVX=1.65, serial=8.23
		//200002*5e4:  AVX=4.46, MT=4.21, MTAVX=3.15, serial=16.92
		Index nThreads = 4;
		ngstd::TaskManager::SetNumThreads(nThreads);

		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		Real time = -EXUstd::GetTimeInSeconds();
		for (Index k = 0; k < runs; k++)
		{
			for (Index i = 0; i < n; i++)
			{
				result.GetUnsafe(i) = v1.GetUnsafe(i) + v2.GetUnsafe(i);
			}
		}
		time += EXUstd::GetTimeInSeconds();
		pout << "time   =" << time << "\n";
		pout << "checksum   =" << (Index)result.Sum() << ", result=" << r << "\n";

		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		Real timeAVX = -EXUstd::GetTimeInSeconds();
		for (Index k = 0; k < runs; k++)
		{
			constexpr Index loopUnrollExpN = 4; //unrolling size = 2^loopUnrollExpN
			Index nShift = n >> loopUnrollExpN;
			PReal* v1p = (PReal*)v1.GetDataPointer();
			PReal* v2p = (PReal*)v2.GetDataPointer();
			PReal* resultp = (PReal*)result.GetDataPointer();
			for (Index i = 0; i < (4*nShift); i+=4)
			{
				//resultp[i] = v1p[i] + v2p[i];
				//resultp[i+1] = v1p[i+1] + v2p[i+1];
				//optimize for latency; using 4 PReal speeds up only slightly
				PReal a = v1p[i] + v2p[i];
				PReal b = v1p[i + 1] + v2p[i + 1];
				resultp[i] = a;
				PReal c = v1p[i + 2] + v2p[i + 2];
				resultp[i + 1] = b;
				PReal d = v1p[i + 3] + v2p[i + 3];
				resultp[i + 2] = c;
				resultp[i + 3] = d;
			}
			for (Index i = nShift << loopUnrollExpN; i < n; i++)
			{
				result.GetUnsafe(i) = v1.GetUnsafe(i) + v2.GetUnsafe(i);
			}
		}
		timeAVX += EXUstd::GetTimeInSeconds();
		pout << "timeAVX=" << timeAVX << "\n";
		pout << "checksumAVX=" << (Index)result.Sum() << ", resultAVX=" << r << "\n";
		
		Index taskmanagerNumthreads = ngstd::EnterTaskManager(); //this is needed in order that any ParallelFor is executed in parallel during solving

		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		Real timeMT = -EXUstd::GetTimeInSeconds();
		for (Index k = 0; k < runs; k++)
		{
			//for (Index i = 0; i < n; i++)
			ngstd::ParallelFor((NGSsizeType)n, [&result, &v1, &v2](NGSsizeType i)
			{
				result.GetUnsafe((Index)i) = v1.GetUnsafe((Index)i) + v2.GetUnsafe((Index)i);
			}, nThreads);
		}

		timeMT += EXUstd::GetTimeInSeconds();
		pout << "timeMT=" << timeMT << "\n";
		pout << "checksumMT=" << (Index)result.Sum() << ", resultMT=" << r << "\n";

		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		Real timeMTAVX = -EXUstd::GetTimeInSeconds();
		for (Index k = 0; k < runs; k++)
		{
			constexpr Index loopUnrollExpN = 4; //unrolling size = 2^loopUnrollExpN
			Index nShift = n >> loopUnrollExpN;
			PReal* v1p = (PReal*)v1.GetDataPointer();
			PReal* v2p = (PReal*)v2.GetDataPointer();
			PReal* resultp = (PReal*)result.GetDataPointer();

			//for (Index i = 0; i < (4 * nShift); i += 4)
			ngstd::ParallelFor((NGSsizeType)(nShift), [&resultp, &v1p, &v2p](NGSsizeType j)
			{
				Index i = (Index)j * 4; //*4
				//resultp[i] = v1p[i] + v2p[i];
				//resultp[i+1] = v1p[i+1] + v2p[i+1];
				//optimize for latency; using 4 PReal speeds up only slightly
				PReal a = v1p[i] + v2p[i];
				PReal b = v1p[i + 1] + v2p[i + 1];
				resultp[i] = a;
				PReal c = v1p[i + 2] + v2p[i + 2];
				resultp[i + 1] = b;
				PReal d = v1p[i + 3] + v2p[i + 3];
				resultp[i + 2] = c;
				resultp[i + 3] = d;
			}, nThreads);
			for (Index i = nShift << loopUnrollExpN; i < n; i++)
			{
				result.GetUnsafe(i) = v1.GetUnsafe(i) + v2.GetUnsafe(i);
			}

		}
		timeMTAVX += EXUstd::GetTimeInSeconds();
		pout << "timeMTAVX=" << timeMTAVX << "\n";
		pout << "checksumMTAVX=" << (Index)result.Sum() << ", resultMTAVX=" << r << "\n";

		ngstd::ExitTaskManager(taskmanagerNumthreads);

	}*/
	
	if (0)
	{
		Vector v1({ 1.,2.,3. });
		Vector v2({ 8.,7.,6. });
		Matrix m(3, 2, { 1,2,3, 4,5,6 });
		Vector r({ 0.,1. });

		EXUmath::MultMatrixVectorAdd(m, v1, r);
		pout << "r1=" << r << "\n";				//=> (14, 33)
		EXUmath::MultMatrixVectorAdd(m, v2, r);
		pout << "r2=" << r << "\n";				//=> (54,136)
	}
	if (0)
	{
		//Matrix m(3, 3, { 1,1,1, 1,1,0, 0,0,1 });
		Matrix m(3, 3, { 1,1,1, 0,1,0, 0,1,0 });
		Matrix minv = m;
		ResizableMatrix temp;
		ArrayIndex rows;
		//minv.Invert();
		Index success = minv.InvertSpecial(temp, rows, true, 0, 0.);
		pout << "success=" << success << "\n";
		//pout << "Minv=" << minv << ", rows=" << rows << ", I=" << minv * m << "\n";
		//pout << "Minv*[1,-2,3]=" << minv * Vector({1,-2,3}) << "\n";

		//minv = m;
		//success = minv.Invert();
		//pout << "success2=" << success << "\n";
		//pout << "Minv2=" << minv << "\n";
	}
	if (0) //test if conversion to sensor and back works
	{
		Index mbsNumber = 0;
		Index i = 0;
		//ItemType myType = ItemType::Sensor;
		ItemType myType = ItemType::Marker;
		Index itemID = Index2ItemID(i, myType, mbsNumber);
		pout << "itemID=" << itemID << "\n";

		Index iNew;
		ItemType typeNew;
		ItemID2IndexType(itemID, iNew, typeNew, mbsNumber);
		pout << "type=" << typeNew << ", index="<< iNew << ", mbs=" << mbsNumber << "\n";
	}
	if (0)
	{
		////! access to internal module
		//void PyGetInternalSysDictionary()
		//{
		//	//internalSystemDictionary["abc"] = 123;
		//	//py::print(internalSystemDictionary["abc"]);
		//	STDstring key = "aaa";
		//	Real item = 1.23;
		//	py::module exudynModule = py::module::import("exudyn");
		//	exudynModule.attr("sys")[key.c_str()] = item;
		//
		//	pout << "test\n";
		//}
	}
	if (0)
	{
		//SlimArray<Real, 3> a({ 3,5,1 });
		//pout << "a=" << a << "\n";
		//a.Sort();
		//pout << "a sorted=" << a << "\n";

	}
}


py::object GetVector()
{
	Vector v({ 42.1234567890123456,43.,44. }); //double precision maintained in NumPy array in python
	return py::array_t<Real>(v.NumberOfItems(), v.GetDataPointer()); //copy array (could also be referenced!)
}

//py::object GetMatrix()
py::array_t<Real> GetMatrix()
{
	ResizableMatrix m(2, 3, { 12.5,13,14,  15,16,17 }); //double precision maintained in NumPy array in python

	return py::array_t<Real>(std::vector<std::ptrdiff_t>{(int)m.NumberOfRows(), (int)m.NumberOfColumns()}, m.GetDataPointer()); //copy array (could also be referenced!)
}

void SeeMatrix(py::array_t<Real> pyArray)
{
	pout << "ndim=" << pyArray.ndim() << "\n";
	pout << "size=" << pyArray.size() << "\n";
	if (pyArray.ndim() == 2)
	{
		auto mat = pyArray.unchecked<2>();
		Index nrows = (Index)mat.shape(0);
		Index ncols = (Index)mat.shape(1);

		Matrix m(nrows, ncols);
		for (Index i = 0; i < nrows; i++)
		{
			for (Index j = 0; j < ncols; j++)
			{
				m(i, j) = mat(i, j);
			}
		}
		pout << "Matrix m=" << m << "\n";
	}
}


//void PythonCreateSys10()
//{
//	py::exec(R"(
//		systemContainer = exu.SystemContainer()
//		mbs = systemContainer.AddSystem()
//		for i in range(10): mbs.AddNode({'nodeType': 'Point','referenceCoordinates': [i, 0.0, 0.0],'initialCoordinates': [0.0, 0.0, (i+1)/10], 'initialVelocities': [0.0, i*2, 0.0],})
//		for i in range(10): mbs.AddObject({'objectType': 'MassPoint', 'physicsMass': 20-i, 'nodeNumber': i})
//		for i in range(10): mbs.AddMarker({'markerType': 'BodyPosition',  'bodyNumber': i,  'localPosition': [0.0, 0.0, 0.0], 'bodyFixed': False})
//		for i in range(9): mbs.AddObject({'objectType': 'SpringDamper', 'stiffness': 4, 'damping': 0, 'force': 0, 'referenceLength':1, 'markerNumbers': [i,i+1]})		
//		mbs
//    )");
//	//pout << "ready to go\n";
//}

//py::object GetIndex()
//{
//	return py::int_(42);
//}
//
//py::object GetString()
//{
//	return py::str("A string");
//}
//
//py::object GetReal()
//{
//	return py::float_(double(42.1234567890123456)); //precision maintained in python: 42.123456789012344
//}
//
//py::object GetVector()
//{
//	Vector v({ 42.1234567890123456,43,44 }); //double precision maintained in NumPy array in python
//	return py::array_t<Real>(v.NumberOfItems(), v.GetDataPointer()); //copy array (could also be referenced!)
//}
//
//py::object GetOutputVariableType() //as OutputVariableType is known, it can return the special type
//{
//	return py::cast(OutputVariableType::Position);
//}

////Exudyn-access to test Vector
//const Vector& GetTest() { return test; }
//void SetTest(const Vector& vector) { test = vector; }

//Numpy-access to test Vector
//py::array_t<Real> GetNumpyTest() { return py::array_t<Real>(test.NumberOfItems(), test.GetDataPointer()); }
//void SetNumpyTest(const std::vector<Real>& v) { test = v; }

//end test functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PythonAlive()
{
	pout << "pout:Exudyn is alive\n";

	EXUmath::MatrixTests(); //perform matrix inverse tests
}

void PythonGo()
{
	py::exec(R"(
import exudyn
systemContainer = exudyn.SystemContainer()
mbs = systemContainer.AddSystem()
    )");
	pout << "main variables:\n systemContainer=exudyn.SystemContainer()\n mbs = systemContainer.AddSystem()\n";
	//pout << "ready to go\n";
}

#ifdef _MYDEBUG //only in debug mode!
void CreateTestSystem(Index systemNumber, Index arg0, Index arg1)
{
	if (systemNumber == 0)
	{
		STDstring str = "n = " + EXUstd::ToString(arg0) + "\n";
		py::exec(str.c_str());
		py::exec(R"(
		from itemInterface import *
		SC = exu.SystemContainer()
		mbs = SC.AddSystem()

		#n = 5    #nBodies
		k = 1000 #stiffness
		d = 10   #damping
		L = 0.25 #distance between nodes
		lastBodyMarker = -1
		mBody = 0
		body = 0
		node = 0

		nodeGround = mbs.AddNode(NodePointGround(referenceCoordinates = [0, 0, 0])) #ground node for coordinate constraint
		mNodeGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber = nodeGround, coordinate = 0)) #Ground node == > no action

		for j in range(n) :
			if (j == 0) :
				body = mbs.AddObject({ 'objectType': 'Ground', 'referencePosition' : [0,j,0] })
			else :
				node = mbs.AddNode(NodePoint2D(referenceCoordinates = [L*(j + 1), 0], initialCoordinates = [0, 0]))
				body = mbs.AddObject(MassPoint2D(physicsMass = 10, nodeNumber = node))
				#constrain y - direction:

			mNode = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber = node, coordinate = 1)) #constrain y - direction
			mbs.AddObject(ObjectConnectorCoordinateSpringDamper(markerNumbers = [mNodeGround, mNode], stiffness = k, damping = d, visualization = VObjectConnectorCoordinateSpringDamper(show = False)))

				#load:
			mBody = mbs.AddMarker(MarkerBodyPosition(bodyNumber = body, localPosition = [0, 0, 0]))
			if j != 0 : mbs.AddLoad({ 'loadType': 'ForceVector',  'markerNumber' : mBody,  'loadVector' : [0, -25, 0] })

			#print("mBody =", mBody, ", lastBodyMarker =", lastBodyMarker)
			#add spring - damper between objects
			if lastBodyMarker != -1:
				mbs.AddObject(ObjectConnectorSpringDamper(markerNumbers = [mBody, lastBodyMarker], referenceLength = L, stiffness = k, damping = d))

			lastBodyMarker = mBody

		mbs.Assemble()
	    )");
	}
	else { PyWarning("CreateTestSystem: no valid System number!"); }

}
#endif






