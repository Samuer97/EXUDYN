/** ***********************************************************************************************
* @file			ReleaseAssert.h
* @brief		Enable asserts in release mode; which show more information on runtime errors in release mode
* @details		Details:
                - helps to detect index and memory allocation errors for large models
*
* @author		Gerstmayr Johannes
* @date			2010-10-01 (created)
* @date			2018-04-30 (update, Exudyn)
* @copyright	This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
* @note			Bug reports, support and further information:
* 				- email: johannes.gerstmayr@uibk.ac.at
* 				- weblink: https://github.com/jgerstmayr/EXUDYN
* 				
*
************************************************************************************************ */
#ifndef EXUDYNEXCEPTIONS__H
#define EXUDYNEXCEPTIONS__H

#include <assert.h>
#include <exception>
#include <stdexcept>

//now defined in preprocessor of Release / ReleaseFast
//#define __FAST_EXUDYN_LINALG //use this to avoid any range checks in linalg; TEST: with __FAST_EXUDYN_LINALG: 2.3s time integration of contact problem, without: 2.9s

//gcc cannot call std::exception() ==> use runtime_error
#ifdef _MSC_VER
//#define EXUexception std::exception
#define EXUexception std::runtime_error
#else
#define EXUexception std::runtime_error
#endif

//#define __FAST_EXUDYN_LINALG //defined as preprocessor flags

#ifndef __FAST_EXUDYN_LINALG
#define __PYTHON_USERFUNCTION_CATCH__  //performs try/catch in all python user functions
#define __EXUDYN_RUNTIME_CHECKS__  //performs several runtime checks, which slows down performance in release or debug mode

//!check if _checkExpression is true; if no, trow std::exception(_exceptionMessage); _exceptionMessage will be a const char*, e.g. "VectorBase::operator[]: invalid index"
//!linalg matrix/vector access functions, memory allocation, array classes and solvers will throw exceptions if the errors are not recoverable
//!this, as a consequence leads to a pybind exception translated to python; the message will be visible in python; for __FAST_EXUDYN_LINALG, no checks are performed

#define CHECKandTHROW(_checkExpression,_exceptionMessage) ((_checkExpression) ? 0 : throw EXUexception(_exceptionMessage))
#define CHECKandTHROWcond(_checkExpression) ((_checkExpression) ? 0 : throw EXUexception("unexpected EXUDYN internal error"))
//always throw:
#define CHECKandTHROWstring(_exceptionMessage) (throw EXUexception(_exceptionMessage))
#else
	//no checks in __FAST_EXUDYN_LINALG mode
#define CHECKandTHROW(_checkExpression,_exceptionMessage)
#define CHECKandTHROWcond(_checkExpression)
#define CHECKandTHROWstring(_exceptionMessage)
#endif

#define __EXUDYN_invalid_local_node0 "Object:GetNodeNumber: invalid call to local node number" //workaround to avoid string in object definition file
#define __EXUDYN_invalid_local_node "Object:GetNodeNumber: invalid local node number > 0" //workaround to avoid string in object definition file
#define __EXUDYN_invalid_local_node1 "Object:GetNodeNumber: invalid local node number > 1" //workaround to avoid string in object definition file
#define __EXUDYN_invalid_local_node2 "Object:GetNodeNumber: invalid local node number > 2" //workaround to avoid string in object definition file

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//a specific flag _MYDEBUG is used as the common _NDEBUG flag does not work in Visual Studio
//use following statements according to msdn.microsoft in order to detect memory leaks and show line number/file where first new to leaked memory has been called
//works only, if dbg_new is used instead of all 'new' commands!
#ifdef _MYDEBUG
#define dbg_new new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#undef NDEBUG
#else
#define dbg_new new
#ifndef NDEBUG
#define NDEBUG //used to avoid range checks e.g. in Eigen
#endif
#endif
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#endif //EXUDYNEXCEPTIONS__H
