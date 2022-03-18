/** ***********************************************************************************************
* @class		CSolverImplicitSecondOrderNew
* @brief		This is the new (2021) implicit second order solver 
* @details		Details:
* 				- solves a dynamic system with constraints
*               - either use Newmark-based formulas with index-2 reduction, or use generalized-alpha with index 3 constraints
*               - in the final version it should include GGL stabilization, Br�ls/Arnold implementation of gen-alpha and Lie group integration
*				- step is solved by nonlinear iteration and Newton's method for accelerations
*
* @author		Gerstmayr Johannes
* @date			2021-01-27 (generated)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
* @note			Bug reports, support and further information:
* 				- email: johannes.gerstmayr@uibk.ac.at
* 				- weblink: https://github.com/jgerstmayr/EXUDYN
* 				
*
************************************************************************************************ */
#ifndef CSOLVERIMPLICITSECONDORDERTIMEINT__H
#define CSOLVERIMPLICITSECONDORDERTIMEINT__H

#include "Solver/CSolverBase.h" 

//! this is the new general implicit second order time integration solver
class CSolverImplicitSecondOrderTimeInt : public CSolverBase
{
public: //made public for access via pybind
	//copy of parameters from integration scheme (cannot be changed during integration!)
	Real newmarkBeta;
	Real newmarkGamma;
	Real alphaM;
	Real alphaF;
	Real spectralRadius;
	Real factJacAlgorithmic;

	bool useScaling; //scaling for ODE2 and AE part to ensure good conditioning of Jacobian
	bool hasConstantMassMatrix; //!< avoid recomputation of mass matrix, if constant


	//bool useIndex2Constraints; ==> directly linked to simulationSettings
public:

	//! return true, if static solver; needs to be overwritten in derived class
	virtual bool IsStaticSolver() const override { return false; }

	//! get solver name - needed for output file header and visualization window
	virtual const STDstring GetSolverName() const override { return "implicit second order time integration"; }

	//! reduce step size (0 .. with given step size, 1..normal, 2..severe problems); return true, if reduction was successful
	virtual bool ReduceStepSize(CSystem& computationalSystem, const SimulationSettings& simulationSettings,
		Index severity, Real suggestedStepSize = -1.) override;

	//! increase step size if convergence is good; if suggestedStepSize == -1, a solver-specific factor will be used
	virtual void IncreaseStepSize(CSystem& computationalSystem, const SimulationSettings& simulationSettings,
		Real suggestedStepSize = -1.) override
	{
		if ((it.currentStepSize != it.maxStepSize))
		{
			it.currentStepSize = EXUstd::Minimum(it.maxStepSize, simulationSettings.timeIntegration.adaptiveStepIncrease*it.currentStepSize);

			if ((IsVerboseCheck(1) && (output.stepInformation & StepInfo::stepIncreaseInfo)) || IsVerboseCheck(2))
			{
				STDstring str = STDstring("  Solve steps: adaptive increase to step size = ") + EXUstd::ToString(it.currentStepSize) + " due to fast convergence";
				if (IsStaticSolver()) { str += ", load factor = " + EXUstd::ToString(computationalSystem.GetSolverData().loadFactor); }
				else { str += ", time = " + EXUstd::ToString(it.currentTime); }
				VerboseWrite(1, str+"\n");
			}
		}
	}

	//! pre-initialize for solver specific tasks; called at beginning of InitializeSolver, right after Solver data reset
	virtual void PreInitializeSolverSpecific(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! post-initialize for solver specific tasks; called at the end of InitializeSolver
	virtual void PostInitializeSolverSpecific(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! set/compute initial conditions (solver-specific!); called from InitializeSolver()
	virtual void InitializeSolverInitialConditions(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! update currentTime (and load factor); MUST be overwritten in special solver class
	virtual void UpdateCurrentTime(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! compute residual for Newton method (e.g. static or time step); store result vector in systemResidual and return scalar residual
	virtual Real ComputeNewtonResidual(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! compute update for currentState from newtonSolution (decrement from residual and jacobian)
	virtual void ComputeNewtonUpdate(CSystem& computationalSystem, const SimulationSettings& simulationSettings, bool initial = false) override;

	//! compute jacobian for newton method of given solver method; store result in systemJacobian
	virtual void ComputeNewtonJacobian(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! finalize algorithmic accelerations in generalized alpha method
	virtual void FinalizeNewton(CSystem& computationalSystem, const SimulationSettings& simulationSettings);

};

class MainSolverImplicitSecondOrder;
typedef std::function<bool(MainSolverImplicitSecondOrder&, MainSystem&, const SimulationSettings&)> MainSolverImplicitSecondOrderUserFunction;
typedef std::function<bool(MainSolverImplicitSecondOrder&, MainSystem&, const SimulationSettings&, bool)> MainSolverImplicitSecondOrderUserFunctionBool;
typedef std::function<Real(MainSolverImplicitSecondOrder&, MainSystem&, const SimulationSettings&)> MainSolverImplicitSecondOrderUserFunctionReal;

//! second order integrator with user functions capability
class CSolverImplicitSecondOrderTimeIntUserFunction : public CSolverImplicitSecondOrderTimeInt
{
private:
	MainSolverImplicitSecondOrderUserFunction userFunctionPreInitializeSolverSpecific;//!< User function to override Newton()
	MainSolverImplicitSecondOrderUserFunction userFunctionInitializeSolverInitialConditions;//!< User function to override Newton()
	MainSolverImplicitSecondOrderUserFunction userFunctionPostInitializeSolverSpecific;//!< User function to override Newton()
	MainSolverImplicitSecondOrderUserFunction userFunctionUpdateCurrentTime;//!< User function to override UpdateCurrentTime()
	MainSolverImplicitSecondOrderUserFunction userFunctionInitializeStep;//!< User function to override InitializeStep()
	MainSolverImplicitSecondOrderUserFunction userFunctionFinishStep;//!< User function to override FinishStep()
	MainSolverImplicitSecondOrderUserFunction userFunctionDiscontinuousIteration;//!< User function to override DiscontinuousIteration()
	MainSolverImplicitSecondOrderUserFunction userFunctionNewton;//!< User function to override Newton()
	MainSolverImplicitSecondOrderUserFunctionBool userFunctionComputeNewtonUpdate;//!< User function to override ComputeNewtonUpdate()
	MainSolverImplicitSecondOrderUserFunctionReal userFunctionComputeNewtonResidual;//!< User function to override ComputeNewtonResidual()
	MainSolverImplicitSecondOrderUserFunction userFunctionComputeNewtonJacobian;//!< User function to override ComputeNewtonJacobian()
	MainSolverImplicitSecondOrderUserFunctionReal userFunctionPostNewton;//!< User function to override PostNewton()

	MainSolverImplicitSecondOrder* mainSolver; //!< pointer to main solver, needed in user function; this is dangerous, but cannot be avoided!
	MainSystem* mainSystem; //!< pointer to main solver, needed in user function; this is dangerous, but cannot be avoided!

public:
	CSolverImplicitSecondOrderTimeIntUserFunction()
	{
		userFunctionPreInitializeSolverSpecific = 0;
		userFunctionInitializeSolverInitialConditions = 0;
		userFunctionPostInitializeSolverSpecific = 0;

		userFunctionUpdateCurrentTime = 0;
		userFunctionInitializeStep = 0;
		userFunctionFinishStep = 0;
		userFunctionDiscontinuousIteration = 0;
		userFunctionNewton = 0;
		userFunctionComputeNewtonUpdate = 0;
		userFunctionComputeNewtonResidual = 0;
		userFunctionComputeNewtonJacobian = 0;
		userFunctionPostNewton = 0;

		mainSolver = nullptr;
		mainSystem = nullptr;
	}

	virtual void SetUserFunctionUpdateCurrentTime(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionUpdateCurrentTime = uf;
	}
	virtual void SetUserFunctionInitializeStep(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionInitializeStep = uf;
	}
	virtual void SetUserFunctionFinishStep(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionFinishStep = uf;
	}
	virtual void SetUserFunctionDiscontinuousIteration(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionDiscontinuousIteration = uf;
	}
	virtual void SetUserFunctionNewton(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionNewton = uf;
	}
	virtual void SetUserFunctionComputeNewtonUpdate(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunctionBool& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionComputeNewtonUpdate = uf;
	}
	virtual void SetUserFunctionComputeNewtonResidual(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunctionReal& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionComputeNewtonResidual = uf;
	}
	virtual void SetUserFunctionComputeNewtonJacobian(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunction& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionComputeNewtonJacobian = uf;
	}
	virtual void SetUserFunctionPostNewton(MainSolverImplicitSecondOrder* mainSolverInit, MainSystem* mainSystemInit, const MainSolverImplicitSecondOrderUserFunctionReal& uf)
	{
		mainSolver = mainSolverInit; mainSystem = mainSystemInit; userFunctionPostNewton = uf;
	}

	//! update currentTime (and load factor); MUST be overwritten in special solver class
	virtual void UpdateCurrentTime(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! initialize static step / time step; python-functions; do some outputs, checks, etc.
	virtual void InitializeStep(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! finish static step / time step; write output of results to file
	virtual void FinishStep(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! perform discontinuousIteration for static step / time step; CALLS ComputeNewtonResidual 
	virtual bool DiscontinuousIteration(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! perform Newton method for given solver method
	virtual bool Newton(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! compute residual for Newton method (e.g. static or time step); store result vector in systemResidual and return scalar residual
	virtual Real ComputeNewtonResidual(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! compute update for currentState from newtonSolution (decrement from residual and jacobian)
	virtual void ComputeNewtonUpdate(CSystem& computationalSystem, const SimulationSettings& simulationSettings, bool initial = false) override;

	//! compute jacobian for newton method of given solver method; store result in systemJacobian
	virtual void ComputeNewtonJacobian(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;

	//! perform PostNewton method for given solver method
	virtual Real PostNewton(CSystem& computationalSystem, const SimulationSettings& simulationSettings) override;
};



#endif
