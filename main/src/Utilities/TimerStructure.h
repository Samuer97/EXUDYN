/** ***********************************************************************************************
* @file			TimerStructure.h
* @brief		This file contains a timer class, which can be used to measure function timings at any place in the code
*
* @author		Gerstmayr Johannes
* @date			2020-03-01 (created)
* @copyright	This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
* @note			Bug reports, support and further information:
* 				- email: johannes.gerstmayr@uibk.ac.at
* 				- weblink: https://github.com/jgerstmayr/EXUDYN
* 				
*
************************************************************************************************ */
#ifndef TIMERSTRUCTURE__H
#define TIMERSTRUCTURE__H

//for chrono / datetime:
#include "Utilities/BasicFunctions.h" //defines Real
#include <vector>

//! use macros to completely avoid timings in fast mode
#ifndef EXUDYN_RELEASE
	//timer registration does not work in debug mode (and is not needed!)
	#ifndef _MYDEBUG
		#define USEGLOBALTIMERS
	#endif
#endif


#ifdef __FAST_EXUDYN_LINALG
	#define STARTGLOBALTIMER(_Expression) 
	#define STOPGLOBALTIMER(_Expression) 
	#define STARTGLOBALTIMERmain(_Expression) 
	#define STOPGLOBALTIMERmain(_Expression) 
#else
    //main timers are always kept!
	#define STARTGLOBALTIMERmain(_Expression) (globalTimers.StartTimer(_Expression))
	#define STOPGLOBALTIMERmain(_Expression) (globalTimers.StopTimer(_Expression))
	#ifdef USEGLOBALTIMERS
		#define STARTGLOBALTIMER(_Expression) (globalTimers.StartTimer(_Expression))
		#define STOPGLOBALTIMER(_Expression) (globalTimers.StopTimer(_Expression))
	#else
		#define STARTGLOBALTIMER(_Expression) 
		#define STOPGLOBALTIMER(_Expression) 
	#endif
#endif


class TimerStructure;
extern TimerStructure globalTimers;

//!special timer structure to measure time spent at certain parts of code
//! put a global variable of this class somewhere to allow micro-timing of functions at any place
class TimerStructure
{
private:
	//this class has two coupled lists, which do the work
	std::vector<Real> counters;
	std::vector<const char*> counterNames;
	Real offsetSecondsPerCall;
public:
	TimerStructure() { offsetSecondsPerCall = 0; };

	//! initialize structure with given (measured offset per call, to obtain more accurate measurements)
	TimerStructure(Real offsetSecondsPerCallInit) { offsetSecondsPerCall = offsetSecondsPerCallInit; };
	
	//! create a new timer; name must be a static name (must exist until end of timer) or dynamically allocated string, may not be deleted
	Index AddTimer(const char* name)
	{
		Index n = (Index)counters.size();
		counters.push_back(0.);
		counterNames.push_back(name);
		return n;
	}

	//!get current value of a timer
	Real GetTiming(Index counterIndex) const { return counters[counterIndex]; }

	//!get current value of a timer
	const char* GetTimerName(Index counterIndex) const { return counterNames[counterIndex]; }

	//!get current value of a timer
	Index NumberOfTimers() const { return (Index)counters.size(); }

	//!reset timers( e.g. before starting simulation):
	void Reset()
	{
		for (auto& item : counters) { item = 0; }
	}

	//! set counter to specific value
	void SetCounter(Index counterIndex, Real value) { counters[counterIndex] = value; }

	//! get counter value
	Real GetCounter(Index counterIndex) { return counters[counterIndex]; }

	//! start measurement
	void StartTimer(Index counterIndex)
	{
		counters[counterIndex] -= EXUstd::GetTimeInSeconds();
	}

	//! stop measurement
	void StopTimer(Index counterIndex)
	{
		counters[counterIndex] += EXUstd::GetTimeInSeconds() - offsetSecondsPerCall;
	}

	//! print current timers into string
	STDstring ToString() const 
	{
		if (!counters.size()) { return ""; }

		std::ostringstream ostr;
		ostr.precision(5); //reduced precision for nicer output...
		for (Index i = 0; i < (Index)counters.size(); i++)
		{
			if (counters[i] != 0.) //exclude timers that are exactly zero:
			{
				ostr << "  " << counterNames[i] << " = " << counters[i] << "s\n";
			}
		}
		return ostr.str();
	}
};


class TimerStructureRegistrator
{
public:
	//! register a timer by creating instance with this constructor:
	TimerStructureRegistrator(const char* timerName, Index& timerNumber, TimerStructure& globalTimerStructure, bool addAlways=false)
	{
	#ifdef USEGLOBALTIMERS
		timerNumber = globalTimerStructure.AddTimer(timerName);
	#else
		if (addAlways)
		{
			timerNumber = globalTimerStructure.AddTimer(timerName);
		}
	#endif
	}
};

#endif
