/**
 * MPCC Solver Interface — includes the acados MPCC solver.
 *
 * Usage:
 *   #include "mpcc_solver_interface.h"
 *   mpcc::AcadosSolver solver;
 *   solver.init(config);
 *   auto result = solver.solve(...);
 */

#ifndef MPCC_SOLVER_INTERFACE_H
#define MPCC_SOLVER_INTERFACE_H

#include "acados_mpcc_solver.h"
namespace mpcc { using ActiveSolver = AcadosSolver; }

#endif  // MPCC_SOLVER_INTERFACE_H
