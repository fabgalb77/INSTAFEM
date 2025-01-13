from FEMSolver3D import *
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import direct
from scipy.optimize import Bounds, OptimizeResult
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from typing import Tuple, List, Optional, Dict, Callable
from scipy.sparse import csr_matrix
from numba import jit
import scipy.sparse.linalg as spla
import pygmsh
import meshio
import pyamg

class DisplacementConstraint:
    """Class representing a target displacement constraint"""
    def __init__(self, node_id: int, dofs: List[int], target_values: List[float],
                 weights: Optional[List[float]] = None):
        self.node_id = node_id
        self.dofs = np.array(dofs)
        self.target_values = np.array(target_values)
        self.weights = np.array(weights) if weights else np.ones_like(target_values)

class UnknownForce:
    """Class representing an unknown force to be optimized"""
    def __init__(self, node_id: int, direction: np.ndarray, 
                 bounds: Optional[Tuple[float, float]] = None):
        self.node_id = node_id
        self.direction = direction / np.linalg.norm(direction)
        self.bounds = bounds if bounds else (-np.inf, np.inf)

class ObjectiveFunction(ABC):
    """Base class for objective functions in inverse problems"""
    @abstractmethod
    def __call__(self, force_magnitudes: np.ndarray, 
                 displacements: np.ndarray,
                 unknown_forces: List[UnknownForce],
                 displacement_constraints: List[DisplacementConstraint]) -> float:
        pass

class MinimumForceObjective(ObjectiveFunction):
    """Minimize sum of squared force magnitudes"""
    def __call__(self, force_magnitudes: np.ndarray, 
                 displacements: np.ndarray,
                 unknown_forces: List[UnknownForce],
                 displacement_constraints: List[DisplacementConstraint]) -> float:
        return np.sum(force_magnitudes**2)

class L1ForceObjective(ObjectiveFunction):
    """Minimize sum of absolute force magnitudes"""
    def __call__(self, force_magnitudes: np.ndarray, 
                 displacements: np.ndarray,
                 unknown_forces: List[UnknownForce],
                 displacement_constraints: List[DisplacementConstraint]) -> float:
        return np.sum(np.abs(force_magnitudes))

class InverseProblem:
    """Class handling nonlinear inverse statics problem"""
    def __init__(self, mesh: Mesh, unknown_forces: List[UnknownForce],
                 displacement_constraints: List[DisplacementConstraint],
                 objective: ObjectiveFunction):
        self.mesh = mesh
        self.unknown_forces = unknown_forces
        self.displacement_constraints = displacement_constraints
        self.objective = objective
        self.solver = NonlinearFEMSolver3D(mesh)
        
        # Optimization parameters
        self.force_tolerance = 1e0  # N
        self.displacement_tolerance = 1e-1  # mm 
        self.n_stable_iterations = 3
        
        # Diagnostic counters
        self.iter_count = 0
        self.func_count = 0
        self.last_forces = None
        self.last_constraint_violations = None
        self.stable_iterations = 0
        self.best_solution = None
        self.best_violation = float('inf')
        self.best_objective = float('inf')
    
    def _estimate_force_scale(self) -> float:
        """
        Estimate characteristic force scale from the problem parameters:
        - External loads
        - Material properties
        - Geometry
        - Bounds on unknown forces
        """
        # Get maximum external force from Neumann BCs
        max_external_force = 0.0
        for bc in self.mesh.neumann_bcs:
            for value in bc.values:
                max_external_force = max(max_external_force, abs(value))
        
        # Get maximum bound from unknown forces
        max_bound_force = 0.0
        for force in self.unknown_forces:
            max_bound_force = max(max_bound_force, abs(force.bounds[0]), abs(force.bounds[1]))
        
        # Get material stiffness scale (E*A/L for a simple bar)
        max_E = 0.0
        for _, material in self.mesh.elements[ElementType.TET4]:
            if isinstance(material, LinearElastic):
                max_E = max(max_E, material.E)
        
        # Get approximate structural dimensions
        coords = self.mesh.nodes
        L = np.max(coords[:, 0]) - np.min(coords[:, 0])  # Length
        H = np.max(coords[:, 1]) - np.min(coords[:, 1])  # Height
        W = np.max(coords[:, 2]) - np.min(coords[:, 2])  # Width
        A = H * W  # Cross-sectional area
        
        # Compute different force scales
        f_external = max_external_force
        f_bound = max_bound_force
        f_material = max_E * A / L  # Approximate force to cause unit strain
        
        # Take minimum of material force scale and maximum of external/bound forces
        force_scale = min(f_material, max(f_external, f_bound))
        
        print(f"\nForce scale estimation:")
        print(f"  Maximum external force: {f_external:.2e} N")
        print(f"  Maximum bound force: {f_bound:.2e} N")
        print(f"  Material force scale (EA/L): {f_material:.2e} N")
        print(f"  Selected force scale: {force_scale:.2e} N")
        
        return force_scale

    def optimize_DIRECT(self, max_iter: int = 100, max_eval: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize using DIRECT algorithm with solution tracking
        """
        self.best_solution = None
        self.best_objective = float('inf')
        self.best_violation = float('inf')
        self.best_displacement = None
        self.n_eval = 0
        
        bounds = [(force.bounds[0], force.bounds[1]) for force in self.unknown_forces]
        bounds = Bounds(
            lb=np.array([b[0] for b in bounds]),
            ub=np.array([b[1] for b in bounds])
        )
        
        def objective(x):
            self.n_eval += 1
            if self.n_eval > max_eval:
                return 1e20
                
            try:
                # Compute displacements
                displacements = self._apply_forces(x)
                current, target = self._compute_displacement_constraints(displacements)
                violations = current - target
                violation_norm = np.linalg.norm(violations)
                
                # Compute force magnitude objective
                force_obj = np.sum(x**2)
                
                # Always update best solution if violation is better
                if violation_norm < self.best_violation:
                    self.best_solution = x.copy()
                    self.best_objective = force_obj
                    self.best_violation = violation_norm
                    self.best_displacement = displacements.copy()
                    print(f"\nNew best solution found:")
                    print(f"Forces: {x}")
                    print(f"Violation: {violation_norm:.6f} mm")
                    print(f"Force magnitude: {np.sqrt(force_obj):.2f} N")
                # Update if violation is similar but force is better
                elif np.abs(violation_norm - self.best_violation) < 1e-6 and force_obj < self.best_objective:
                    self.best_solution = x.copy()
                    self.best_objective = force_obj
                    self.best_violation = violation_norm
                    self.best_displacement = displacements.copy()
                    print(f"\nNew best solution found (lower force):")
                    print(f"Forces: {x}")
                    print(f"Violation: {violation_norm:.6f} mm")
                    print(f"Force magnitude: {np.sqrt(force_obj):.2f} N")
                
                # Compute objective with stronger penalty
                if violation_norm > self.displacement_tolerance:
                    penalty = 1e12 * (violation_norm/self.displacement_tolerance)**4 
                    total_obj = penalty
                else:
                    total_obj = force_obj
                
                if self.n_eval % 50 == 0:
                    print(f"\nEvaluation {self.n_eval}:")
                    print(f"Current violation: {violation_norm:.6f} mm")
                    print(f"Best violation so far: {self.best_violation:.6f} mm")
                    print(f"Current force magnitude: {np.sqrt(force_obj):.2f} N")
                
                return total_obj
                
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                return float('inf')
        
        print("\nRunning DIRECT optimization...")
        result = direct(
            objective,
            bounds=bounds,
            maxiter=max_iter,
            eps=1e-8,
            locally_biased=True
        )
        
        if self.best_solution is None:
            raise RuntimeError("No valid solution found")
        
        print(f"\nDIRECT Optimization completed:")
        print(f"Best force magnitudes: {self.best_solution}")
        print(f"Total force magnitude: {np.sqrt(np.sum(self.best_solution**2)):.2f} N")
        print(f"Best violation: {self.best_violation:.6f} mm")
        print(f"Function evaluations: {self.n_eval}")
        
        return self.best_solution, self.best_displacement

    def optimize_bayesian(self, n_calls: int = 100, n_initial_points: int = 20):
        import torch
        import numpy as np
        from botorch.fit import fit_gpytorch_model
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.acquisition.monte_carlo import qExpectedImprovement
        from botorch.acquisition.objective import ConstrainedMCObjective
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.utils.transforms import standardize

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.double

        # 0. Define helper functions for normalization
        def normalize(X_raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
            """Scale `X_raw` to [0,1]^d based on `bounds` (d x 2)."""
            lb = bounds[:, 0]
            ub = bounds[:, 1]
            return (X_raw - lb) / (ub - lb)

        def unnormalize(X_norm: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
            """Convert normalized inputs in [0,1]^d back to real-world domain."""
            lb = bounds[:, 0]
            ub = bounds[:, 1]
            return X_norm * (ub - lb) + lb

        # 1. Reset best-solution trackers
        self.best_solution = None
        self.best_objective = float('inf')
        self.best_violation = float('inf')
        self.best_displacement = None

        # 2. Prepare real-world bounds and a normalized domain
        bounds_raw = torch.tensor(
            [[f.bounds[0], f.bounds[1]] for f in self.unknown_forces],
            device=device,
            dtype=dtype
        )
        d = bounds_raw.shape[0]

        # For BoTorch optimize_acqf, pass [0,1]^d as bounds
        bounds_norm = torch.stack([
            torch.zeros(d, dtype=dtype, device=device),
            torch.ones(d, dtype=dtype, device=device)
        ])  # shape = (2, d)

        # 3. Generate initial points in [0,1]^d and evaluate
        X_norm = torch.rand(n_initial_points, d, device=device, dtype=dtype)
        obj_vals = []
        constr_vals = []

        for i in range(n_initial_points):
            x_norm_i = X_norm[i]  # shape (d,)
            x_real_i = unnormalize(x_norm_i, bounds_raw).cpu().numpy()

            # Objective: sum of squares of forces
            obj_val = np.sum(x_real_i**2)

            # Constraint: displacement violation
            displacements = self._apply_forces(x_real_i)
            current, target = self._compute_displacement_constraints(displacements)
            violation_norm = np.linalg.norm(current - target)
            constr_val = violation_norm - self.displacement_tolerance

            obj_vals.append(obj_val)
            constr_vals.append(constr_val)

            # Track best feasible solution
            if violation_norm < self.best_violation:
                self.best_solution = x_real_i
                self.best_objective = obj_val
                self.best_violation = violation_norm
                self.best_displacement = displacements
                print(f"\nNew best solution found (init {i+1}):")
                print(f"Forces: {x_real_i}")
                print(f"Violation: {violation_norm:.6f} mm")
                print(f"Force magnitude: {np.sqrt(obj_val):.2f} N")

        # Convert to Tensors
        Y = torch.tensor(obj_vals, device=device, dtype=dtype).unsqueeze(-1)
        C = torch.tensor(constr_vals, device=device, dtype=dtype).unsqueeze(-1)

        print("\nRunning constrained Bayesian optimization...")

        # 4. Main optimization loop
        for iteration in range(n_calls - n_initial_points):
            # (a) Fit separate GPs for objective and constraint
            obj_model = SingleTaskGP(X_norm, standardize(Y))  # standardize objective
            constr_model = SingleTaskGP(X_norm, C)            # skip standardizing constraint
            obj_mll = ExactMarginalLogLikelihood(obj_model.likelihood, obj_model)
            fit_gpytorch_model(obj_mll)
            constr_mll = ExactMarginalLogLikelihood(constr_model.likelihood, constr_model)
            fit_gpytorch_model(constr_mll)

            # (b) Combine into multi-output model
            model = ModelListGP(obj_model, constr_model).to(device=device, dtype=dtype)

            # (c) Define constrained MC objective
            objective_fn = lambda samples, X=None: -samples[..., 0]
            constraint_fn = [lambda samples, X=None: samples[..., 1]]
            cobjective = ConstrainedMCObjective(objective=objective_fn, constraints=constraint_fn)

            # (d) Determine best_f from feasible points only
            feasible_mask = (C <= 0)
            if feasible_mask.any():
                best_f = Y[feasible_mask].min().item()
            else:
                best_f = Y.min().item()

            # (e) Construct qEI with the constrained objective
            acq_func = qExpectedImprovement(
                model=model,
                objective=cobjective,
                best_f=best_f
            )

            # (f) Optimize acquisition function in the normalized domain
            candidates_norm, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds_norm,
                q=1,
                num_restarts=10,
                raw_samples=100,
                options={"batch_limit": 5, "maxiter": 200}
            )

            # Evaluate new candidate in real domain
            x_new_norm = candidates_norm[0]
            x_new_real = unnormalize(x_new_norm, bounds_raw).cpu().numpy()

            # >>>> REAL-TIME FEEDBACK PRINTS <<<<
            print(f"\n== Iteration {iteration+1} candidate ==")
            print(f"Normalized:  {x_new_norm.detach().cpu().numpy()}")
            print(f"Real Forces: {x_new_real}")

            obj_val = np.sum(x_new_real**2)
            displacements = self._apply_forces(x_new_real)
            current, target = self._compute_displacement_constraints(displacements)
            violation_norm = np.linalg.norm(current - target)
            constr_val = violation_norm - self.displacement_tolerance

            print(f"Objective value (sum of squares): {obj_val:.4f}")
            print(f"Constraint value:                {constr_val:.4f}  (≤ 0 is feasible)")
            if constr_val <= 0.0:
                print("Candidate is FEASIBLE!")
            else:
                print("Candidate is NOT feasible.")

            # (g) Update training sets in normalized space
            X_norm = torch.cat([X_norm, candidates_norm], dim=0)
            Y = torch.cat([Y, torch.tensor([[obj_val]], device=device, dtype=dtype)], dim=0)
            C = torch.cat([C, torch.tensor([[constr_val]], device=device, dtype=dtype)], dim=0)

            # (h) Track best feasible solution so far
            if violation_norm < self.best_violation:
                self.best_solution = x_new_real
                self.best_objective = obj_val
                self.best_violation = violation_norm
                self.best_displacement = displacements
                print(f"\nNew best solution found (iteration {iteration+1}):")
                print(f"Forces: {x_new_real}")
                print(f"Violation: {violation_norm:.6f} mm")
                print(f"Force magnitude: {np.sqrt(obj_val):.2f} N")

        # 5. Final checks & return
        if self.best_solution is None:
            raise RuntimeError("No valid (feasible) solution found.")

        print("\nBayesian Optimization completed:")
        print(f"Best force magnitudes: {self.best_solution}")
        print(f"Total force magnitude: {np.sqrt(np.sum(self.best_solution ** 2)):.2f} N")
        print(f"Best violation: {self.best_violation:.6f} mm")
        print(f"Number of evaluations: {n_calls}")

        return self.best_solution, self.best_displacement


    
    def _apply_forces(self, force_magnitudes: np.ndarray) -> np.ndarray:
        """Apply forces and solve nonlinear problem"""
        # Initialize force vector
        forces = np.zeros(self.solver.n_dof)
        
        # Add unknown forces
        for mag, force in zip(force_magnitudes, self.unknown_forces):
            force_vector = mag * force.direction
            forces[3*force.node_id:3*force.node_id + 3] += force_vector
        
        # Solve nonlinear problem (Neumann BCs are handled by solver)
        return self.solver.solve_nonlinear(forces)
    
    def _compute_displacement_constraints(self, displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute current and target values for displacement constraints
        Note: displacements are already in mm
        """
        current_values = []
        target_values = []
        
        for constraint in self.displacement_constraints:
            for dof, target, weight in zip(constraint.dofs, 
                                         constraint.target_values,
                                         constraint.weights):
                current = displacements[3*constraint.node_id + dof]  # Already in mm
                current_values.append(current * weight)
                target_values.append(target * weight)
        
        return np.array(current_values), np.array(target_values)
