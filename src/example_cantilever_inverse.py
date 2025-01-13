from InverseSolver import *

def create_inverse_example():
    """
    Create example with known and unknown forces:
    - Cantilever beam fixed on left end
    - Known force (1000N down) at right end
    - An unknown force at L/3
    - Target: Zero vertical displacement at midpoint
    All lengths in mm, forces in N
    """

    # Create simple mesh with forced nodes at L/3 and midpoint
    length = 100.0  # mm
    height = 10.0   # mm
    width = 10.0    # mm
    E = 8000.0    # MPa (N/mm²)
    nu = 0.3
    
    # Create x coordinates to ensure we have nodes at L/3 and L/2
    x_first = np.linspace(0, length/3, 4)     # Points from 0 to L/3
    x_second = np.linspace(length/3, length/2, 4)  # Points from L/3 to L/2
    x_third = np.linspace(length/2, length, 8)     # Points from L/2 to end
    x = np.unique(np.concatenate([x_first, x_second, x_third]))  # Combine and remove duplicates
    
    y = np.linspace(0, height, 4)
    z = np.linspace(0, width, 4)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    mesh = Mesh(nodes)
    
    # Create elements
    elements = []
    nx = len(x) - 1
    ny = len(y) - 1
    nz = len(z) - 1
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                n0 = ix * len(y) * len(z) + iy * len(z) + iz
                n1 = n0 + len(z)
                n2 = n1 + 1
                n3 = n0 + 1
                n4 = n0 + len(y) * len(z)
                n5 = n4 + len(z)
                n6 = n5 + 1
                n7 = n4 + 1
                
                elements.extend([
                    [n0, n1, n3, n4],
                    [n1, n2, n3, n6],
                    [n1, n4, n5, n6],
                    [n3, n4, n6, n7],
                    [n1, n3, n4, n6],
                ])
    
    # Add material
    steel = LinearElastic("material", E=E, nu=nu)  # Properties in MPa
    mesh.add_elements(ElementType.TET4, np.array(elements), steel)
    
    # Add fixed boundary condition at x=0
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < 1e-6)[0]
    mesh.add_dirichlet_bc(BoundaryCondition(
        fixed_nodes,
        [0, 1, 2] * len(fixed_nodes),
        [0.0] * (3 * len(fixed_nodes))
    ))
    
    # Find nodes for loads and constraints
    end_node = mesh.find_closest_node(
        np.array([length, height/2, width/2])
    )
    
    third_node = mesh.find_closest_node(
        np.array([length/3, height/2, width/2])
    )
    
    mid_node = mesh.find_closest_node(
        np.array([length/2, height/2, width/2])
    )
    
    # Print node locations to verify
    print(f"\nNode locations:")
    print(f"End node: {end_node} at coordinates: ({nodes[end_node][0]:.1f}, {nodes[end_node][1]:.1f}, {nodes[end_node][2]:.1f}) mm")
    print(f"L/3 node: {third_node} at coordinates: ({nodes[third_node][0]:.1f}, {nodes[third_node][1]:.1f}, {nodes[third_node][2]:.1f}) mm")
    print(f"Mid node: {mid_node} at coordinates: ({nodes[mid_node][0]:.1f}, {nodes[mid_node][1]:.1f}, {nodes[mid_node][2]:.1f}) mm")
    
    # Add known force (1000N downward at end)
    mesh.add_neumann_bc(BoundaryCondition(
        [end_node],
        [1],  # y-direction
        [-1000.0]  # downward force
    ))
    
    # Define unknown force at L/3
    unknown_forces = [
        UnknownForce(third_node, np.array([0, 1, 0]), bounds=(-5000, 5000)),
    ]
    
    # Define displacement constraint (zero vertical displacement at midpoint)
    displacement_constraints = [
        DisplacementConstraint(mid_node, [1], [0.0])  # Zero y-displacement at midpoint
    ]
    
    return InverseProblem(mesh, unknown_forces, displacement_constraints, MinimumForceObjective())

def run_inverse_example():
    print("\n=== Starting Inverse Problem Example ===")
    print("Problem setup:")
    print("- Cantilever beam (100 x 10 x 10 mm)")
    print("- Fixed at left end")
    print("- Known force: 1000N down at right end")
    print("- Unknown force at L/3 (33.3 mm)")
    print("- Target: Zero vertical displacement at midpoint (50 mm)")
    
    # Create problem
    problem = create_inverse_example()
    
    # Compute initial state (known force only)
    print("\nComputing initial state (known force only)...")
    initial_displacements = problem.solver.solve_nonlinear()
    initial_stresses = problem.solver.compute_stresses(initial_displacements)
    initial_vm = problem.solver.compute_von_mises_stresses(initial_stresses)
    
    # Export initial state
    problem.solver.export_to_vtk(
        "inverse_validation_initial.vtu",
        initial_displacements,
        initial_stresses
    )
    
    print("\nInitial state results:")
    print(f"Maximum displacement: {np.max(np.abs(initial_displacements)):.3f} mm")
    print(f"Maximum von Mises stress: {np.max(initial_vm):.1f} MPa")
    
    # Run optimization
    print("\nRunning optimization to find unknown forces...")
    forces, final_displacements = problem.optimize_DIRECT(max_iter=100, max_eval=1000)
    #forces, final_displacements = problem.optimize_bayesian(n_calls=200, n_initial_points=50)
    
    # Compute final state
    final_stresses = problem.solver.compute_stresses(final_displacements)
    final_vm = problem.solver.compute_von_mises_stresses(final_stresses)
    
    # Export final state
    problem.solver.export_to_vtk(
        "inverse_validation_final.vtu",
        final_displacements,
        final_stresses
    )
    
    print("\n=== Final Results ===")
    print(f'Found forces:')
    print(f"F (at L/3):   {forces[0]:8.2f} N")
    print(f"Total magnitude: {np.sqrt(np.sum(forces**2)):8.2f} N")
    
    # Check midpoint displacement
    mid_node = problem.displacement_constraints[0].node_id
    mid_disp = final_displacements[3*mid_node + 1]  # y-displacement
    print(f"\nMidpoint displacement:")
    print(f"Target: 0.000000 mm")
    print(f"Actual: {mid_disp:.6f} mm")
    print(f"Error:  {abs(mid_disp):.6f} mm")
    
    print("\nOverall results:")
    print(f"Maximum displacement: {np.max(np.abs(final_displacements)):.3f} mm")
    print(f"Maximum von Mises stress: {np.max(final_vm):.1f} MPa")
    
    return problem, forces, final_displacements

if __name__ == "__main__":
    run_inverse_example()