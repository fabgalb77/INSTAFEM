from FEMSolver3D import *

def example_gapuni():
    """Test gap element with two blocks connected by a single gap element"""
    # Geometry parameters (lengths in mm)
    block_size = 10.0          # Size of blocks in mm
    block_height = 5.0         # Height of blocks
    gap = 1.0                  # Initial gap between blocks

    # Material properties
    E = 200e3     # Young's modulus in MPa (N/mm^2)
    nu = 0.3      # Poisson's ratio

    # Mesh parameters
    mesh_size = 3.0   # Mesh element size in mm

    # Convert lengths from mm to meters for pygmsh
    mm_to_m = 1e-3
    block_size_m = block_size * mm_to_m
    block_height_m = block_height * mm_to_m
    gap_m = gap * mm_to_m
    mesh_size_m = mesh_size * mm_to_m

    # Create geometry using pygmsh
    with pygmsh.geo.Geometry() as geom:
        # Top block
        top_block = geom.add_box(
            x0=-block_size_m / 2,
            y0=-block_size_m / 2,
            z0=gap_m,  # Offset by gap
            x1=block_size_m / 2,
            y1=block_size_m / 2,
            z1=gap_m + block_height_m,
            mesh_size=mesh_size_m
        )

        # Bottom block
        bottom_block = geom.add_box(
            x0=-block_size_m / 2,
            y0=-block_size_m / 2,
            z0=-block_height_m,
            x1=block_size_m / 2,
            y1=block_size_m / 2,
            z1=0.0,
            mesh_size=mesh_size_m
        )

        # Generate mesh
        pygmsh_mesh = geom.generate_mesh()

    # Extract nodes and elements, converting to mm
    nodes = pygmsh_mesh.points * 1000.0  # Convert to mm
    tets = pygmsh_mesh.cells_dict['tetra'].astype(int)

    # Create mesh object
    mesh = Mesh(nodes)

    # Add tetrahedral elements with material properties
    material = LinearElastic("Steel", E=E, nu=nu)
    mesh.add_elements(ElementType.TET4, tets, material)

    # Find nodes for gap element
    tol = 1e-6
    # Bottom block top center node
    bottom_nodes = np.where(np.abs(nodes[:, 2] - 0.0) < tol)[0]
    bottom_center_node = bottom_nodes[np.argmin(nodes[bottom_nodes, 0]**2 + nodes[bottom_nodes, 1]**2)]
    
    # Top block bottom center node
    top_nodes = np.where(np.abs(nodes[:, 2] - gap) < tol)[0]
    top_center_node = top_nodes[np.argmin(nodes[top_nodes, 0]**2 + nodes[top_nodes, 1]**2)]

    print(f"\nGap element nodes:")
    print(f"Bottom node: {nodes[bottom_center_node]}")
    print(f"Top node: {nodes[top_center_node]}")

    # Create gap element
    gap_material = GapUniMaterial(
        name="GapMaterial",
        initial_clearance=gap * 0.9,  # Slightly less than geometric gap
        contact_direction=np.array([0.0, 0.0, 1.0]),  # Contact along Z axis
        penalty_parameter=E/5  # Penalty relative to material stiffness
    )
    mesh.add_elements(ElementType.GAPUNI, [np.array([bottom_center_node, top_center_node])], gap_material)

    # Boundary conditions
    # 1. Fix bottom of bottom block
    fixed_nodes = np.where(np.abs(nodes[:, 2] + block_height) < tol)[0]
    node_ids = []
    dofs = []
    values = []
    for node in fixed_nodes:
        for dof in [0, 1, 2]:
            node_ids.append(node)
            dofs.append(dof)
            values.append(0.0)
    mesh.add_dirichlet_bc(BoundaryCondition(node_ids, dofs, values))

    # 2. Apply displacement to top block
    top_nodes = np.where(np.abs(nodes[:, 2] - (gap + block_height)) < tol)[0]
    displacement = -0.5  # Downward displacement
    node_ids = []
    dofs = []
    values = []
    for node in top_nodes:
        for dof in [0, 1, 2]:
            node_ids.append(node)
            dofs.append(dof)
            if dof == 2:
                values.append(displacement)  # Z displacement
            else:
                values.append(0.0)  # Fix X and Y
    mesh.add_dirichlet_bc(BoundaryCondition(node_ids, dofs, values))

    # Create solver
    solver = NonlinearFEMSolver3D(mesh)
    solver.tolerance = 1e-5
    solver.max_iterations = 30
    solver.min_delta_lambda = 1e-4
    solver.delta_lambda = 0.1
    solver.damping_factor = 0.7

    """
    # Add debugging for gap element
    original_compute_matrices = solver.compute_gapuni_element_matrices

    def debug_gapuni_matrices(node_coords, u_e, material):
        x1, x2 = node_coords
        u1, u2 = u_e[:3], u_e[3:]
        
        current_x1 = x1 + u1
        current_x2 = x2 + u2
        current_vector = current_x2 - current_x1
        current_distance = np.dot(current_vector, material.contact_direction)
        
        print(f"\nGap element state:")
        print(f"Current distance: {current_distance:.6f}")
        print(f"Initial clearance: {material.initial_clearance:.6f}")
        print(f"Penetration: {material.initial_clearance - current_distance:.6f}")
        
        K_e, f_int_e = original_compute_matrices(node_coords, u_e, material)
        contact_force = np.linalg.norm(f_int_e[:3])
        print(f"Contact force: {contact_force:.2f} N")
        
        return K_e, f_int_e
    
    solver.compute_gapuni_element_matrices = debug_gapuni_matrices
    """

    # Solve
    print("\nSolving system...")
    try:
        displacements = solver.solve_nonlinear()
        
        # Compute stresses
        stresses = solver.compute_stresses(displacements)
        von_mises = solver.compute_von_mises_stresses(stresses)
        
        print("\nResults summary:")
        print(f"Maximum von Mises stress: {np.max(von_mises):.2f} MPa")
        
        # Final gap state
        bottom_final = nodes[bottom_center_node] + displacements[3*bottom_center_node:3*bottom_center_node+3]
        top_final = nodes[top_center_node] + displacements[3*top_center_node:3*top_center_node+3]
        final_gap = np.dot(top_final - bottom_final, gap_material.contact_direction)
        
        print(f"\nFinal gap configuration:")
        print(f"Initial gap: {gap:.6f} mm")
        print(f"Final gap: {final_gap:.6f} mm")
        print(f"Change: {final_gap - gap:.6f} mm")
        
        # Export results
        solver.export_to_vtk('gap_test_results.vtu', displacements, stresses)
        print("\nResults exported to 'gap_test_results.vtu'")
        
    except Exception as e:
        print(f"\nSimulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    example_gapuni()