from FEMSolver3D import *

def example_spring():
    # Beam geometry parameters
    length = 100.0  # Beam length in mm
    height = 10.0   # Beam height in mm
    width = 10.0    # Beam width in mm
    E = 200000      # Young's modulus in MPa (steel)
    nu = 0.3        # Poisson's ratio

    # Mesh parameters
    mesh_size = 2.0  # Approximate element size (in mm)

    # Convert mesh size to meters for Gmsh
    mesh_size_meters = mesh_size / 1000.0

    # Create beam geometry and mesh using pygmsh
    with pygmsh.geo.Geometry() as geom:
        # Define the beam as a box (dimensions converted to meters)
        beam = geom.add_box(
            x0=0.0,
            x1=length / 1000.0,  # Convert mm to meters
            y0=-width / 2 / 1000.0,
            y1=width / 2 / 1000.0,
            z0=-height / 2 / 1000.0,
            z1=height / 2 / 1000.0,
            mesh_size=mesh_size_meters 
        )

        # Generate mesh
        pygmsh_mesh = geom.generate_mesh()

    # Extract mesh data and convert back to mm
    nodes = pygmsh_mesh.points * 1000.0  # Convert from meters to mm
    cells_dict = pygmsh_mesh.cells_dict

    if "tetra" in cells_dict:
        cells = cells_dict["tetra"].astype(int)
    elif "tetra10" in cells_dict:
        # If higher-order elements are generated, extract their nodes
        cells = cells_dict["tetra10"][:, :4].astype(int)
    else:
        raise ValueError("No tetrahedral elements found in the mesh.")

    # Create mesh object
    mesh = Mesh(nodes)

    # Add material
    steel = LinearElastic("steel", E=E, nu=nu)
    mesh.add_elements(ElementType.TET4, cells, steel)

    # Fixed end
    fixed_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    node_ids = []
    dofs = []
    values = []
    for node in fixed_nodes:
        for dof in [0, 1, 2]:
            node_ids.append(node)
            dofs.append(dof)
            values.append(0.0)
    mesh.add_dirichlet_bc(BoundaryCondition(
        node_ids=node_ids,
        dofs=dofs,
        values=values
    ))

    # Identify free end nodes (beam tip)
    free_end_nodes = np.where(np.isclose(nodes[:, 0], length))[0]
    # Get the node closest to the center of the beam tip
    tip_node = free_end_nodes[np.argmin(np.linalg.norm(nodes[free_end_nodes][:, 1:], axis=1))]

    # Constrain y and z DOFs of the tip node to prevent rigid body motion
    mesh.add_dirichlet_bc(BoundaryCondition(
        node_ids=[tip_node, tip_node],
        dofs=[1, 2],
        values=[0.0, 0.0]
    ))

    # Add an extra node for the spring attachment point, offset slightly in x-direction
    spring_offset = 0.01  # Small offset in mm
    spring_node_coords = nodes[tip_node] + np.array([spring_offset, 0.0, 0.0])
    nodes = np.vstack([nodes, spring_node_coords])
    spring_node_id = len(nodes) - 1
    mesh.nodes = nodes  # Update mesh nodes

    # Add the nonlinear spring element between tip node and spring node
    # Define force-elongation data for the spring
    force_elongation_data = np.array([
        [0.0, 0.0],
        [1., 10000.0],
        [2., 20000.0],
        [3., 30000.0],
        [4., 40000.0],
        [5., 50000.0]
    ])
    spring_material = NonlinearSpring("nonlinear_spring", force_elongation_data)
    mesh.add_elements(ElementType.SPRING, np.array([[tip_node, spring_node_id]]), spring_material)

    # Constrain the spring node in y and z directions
    mesh.add_dirichlet_bc(BoundaryCondition(
        node_ids=[spring_node_id, spring_node_id],
        dofs=[1, 2],
        values=[0.0, 0.0]
    ))

    # Apply load at the spring node in x-direction
    applied_force = 1000.0  # N
    mesh.add_neumann_bc(BoundaryCondition(
        node_ids=[spring_node_id],
        dofs=[0],
        values=[applied_force]
    ))

    # Create solver
    solver = NonlinearFEMSolver3D(mesh)
    solver.tolerance = 1e-6
    solver.max_iterations = 50

    # Solve
    displacements = solver.solve_nonlinear()

    # Compute stresses in the beam
    stresses = solver.compute_stresses(displacements)

    # Get tip displacement
    tip_disp = displacements[3 * tip_node:3 * tip_node + 3]
    spring_node_disp = displacements[3 * spring_node_id:3 * spring_node_id + 3]
    spring_elongation = np.linalg.norm((nodes[spring_node_id] + spring_node_disp) - (nodes[tip_node] + tip_disp)) - spring_offset

    # Expected elongation from force-elongation data
    expected_elongation = np.interp(applied_force, force_elongation_data[:, 1], force_elongation_data[:, 0])

    # Print results
    print("\nValidation Test: Mixed Tetrahedral and Nonlinear Spring Elements")
    print(f"Applied force at spring node: {applied_force:.1f} N")
    print(f"Tip node displacement: {tip_disp}")
    print(f"Spring node displacement: {spring_node_disp}")
    print(f"Spring elongation: {spring_elongation:.6f} mm")
    print(f"Expected elongation from data: {expected_elongation:.6f} mm")
    print(f"Difference: {spring_elongation - expected_elongation:.6e} mm")

    # Export results for visualization
    solver.export_to_vtk('example_springs_results.vtu', displacements, stresses)

    return displacements, stresses

if __name__ == "__main__":
    example_spring()