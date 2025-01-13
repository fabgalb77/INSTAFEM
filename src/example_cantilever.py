from FEMSolver3D import *

def create_mesh_example():
    # Dimensions and material
    length = 100.0  # mm
    height = 10.0   # mm
    width = 10.0    # mm
    E = 200000.0    # MPa
    nu = 0.3
    P = 20000.0       # N (downward force)
    
    # Finer mesh discretization
    nx = 20 
    ny = 5  
    nz = 5  
    
    # Create nodal grid
    x = np.linspace(0, length, nx + 1)
    y = np.linspace(0, height, ny + 1)
    z = np.linspace(0, width, nz + 1)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    # Create mesh object
    mesh = Mesh(nodes)
    
    # Create elements
    elements = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                n0 = ix * (ny+1) * (nz+1) + iy * (nz+1) + iz
                n1 = n0 + (nz+1)
                n2 = n1 + 1
                n3 = n0 + 1
                n4 = n0 + (ny+1) * (nz+1)
                n5 = n4 + (nz+1)
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
    steel = LinearElastic("steel", E=E, nu=nu)
    mesh.add_elements(ElementType.TET4, np.array(elements), steel)
    
    # Fixed end
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < 1e-6)[0]
    mesh.add_dirichlet_bc(BoundaryCondition(
        fixed_nodes,
        [0, 1, 2] * len(fixed_nodes),
        [0.0] * (3 * len(fixed_nodes))
    ))
    
    # Find end nodes for load application
    end_nodes = np.where(np.abs(nodes[:, 0] - length) < 1e-6)[0]
    end_center_nodes = end_nodes[
        np.logical_and(
            np.abs(nodes[end_nodes, 1] - height/2) < height/4,
            np.abs(nodes[end_nodes, 2] - width/2) < width/4
        )
    ]
    
    # Apply load (negative for downward)
    P_per_node = -P / len(end_center_nodes)
    mesh.add_neumann_bc(BoundaryCondition(
        end_center_nodes.tolist(),
        [1] * len(end_center_nodes),
        [P_per_node] * len(end_center_nodes)
    ))
    
    # Store center node
    mesh.center_node = end_nodes[np.argmin(
        (nodes[end_nodes, 1] - height/2)**2 +
        (nodes[end_nodes, 2] - width/2)**2
    )]
       
    print("\nMesh properties:")
    print(f"Elements: {nx}x{ny}x{nz}")
    print(f"Number of nodes: {mesh.n_nodes}")
    print(f"Number of elements: {len(elements)}")
    print(f"Element size: {length/nx:.2f} x {height/ny:.2f} x {width/nz:.2f} mm")
    
    print("\nLoading:")
    print(f"Total load: {P:.1f} N downward")
    print(f"Load per node: {-P_per_node:.2f} N over {len(end_center_nodes)} nodes")
        
    return mesh

def test_nonlinear_solver():
    # Create mesh
    mesh = create_mesh_example()
    
    # Create solver
    solver = NonlinearFEMSolver3D(mesh)
    solver.tolerance = 1e-6
    solver.max_iterations = 25
    
    # Solve
    displacements = solver.solve_nonlinear()
    
    # Compute stresses
    stresses = solver.compute_stresses(displacements)
    von_mises = solver.compute_von_mises_stresses(stresses)
    
    # Get tip displacement
    center_node = mesh.center_node
    end_disp = displacements[3*center_node:3*center_node+3]
    
    # Find total applied force
    total_force = 0.0
    for bc in mesh.neumann_bcs:
        total_force += sum(bc.values)
 
    # Export results
    solver.export_to_vtk('cantilever_results.vtu', displacements, stresses)

    return {
        'displacements': displacements,
        'stresses': stresses,
        'von_mises': von_mises
    }

if __name__ == "__main__":
    test_nonlinear_solver()