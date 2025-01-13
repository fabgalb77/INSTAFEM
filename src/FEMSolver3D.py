import numpy as np
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

class ElementType(Enum):
    """Enumeration of supported element types"""
    TET4 = auto()   # 4-node tetrahedral element
    SPRING = auto() # 2-node spring element
    GAPUNI = auto() # 2-node gap element for node-to-node contact

class BoundaryCondition:
    """Class representing a boundary condition"""
    def __init__(self, node_ids: List[int], dofs: List[int], values: List[float]):
        """
        Parameters:
        node_ids: List of node IDs where BC is applied
        dofs: List of DOFs (0,1,2 for x,y,z) to constrain for each node
        values: Values to apply (displacement for Dirichlet, force for Neumann)
        """
        self.node_ids = np.array(node_ids)
        self.dofs = np.array(dofs)
        self.values = np.array(values)

class Material:
    """Base class for materials"""
    def __init__(self, name: str):
        self.name = name

class LinearElastic(Material):
    """Linear elastic material"""
    def __init__(self, name: str, E: float, nu: float):
        super().__init__(name)
        self.E = E     # Expect E in MPa
        self.nu = nu
        self.G = E / (2 * (1 + nu))
        self.lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    def get_D_matrix(self) -> np.ndarray:
        """Return the constitutive matrix"""
        D = np.zeros((6, 6))
        # Normal stress terms
        for i in range(3):
            for j in range(3):
                D[i, j] = self.lambda_
            D[i, i] += 2 * self.G
        # Shear stress terms
        for i in range(3, 6):
            D[i, i] = self.G
        return D

class NonlinearSpring(Material):
    """Nonlinear spring material defined by (force, elongation) data points"""
    def __init__(self, name: str, force_elongation_data: np.ndarray):
        super().__init__(name)
        # force_elongation_data should be a 2D array with columns: [elongation, force]
        self.data = force_elongation_data
        # Sort data by elongation
        self.data = self.data[self.data[:, 0].argsort()]
    
    def get_force(self, elongation: float) -> float:
        """Interpolate to get force at given elongation"""
        elongations = self.data[:, 0]
        forces = self.data[:, 1]
        return np.interp(elongation, elongations, forces)
    
    def get_stiffness(self, elongation: float) -> float:
        """Compute stiffness as derivative of force w.r.t elongation"""
        elongations = self.data[:, 0]
        forces = self.data[:, 1]
        if elongation <= elongations[0]:
            # Use first two points to compute derivative
            k = (forces[1] - forces[0]) / (elongations[1] - elongations[0])
        elif elongation >= elongations[-1]:
            # Use last two points
            k = (forces[-1] - forces[-2]) / (elongations[-1] - elongations[-2])
        else:
            # Find interval containing elongation
            idx = np.searchsorted(elongations, elongation, side='right') - 1
            # Ensure idx is within valid range
            idx = min(idx, len(elongations) - 2)
            idx = max(idx, 0)
            k = (forces[idx+1] - forces[idx]) / (elongations[idx+1] - elongations[idx])
        return k

class GapUniMaterial(Material):
    """Gap element material for node-to-node contact"""
    def __init__(self, name: str, initial_clearance: Optional[float] = None, 
                 contact_direction: Optional[np.ndarray] = None, penalty_parameter: float = 1e6):
        super().__init__(name)
        self.initial_clearance = initial_clearance
        self.contact_direction = contact_direction
        self.penalty_parameter = penalty_parameter  # Penalty parameter for enforcing contact

class ContactSurface:
    """Class representing a contact surface."""
    def __init__(self, name: str, element_indices: List[int], mesh: 'Mesh'):
        self.name = name
        self.element_indices = element_indices
        self.faces = []       # List of faces (tuples of node indices)
        self.face_normals = []  # Corresponding face normals
        self.nodes = set()    # Set of node indices involved in the surface
        self.extract_external_faces(mesh)

    def extract_external_faces(self, mesh: 'Mesh'):
        """Extract external faces and compute their normals."""
        face_dict = {}
        for elem_idx in self.element_indices:
            element_nodes, _ = mesh.elements[ElementType.TET4][elem_idx]
            # Define faces with consistent node ordering
            faces = [
                (element_nodes[0], element_nodes[1], element_nodes[2]),
                (element_nodes[0], element_nodes[1], element_nodes[3]),
                (element_nodes[0], element_nodes[2], element_nodes[3]),
                (element_nodes[1], element_nodes[2], element_nodes[3]),
            ]
            for face in faces:
                # Sort face nodes for consistent identification
                sorted_face = tuple(sorted(face))
                face_dict.setdefault(sorted_face, []).append((face, elem_idx))

        # External faces are those that appear only once
        external_faces = [faces[0][0] for faces in face_dict.values() if len(faces) == 1]

        # Compute normals for external faces
        for face in external_faces:
            n0, n1, n2 = face
            p0, p1, p2 = mesh.nodes[n0], mesh.nodes[n1], mesh.nodes[n2]
            normal = np.cross(p1 - p0, p2 - p0)
            normal /= np.linalg.norm(normal)
            self.faces.append(face)
            self.face_normals.append(normal)
            self.nodes.update(face)
        self.face_normals = np.array(self.face_normals)

class Mesh:
    """Class representing a finite element mesh"""
    def __init__(self, nodes: np.ndarray):
        """
        Parameters:
        nodes: Array of node coordinates (N x 3)
        """
        self.nodes = nodes
        self.elements: Dict[ElementType, List[Tuple[np.ndarray, Material]]] = {
            ElementType.TET4: [],
            ElementType.SPRING: [],
            ElementType.GAPUNI: []
        }
        self.dirichlet_bcs: List[BoundaryCondition] = []
        self.neumann_bcs: List[BoundaryCondition] = []
    
    def add_elements(self, element_type: ElementType, connectivity: np.ndarray, 
                    material: Material):
        """Add elements of a given type with their material"""
        for conn in connectivity:
            self.elements[element_type].append((conn, material))
    
    def add_dirichlet_bc(self, bc: BoundaryCondition):
        """Add a Dirichlet (displacement) boundary condition"""
        self.dirichlet_bcs.append(bc)
    
    def add_neumann_bc(self, bc: BoundaryCondition):
        """Add a Neumann (force) boundary condition"""
        self.neumann_bcs.append(bc)
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    def get_node_coordinates(self, element_nodes: np.ndarray) -> np.ndarray:
        """Get coordinates of nodes for a given element"""
        return self.nodes[element_nodes]
    
    def find_closest_node(self, target_coords: np.ndarray, 
                         tolerance: float = 1e-6) -> int:
        """Find the node closest to the target coordinates"""
        distances = np.linalg.norm(self.nodes - target_coords, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] > tolerance:
            print(f"Warning: Closest node is {distances[closest_idx]*1000:.2f} mm from target position")
        return closest_idx

    def add_contact_surface(self, name: str, element_indices: List[int]):
        if not hasattr(self, 'contact_surfaces'):
            self.contact_surfaces: Dict[str, ContactSurface] = {}
        self.contact_surfaces[name] = ContactSurface(name, element_indices, self)

@jit(nopython=True)
def _compute_geometric_stiffness_core(dN_stress, dN_xyz, eye3):
    K_g = np.zeros((12, 12))  # Could be pre-allocated if Numba allows
    for i in range(4):
        for j in range(4):
            scalar = 0.0
            for k in range(3):
                scalar += dN_stress[i, k] * dN_xyz[j, k]
            K_g[3*i:3*i+3, 3*j:3*j+3] = scalar * eye3
    return K_g


class NonlinearFEMSolver3D:
    """FEM solver for 3D problems with geometric nonlinearity"""
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.n_dof = 3 * mesh.n_nodes
        
        # Tetrahedral shape function derivatives with respect to natural coordinates
        self.dN = np.array([
            [-1.0, -1.0, -1.0],  # Node 0
            [ 1.0,  0.0,  0.0],  # Node 1
            [ 0.0,  1.0,  0.0],  # Node 2
            [ 0.0,  0.0,  1.0]   # Node 3
        ])
        
        # Single integration point coordinates (barycentric)
        self.gp = np.array([[0.25, 0.25, 0.25]])
        self.weights = np.array([1.0])  # Full integration for tetrahedral

        # Solution parameters
        self.max_iterations = 20
        self.tolerance = 1e-8
        
        self.eye3 = np.eye(3)
        self.J_template = np.array([
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1]
        ], dtype=np.float64)
        
        # Pre-allocated arrays
        self.stress_matrix = np.zeros((3, 3))
        self.dN_xyz = np.zeros((4, 3))
        self.dN_stress = np.zeros((4, 3))
        self.K_g = np.zeros((12, 12))
        
        # For assembly
        self.max_elements = 15000
        self.element_matrices = np.zeros((self.max_elements, 12, 12))
        self.element_forces = np.zeros((self.max_elements, 12))
        self.element_dofs = np.zeros((self.max_elements, 12), dtype=np.int32)

        # Adaptive load stepping parameters
        self.damping_factor = 1.
        self.lambda_target = 1.0      # Target load factor (total load to be applied)
        self.delta_lambda = 0.1       # Initial load increment
        self.min_delta_lambda = 0.01  # Minimum allowable load increment
        self.max_delta_lambda = 0.5   # Maximum allowable load increment
        self.iteration_threshold = 5  # Iterations after which to adjust load step

        self.lagrange_multipliers = {}
        if hasattr(self.mesh, 'contact_pairs'):
            for slave_name, _ in self.mesh.contact_pairs:
                slave_surface = self.mesh.contact_surfaces[slave_name]
                for node_idx in slave_surface.nodes:
                    self.lagrange_multipliers[node_idx] = 0.0  # Initialize λ_n to zero

    def compute_contact_contributions(self, u: np.ndarray) -> Tuple[List, List, List, np.ndarray, Dict[int, float]]:
        """Compute contact forces and stiffness matrix contributions."""
        data = []
        rows = []
        cols = []
        f_contact = np.zeros(self.n_dof)
        penetrations = {}

        if not hasattr(self.mesh, 'contact_pairs'):
            return data, rows, cols, f_contact, penetrations

        penalty = self.penalty_parameter  # Ensure penalty parameter is defined

        for slave_name, master_name in self.mesh.contact_pairs:
            slave_surface = self.mesh.contact_surfaces[slave_name]
            master_surface = self.mesh.contact_surfaces[master_name]

            for node_idx in slave_surface.nodes:
                # Current position of the slave node
                node_disp = u[3 * node_idx:3 * node_idx + 3]
                node_coords = self.mesh.nodes[node_idx] + node_disp

                # Initialize minimum distance
                min_distance = None
                closest_face = None
                closest_normal = None
                master_face_nodes = None

                # Check for penetration with master faces
                for face in master_surface.faces:
                    n0, n1, n2 = face
                    # Get current positions of master face nodes
                    p0 = self.mesh.nodes[n0] + u[3 * n0:3 * n0 + 3]
                    p1 = self.mesh.nodes[n1] + u[3 * n1:3 * n1 + 3]
                    p2 = self.mesh.nodes[n2] + u[3 * n2:3 * n2 + 3]

                    # Compute normal
                    normal = np.cross(p1 - p0, p2 - p0)
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm == 0:
                        continue  # Degenerate face
                    normal /= normal_norm

                    # Compute signed distance from node to plane of the face
                    distance = np.dot(normal, node_coords - p0)

                    # Ensure the normal points from the master to the slave
                    if distance > 0:
                        normal = -normal
                        distance = -distance

                    if distance < 0:
                        # Node is penetrating the face
                        # Project slave node onto master face plane
                        proj_point = node_coords - distance * normal

                        # Check if projection is inside the triangle using barycentric coordinates
                        v0 = p2 - p0
                        v1 = p1 - p0
                        v2 = proj_point - p0

                        d00 = np.dot(v0, v0)
                        d01 = np.dot(v0, v1)
                        d11 = np.dot(v1, v1)
                        d20 = np.dot(v2, v0)
                        d21 = np.dot(v2, v1)

                        denom = d00 * d11 - d01 * d01
                        if denom == 0:
                            continue  # Degenerate triangle
                        v = (d11 * d20 - d01 * d21) / denom
                        w = (d00 * d21 - d01 * d20) / denom
                        u_bary = 1 - v - w

                        if (u_bary >= -1e-6) and (v >= -1e-6) and (w >= -1e-6):
                            # Projection is inside the triangle
                            if min_distance is None or distance < min_distance:
                                min_distance = distance
                                closest_face = face
                                closest_normal = normal
                                master_face_nodes = [n0, n1, n2]
                                barycentric_coords = [u_bary, v, w]

                if min_distance is not None:
                    # Penetration depth (positive value)
                    penetration = -min_distance
                    penetrations[node_idx] = penetration

                    # Compute contact force
                    f_n = penalty * penetration * closest_normal

                    dofs_slave = [3 * node_idx + i for i in range(3)]
                    f_contact[dofs_slave] += f_n

                    # Distribute the reaction force to master nodes using barycentric coordinates
                    N_master = np.array(barycentric_coords)
                    master_nodes = master_face_nodes

                    for i, node in enumerate(master_nodes):
                        dofs_master = [3 * node + j for j in range(3)]
                        f_contact[dofs_master] -= N_master[i] * f_n  # Reaction force

                    # Compute contact stiffness contributions (simplified)
                    K_c_ss = penalty * np.outer(closest_normal, closest_normal)

                    # Assemble stiffness matrix contributions for slave node
                    for i_local, i_global in enumerate(dofs_slave):
                        for j_local, j_global in enumerate(dofs_slave):
                            data.append(K_c_ss[i_local, j_local])
                            rows.append(i_global)
                            cols.append(j_global)

                    # Assemble stiffness matrix contributions for master nodes
                    for i_master_i, node_i in enumerate(master_nodes):
                        dofs_i = [3 * node_i + a for a in range(3)]
                        N_i = N_master[i_master_i]
                        for i_master_j, node_j in enumerate(master_nodes):
                            dofs_j = [3 * node_j + b for b in range(3)]
                            N_j = N_master[i_master_j]
                            # K_c_mm = N_i * N_j * K_c_ss
                            for a in range(3):
                                for b in range(3):
                                    val = N_i * N_j * K_c_ss[a, b]
                                    data.append(val)
                                    rows.append(dofs_i[a])
                                    cols.append(dofs_j[b])

                    # Assemble cross terms between slave and master nodes
                    for i_local_slave, i_global_slave in enumerate(dofs_slave):
                        for i_master, node_master in enumerate(master_nodes):
                            dofs_master = [3 * node_master + j for j in range(3)]
                            N = N_master[i_master]
                            # K_c_sm = - N * K_c_ss
                            for j_local_master, j_global_master in enumerate(dofs_master):
                                val = -N * K_c_ss[i_local_slave, j_local_master % 3]
                                data.append(val)
                                rows.append(i_global_slave)
                                cols.append(j_global_master)

        return data, rows, cols, f_contact, penetrations

    def get_B_matrix(self, natural_coords: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """Compute B matrix for tetrahedral element using volume coordinates"""
        # Shape function derivatives in natural coordinates
        dN = np.array([
            [-1, -1, -1],  # Node 0 (origin)
            [ 1,  0,  0],  # Node 1 (along x)
            [ 0,  1,  0],  # Node 2 (along y)
            [ 0,  0,  1]   # Node 3 (along z)
        ])

        # Transform to physical coordinates directly
        dN_xyz = np.zeros((4, 3))
        for i in range(4):
            dN_xyz[i] = np.linalg.solve(jacobian.T, dN[i])
        
        # Initialize B matrix
        B = np.zeros((6, 12))
        
        # Fill B matrix with proper strain-displacement relations
        for i in range(4):  # For each node
            idx = 3 * i  # Starting index for this node's DOFs
            
            # Normal strains (diagonal terms)
            B[0, idx]   = dN_xyz[i, 0]          # εxx = du/dx
            B[1, idx+1] = dN_xyz[i, 1]          # εyy = dv/dy
            B[2, idx+2] = dN_xyz[i, 2]          # εzz = dw/dz
            
            # Shear strains (off-diagonal terms)
            # γxy = du/dy + dv/dx
            B[3, idx]   = dN_xyz[i, 1]  # du/dy
            B[3, idx+1] = dN_xyz[i, 0]  # dv/dx
            
            # γyz = dv/dz + dw/dy
            B[4, idx+1] = dN_xyz[i, 2]  # dv/dz
            B[4, idx+2] = dN_xyz[i, 1]  # dw/dy
            
            # γxz = du/dz + dw/dx
            B[5, idx]   = dN_xyz[i, 2]  # du/dz
            B[5, idx+2] = dN_xyz[i, 0]  # dw/dx
        
        return B

    def compute_geometric_stiffness(self, B: np.ndarray, stress: np.ndarray, 
                                  node_coords: np.ndarray) -> np.ndarray:
        """Compute geometric stiffness matrix for an element"""
        # Reuse pre-allocated stress matrix
        self.stress_matrix[0, 0] = stress[0]
        self.stress_matrix[0, 1] = self.stress_matrix[1, 0] = stress[3]
        self.stress_matrix[0, 2] = self.stress_matrix[2, 0] = stress[5]
        self.stress_matrix[1, 1] = stress[1]
        self.stress_matrix[1, 2] = self.stress_matrix[2, 1] = stress[4]
        self.stress_matrix[2, 2] = stress[2]
        
        # Use matrix multiplication with pre-allocated template
        J = self.J_template @ node_coords
        
        # Solve for dN_xyz - can't use out parameter here
        self.dN_xyz[:] = np.linalg.solve(J.T, self.dN.T).T
        
        # Compute dN_stress using pre-allocated array
        np.matmul(self.dN_xyz, self.stress_matrix, out=self.dN_stress)
        
        # Use the Numba-optimized core function
        return _compute_geometric_stiffness_core(self.dN_stress, self.dN_xyz, self.eye3)

    def compute_element_matrices(self, node_coords: np.ndarray, u_e: np.ndarray, 
                           material: LinearElastic) -> Tuple[np.ndarray, np.ndarray]:
        """Compute element matrices with consistent sign convention"""
        # Get edge vectors and volume
        x21 = node_coords[1] - node_coords[0]
        x31 = node_coords[2] - node_coords[0]
        x41 = node_coords[3] - node_coords[0]
        volume = abs(np.dot(x21, np.cross(x31, x41))) / 6.0 # Volume in mm^3
        
        # Form Jacobian from edge vectors
        J = np.column_stack([x21, x31, x41])
        J_inv = np.linalg.inv(J)
        
        # Get shape function derivatives in physical coordinates
        dN = np.array([
            [-1.0, -1.0, -1.0],  # Node 0
            [ 1.0,  0.0,  0.0],  # Node 1
            [ 0.0,  1.0,  0.0],  # Node 2
            [ 0.0,  0.0,  1.0]   # Node 3
        ])
        dN_xyz = dN @ J_inv
        
        # Form B matrix
        B = np.zeros((6, 12))
        for i in range(4):
            i3 = 3*i
            # Normal strains
            B[0, i3]   = dN_xyz[i,0]  # εxx
            B[1, i3+1] = dN_xyz[i,1]  # εyy
            B[2, i3+2] = dN_xyz[i,2]  # εzz
            
            # Engineering shear strains
            B[3, i3]   = dN_xyz[i,1]  # γxy
            B[3, i3+1] = dN_xyz[i,0]
            
            B[4, i3+1] = dN_xyz[i,2]  # γyz
            B[4, i3+2] = dN_xyz[i,1]
            
            B[5, i3]   = dN_xyz[i,2]  # γxz
            B[5, i3+2] = dN_xyz[i,0]
        
        # Compute strain and stress
        strain = B @ u_e
        D = material.get_D_matrix()
        stress = D @ strain
        
        # Element matrices
        K_e = volume * B.T @ D @ B
        f_int = volume * B.T @ stress
        
        return K_e, f_int, strain, stress, volume, B

    def compute_spring_element_matrices(self, node_coords: np.ndarray, u_e: np.ndarray, 
                                        material: NonlinearSpring) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness matrix and internal force vector for a nonlinear spring element.
        
        Parameters:
        node_coords: Coordinates of the two nodes (2 x 3)
        u_e: Displacements of the two nodes (6,)
        material: NonlinearSpring material object
        
        Returns:
        K_e: Element stiffness matrix (6 x 6)
        f_int_e: Element internal force vector (6,)
        """
        # Get initial length and direction
        x1 = node_coords[0]
        x2 = node_coords[1]
        u1 = u_e[:3]
        u2 = u_e[3:]
        
        delta_u = u2 - u1
        delta_x = x2 - x1
        L0 = np.linalg.norm(delta_x)
        if L0 == 0:
            raise ValueError("Zero-length spring element")
        # Current length
        L = np.linalg.norm(delta_x + delta_u)
        # Elongation
        elongation = L - L0
        # Unit vector in the current configuration
        n = (delta_x + delta_u) / L
        # Get force and stiffness from material
        force = material.get_force(elongation)
        stiffness = material.get_stiffness(elongation)
        # Internal force vector (6,)
        f_int_e = np.zeros(6)
        f_int_e[:3] = -force * n
        f_int_e[3:] = force * n
        # Element stiffness matrix (6 x 6)
        K_e = stiffness * np.outer(np.r_[-n, n], np.r_[-n, n])
        return K_e, f_int_e

    def compute_gapuni_element_matrices(self, node_coords: np.ndarray, u_e: np.ndarray, 
                                  material: GapUniMaterial) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness matrix and internal force vector for a GAPUNI element.
        
        Parameters:
        -----------
        node_coords: np.ndarray (2, 3)
            Initial coordinates of the two nodes
        u_e: np.ndarray (6,)
            Current displacement vector [u1x, u1y, u1z, u2x, u2y, u2z]
        material: GapUniMaterial
            Material properties including penalty parameter
        
        Returns:
        --------
        K_e: np.ndarray (6, 6)
            Element stiffness matrix
        f_int_e: np.ndarray (6,)
            Internal force vector
        """
        # Get node positions and displacements
        x1, x2 = node_coords
        u1, u2 = u_e[:3], u_e[3:]
        
        # Current positions
        x1_current = x1 + u1
        x2_current = x2 + u2
        
        # Get contact direction
        if material.contact_direction is None:
            initial_vector = x2 - x1
            initial_distance = np.linalg.norm(initial_vector)
            if initial_distance == 0:
                raise ValueError("Zero initial distance in GAPUNI element")
            contact_direction = initial_vector / initial_distance
        else:
            contact_direction = material.contact_direction
            contact_direction = contact_direction / np.linalg.norm(contact_direction)
        
        # Get initial clearance
        if material.initial_clearance is None:
            initial_clearance = np.dot(x2 - x1, contact_direction)
        else:
            initial_clearance = material.initial_clearance
        
        # Current gap vector and distance
        gap_vector = x2_current - x1_current
        current_distance = np.dot(gap_vector, contact_direction)
        
        # Compute penetration (negative means penetration)
        penetration = initial_clearance - current_distance
        
        # Initialize matrices
        K_e = np.zeros((6, 6))
        f_int_e = np.zeros(6)
        
        if penetration > 0:
            # Gap is closed - apply penalty force
            penalty = material.penalty_parameter
            
            # Contact force magnitude (positive in compression)
            force_magnitude = penalty * penetration
            
            # Internal force vector (opposite forces on each node)
            f1 = force_magnitude * contact_direction  # Force on node 1
            f2 = -f1                                 # Equal and opposite force on node 2
            
            f_int_e[:3] = f1
            f_int_e[3:] = f2
            
            # Stiffness matrix
            K_local = penalty * np.outer(contact_direction, contact_direction)
            
            # Assemble into global stiffness matrix
            K_e[:3, :3] = K_local
            K_e[:3, 3:] = -K_local
            K_e[3:, :3] = -K_local
            K_e[3:, 3:] = K_local
        
        return K_e, f_int_e

    def assemble_system_nonlinear(self, displacements: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        #Assemble global system with geometric nonlinearity and spring elements.
        # Use lists to store element matrices, forces, and DOFs
        element_matrices = []
        element_forces = []
        element_dofs = []

        # Process TET4 elements
        for i, (element_nodes, material) in enumerate(self.mesh.elements[ElementType.TET4]):
            #if i % 500 == 0:
            #    print(f"Processing TET4 element {i}/{len(self.mesh.elements[ElementType.TET4])}")
            # Get element displacements
            dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in element_nodes]).flatten()
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            u_e = displacements[dofs]
            # Compute element matrices
            K_e, f_e, _, _, _, _ = self.compute_element_matrices(node_coords, u_e, material)
            element_matrices.append(K_e)
            element_forces.append(f_e)
            element_dofs.append(dofs)

        # Process SPRING elements
        for i, (element_nodes, material) in enumerate(self.mesh.elements.get(ElementType.SPRING, [])):
            print(f"Processing SPRING element {i}/{len(self.mesh.elements[ElementType.SPRING])}")
            node1, node2 = element_nodes
            dofs = np.array([3 * node1, 3 * node1 + 1, 3 * node1 + 2,
                             3 * node2, 3 * node2 + 1, 3 * node2 + 2])
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            u_e = displacements[dofs]
            K_e, f_e = self.compute_spring_element_matrices(node_coords, u_e, material)
            element_matrices.append(K_e)
            element_forces.append(f_e)
            element_dofs.append(dofs)

        # Process GAPUNI elements
        for i, (element_nodes, material) in enumerate(self.mesh.elements.get(ElementType.GAPUNI, [])):
            if i % 500 == 0:
                print(f"Processing GAPUNI element {i}/{len(self.mesh.elements[ElementType.GAPUNI])}")
            node1, node2 = element_nodes
            dofs = np.array([3 * node1, 3 * node1 + 1, 3 * node1 + 2,
                             3 * node2, 3 * node2 + 1, 3 * node2 + 2])
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            u_e = displacements[dofs]
            K_e, f_e = self.compute_gapuni_element_matrices(node_coords, u_e, material)
            element_matrices.append(K_e)
            element_forces.append(f_e)
            element_dofs.append(dofs)

        # Assemble global stiffness matrix and internal force vector
        n_dofs = self.n_dof
        data = []
        rows = []
        cols = []
        f_int = np.zeros(n_dofs)
        for K_e, f_e, dofs in zip(element_matrices, element_forces, element_dofs):
            for i_local, i_global in enumerate(dofs):
                f_int[i_global] += f_e[i_local]
                for j_local, j_global in enumerate(dofs):
                    data.append(K_e[i_local, j_local])
                    rows.append(i_global)
                    cols.append(j_global)

        # Contact contributions
        contact_data, contact_rows, contact_cols, f_contact, penetrations = self.compute_contact_contributions(displacements)
        data.extend(contact_data)
        rows.extend(contact_rows)
        cols.extend(contact_cols)
        f_int += f_contact

        K = csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
        return K, f_int, penetrations


    def solve_nonlinear(self, forces: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve nonlinear system including contact with adaptive load stepping."""
        lambda_total = 0.0       # Current total load factor

        if forces is None:
            forces = np.zeros(self.n_dof)
        
        # Initialize
        u = np.zeros(self.n_dof)
        u_previous = u.copy()

        # Identify fixed DOFs and their prescribed values
        fixed_dofs = []
        prescribed_displacements = {}
        for bc in self.mesh.dirichlet_bcs:
            for node, dof, value in zip(bc.node_ids, bc.dofs, bc.values):
                global_dof = 3 * node + dof
                fixed_dofs.append(global_dof)
                prescribed_displacements[global_dof] = value

        fixed_dofs = np.array(fixed_dofs)
        free_dofs = np.setdiff1d(np.arange(self.n_dof), fixed_dofs)

        print("\nStarting nonlinear solution with adaptive load stepping:")
        print(f"Number of DOFs: {len(free_dofs)} free, {len(fixed_dofs)} fixed")

        while lambda_total < self.lambda_target:
            # Ensure the last load step reaches the target load factor
            if lambda_total + self.delta_lambda > self.lambda_target:
                self.delta_lambda = self.lambda_target - lambda_total

            lambda_total += self.delta_lambda
            print(f"\nApplying load factor λ = {lambda_total:.4f}")

            # Store the previous solution in case we need to retry the load step
            u_trial = u.copy()
            converged = False

            for iteration in range(1, self.max_iterations + 1):
                # Apply prescribed displacements for the current load factor
                for dof in fixed_dofs:
                    u[dof] = lambda_total * prescribed_displacements[dof]

                # Assemble system
                K, f_int, penetrations = self.assemble_system_nonlinear(u)

                # Compute residual
                residual = forces * lambda_total - f_int

                # Apply Neumann boundary conditions (if any)
                for bc in self.mesh.neumann_bcs:
                    for node, dof, value in zip(bc.node_ids, bc.dofs, bc.values):
                        global_dof = 3 * node + dof
                        residual[global_dof] += lambda_total * value

                # Apply displacement boundary conditions to residual
                residual[fixed_dofs] = 0.0  # Prescribed DOFs have zero residual

                # Check convergence
                residual_norm = np.linalg.norm(residual[free_dofs])
                if iteration == 1:
                    initial_residual_norm = residual_norm

                relative_residual = residual_norm / (initial_residual_norm + 1e-15)

                print(f"  Iteration {iteration}: Residual = {residual_norm:.3e}, Relative = {relative_residual:.3e}")

                if residual_norm < self.tolerance:
                    print(f"  Converged in {iteration} iterations")
                    converged = True
                    break

                # Solve system using iterative solver with preconditioning
                K_ff = K[free_dofs, :][:, free_dofs]
                residual_f = residual[free_dofs]

                # Create AMG preconditioner
                ml = pyamg.smoothed_aggregation_solver(K_ff)
                M = ml.aspreconditioner()

                try:
                    # Solve using LGMRES from scipy.sparse.linalg with AMG preconditioner
                    du_free, info = spla.lgmres(K_ff, residual_f, M=M, maxiter=1000)
                    du_free *= self.damping_factor
                except Exception as e:
                    print(f"    Linear solver failed: {str(e)}")
                    converged = False
                    break

                if info != 0:
                    print(f"    Iterative solver did not converge within the allowed iterations. Info: {info}")
                    converged = False
                    break

                # Update displacements directly without line search
                u[free_dofs] += du_free

                # Apply prescribed displacements for the current load factor
                for dof in fixed_dofs:
                    u[dof] = lambda_total * prescribed_displacements[dof]

                # Update Lagrange multipliers if using augmented Lagrangian
                if hasattr(self, 'lagrange_multipliers'):
                    # Recompute penetrations with updated displacements
                    _, _, _, _, penetrations = self.compute_contact_contributions(u)
                    for node_idx in self.lagrange_multipliers.keys():
                        penetration = penetrations.get(node_idx, 0.0)
                        self.lagrange_multipliers[node_idx] += self.penalty_parameter * penetration

            if converged:
                # Adjust load increment if convergence was too fast or slow
                if iteration <= self.iteration_threshold and self.delta_lambda < self.max_delta_lambda:
                    self.delta_lambda *= 1.5  # Increase load increment
                    self.delta_lambda = min(self.delta_lambda, self.max_delta_lambda)
            else:
                # If not converged, reduce the load increment and retry
                print("  Did not converge, reducing load increment and retrying...")
                lambda_total -= self.delta_lambda  # Roll back the load factor
                u = u_previous.copy()              # Roll back to the previous solution
                self.delta_lambda *= 0.5           # Reduce load increment
                if self.delta_lambda < self.min_delta_lambda:
                    print("  Load increment too small, stopping simulation.")
                    break
                continue  # Retry the load step with the reduced load increment

            # Store the solution from the converged load step
            u_previous = u.copy()

        return u


    def compute_stresses(self, displacements: np.ndarray) -> np.ndarray:
        """Compute stresses with improved analysis"""
        n_elements = len(self.mesh.elements[ElementType.TET4])
        stresses = np.zeros((n_elements, 6))
        element_centroids = np.zeros((n_elements, 3))
        
        # Track maximum stresses and locations
        max_stress = 0.0
        max_elem = 0
        
        print("\nComputing stresses...")
        for i, (element_nodes, material) in enumerate(self.mesh.elements[ElementType.TET4]):
            # Get element data
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in element_nodes]).flatten()
            u_e = displacements[dofs]
            
            # Compute centroid
            centroid = np.mean(node_coords, axis=0)
            element_centroids[i] = centroid
            
            # Compute stress
            _, _, _, stress, _, _ = self.compute_element_matrices(node_coords, u_e, material)
            stresses[i] = stress
            
            # Track maximum
            s_max = np.max(np.abs(stress))
            if s_max > max_stress:
                max_stress = s_max
                max_elem = i
        
        # Get maximum stress location
        max_location = element_centroids[max_elem]
        max_stress_tensor = stresses[max_elem]
        
        print("\nStress analysis:")
        print(f"Maximum stress location: x = {max_location[0]:.1f} mm (from fixed end)")
        print("\nMaximum stress tensor (MPa):")
        print(f"σxx = {max_stress_tensor[0]:.1f}")
        print(f"σyy = {max_stress_tensor[1]:.1f}")
        print(f"σzz = {max_stress_tensor[2]:.1f}")
        print(f"τxy = {max_stress_tensor[3]:.1f}")
        print(f"τyz = {max_stress_tensor[4]:.1f}")
        print(f"τxz = {max_stress_tensor[5]:.1f}")
        
        # Analyze stresses at mid-length (where beam theory should be valid)
        length = np.max(element_centroids[:, 0])
        mid_elements = np.where(
            np.abs(element_centroids[:, 0] - length/2) < length/20
        )[0]
        
        mid_stresses = stresses[mid_elements]
        mid_max_stress = np.max(np.abs(mid_stresses[:, 0]))  # Max normal stress
        
        print("\nMid-length stress analysis:")
        print(f"Number of elements analyzed: {len(mid_elements)}")
        print(f"Maximum normal stress: {mid_max_stress:.1f} MPa")
        
        # Compute stress statistics by region
        regions = [
            (0, 0.2),      # Near fixed end
            (0.2, 0.4),    # First quarter
            (0.4, 0.6),    # Middle
            (0.6, 0.8),    # Third quarter
            (0.8, 1.0)     # Near load
        ]
        
        print("\nStress distribution along beam:")
        for start, end in regions:
            region_elements = np.where(
                (element_centroids[:, 0] >= start*length) & 
                (element_centroids[:, 0] < end*length)
            )[0]
            
            region_stresses = stresses[region_elements]
            max_normal = np.max(np.abs(region_stresses[:, 0]))
            max_shear = np.max(np.abs(region_stresses[:, 3:]))
            
            print(f"\nRegion {start*100:.0f}%-{end*100:.0f}% of length:")
            print(f"  Maximum normal stress: {max_normal:.1f} MPa")
            print(f"  Maximum shear stress: {max_shear:.1f} MPa")
        
        return stresses

    def compute_von_mises_stresses(self, stresses: np.ndarray) -> np.ndarray:
        """Compute von Mises stress from stress tensor components"""
        σxx, σyy, σzz, τxy, τyz, τxz = stresses.T
        
        von_mises = np.sqrt(0.5 * ((σxx - σyy)**2 + 
                                  (σyy - σzz)**2 + 
                                  (σzz - σxx)**2 + 
                                  6*(τxy**2 + τyz**2 + τxz**2)))
        
        return von_mises

    def compute_spring_element_data(self, displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute data for spring element visualization"""
        spring_elements = self.mesh.elements[ElementType.SPRING]
        n_springs = len(spring_elements)
        
        # Initialize arrays for spring data
        spring_forces = np.zeros(n_springs)
        spring_elongations = np.zeros(n_springs)
        spring_lengths = np.zeros(n_springs)
        
        for i, (element_nodes, material) in enumerate(spring_elements):
            # Get node coordinates and displacements
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            dofs = np.array([3*n + j for n in element_nodes for j in range(3)])
            u_e = displacements[dofs]
            
            # Compute initial and current configurations
            x1, x2 = node_coords
            u1, u2 = u_e[:3], u_e[3:]
            
            # Initial length
            L0 = np.linalg.norm(x2 - x1)
            
            # Current configuration
            x1_current = x1 + u1
            x2_current = x2 + u2
            L = np.linalg.norm(x2_current - x1_current)
            
            # Compute elongation and force
            elongation = L - L0
            force = material.get_force(elongation)
            
            # Store results
            spring_forces[i] = force
            spring_elongations[i] = elongation
            spring_lengths[i] = L
        
        return spring_forces, spring_elongations, spring_lengths

    def export_to_vtk(self, filename: str, displacements: np.ndarray, stresses: np.ndarray):
        """Export mesh and results to VTK format with spring visualization"""
        if not filename.endswith('.vtu'):
            filename += '.vtu'

        # Reshape displacements for point data
        disp_field = displacements.reshape(-1, 3)

        # Compute von Mises stress for tetrahedral elements
        von_mises = self.compute_von_mises_stresses(stresses)

        # Convert stresses to tensor form for tetrahedral elements
        stress_tensor = np.zeros((len(stresses), 9))
        stress_tensor[:, 0] = stresses[:, 0]  # σxx
        stress_tensor[:, 4] = stresses[:, 1]  # σyy
        stress_tensor[:, 8] = stresses[:, 2]  # σzz
        stress_tensor[:, 1] = stress_tensor[:, 3] = stresses[:, 3]  # τxy
        stress_tensor[:, 5] = stress_tensor[:, 7] = stresses[:, 4]  # τyz
        stress_tensor[:, 2] = stress_tensor[:, 6] = stresses[:, 5]  # τxz

        # Prepare elements and types
        elements_tet = [e[0] for e in self.mesh.elements[ElementType.TET4]]
        elements_spring = [e[0] for e in self.mesh.elements[ElementType.SPRING]]
        
        # Get spring data
        spring_forces = []
        spring_elongations = []
        spring_lengths = []
        for element_nodes, material in self.mesh.elements[ElementType.SPRING]:
            # Get node coordinates and displacements
            node_coords = self.mesh.get_node_coordinates(element_nodes)
            dofs = np.array([3*n + j for n in element_nodes for j in range(3)])
            u_e = displacements[dofs]
            
            # Compute initial and current configurations
            x1, x2 = node_coords
            u1, u2 = u_e[:3], u_e[3:]
            
            # Initial length
            L0 = np.linalg.norm(x2 - x1)
            
            # Current configuration
            x1_current = x1 + u1
            x2_current = x2 + u2
            L = np.linalg.norm(x2_current - x1_current)
            
            # Compute elongation and force
            elongation = L - L0
            force = material.get_force(elongation)
            
            # Store results
            spring_forces.append(force)
            spring_elongations.append(elongation)
            spring_lengths.append(L)
        
        spring_forces = np.array(spring_forces)
        spring_elongations = np.array(spring_elongations)
        spring_lengths = np.array(spring_lengths)

        # Create separate VTU files for springs and tetrahedral elements
        # 1. Springs file
        spring_filename = filename.replace('.vtu', '_springs.vtu')
        
        # Create point data dictionary for springs
        spring_point_data = {
            'Displacement': disp_field,
            'SpringForce': np.zeros(len(self.mesh.nodes)),  # Initialize with zeros
            'SpringElongation': np.zeros(len(self.mesh.nodes))  # Initialize with zeros
        }
        
        # Map spring values to their nodes
        for i, (element_nodes, _) in enumerate(self.mesh.elements[ElementType.SPRING]):
            for node in element_nodes:
                spring_point_data['SpringForce'][node] = spring_forces[i]
                spring_point_data['SpringElongation'][node] = spring_elongations[i]
        
        # Write springs file
        write_vtu_file(spring_filename, self.mesh.nodes, elements_spring, 
                       [3] * len(elements_spring),  # VTK_LINE = 3
                       spring_point_data, 
                       {'SpringForce': spring_forces,
                        'SpringElongation': spring_elongations,
                        'SpringLength': spring_lengths}, 
                       ['Displacement'])
        
        # 2. Tetrahedral elements file
        tet_filename = filename.replace('.vtu', '_tets.vtu')
        
        # Point data for tetrahedral elements
        tet_point_data = {
            'Displacement': disp_field,
            'DisplacementMagnitude': np.linalg.norm(disp_field, axis=1)
        }
        
        # Cell data for tetrahedral elements
        tet_cell_data = {
            'StressTensor': stress_tensor,
            'vonMises': von_mises
        }
        
        # Write tetrahedral elements file
        write_vtu_file(tet_filename, self.mesh.nodes, elements_tet,
                       [10] * len(elements_tet),  # VTK_TETRA = 10
                       tet_point_data, tet_cell_data, ['Displacement'])
        
        print(f"\nResults exported to separate files:")
        print(f"Springs: {spring_filename}")
        print(f"Tetrahedral elements: {tet_filename}")
        print("\nTo visualize springs in Paraview:")
        print("1. Load both VTU files")
        print("2. For the springs file:")
        print("   - Select 'SpringForce' or 'SpringElongation' in the coloring dropdown")
        print("   - Apply 'Cell Data to Point Data' filter")
        print("   - Then apply 'Tube' filter")
        print("   - Adjust tube radius in Properties panel")
        print("3. For the tetrahedral file:")
        print("   - Color by 'vonMises' or other stress measures")
        print("4. Optional: Use 'Warp By Vector' filter with 'Displacement' to see deformation")

def write_vtu_file(filename: str,
                   nodes: np.ndarray,
                   elements: List[np.ndarray],
                   element_types: List[int],
                   point_data: Optional[Dict[str, np.ndarray]] = None,
                   cell_data: Optional[Dict[str, np.ndarray]] = None,
                   vector_fields: Optional[List[str]] = None):
    """
    Write mesh and results to VTU file format for Paraview with proper handling of mixed elements

    Parameters:
    filename: Output filename
    nodes: Node coordinates (N x 3)
    elements: List of element connectivity arrays
    element_types: List of VTK element types corresponding to each element
    point_data: Dictionary of nodal data
    cell_data: Dictionary of element data
    vector_fields: List of field names that should be treated as vectors
    """
    if vector_fields is None:
        vector_fields = []

    # Create XML header
    root = ET.Element('VTKFile')
    root.set('type', 'UnstructuredGrid')
    root.set('version', '1.0')
    root.set('byte_order', 'LittleEndian')
    root.set('header_type', 'UInt64')

    # Create unstructured grid element
    grid = ET.SubElement(root, 'UnstructuredGrid')
    piece = ET.SubElement(grid, 'Piece')
    piece.set('NumberOfPoints', str(len(nodes)))
    piece.set('NumberOfCells', str(len(elements)))

    # Write points
    points = ET.SubElement(piece, 'Points')
    points_data = ET.SubElement(points, 'DataArray')
    points_data.set('type', 'Float64')
    points_data.set('Name', 'Points')
    points_data.set('NumberOfComponents', '3')
    points_data.set('format', 'ascii')
    points_data.text = '\n'.join(' '.join(f'{coord:.10e}' for coord in node) for node in nodes)

    # Write cells (elements)
    cells = ET.SubElement(piece, 'Cells')

    # Connectivity
    connectivity_data = ET.SubElement(cells, 'DataArray')
    connectivity_data.set('type', 'Int64')
    connectivity_data.set('Name', 'connectivity')
    connectivity_data.set('format', 'ascii')
    connectivity_list = []
    for element in elements:
        connectivity_list.extend(element)
    connectivity_data.text = ' '.join(map(str, connectivity_list))

    # Offsets
    offsets_data = ET.SubElement(cells, 'DataArray')
    offsets_data.set('type', 'Int64')
    offsets_data.set('Name', 'offsets')
    offsets_data.set('format', 'ascii')
    offsets = []
    offset = 0
    for element in elements:
        offset += len(element)
        offsets.append(offset)
    offsets_data.text = ' '.join(map(str, offsets))

    # Types
    types_data = ET.SubElement(cells, 'DataArray')
    types_data.set('type', 'UInt8')
    types_data.set('Name', 'types')
    types_data.set('format', 'ascii')
    types_data.text = ' '.join(map(str, element_types))

    # Write point data
    if point_data:
        point_data_element = ET.SubElement(piece, 'PointData')
        for name, data in point_data.items():
            data_array = ET.SubElement(point_data_element, 'DataArray')
            data_array.set('type', 'Float64')
            data_array.set('Name', name)
            
            if name in vector_fields:
                data_array.set('NumberOfComponents', str(data.shape[1]))
            else:
                data_array.set('NumberOfComponents', '1')
            
            data_array.set('format', 'ascii')
            if data.ndim > 1 and name in vector_fields:
                data_array.text = '\n'.join(' '.join(f'{value:.10e}' for value in row) for row in data)
            else:
                data_array.text = '\n'.join(f'{value:.10e}' for value in data.flatten())

    # Write cell data with proper handling for different element types
    if cell_data:
        cell_data_element = ET.SubElement(piece, 'CellData')
        for name, data in cell_data.items():
            data_array = ET.SubElement(cell_data_element, 'DataArray')
            data_array.set('type', 'Float64')
            data_array.set('Name', name)
            
            # Determine components based on data shape and element type
            if data.ndim > 1:
                data_array.set('NumberOfComponents', str(data.shape[1]))
                formatted_data = []
                for i, elem_type in enumerate(element_types):
                    if elem_type == 3:  # Spring element
                        # For springs, pad with zeros to match tensor size
                        if name == "StressTensor":
                            formatted_data.extend(['0.0'] * 9)  # 9 components for stress tensor
                        else:
                            formatted_data.extend([f'{data[i, 0]:.10e}'] + ['0.0'] * (data.shape[1] - 1))
                    else:  # Tetrahedral element
                        formatted_data.extend(f'{value:.10e}' for value in data[i])
                data_array.text = '\n'.join(' '.join(row) for row in np.array(formatted_data).reshape(-1, data.shape[1]))
            else:
                data_array.set('NumberOfComponents', '1')
                data_array.text = '\n'.join(f'{value:.10e}' if not np.isnan(value) else '0.0' for value in data)

    # Write to file
    tree = ET.ElementTree(root)
    with open(filename, 'wb') as f:
        f.write(b'<?xml version="1.0"?>\n')
        tree.write(f, encoding='ascii', xml_declaration=False)
