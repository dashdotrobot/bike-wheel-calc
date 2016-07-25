import numpy as np
from femsolution import FEMSolution
from helpers import *
from bicycle_wheel import *
from rigidbody import *

EL_RIM = 1
EL_SPOKE = 2
N_RIM = 1
N_HUB = 2
N_REF = 3


class BicycleWheelFEM:
    """Finite-element solver for performing stress analysis bicycle wheels.

    Creates a finite-element model from a BicycleWheel object and solves the
    linear elasticity equations K*u = f subject to constraints and boundary
    conditions.
    """

    def get_node_pos(self, node_id):
        'Return the [X,Y,Z] position of a node as an NdArray.'
        return np.array([self.x_nodes[node_id],
                         self.y_nodes[node_id],
                         self.z_nodes[node_id]])

    def get_rim_nodes(self):
        'Return node IDs of all nodes on the rim.'
        return np.where(self.type_nodes == N_RIM)[0]

    def get_hub_nodes(self):
        'Return node IDs of all nodes on the hub.'
        return np.where(self.type_nodes == N_HUB)[0]

    def get_rim_elements(self):
        'Return element IDs of all rim elements.'
        return np.where(self.el_type == EL_RIM)[0]

    def get_spoke_elements(self):
        'Return element IDs of all hub elements.'
        return np.where(self.el_type == EL_SPOKE)[0]

    def calc_spoke_stiff(self, el_id, s):
        'Calculate stiffness matrix for a single spoke.'

        n2 = self.el_n2[el_id]

        nip_pt = pol2rect(s.rim_pt)     # spoke nipple
        hub_pt = pol2rect(s.hub_pt)     # hub eyelet
        rim_pt = self.get_node_pos(n2)  # point on rim centroid

        # Beam coordinate system
        e1 = hub_pt - nip_pt                    # tangent vector
        l = np.sqrt(e1.dot(e1))
        e1 = e1 / l
        e2 = np.cross(e1, np.array([0, 0, 1]))  # normal vector
        e2 = e2 / np.sqrt(e2.dot(e2))
        e3 = np.cross(e1, e2)                   # second normal vector

        # axial stiffness (normal)
        k_n = s.EA / l

        # tension stiffness (transverse). No negative tension-stiffness
        k_t = max(0.0, self.el_prestress[el_id] / l)

        # bending stiffness (transverse)
        # Generally, bending stiffness is negligible. It is only present for
        # numerical stability in the case of radial spokes (vanishing torsional
        # stiffness).
        k_b = 3 * s.EA * (s.diameter**2 / 16) / l**3

        # Bar element stiffness matrix (FORCES ONLY) in beam coordinates
        k_spoke = np.matrix(np.zeros((12, 12)))
        k_spoke[0::6, 0::6] = k_n * np.matrix([[1, -1], [-1, 1]])
        k_spoke[1::6, 1::6] = (k_t + k_b) * np.matrix([[1, -1], [-1, 1]])
        k_spoke[2::6, 2::6] = (k_t + k_b) * np.matrix([[1, -1], [-1, 1]])

        # rotation matrix to global coordinates
        Tg = np.matrix(np.zeros((3, 3)))
        Tg[:, 0] = e1.reshape((3, 1))
        Tg[:, 1] = e2.reshape((3, 1))
        Tg[:, 2] = e3.reshape((3, 1))

        # Apply rotation matrix to each sub matrix
        for i in range(4):
            for j in range(4):
                k_spoke[3*i:3*(i+1), 3*j:3*(j+1)] = \
                    Tg * k_spoke[3*i:3*(i+1), 3*j:3*(j+1)] * Tg.T

        # Transformation matrices to account for spoke offset
        r = nip_pt - rim_pt
        Omega_r = skew_symm(r)

        # Right-multiply k_spoke by C to transform from u_nip -> u_rim
        for i in range(4):
            k_spoke[3*i:3*(i+1), 9::] = k_spoke[3*i:3*(i+1), 9::] - \
                Omega_r * k_spoke[3*i:3*(i+1), 6:9]

        # Left-multiply k_spoke by B to transform from f_nip -> f_rim
        for i in range(4):
            k_spoke[9::, 3*i:3*(i+1)] = k_spoke[9::, 3*i:3*(i+1)] - \
                Omega_r * k_spoke[6:9, 3*i:3*(i+1)]

        return k_spoke

    def calc_rim_stiff(self, el_id):
        'Calculate stiffness matrix for a single rim element.'

        n1 = self.el_n1[el_id]
        n2 = self.el_n2[el_id]

        # For details, see R. Palaninathan, P.S. Chandrasekharan,
        # Computers and Structures, 4(21), pp. 663-669, 1985.

        node1_pos = self.get_node_pos(n1)
        node2_pos = self.get_node_pos(n2)

        ref = np.array([0, 0, 0])  # reference point at wheel center

        d = node2_pos - node1_pos  # beam orientation vector
        r1 = node1_pos - ref       # radial vector to node 1
        R = np.sqrt(r1.dot(r1))    # radius of curvature

        # angle subtended by arc segment
        phi0 = 2*np.arcsin(np.sqrt(d.dot(d)) / (2*R))

        # local coordinate system
        e1 = d / np.sqrt(d.dot(d))  # radial vector
        e3 = np.array([0, 0, -1])   # axial vector
        e2 = np.cross(e3, e1)       # tangential vector

        # Material and section properties
        # Beam warping is neglected
        A = self.wheel.rim.area
        E = self.wheel.rim.young_mod
        G = self.wheel.rim.shear_mod
        I11 = self.wheel.rim.I11
        I22 = self.wheel.rim.I22
        I33 = self.wheel.rim.I33
        K2 = 0  # shear flexibility constant (0 = Euler-Bernoulli beam)
        K3 = 0  # shear flexibility constant (0 = Euler-Bernoulli beam)

        # Constants
        N = phi0 + np.sin(2*phi0)/2
        B = phi0 - np.sin(2*phi0)/2
        C = 3*phi0 + np.sin(2*phi0)/2 - 4*np.sin(phi0)
        S = 0.75 - np.cos(phi0) + np.cos(2*phi0)/4
        F = np.sin(phi0) - phi0
        H = np.cos(phi0) - 1
        V = 2*np.sin(phi0) - phi0 - np.sin(2*phi0)/2
        D = np.cos(2*phi0)/2 - 0.5

        # Initialize stiffness matrix
        k_r = np.matrix(np.zeros((12, 12)))

        # Flexibility matrix for node 1 DOFs
        a = np.matrix(np.zeros((6, 6)))

        a[0, 0] = R*N/(2*E*A) + K2*R*B/(2*G*A) + C*R**3/(2*E*I33)
        a[0, 1] = R*D/(2*E*A) - K2*R*D/(2*G*A) + S*R**3/(E*I33)
        a[1, 0] = a[0, 1]

        a[1, 1] = B*R/(2*E*A) + K2*R*N/(2*G*A) + B*R**3/(2*E*I33)

        a[0, 5] = F*R**2/(E*I33)
        a[5, 0] = F*R**2/(E*I33)
        a[1, 5] = H*R**2/(E*I33)
        a[5, 1] = a[1, 5]

        a[2, 2] = K3*R*phi0/(G*A) + C*R**3/(2*G*I11) + B*R**3/(2*E*I22)

        a[2, 3] = R**2/2*(B/(E*I22) - V/(G*I11))
        a[3, 2] = a[2, 3]
        a[2, 4] = R**2/2*(2*S/(G*I11) - D/(E*I22))
        a[4, 2] = a[2, 4]
        a[3, 3] = R/2*(N/(G*I11) + B/(E*I22))
        a[3, 4] = D*R/2*(1/(G*I11) - 1/(E*I22))
        a[4, 3] = a[3, 4]
        a[4, 4] = R/2*(B/(G*I11) + N/(E*I22))

        a[5, 5] = R*phi0/(E*I33)

        # Flexibility matrix for node 2 DOFs
        b = a.copy()
        b[0, 1] = -a[0, 1]
        b[1, 0] = -a[1, 0]
        b[1, 5] = -a[1, 5]
        b[5, 1] = -a[5, 1]
        b[2, 4] = -a[2, 4]
        b[4, 2] = -a[4, 2]
        b[3, 4] = -a[3, 4]
        b[4, 3] = -a[4, 3]

        # Transformation matrix from node 1 -> node 2
        al = np.cos(phi0)
        bt = np.sin(phi0)

        Tbar = np.matrix([[-al,  bt, 0, 0, 0, 0],
                          [-bt, -al, 0, 0, 0, 0],
                          [0,    0, -1, 0, 0, 0],
                          [0, 0, R*(1-al), -al, bt, 0],
                          [0, 0, -R*bt,    -bt, -al, 0],
                          [R*(1-al), R*bt, 0, 0, 0, -1]])

        # Transformation matrix from node 1 -> beam coordinates
        Tb = np.matrix(np.zeros((6, 6)))
        Tb[:3:, :3:] = np.matrix([[np.cos(phi0/2), -np.sin(phi0/2), 0],
                                 [np.sin(phi0/2),  np.cos(phi0/2), 0],
                                 [0,               0,              1]])
        Tb[3::, 3::] = Tb[:3:, :3:]

        # Transformation matrix from beam coordinates to global coordinates
        Tg = np.matrix(np.zeros((6, 6)))
        Tg[:3:, :3:] = np.matrix(np.vstack((e1, e2, e3)).T)
        Tg[3::, 3::] = Tg[:3:, :3:]

        # Assemble submatrices
        k_r[:6:, :6:] = np.linalg.inv(a)      # K_II
        k_r[6::, 6::] = np.linalg.inv(b)      # K_JJ
        k_r[6::, :6:] = Tbar * k_r[:6:, :6:]  # K_JI

        k_r[:6:, :6:] = Tg*Tb   * k_r[:6:, :6:] * Tb.T*Tg.T
        k_r[6::, 6::] = Tg*Tb.T * k_r[6::, 6::] * Tb  *Tg.T
        k_r[6::, :6:] = Tg*Tb.T * k_r[6::, :6:] * Tb.T*Tg.T
        k_r[:6:, 6::] = k_r[6::,:6:].T      # K_IJ (symm.)

        return k_r

    def calc_global_stiff(self):
        'Calculate global stiffness matrix by element scatter algorithm.'

        if self.verbose:
            print('# Calculating global stiffness matrix ---')
            print('# -- Nodes: {:d}'.format(self.n_nodes))
            print('# -- DOFs : {:d}'.format(6*self.n_nodes))
            print('# ---------------------------------------')
            print('')

        # Initialize empty stiffness matrix
        self.k_global = np.matrix(np.zeros((6*self.n_nodes, 6*self.n_nodes)))

        # Loop over all elements and scatter to global K matrix
        for el in range(len(self.el_type)):

            n1 = self.el_n1[el]
            n2 = self.el_n2[el]

            dofs = np.concatenate((6*n1 + np.arange(6), 6*n2 + np.arange(6)))

            if self.el_type[el] == EL_RIM:
                k_el = self.calc_rim_stiff(el)
            elif self.el_type[el] == EL_SPOKE:
                s = self.wheel.spokes[n1 - self.n_rim_nodes]
                k_el = self.calc_spoke_stiff(el, s)

            # Scatter to global matrix
            self.k_global[np.ix_(dofs, dofs)] = self.k_global[dofs][:, dofs] +\
                k_el

    def calc_spoke_stress(self, el_id, u):
        'Calculate tension in a spoke element.'

        n1 = self.el_n1[el_id]
        n2 = self.el_n2[el_id]

        s_num = self.el_s_num[el_id]
        s = self.wheel.spokes[s_num]  # spoke object

        # spoke vector
        nip_pt = pol2rect(s.rim_pt)     # spoke nipple
        hub_pt = pol2rect(s.hub_pt)     # hub eyelet
        e1 = nip_pt - hub_pt
        e1 = e1 / np.sqrt(e1.dot(e1))

        dofs = np.concatenate((6*n1 + np.arange(6), 6*n2 + np.arange(6)))

        k_el = self.calc_spoke_stiff(el_id, s)
        u_el = u[dofs]
        f_el = np.array(k_el.dot(u_el)).flatten()

        # Generalized stress tuple:
        #  Tension
        return (e1.dot(f_el[6:9]), )

    def calc_rim_stress(self, el_id, u):
        """Calculate internal forces in a rim element.

        Returns the internal forces at the first node of the rim element. The
        internal forces are defined at the nodes (not integration points)
        because the stiffness matrix is obtained by Castiliano's method.

        Returns:
            tuple:
                0: axial force
                1: transverse force (in-plane shear)
                2: transverse force (out-of-plane shear)
                3: twisting moment
                4: bending moment (out-of-plane)
                5: bending moment (in-plane)
        """

        n1 = self.el_n1[el_id]
        n2 = self.el_n2[el_id]

        # Local coordinates system at node 1
        n1_pos = self.get_node_pos(n1)
        n2_pos = self.get_node_pos(n2)

        e3_1 = np.cross(n1_pos, n2_pos) /\
            np.sqrt(n1_pos.dot(n1_pos) * n2_pos.dot(n2_pos))
        e1_1 = np.cross(n1_pos, e3_1) / np.sqrt(n1_pos.dot(n1_pos))
        e2_1 = np.cross(e3_1, e1_1)

        # Nodal degrees of freedom
        dofs = np.concatenate((6*n1 + np.arange(6), 6*n2 + np.arange(6)))

        # Calculate nodal forces
        k_el = self.calc_rim_stiff(el_id)
        u_el = u[dofs]
        f_el = np.array(k_el.dot(u_el)).flatten()

        return (e1_1.dot(f_el[0:3]), e2_1.dot(f_el[0:3]), e3_1.dot(f_el[0:3]),
                e1_1.dot(f_el[3:6]), e2_1.dot(f_el[3:6]), e3_1.dot(f_el[3:6]))

    def add_rigid_body(self, rigid_body):
        'Add a rigid body defined by the arg rigid_body.'

        # Check that nodes are not already assigned to rigid bodies
        for rig in self.rigid:
            in_rig = [i in rig.nodes for i in rigid_body.nodes]
            if any(in_rig):
                print('*** Nodes cannot belong to multiple rigid bodies')
                print('***  -- Node {:d}\n'.format(rigid_body.nodes[in_rig.index(True)]))
                return

        # Add new rigid body
        self.rigid.append(rigid_body)

        # Create a new node to reference the rigid body to
        self.n_nodes += 1
        rigid_body.node_id = self.n_nodes - 1
        self.x_nodes = np.append(self.x_nodes, rigid_body.pos[0])
        self.y_nodes = np.append(self.y_nodes, rigid_body.pos[1])
        self.z_nodes = np.append(self.z_nodes, rigid_body.pos[2])
        self.type_nodes = np.append(self.type_nodes, N_REF)
        self.bc_const.extend(6*[False])
        self.bc_force.extend(6*[False])
        self.bc_u = np.append(self.bc_u, 6*[0])
        self.bc_f = np.append(self.bc_f, 6*[0])

        if self.verbose:
            print('# Adding new rigid body: {:s}'.format(rigid_body.name))
            print('# -- Reference node {:d}\n'.format(rigid_body.node_id))

        # Recalculate reduction matrices
        self.calc_reduction_matrices()

    def calc_reduction_matrices(self):
        """Calculate matrices which encode rigid body constraints.

        Convert stiffness equation into reduced stiffness equation:
            U  = C * U_reduced
            F_reduced = B * F
            K_reduced = (B * K * C)
        """

        if not self.rigid:  # if there are no rigid bodies
            self.B = 1
            self.C = 1
            self.node_r_id = range(self.n_nodes)
            return

        # Re-calculate B and C matrices
        n_c = np.sum([r.n_nodes for r in self.rigid])
        self.C = np.mat(np.zeros((6*self.n_nodes, 6*(self.n_nodes - n_c))))
        self.B = np.mat(np.zeros((6*(self.n_nodes - n_c), 6*self.n_nodes)))

        self.node_r_id = [-1] * self.n_nodes
        for rig_id in range(len(self.rigid)):
            self.node_r_id[self.rigid[rig_id].node_id] = self.n_nodes - len(self.rigid) - n_c + rig_id

        n_r_n = 0
        for n in range(self.n_nodes):
            in_rigid = [n in r.nodes for r in self.rigid]  # list of logicals
            dof_n = 6*n + np.arange(6)  # IDs of DOFs associated with this node

            if not any(in_rigid):
                # new re-numbered node ID
                self.node_r_id[n] = n_r_n
                dof_r_n = 6*n_r_n + np.arange(6)
                n_r_n += 1

                self.C[dof_n, dof_r_n] = 1  # identity matrix
                self.B[dof_r_n, dof_n] = 1
            else:
                rig_i = in_rigid.index(True)  # Index of rigid body
                n_r_r = self.node_r_id[self.rigid[rig_i].node_id]  # Reduced index of rigid body node
                dof_r_r = 6*n_r_r + np.arange(6)

                r_c = self.get_node_pos(n) - self.rigid[rig_i].pos
                R = skew_symm(r_c)

                self.C[dof_n, dof_r_r] = 1
                self.C[np.ix_(dof_n[:3:], dof_r_r[3::])] = R

                self.B[dof_r_r, dof_n] = 1
                self.B[np.ix_(dof_r_r[3::], dof_n[:3:])] = R

        self.soln_updated = False

    def remove_rigid_body(self, rigid_body):
        'Remove a rigid body constraint.'

        # Confirm that the rigid body belongs to this model
        if rigid_body not in self.rigid:
            print('*** This rigid body does not exist in this model.')
            return

        # Remove from rigid bodies list
        self.rigid.remove(rigid_body)

        # Delete the reference node
        n = rigid_body.node_id
        self.n_nodes -= 1
        self.x_nodes = np.delete(self.x_nodes, n)
        self.y_nodes = np.delete(self.y_nodes, n)
        self.z_nodes = np.delete(self.z_nodes, n)
        self.type_nodes = np.delete(self.type_nodes, n)
        self.bc_u = np.delete(self.bc_u, 6*n + np.arange(6))
        self.bc_f = np.delete(self.bc_f, 6*n + np.arange(6))
        for _ in range(6):
            self.bc_const.pop(n)
            self.bc_force.pop(n)

        # Shift the node id for any subsequent rigid bodies down
        for r in self.rigid:
            if r.node_id > n:
                r.node_id -= 1

        # Unset reference node
        rigid_body.node_id = None

        # Recalculate reduction matrices
        self.calc_reduction_matrices()

    def add_constraint(self, node_id, dof, u=0):
        'Add a displacement constraint (Dirichlet boundary condition).'

        # Allow array input for node_id and/or dof
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]
        if not hasattr(dof, '__iter__'):
            dof = [dof]

        for n in node_id:
            for d in dof:

                dof_r = 6*n + d

                if not self.bc_force[dof_r]:
                    if not any([n in r.nodes for r in self.rigid]):
                        self.bc_const[dof_r] = True
                        self.bc_u[dof_r] = u
                        self.soln_updated = False
                    else:
                        print('\n*** Node {:d}: Cannot assign a force to a node in a rigid body\n'.format(n))
                else:
                    print('\n*** Node {:d}, DOF {:d}: Cannot assign a constraint and force simultaneously\n',format(n,d))

    def add_force(self, node_id, dof, f):
        'Add a concentrated force or moment (Neumann boundary condition).'

        dof_r = 6*node_id + dof

        if not self.bc_const[dof_r]:
            if not any([node_id in r.nodes for r in self.rigid]):
                self.bc_force[dof_r] = True
                self.bc_f[dof_r] = f
                self.soln_updated = False
            else:
                print('\n*** Node {:d}: Cannot assign a force to a node in a rigid body\n'.format(node_id))
        else:
            print('\n*** Node {:d}: Cannot assign a constraint and force simultaneously\n'.format(node_id))

    def remove_bc(self, node_id, dof):
        'Remove one or more boundary conditions.'

        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]
        if not hasattr(dof, '__iter__'):
            dof = [dof]

        for n in node_id:
            for d in dof:

                dof_r = 6*n + d

                self.bc_const[dof_r] = False
                self.bc_force[dof_r] = False

                self.bc_u[dof_r] = 0
                self.bc_f[dof_r] = 0

                self.soln_updated = False

    def solve_iteration(self):
        'Solve elasticity equations for nodal displacements.'

        # Form augmented, reduced stiffness matrix
        self.calc_global_stiff()

        if len(self.rigid) == 0:
            # No rigid bodies. Reduced node IDs are equal to node IDs
            self.node_r_id = np.arange(self.n_nodes, dtype=np.int16)

        k_red = self.B * self.k_global * self.C
        k_aug = k_red.copy()
        f_aug = np.zeros(k_aug.shape[0])

        # Apply constraints to nodes
        for dof_c in [d for d in range(6*self.n_nodes) if self.bc_const[d]]:
            dof_r = 6*self.node_r_id[int(dof_c) / 6] + dof_c % 6

            k_aug[dof_r] = 0
            k_aug[dof_r, dof_r] = 1

            f_aug[dof_r] = self.bc_u[dof_c]

        # Apply forces to nodes
        for dof_c in [d for d in range(6*self.n_nodes) if self.bc_force[d]]:
            dof_r = 6*self.node_r_id[int(dof_c) / 6] + dof_c % 6
            f_aug[dof_r] = self.bc_f[dof_c]

        # Solve for reduced nodal displacements
        if self.verbose:
            print('# Solving for nodal displacements -------')
            print('# -- Reduced system of equations:')
            print('#      Reduced DOFs: {:d}'.format(k_aug.shape[0]))
            print('#      Rank = {:d} / ({:d})'.format(np.linalg.matrix_rank(k_aug), k_aug.shape[0]))

        try:
            u_red = np.linalg.solve(k_aug, f_aug)
        except np.linalg.LinAlgError as e:
            print('\n*** ERROR: {:s}. Did you properly constrain all DOFs?'.format(e.message))

            # TODO: Give user meaningful info about missing constraints

            return False

        self.soln_updated = True

        # Create solution object
        soln = FEMSolution(self)

        # nodal displacements
        if self.rigid:
            u = np.array(self.C.dot(u_red)).flatten()
        else:
            u = u_red.flatten()

        soln.nodal_disp = np.zeros((self.n_nodes, 6))
        for d in range(6):
            soln.nodal_disp[:, d] = u[d::6]

        # nodal reaction forces
        rxn_red = np.array(k_red.dot(u_red) - f_aug).flatten()
        dof_rxn_red = [6*self.node_r_id[i/6] + i % 6
                       for i in range(6*self.n_nodes)
                       if self.bc_const[i]]
        dof_rxn = np.where(self.bc_const)[0]
        rxn = rxn_red[dof_rxn_red]

        soln.nodal_rxn = np.zeros((self.n_nodes, 6))
        soln.nodal_rxn[dof_rxn / 6, dof_rxn % 6] = rxn

        # TODO Calculate element stresses
        soln.el_prestress = self.el_prestress
        for el in range(len(self.el_type)):
            if self.el_type[el] == EL_SPOKE:
                soln.el_stress.append(self.calc_spoke_stress(el, u))
            else:
                soln.el_stress.append(self.calc_rim_stress(el, u))

        if self.verbose:
            print('# ---------------------------------------')

        return soln

    def solve(self, pretension=None, verbose=True):
        """Solve elasticity equations including the effect of prestress.

        Since the spoke stiffness depends on spoke tension, the elasticity
        equations are technically non-linear. If the pretension keyword is used
        the solve method first initializes the spoke tensions and then
        solves the linear stiffness equation by calling the solve_iteration()
        method. The changes in spoke tension are calculated and used to update
        the spoke tensions. The solve_iteration() method is called again using
        the updated spoke tensions. This method only requires 2 iterations to
        converge because the axial stiffness is orthogonal to the tension
        stiffness.
        """

        self.verbose = verbose

        if pretension is not None:

            # set initial pretension
            for e in self.get_spoke_elements():
                self.el_prestress[e] = pretension

            # solve
            soln1 = self.solve_iteration()

            # update spoke tensions
            for e in self.get_spoke_elements():
                self.el_prestress[e] = self.el_prestress[e] +\
                    soln1.el_stress[e][0]
        else:
            pretension = 0.0

        # solve with updated element tensions
        soln_2 = self.solve_iteration()

        # reset spoke prestress to initial prestress
        for e in self.get_spoke_elements():
            self.el_prestress[e] = pretension

        return soln_2

    def __init__(self, wheel, verbose=False):

        self.verbose = verbose

        self.wheel = wheel

        # Create a rim node at each unique spoke attachment point
        theta_rim_nodes = set()
        for s in self.wheel.spokes:
            theta_rim_nodes.add(s.rim_pt[1])

        theta_rim_nodes = sorted(list(theta_rim_nodes))

        # Rim nodes
        self.x_nodes = wheel.rim.radius * np.sin(theta_rim_nodes)
        self.y_nodes = -wheel.rim.radius * np.cos(theta_rim_nodes)
        self.z_nodes = np.zeros(len(self.x_nodes))
        self.type_nodes = N_RIM * np.ones(len(self.x_nodes))

        self.n_rim_nodes = len(self.type_nodes)

        # Hub nodes
        for s in self.wheel.spokes:
            r_h = s.hub_pt[0]
            theta_h = s.hub_pt[1]
            z_h = s.hub_pt[2]
            self.x_nodes = np.append(self.x_nodes, r_h*np.sin(theta_h))
            self.y_nodes = np.append(self.y_nodes, -r_h*np.cos(theta_h))
            self.z_nodes = np.append(self.z_nodes, z_h)
            self.type_nodes = np.append(self.type_nodes, N_HUB)

        self.n_nodes = len(self.x_nodes)

        # Create element connectivity matrix for rim nodes
        self.el_n1 = np.arange(self.n_rim_nodes)
        self.el_n2 = np.append(np.arange(1, self.n_rim_nodes), 0)
        self.el_type = EL_RIM * np.ones(len(self.el_n1))
        self.el_s_num = np.zeros(len(self.el_n1), dtype=np.int)

        # Add spoke elements
        for s_num, s in enumerate(self.wheel.spokes):
            r_node = np.where(s.rim_pt[1] == np.array(theta_rim_nodes))[0][0]

            self.el_n1 = np.append(self.el_n1, self.n_rim_nodes + s_num)
            self.el_n2 = np.append(self.el_n2, r_node)
            self.el_type = np.append(self.el_type, EL_SPOKE)
            self.el_s_num = np.append(self.el_s_num, s_num)

        # Spoke tension vector
        self.el_prestress = np.zeros(len(self.el_type))

        # rigid bodies
        self.rigid = []
        self.C = 1
        self.B = 1
        self.node_r_id = range(self.n_nodes)

        # constraints
        self.bc_const = [False]*(6*self.n_nodes)
        self.bc_force = [False]*(6*self.n_nodes)
        self.bc_f = np.zeros(6*self.n_nodes)
        self.bc_u = np.zeros(6*self.n_nodes)

        # stiffness matrices
        self.k_global = None

        # solution arrays
        self.soln_updated = False

# Testing code
if False:

    w = BicycleWheel()
    w.hub = w.Hub(diam1=0.04, width1=0.025)
    w.rim = w.Rim.general(radius=0.3,
                          area=100e-6,
                          I11=1000e-12,
                          I22=1000e-12,
                          I33=1000e-12,
                          Iw=0.0,
                          young_mod=69.0e9,
                          shear_mod=26.0e9)

    # w.lace_radial(n_spokes=36, diameter=1.5e-3, young_mod=210e9, offset=0.0)
    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.5e-3,
                 young_mod=210e9, offset=0.0)

    fem = BicycleWheelFEM(w, verbose=True)

    # Create a rigid body to constrain the hub nodes
    r_hub = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(r_hub)

    # Calculate radial stiffness. Apply an upward force to the bottom node
    fem.add_constraint(r_hub.node_id, range(6))
    fem.add_force(0, 1, 500)

    soln = fem.solve(pretension=1000)

    rim_bend_moment = [soln.el_stress[i][5] for i in range(36)]
    pp.plot(rim_bend_moment)
    pp.show()
