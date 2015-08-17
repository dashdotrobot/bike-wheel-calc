#!/usr/bin/env python

'Finite-element solver for performing stress analysis on a bicycle wheel.'

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as pp
import copy
import re

from femsolution import FEMSolution
from rigidbody import RigidBody
from spokesection import SpokeSection
from rimsection import RimSection
from wheelgeometry import WheelGeometry
from helpers import *

EL_RIM = 1
EL_SPOKE = 2
N_RIM = 1
N_HUB = 2
N_REF = 3
TWOPI = 2*np.pi


class RimElement:

    # MEMBERS
    # id       unique number
    # fem      parent FEM model
    # section  RimSection object containing rim properties
    # nodes

    def calc_el_stiff(self):
        """Calculate stiffness matrix for a single rim element.

        For details, see R. Palaninathan, P.S. Chandrasekharan,
        Computers and Structures, 4(21), pp. 663-669, 1985."""

        node1_pos = self.fem.get_node_pos(self.nodes[0])
        node2_pos = self.fem.get_node_pos(self.nodes[1])

        ref = np.array([0, 0, 0])
        sec = self.section

        d = node2_pos - node1_pos  # beam orientation vector
        r1 = node1_pos - ref       # radial vector to node 1
        R = np.sqrt(r1.dot(r1))    # radius of curvature

        # angle subtended by arc segment
        phi0 = 2*np.arcsin(np.sqrt(d.dot(d)) / (2*R))

        # local coordinate system
        e1 = d / np.sqrt(d.dot(d))  # radial vector
        e3 = np.array([0, 0, -1])   # axial vector
        e2 = np.cross(e3, e1)       # tangential vector

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

        a[0,0] = R*N/(2*sec.young_mod*sec.area) + sec.K2*R*B/(2*sec.shear_mod*sec.area) + C*R**3/(2*sec.young_mod*sec.I33)
        a[0,1] = R*D/(2*sec.young_mod*sec.area) - sec.K2*R*D/(2*sec.shear_mod*sec.area) + S*R**3/(sec.young_mod*sec.I33)
        a[1,0] = a[0,1]
        
        a[1,1] = B*R/(2*sec.young_mod*sec.area) + sec.K2*R*N/(2*sec.shear_mod*sec.area) + B*R**3/(2*sec.young_mod*sec.I33)

        a[0,5] = F*R**2/(sec.young_mod*sec.I33)
        a[5,0] = F*R**2/(sec.young_mod*sec.I33)
        a[1,5] = H*R**2/(sec.young_mod*sec.I33)
        a[5,1] = a[1,5]
  
        a[2,2] = sec.K3*R*phi0/(sec.shear_mod*sec.area) + C*R**3/(2*sec.shear_mod*sec.I11) + B*R**3/(2*sec.young_mod*sec.I22);
        
        a[2,3] = R**2/2*(B/(sec.young_mod*sec.I22) - V/(sec.shear_mod*sec.I11))
        a[3,2] = a[2,3]
        a[2,4] = R**2/2*(2*S/(sec.shear_mod*sec.I11) - D/(sec.young_mod*sec.I22))
        a[4,2] = a[2,4]
        a[3,3] = R/2*(N/(sec.shear_mod*sec.I11) + B/(sec.young_mod*sec.I22))
        a[3,4] = D*R/2*(1/(sec.shear_mod*sec.I11) - 1/(sec.young_mod*sec.I22))
        a[4,3] = a[3,4]
        a[4,4] = R/2*(B/(sec.shear_mod*sec.I11) + N/(sec.young_mod*sec.I22))
        
        a[5,5] = R*phi0/(sec.young_mod*sec.I33)

        # Flexibility matrix for node 2 DOFs
        b = a.copy()
        b[0,1] = -a[0,1]
        b[1,0] = -a[1,0]
        b[1,5] = -a[1,5]
        b[5,1] = -a[5,1]
        b[2,4] = -a[2,4]
        b[4,2] = -a[4,2]
        b[3,4] = -a[3,4]
        b[4,3] = -a[4,3]

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

    def calc_stresses(self, u):
        'Calculate average stresses at node 1'

        k_rim = self.calc_el_stiff()
        u_el = u[self.get_dofs()]

        f_el = np.array(k_rim.dot(u_el)).flatten()

        r = self.fem.get_node_pos(self.nodes[0])  # radial vector

        e2 = r / np.sqrt(r.dot(r))  # radial unit vector
        e3 = np.array([0, 0, 1])    # outward wheel normal (axial) vector
        e1 = np.cross(e2, e3)       # tangential unit vector

        f = f_el[0:3]  # total force at node 1
        m = f_el[3:6]  # total moment at node 1

        # Generalized internal force tuple:
        #  Tension
        #  Shear (in-plane)
        #  Shear (out-of-plane)
        #  Twisting moment
        #  Wobbling moment
        #  Squashing moment
        f_int = (f.dot(e1), f.dot(e2), f.dot(e3),
                 m.dot(e1), m.dot(e2), m.dot(e3))

        return f_int

    def get_dofs(self):
        dof_n1 = 6*self.nodes[0] + np.arange(6)
        dof_n2 = 6*self.nodes[1] + np.arange(6)
        dofs = np.concatenate((dof_n1, dof_n2))

        return dofs

    def __init__(self, fem, section, n1, n2):
        self.fem = fem
        self.section = section
        self.nodes = [n1, n2]

        self.type = EL_RIM


class SpokeElement:

    # MEMBERS
    # id       unique number
    # fem      parent FEM model
    # section  SpokeSection object containing spoke properties
    # nodes    node ids
    # side:    1 (drive side) or -1 (non-drive side)
    # offset:  nipple offset from rim centerline (in direction of spoke side)

    def calc_el_stiff(self):

        node1_pos = self.fem.get_node_pos(self.nodes[0])
        node2_pos = self.fem.get_node_pos(self.nodes[1])

        # Tangent vector
        e1 = node2_pos - node1_pos
        l = np.sqrt(e1.dot(e1))
        e1 = e1 / l  # convert to unit vector
        e2 = np.cross(e1, np.array([0, 0, 1]))
        e2 = e2 / np.sqrt(e2.dot(e2))
        e3 = np.cross(e1, e2)

        # do not allow negative spoke tension (compression)
        tension = self.tension
        if tension < 0:
            # if self.verbose:
                # print('*** Spoke {:d} has negative pre-tension. Result may be inaccurate'.format(e))
            tension = 0

        k_n = self.section.area*self.section.young_mod / l
        k_t = tension / l
        k_b = 3 * self.section.young_mod*self.section.I / l**3

        k_spoke = np.matrix(np.zeros((12, 12)))
        k_spoke[0::6, 0::6] = k_n * np.matrix([[1, -1], [-1, 1]])
        k_spoke[1::6, 1::6] = (k_t + k_b) * np.matrix([[1, -1], [-1, 1]])
        k_spoke[2::6, 2::6] = (k_t + k_b) * np.matrix([[1, -1], [-1, 1]])

        # rotation matrix to global coordinates
        Tg = np.matrix(np.zeros((3, 3)))
        Tg[:, 0] = e1.reshape((3, 1))
        Tg[:, 1] = e2.reshape((3, 1))
        Tg[:, 2] = e3.reshape((3, 1))
        # Tg[3::, 3] = e1.reshape((3, 1))
        # Tg[3::, 4] = e2.reshape((3, 1))
        # Tg[3::, 5] = e3.reshape((3, 1))

        # Apply rotation matrix to each sub matrix
        for i in range(4):
            for j in range(4):
                k_spoke[3*i:3*(i+1), 3*j:3*(j+1)] = \
                    Tg * k_spoke[3*i:3*(i+1), 3*j:3*(j+1)] * Tg.T

        return k_spoke

    def calc_stresses(self, u):
        n1 = self.nodes[0]  # hub node
        n2 = self.nodes[1]  # rim node

        # radial vector
        e1 = self.fem.get_node_pos(n2) - self.fem.get_node_pos(n1)
        e1 = e1 / np.sqrt(e1.dot(e1))  # convert to unit vector

        k_spoke = self.calc_el_stiff()
        u_el = u[self.get_dofs()]

        f_el = np.array(k_spoke.dot(u_el)).flatten()

        # Generalized stress tuple:
        #  Tension
        return (e1.dot(f_el[6:9]), )

    def get_dofs(self):
        dof_n1 = 6*self.nodes[0] + np.arange(6)
        dof_n2 = 6*self.nodes[1] + np.arange(6)
        dofs = np.concatenate((dof_n1, dof_n2))

        return dofs

    def __init__(self, fem, section, n1, n2, side):
        self.fem = fem
        self.section = section
        self.nodes = [n1, n2]
        self.side = side
        self.tension = 0

        self.type = EL_SPOKE


class BicycleWheelFEM:
    'Finite-element implementation and methods'

    def get_node_pos(self, node_id):
        'Return an NdArray of [X,Y,Z] position of node.'
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

    def calc_stiff_mat(self):
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
        for el in self.elements:

            dofs = el.get_dofs()

            node1_pos = self.get_node_pos(el.nodes[0])
            node2_pos = self.get_node_pos(el.nodes[1])

            k_el = el.calc_el_stiff()

            self.k_global[np.ix_(dofs, dofs)] = self.k_global[dofs][:, dofs] +\
                k_el

        # print "***"
        # print "*** STIFFNESS MATRIX"
        # print "***"
        # for i in range(len(self.x_nodes)):
            # for j in range(len(self.x_nodes)):
                # print "{:d}, {:d}".format(i, j)
                # submat = self.k_global[6*i:6*(i+1), 6*j:6*(j+1)]
                # if np.any(submat):
                    # print self.k_global[6*i:6*(i+1), 6*j:6*(j+1)]
                # else:
                    # print "ZEROS"

    def add_rigid_body(self, rigid_body):

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
        'Calculates matrices which encode rigid body constraints'

        # Convert stiffness equation into reduced equation
        #   U  = C * U_reduced
        #   F_reduced = B * F
        #   F_reduced = (B * K * C) * U_reduced

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
        'Remove a rigid body constraint'

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
        'Add a concentrated force (Neumann boundary condition).'

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
        self.calc_stiff_mat()

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
        dof_rxn_red = [6*self.node_r_id[i/6] + i % 6 for i in range(6*self.n_nodes) if self.bc_const[i]]
        dof_rxn = np.where(self.bc_const)[0]
        rxn = rxn_red[dof_rxn_red]

        soln.nodal_rxn = np.zeros((self.n_nodes, 6))
        soln.nodal_rxn[dof_rxn / 6, dof_rxn % 6] = rxn

        # Calculate element stresses
        for el in self.elements:
            soln.el_stress.append(el.calc_stresses(u))

        if self.verbose:
            print('# ---------------------------------------')

        return soln

    def solve(self, pretension=None, verbose=True):

        self.verbose = verbose
        self.el_pretension = np.zeros(len(self.el_type))

        if pretension is not None:

            # set initial pretension
            self.el_pretension = pretension * np.ones(len(self.el_type))

            # solve
            soln_1 = self.solve_iteration()

            # update spoke tensions
            el_spokes = np.where(self.el_type == EL_SPOKE)
            self.el_pretension[el_spokes] = self.el_pretension[el_spokes] + \
                np.array(soln_1.get_spoke_tension())

        # solve with updated element tensions
        soln_2 = self.solve_iteration()

        return soln_2

    def __init__(self, geom, rim_sec, spoke_sec, verbose=False):

        self.verbose = verbose

        self.geom = geom
        self.rim_sec = rim_sec
        self.spoke_sec = spoke_sec

        # Rim nodes
        self.x_nodes = geom.d_rim/2 * np.sin(geom.a_rim_nodes)
        self.y_nodes = -geom.d_rim/2 * np.cos(geom.a_rim_nodes)
        self.z_nodes = np.zeros(len(self.x_nodes))
        self.type_nodes = N_RIM * np.ones(len(self.x_nodes))

        # Hub nodes
        diam_hub = np.array([geom.d1_hub*(geom.s_hub_nodes[i] < 0) +
                             geom.d2_hub*(geom.s_hub_nodes[i] > 0)
                             for i in range(geom.n_hub_nodes)])

        width_hub = np.array([-geom.w1_hub*(geom.s_hub_nodes[i] < 0) +
                              geom.w2_hub*(geom.s_hub_nodes[i] > 0)
                              for i in range(geom.n_hub_nodes)])

        self.x_nodes = np.append(self.x_nodes,
                                 diam_hub/2 * np.sin(geom.a_hub_nodes))
        self.y_nodes = np.append(self.y_nodes,
                                 -diam_hub/2 * np.cos(geom.a_hub_nodes))
        self.z_nodes = np.append(self.z_nodes, width_hub)
        self.type_nodes = np.append(self.type_nodes,
                                    N_HUB * np.ones(len(geom.a_hub_nodes)))

        self.n_nodes = len(self.x_nodes)

        # rim elements connectivity
        self.el_n1 = np.array(range(geom.n_rim_nodes))
        self.el_n2 = np.array(range(1, geom.n_rim_nodes) + [0])
        self.el_type = EL_RIM * np.ones(geom.n_rim_nodes)

        self.elements = []
        for e in range(geom.n_rim_nodes):
            n1 = self.el_n1[e]
            n2 = self.el_n2[e]
            self.elements.append(RimElement(self, rim_sec, n1, n2))
            self.elements[-1].id = e

        n_el = len(self.elements)

        # spoke elements connectivity
        self.el_n1 = np.concatenate((self.el_n1,
                                     geom.lace_hub_n - 1 + geom.n_rim_nodes))
        self.el_n2 = np.concatenate((self.el_n2, geom.lace_rim_n - 1))
        self.el_type = np.concatenate((self.el_type,
                                       EL_SPOKE*np.ones(len(geom.lace_hub_n))))

        for e in range(len(geom.lace_hub_n)):
            # subtract 1 to convert human-friendly ID to 0-based index
            n1 = geom.lace_hub_n[e] + geom.n_rim_nodes - 1
            n2 = geom.lace_rim_n[e] - 1
            side = geom.s_hub_nodes[geom.lace_hub_n[e] - 1]
            self.elements.append(SpokeElement(self, spoke_sec, n1, n2, side))
            self.elements[-1].id = e + n_el

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
