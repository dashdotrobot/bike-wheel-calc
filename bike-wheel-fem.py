#!/usr/bin/env python

'Finite-element solver for performing stress analysis on a bicycle wheel.'

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as pp
import copy
import re

# Element type definitions
EL_RIM = 1
EL_SPOKE = 2
TWOPI = 2*np.pi


def skew_symm(v):
    'Create a skew-symmetric tensor V from vector v such that V*u = v cross u.'

    return np.matrix([[0,     v[2], -v[1]],
                      [-v[2], 0,     v[0]],
                      [v[1], -v[0],  0   ]])
                    
def interp_periodic(x, y, xx, period=TWOPI):
    'Interpolate a periodic function with cubic spline matching slope at ends.'

    # Pad data by wrapping beginning and end to match derivatives
    x_pad = np.concatenate(([x[-1] - period], x, [x[0] + period, x[1] + period]))
    y_pad = np.concatenate(([y[-1]], y, [y[0], y[1]]))

    f_spline = interpolate.splrep(x_pad,y_pad)

    return interpolate.splev(xx, f_spline)


class BicycleWheelFEM:
    'Finite-element implementation and methods'
    
    def get_node_pos(self, node_id):
        'Return an NdArray of [X,Y,Z] position of node.'
        return np.array([self.x_nodes[node_id], 
                         self.y_nodes[node_id], 
                         self.z_nodes[node_id]])
    
    def get_rim_nodes(self):
        'Return node IDs of all nodes on the rim.'
        return np.arange(self.geom.n_rim_nodes)

    def get_hub_nodes(self):
        'Return node IDs of all nodes on the hub.'
        return np.arange(self.geom.n_hub_nodes) + self.geom.n_rim_nodes

    def get_spoke_tension(self, u):
        'Calculate tension in all spokes.'
        
        f_spokes = np.array(self.k_spokes.dot(u)).flatten()
        t_spokes = []
        
        # Iterate over spoke elements
        for e in [x for x in range(len(self.el_type)) if self.el_type[x] == EL_SPOKE]:
            
            n1 = self.el_n1[e]  # hub node
            n2 = self.el_n2[e]  # rim node
            
            e1 = self.get_node_pos(n2) - self.get_node_pos(n1)  # vector along spoke
            f2 = np.array(f_spokes[6*n2:6*n2+3:])               # force vector at rim node
            t_spokes.append(e1.dot(f2) / np.sqrt(e1.dot(e1)))
            
        return np.array(t_spokes)
    
    def calc_k_rim(self, node1, node2, ref, sec):
        """Calculate stiffness matrix for a single rim element.

        For details, see R. Palaninathan, P.S. Chandrasekharan,
        Computers and Structures, 4(21), pp. 663-669, 1985."""

        node1_pos = self.get_node_pos(node1)
        node2_pos = self.get_node_pos(node2)

        d = node2_pos - node1_pos  # beam orientation vector
        r1 = node1_pos - ref       # radial vector to node 1
        R = np.sqrt(r1.dot(r1))    # radius of curvature
        
        # angle subtended by arc segment
        phi0 = 2*np.arcsin(np.sqrt(d.dot(d)) / (2*R))
        
        # local coordinate system
        e1 =  d / np.sqrt(d.dot(d))
        e3 = -self.geom.n_vec
        e2 =  np.cross(e3,e1)
        
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
        k_r = np.matrix(np.zeros((12,12)))
        
        # Flexibility matrix for node 1 DOFs
        a = np.matrix(np.zeros((6,6)))
        
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
        a[2,4] = R**2/2*(S/(sec.shear_mod*sec.I11) - D/(sec.young_mod*sec.I22))
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
        Tb = np.matrix(np.zeros((6,6)))
        Tb[:3:,:3:] = np.matrix([[np.cos(phi0/2), -np.sin(phi0/2), 0],
                                 [np.sin(phi0/2),  np.cos(phi0/2), 0],
                                 [0,               0,              1]])
        Tb[3::,3::] = Tb[:3:,:3:]

        # Transformation matrix from beam coordinates to global coordinates
        Tg = np.matrix(np.zeros((6,6)))
        Tg[:3:,:3:] = np.matrix(np.vstack((e1,e2,e3)).T)
        Tg[3::,3::] = Tg[:3:,:3:]

        # Assemble submatrices
        k_r[:6:,:6:] = np.linalg.inv(a)    # K_II
        k_r[6::,6::] = np.linalg.inv(b)    # K_JJ
        k_r[6::,:6:] = Tbar * k_r[:6:,:6:] # K_JI
        
        k_r[:6:,:6:] = Tg*Tb   * k_r[:6:,:6:] * Tb.T*Tg.T
        k_r[6::,6::] = Tg*Tb.T * k_r[6::,6::] * Tb  *Tg.T
        k_r[6::,:6:] = Tg*Tb.T * k_r[6::,:6:] * Tb.T*Tg.T
        k_r[:6:,6::] = k_r[6::,:6:].T      # K_IJ (symm.)
        
        return k_r
    
    def calc_k_spoke(self, node1, node2, sec):
        'Calculate stiffness matrix for thin elastic rod (no bending or torsion).'

        node1_pos = self.get_node_pos(node1)
        node2_pos = self.get_node_pos(node2)
        
        # Tangent vector
        e1 = node2_pos - node1_pos
        l = np.sqrt(e1.dot(e1))
        e1 = e1 / l # convert to unit vector
        
        # Rotation matrix to global coordinates
        Tg = np.matrix(np.zeros((6,6)))
        Tg[:3:,0] = e1.reshape((3,1))
        Tg[3::,3] = e1.reshape((3,1))
        
        k_spoke = np.matrix(np.zeros((6,6)))
        k_spoke[::3,::3] = sec.area*sec.young_mod/l * np.matrix([[1,-1],[-1,1]])
        
        k_spoke = Tg * k_spoke * Tg.T
        
        return k_spoke
    
    def calc_stiff_mat(self):
        'Calculate global stiffness matrix by element scatter algorithm.'
    
        print('# Calculating global stiffness matrix ---')
        print('# -- Nodes: {:d}'.format(self.n_nodes))
        print('# -- DOFs : {:d}'.format(6*self.n_nodes))
        print('# ---------------------------------------')
        print('')
    
        self.k_rim    = np.matrix(np.zeros((6*self.n_nodes,6*self.n_nodes)))
        self.k_spokes = self.k_rim.copy()
        
        # Loop over element matrices and scatter to global K matrix
        for e in range(len(self.el_type)):
            if self.el_type[e] == EL_RIM:
                dof_n1 = 6*self.el_n1[e] + np.arange(6)
                dof_n2 = 6*self.el_n2[e] + np.arange(6)
                
                dof = np.concatenate((dof_n1, dof_n2))

                self.k_rim[np.ix_(dof,dof)] = self.k_rim[dof][:,dof] + \
                                              self.calc_k_rim(self.el_n1[e], 
                                                              self.el_n2[e], 
                                                              np.array([0,0,0]), 
                                                              self.rim_sec)
            if self.el_type[e] == EL_SPOKE:
                dof_n1 = 6*self.el_n1[e] + np.arange(3)
                dof_n2 = 6*self.el_n2[e] + np.arange(3)
                
                dof = np.concatenate((dof_n1,dof_n2))
                
                self.k_spokes[np.ix_(dof,dof)] = self.k_spokes[dof][:,dof] + \
                                                 self.calc_k_spoke(self.el_n1[e], 
                                                                   self.el_n2[e],
                                                                   self.spoke_sec)
                                                                   
        self.k_global = self.k_rim + self.k_spokes
    
    def add_rigid_body(self, rigid_body):
        'Constrain nodes to move together as a rigid body.'

        # Convert stiffness equation into reduced equation
        #   U  = C * U_reduced
        #   F_reduced = B * F
        #   F_reduced = (B * K * C) * U_reduced
        
        # Check that the constrained nodes are not already assigned to rigid bodies
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
        self.bc_const.extend(6*[False])
        self.bc_force.extend(6*[False])
        self.bc_u = np.append(self.bc_u, 6*[0])
        self.bc_f = np.append(self.bc_f, 6*[0])

        print('# Adding new rigid body: {:s}'.format(rigid_body.name))
        print('# -- Reference node {:d}\n'.format(rigid_body.node_id))
        
        # Re-calculate B and C matrices
        n_c = np.sum([r.n_nodes for r in self.rigid])
        self.C = np.mat(np.zeros((6*self.n_nodes,6*(self.n_nodes - n_c))))
        self.B = np.mat(np.zeros((6*(self.n_nodes - n_c),6*self.n_nodes)))
        
        self.node_r_id = [-1] * self.n_nodes
        for rig_id in range(len(self.rigid)):
            self.node_r_id[self.rigid[rig_id].node_id] = self.n_nodes - len(self.rigid) - n_c + rig_id
        
        n_r_n = 0
        for n in range(self.n_nodes):
            in_rigid = [n in r.nodes for r in self.rigid]
            dof_n = 6*n + np.arange(6)
            
            if not any(in_rigid):
                # new re-numbered node ID
                self.node_r_id[n] = n_r_n
                dof_r_n = 6*n_r_n + np.arange(6)
                n_r_n += 1

                self.C[dof_n,dof_r_n] = 1 # identity matrix
                self.B[dof_r_n,dof_n] = 1
            else:
                rig_i = in_rigid.index(True) # Index of rigid body
                n_r_r = self.node_r_id[self.rigid[rig_i].node_id] # Reduced index of rigid body node
                dof_r_r = 6*n_r_r + np.arange(6)
                
                r_c = self.get_node_pos(n) - self.rigid[rig_i].pos
                R = skew_symm(r_c)

                self.C[dof_n,dof_r_r] = 1
                self.C[np.ix_(dof_n[:3:],dof_r_r[3::])] = R
                
                self.B[dof_r_r,dof_n] = 1
                self.B[np.ix_(dof_r_r[3::],dof_n[:3:])] = R
                
        self.soln_updated = False
        
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
    
                print(dof_r)

                self.bc_const[dof_r] = False
                self.bc_force[dof_r] = False
        
                self.bc_u[dof_r] = 0
                self.bc_f[dof_r] = 0
        
                self.soln_updated = False
    
    def solve(self):
        'Solve elasticity equations for nodal displacements.'

        # Form augmented, reduced stiffness matrix
        if self.k_global == None:
            self.calc_stiff_mat()
            
        if len(self.rigid) == 0:
            # No rigid bodies. Reduced node IDs are equal to node IDs
            self.node_r_id = np.arange(self.n_nodes, dtype=np.int16)
        
        k_red = self.B * self.k_global * self.C
        k_aug = k_red.copy()
        f_aug = np.zeros(k_aug.shape[0])
        
        print('# Solving for nodal displacements -------')
        print('# -- Reduced system of equations:')
        print('#      Reduced DOFs: {:d}'.format(k_aug.shape[0]))
        
        # Apply constraints to nodes
        n_c = np.sum([r.n_nodes for r in self.rigid])
        for dof_c in [d for d in range(6*self.n_nodes) if self.bc_const[d]]:
            dof_r = 6*self.node_r_id[int(dof_c) / 6] + dof_c % 6
            
            k_aug[dof_r] = 0
            k_aug[dof_r,dof_r] = 1
            
            f_aug[dof_r] = self.bc_u[dof_c]
        
        # Apply forces to nodes
        for dof_c in [d for d in range(6*self.n_nodes) if self.bc_force[d]]:
            dof_r = 6*self.node_r_id[int(dof_c) / 6] + dof_c % 6
            f_aug[dof_r] = self.bc_f[dof_c]
    
        # Solve for reduced nodal displacements
        print('#      Rank = {:d} / ({:d})'.format(np.linalg.matrix_rank(k_aug),k_aug.shape[0]))
        
        try:
            u_red = np.linalg.solve(k_aug,f_aug)    
        except np.linalg.LinAlgError as e:
            print('\n*** ERROR: {:s}. Did you properly constrain all DOFs?'.format(e.message))

            # u, s, v = np.linalg.svd(k_aug)
            # nnz = (s >= 1e-12*np.mean(s)).sum()
            # print(s)
            # print(v[nnz:].conj().T)
 
            # for dof_u in [i for i in range(u.shape[0]) if s[i] < 1e-12*np.mean(s)]:
                # node_id = self.node_r_id.index(dof_u/6)
                # print('*** -- Node {:3d} DOF {:d}'.format(node_id,dof_u % 6))
            return False
        
        self.soln_updated = True

        # Create solution object
        soln = FEMSolution(self)

        # nodal displacements
        if 'numpy' in str(type(self.C)):
            soln.nodal_disp = np.array(self.C.dot(u_red)).flatten()
        else:
            soln.nodal_disp = u_red.flatten() # No rigid bodies. C is a scalar
        
        # nodal reaction forces
        rxn_red = np.array(k_red.dot(u_red) - f_aug).flatten()
        dof_rxn = [6*self.node_r_id[i/6] + i%6 for i in range(6*self.n_nodes) if self.bc_const[i]]
        soln.nodal_rxn = rxn_red[dof_rxn]
        soln.dof_rxn = dof_rxn

        # spoke tension
        soln.spoke_t = self.get_spoke_tension(soln.nodal_disp)
        
        print('# ---------------------------------------')

        return soln
        
    def __init__(self,geom,rim_sec,spoke_sec):
        self.geom = geom
        self.rim_sec = rim_sec
        self.spoke_sec = spoke_sec

        print(geom.a_rim_nodes)

        # Rim nodes
        self.x_nodes = geom.d_rim/2 * np.sin(geom.a_rim_nodes)
        self.y_nodes = -geom.d_rim/2 * np.cos(geom.a_rim_nodes)
        self.z_nodes = np.zeros(len(self.x_nodes))

        # Hub nodes
        diam_hub = np.array([geom.d1_hub*(geom.s_hub_nodes[i] < 0) + 
                             geom.d2_hub*(geom.s_hub_nodes[i] > 0)
                             for i in range(geom.n_hub_nodes)])

        width_hub = np.array([-geom.w1_hub*(geom.s_hub_nodes[i] < 0) + 
                              geom.w2_hub*(geom.s_hub_nodes[i] > 0)
                              for i in range(geom.n_hub_nodes)])

        self.x_nodes = np.append(self.x_nodes,  diam_hub/2 * np.sin(geom.a_hub_nodes))
        self.y_nodes = np.append(self.y_nodes, -diam_hub/2 * np.cos(geom.a_hub_nodes))
        self.z_nodes = np.append(self.z_nodes, width_hub)

        self.n_nodes = len(self.x_nodes)

        # rim elements connectivity
        self.el_n1 = np.array(range(geom.n_rim_nodes))
        self.el_n2 = np.array(range(1, geom.n_rim_nodes) + [0])

        # spoke elements connectivity
        self.el_n1 = np.concatenate((self.el_n1, geom.lace_hub_n - 1 + geom.n_rim_nodes))
        self.el_n2 = np.concatenate((self.el_n2, geom.lace_rim_n - 1))
        
        self.el_type = np.array([EL_RIM]*geom.n_spokes)
        self.el_type = np.concatenate((self.el_type, EL_SPOKE*np.ones(len(geom.lace_hub_n))))

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
        self.k_rim = None
        self.k_spokes = None
        self.k_global = None

        # solution arrays
        self.soln_updated = False

class BicycleWheelGeom:
    'Geometric parameters including size and lacing pattern'

    def parse_wheel_file(self, fname):

        def keyword_rim(l, f):

            # Default values
            args = {'diam': 0.6, 'spacing': 'default', 'N':36}

            # Parse arguments
            l_args = l.strip().split()
            for c in l_args[1::]:
                # Set option
                key = c.split('=')[0]
                arg = c.split('=')[1]
                if key in args:
                    args[key] = arg
                else:
                    raise ValueError('Invalid parameter: {:s}'.format(key))

            self.d_rim = float(args['diam'])
            N = int(args['N'])
            self.n_rim_nodes = N

            if args['spacing'] == 'default':

                # Space rim nodes evenly
                self.a_rim_nodes = np.linspace(0, 2*np.pi * (N-1)/N, N)
                l = f.readline()

            elif args['spacing'] == 'custom':

                # Read rim node positions from input file
                l = f.readline()

                self.a_rim_nodes = np.zeros(N, dtype=np.float)
                while len(l) > 0:

                    l_rim_node = l.strip().split()

                    if re.match('^\d+$',l_rim_node[0]): # check if line is an integer
                        self.a_rim_nodes[int(l_rim_node[0]) - 1] = np.radians(float(l_rim_node[1]))
                        l = f.readline()
                    elif l_rim_node[0][0] == '#':
                        l = f.readline()
                    else:
                        break

                # Make sure all angles are unique
                if not len(self.a_rim_nodes) == len(set(self.a_rim_nodes)):
                    raise ValueError('When using spacing=custom, all node angles must be defined')

            else:
                raise ValueError('Invalid value for parameter \'spacing\': ' + args['spacing'])

            # Increment to the next line
            return l

        def keyword_hub(l, f):

            # Default values
            args = {'diam': 0.6, 'diam_drive': None, 'width':0.035, 'width_drive':None, 
                    'spacing': 'default', 'N':36}

            # Parse arguments
            l_args = l.strip().split()
            for c in l_args[1::]:
                # Set option
                key = c.split('=')[0]
                arg = c.split('=')[1]
                if key in args:
                    args[key] = arg
                else:
                    raise ValueError('Invalid parameter: {:s}'.format(key))

            self.d1_hub = float(args['diam'])
            self.d2_hub = self.d1_hub
            if args['diam_drive'] != None:
            	self.d2_hub = float(args['diam_drive'])

            self.w1_hub = float(args['width'])
            self.w2_hub = self.w1_hub
            if args['width_drive'] != None:
            	self.w2_hub = float(args['width_drive'])

            N = int(args['N'])
            self.n_hub_nodes = N
            self.s_hub_nodes = np.zeros(N, dtype=np.int8)

            if args['spacing'] == 'default':

                # Space rim nodes evenly
                self.a_hub_nodes = np.linspace(0, 2*np.pi * (N-1)/N, N)
                self.s_hub_nodes[::2] = 1    # Drive-side nodes
                self.s_hub_nodes[1::2] = -1  # Left-side nodes
                l = f.readline()

            elif args['spacing'] == 'custom':

                # Read hub node positions from input file
                l = f.readline()

                self.a_hub_nodes = np.zeros(N, dtype=np.float)

                while len(l) > 0:

                    l_hub_node = l.strip().split()

                    if re.match('^\d+$',l_hub_node[0]): # check if line is an integer
                        i_node = int(l_hub_node[0]) - 1
                        self.a_hub_nodes[i_node] = np.radians(float(l_hub_node[1]))

                        if len(l_hub_node) > 2 and l_hub_node[2].upper() == 'D':
                            self.s_hub_nodes[i_node] = 1  # Drive-side node
                        else:
                            self.s_hub_nodes[i_node] = -1  # Left-side node

                        l = f.readline()
                    elif l_hub_node[0][0] == '#':
                        l = f.readline()
                    else:
                        break

                # Make sure all angles are unique
                if not len(self.a_hub_nodes) == len(set(self.a_hub_nodes)):
                    raise ValueError('When using spacing=custom, all node angles must be defined')

            else:
                raise ValueError('Invalid value for parameter \'spacing\': ' + args['spacing'])

            # Increment to the next line
            return l

        def keyword_lacing(l, f):
            # Default values
            args = {'pattern':'default'}

            # Parse arguments
            l_args = l.strip().split()
            for c in l_args[1::]:
                # Set option
                key = c.split('=')[0]
                arg = c.split('=')[1]
                if key in args:
                    args[key] = arg
                else:
                    raise ValueError('Invalid parameter: {:s}'.format(key))

            if args['pattern'] == 'custom':

                # Read hub node positions from input file
                l = f.readline()

                self.lace_hub_n = np.array([], dtype=np.int32)
                self.lace_rim_n = np.array([], dtype=np.int32)

                while len(l) > 0:

                    l_lace = l.strip().split()

                    if re.match('^\d+$',l_lace[0]): # check if line is an integer
                        i_node_hub = int(l_lace[0])
                        i_node_rim = int(l_lace[1])

                        self.lace_hub_n = np.append(self.lace_hub_n, i_node_hub)
                        self.lace_rim_n = np.append(self.lace_rim_n, i_node_rim)

                        l = f.readline()
                    elif l_lace[0][0] == '#':
                        l = f.readline()
                    else:
                        break

            self.n_spokes = len(self.lace_hub_n)

            # Increment to the next line
            return f.readline()

        try:
            with open(fname) as f:
                l = f.readline()
                while len(l) > 0:

                    l_args = l.strip().split()

                    if l_args[0].lower() == 'rim':
                        l = keyword_rim(l, f)
                    elif l_args[0].lower() == 'hub':
                        l = keyword_hub(l, f)
                    elif l_args[0].lower() == 'lacing':
                        l = keyword_lacing(l, f)
                    else:
                        # Nothing interesting on this line. Skip it.
                        l = f.readline()
        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))
            raise
        except:
            print('')
            raise

    def __init__(self, wheel_file=None, n_vec=np.array([0,0,1])):
    
        print('# Initializing wheel geometry -----------')

        # self.d_rim = d_rim   # rim diameter
        # self.d_hub = d_hub   # hub diameter
        # self.w1_hub = w1_hub # width from rim center to left flange
        # self.w2_hub = w2_hub # width from rim center to right flange
        self.n_vec = n_vec   # axial vector from hub center to drive side nut

        if wheel_file is not None:
            self.parse_wheel_file(wheel_file)

class RimSection:
    'Section definition for rim'
    
    def __init__(self, area, I11, I22, I33, young_mod, shear_mod, K2=1.0, K3=1.0):
        self.area = area
        self.I11 = I11
        self.I22 = I22
        self.I33 = I33
        self.young_mod = young_mod
        self.shear_mod = shear_mod
        self.K2 = K2
        self.K3 = K3

class SpokeSection:
    'Section definition for spoke'
    
    def __init__(self, d_spoke, young_mod):
        self.young_mod = young_mod
        self.area = np.pi / 4 * d_spoke**2

class RigidBody:
    'Set of nodes with constained DOFs'
    
    def __init__(self, name, pos, nodes):
    
        self.name = name
        
        self.pos = np.array(pos)
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)

class FEMSolution:

    def get_polar_displacements(self, node_id):
        'Convert nodal displacements to polar form'

        # Allow array input for node_id and/or dof
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]

        u_rad = np.array([])
        u_tan = np.array([])

        for n in node_id:

            x = np.array([self.x_nodes[n], self.y_nodes[n], self.z_nodes[n]])
            n_tan = np.cross(np.array([0,0,1]),x)

            u_rim = self.nodal_disp[6*n + np.arange(3)]

            u_rad = np.append(u_rad, u_rim.dot(x) / np.sqrt(x.dot(x)))
            u_tan = np.append(u_tan, u_rim.dot(n_tan) / np.sqrt(n_tan.dot(n_tan)))

        u_rad = np.array(u_rad)
        u_tan = np.array(u_tan)

        return u_rad, u_tan

    def get_deformed_coords(self, node_id, scale_rad, scale_tan):

        # Allow array input for node_id and/or dof
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]

        x_def = np.array([])
        y_def = np.array([])
        z_def = np.array([])

        u_rad, u_tan = self.get_polar_displacements(node_id)

        for n in range(len(node_id)):
            n_id = node_id[n]

            x = np.array([self.x_nodes[n_id], self.y_nodes[n_id], self.z_nodes[n_id]])
            n_tan = np.cross(np.array([0,0,1]), x)

            u = scale_rad*u_rad * x/np.sqrt(x.dot(x)) + scale_tan*u_tan * n_tan

            x_def = np.append(x_def, x[0] + u[0])
            y_def = np.append(y_def, x[1] + u[1])

        return x_def, y_def

    def plot_deformed_wheel(self, scale_rad=0.1, scale_tan=0.0):

        rim_nodes = [self.el_n1[e] for e in range(len(self.el_type)) if self.el_type[e] == EL_RIM]

        u_rad, u_tan = self.get_polar_displacements(rim_nodes)

        # Scale the largest displacement to a percentage of the rim radius
        scale_rad = self.geom.d_rim/2 / max(np.abs(u_rad)) * scale_rad
        scale_tan = self.geom.d_rim/2 / max(np.abs(u_rad)) * scale_tan

        theta = self.geom.a_rim_nodes - np.pi/2

        # Calculate coordinates in deformed configuration
        theta_def = theta + scale_tan * u_tan / (self.geom.d_rim / 2)
        r_def = (self.geom.d_rim / 2) + scale_rad * u_rad

        theta_ii = np.linspace(-np.pi/2, 3*np.pi/2, 1000)

        theta_interp = interpolate.interp1d(np.append(theta,theta[0] + 2*np.pi),
                                            np.append(theta_def,theta_def[0] + 2*np.pi))
        theta_def_ii = theta_interp(theta_ii)

        r_def_ii = interp_periodic(theta, r_def, theta_ii)

        # Plot undeformed rim
        pp.plot(self.geom.d_rim/2 * np.cos(theta_ii),
                self.geom.d_rim/2 * np.sin(theta_ii),'k:')

        # Plot deformed rim
        pp.plot(r_def_ii * np.cos(theta_def_ii), r_def_ii * np.sin(theta_def_ii), 'k', linewidth=3.0)


        # Plot spokes in deformed configuration
        for e in [x for x in range(len(self.el_type)) if self.el_type[x] == EL_SPOKE]:
            n_hub = self.el_n1[e]
            n_rim = self.el_n2[e]

            x_hub, y_hub = self.get_deformed_coords(n_hub, scale_rad, scale_tan)
            x_rim, y_rim = self.get_deformed_coords(n_rim, scale_rad, scale_tan)

            pp.plot([x_hub, x_rim], [y_hub, y_rim], 'k-')

        # Axis properties
        ax = pp.gca()
        ax.set_ylim([-1.2*self.geom.d_rim/2, 1.2*self.geom.d_rim/2])
        ax.set_xlim([-1.2*self.geom.d_rim/2, 1.2*self.geom.d_rim/2])

        ax.set_aspect('equal', 'datalim')

        ax.xaxis.set_major_locator(pp.NullLocator())
        ax.yaxis.set_major_locator(pp.NullLocator())

        return pp.gcf()


    def __init__(self, fem):
        self.updated = False

        self.geom = copy.copy(fem.geom)
        
        self.x_nodes = fem.x_nodes.copy()
        self.y_nodes = fem.y_nodes.copy()
        self.z_nodes = fem.z_nodes.copy()
        
        self.el_type = fem.el_type.copy()
        self.el_n1 = fem.el_n1.copy()
        self.el_n2 = fem.el_n2.copy()
        
        self.nodal_disp = []
        self.nodal_rxn = []
        self.dof_rxn = []
        self.spoke_t = []
        self.rim_n = []
        self.rim_v = []
        self.rim_m = []
