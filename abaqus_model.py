#!/usr/bin/env python

"""Generate an ABAQUS input file from a BicycleWheel object."""

import numpy as np
from bikewheelcalc import BicycleWheel


class AbaqusModel:

    node_fmt = ' {:5d}, {:11.4E}, {:11.4E}, {:11.4E}\n'

    def get_spokes_at_rim_node(self, r):
        'Get indices of spokes connected to a specific rim node.'

        theta_r = self.theta_rim_nodes[r]
        theta_s = np.array([s.rim_pt[1] for s in self.wheel.spokes])
        return np.where(theta_s == theta_r)[0]

    def create_rim_nodes(self):
        """Create rim nodes for each unique spoke connection point, then
        create nodes to interpolate rim curvature."""

        # Find rim attachment points (multiple spokes may attach at 1 point)
        theta_s = set()
        for s in self.wheel.spokes:
            theta_s.add(s.rim_pt[1])

        theta_s = sorted(list(theta_s))

        # Insert additional nodes where needed to interpolate curvature
        theta_interp = []
        for g in range(len(theta_s)):
            nxt = (g+1) % len(theta_s)
            if theta_s[nxt] < theta_s[g]:
                da = theta_s[nxt] + 2*np.pi - theta_s[g]
            else:
                da = theta_s[nxt] - theta_s[g]

            # Determine how many nodes to insert
            n_add = np.floor(da / self.max_curve)
            theta_interp.extend(theta_s[g] +
                                da*np.arange(1, n_add + 1) / (n_add + 1))

        self.theta_rim_nodes = sorted(theta_s + theta_interp)
        self.i_spoke_nips = [self.theta_rim_nodes.index(s) for s in theta_s]

    def write_rim_nodes(self, nset='nsetRim', f_perturb=None):
        out_str = '*NODE, nset={:s}\n'.format(nset)

        R = self.wheel.rim.radius

        for i in range(len(self.theta_rim_nodes)):
            x = R*np.sin(self.theta_rim_nodes[i])
            y = -R*np.cos(self.theta_rim_nodes[i])

            # Add an imperfection for buckling analysis
            if f_perturb is not None:
                z = f_perturb(self.theta_rim_nodes[i])
            else:
                z = 0.0
            out_str += self.node_fmt.format(i+1, x, y, z)

        out_str += '*NSET, nset=nsetSpokeNip\n'
        for i in self.i_spoke_nips:
            out_str += ' {:5d}\n'.format(i+1)

        # out_str += '*TRANSFORM, nset=nsetSpokeNip, type=C'
        # out_str += '\n 0, 0, 0, 0, 0, 1.0\n'

        return out_str

    def write_spoke_nodes(self, nset='nsetSpokes', f_perturb=None):
        out_str = '*NODE, nset={:s}\n'.format(nset)

        for s in range(len(self.wheel.spokes)):
            rim_pt = self.wheel.spokes[s].rim_pt
            hub_pt = self.wheel.spokes[s].hub_pt

            x = np.linspace(rim_pt[0]*np.sin(rim_pt[1]),
                            hub_pt[0]*np.sin(hub_pt[1]), self.n_spk+1)
            y = np.linspace(-rim_pt[0]*np.cos(rim_pt[1]),
                            -hub_pt[0]*np.cos(hub_pt[1]), self.n_spk+1)

            z = np.linspace(rim_pt[2], hub_pt[2], self.n_spk+1)
            if f_perturb is not None:
                z += f_perturb(np.linspace(0.0, 1.0, self.n_spk+1))

            for i in range(0, self.n_spk+1):
                out_str += self.node_fmt.format(1000*(i+1) + s+1,
                                                x[i], y[i], z[i])

        out_str += '*NSET, nset=nsetHub\n'
        for s in range(len(self.wheel.spokes)):
            out_str += ' {:d}\n'.format(1000*(self.n_spk+1) + s + 1)

        return out_str

    def write_pretension_nodes(self):
        out_str = '*NODE, nset=nsetPreT\n'

        for s in range(len(self.wheel.spokes)):
            out_str += self.node_fmt.format(99000 + s + 1, 0.0, 0.0, 0.0)

        return out_str

    def write_rim_elems(self, elset='elsetRim', eltype='B31'):
        n_rim_nodes = len(self.theta_rim_nodes)

        out_str = '*ELEMENT, type={:s}, elset={:s}\n     1,     1,     2\n'\
            .format(eltype, elset)
        out_str += ' {:5d}, {:5d},     1\n'.format(n_rim_nodes, n_rim_nodes)

        out_str += '*ELGEN, elset={:s}\n     1, {:5d}\n'\
            .format(elset, n_rim_nodes-1)

        return out_str

    def write_spoke_elems(self, elset='elsetSpokes', eltype='B31'):
        self.spoke_eltype = eltype
        out_str = ''

        for s in range(len(self.wheel.spokes)):

            side = (-np.sign(self.wheel.spokes[s].hub_pt[2])+1)/2 + 1
            elset_s = elset + '{:d}'.format(int(side))

            out_str += '*ELEMENT, type={eltype:s}, elset={elset:s}\n'\
                .format(eltype=eltype, elset=elset_s)
            out_str += '{elnum:d}, {n1:d}, {n2:d}\n'\
                .format(elnum=1000 + s+1, n1=1000 + s+1, n2=2000 + s+1)

            if self.n_spk > 1:
                out_str += '*ELGEN, elset={:s}\n'.format(elset_s)
                out_str += '{mel:d}, {nel:d}, {ninc:d}, {einc:d}\n'\
                    .format(mel=1000 + s+1, nel=self.n_spk,
                            ninc=1000, einc=1000)

        # Add all spokes to primary elset
        out_str += '*ELSET, elset={:s}\n{:s}, {:s}\n'.format(elset,
                                                             elset+'1',
                                                             elset+'2')

        # Create element set containing one element from each spoke
        out_str += '*ELSET, elset=elsetSpokesRef, generate\n'
        out_str += ' {0:d}, {1:d}, {2:d}\n'\
            .format(1001, 1000 + len(self.wheel.spokes), 1)

        return out_str

    def write_pretension_section(self):
        out_str = ''

        for s in range(len(self.wheel.spokes)):
            out_str += '*PRE-TENSION SECTION, node={:d}, element={:d}\n'\
                .format(99000 + s+1, 1000 + s+1)

        return out_str

    def write_rigid_ties(self):
        'Create rigid bodies to connect spokes to rim.'

        out_str = ''

        for s in range(len(self.i_spoke_nips)):
            # Create a node set for all nodes to be connected
            i_rim = self.i_spoke_nips[s]
            out_str += '*NSET, nset=nsetRig{:d}\n'.format(i_rim+1)
            out_str += ' {:5d}\n'.format(i_rim+1)
            i_spk = self.get_spokes_at_rim_node(i_rim)
            for i in i_spk:
                out_str += ' {:5d}\n'.format(i+1001)

            # Create rigid body
            out_str += '*RIGID BODY, ref node={:d}, tie nset=nsetRig{:d}\n'\
                .format(i_rim+1, i_rim+1)

        return out_str

    def write_beam_sections(self, alpha1=1.0, alpha2=1.0):
        'Write material and beam section block for rim and spokes.'

        r = self.wheel.rim
        s1 = self.wheel.spokes[0]
        s2 = self.wheel.spokes[1]

        if self.spoke_eltype[0].lower() == 'b':  # beam elements
            out_str = '*BEAM SECTION, elset=elsetSpokes1, material=steel1'
            out_str += ', section=CIRC\n{:e}\n0.,0.,-1.\n'.format(s1.diameter/2)
            out_str += '*BEAM SECTION, elset=elsetSpokes2, material=steel2'
            out_str += ', section=CIRC\n{:e}\n0.,0.,-1.\n'.format(s2.diameter/2)
        elif self.spoke_eltype[0].lower() == 't':  # truss element
            out_str = '*SOLID SECTION, elset=elsetSpokes, material=steel1\n'
            out_str += '{:e}\n'.format(np.pi/4 * s1.diameter**2)

        out_str += '*MATERIAL, name=steel1\n*ELASTIC\n {:e}, 0.33\n'\
            .format(s1.EA / (np.pi/4 * s1.diameter**2))
        out_str += '*DENSITY\n 8050.0\n'
        out_str += '*DAMPING, alpha={:e}\n'.format(1000.0)
        out_str += '*EXPANSION, type=iso\n {:e}\n'.format(alpha1)

        out_str += '*MATERIAL, name=steel2\n*ELASTIC\n {:e}, 0.33\n'\
            .format(s2.EA / (np.pi/4 * s2.diameter**2))
        out_str += '*DENSITY\n 8050.0\n'
        out_str += '*DAMPING, alpha={:e}\n'.format(1000.0)
        out_str += '*EXPANSION, type=iso\n {:e}\n'.format(alpha2)

        out_str += '*BEAM GENERAL SECTION, elset=elsetRim, '
        out_str += 'section=GENERAL, density=2700.0\n'

        warping = True
        if warping:
            out_str += '{A:10.4E}, {I33:10.4E}, 0., {I22:10.4E}, {I11:10.4E}, {G0:10.4E}, {Iw:10.4E}\n'\
                .format(A=r.area, I11=r.I11, I22=r.I22, I33=r.I33,
                        G0=0.0, Iw=r.Iw)
        else:
            out_str += '{A:10.4E}, {I33:10.4E}, 0., {I22:10.4E}, {I11:10.4E}\n'\
                .format(A=self.wheel.rim.area,
                        I11=r.I11, I22=r.I22, I33=r.I33)
        out_str += '0.,0.,-1.\n{E:10.4e}, {G:10.4}\n'\
            .format(E=r.young_mod, G=r.shear_mod)

        return out_str

    def write_bc_fix_hub(self):
        'Write ENCASTRE boundary conditions for hub nodes.'

        return '*BOUNDARY\n nsetHub, ENCASTRE\n'

    def write_heading(self, heading):
        'Write a nicely formatted section heading.'

        return ('** ' + (50*'-') + '\n** -- {:s}\n** ' + (50*'-') + '\n')\
            .format(heading)

    def __init__(self, wheel, max_curve=0.1, n_spk=10):
        self.wheel = wheel
        self.max_curve = max_curve
        self.n_spk = n_spk

        self.create_rim_nodes()


class AbaqusModelShell:

    node_fmt = ' {:5d}, {:11.4E}, {:11.4E}, {:11.4E}\n'

    class RimXSection:
        def __init__(self, rim, n_e_xc):

            if rim.sec_type == 'C':

                h = rim.sec_params['h']
                w = rim.sec_params['w']
                t = rim.sec_params['t']
                self.thick = t

                n_el_wall = np.ceil(n_e_xc * h / (2*h + w))
                n_el_web = 2*np.ceil(n_e_xc * w / (4*h + 2*w))

                x_wall = w/2 * np.ones(n_el_wall)
                x_web = np.linspace(-w/2, w/2, n_el_web + 1)

                y_wall = np.linspace(0, h, n_el_wall + 1)
                y_web = np.zeros(n_el_web - 1)

                self.x_pts = np.concatenate((-x_wall, x_web, x_wall))
                self.y_pts = np.concatenate((y_wall[::-1], y_web, y_wall)) -\
                    rim.sec_params['y_s']

                self.ref_pt = int(np.ceil(len(self.x_pts)/2))

            elif rim.sec_type == 'box':
                h = rim.sec_params['h']
                w = rim.sec_params['w']
                t = rim.sec_params['t']
                self.thick = t

                n_el_w = 2*np.ceil(n_e_xc * w / (4*h + 4*w))
                n_el_h = 2*np.ceil(n_e_xc * h / (4*h + 4*w))

                x_w = np.linspace(-w/2, w/2, n_el_w+1)
                x_h = w/2 * np.ones(n_el_h - 1)

                y_w = h/2 * np.ones(n_el_w + 1)
                y_h = np.linspace(-h/2, h/2, n_el_h + 1)
                y_h = y_h[1:-1]

                self.x_pts = np.concatenate((x_w, x_h, x_w[::-1], -x_h))
                self.y_pts = np.concatenate((-y_w, y_h, y_w, y_h[::-1]))

                self.ref_pt = int(n_el_w/2) + 1

            else:
                raise ValueError('Unknown rim section type {:s}'
                                 .format(rim.sec_type))

        def rotate_x_sec(self, theta):
            'Rotate cross-section points to global coordinates.'
            x_rot = self.y_pts * np.sin(theta)
            y_rot = -self.y_pts * np.cos(theta)
            z_rot = self.x_pts

            return x_rot, y_rot, z_rot

    def get_spokes_at_rim_point(self, r):
        'Get indices of spokes connected to a specific rim node.'

        theta_r = self.theta_rim_pts[r]
        theta_s = np.array([s.rim_pt[1] for s in self.wheel.spokes])
        return np.where(theta_s == theta_r)[0]

    def create_rim_points(self):
        """Create a rim cross-section for each unique spoke connection point,
        then create extra points to interpolate rim curvature."""

        # Find rim attachment points (multiple spokes may attach at 1 point)
        theta_s = set()
        for s in self.wheel.spokes:
            theta_s.add(s.rim_pt[1])

        theta_s = sorted(list(theta_s))

        # Insert additional nodes where needed to interpolate curvature
        theta_interp = []
        for g in range(len(theta_s)):
            nxt = (g+1) % len(theta_s)
            if theta_s[nxt] < theta_s[g]:
                da = theta_s[nxt] + 2*np.pi - theta_s[g]
            else:
                da = theta_s[nxt] - theta_s[g]

            # Determine how many nodes to insert
            n_add = np.floor(da / self.max_curve)
            theta_interp.extend(theta_s[g] +
                                da*np.arange(1, n_add + 1) / (n_add + 1))

        self.theta_rim_pts = sorted(theta_s + theta_interp)
        self.i_spoke_nips = [self.theta_rim_pts.index(s) for s in theta_s]

    def write_rim_nodes(self, nset='nsetRim', f_pert=None):
        out_str = '*NODE, nset={:s}\n'.format(nset)

        R = self.wheel.rim.radius

        for i in range(len(self.theta_rim_pts)):
            theta = self.theta_rim_pts[i]

            # Add an imperfection for buckling analysis
            if f_pert is not None:
                z_pert = f_pert(self.theta_rim_pts[i])
            else:
                z_pert = 0.0

            # Write all nodes in this cross-section
            x_rot, y_rot, z_rot = self.rim_xc.rotate_x_sec(theta)

            for j in range(len(x_rot)):
                x = x_rot[j] + R*np.sin(self.theta_rim_pts[i])
                y = y_rot[j] - R*np.cos(self.theta_rim_pts[i])
                z = z_rot[j] + z_pert

                i_node = len(x_rot)*i + j + 1
                out_str += self.node_fmt.format(i_node, x, y, z)

        out_str += '*NSET, nset=nsetRefNode\n {:d}\n'\
            .format(self.rim_xc.ref_pt)

        return out_str

    def write_spoke_nodes(self, nset='nsetSpokes', f_perturb=None):
        out_str = '*NODE, nset={:s}\n'.format(nset)

        for s in range(len(self.wheel.spokes)):
            rim_pt = self.wheel.spokes[s].rim_pt
            hub_pt = self.wheel.spokes[s].hub_pt

            x = np.linspace(rim_pt[0]*np.sin(rim_pt[1]),
                            hub_pt[0]*np.sin(hub_pt[1]), self.n_spk+1)
            y = np.linspace(-rim_pt[0]*np.cos(rim_pt[1]),
                            -hub_pt[0]*np.cos(hub_pt[1]), self.n_spk+1)

            z = np.linspace(rim_pt[2], hub_pt[2], self.n_spk+1)
            if f_perturb is not None:
                z += f_perturb(np.linspace(0.0, 1.0, self.n_spk+1))

            for i in range(0, self.n_spk+1):
                out_str += self.node_fmt.format(10000 + 100*(i+1) + s+1,
                                                x[i], y[i], z[i])

        out_str += '*NSET, nset=nsetHub\n'
        for s in range(len(self.wheel.spokes)):
            out_str += ' {:d}\n'.format(10000 + 100*(self.n_spk+1) + s+1)

        return out_str

    def write_pretension_nodes(self):
        out_str = '*NODE, nset=nsetPreT\n'

        for s in range(len(self.wheel.spokes)):
            out_str += self.node_fmt.format(99000 + s + 1, 0.0, 0.0, 0.0)

        return out_str

    def write_rim_elems(self, elset='elsetRim', eltype='S4R'):
        n_n_xc = len(self.rim_xc.x_pts)
        n_e_xc = n_n_xc - 1*(not self.wheel.rim.sec_params['closed'])
        N_r = len(self.theta_rim_pts)

        # create master element
        out_str = '*ELEMENT, type={:s}, elset={:s}\n'.format(eltype, elset)
        out_str += '     1,     1,     2, {:5d}, {:5d}\n'\
            .format(n_n_xc + 2, n_n_xc + 1)

        out_str += '*ELGEN, elset={:s}\n'.format(elset)
        out_str += '   1, {:5d},   1,   1, {:5d}, {:5d}, {:5d}\n'\
            .format(n_e_xc-1, N_r-1,
                    n_n_xc, n_e_xc)

        # create last row of elements
        out_str += '*ELEMENT, type={:s}, elset={:s}\n'.format(eltype, elset)
        out_str += ' {:5d}, {:5d}, {:5d}, {:5d}, {:5d}\n'\
            .format(n_e_xc*(N_r-1) + 1,
                    n_n_xc*(N_r-1) + 1,
                    n_n_xc*(N_r-1) + 2,
                    2, 1)

        out_str += '*ELGEN, elset={:s}\n'.format(elset)
        out_str += ' {:5d}, {:5d},   1,   1\n'\
            .format(n_e_xc*(N_r-1) + 1,
                    n_e_xc - 1)

        # If rim profile is a closed section
        if self.wheel.rim.sec_params['closed']:
            # Create master closing element
            out_str += '*ELEMENT, type={:s}, elset={:s}\n'.format(eltype, elset)
            out_str += ' {:5d}, {:5d}, {:5d}, {:5d}, {:5d}\n'\
                .format(n_e_xc, n_n_xc, 1, n_n_xc + 1, 2*n_n_xc)
            out_str += ' {:5d}, {:5d}, {:5d}, {:5d}, {:5d}\n'\
                .format(N_r*n_e_xc,
                        N_r*n_n_xc, (N_r-1)*n_n_xc + 1, 1, n_n_xc)

            out_str += '*ELGEN, elset={:s}\n'.format(elset)
            out_str += ' {:5d}, {:5d}, {:5d}, {:5d}\n'\
                .format(n_e_xc, len(self.theta_rim_pts) - 1, n_n_xc, n_e_xc)

        return out_str

    def write_spoke_elems(self, elset='elsetSpokes'):
        out_str = ''
        ns = len(self.wheel.spokes)
        for s in range(ns):

            side = (-np.sign(self.wheel.spokes[s].hub_pt[2])+1)/2 + 1
            elset_s = elset + '{:d}'.format(int(side))

            out_str += '*ELEMENT, type=B31, elset={:s}\n'.format(elset_s)
            out_str += '{elnum:d}, {n1:d}, {n2:d}\n'\
                .format(elnum=10100 + s+1, n1=10100 + s+1, n2=10200 + s+1)
            out_str += '*ELGEN, elset={:s}\n'.format(elset_s)
            out_str += '{mel:d}, {nel:d}, {ninc:d}, {einc:d}\n'\
                .format(mel=10100 + s+1, nel=self.n_spk,
                        ninc=100, einc=100)

        # Add all spokes to primary elset
        out_str += '*ELSET, elset={:s}\n{:s}, {:s}\n'.format(elset,
                                                             elset+'1',
                                                             elset+'2')

        # Create elset for spoke reference elements
        out_str += '*ELSET, elset=elsetSpokeRef, generate\n'
        out_str += ' {first:5d}, {last:5d}, {inc:5d}\n'\
            .format(first=10201, last=(10200+ns), inc=1)

        return out_str

    def write_rigid_ties(self):
        'Create rigid bodies to connect spokes to rim.'

        out_str = ''

        for s in range(len(self.i_spoke_nips)):
            # Create a node set for all nodes to be connected
            i_rim = self.i_spoke_nips[s]
            nn_rim = i_rim*len(self.rim_xc.x_pts) + self.rim_xc.ref_pt
            out_str += '*NSET, nset=nsetRig{:d}\n'.format(s+1)
            out_str += ' {:5d}\n'.format(nn_rim)
            i_spk = self.get_spokes_at_rim_point(i_rim)
            for i in i_spk:
                out_str += ' {:5d}\n'.format(i+10101)

            # Create rigid body
            out_str += '*RIGID BODY, ref node={:d}, tie nset=nsetRig{:d}\n'\
                .format(nn_rim, s+1)

        return out_str

    def write_pretension_section(self):
        out_str = ''

        for s in range(len(self.wheel.spokes)):
            out_str += '*PRE-TENSION SECTION, node={:d}, element={:d}\n'\
                .format(99000 + s+1, 2000 + s+1)

        return out_str

    def write_beam_sections(self):
        'Write material and beam section block for rim and spokes.'

        r = self.wheel.rim
        s = self.wheel.spokes[0]

        out_str = '*BEAM SECTION, elset=elsetSpokes, material=steel'
        out_str += ', section=CIRC\n{:e}\n0.,0.,-1.\n'.format(s.diameter/2)

        out_str += '*MATERIAL, name=steel\n*ELASTIC\n {:e}, 0.33\n'\
            .format(s.EA / (np.pi/4 * s.diameter**2))
        out_str += '*DENSITY\n 8050.0\n'
        out_str += '*DAMPING, alpha={:e}\n'.format(1000.0)
        out_str += '*EXPANSION, type=iso\n {:e}\n'.format(1.0)

        out_str += '*MATERIAL, name=alu\n*ELASTIC\n {:e}, 0.33\n'\
            .format(r.young_mod)
        out_str += '*DENSITY\n 2700.0\n'
        out_str += '*DAMPING, alpha={:e}\n'.format(1000.0)

        out_str += '*SHELL SECTION, elset=elsetRim, poisson=elastic, '
        out_str += 'material=alu\n {:e}\n'.format(self.rim_xc.thick)

        return out_str

    def __init__(self, wheel, max_curve=0.1, n_spk=10, n_xc=9):
        self.wheel = wheel
        self.max_curve = max_curve
        self.n_spk = n_spk

        self.rim_xc = self.RimXSection(wheel.rim, n_xc)

        self.create_rim_points()


# Testing code
if False:
    w = BicycleWheel()
    w.rim = w.Rim(radius=0.3,
                  area=100e-6,
                  I11=1000.0e-12,    # Saint-Venant torsion constant, J
                  I22=1000.0e-12,    # area moment of inertia (wobble)
                  I33=1000.0e-12,    # area moment of inertia (squish)
                  Iw=0.0,            # warping constant
                  young_mod=69.0e9,  # Young's modulus - aluminum
                  shear_mod=26.0e9)  # shear modulus - aluminum)

    w.hub = w.Hub(diam1=0.04, width1=0.025)
    w.lace_radial(n_spokes=36, diameter=1.0e-3, young_mod=210e9)

    am = AbaqusModel(w)
    print am.write_rim_nodes()
    print am.write_spoke_nodes()
    print am.write_pretension_nodes()
    print am.write_rim_elems()
    print am.write_spoke_elems()
    print am.write_rigid_ties()
    # print am.write_pretension_section()
