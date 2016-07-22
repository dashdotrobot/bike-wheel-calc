import numpy as np
import matplotlib.pyplot as pp
import copy
from scipy import interpolate
from helpers import *

EL_RIM = 1
EL_SPOKE = 2
N_RIM = 1
N_HUB = 2
N_REF = 3


class FEMSolution:
    el_rim = 1
    el_spoke = 2

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

            u_rim = self.nodal_disp[n, 0:3]

            u_rad = np.append(u_rad, u_rim.dot(x) / np.sqrt(x.dot(x)))
            u_tan = np.append(u_tan, u_rim.dot(n_tan) / np.sqrt(n_tan.dot(n_tan)))

        u_rad = np.array(u_rad)
        u_tan = np.array(u_tan)

        return u_rad, u_tan

    def get_deformed_coords(self, node_id, scale_rad=1.0, scale_tan=1.0):
        'Get coordinates of nodes in scaled deformed configuration.'

        # Allow array input for node_id and/or dof
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]

        x_def = np.array([])
        y_def = np.array([])
        z_def = np.array([])  # TODO

        u_rad, u_tan = self.get_polar_displacements(node_id)

        for n in range(len(node_id)):
            n_id = node_id[n]

            x = np.array([self.x_nodes[n_id], self.y_nodes[n_id], self.z_nodes[n_id]])
            n_tan = np.cross(np.array([0, 0, 1]), x)

            u = scale_rad*u_rad * x/np.sqrt(x.dot(x)) + scale_tan*u_tan * n_tan

            x_def = np.append(x_def, x[0] + u[0])
            y_def = np.append(y_def, x[1] + u[1])

        return x_def, y_def

    def get_spoke_tension(self):
        'Return a list of spoke tensions'
        spoke_tension = [self.el_stress[e][0]
                         for e in range(len(self.el_type))
                         if self.el_type[e] == EL_SPOKE]

        return np.array(spoke_tension)

    def get_rim_stress(self, comp):
        """Return a list of rim stresses, specified by comp.'

        Stress components are identified as follows:
        0 = normal tension
        1 = in-plane shear
        2 = out-of-plane shear
        3 = torsion (twisting) moment
        4 = out-of-plane (wobbling) moment
        5 = in-plane (squashing) moment
        """

        rim_stress = [self.el_stress[e][comp]
                      for e in range(len(self.el_type))
                      if self.el_type[e] == EL_RIM]

        return np.array(rim_stress)

    def plot_deformed_wheel(self, scale_rad=0.1, scale_tan=0.0):
        'Plot the exaggerated, deformed wheel shape.'

        rim_nodes = np.where(self.type_nodes == N_RIM)[0]

        u_rad, u_tan = self.get_polar_displacements(rim_nodes)

        # Scale the largest displacement to a percentage of the rim radius
        if max(np.abs(u_rad)) > 0:
            scale_rad = self.geom.d_rim/2 / max(np.abs(u_rad)) * scale_rad
            scale_tan = self.geom.d_rim/2 / max(np.abs(u_rad)) * scale_tan
        else:
            scale_rad = 0
            scale_tan = 0

        theta = self.geom.a_rim_nodes - np.pi/2

        # Calculate coordinates in deformed configuration
        theta_def = theta + scale_tan * u_tan / (self.geom.d_rim / 2)
        r_def = (self.geom.d_rim / 2) + scale_rad * u_rad

        theta_ii = np.linspace(-np.pi/2, 3*np.pi/2, 1000)

        theta_interp = interpolate.interp1d(np.append(theta, theta[0] + 2*np.pi),
                                            np.append(theta_def, theta_def[0] + 2*np.pi))
        theta_def_ii = theta_interp(theta_ii)

        r_def_ii = interp_periodic(theta, r_def, theta_ii)

        # Plot undeformed rim
        pp.plot(self.geom.d_rim/2 * np.cos(theta_ii),
                self.geom.d_rim/2 * np.sin(theta_ii), 'k:')

        # Plot deformed rim
        pp.plot(r_def_ii * np.cos(theta_def_ii), r_def_ii * np.sin(theta_def_ii), 'k', linewidth=2.0)

        # Plot spokes in deformed configuration
        for e in np.where(self.el_type == EL_SPOKE)[0]:
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

    def plot_spoke_tension(self, fig=None):
        'Plot the spoke tensions on a polar plot'

        if fig is None:
            fig = pp.figure()

        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax1.set_xticklabels([])

        # Get list of spoke tension
        spoke_tension = self.get_spoke_tension()

        # drive-side spokes
        angle = self.geom.a_rim_nodes[np.where(self.geom.s_hub_nodes == 1)[0]] - np.pi/2
        tension = spoke_tension[np.where(self.geom.s_hub_nodes == 1)[0]]

        angle = np.append(angle, angle[0])
        tension = np.append(tension, tension[0])

        l_drive, = ax1.plot(angle, tension, '.-',
                            color='#69D2E7', linewidth=3, markersize=15)

        # non-drive-side spokes
        angle = self.geom.a_rim_nodes[np.where(self.geom.s_hub_nodes == -1)[0]] - np.pi/2
        tension = spoke_tension[np.where(self.geom.s_hub_nodes == -1)[0]]

        angle = np.append(angle, angle[0])
        tension = np.append(tension, tension[0])

        l_nondrive, = ax1.plot(angle, tension, '.-',
                               color='#F38630', linewidth=3, markersize=15)

        l_drive.set_label('right')
        l_nondrive.set_label('left')

        ax1.legend(loc='center')

    def __init__(self, fem):
        self.updated = False

        self.wheel = copy.copy(fem.wheel)

        self.x_nodes = fem.x_nodes.copy()
        self.y_nodes = fem.y_nodes.copy()
        self.z_nodes = fem.z_nodes.copy()
        self.type_nodes = fem.type_nodes.copy()

        # Elements and connectivity
        self.el_type = fem.el_type.copy()
        self.el_n1 = fem.el_n1.copy()
        self.el_n2 = fem.el_n2.copy()

        # nodal displacements and reation forces
        self.nodal_disp = None
        self.nodal_rxn = None
        self.dof_rxn = None

        # element stresses
        self.el_stress = []
