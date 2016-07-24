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

    def get_polar_displacements(self, node_id):
        'Convert nodal displacements to polar form.'

        # Allow array input for node_id
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]

        u_rad = np.array([])
        u_tan = np.array([])
        u_z = np.array([])

        for n in node_id:

            x = np.array([self.x_nodes[n], self.y_nodes[n], self.z_nodes[n]])
            n_tan = np.cross(np.array([0, 0, 1]), x)

            u_rim = self.nodal_disp[n, 0:3]

            u_rad = np.append(u_rad, u_rim.dot(x) / np.sqrt(x.dot(x)))
            u_tan = np.append(u_tan, u_rim.dot(n_tan) /
                              np.sqrt(n_tan.dot(n_tan)))
            u_z = np.append(u_z, u_rim.dot(np.array([0, 0, 1])))

        return u_rad, u_tan, u_z

    def get_deformed_coords(self, node_id, def_scale=1.0):
        'Get coordinates of nodes in scaled deformed configuration.'

        # Allow array input for node_id and/or dof
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]

        u_x = self.nodal_disp[node_id, 0]
        u_y = self.nodal_disp[node_id, 1]
        u_z = self.nodal_disp[node_id, 2]

        x_def = self.x_nodes[node_id] + def_scale*u_x
        y_def = self.y_nodes[node_id] + def_scale*u_y
        z_def = self.z_nodes[node_id] + def_scale*u_z

        return x_def, y_def, z_def

    def get_spoke_tension(self):
        'Return a list of spoke tensions'
        spoke_tension = [self.el_stress[e][0] + self.el_prestress[e]
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

    def plot_deformed_wheel(self, rel_scale=0.1):
        'Plot the exaggerated, deformed wheel shape.'

        rim_nodes = np.where(self.type_nodes == N_RIM)[0]

        u_rad, u_tan, u_z = self.get_polar_displacements(rim_nodes)

        # Scale the largest displacement to a percentage of the rim radius
        u_mag = np.sqrt(u_rad**2 + u_tan**2 + u_z**2)
        if max(u_mag) > 0:
            def_scale = self.wheel.rim.radius / max(u_mag) * rel_scale
        else:
            def_scale = 0.0

        # Angular positions of spoke nipples (x-axis is theta=0)
        theta = np.array([rect2pol(np.array([self.x_nodes[i],
                                             self.y_nodes[i],
                                             self.z_nodes[i]]))[1]
                          for i in range(len(self.x_nodes))
                          if self.type_nodes[i] == N_RIM])

        theta = np.mod(theta + 2*np.pi, 2*np.pi) - np.pi/2

        # Calculate coordinates in deformed configuration
        theta_def = theta + def_scale * u_tan / (self.wheel.rim.radius)
        r_def = (self.wheel.rim.radius) + def_scale * u_rad

        theta_ii = np.linspace(-np.pi/2, 3*np.pi/2, 1000)

        # Wrap the first value to the end to ensure continuity of the interpolation
        theta_interp = interpolate.interp1d(np.append(theta, theta[0] + 2*np.pi),
                                            np.append(theta_def, theta_def[0] + 2*np.pi))

        # Interpolate radial and tangential displacements
        theta_def_ii = theta_interp(theta_ii)
        r_def_ii = interp_periodic(theta, r_def, theta_ii)

        # Plot undeformed rim
        pp.plot(self.wheel.rim.radius * np.cos(theta_ii),
                self.wheel.rim.radius * np.sin(theta_ii), 'k:')

        # Plot deformed rim
        pp.plot(r_def_ii * np.cos(theta_def_ii),
                r_def_ii * np.sin(theta_def_ii), 'k', linewidth=2.0)

        # Plot spokes in deformed configuration
        for e in np.where(self.el_type == EL_SPOKE)[0]:
            n_hub = self.el_n1[e]
            n_rim = self.el_n2[e]

            x_hub, y_hub, z_hub = self.get_deformed_coords(n_hub, def_scale)
            x_rim, y_rim, z_rim = self.get_deformed_coords(n_rim, def_scale)

            pp.plot([x_hub, x_rim], [y_hub, y_rim], 'k-')

        # Axis properties
        ax = pp.gca()
        ax.set_ylim([-1.2*self.wheel.rim.radius, 1.2*self.wheel.rim.radius])
        ax.set_xlim([-1.2*self.wheel.rim.radius, 1.2*self.wheel.rim.radius])

        ax.set_aspect('equal', 'datalim')

        ax.xaxis.set_major_locator(pp.NullLocator())
        ax.yaxis.set_major_locator(pp.NullLocator())

        return pp.gcf()

    def plot_spoke_tension(self, fig=None):
        'Plot the spoke tensions on a polar plot.'

        if fig is None:
            fig = pp.figure()

        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax1.set_xticklabels([])

        # Get list of spoke tension
        spoke_tension = self.get_spoke_tension()

        # Angular positions of spoke nipples (x-axis is theta=0)
        theta_rim = [rect2pol(np.array([self.x_nodes[self.el_n2[i]],
                                        self.y_nodes[self.el_n2[i]],
                                        self.z_nodes[self.el_n2[i]]]))[1]
                     for i in range(len(self.el_type))
                     if self.el_type[i] == EL_SPOKE]

        # Rotate so that theta=0 is down (bottom of the wheel)
        theta_rim = np.array(theta_rim) - np.pi/2

        z_hub = np.array([self.z_nodes[self.el_n1[i]]
                          for i in range(len(self.el_type))
                          if self.el_type[i] == EL_SPOKE])

        # Plot spoke tensions for drive-side spokes
        theta = np.append(theta_rim[z_hub > 0], theta_rim[z_hub > 0][0])
        tension = np.append(spoke_tension[z_hub > 0],
                            spoke_tension[z_hub > 0][0])

        l_drive, = ax1.plot(theta, tension, '.-',
                            color='#69D2E7', linewidth=3, markersize=15)

        # non-drive-side spokes
        theta = np.append(theta_rim[z_hub < 0], theta_rim[z_hub < 0][0])
        tension = np.append(spoke_tension[z_hub < 0],
                            spoke_tension[z_hub < 0][0])

        l_nondrive, = ax1.plot(theta, tension, '.-',
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
        self.el_prestress = []
        self.el_stress = []
