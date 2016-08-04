#!/usr/bin/env python

"""This module uses an old wheel definition and is depricated."""

import numpy as np


class AbaqusMesh:

    node_fmt_str = ' {:5d}, {:11.4E}, {:11.4E}, {:11.4E}'

    def calc_n_rim_nodes(self):
        a_aug = self.geom.a_rim_nodes.tolist() +\
            [2*np.pi + self.geom.a_rim_nodes[0]]

        nn_spoke_nips = []  # node numbers of spoke nipples
        nn = 1              # current node number
        for n in range(len(self.geom.a_rim_nodes)):
            a_1 = a_aug[n]
            a_2 = a_aug[n + 1]

            # Determine number of elements between here and the next spoke
            n_el = int(np.ceil((a_2 - a_1) / self.a_min))

            nn_spoke_nips.append(nn)
            nn += n_el

        self.nn_spoke_nips = nn_spoke_nips
        self.n_rim_nodes = nn-1

    def rim_nodes(self, nset='nsetRim'):
        out_string = '*NODE, nset={:s}\n'.format(nset)

        a_aug = self.geom.a_rim_nodes.tolist() +\
            [2*np.pi + self.geom.a_rim_nodes[0]]

        nn_spoke_nips = []  # node numbers of spoke nipples
        nn = 1              # current node number
        for n in range(len(self.geom.a_rim_nodes)):
            a_1 = a_aug[n]
            a_2 = a_aug[n + 1]

            # Determine number of elements between here and the next spoke
            n_el = int(np.ceil((a_2 - a_1) / self.a_min))

            nn_spoke_nips.append(nn)
            for i in range(n_el):
                a = a_1 + (a_2 - a_1) * float(i)/n_el
                x = self.geom.d_rim/2 * np.sin(a)
                y = -self.geom.d_rim/2 * np.cos(a)

                out_string += self.node_fmt_str.format(nn, x, y, 0.0) + '\n'
                nn += 1

        out_string += '*NSET, nset=nsetSpokeNip\n'
        for n in nn_spoke_nips:
            out_string += '{:d}\n'.format(n)

        out_string = out_string + '*TRANSFORM, nset=nsetSpokeNip, type=C\n0, 0, 0, 0, 0, 1.0\n'

        return out_string

    def spoke_nodes(self, nset='nsetSpokes'):
        out_str = '*NODE, nset={:s}\n'.format(nset)

        # Iterate over spokes
        for n in range(len(self.geom.lace_rim_n)):
            n_rim = self.geom.lace_rim_n[n]
            n_hub = self.geom.lace_hub_n[n]
            a_rim = self.geom.a_rim_nodes[n_rim-1]
            a_hub = self.geom.a_hub_nodes[n_hub-1]

            s = self.geom.s_hub_nodes[n_hub - 1]

            if s == 1:
                d_hub = self.geom.d1_hub
                w_hub = self.geom.w1_hub
            elif s == -1:
                d_hub = self.geom.d2_hub
                w_hub = self.geom.w2_hub

            x_r = self.geom.d_rim/2 * np.sin(a_rim)
            y_r = -self.geom.d_rim/2 * np.cos(a_rim)
            z_r = 0.0

            x_h = d_hub/2 * np.sin(a_hub)
            y_h = -d_hub/2 * np.cos(a_hub)
            z_h = w_hub * (2*(n % 2) - 1)

            x = np.linspace(x_r, x_h, self.n_spk+1)
            y = np.linspace(y_r, y_h, self.n_spk+1)
            z = np.linspace(z_r, z_h, self.n_spk+1)

            for i in range(1, self.n_spk+1):
                nn = self.nn_spoke_nips[n] + 1000*i

                out_str += self.node_fmt_str \
                    .format(nn, x[i], y[i], z[i]) + '\n'

        # Node set for hub nodes
        out_str += '*NSET, nset=nsetHub\n'
        for n in self.nn_spoke_nips:
            out_str += '{:d}\n'.format(1000*self.n_spk + n)

        return out_str

    def pretension_nodes(self):
        out_str = ''

        for n in self.nn_spoke_nips:
            out_str += '*NODE, nset=nsetPreT_{:d}\n'.format(n)
            out_str += '{:d}, 0.0, 0.0, 0.0\n'.format(99000 + n)

        return out_str

    def rim_elems(self, elset='elsetRim'):

        out_str = '*ELEMENT, type=B31, elset={:s}\n1, 1, 2\n'.format(elset)
        out_str += '{:d}, {:d}, 1\n'.format(self.n_rim_nodes, self.n_rim_nodes)
        out_str += '*ELGEN, elset={:s}\n1, {:d}\n'\
            .format(elset, self.n_rim_nodes-1)

        return out_str

    def spoke_elems(self, elset='elsetSpokes'):
        out_str = ''

        # Create spoke elements
        for s in range(len(self.geom.lace_hub_n)):
            n_hub = self.geom.lace_hub_n[s]
            nn_rim = self.nn_spoke_nips[self.geom.lace_rim_n[s] - 1]

            # Determine which side of the wheel it is on
            side = self.geom.s_hub_nodes[n_hub - 1]
            if side == 1:
                elset_s = elset + '1'
            else:
                elset_s = elset + '2'

            out_str += '*ELEMENT, type=B31, elset={:s}\n'.format(elset_s)
            out_str += '{elnum:d}, {n1:d}, {n2:d}\n'\
                .format(elnum=1000 + nn_rim, n1=nn_rim, n2=1000 + nn_rim)
            out_str += '*ELGEN, elset={:s}\n'.format(elset_s)
            out_str += '{mel:d}, {nel:d}, {ninc:d}, {einc:d}\n'\
                .format(mel=1000 + nn_rim, nel=self.n_spk,
                        ninc=1000, einc=1000)

        # Add all spokes to primary elset
        out_str += '*ELSET, elset={:s}\n{:s}, {:s}\n'.format(elset,
                                                             elset+'1',
                                                             elset+'2')

        # Create elset for spoke elements contacting rim
        out_str += '*ELSET, elset=elsetSpokesAtRim\n'
        for s in range(len(self.geom.lace_hub_n)):
            nn_rim = self.nn_spoke_nips[self.geom.lace_rim_n[s] - 1]
            out_str += '{:d}\n'.format(1000 + nn_rim)

        return out_str

    def preT_section(self):
        out_str = ''
        for n in self.nn_spoke_nips:
            out_str += '*PRE-TENSION SECTION, node={:d}, element={:d}\n'\
                .format(99000 + n, 2000 + n)

        return out_str

    def __init__(self, geom, r_sec, s_sec, n_bspk=2, n_spk=10):
        self.geom = geom
        self.r_sec = r_sec
        self.s_sec = s_sec

        self.n_bspk = n_bspk
        self.n_spk = n_spk

        self.a_min = 2*np.pi/(2*36 - 1)

        self.calc_n_rim_nodes()
