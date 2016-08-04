#!/usr/bin/env python

import numpy as np


def write_wheel_mesh(geom, rsec, ssec, f, n_bspk=2, n_spk=10, preT=True):
    'Write ABAQUS input file for a radially-spoked wheel.'

    n_spokes = len(geom.lace_hub_n)

    # -------- NODES -------- #

    node_format_string = ' {:5d}, {:11.4E}, {:11.4E}, {:11.4E}'

    # rim nodes
    f.write('*NODE, nset=nsetRim\n')

    a_rim = 2*np.pi*np.linspace(0, float((n_spokes*n_bspk-1))/(n_spokes*n_bspk),
                                n_spokes * n_bspk)

    n_rim = range(1, n_spokes*n_bspk + 1)
    for n, a in zip(n_rim, a_rim):
        x = geom.d_rim/2 * np.sin(a)
        y = -geom.d_rim/2 * np.cos(a)
        z = 0.0

        f.write(node_format_string.format(n, x, y, z) + '\n')

    # rim-spoke connection nodes
    # n_sp_nip = np.arange(n_bspk/2 + 1,
                         # n_bspk/2 + (n_spokes-1)*n_bspk + 2,
                         # n_bspk)
    n_sp_nip = np.arange(1,
                         (n_spokes-1)*n_bspk + 2,
                         n_bspk)

    f.write("""*NSET, nset=nsetSpokeNip, generate
{:d}, {:d}, {:d}\n""".format(n_sp_nip[0], n_sp_nip[-1], n_bspk))

    f.write("""*TRANSFORM, nset=nsetSpokeNip, type=C
0, 0, 0, 0, 0, 1.0\n""")

    # spoke nodes
    f.write('*NODE, nset=nsetSpokes\n')

    a_hub = 2*np.pi*np.linspace(0, float((n_spokes-1))/n_spokes, n_spokes)  # +\
        # np.pi / (n_spokes*n_bspk)

    for n, a in zip(n_sp_nip, a_hub):

        # determine proper hub width (left or drive side)
        if (n-1)/n_bspk % 2 == 0:
            z_max = geom.w1_hub
            d_hub = geom.d1_hub
        else:
            z_max = -geom.w2_hub
            d_hub = geom.d2_hub

        # spoke nodes
        for n_along_spk in range(1, n_spk + 1):
            r_node = geom.d_rim/2 -\
                float(n_along_spk)/n_spk * (geom.d_rim/2 - d_hub/2)

            x = r_node * np.sin(a)
            y = -r_node * np.cos(a)
            z = float(n_along_spk)/n_spk * z_max

            f.write(node_format_string.
                    format(n + n_along_spk*1000, x, y, z) + '\n')

    # pre-tension nodes
    if preT:
        for n in n_sp_nip:
            f.write('*NODE, nset=nsetPreT_{:d}\n'.format(n))
            f.write('{:d}, 0.0, 0.0, 0.0\n'.format(99000 + n))

    f.write("""*NSET, nset=nsetHub, generate
{:d}, {:d}, {:d}\n""".format(1000*n_spk + n_sp_nip[0],
                                 1000*n_spk + n_sp_nip[-1],
                                 n_bspk))

    # -------- ELEMENTS -------- #

    f.write("""*ELEMENT, type=B31, elset=elsetRim
1, 1, 2
{:d}, {:d}, 1\n""".format(n_spokes*n_bspk, n_spokes*n_bspk))

    f.write("""*ELGEN, elset=elsetRim
1, {:d}\n""".format(n_spokes*n_bspk - 1))

    f.write("""*ELEMENT, type=B31, elset=elsetSpokes
{:d}, {:d}, {:d}\n""".format(1000 + n_sp_nip[0],   # element number
                             n_sp_nip[0],          # rim node
                             1000 + n_sp_nip[0]))  # spoke node

    f.write("""*ELGEN, elset=elsetSpokes
{first:d}, {ns:d}, {nb:d}, {nb:d}, {nrows:d}, 1000, 1000\n"""
            .format(first=1000 + n_sp_nip[0],
                    ns=n_spokes,
                    nb=n_bspk,
                    nrows=n_spk))

    f.write("""*ELSET, elset=elsetSpokeAtRim, generate
{first:d}, {last:d}, {inc:d}\n""".format(first=1000 + n_sp_nip[0],
                                         last=1000 + n_sp_nip[-1],
                                         inc=n_bspk))

    f.write("""** Section definitions
*BEAM SECTION, elset=elsetSpokes, material=steel, section=CIRC
{:e}
0.,0.,-1.""".format(np.sqrt(ssec.area/np.pi)))

    f.write("""
*BEAM GENERAL SECTION, elset=elsetRim, section=GENERAL, density=2700.0
{A:10.4E}, {I33:10.4E}, 0., {I22:10.4E}, {I11:10.4E}
0.,0.,-1.
{E:10.4e}, {G:10.4}\n""".format(A=rsec.area,
                                I11=rsec.I11, I22=rsec.I22, I33=rsec.I33,
                                E=rsec.young_mod, G=rsec.shear_mod))

    # spoke pretension sections
    if preT:
        for n in n_sp_nip:
            f.write('*PRE-TENSION SECTION, node={:d}, element={:d}\n'
                    .format(99000 + n, 2000 + n))

    f.write("""**
**
** MATERIALS
**
*MATERIAL, name=steel
*ELASTIC
{E:10.4e}, 0.33
*DENSITY
8050.0
*DAMPING, alpha=10.0
*EXPANSION, type=iso
1.0
""".format(E=ssec.young_mod))

    return (n_bspk + n_spk + 1) * n_spokes, n_rim, n_sp_nip
