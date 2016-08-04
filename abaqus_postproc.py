from abaqusConstants import *
import numpy as np


def num_frames(odb):
    'Returns the number of frames in an output database.'
    return len(odb.steps[odb.steps.keys()[0]].frames)


def T_avg(odb, f_id):
    'Returns the average spoke tension in frame f.'

    # Get part information
    part = odb.rootAssembly.instances['PART-1-1']
    regionSpokes = part.elementSets['ELSETSPOKES']

    # Get step and field output
    step = odb.steps[odb.steps.keys()[0]]

    # Print table of spoke tension over time
    f = step.frames[f_id]
    sf = f.fieldOutputs['SF']

    sf_at_rim = sf.getSubset(region=regionSpokes,
                             position=INTEGRATION_POINT,
                             elementType='B31')

    T_avg = 0.0
    n = 0
    for T in sf_at_rim.values:
        T_avg = T_avg + T.data[0]
        n += 1
    T_avg = T_avg / n

    return T_avg


def get_node_field(odb, f_id, field_name, comp, n_id=0, nset=None):
    'Return the value of a field at a particular node.'

    part = odb.rootAssembly.instances['PART-1-1']
    step = odb.steps[odb.steps.keys()[0]]

    frame = step.frames[f_id]
    f_out = frame.fieldOutputs[field_name]

    if nset is not None:
        region = part.nodeSets[nset]
    else:
        region = part.nodes[n_id]

    d = f_out.getSubset(region=region,
                        position=NODAL).values[0].data[comp]
    return d


def get_frame_temp(odb, f_id, elset='ELSETSPOKES'):
    'Get spoke temperature (equal to tightening strain).'

    part = odb.rootAssembly.instances['PART-1-1']
    step = odb.steps[odb.steps.keys()[0]]
    frame = step.frames[f_id]

    if 'TEMP' in frame.fieldOutputs.keys():
        f_out = frame.fieldOutputs['TEMP']
        temp = f_out.getSubset(region=part.elementSets[elset],
                               position=INTEGRATION_POINT).values[0].data[0]
    else:
        temp = frame.frameValue

    return temp


def find_T_max(odb):
    'Find maximum average tension.'

    n_f = num_frames(odb)
    return max([T_avg(odb, f) for f in range(0, n_f)])


def write_buckle_data(odb, ref_node=0, fname=None, verbose=False):
    'Write time step, average tension, and U3 displacement.'

    num_f = num_frames(odb)

    t = [get_frame_temp(odb, f) for f in range(num_f)]
    u = [get_node_field(odb, f, 'U', 2, n_id=ref_node)
         for f in range(num_f)]
    T = [T_avg(odb, f) for f in range(num_f)]

    if fname is not None:
        with open(fname, 'w') as f_out:
            f_out.write('{:12s} {:12s} {:12s}\n'.format('time', 'u [m]', 'tension [N]'))
            for tt, uu, TT in zip(t, u, T):
                f_out.write('{:e} {:e} {:e}\n'.format(tt, uu, TT))

    if verbose:
        print '{:12s} {:12s} {:12s}'.format('time', 'u [m]', 'tension [N]')
        for tt, uu, TT in zip(t, u, T):
            print '{:e} {:e} {:e}'.format(tt, uu, TT)


def calc_Tc_Southwell(odb, node_id=0, nset=None, end=1, verbose=False):
    'Calculate buckling tension from Southwell plot.'

    num_f = num_frames(odb)

    u = [get_node_field(odb, f, 'U', 2, n_id=node_id, nset=nset)
         for f in range(1, num_f-(end+1))]
    T = [T_avg(odb, f) for f in range(1, num_f-(end+1))]
    y = [uu/TT for uu, TT in zip(u, T)]

    if verbose:
        for uu, yy in zip(u, y):
            print '{:e} {:e}'.format(uu, yy)

    # Fit a straight line to T/u vs. u
    a = np.polyfit(u, y, 1)
    return 1.0/a[0]


def calc_Tc_nonlinear(odb, tol=0.03):
    'Find critical tension by departure from linearity'

    n_f = num_frames(odb)

    T = [T_avg(odb, f) for f in range(0, n_f)]

    # Derivative of spoke tension w/r.t. time
    dTdt = [T[i] / i for i in range(1, n_f)]

    # Average slope of T vs. time
    dTdt_avg = sum(dTdt[0:20]) / 20

    # Error between T and (dT/dt)*t
    dT_err = [1 - dTdt[i] / dTdt_avg for i in range(len(dTdt))]

    T_c = 0.0
    frame_c = 0
    for d in range(len(dT_err)):
        Ti = T[d + 1]
        e = dT_err[d]

        if e >= tol:
            T_c = Ti
            frame_c = d
            break

    return T_c, frame_c


def print_spoke_tension(odb, f_out):

    # Get part information
    part = odb.rootAssembly.instances['PART-1-1']
    regionSpokesAtRim = part.elementSets['ELSETSPOKESATRIM']

    # Get step and field output
    step = odb.steps[odb.steps.keys()[0]]

    # Print table of spoke tension over time
    for f in step.frames:

        f_out.write('{:12.5e} '.format(f.frameValue))

        sf = f.fieldOutputs['SF']

        sf_at_rim = sf.getSubset(region=regionSpokesAtRim,
                                 position=INTEGRATION_POINT,
                                 elementType='B31')

        for T in sf_at_rim.values:
            f_out.write('{:12.5e} '.format(T.data[0]))

        f_out.write('\n')

    f_out.close()
