import numpy as np
import re

class WheelGeometry:
    'Geometric parameters including size and lacing pattern'

    def __str__(self):
        return ("Bicycle wheel geometry object\n"
                "  rim diameter        :{rd:6.1f} mm\n"
                "  hub diameter        :{hd1:6.1f} mm\n"
                "  hub diameter (drive):{hd2:6.1f} mm\n"
                "  hub width           :{hw1:6.1f} mm\n"
                "  hub width (drive)   :{hw2:6.1f} mm\n"
                "  number of spokes    :{ns:6d}\n").format(rd=self.d_rim*1000,
                                                         hd1=self.d1_hub*1000,
                                                         hd2=self.d2_hub*1000,
                                                         hw1=self.w1_hub*1000,
                                                         hw2=self.w2_hub*1000,
                                                         ns=len(self.lace_hub_n))

    def sort_spokes(self):
        'Sort spokes by rim node'

        self.lace_hub_n = self.lace_hub_n[self.lace_rim_n.argsort()]
        self.lace_rim_n.sort()

    def parse_wheel_file(self, fname):

        def keyword_rim(l, f):

            # Default values
            args = {'diam': 0.6, 'spacing': 'default', 'N': 36}

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

                    if re.match('^\d+$', l_rim_node[0]):  # check if line is an integer
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
            args = {'diam': 0.6, 'diam_drive': None, 'width': 0.035, 'width_drive': None, 
                    'spacing': 'default', 'N': 36}

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
            if args['diam_drive'] is not None:
                self.d2_hub = float(args['diam_drive'])

            self.w1_hub = float(args['width'])
            self.w2_hub = self.w1_hub
            if args['width_drive'] is not None:
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

                    if re.match('^\d+$', l_hub_node[0]):  # check if line is an integer
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
            args = {'pattern': 'default'}

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

                    if re.match('^\d+$', l_lace[0]):  # check if line is an integer
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

    def add_spoke(self, hub_eyelet, rim_nipple):
        """Add a spoke connecting hub_eyelet and rim_nipple. Inputs can be
           integers, or lists of equal length"""

        # Allow array input for eyelets or nipples
        if not hasattr(hub_eyelet, '__iter__'):
            hub_eyelet = [hub_eyelet]
        if not hasattr(rim_nipple, '__iter__'):
            rim_nipple = [rim_nipple]

        # Make sure there are enough eyelets and nipples
        cond1 = np.max(hub_eyelet) <= self.n_hub_nodes
        cond2 = np.max(rim_nipple) <= self.n_rim_nodes

        if cond1 and cond2:
            for h, r in zip(hub_eyelet, rim_nipple):
                self.lace_hub_n = np.append(self.lace_hub_n, h)
                self.lace_rim_n = np.append(self.lace_rim_n, r)
        else:
            print('*** Not enough hub eyelets or spoke nipples have been defined.')

        self.sort_spokes()

    def remove_spoke(self, hub_eyelet, rim_nipple):
        for s in range(len(self.lace_hub_n)):
            if self.lace_hub_n[s] == hub_eyelet and self.lace_rim_n[s] == rim_nipple:
                self.lace_hub_n = np.delete(self.lace_hub_n, s)
                self.lace_rim_n = np.delete(self.lace_rim_n, s)
                return

    def add_nipple(self, angle):

        if not hasattr(angle, '__iter__'):
            angle = [angle]

        angle = np.array(angle) * np.pi / 180

        self.a_rim_nodes = np.concatenate((self.a_rim_nodes, angle))
        self.n_rim_nodes = len(self.a_rim_nodes)

    def add_eyelet(self, angle, side):

        if not hasattr(angle, '__iter__'):
            angle = [angle]
        if not hasattr(side, '__iter__'):
            side = [side]

        angle = np.array(angle) * np.pi / 180

        self.a_hub_nodes = np.concatenate((self.a_hub_nodes, angle))
        self.s_hub_nodes = np.concatenate((self.s_hub_nodes, side))

        self.n_hub_nodes = len(self.a_hub_nodes)

    def __init__(self, wheel_file=None, n_spokes=None, rim_diam=0.6,
                 hub_diam=0.04, hub_diam_drive=None, hub_width=0.035, hub_width_drive=None):

        # axial vector from hub center to drive side nut
        self.n_vec = np.array([0, 0, 1])

        if not n_spokes is None:
            # Create equally spaced hub eyelets and spoke nipples
            self.a_rim_nodes = np.linspace(0, 2*np.pi * (n_spokes-1)/n_spokes, n_spokes)
            self.a_hub_nodes = np.linspace(0, 2*np.pi * (n_spokes-1)/n_spokes, n_spokes)
            self.s_hub_nodes = np.zeros(n_spokes, dtype=np.int8)
            self.s_hub_nodes[::2] = 1    # Drive-side nodes
            self.s_hub_nodes[1::2] = -1  # Left-side nodes
            self.n_hub_nodes = n_spokes
            self.n_rim_nodes = n_spokes
        else:
            self.a_rim_nodes = np.array([], dtype=np.float)
            self.a_hub_nodes = np.array([], dtype=np.float)
            self.s_hub_nodes = np.array([], dtype=np.int8)

        self.lace_hub_n = np.array([], dtype=np.int32)
        self.lace_rim_n = np.array([], dtype=np.int32)

        if hub_diam_drive is None:
            hub_diam_drive = hub_diam
        if hub_width_drive is None:
            hub_width_drive = hub_width

        self.d_rim = rim_diam
        self.d1_hub = hub_diam
        self.d2_hub = hub_diam_drive
        self.w1_hub = hub_width
        self.w2_hub = hub_width_drive

        if wheel_file is not None:
            self.parse_wheel_file(wheel_file)
            self.sort_spokes()
