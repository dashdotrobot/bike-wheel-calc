import numpy as np
import re

class WheelGeometry:
    'Geometric parameters including size and lacing pattern'

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
        if hub_eyelet <= self.n_hub_nodes and rim_nipple <= self.n_rim_nodes:
            self.lace_hub_n = np.append(self.lace_hub_n, hub_eyelet)
            self.lace_rim_n = np.append(self.lace_rim_n, rim_nipple)
        else:
            print('*** Not enough hub eyelets or spoke nipples have been defined.')

    def remove_spoke(self, hub_eyelet, rim_nipple):
        # TODO
        for s in range(len(self.lace_hub_n)):
            if self.lace_hub_n[s] == hub_eyelet and self.lace_rim_n[s] == rim_nipple:
                print s

    def __init__(self, wheel_file=None, n_vec=np.array([0, 0, 1])):

        print('# Initializing wheel geometry -----------')

        self.n_vec = n_vec  # axial vector from hub center to drive side nut

        if wheel_file is not None:
            self.parse_wheel_file(wheel_file)
