import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from ipywidgets import interact

from util import lerp, trilerp_deposit

DATA_NAME =  'TNG-100' #'TNG-100' #'back_trace' #'2020-05-01/global'
SCALING_FACTOR = 1 #for local #20 # for global 10 # for galaxy 1
DATA_PATH = 'data/' + DATA_NAME

# scaling: local = 20, global = 10
# standard: longest axis is 0 to 1, center would be on the origin
# world coordinate -> grid coordinate (switch x and z, scale, shift)

class Deposit:
    def __init__(self, data_path):
        self.data_path = data_path
        self.grid_res = None
        self.grid_size = None # This will be the actual simulation box
        self.grid_center = None
        self.n_channels = 1

        self.params = {
            "sense_angle":        0.2,
            "sense_distance":     0.4,
            "sense_distance_grid":     0.4,
            "sharpness":          2,
            "move_angle":         0.2,
            "move_distance":      0.4,
            "move_distance_grid":      0.4,
            "agent_deposit":      0.1,

            "min": np.array([-1, -1, -1]),
            "max": np.array([-1, -1, -1]),
            "min_x": -100,
            "max_x": 100,
            "min_y": -100,
            "max_y": 100,
            "min_z": -100,
            "max_z": 100,
            "num_agent": 300
            }

        self.read_meta(data_path)
        self.grid_center = self.grid_center - self.params["min"]
        self.grid_adjusted_res = np.array([self.grid_res[2], self.grid_res[1], self.grid_res[0]])
        self.grid_ratio = self.grid_res[0] / self.grid_size[0]
        self.deposit = \
            np.fromfile(data_path + '/trace.bin', dtype=np.float16).reshape(\
                self.grid_res[2],\
                self.grid_res[1],\
                self.grid_res[0],\
                self.n_channels) # Notice that x and z axis are switched
        self.point_coord, self.point_info, self.point_weight \
             = self.read_point_data()
        # point_grid and sorted_point are for detecting agents passing through data points
        self.point_grid, self.sorted_point = self.preprocess_ptdata(downsample=2)
        

    def get_deposit(self, point, interpolation='trilinear', verbose=False):
        # Adjust for center TODO: Reactivate this
        #point = point - (self.grid_center - 0.5 * self.grid_size)

        # Wrap around the box
        if (np.any(point >= self.grid_size) or np.any(point < 0)):
            print("WARNING: get_deposit: Point out of bounds {}".format(point))
            point = point % self.grid_size

        # convert to grid number
        point_g = point * self.grid_ratio

        # Switch x and z because of the data format
        point_g[0], point_g[2] = point_g[2], point_g[0]

        # Get bounds
        point_f, point_c = np.floor(point_g), np.ceil(point_g)

        # Check if ceil is past bounds
        # TODO: Try not to do this
        #if np.any(point_c.astype(int) >= self.grid_adjusted_res):
            #print("!!! {}, {}, {} !!!".format(point_c, self.grid_adjusted_res, point_c % self.grid_adjusted_res))
            #point_c = point_c % self.grid_adjusted_res
        #    return 0.01

        if verbose:
            print("point: {}".format(point))
            print("point_g: ".format(point_g))
            print("point_f: ".format(point_f))
            print("point_c: ".format(point_c))

        if interpolation is 'trilinear':
            return trilerp_deposit(
                [
                    [int(point_f[0]), int(point_c[0])],
                    [int(point_f[1]), int(point_c[1])],
                    [int(point_f[2]), int(point_c[2])]
                ],
                self.deposit,
                point_g,
                self.grid_adjusted_res
                )
        else:
            print("get_deposit: interpolation unknown")
            exit()

    def display_data(self, interactive = False, slice_num = 100):
        voxels_u8 = self.deposit.astype(np.uint8)

        gamma = 0.2
        figsize = 10.0

        if one_slice:
            def plot_voxels_slice(slice, channel):
                plt.figure(figsize = (figsize, figsize))
                plt.imshow(voxels_u8[:, :, slice, channel] ** gamma)
            interact(plot_voxels_slice,\
                slice = (0, self.grid_size[0] - 1, 1),\
                channel = (0, self.n_channels - 1, 1))
        else:
            plt.figure(figsize = (figsize, figsize))
            plt.imshow(voxels_u8[:, :, slice_num, 0] ** gamma)
            plt.show()

    """
        Check if the position is close to one of the point, return information of that point
        TODO: Make it more efficient
    """
    def check_point(self, pt, threshold=0.2):
        for pt_data in self.point_coord:
            if distance.euclidean(pt_data, pt) <= threshold:
                return pt_data

    def check_bound(self, pt, adjust=True, verbose=False):
        # Adjust for center
        if(adjust):
            pt = pt - (self.grid_center - 0.5 * self.grid_size)

        if np.any(pt < 0) or np.any(pt > self.grid_size):
            if verbose:
                print("Out of bound detected: val: {}, bound: {}".format(pt, self.grid_size))
            return False
        return True
    
    def check_bound_grid(self, pt, point_grid=False):
        grid_res = self.grid_res
        
        if point_grid:
            grid_res = self.point_grid_res

        for i in range(len(pt)):
            if pt[i] < 0 or pt[i] >= grid_res[i]:
                return False
        return True

    def read_point_data(self, dimension=3):
        point_coord = []
        point_info = []
        point_weight = []

        # Read point data
        with open(self.data_path + '/pt_data') as pt_data:
            with open(self.data_path + '/full_data', encoding='utf-8') as pt_info_data:
                for ln in pt_data:
                    coord = ln.split(' ')

                    weight = None
                    if len(coord) == 4:
                        weight = float(coord[3])
                        coord = coord[:3]

                    # Dimension checking
                    # WARNING: Currently working with 3D data with weights (4d)
                    #if len(coord) is not dimension:
                    #    continue

                    # Processing point data
                    coord = np.array(list(map(lambda x: float(x), coord)))
                    coord = self.adjust_coord(coord)
                    info = pt_info_data.readline()

                    # Check if point is out of bounds
                    if self.check_bound(coord, verbose=True):
                        if info[-1] == '\n':
                            info = info[:-1]

                        point_coord.append(coord)
                        point_info.append(info)
                        
                        if weight:
                            point_weight.append(weight)
                        else:
                            point_weight.append(1.0)
                        
                    pt_info_data.readline()

        return [point_coord, point_info, point_weight]
    
    def adjust_coord(self, coord):
        return coord * SCALING_FACTOR - self.params['min']

    # Given a point coordinate, return grid coordinate
    def get_grid_coord(self, pt, grid_ratio = None, grid_res = None):
        if not grid_ratio:
            grid_ratio = self.grid_ratio
        if not grid_res:
            grid_res = self.grid_res

        output = np.floor(pt * grid_ratio).astype(int)

        # This is for adjusting for math error
        # The agent grid position might go just a little bit over the bound
        for i in range(len(output)):
            if output[i] == grid_res[i]:
                output[i] -= 1

        return np.floor(pt * grid_ratio).astype(int)

    """
        Generate a sorted list of point coordinates

        From that sorted list, generate a grid
        The grid stores the same resolution as the original deposit
        But for each cell, it stores the index of points belonging to that cell
        The format of each cell is simply the start and end index
    """
    def preprocess_ptdata(self, downsample = 2):
        sorted_point = np.lexsort(([i[2] for i in self.point_coord],
                                    [i[1] for i in self.point_coord],
                                    [i[0] for i in self.point_coord]))

        # Generate grid
        point_grid = np.full((int(self.grid_res[0] / downsample),
                                int(self.grid_res[1] / downsample),
                                int(self.grid_res[2] / downsample), 
                                2), 0)

        self.point_grid_ratio = point_grid.shape[0] / self.grid_size[0]
        self.point_grid_res = point_grid.shape

        start, end = 0, 0
        start_pt = self.point_coord[sorted_point[start]]
        start_pt = self.get_grid_coord(start_pt, self.point_grid_ratio, self.point_grid_res)
        for i in range(len(sorted_point)):
            i_pt = self.point_coord[sorted_point[i]]
            i_pt = self.get_grid_coord(i_pt, self.point_grid_ratio, self.point_grid_res)

            if (not np.all(i_pt == start_pt)):
                end = i

                # Assign point
                point_grid[tuple(start_pt)][0] = start
                point_grid[tuple(start_pt)][1] = end
                #print(point_grid[tuple(start_pt)])
                start = end
                start_pt = self.point_coord[sorted_point[start]]
                start_pt = self.get_grid_coord(start_pt, self.point_grid_ratio, self.point_grid_res)

        point_grid[tuple(start_pt)][0] = start
        point_grid[tuple(start_pt)][1] = len(sorted_point)

        return [point_grid, sorted_point]

    def read_meta(self, data_path):
        with open(data_path + "/export_metadata.txt") as meta_file:
            for ln in meta_file:
                if ':' not in ln:
                    continue

                ln_name, ln_value = ln.split(':')

                if ln_name == "simulation grid resolution":
                    vec = ln_value.split('[')[0].split('x')
                    self.grid_res = np.array([int(vec[0]), int(vec[1]), int(vec[2])])
                elif ln_name == "simulation grid size":
                    vec = ln_value.split('[')[0].split('x')
                    self.grid_size = np.array([float(vec[0]), float(vec[1]), float(vec[2])])
                elif ln_name == "simulation grid center":
                    vec = ln_value.split('[')[0][2:-2].split(',')
                    self.grid_center = np.array([float(vec[0]), float(vec[1]), float(vec[2])])
                elif ln_name == "move distance":
                    self.params["move_distance"] = float(ln_value.split('[')[0])
                elif ln_name == "move distrance grid":
                    self.params["move_distance_grid"] = float(ln_value.split('[')[0])
                elif ln_name == "sense distance":
                    self.params["sense_distance"] = float(ln_value.split('[')[0])
                elif ln_name == "sense distance grid":
                    self.params["sense_distance_grid"] = float(ln_value.split('[')[0])
                elif ln_name == "move spread":
                    self.params["move_angle"] = np.radians(float(ln_value.split('[')[0]))
                elif ln_name == "sense spread":
                    self.params["sense_angle"] = np.radians(float(ln_value.split('[')[0]))
                elif ln_name == "persistence coefficient":
                    self.params["persistence"] = float(ln_value.split('[')[0])
                elif ln_name == "agent deposit":
                    self.params["agent_deposit"] = float(ln_value.split('[')[0])
                elif ln_name == "sampling sharpness":
                    self.params["sharpness"] = float(ln_value.split('[')[0])
    
        self.params['min'] = self.grid_center - self.grid_size / 2
        self.params['max'] = self.grid_center + self.grid_size / 2

    """
        TODO: Testing use, but reserved for future method implementation
    """
    def read_point_density(self, dimension=3, verbose=False):
        densities = []

        for pt in self.point_coord:
            densities.append(np.log10(self.get_deposit(pt, verbose=verbose)))

        return(densities)

    # Makes all deposit zero
    def clear_deposit(self):
        self.deposit = np.zeros(self.deposit.shape)
