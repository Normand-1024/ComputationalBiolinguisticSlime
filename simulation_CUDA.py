# TODO: Passing deposit into the thing

# TUTORIAL: 
#           
#           imagetexture = mod.get_texref("imagetexture")
#           cuda.matrix_to_texref(imagearray, imagetexture, order='C')
#           texrefs=[imagetexture]
#           texture<float, 2> imagetexture;
#           tex2D(imagetexture, 0, 0);

# Note: Tested random function for 64 * 50000 threads, the distribution looks pretty good

import random
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\bin\HostX64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

from simulation import Agent, SlimeSimulation
from deposit_reader import DATA_PATH, Deposit

# >>> Macro Variables <<<
BLOCKDIM = (64, 1, 1)
AGENT_POS, AGENT_DIR = range(2)
SENSE_ANGLE, SENSE_DIST,\
 SHARPNESS, MOVE_ANGLE, MOVE_DISTANCE = range(5)

# >>> Help functions <<<
def get_agent_array(list_of_agents):
    agent_array = np.zeros((len(list_of_agents), 2, 3))
    
    for i in range(len(list_of_agents)):
        agent_array[i][AGENT_POS] = list_of_agents[i].pos
        agent_array[i][AGENT_DIR] = list_of_agents[i].dir

    return agent_array.astype(np.float32)

def array_to_cuda(nparray):
    descr = cuda.ArrayDescriptor3D()

    descr.width, descr.height, descr.depth = nparray.shape[:3]
    descr.format = cuda.dtype_to_array_format(nparray.dtype)
    descr.num_channels = 1
    ary = cuda.Array(descr)

    copy = cuda.Memcpy3D()
    copy.set_src_host(nparray)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = \
        nparray.strides[0]
    copy.height = descr.height
    copy.depth = copy.depth
    copy()

    return ary

# >>> Main Simulation <<<
class CUDA_slime:
    def __init__(self, deposit, agent_array,\
                    grid_res, grid_size,\
                    parameter):
        self.blockdim = (64, 1, 1)
        self.deposit = deposit
        self.agent_array = agent_array
        self.grid_res = grid_res
        self.grid_size = grid_size
        self.parameter = parameter

        sourceFile = open("CUDAstep.cpp")
        self.mod = SourceModule(sourceFile.read())
        self.slimePropagate = self.mod.get_function("slimePropagate")

        self.deposit_texture = self.mod.get_texref("depositTexture")
        self.deposit_texture.set_array(cuda.np_to_array(deposit, order='C'))
        self.deposit_texture.set_address_mode(0, cuda.address_mode.CLAMP) # Use CLAMP because WRAP doesnt work
        self.deposit_texture.set_address_mode(1, cuda.address_mode.CLAMP)
        self.deposit_texture.set_address_mode(2, cuda.address_mode.CLAMP)
        self.deposit_texture.set_filter_mode(cuda.filter_mode.LINEAR)

    def step(self):
        agentNum = np.int32(self.agent_array.shape[0])
        self.slimePropagate(cuda.InOut(self.agent_array), agentNum,\
             cuda.In(self.grid_res), cuda.In(self.grid_size),\
             cuda.In(self.parameter),\
             texrefs=[self.deposit_texture],\
             block=self.blockdim, \
             grid=(int(agentNum // self.blockdim[0] + 1),1))

# ================
# MAIN HERE
# ================

if __name__ == "__main__":
    depo = Deposit(DATA_PATH)
    simu = SlimeSimulation(depo)
    simu.initialize_agents(num_agents=300, spawn_at=simu.depo.point_coord[0])
    print("Spawn at: ", simu.depo.point_coord[3])

    grid_res = depo.grid_res.astype(np.int32) # unadjusted, dimension of the grid in array coordinate
    grid_size = depo.grid_size.astype(np.float32) # unadjusted, dimension of the grid in world coordinate
    grid_ratio = depo.grid_ratio # res / size
    grid_adjusted_res = depo.grid_adjusted_res.astype(np.int32) # x and z exchange position
    parameter = np.array([depo.params["sense_angle"],\
                depo.params["sense_distance"],\
                depo.params["sharpness"],\
                depo.params["move_angle"],\
                depo.params["move_distance"]]).astype(np.float32)

    agent_array = get_agent_array(simu.agents)
    deposit = np.squeeze(depo.deposit.astype(np.float32), axis=3)
    em = np.array([])

    slime = CUDA_slime(deposit, agent_array,\
        grid_res, grid_size,\
        parameter)
    slime.step()

    exit()