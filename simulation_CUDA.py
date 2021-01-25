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
import math
import re
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

def array_to_cuda(nparray, if_surface=False):
    descr = cuda.ArrayDescriptor3D()

    descr.width, descr.height, descr.depth = nparray.shape[:3]
    descr.format = cuda.dtype_to_array_format(nparray.dtype)
    descr.num_channels = 1

    if if_surface:
        descr.flags  = cuda.array3d_flags.SURFACE_LDST

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

def slugify(value):
    value = re.sub('[^\w\-_\. ]', '_', value)
    return value

# >>> Main Simulation <<<
class CUDA_slime:
    def __init__(self, deposit, agent_array,\
                    point_coord, point_info, point_weight,\
                    grid_res, grid_size,\
                    parameter,\
                    trace_downsample_factor=1.5,\
                    ranking_threshold=10):
        self.ranking_threshold = ranking_threshold
        self.blockdim = BLOCKDIM
        self.deposit = deposit
        self.agent_array = agent_array
        self.grid_res = grid_res
        self.grid_size = grid_size
        self.parameter = parameter
        self.agent_trace = []
        self.agent_trace_texture = np.zeros(
                    (math.floor(deposit.shape[2] / trace_downsample_factor),
                     math.floor(deposit.shape[1] / trace_downsample_factor),
                     math.floor(deposit.shape[0] / trace_downsample_factor)),
                     dtype=np.int32)
        self.traceshape = np.int32(self.agent_trace_texture.shape)
        self.worldToTraceGridRatio = np.float32(self.traceshape[0] / self.grid_size[0])
        self.point_coord, self.point_info, self.point_weight = point_coord, point_info, point_weight
        self.point_coord = np.array(point_coord, dtype=np.float32)
        self.similarity_rank = np.zeros(
            (len(self.point_info),), dtype=np.int32)

        sourceFile = open("CUDAstep.cpp")
        self.mod = SourceModule(sourceFile.read())
        self.slimePropagate = self.mod.get_function("slimePropagate")
        self.slimePropagateRecord = self.mod.get_function("slimePropagateAndRecord")

        self.deposit_texture = self.mod.get_texref("depositTexture")
        self.deposit_texture.set_array(cuda.np_to_array(deposit, order='C'))
        self.deposit_texture.set_address_mode(0, cuda.address_mode.CLAMP) # Use CLAMP because WRAP doesnt work
        self.deposit_texture.set_address_mode(1, cuda.address_mode.CLAMP)
        self.deposit_texture.set_address_mode(2, cuda.address_mode.CLAMP)
        self.deposit_texture.set_filter_mode(cuda.filter_mode.LINEAR)
        #self.deposit_texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

    """
        Slime Propagate and record trace at the same time, if memory permits
    """
    # Add_trace is set to false by default because of no use for it
    def step(self,add_trace=False):
        if not hasattr(self, "trace_texture_gpu"):
            print("WARNING: slime CUDA simulation instance doesn't have trace_texture_gpu. Call transferAgentTraceTex()")
            return

        agentNum = np.int32(self.agent_array.shape[0])

        self.slimePropagateRecord(\
             cuda.InOut(self.agent_array), agentNum,\
             cuda.In(self.grid_res), cuda.In(self.grid_size),\
             cuda.In(self.parameter),\
             self.trace_texture_gpu, self.worldToTraceGridRatio,\
             self.traceshape[2], self.traceshape[1],\
             texrefs=[self.deposit_texture],\
             block=self.blockdim,\
             grid=(int(agentNum // self.blockdim[0] + 1),1))

        if (add_trace):
            for agent in self.agent_array:
                self.agent_trace += [agent[0].copy()] 

    def stepWithoutRecord(self, add_trace=True):
        agentNum = np.int32(self.agent_array.shape[0])
        self.slimePropagate(cuda.InOut(self.agent_array), agentNum,\
             cuda.In(self.grid_res), cuda.In(self.grid_size),\
             cuda.In(self.parameter),\
             texrefs=[self.deposit_texture],\
             block=self.blockdim, \
             grid=(int(agentNum // self.blockdim[0] + 1),1))

        if (add_trace):
            for agent in self.agent_array:
                self.agent_trace += [agent[0].copy()] 

    """
        Record Trace, delete the deposit to save memory
    """
    def recordTrace(self):
        self.recordTrace = self.mod.get_function("recordTrace")
        self.deposit_texture = None

        self.agent_trace = np.array(self.agent_trace, dtype=np.float32)
        agentNum = np.int32(self.agent_array.shape[0])
        traceLen = np.int32(len(self.agent_trace))
        traceshape = np.int32(self.agent_trace_texture.shape)
        self.worldToTraceGridRatio = np.float32(traceshape[0] / self.grid_size[0])

        self.recordTrace(cuda.In(self.agent_trace), traceLen,\
                 self.worldToTraceGridRatio,\
                 cuda.InOut(self.agent_trace_texture), \
                 traceshape[2], traceshape[1],\
                 cuda.In(traceshape),\
                 block=self.blockdim, \
                 grid=(int(agentNum // self.blockdim[0] + 1),1))

        print("Out texture sum: " + str(np.sum(self.agent_trace_texture)))

    def generateSimilarity(self, sense_dist = 1):
        self.generateSimilairy = self.mod.get_function("generateSimilairy")

        grid_sense_dist = np.int32(np.ceil(sense_dist * self.worldToTraceGridRatio))
        agentNum = np.int32(self.agent_array.shape[0])

        self.generateSimilairy(
                 cuda.InOut(self.similarity_rank),\
                 cuda.In(self.point_coord),\
                 np.int32(len(self.point_coord)), grid_sense_dist,\
                 cuda.In(self.agent_trace_texture),\
                 self.worldToTraceGridRatio,\
                 self.traceshape[2], self.traceshape[1], self.traceshape[0],\
                 block=self.blockdim, \
                 grid=(int(agentNum // self.blockdim[0] + 1),1))

    # Rank format: (index, similarity, weight)
    #   sorted by similarity
    def getSimilarityRank(self, sort = True, reverse = True):
        ranking = []
        for i in range(len(self.similarity_rank)):
            if (self.similarity_rank[i] >= self.ranking_threshold):
                ranking.append((i,\
                     self.similarity_rank[i],\
                     self.point_weight[i],\
                     list(self.point_coord[i])))
        if sort:
            ranking.sort(reverse=reverse, key=lambda x: x[1])
        return ranking
        
    def writeSimilarity(self, filename):
        with open(filename, 'w+', encoding='utf-8') as f:
            for i in range(len(self.similarity_rank)):
                if (self.similarity_rank[i] >= self.ranking_threshold):
                    f.write(self.point_info[i] + '\n' +\
                        str(self.similarity_rank[i]) + '\n')

    def transferAgentTraceTex(self):
        self.trace_texture_gpu = cuda.mem_alloc(self.agent_trace_texture.nbytes)
        cuda.memcpy_htod(self.trace_texture_gpu, self.agent_trace_texture)
        
    def retrieveAgentTraceTex(self):
        cuda.memcpy_dtoh(self.agent_trace_texture, self.trace_texture_gpu)

# ================
# MAIN HERE
# ================

if __name__ == "__main__":
    depo = Deposit(DATA_PATH)
    simu = SlimeSimulation(depo)
    simu.initialize_agents(num_agents=300, spawn_at=simu.depo.point_coord[0])

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

    print("Grid Size: ", grid_size)
    print("Grid Res: ", grid_res)
    print("Grid Ratio: ", grid_ratio)

    #print("In Python: \n  Position: ", agent_array[0][0], "\n  Deposit: ", depo.get_deposit(agent_array[0][0]))
    p0 = simu.depo.get_deposit(agent_array[0][0] + agent_array[0][1] * depo.params["sense_distance"])
    slime = CUDA_slime(deposit, agent_array,\
        depo.point_coord, depo.point_info,\
        grid_res, grid_size,\
        parameter)
    for i in range(500):
        slime.step()
    slime.recordTrace()
    slime.generateSimilarity()
    slime.outputSimilarity("1111.txt")
    for i, item in enumerate(slime.similarity_rank):
        if item > 0:
            print(depo.point_info[i], " ", item, " ", depo.point_coord[i])
            
    print(">>>>>", depo.point_info[0], " ", slime.similarity_rank[0], " ", depo.point_coord[0])

    point = np.int32(np.floor(simu.depo.point_coord[0] * depo.grid_ratio))
    lowerb = math.floor((simu.depo.point_coord[0][2] - 10) * depo.grid_ratio)
    upperb = math.floor((simu.depo.point_coord[0][2] + 10) * depo.grid_ratio)
    trace_tex = slime.agent_trace_texture[:,:,lowerb : upperb]
    print(np.sum(trace_tex))
    trace_tex = np.sum(trace_tex, axis=2)
    plt.imshow(trace_tex, origin='lower')
    plt.show()  

    exit()