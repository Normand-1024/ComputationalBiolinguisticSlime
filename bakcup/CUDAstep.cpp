#include <stdio.h>
#include <math.h>

#define PI 3.1415926535f

#define SENSE_ANGLE 0
#define SENSE_DIST 1
#define SHARPNESS 2
#define MOVE_ANGLE 3
#define MOVE_DISTANCE 4

typedef unsigned int uint;

texture<float, 3> depositTexture;

// Given a, b with shape (3,), return the cross product
__device__ void cross(float* a, float* b, float* output){
    output[0] = a[1] * b[2] - a[2] * b[1];
    output[1] = - (a[0] * b[2] - a[2] * b[0]);
    output[2] = a[0] * b[1] - a[1] * b[0];
}

// Return the dot product
__device__ float dot(float* a, float* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Given vector and axis, rotate an angle
__device__ void rotate(float* vec, float* axis, float angle, float* output){
    float crossP[] = {0, 0, 0};
    float innerP = dot(vec, axis);
    float cosAngle = cosf(angle);
    float sinAngle = sinf(angle);
    cross(axis, vec, crossP);
    output[0] = vec[0] * cosAngle + 
            crossP[0] * sinAngle + 
            axis[0] * innerP * (1 - cosAngle);
    output[1] = vec[1] * cosAngle + 
            crossP[1] * sinAngle + 
            axis[1] * innerP * (1 - cosAngle);
    output[2] = vec[2] * cosAngle + 
            crossP[2] * sinAngle + 
            axis[2] * innerP * (1 - cosAngle);
}

// Rotationally interpolate between the two directions
__device__ void rotateLerpReplace(float* vec1, float* vec2, float angle){
    float crossP[] = {0, 0, 0};
    float output[] = {0, 0, 0};
    cross(vec1, vec2, crossP);

    // Normalize
    float dim = sqrtf(powf(crossP[0], 2) + powf(crossP[1], 2) + powf(crossP[2], 2));
    crossP[0] /= dim; crossP[1] /= dim; crossP[2] /= dim;

    // Rotate
    rotate(vec1, crossP, angle, output);
    vec1[0] = output[0]; vec1[1] = output[1]; vec1[2] = output[2];
}

struct RNG {
    #define BAD_W 0x464fffffU
    #define BAD_Z 0x9068ffffU
    uint m_w;
	uint m_z;

    __device__ void set_seed(uint seed1, uint seed2) {
		m_w = seed1;
		m_z = seed2;
		if (m_w == 0U || m_w == BAD_W) ++m_w;
		if (m_w == 0U || m_z == BAD_Z) ++m_z;
	}

    __device__ void get_seed(uint& seed1, uint& seed2) {
        seed1 = m_w;
        seed2 = m_z;
    }

    __device__ uint random_uint() {
		m_z = 36969U * (m_z & 65535U) + (m_z >> 16U);
		m_w = 18000U * (m_w & 65535U) + (m_w >> 16U);
		return uint((m_z << 16U) + m_w);
	}

    __device__ float random_float() {
		return float(random_uint()) / float(0xFFFFFFFFU);
	}

    __device__ uint wang_hash(uint seed) {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }
};

__global__ void slimePropagate(float *agentArray, int agentNum, 
                 int *grid_res, float *grid_size,
                 float *parameter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float grid_ratio = grid_res[0] / grid_size[0];
    float *pos = &agentArray[idx * 6 + 0 * 3];
    float *dir = &agentArray[idx * 6 + 1 * 3];
    float dirCurrent[] = {dir[0], dir[1], dir[2]};

    if (idx >= agentNum){
        return;
    }

    RNG rng;
    rng.set_seed(
        rng.wang_hash(73*idx),
        rng.wang_hash(pos[0] * pos[1] * pos[2])
    );
    
    // === SENSE PHASE ===
    // Produce random sensing direction
    float dir2[] = {0, 0, 0};

    // Convert current direction to sphere coordinate
    float th = atanf(dir[1] / dir[0]);
    if (th < 0 && dir[0] < 0) th += PI;
    else if (th > 0 && dir[0] < 0) th -= PI;
    float ph = atanf(sqrtf(dir[0] * dir[0] + dir[1] * dir[1]) / dir[2]);
    if (ph < 0 && dir[2] < 0) ph += PI;
    else if (ph > 0 && dir[2] < 0) ph -= PI;

    // Obtain the new sensing direction
    float offph = ph - parameter[SENSE_ANGLE];
    float offdir[] = {sinf(offph) * cosf(th),
                    sinf(offph) * sinf(th),
                    cosf(offph)};
    float randAngle = rng.random_float() * PI * 2 - PI;
    rotate(offdir, dir, randAngle, dir2);

    float dist_s = rng.random_float() * parameter[SENSE_DIST];
    float p0 = tex3D(depositTexture, 
                    fmod((pos[0] + dist_s * dir[0]),
                         grid_size[2]) * grid_ratio + 0.5f,
                    fmod((pos[1] + dist_s * dir[1]),
                         grid_size[1]) * grid_ratio + 0.5f,
                    fmod((pos[2] + dist_s * dir[2]),
                         grid_size[0]) * grid_ratio + 0.5f
                    );
    float p1 = tex3D(depositTexture, 
                    fmod((pos[0] + dist_s * dir2[0]),
                         grid_size[2]) * grid_ratio + 0.5f,
                    fmod((pos[1] + dist_s * dir2[1]),
                         grid_size[1]) * grid_ratio + 0.5f,
                    fmod((pos[2] + dist_s * dir2[2]),
                         grid_size[0]) * grid_ratio + 0.5f
                    );
    p0 = powf(p0, parameter[SHARPNESS]);
    p1 = powf(p1, parameter[SHARPNESS]);

    // === SAMPLE PHASE ===
    if (rng.random_float() > p0 / (p0 + p1)){
        rotateLerpReplace(dir, dir2, parameter[MOVE_ANGLE]);
    }
    float dist_m = rng.random_float() * parameter[MOVE_DISTANCE];

    // === UPDATE PHASE ===
    pos[0] = fmod(pos[0] + dir[0] * dist_m, grid_size[0]); 
    pos[1] = fmod(pos[1] + dir[1] * dist_m, grid_size[1]);
    pos[2] = fmod(pos[2] + dir[2] * dist_m, grid_size[2]);
    
    if (idx == 1){
        //printf("%f\n", fmod(pos[0] + dist_s * dir[0], grid_size[0]));
        //printf("%f\n", parameter[SENSE_ANGLE]);
        //printf("%f\n", dir[0] * dir2[0] + dir[1] * dir2[1] + dir[2] * dir2[2]);
        //printf("%f, %f", p0, p1);
        //printf(" - (%f, %f, %f)\n", pos[0], pos[1], pos[2]);
        //cross(pos, dir, crossproduct);
        //float dotproduct = dot(pos, dir);
        //printf("%f, %f, %f", crossproduct[0], crossproduct[1], crossproduct[2]);
        //printf("%f", dotproduct);
        //printf("%f, %f\n", tex3D(depositTexture, 0.0, 1.0, 0.0), tex3D(depositTexture, 0.0, 1.2, 0.0));
        //printf("%d, %d, %d\n", grid_res[0], grid_res[1], grid_res[2]);
        //printf("%f, %f, %f\n", grid_size[0], grid_size[1], grid_size[2]);
        //printf("%d, %d, %d\n", grid_adjusted_res[0], grid_adjusted_res[1], grid_adjusted_res[2]);
        //printf("%f, %f, %f, %f, %f", parameter[0], parameter[1], parameter[2], parameter[3], parameter[4]);
        //printf("%d, %f", length, outputArray[0]);
        //printf("%f, %f", agentArray[0], agentArray[1]);
    }
}
