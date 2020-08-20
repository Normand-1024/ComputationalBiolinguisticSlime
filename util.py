import numpy as np

"""
    Linear interpolation, return y
"""
def lerp(x_bound_0, x_bound_1, y_bound_0, y_bound_1, x):
    return y_bound_0 + (y_bound_1 - y_bound_0) * (x - x_bound_0) / (x_bound_1 - x_bound_0)

"""
    Bilinear interpolation

    format: 
        1. coordinates array [x_left, x_right, y_bottom, y_top]
        2. values array [val_bl, val_br, val_tl, val_tr]
        3. point coordinate
"""
def bilerp(coord, val, pt):
    if (type(coord) is not list or type(val) is not list or type(pt) is not tuple):
        print("bilerp: either coord or val is not list or pt is not tuple")
        exit()
    if (len(coord) is not 4 or len(val) is not 4 or len(pt) is not 2):
        print("bilerp: either coord or val or pt has incorrect length")
        exit()

    botm_lerp = lerp(coord[0], coord[1], val[0], val[1], pt[0])
    top_lerp = lerp(coord[0], coord[1], val[2], val[3], pt[0])
    return lerp(coord[2], coord[3], botm_lerp, top_lerp, pt[1])


"""
    Trilinear interpolation
    
    format: 
        1. coordinates array [
                                [x_left, x_right],
                                [y_bottom, y_top],
                                [z_back, z_front]
                            ]
        2. values array
                 (blb: bottom left back, trf: top right front)
                 [val_blb, val_brb, val_tlb, val_trb, val_blf, val_brf, val_tlf, val_trf]
        3. point coordinate
"""
def trilerp(coord, val, pt):
    if (type(coord) is not list or type(val) is not list or type(pt) is not np.ndarray):
        print("trilerp: either coord or val is not list or pt is not tuple")
        exit()
    if (len(coord) is not 3 or len(val) is not 8 or len(pt) is not 3):
        print("trilerp: either coord or val or pt has incorrect length")
        exit()

    back_lerp = bilerp([coord[0][0], coord[0][1], coord[1][0], coord[1][1]],
                    [val[0], val[1], val[2], val[3]],
                    (pt[0], pt[1]))
    front_lerp = bilerp([coord[0][0], coord[0][1], coord[1][0], coord[1][1]],
                    [val[4], val[5], val[6], val[7]],
                    (pt[0], pt[1]))
    return lerp(coord[2][0], coord[2][1], back_lerp, front_lerp, pt[2])

"""
    Trilinear interpolation with deposit field.

        format: 
        1. coordinates array [
                                [x_left, x_right],
                                [y_bottom, y_top],
                                [z_back, z_front]
                            ]
        2. deposit field
        3. point coordinate
"""
def trilerp_deposit(coord, deposit, pt, bounds):
    # Change coord so ceilings wrap around the box
    bcoord =    [
                    [coord[0][0] % bounds[0], coord[0][1] % bounds[0]],
                    [coord[1][0] % bounds[1], coord[1][1] % bounds[1]],
                    [coord[2][0] % bounds[2], coord[2][1] % bounds[2]]
                ]

    return trilerp(coord, 
        [
            deposit[bcoord[0][0]][bcoord[1][0]][bcoord[2][0]], deposit[bcoord[0][1]][bcoord[1][0]][bcoord[2][0]],
            deposit[bcoord[0][0]][bcoord[1][1]][bcoord[2][0]], deposit[bcoord[0][1]][bcoord[1][1]][bcoord[2][0]],
            deposit[bcoord[0][0]][bcoord[1][0]][bcoord[2][1]], deposit[bcoord[0][1]][bcoord[1][0]][bcoord[2][1]],
            deposit[bcoord[0][0]][bcoord[1][1]][bcoord[2][1]], deposit[bcoord[0][1]][bcoord[1][1]][bcoord[2][1]]         
        ],
        pt)[0]

"""
    Given a vector, return a random perpendicular unit vector
"""
def rand_perpendicular(vec, dim=3):
    rand_perpen = np.cross(vec, np.random.normal(size=dim))
    norm = np.linalg.norm(rand_perpen)
    return rand_perpen / norm

"""
    Given an original vector, axis vector and angle, return rotated vector
"""
def axis_rotate(vec, axis, angle):
    if (type(axis) != np.ndarray):
        axis = np.array(axis)
    if (type(vec) != np.ndarray):
        vec = np.array(vec)#

    # k is axis, v is vec
    # v_rot = v * cos(theta) + k cross v * sin(theta) + k * (k inner v) * (1 - cos(theta))
    return vec * np.cos(angle) +\
            np.cross(axis, vec) * np.sin(angle) +\
            axis * np.inner(axis, vec) * (1 - np.cos(angle))