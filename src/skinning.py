import numpy as np


def lbs(vertices, joint_transforms, skinning_weights):
    T = np.tensordot(skinning_weights, joint_transforms, axes=[[1], [0]])
    vertices_homogeneous = np.hstack((vertices, np.ones([vertices.shape[0], 1])))
    vertices_posed_homogeneous = np.matmul(T, vertices_homogeneous.reshape([-1, 4, 1])).reshape([-1, 4])
    vertices_posed = vertices_posed_homogeneous[:, :3]

    return vertices_posed
