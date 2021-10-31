import numpy as np


def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances


def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def compute_vertex_normals(vertices, faces):
    # Vertex normals weighted by triangle areas:
    # http://www.iquilezles.org/www/articles/normals/normals.htm

    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    triangles = vertices[faces]

    e1 = triangles[::, 0] - triangles[::, 1]
    e2 = triangles[::, 2] - triangles[::, 1]
    n = np.cross(e2, e1) 

    np.add.at(normals, faces[:,0], n)
    np.add.at(normals, faces[:,1], n)
    np.add.at(normals, faces[:,2], n)

    return normalize(normals)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

