import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def read_file(path, number_of_points):
    number_to_generate = 1

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=5)
    numpy_pcd = np.asarray(pcd.points)
    numpy_pcd = (numpy_pcd - np.min(numpy_pcd)) / (np.max(numpy_pcd) - np.min(numpy_pcd))
    numpy_pcd = (numpy_pcd * 2) - 1
    return numpy_pcd


def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    #random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    random_p1 = better_random_corner_sensor_imitation()
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :]


def better_random_corner_sensor_imitation():
    step = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]])
    corner = np.array(
        [np.random.choice([1, -1, 2, -2]), np.random.choice([1, -1, 2, -2]), np.random.choice([1, -1, 2, -2])])
    result = step * corner
    return result


def display_open3d(template):
    template_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    template_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([template_])

while True:
    path = "data/Planierraupe.stl"
    pcd = read_file(path, number_of_points=20000)
    pcd = farthest_subsample_points(pcd, num_subsampled_points=5000)
    display_open3d(pcd)
