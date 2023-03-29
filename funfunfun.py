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
    return numpy_pcd, pcd


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


def generate_stanford_data():
    characs = ["a", "b", "c", "d", "e", "f", "g", "h"]
    path = "data/stanford/xyzrgb_statuette.ply"
    numpy_pcd1, pcd1 = read_file(path, number_of_points=1024)
    numpy_pcd2, pcd2 = read_file(path, number_of_points=1024)
    for c in characs:
        numpy_pcd_a = farthest_subsample_points(numpy_pcd1, num_subsampled_points=512)
        name = path[:-4] + "_" + c + "1.pcd"
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(numpy_pcd_a)
        o3d.io.write_point_cloud(name, pcd_a)
        #display_open3d(numpy_pcd_a)
        # ---------------------------
        numpy_pcd_b = farthest_subsample_points(numpy_pcd2, num_subsampled_points=512)
        name = path[:-4] + "_" + c + "2.pcd"
        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(numpy_pcd_b)
        o3d.io.write_point_cloud(name, pcd_b)
        #display_open3d(numpy_pcd_b)


def segement_plane(pcd):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.8, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud, outlier_cloud


path = "registration/data/14_rampsheres_clustered.pcd"


data = o3d.io.read_point_cloud(path)

o3d.visualization.draw_geometries([data])
