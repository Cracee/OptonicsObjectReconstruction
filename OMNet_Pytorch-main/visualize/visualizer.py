import torch
import open3d as o3d
from common import se3
import numpy as np


def translate_np_to_pcd(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


def generate_pointcloud(path, number_of_points):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=5)
    numpy_pcd = np.asarray(pcd.points)
    numpy_pcd = (numpy_pcd - np.min(numpy_pcd)) / (np.max(numpy_pcd) - np.min(numpy_pcd))
    numpy_pcd = (numpy_pcd * 2) - 1
    return numpy_pcd


def visualize_result(net_output, data_batch):

    points_src = data_batch["points_src"].cpu().numpy()
    points_ref = data_batch["points_ref"].cpu().numpy()
    transformation = net_output["transform_pair"][1].cpu().numpy()

    points_transformed = se3.np_transform(transformation, points_src)

    for i in range(points_transformed.shape[0]):
        item = points_transformed[i]
        item2 = points_src[i]
        ref = points_ref[i]
        point = translate_np_to_pcd(item)
        point.paint_uniform_color([0.0, 1.0, 0.0])
        point2 = translate_np_to_pcd(item2)
        point2.paint_uniform_color([1.0, 0.0, 0.0])
        ref_point = translate_np_to_pcd(ref)
        ref_point.paint_uniform_color([0.0, 0.0, 1.0])

        o3d.visualization.draw_geometries([point, point2])
        o3d.visualization.draw_geometries([point, ref_point])

