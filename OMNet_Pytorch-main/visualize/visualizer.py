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
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    print(points_src)
    print(points_ref)

    points_transformed = se3.np_transform(transformation, points_src)

    for i in range(points_transformed.shape[0]):
        item_point_trans = points_transformed[i]
        item_point_source = points_src[i]
        item_ref = points_ref[i]
        point_trans = translate_np_to_pcd(item_point_trans)
        point_trans.paint_uniform_color([0.0, 1.0, 0.0])
        point_source = translate_np_to_pcd(item_point_source)
        point_source.paint_uniform_color([1.0, 0.0, 0.0])
        point_ref = translate_np_to_pcd(item_ref)
        point_ref.paint_uniform_color([0.0, 0.0, 1.0])

        o3d.visualization.draw_geometries([point_ref, point_source, axes])
        o3d.visualization.draw_geometries([point_ref, point_trans, axes])

