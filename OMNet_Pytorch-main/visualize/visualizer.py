import torch
import open3d as o3d
from common import se3
import numpy as np


def translate_np_to_pcd(point_cloud):
    """
    Simple function to translate a numpy point cloud into an open3d one
    Args:
        point_cloud: point cloud in numpy format and dimensions: [n, 3]

    Returns: an open3d pcd format

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


def generate_pointcloud(path, number_of_points, resampled=False, resample_with_bigger_point_clouds=False):
    """
    A function to generate a point cloud in numpy format from a open3d pcd file. There is the option to have a second
    point cloud sampled from the same source (to have slightly different points)
    Args:
        path: string, the path to the pcd file
        number_of_points: int, the number of points, that should be sampled
        resampled: bool, should there be a second point cloud with different points
        resample_with_bigger_point_clouds: bool, a greater technique to have even more different point clouds

    Returns: one or two point clouds in numpy [number_of_points, 3] format

    """
    numper_to_generate = 1
    if resampled:
        numper_to_generate += 1
    pcds = []
    for i in range(numper_to_generate):
        if resample_with_bigger_point_clouds:
            index_list = np.arange(0, number_of_points)
            shift = np.random.randint(int(number_of_points*0.25))
        else:
            shift = 0
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points + shift, init_factor=5)
        numpy_pcd = np.asarray(pcd.points)
        if resample_with_bigger_point_clouds:
            numpy_pcd = numpy_pcd[index_list]
        numpy_pcd = (numpy_pcd - np.min(numpy_pcd)) / (np.max(numpy_pcd) - np.min(numpy_pcd))
        numpy_pcd = (numpy_pcd * 2) - 1
        pcds.append(numpy_pcd)
    if len(pcds) == 1:
        pcds = pcds[0]
    return pcds


def generate_pointcloud_ply(path, number_of_points, resampled=False, resample_with_bigger_point_clouds=False):
    """
    A function to generate a point cloud in numpy format from an open3d pcd file. There is the option to have a second
    point cloud sampled from the same source (to have slightly different points)
    Args:
        path: string, the path to the pcd file
        number_of_points: int, the number of points, that should be sampled
        resampled: bool, should there be a second point cloud with different points
        resample_with_bigger_point_clouds: bool, a greater technique to have even more different point clouds

    Returns: one or two point clouds in numpy [number_of_points, 3] format

    """
    number_to_generate = 1
    if resampled:
        number_to_generate += 1
    pcds = []
    for i in range(number_to_generate):
        if resample_with_bigger_point_clouds:
            index_list = np.arange(0, number_of_points)
            shift = np.random.randint(int(number_of_points*0.25))
        else:
            shift = 0
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points + shift, init_factor=5)
        numpy_pcd = np.asarray(pcd.points)
        if resample_with_bigger_point_clouds:
            numpy_pcd = numpy_pcd[index_list]
        numpy_pcd = (numpy_pcd - np.min(numpy_pcd)) / (np.max(numpy_pcd) - np.min(numpy_pcd))
        numpy_pcd = (numpy_pcd * 2) - 1
        pcds.append(numpy_pcd)
    if len(pcds) == 1:
        pcds = pcds[0]
    return pcds

def visualize_result(net_output, data_batch):
    """
    A simple function to visualize the output of OMNet with open3d.

    Args:
        net_output: dict, the output of the forward pass of OMNet
        data_batch: dict, the data with ground truths

    """
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
        point_trans.paint_uniform_color([1.0, 0.0, 0.0])
        point_source = translate_np_to_pcd(item_point_source)
        point_source.paint_uniform_color([1.0, 0.0, 0.0])
        point_ref = translate_np_to_pcd(item_ref)
        point_ref.paint_uniform_color([0.0, 1.0, 0.0])

        #o3d.visualization.draw_geometries([point_ref, point_source, axes])
        #o3d.visualization.draw_geometries([point_ref, point_source])
        #o3d.visualization.draw_geometries([point_ref, point_trans, axes])
        o3d.visualization.draw_geometries([point_ref, point_trans, point_source])
        break

