import numpy
import numpy as np
import open3d as o3d
from os.path import exists


def filter_nans(point_map):
    """Filter NaN values."""
    return point_map[~np.isnan(point_map).any(axis=1)]


def reshape_point_map(point_map):
    """Reshape the point map array from (m x n x 3) to ((m*n) x 3)."""
    return point_map.reshape(
        (point_map.shape[0] * point_map.shape[1]), point_map.shape[2]
    )


def convert_to_open3d_point_cloud(point_map):
    """Convert numpy array to Open3D format."""
    point_map = reshape_point_map(point_map)
    point_map = filter_nans(point_map)
    open3d_point_cloud = o3d.geometry.PointCloud()
    open3d_point_cloud.points = o3d.utility.Vector3dVector(point_map)
    return open3d_point_cloud


def read_pcd_file(pcd_file="data/t_piece.pcd", visualize=True):
    """
    Simple read function, to open a point cloud from a file, with the option to directly visualize it.

    :param pcd_file: string - the file to open
    :param visualize: bool - if you want to directly visualize it
    :return: the opened file as open3d point cloud
    """

    if not exists(pcd_file) or not pcd_file.endswith("pcd"):
        raise NameError("The file name is not correct!")

    point_cloud = o3d.io.read_point_cloud(pcd_file)
    if visualize:
        o3d.visualization.draw_geometries([point_cloud])
    return point_cloud


def save_pcd_file(pcd_file, save_path):
    """
    Simple saving function for point clouds for easier writing.

    :param pcd_file: An open3D point cloud
    :param save_path: string - path to save on
    :return:
    """
    if not save_path.endswith(".pcd"):
        raise NameError("The save file must end on .pcd")
    o3d.io.write_point_cloud(save_path, pcd_file)


def sort_point_clouds_by_size(list_of_point_clouds):
    """
    Function, that gets an already separated list of point clouds and sorts them by their respective point volume.

    :param list_of_point_clouds: list - of open3d point clouds
    :return: a sorted list of point clouds, biggest element at index 0
    """
    all_lengths = []
    for item in list_of_point_clouds:
        all_lengths.append(len(item.points))
    all_lengths = numpy.array(all_lengths)
    sort_index = numpy.argsort(all_lengths)
    result = [list_of_point_clouds[i] for i in sort_index]
    result.reverse()
    return result
