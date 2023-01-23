import numpy as np
import open3d as o3d


def generate_pointcloud(path, number_of_points, resampled=False, resample_with_bigger_point_clouds=False):
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