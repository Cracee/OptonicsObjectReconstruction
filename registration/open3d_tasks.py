import os.path

import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
import time
import sys


from nxlib import NxLib, NxLibItem
from nxlib import Camera
from nxlib.constants import (
    ITM_CAMERAS,
    VAL_STRUCTURED_LIGHT,
    VAL_STEREO,
    ITM_OBJECTS,
    CMD_CAPTURE,
    ITM_TIMEOUT,
)
from nxlib import NxLibCommand
from util_functions import (
    convert_to_open3d_point_cloud,
    read_pcd_file,
    save_pcd_file,
    sort_point_clouds_by_size,
)


def create_virtual_object(
    object_file="data/t_piece.json",
    camera_serial_number="N36-804-16-BL",
    save_file="",
    save_raw=False,
):
    """
    A function that loads the given objects into a virtual environment with the given camera number. In there it will
    get a small Region of Interest Cut, the floor gets eliminated. After that, if specified, the resulting point cloud
    will get saved.

    :param save_raw: True, when saved with floor, False without
    :param object_file: a json file containing the objects in Nx Tree Structure
    :param camera_serial_number: the serial number of the virtual camera
    :param save_file: if empty, there will be no save. Otherwise, the name of the savefile.
    :return:
    """
    try:
        f = open(object_file)
        data = json.load(f)
        jsonString = json.dumps(data)
        print("Object File has been read successful")
    except:
        raise Exception("Sorry, the .json Object file can`t be read the intended way.")

    with NxLib(), Camera.from_serial(
        camera_serial_number, [VAL_STRUCTURED_LIGHT, VAL_STEREO]
    ) as camera:

        # overwrite the Objects to define the scene as you want it to be
        try:
            NxLibItem()[ITM_OBJECTS].set_json(jsonString)
            print("Loading of Objects into virtual scene has worked")
        except:
            raise Exception("Sorry, the Object file doesn`t work the intended way.")

        # capture the scene
        with NxLibCommand(CMD_CAPTURE) as cmd:
            cmd.parameters()[ITM_CAMERAS] = camera_serial_number
            cmd.parameters()[ITM_TIMEOUT] = 5000
            cmd.execute()
        camera.rectify()
        camera.compute_disparity_map()
        camera.compute_point_map()

        # transfer the point cloud to open3D
        point_cloud = convert_to_open3d_point_cloud(camera.get_point_map())

        bounding_box_a = point_cloud.get_axis_aligned_bounding_box()
        print(bounding_box_a)
        list_of_points = bounding_box_a.get_box_points()
        new_list = []
        for point in list_of_points:
            a, b, c = point
            if c > -60:
                c = -65
            new_list.append([a, b, c])

        bounding_polygon = np.asarray(new_list)
        vol = o3d.visualization.SelectionPolygonVolume()

        vol.orthogonal_axis = "Y"
        vol.axis_max = np.max(bounding_polygon[:, 1])
        vol.axis_min = np.min(bounding_polygon[:, 1])

        bounding_polygon[:, 1] = 0
        foo = o3d.utility.Vector3dVector(bounding_polygon)

        vol.bounding_polygon = foo

        cropped_pcd = vol.crop_point_cloud(point_cloud)

        bounding_box_crop = cropped_pcd.get_axis_aligned_bounding_box()
        bounding_box_crop.color = (1, 0, 0)

        bounding_box_crop.color = (0, 1, 0)
        print("The new bounding box is", bounding_box_crop)
        bounding_box_a.color = (1, 0, 0)
        o3d.visualization.draw_geometries(
            [point_cloud, bounding_box_a, bounding_box_crop]
        )

        o3d.visualization.draw_geometries([cropped_pcd, bounding_box_crop])

        if save_file:
            assert save_file.endswith(".pcd"), "The save file must end on .pcd"
            if save_raw:
                new_path = save_file[:-4] + "_raw" + save_file[-4:]
                o3d.io.write_point_cloud(new_path, point_cloud)
                # TODO Change to personal save function

            o3d.io.write_point_cloud(save_file, cropped_pcd)
            # TODO Change to personal save function


def cluster_objects(
    point_cloud, eps=2, min_size=50, print_progress=False, visualize=False
):
    """
    Function, that takes a point_cloud and clusters it into elements via the DBSCAN clustering algorithm. Result will
    be colored by cluster.

    :param point_cloud: open3d point cloud
    :param eps: distances to neighbours (param of DBSCAN)
    :param min_size: minimum size of a cluster (param of DBSCAN)
    :param print_progress: show the progress_bar
    :param visualize: visualize the resulting clusters
    :return: cluster number, point cloud file, labels of the points
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(
            point_cloud.cluster_dbscan(
                eps=eps, min_points=min_size, print_progress=False
            )
        )

    max_label = labels.max()
    if print_progress:
        print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if visualize:
        o3d.visualization.draw_geometries(
            [point_cloud],
            zoom=0.455,
            front=[-0.4999, -0.1659, -0.8499],
            lookat=[2.1813, 2.0619, 2.0999],
            up=[0.1204, -0.9852, 0.1215],
        )
    return max_label + 1, point_cloud, labels


def cluster_trial(point_cloud):
    """
    An analysis run of the DBSCAN clustering algorithm on the point clouds of this project. Tries out different
    parameters (eps and min_sizes) and prints out the time and found clusters.

    :param point_cloud: open3D point cloud for clustering
    :return:
    """
    different_eps = [0.01, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    different_min_sizes = [*range(10, 110, 10)]

    results = []

    for eps in different_eps:
        for size in different_min_sizes:
            start = time.time()
            clusters, _, _ = cluster_objects(
                point_cloud, eps=eps, min_size=size, print_progress=False
            )
            end = time.time()
            elapsed_time = end - start

            results.append(
                f"{str(eps):<8} {str(size):<8} {str(elapsed_time):<8.4} {str(clusters):<8}"
            )

    print("eps   Min Size   time    clusters")
    for item in results:
        print(item)


def split_point_cloud_by_clusters(point_cloud, save_clusters=""):
    """
    Function, that takes a point cloud and seperates the clusters (see cluster_object function) into unique files and
    elements.

    :param point_cloud: string or open3d point cloud - original point cloud
    :param save_clusters: if string is given, then in this folder the clusters will be saved
    :return: list of sorted clusters by size
    """
    if isinstance(point_cloud, str):
        point_cloud = read_pcd_file(point_cloud, visualize=False)

    num_clus, pcd, labels = cluster_objects(
        point_cloud, eps=4.0, min_size=150, visualize=False
    )

    point_clouds_clustered = []

    for i in range(num_clus):
        numpy_index_list = np.asarray(labels == i).nonzero()
        index_list = np.asarray(numpy_index_list)[0]
        point_clouds_clustered.append(point_cloud.select_by_index(index_list))

    clusters_sorted = sort_point_clouds_by_size(point_clouds_clustered)

    if save_clusters:
        print("Saving all the single clusters!")
        current_wd = os.getcwd()
        save_clusters = current_wd + "/" + save_clusters
        print(save_clusters)
        if not os.path.exists(save_clusters):
            os.makedirs(save_clusters)
            print("A new directory has been created!")
        counter = 0
        for item in clusters_sorted:
            save_string = save_clusters + "/cluster_" + str(counter) + ".pcd"
            save_pcd_file(item, save_string)
            counter += 1

    return clusters_sorted


def create_mesh_from_point_cloud(point_cloud, visualize=True, mode="alpha"):
    # bunny = o3d.data.BunnyMesh()
    # meshi = o3d.io.read_triangle_mesh(bunny.path)

    if mode == "poisson":
        print("run Poisson surface reconstruction")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=6
            )
        print(mesh)
    elif mode == "alpha":
        alpha = 0.8
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            point_cloud, alpha
        )

    elif mode == "ball_pivot":
        radii = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud, o3d.utility.DoubleVector(radii)
        )
        mesh.compute_vertex_normals()

    if visualize:
        o3d.visualization.draw_geometries([mesh])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def outlier_points_removal(point_cloud, mode="statistical"):
    if mode == "statistical":
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    elif mode == "radius":
        cl, ind = point_cloud.remove_radius_outlier(nb_points=20, radius=1.2)
    display_inlier_outlier(point_cloud, ind)
    return point_cloud.select_by_index(ind)


def create_convex_hull(point_cloud, visualize=False):
    hull, _ = point_cloud.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    if visualize:
        o3d.visualization.draw_geometries([point_cloud, hull_ls])
    hull.paint_uniform_color([0, 0.706, 0.805])
    return hull
