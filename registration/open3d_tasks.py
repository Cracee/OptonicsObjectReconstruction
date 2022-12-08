import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt

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
from util_functions import convert_to_open3d_point_cloud


def create_virtual_object(
    object_file="data/t_piece.json", camera_serial_number="N36-804-16-BL", save_file=""
):
    """
    A function that loads the given objects into a virtual environment with the given camera number. In there it will
    get a small Region of Interest Cut, the floor gets eliminated. After that, if specified, the resulting point cloud
    will get saved.

    :param object_file: a json file containing the objects in Nx Tree Structure
    :param camera_serial_number: the serial number of the virtual camera
    :param save_file: if empty, there will be no save. Otherwise, the name of the savefile.
    :return:
    """
    try:
        f = open(object_file)
        data = json.load(f)
        jsonString = json.dumps(data)
        print("Object File has bean read successful")
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
            o3d.io.write_point_cloud(save_file, cropped_pcd)


def cluster_objects(
    point_cloud, eps=2, min_size=50, print_progress=False, visualize=False
):
    """
    Function, that takes a point_cloud and clusters it into elements.
    :param point_cloud:
    :return:
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
    return max_label + 1
