#!/usr/bin/env python

"""create_virtual_object.py: The starting file to obtain a
3D Object point cloud for testing registration."""

__author__ = "Gregor Stief"
__copyright__ = "Copyright 2022, Gregor Stief and Optonics"
__email__ = "Gregormax.Stief@web.de"
__status__ = "Work in Progress"

import numpy as np
import open3d as o3d
import json


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

f = open("data/t_piece.json")
data = json.load(f)
jsonString = json.dumps(data)

serial_number = "N36-804-16-BL"

with NxLib(), Camera.from_serial(
    serial_number, [VAL_STRUCTURED_LIGHT, VAL_STEREO]
) as camera:

    # overwrite the Objects to define the scene as you want it to be
    NxLibItem()[ITM_OBJECTS].set_json(jsonString)
    print("Loading of Objects has worked")
    # print(NxLibItem().as_json())

    # capture the scene
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS] = serial_number
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
    o3d.visualization.draw_geometries([point_cloud, bounding_box_a, bounding_box_crop])

    o3d.visualization.draw_geometries([cropped_pcd, bounding_box_crop])

    # o3d.io.write_point_cloud("data/t_piece.pcd", cropped_pcd)
