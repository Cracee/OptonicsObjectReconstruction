#!/usr/bin/env python

"""create_virtual_object.py: The starting file to obtain a
3D Objetct point cloud for testing registration."""

__author__ = "Gregor Stief"
__copyright__ = "Copyright 2022, Gregor Stief and Optonics"
__email__ = "Gregormax.Stief@web.de"
__status__ = "Work in Progress"

import open3d
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

f = open("data/box_and_t_piece.json")
data = json.load(f)
jsonString = json.dumps(data)

serial_number = "N36-804-16-BL"

with NxLib(), Camera.from_serial(
    serial_number, [VAL_STRUCTURED_LIGHT, VAL_STEREO]
) as camera:

    NxLibItem()[ITM_OBJECTS].set_json(jsonString)
    print("Loading of Objects has worked")
    # print(NxLibItem().as_json())

    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS] = serial_number
        cmd.parameters()[ITM_TIMEOUT] = 5000
        cmd.execute()
    camera.rectify()
    camera.compute_disparity_map()
    camera.compute_point_map()

    # Watch the captured point cloud with open3d
    point_cloud = convert_to_open3d_point_cloud(camera.get_point_map())
    open3d.visualization.draw_geometries([point_cloud])
