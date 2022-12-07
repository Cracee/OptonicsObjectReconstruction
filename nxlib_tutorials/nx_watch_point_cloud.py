# This example requires the open3d package to be installed.
# Install it with: python3 -m pip install open3d

import numpy as np
import open3d

from nxlib import NxLib
from nxlib import Camera
from nxlib.constants import (
    ITM_CAMERAS,
    VAL_STRUCTURED_LIGHT,
    VAL_STEREO,
    CMD_CAPTURE,
    ITM_TIMEOUT,
)
from nxlib import NxLibCommand


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
    open3d_point_cloud = open3d.geometry.PointCloud()
    open3d_point_cloud.points = open3d.utility.Vector3dVector(point_map)
    return open3d_point_cloud


"""
parser = argparse.ArgumentParser()
parser.add_argument("serial", type=str,
                    help="the serial of the depth camera to open")
args = parser.parse_args()
"""

serial_number = "N36-804-16-BL"


with NxLib(), Camera.from_serial(
    serial_number, [VAL_STRUCTURED_LIGHT, VAL_STEREO]
) as camera:
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS] = serial_number
        cmd.parameters()[ITM_TIMEOUT] = 5000
        cmd.execute()
    camera.compute_disparity_map()
    camera.compute_point_map()
    # Watch the captured point cloud with open3d
    point_cloud = convert_to_open3d_point_cloud(camera.get_point_map())
    open3d.visualization.draw_geometries([point_cloud])
