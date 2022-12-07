import numpy as np

from nxlib import NxLib, Camera, NxLibItem
from nxlib.constants import (
    ITM_CAMERAS,
    ITM_OBJECTS,
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


def compute_z_value_average(point_map):
    """Reshape and filter the point map, then calculate z-value average."""
    point_map = reshape_point_map(point_map)
    point_map = filter_nans(point_map)

    if point_map.shape[0] == 0:
        return 0.0

    z_idx = 2
    z_count, z_sum = 0, 0.0
    for i in range(point_map.shape[0]):
        z_sum += point_map[i][z_idx]
        z_count += 1

    return z_sum / z_count


# parser = argparse.ArgumentParser()
# parser.add_argument("serial", type=str,
#                    help="the serial of the stereo camera to open")
# args = parser.parse_args()

serial_number = "N35-801-16-BL"
# serial_number = "FileN36-804-16-BL"

print(NxLibItem()[ITM_OBJECTS].as_json())

with NxLib(), Camera.from_serial(
    serial_number, [VAL_STRUCTURED_LIGHT, VAL_STEREO]
) as camera:
    # Capture with the previously opened camera
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS] = serial_number
        cmd.parameters()[ITM_TIMEOUT] = 5000
        cmd.execute()
    camera.rectify()
    camera.compute_disparity_map()
    camera.compute_point_map()

    z_value_average = compute_z_value_average(camera.get_point_map())
    print(f"The average z value in the point map is {z_value_average:.1f}mm")
