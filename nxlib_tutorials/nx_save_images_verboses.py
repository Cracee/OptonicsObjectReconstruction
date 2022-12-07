import nxlib.api as api
from nxlib import NxLibCommand, NxLibException, NxLibItem
from nxlib.constants import (
    ITM_CAMERAS,
    ITM_BY_SERIAL_NO,
    CMD_CAPTURE,
    ITM_TIMEOUT,
    CMD_RECTIFY_IMAGES,
    VAL_STEREO,
    ITM_NODE,
    ITM_IMAGES,
    ITM_RAW,
    ITM_LEFT,
    ITM_FILENAME,
    CMD_OPEN,
    CMD_SAVE_IMAGE,
    ITM_TYPE,
    ITM_RECTIFIED,
    ITM_RIGHT,
    CMD_CLOSE,
)


def get_camera_node(serial):
    # Get the root of the tree.
    root = NxLibItem()
    # From here on we can use the [] operator to walk the tree.
    cameras = root[ITM_CAMERAS][ITM_BY_SERIAL_NO]
    for i in range(cameras.count()):
        found = cameras[i].name() == serial
        if found:
            return cameras[i]


# parser = argparse.ArgumentParser()
# parser.add_argument("serial", type=str,
#                    help="the serial of the stereo camera to open")
# args = parser.parse_args()

camera_serial = "N36-804-16-BL"

try:
    # Wait for the cameras to be initialized
    api.initialize()

    tree = NxLibItem()

    # Open the camera with the given serial
    with NxLibCommand(CMD_OPEN) as cmd:
        cmd.parameters()[ITM_CAMERAS] = camera_serial
        cmd.execute()

    # Capture with the previously opened camera
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS] = camera_serial
        cmd.parameters()[ITM_TIMEOUT] = 5000
        cmd.execute()

    # Rectify the images
    with NxLibCommand(CMD_RECTIFY_IMAGES) as cmd:
        cmd.execute()

    # Get the NxLib node of the open camera
    camera = get_camera_node(camera_serial)

    # Save the raw and rectified images
    with NxLibCommand(CMD_SAVE_IMAGE) as cmd:
        if camera[ITM_TYPE].as_string() == VAL_STEREO:
            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RAW][ITM_LEFT].path
            cmd.parameters()[ITM_FILENAME] = "raw_left.png"
            cmd.execute()

            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RAW][ITM_RIGHT].path
            cmd.parameters()[ITM_FILENAME] = "raw_right.png"
            cmd.execute()

            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RECTIFIED][
                ITM_LEFT
            ].path
            cmd.parameters()[ITM_FILENAME] = "rectified_left.png"
            cmd.execute()

            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RECTIFIED][
                ITM_RIGHT
            ].path
            cmd.parameters()[ITM_FILENAME] = "rectified_right.png"
            cmd.execute()
        else:
            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RAW].path
            cmd.parameters()[ITM_FILENAME] = "raw.png"
            cmd.execute()

            cmd.parameters()[ITM_NODE] = camera[ITM_IMAGES][ITM_RECTIFIED].path
            cmd.parameters()[ITM_FILENAME] = "rectified.png"
            cmd.execute()

    # Close the open camera
    with NxLibCommand(CMD_CLOSE) as cmd:
        cmd.execute()

except NxLibException as e:
    print(f"An NxLib error occured: Error Text: {e.get_error_text()}")
except Exception:
    print("An NxLib unrelated error occured:\n")
    raise
