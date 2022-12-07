#! /usr/bin/env python3
# -*- coding: utf8 -*-

from nxlib import NxLibItem
from nxlib.constants import (
    ITM_CAMERAS,
    ITM_STATUS,
    ITM_VALID_IP_ADDRESS,
    ITM_OPEN,
    ITM_AVAILABLE,
    ITM_VALID_CAMERA_FIRMWARE,
    ITM_CALIBRATED,
    ITM_VALID_PROJECTOR_FIRMWARE,
    ITM_MODEL_NAME,
)
from nxlib.context import NxLib


def check_true(item):
    return item.exists() and item.as_bool() is True


def check_false(item):
    return item.exists() and item.as_bool() is False


with NxLib():
    # Reference to the serials subnode of all cameras
    cameras = NxLibItem()[ITM_CAMERAS]

    # Print status information for each camera
    print("SerialNo", " " * 8, "Model", " " * 10, "Status")
    for i in range(cameras.count()):
        if not cameras[i][ITM_STATUS].exists():
            continue
        if check_false(cameras[i][ITM_STATUS][ITM_VALID_IP_ADDRESS]):
            status = "Invalid Ip"
        elif check_true(cameras[i][ITM_STATUS][ITM_OPEN]):
            status = "Open"
        elif check_false(cameras[i][ITM_STATUS][ITM_AVAILABLE]):
            status = "In Use"
        elif check_false(cameras[i][ITM_STATUS][ITM_VALID_CAMERA_FIRMWARE]):
            status = "Invalid Camera Firmware"
        elif check_false(cameras[i][ITM_STATUS][ITM_VALID_PROJECTOR_FIRMWARE]):
            status = "Invalid Projector Firmware"
        elif check_false(cameras[i][ITM_STATUS][ITM_CALIBRATED]):
            status = "Not Calibrated"
        else:
            status = "Available"
        serial = cameras[i].name()
        model = cameras[i][ITM_MODEL_NAME].as_string()
        print(f"{serial:<17} {model:<16} {status:<16}")
