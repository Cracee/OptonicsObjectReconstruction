from nxlib import NxLib
from nxlib import NxLibItem
from nxlib import NxLibError
from nxlib.constants import (
    ITM_VERSION,
    ITM_NX_LIB,
    ITM_MAJOR,
    ITM_MINOR,
    ITM_BUILD,
)

with NxLib():
    major = NxLibItem()[ITM_VERSION][ITM_NX_LIB][ITM_MAJOR]
    minor = NxLibItem()[ITM_VERSION][ITM_NX_LIB][ITM_MINOR]
    build = NxLibItem()[ITM_VERSION][ITM_NX_LIB][ITM_BUILD]
    try:
        version = major.as_string() + minor.as_string() + build.as_string()
    except NxLibError:
        # "The 'as_*()' methods cannot be used to convert an item's value."
        pass
    if major.is_int() and minor.is_int() and build.is_int():
        versio = ".".join(str(node.as_int()) for node in [major, minor, build])
        if major >= 4 or (major == 3 and minor > 3):
            print("Ensenso SDK installation contains Python interface")
        print(f"NxLib Version {versio}")
    else:
        print("Unexpected node format detected")
