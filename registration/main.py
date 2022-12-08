#!/usr/bin/env python

"""open3d_tasks.py: The starting file to obtain a
3D Object point cloud for testing registration."""

__author__ = "Gregor Stief"
__copyright__ = "Copyright 2022, Gregor Stief and Optonics"
__email__ = "Gregormax.Stief@web.de"
__status__ = "Work in Progress"

import open3d as o3d
import copy
import numpy
from open3d_tasks import (
    create_virtual_object,
    cluster_objects,
    cluster_trial,
    split_point_cloud_by_clusters,
)
from util_functions import read_pcd_file, save_pcd_file
from fast_global_registration import start_fast_global_registration

# create_virtual_object(object_file="data/14_rampspheres.json", save_file="data/14_rampsheres.pcd", save_raw=True)
# point_cloud = read_pcd_file("data/14_rampsheres_clustered.pcd", visualize=False)
# sorted_clusters = split_point_cloud_by_clusters("data/14_rampsheres_clustered.pcd")

start_fast_global_registration()
