#!/usr/bin/env python

"""open3d_tasks.py: The starting file to obtain a
3D Object point cloud for testing registration."""

__author__ = "Gregor Stief"
__copyright__ = "Copyright 2022, Gregor Stief and Optonics"
__email__ = "Gregormax.Stief@web.de"
__status__ = "Work in Progress"


from open3d_tasks import create_virtual_object, cluster_objects
from util_functions import read_pcd_file

# create_virtual_object(save_file="data/t_piece_raw.pcd")
point_cloud = read_pcd_file("data/t_piece_raw.pcd", visualize=False)
cluster_objects(point_cloud)
