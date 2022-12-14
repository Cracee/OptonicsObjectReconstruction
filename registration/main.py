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
    create_mesh_from_point_cloud,
    outlier_points_removal,
    create_convex_hull,
)
from util_functions import (
    read_pcd_file,
    save_pcd_file,
    read_stl_file_to_point_cloud,
    move_point_cloud_close_to_zero,
    calculate_dist,
    move_points_by,
)
from registration_algorithms import (
    start_fast_global_registration,
    start_transformation_pipeline,
    execute_global_registration,
    start_icp_ptp,
    draw_registration_result,
)

# At the moment this algorithm starts a working RANSAC, but maybe it will not be a perfect alignment

pcd_a = read_pcd_file("data/7_cylin_order/cluster_0.pcd", visualize=True)
pcd_b = read_pcd_file("data/7_cylin_order/cluster_3.pcd", visualize=True)

go_on = True

if go_on:
    reso = execute_global_registration(pcd_a, pcd_b)

    pcd, source_down, target_down, _, init_trans = reso

    resulto, final_trans = start_icp_ptp(source_down, target_down, init_trans)

    """
    reso = execute_global_registration(pcd_a, pcd_b)

    pcd, source_down, target_down, _, init_trans = reso

    resulto = start_icp_ptp(source_down, target_down, init_trans)
    """
    resulting_pcd = draw_registration_result(pcd_a, pcd_b, final_trans)
    save_pcd_file(resulting_pcd, "data/frankenstein_cylin/cluster_comb_0_3" "" ".pcd")
