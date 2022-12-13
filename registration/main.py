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
from util_functions import read_pcd_file, save_pcd_file, read_stl_file_to_point_cloud
from fast_global_registration import (
    start_fast_global_registration,
    start_transformation_pipeline,
    execute_global_registration,
)
import pyransac3d as pyrsc

"""
pcd = read_pcd_file("data/14_rampsheres_raw.pcd", visualize=True)
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.5, ransac_n=3, num_iterations=1000
)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
o3d.visualization.draw_geometries([outlier_cloud])

plane_model, inliers = outlier_cloud.segment_plane(
    distance_threshold=0.5, ransac_n=3, num_iterations=1000
)
inlier_cloud = outlier_cloud.select_by_index(inliers)
outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
o3d.visualization.draw_geometries([outlier_cloud])

sorted_clusters = split_point_cloud_by_clusters(outlier_cloud)


o3d.visualization.draw_geometries(sorted_clusters)
"""

# At the moment this algorithm starts a working RANSAC, but maybe it will not be a perfect alignment

pcd_a = read_pcd_file("data/14_ramp_order/cluster_13.pcd", visualize=False)
pcd_b = read_pcd_file("data/14_ramp_order/cluster_11.pcd", visualize=False)

reso = execute_global_registration(pcd_a, pcd_b)

save_pcd_file(reso, "data/frankenstein/2_cluster_comb_11_u_13.pcd")
# o3d.visualization.draw_geometries([reso])
