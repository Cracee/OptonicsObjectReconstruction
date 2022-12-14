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
import pyransac3d as pyrsc

# create_virtual_object(object_file="data/7_RAMPSHERES.json", save_file="data/7_RAMPSHERES.pcd", save_raw=True)

# split_point_cloud_by_clusters("data/7_RAMPSHERES.pcd", save_clusters="data/7_RAMP_order")

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

pcd_a = read_pcd_file("data/7_RAMP_order/cluster_5.pcd", visualize=False)
pcd_b = read_pcd_file("data/7_RAMP_order/cluster_0.pcd", visualize=False)

pcd_a = move_point_cloud_close_to_zero(pcd_a)
pcd_b = move_point_cloud_close_to_zero(pcd_b)

a_box = pcd_a.get_oriented_bounding_box()
a_box.color = (1, 0, 0)
a_x, a_y, a_z = a_box.center
b_box = pcd_b.get_oriented_bounding_box()
b_box.color = (0, 1, 1)
b_x, b_y, b_z = b_box.center
o3d.visualization.draw_geometries([a_box, b_box])
mov_coord = calculate_dist((a_x, a_y, a_z), (b_x, b_y, b_z))
pcd_a = move_points_by(pcd_a, mov_coord)
a_box = pcd_a.get_oriented_bounding_box()
a_box.color = (1, 0, 0)
o3d.visualization.draw_geometries([a_box, b_box])
o3d.visualization.draw_geometries([pcd_a, pcd_b])
print(a_box.volume())
print(b_box.volume())

exit()
go_on = False

new_cloud = pcd_a
inliner_clouds = []
for i in range(4):
    plane_model, inliers = new_cloud.segment_plane(
        distance_threshold=0.4, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = new_cloud.select_by_index(inliers)
    outlier_cloud = new_cloud.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    inliner_clouds.append(inlier_cloud)
    new_cloud = outlier_cloud

number = int(input("Which plane are we choosing? >>> "))

plane = inliner_clouds[number]

new_cloud = pcd_b
inliner_clouds = []
for i in range(3):
    plane_model, inliers = new_cloud.segment_plane(
        distance_threshold=0.4, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = new_cloud.select_by_index(inliers)
    outlier_cloud = new_cloud.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    inliner_clouds.append(inlier_cloud)
    new_cloud = outlier_cloud

number = int(input("Which plane are we choosing? >>> "))

other_plane = inliner_clouds[number]
# R = pcd_b.get_rotation_matrix_from_xyz((0, numpy.pi, numpy.pi))
# pcd_b.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([pcd_b])
if go_on:
    reso = execute_global_registration(plane, other_plane)

    pcd, source_down, target_down, _, init_trans = reso

    resulto, final_trans = start_icp_ptp(source_down, target_down, init_trans)

    """
    reso = execute_global_registration(pcd_a, pcd_b)

    pcd, source_down, target_down, _, init_trans = reso

    resulto = start_icp_ptp(source_down, target_down, init_trans)
    """
    resulting_pcd = draw_registration_result(pcd_a, pcd_b, final_trans)
    save_pcd_file(resulting_pcd, "data/reg_by_plane/cluster_comb_0_5.pcd")
# o3d.visualization.draw_geometries([reso])
