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
)
import pyransac3d as pyrsc

# create_virtual_object(object_file="data/14_rampspheres.json", save_file="data/14_rampsheres.pcd", save_raw=True)
# point_cloud = read_pcd_file("data/14_rampsheres_clustered.pcd", visualize=False)
# sorted_clusters = split_point_cloud_by_clusters("data/14_rampsheres_clustered.pcd", save_clusters="data/14_ramp_order")

# start_fast_global_registration()
#
# point_cloud = read_stl_file_to_point_cloud(
#    "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/Ramp_sphere.stl"
# )

"""
tester = True
while tester:
    result = start_transformation_pipeline()

    question = input("Do you want to save that point_cloud? (yes: y / no: n) >>> ")

    if question == "y":
        save_path = input("how to call the file? >>>")
        save_pcd_file(result, save_path)
        tester = False
    if question == "quit":
        tester = False



for i in range(15):
    save_path = "data/14_ramp_order/cluster_" + str(i) + ".pcd"
    clusterino = read_pcd_file(save_path, visualize=True)
    print("The length of cluster ", str(i), " is ", str(len(clusterino.points)))

print("We are searching for these lengths:")
clusterino = read_pcd_file("data/14_ramsphere/cluster_1.pcd", visualize=False)
print("The length of searched cluster 1 is ", str(len(clusterino.points)))
clusterino = read_pcd_file("data/14_ramsphere/cluster_3.pcd", visualize=False)
print("The length of searched cluster 3 is ", str(len(clusterino.points)))
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


# sorted_clusters.append(inlier_cloud)
o3d.visualization.draw_geometries(sorted_clusters)

# new_pcd = outlier_points_removal(pcd)
# mesh = create_convex_hull(new_pcd)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])

"""
all_inlier_clouds = []
all_hulls = []
counter = 0
while counter < 5:

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    all_inlier_clouds.append(inlier_cloud)
    all_inlier_clouds.append(outlier_cloud)
    # o3d.visualization.draw_geometries(all_inlier_clouds)
    new_hull = create_convex_hull(inlier_cloud)
    new_hull.compute_vertex_normals()
    all_hulls.append(new_hull)
    all_inlier_clouds = all_inlier_clouds[:-1]
    pcd = outlier_cloud
    counter += 1
# create_mesh_from_point_cloud(new_pcd)
o3d.visualization.draw_geometries(all_inlier_clouds)
o3d.visualization.draw_geometries(all_hulls)
"""
