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
from util_functions import read_pcd_file, save_pcd_file, read_stl_file_to_point_cloud
from fast_global_registration import (
    start_fast_global_registration,
    start_transformation_pipeline,
)

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

pcd = read_pcd_file("data/frankenstein/cluster_comb_9u11U13_1.pcd", visualize=True)
