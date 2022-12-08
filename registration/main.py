#!/usr/bin/env python

"""open3d_tasks.py: The starting file to obtain a
3D Object point cloud for testing registration."""

__author__ = "Gregor Stief"
__copyright__ = "Copyright 2022, Gregor Stief and Optonics"
__email__ = "Gregormax.Stief@web.de"
__status__ = "Work in Progress"


from open3d_tasks import create_virtual_object, cluster_objects
from util_functions import read_pcd_file
import time

# create_virtual_object(save_file="data/t_piece_raw.pcd")
point_cloud = read_pcd_file("data/box_3_piece.pcd", visualize=False)

different_eps = [0.01, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
different_min_sizes = [*range(10, 110, 10)]

results = []

for eps in different_eps:
    for size in different_min_sizes:
        start = time.time()
        clusters = cluster_objects(
            point_cloud, eps=eps, min_size=size, print_progress=False
        )
        end = time.time()
        elapsed_time = end - start

        results.append(
            f"{str(eps):<8} {str(size):<8} {str(elapsed_time):<8.4} {str(clusters):<8}"
        )

print("eps   Min Size   time    clusters")
for item in results:
    print(item)
