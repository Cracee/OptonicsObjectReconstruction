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


# pcd = read_pcd_file("data/multiway_registration/multiway_registration.pcd", visualize=True)

# create_virtual_object(object_file="data/7_cylin.json", save_file="data/7_cylin.pcd", save_raw=True)

split_point_cloud_by_clusters("data/7_cylin.pcd", save_clusters="data/7_cylin_order")
