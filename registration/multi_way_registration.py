"""
This file is a copy of the Multiway registration Tutorial of the open3d docs. It is fitted to the personal needs,
therefore, the purpose is to register fragments instead of the whole scenes. Also it has an added choice between
RANSAC and ICP PtPlane
"""

import open3d as o3d
import numpy as np
from registration_algorithms import execute_RANSAC_for_multiway, icp_ptp_multiway
from util_functions import move_point_cloud_close_to_zero
from synthetic_data_creation import generate_fragments

VOXEL_SIZE = 0.5


def preprocess_point_cloud(pcd):
    """
    Simple preprocessing step that needs to be done for every point cloud
    :param pcd: Raw Point Cloud
    :return: Preprocessed Point Cloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd_down = move_point_cloud_close_to_zero(pcd_down)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd_down


def load_synthetic_pcd():
    """
    Instead of loading real measurements from the virtual camerea, create synthetic dataset instead
    :return: list of fragments
    """
    fragment_list = generate_fragments("rampshere", 7)
    post_process_fragments = []
    for item in fragment_list:
        post_process_fragments.append(preprocess_point_cloud(item))
    return post_process_fragments


def load_point_clouds():
    """
    A function to call the different point clouds you want to register with multiway
    :return: List of open3d Point Clouds
    """
    pcds = []
    for i in range(7):
        path = "dataset/7_cylin_order/cluster_" + str(i) + ".pcd"
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = preprocess_point_cloud(pcd)
        pcds.append(pcd_down)
    return pcds


def pairwise_registration(source, target, mode="RANSAC"):
    """
    Check for pairwise registration of two point clouds
    :param source: Point cloud A in open3d style
    :param target: Point cloud B in open3d style
    :param mode: "RANSAC" or "PtPlane" - the registration method you want to use
    :return:
    """
    if mode == "RANSAC":
        print("Apply RANSAC")
        _, _, _, icp_coarse, _ = execute_RANSAC_for_multiway(
            source, target, voxel_size=VOXEL_SIZE
        )
        icp_fine = icp_ptp_multiway(source, target, icp_coarse.transformation)

    elif mode == "PtPlane":

        print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance_coarse,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        icp_fine = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

    transformation_icp = icp_fine.transformation
    information_icp = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )
    return transformation_icp, information_icp


def full_registration(
    pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine
):
    """
    Main function to diregent the multiway registration. It implements a pose_graph for registering multiple
    fragments of point clouds.
    :param pcds: List of open3D point clouds
    :param max_correspondence_distance_coarse: for ICP PLane the max coarse distance
    :param max_correspondence_distance_fine:  for ICP PLane the max fine distance
    :return: the pose_graph of the registrations
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id]
            )
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


# pcds_down = load_point_clouds()
pcds_down = load_synthetic_pcd()
o3d.visualization.draw_geometries(pcds_down)

print("Full registration ...")
max_correspondence_distance_coarse = VOXEL_SIZE * 15
max_correspondence_distance_fine = VOXEL_SIZE * 1.5
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(
        pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine
    )

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0,
)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

print("Transform points and display")
for point_id in range(len(pcds_down)):
    print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(
    pcds_down,
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024],
)

pcds = load_point_clouds()
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=VOXEL_SIZE)
o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries(
    [pcd_combined_down],
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024],
)
