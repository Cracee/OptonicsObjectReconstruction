"""
Most of this File are functions from the open3d tutorials towards Registration. Some of them are altered and
a few are of my own origin. Still, most credits go out to open3d.
"""

import open3d as o3d
import copy
import numpy

from util_functions import read_pcd_file, save_pcd_file


def draw_registration_result(source, target, transformation, visualize=True):
    """
    Draw the registration result, visualizing the two point clouds in different colours

    :param source: The point cloud you wanted to transform
    :param target: The point cloud you wanted to aim for
    :param transformation: the transformation matrix you calculated
    :param visualize: if you want to visualize the result
    :return: Point Cloud, both added together
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if visualize:
        o3d.visualization.draw_geometries([source_temp, target_temp])
    new_pcd = source_temp + target_temp
    return new_pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    A preprocessing step of a point cloud, to downsample for better calculation later.
    :param pcd: the given pointcloud, in open3D style
    :param voxel_size: the size of the voxel, you want to downpixel to
    :return: a smaller point cloud, and the fpfh feature
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def preprocess_normals_fpfh(pcd, voxel_size=0.5):
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd, pcd_fpfh


def prepare_dataset(voxel_size, cloud_a=None, cloud_b=None):
    """
    A preprocessing stept to prepare your dataset for registration

    :param voxel_size: the voxel size you want to downpixel to
    :param cloud_a: point cloud A
    :param cloud_b: point cloud B
    :return: both point clouds, both smaller point clouds, both fpfh features
    """
    print(":: Load two point clouds and disturb initial pose.")
    if cloud_a is None:
        source = read_pcd_file("data/14_ramp_order/cluster_1.pcd", visualize=False)
    else:
        source = cloud_a
    if cloud_b is None:
        target = read_pcd_file(
            "data/frankenstein/cluster_comb_9u11_13.pcd", visualize=False
        )
    else:
        target = cloud_b
    trans_init = numpy.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source, target, voxel_size=0.5, visualize=True):
    """
    The main function for executing the global registration based on RANSAC

    :param source_down: point cloud A in smaller resolution
    :param target_down: point cloud B in smaller resolution
    :param source_fpfh: fpfh feature of point cloud A
    :param target_fpfh: fpfh feature of point cloud B
    :param voxel_size: the voxel size, you downsampled with
    :return: transformation matrix of point cloud A to B
    """

    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(voxel_size, cloud_a=source, cloud_b=target)
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999),
    )

    pcd = draw_registration_result(
        source_down, target_down, result.transformation, visualize=visualize
    )
    return pcd, source_down, target_down, result, result.transformation


def execute_RANSAC_for_multiway(source, target, voxel_size=0.5):
    """
    The main function for executing the global registration based on RANSAC

    :param source: point cloud A in smaller resolution
    :param target: point cloud B in smaller resolution
    :param voxel_size: the voxel size, you downsampled with
    :return: transformation matrix of point cloud A to B
    """
    source_down, source_fpfh = preprocess_normals_fpfh(source, voxel_size=voxel_size)
    target_down, target_fpfh = preprocess_normals_fpfh(target, voxel_size=voxel_size)
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999),
    )

    pcd = draw_registration_result(
        source_down, target_down, result.transformation, visualize=False
    )
    return pcd, source_down, target_down, result, result.transformation


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    """
    The main function for the registration based on Fast Global Registration
    :param source_down: point cloud A in smaller resolution
    :param target_down: point cloud B in smaller resolution
    :param source_fpfh: fpfh feature of point cloud A
    :param target_fpfh: fpfh feature of point cloud B
    :param voxel_size: the voxel size, you downsampled with
    :return: transformation matrix of point cloud A to B
    """
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def start_fast_global_registration(cloud_a=None, cloud_b=None):
    """
    The function to call from outer scope to start the fast global registration in this file
    :return:
    """
    voxel_size = 0.5
    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(voxel_size, cloud_a=cloud_a, cloud_b=cloud_b)

    result_fast = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)


def start_icp_ptp(source, target, trans_init):
    """
    The main funtcion to start the ICP algorithm for two point clouds (local transformation)

    :param source: Point Cloud A
    :param target: Point Cloud B
    :param trans_init: Initial transformation to gain a very close matching already
    :return: the added together point clouds
    """
    print("Initial alignment")
    threshold = 1.0
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    resulting_pcd = draw_registration_result(source, target, reg_p2p.transformation)
    return resulting_pcd, reg_p2p.transformation


def icp_ptp_multiway(source, target, trans_init):
    """
    The main funtcion to start the ICP algorithm for two point clouds (local transformation)

    :param source: Point Cloud A
    :param target: Point Cloud B
    :param trans_init: Initial transformation to gain a very close matching already
    :return: the added together point clouds
    """
    print("Initial alignment")
    threshold = 1.0
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
    )
    return reg_p2p


def start_transformation_pipeline(cloud_a=None, cloud_b=None):
    """
    A Pipeline function to start first the Fast Global Transformation, followed by the ICP PtP Transformation
    :param cloud_a: Point Cloud A
    :param cloud_b: Point Cloud B
    :return: The two matched point clouds
    """
    voxel_size = 0.1
    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(voxel_size, cloud_a=cloud_a, cloud_b=cloud_b)

    result_fast = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    draw_registration_result(source_down, target_down, result_fast.transformation)

    result = start_icp_ptp(source_down, target_down, result_fast.transformation)

    return result
