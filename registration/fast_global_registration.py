import open3d as o3d
import copy
import numpy

from util_functions import read_pcd_file, save_pcd_file


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    new_pcd = source_temp + target_temp
    return new_pcd


def preprocess_point_cloud(pcd, voxel_size):
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


def prepare_dataset(voxel_size, cloud_a=None, cloud_b=None):
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


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
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
    voxel_size = 2.0
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
    return resulting_pcd


def start_transformation_pipeline(cloud_a=None, cloud_b=None):
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
