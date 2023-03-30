""""""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import transformations as transform
import open3d as o3d
import torch
import torch.nn as nn
from tqdm import tqdm

RADIUS_NORMAL = 0.1


def best_fit_transform_point2plane(A, B, normals):
    """
    reference: https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
    Input:
      A: Nx3 numpy array of corresponding points
      B: Nx3 numpy array of corresponding points
      normals: Nx3 numpy array of B's normal vectors
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape
    assert A.shape == normals.shape

    H = []
    b = []
    for i in range(A.shape[0]):
        dx = B[i, 0]
        dy = B[i, 1]
        dz = B[i, 2]
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]
        sx = A[i, 0]
        sy = A[i, 1]
        sz = A[i, 2]

        _a1 = (nz * sy) - (ny * sz)
        _a2 = (nx * sz) - (nz * sx)
        _a3 = (ny * sx) - (nx * sy)

        _a = np.array([_a1, _a2, _a3, nx, ny, nz])
        _b = (nx * dx) + (ny * dy) + (nz * dz) - (nx * sx) - (ny * sy) - (nz * sz)

        H.append(_a)
        b.append(_b)

    H = np.array(H)
    b = np.array(b)

    tr = np.dot(np.linalg.pinv(H), b)
    T = transform.euler_matrix(tr[0], tr[1], tr[2])
    T[0, 3] = tr[3]
    T[1, 3] = tr[4]
    T[2, 3] = tr[5]

    R = T[:3, :3]
    t = T[:3, 3]

    return T, R, t


def best_fit_transform_point2point(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1, :] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t



def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(src_tm, dst_tm, init_pose=None, max_iterations=20, tolerance=None, samplerate=1):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        samplerate: subsampling rate
    Output:
        T: final homogeneous transformation that maps A on to B
        MeanError: list, report each iteration's distance mean error
    """

    style = "open3d"

    if style == "trimesh":
        # get vertices and their normals from trimesh
        src_pts = np.array(src_tm.vertices)
        dst_pts = np.array(dst_tm.vertices)
        src_pt_normals = np.array(src_tm.vertex_normals)
        dst_pt_normals = np.array(dst_tm.vertex_normals)
    elif style == "open3d":
        # get vertices and their normals from point cloud
        src_pts = np.asarray(src_tm.points)
        dst_pts = np.asarray(dst_tm.points)
        src_tm.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL, max_nn=30)
        )
        dst_tm.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL, max_nn=30)
        )
        src_pt_normals = np.asarray(src_tm.normals)[:]
        dst_pt_normals = np.asarray(dst_tm.normals)[:]

    # subsampling
    ids = np.random.uniform(0, 1, size=src_pts.shape[0])
    A = src_pts[ids < samplerate, :]
    A_normals = src_pt_normals[ids < samplerate, :]
    ids = np.random.uniform(0, 1, size=dst_pts.shape[0])
    B = dst_pts[ids < samplerate, :]
    B_normals = dst_pt_normals[ids < samplerate, :]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    MeanError = []

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # match each point of source-set to closest point of destination-set,
        matched_src_pts = src[:m, :].T.copy()
        matched_dst_pts = dst[:m, :].T

        # compute angle between 2 matched vertexs' normals
        matched_src_pt_normals = A_normals.copy()
        matched_dst_pt_normals = B_normals.copy()

        angles = np.zeros(matched_src_pt_normals.shape[0])
        for k in range(matched_src_pt_normals.shape[0]):
            v1 = matched_src_pt_normals[k, :]
            v2 = matched_dst_pt_normals[k, :]
            cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles[k] = np.arccos(cos_angle) / np.pi * 180

        # and reject the bad corresponding
        dist_threshold = np.inf
        dist_bool_flag = (distances < dist_threshold)
        angle_threshold = 20
        angle_bool_flag = (angles < angle_threshold)
        reject_part_flag = dist_bool_flag * angle_bool_flag

        # get matched vertexes and dst_vertexes' normals
        matched_src_pts = matched_src_pts[reject_part_flag, :]
        matched_dst_pts = matched_dst_pts[reject_part_flag, :]
        matched_dst_pt_normals = matched_dst_pt_normals[reject_part_flag, :]

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform_point2plane(matched_src_pts, matched_dst_pts, matched_dst_pt_normals)

        # update the current source
        src = np.dot(T, src)

        # print iteration
        print('\ricp iteration: %d/%d  %s...' % (i+1, max_iterations, str(distances.mean())), end='', flush=True)

        # check error
        mean_error = np.mean(distances[reject_part_flag])
        MeanError.append(mean_error)
        if tolerance is not None:
            if np.abs(prev_error - mean_error) < tolerance:
                print('\nbreak iteration, the distance between two adjacent iterations '
                      'is lower than tolerance (%.f < %f)'
                      % (np.abs(prev_error - mean_error), tolerance))
                break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform_point2point(A, src[:m, :].T)
    print()

    return T, MeanError


def icp_with_DL(model_predict_func, params, init_pose=None, max_iterations=20, tolerance=None, samplerate=1):
    prnet_args, prnet, dataloader = params
    device = prnet_args.device

    src, target, source_full, target_full, R_ab, translation_ab = model_predict_func(prnet_args, prnet, dataloader)
    src, target, _, _ = pcd_to_numpy(src, target)
    overall_mse = []

    R_ab = R_ab[0].detach().cpu().numpy()
    translation_ab = translation_ab[0].detach().cpu().numpy()

    M_gt = np.vstack((np.hstack((R_ab, translation_ab[:, None])), [0, 0, 0, 1]))

    for o, data in enumerate(tqdm(dataloader)):
        template, source, igt = data

        src_og = source[0].detach().cpu().numpy().copy()
        tar_og = template[0].detach().cpu().numpy().copy()

        transformations = get_transformations(igt)
        transformations = [t.to(device) for t in transformations]
        R_ab, translation_ab, R_ba, translation_ba = transformations
        translation_ab = translation_ab.permute(0, 2, 1)

        R_ab = R_ab[0].detach().cpu().numpy()
        translation_ab = translation_ab[0].detach().cpu().numpy()[0]

        M_gt = np.vstack((np.hstack((R_ab, translation_ab[:, None])), [0, 0, 0, 1]))

        source = src_og
        target = tar_og

        for i in range(max_iterations):
            src = source.copy()
            src, target = model_matching_prediction(prnet, src, target, device)

            src, target, src_pt_normals, matched_dst_pt_normals = pcd_to_numpy(src, target)

            src_4 = np.ones((4, 512))
            src_4[:3, :] = np.copy(source.T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = best_fit_transform_point2plane(src, target, matched_dst_pt_normals)

            # update the current source
            src_4 = np.dot(T, src_4)
            source = src_4.T[:, :3]

            mse = numpy_MSE_loss(T, M_gt)

            #print('\ricp iteration: %d/%d  %s...' % (i + 1, max_iterations, str(mse)), end='', flush=True)
        overall_mse.append(mse)
        if o % 25 == 0:
            the_MSE = sum(overall_mse) / len(overall_mse)
            print("Overall MSE %s" % the_MSE)
        #display_result(source, tar_og)
    the_MSE = sum(overall_mse) / len(overall_mse)
    print("Overall MSE %s" % the_MSE)


def pcd_to_numpy(src_tm, dst_tm):
    # from point cloud to numpy
    src_pts = np.asarray(src_tm.points)
    dst_pts = np.asarray(dst_tm.points)
    src_tm.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL, max_nn=30)
    )
    dst_tm.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL, max_nn=30)
    )
    src_pt_normals = np.asarray(src_tm.normals)[:]
    dst_pt_normals = np.asarray(dst_tm.normals)[:]
    return src_pts, dst_pts, src_pt_normals, dst_pt_normals


def numpy_to_tensor(src, temp):
    src = torch.tensor(src, dtype=torch.float)
    src = torch.unsqueeze(src, 0)
    temp = torch.tensor(temp, dtype=torch.float)
    temp = torch.unsqueeze(temp, 0)
    return src, temp


def numpy_MSE_loss(A, B):
    mse = (np.square(A - B)).mean(axis=None)
    return mse


def model_matching_prediction(model, source, template, device):
    source, template = numpy_to_tensor(source, template)

    template = template.to(device)
    source = source.to(device)
    extra_data = model.predict_keypoint_correspondence(template, source)

    extra_source, extra_target, extra_scores = extra_data
    extra_source = extra_source[0].permute(1, 0)
    extra_target = extra_target[0].permute(1, 0)
    extra_source = extra_source.detach().cpu().numpy()
    extra_target_plus = extra_target.detach().cpu().numpy()
    extra_target_ = o3d.geometry.PointCloud()
    extra_source_ = o3d.geometry.PointCloud()
    extra_target_.points = o3d.utility.Vector3dVector(extra_target_plus)
    extra_source_.points = o3d.utility.Vector3dVector(extra_source + np.array([0, 0, 0]))

    extra_scores = extra_scores[0].detach().cpu().numpy()
    length = len(extra_target_plus)

    matching_order_ind = [argmax(extra_scores[i]) for i in range(length)]
    target_return = extra_target_plus[matching_order_ind]
    extra_target_.points = o3d.utility.Vector3dVector(target_return)

    return extra_source_, extra_target_


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def display_result(source, target):
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd.points = o3d.utility.Vector3dVector(target)
    source_pcd.paint_uniform_color([1, 0, 0])
    target_pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([source_pcd, target_pcd])


def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]  # Ps = R_ba * Pt
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)  # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)  # Pt = R_ab * Ps
    translation_ab = -torch.bmm(R_ab, translation_ba)  # Pt = Ps + t_ab
    return R_ab, translation_ab, R_ba, translation_ba
