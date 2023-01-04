import torch
import open3d as o3d
from common import se3


def translate_np_to_pcd(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


def visualize_foursome(np_pairs):

    for item in np_pairs:
        src = item[0][0].cpu().numpy()
        cls = item[1][0].cpu().numpy()
        print(src.shape)
        print(cls.shape)

        src = translate_np_to_pcd(src)
        cls_1 = translate_np_to_pcd(cls[0])
        cls_2 = translate_np_to_pcd(cls[1])
        o3d.visualization.draw_geometries([src])
        o3d.visualization.draw_geometries([cls_1])
        o3d.visualization.draw_geometries([cls_2])
        o3d.visualization.draw_geometries([src, cls_1, cls_2])


def visualize_result(net_output, data_batch):

    points_src = data_batch["points_src"].cpu().numpy()
    points_ref = data_batch["points_ref"].cpu().numpy()
    transformation = net_output["transform_pair"][1].cpu().numpy()

    points_transformed = se3.np_transform(transformation, points_src)

    for i in range(points_transformed.shape[0]):
        item = points_transformed[i]
        item2 = points_src[i]
        ref = points_ref[i]
        print("We attempt")
        point = translate_np_to_pcd(item)
        point2 = translate_np_to_pcd(item2)
        ref_point = translate_np_to_pcd(ref)
        o3d.visualization.draw_geometries([point])
        o3d.visualization.draw_geometries([point, point2])
        o3d.visualization.draw_geometries([point, ref_point])
    print("finished")
    src = translate_np_to_pcd(points_src[0])
    raw = translate_np_to_pcd(points_ref[0])

    o3d.visualization.draw_geometries([src, raw])

