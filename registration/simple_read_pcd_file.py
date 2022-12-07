import open3d as o3d

point_cloud = o3d.io.read_point_cloud("data/t_piece.pcd")

o3d.visualization.draw_geometries([point_cloud])
