import open3d as o3d
import random
import numpy as np

paths = {
    "cylin": "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/cylin.stl",
    "rampshere": "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/Ramp_sphere.stl",
    "rampshere_upscaled": "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/Ramp_sphere_upscale.stl",
    "bunny": "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/Stanford_Bunny.stl",
    "bunny_low": "C:/Users/Grego/Documents/Universität/Master V/Optonics Projekt/3D Objekte/Bunny-LowPoly.stl",
}


def generate_fragments(object_type, n):
    if object_type not in paths:
        raise TypeError(
            "This object seems to not exist. Perhaps the archives are incomplete."
        )
    path = paths[object_type]
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    all_fragments = []
    for i in range(n):
        shift = random.randint(-1000, 1000)
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=5000 + shift, init_factor=5
        )
        random_point = random.randint(0, len(pcd.points) - 1)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd.colors[random_point] = [1, 0, 0]
        shift = random.randint(-500, 500)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(
            pcd.points[random_point], 2000 + shift
        )
        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        pcd_fragment = pcd.select_by_index(idx)
        print(pcd_fragment)
        all_fragments.append(pcd_fragment)
        # o3d.visualization.draw_geometries([pcd])
    return all_fragments


_ = generate_fragments("rampshere", 5)
