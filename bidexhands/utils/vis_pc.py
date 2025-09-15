import open3d as o3d
import trimesh

num_surface_samples = 512
object_mesh_path = "../assets/botyard/panda_by_description/meshes/object/box_50mm/box.stl"
mesh = trimesh.load(object_mesh_path)
point_clouds, _ = trimesh.sample.sample_surface(mesh, num_surface_samples)
points = point_clouds[:, :3]
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])