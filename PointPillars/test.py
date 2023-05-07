import open3d as o3d
import numpy as np

def create_lineset_from_obb(obb, color=(1, 0, 0)):
    # Get the corner points of the oriented bounding box
    corners = np.asarray(obb.get_box_points())

    # Define the edges of the bounding box using corner indices
    # edges = [
    #     [0, 1], [1, 5], [5, 4], [4, 0],
    #     [2, 3], [3, 7], [7, 6], [6, 2],
    #     [0, 2], [1, 3], [5, 7], [4, 6]
    # ]
    print(corners)
    edges = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create the LineSet and set the points and lines
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(edges)

    # Set the color for the lines
    lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(edges))])

    return lineset

# Create a random point cloud
points = np.random.rand(100, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Calculate the oriented bounding box
obb = point_cloud.get_oriented_bounding_box()

# print(obb)

# Create a LineSet representing the oriented bounding box with the desired color
obb_lineset = create_lineset_from_obb(obb, color=(1, 0, 0))  # Red color

# Visualize the point cloud and the colored oriented bounding box
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([point_cloud, obb_lineset, mesh_frame])
