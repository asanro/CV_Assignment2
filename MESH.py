import numpy as np
import pyvista as pv

"""
Creates a mesh from the data_mesh_good file. 
"""

with open('data_mesh_good.txt', 'r') as f:
    file_contents = f.read()
    data_mesh = eval(file_contents)
    f.close()

# NumPy array with shape (n_points, 3)
points = np.array(data_mesh)  # np.genfromtxt('points.csv', delimiter=",", dtype=np.float32)

# points is a 3D numpy array (n_points, 3) coordinates of a sphere
cloud = pv.PolyData(points)

volume = cloud.delaunay_3d(alpha=0.7)
shell = volume.extract_geometry()
shell.plot()