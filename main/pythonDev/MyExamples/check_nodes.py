import numpy as np
import struct

def load_binary_stl(filename):
    vertices = []
    vertex_map = {}
    with open(filename, 'rb') as f:
        f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        for i in range(num_triangles):
            f.read(12)
            for j in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                vertex = (x, y, z)
                if vertex not in vertex_map:
                    vertex_map[vertex] = len(vertices)
                    vertices.append([x, y, z])
            f.read(2)
    return np.array(vertices)

vertices_3d = load_binary_stl('d:/NTULearning/课题进展/202502叶盘/test/板_梁单元建模/ANSYS_DATA/blisk_65.stl')
nodes_2d = vertices_3d[:, :2]
distances = np.linalg.norm(nodes_2d, axis=1)
print(f'Min distance: {np.min(distances):.6f} m')
print(f'Max distance: {np.max(distances):.6f} m')
print(f'Nodes with r < 0.1: {np.sum(distances < 0.1)}')
print(f'Nodes with r <= 0.1: {np.sum(distances <= 0.1)}')
print(f'Nodes with r < 0.15: {np.sum(distances < 0.15)}')
inner_nodes = np.where(distances < 0.1)[0]
print(f'Inner nodes (r < 0.1m): {len(inner_nodes)}')
if len(inner_nodes) > 0:
    print(f'Inner node distances: {distances[inner_nodes]}')
