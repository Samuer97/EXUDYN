"""
Blisk STL Mesh Visualization
显示叶盘网格结构，分析节点分布和单元连接关系
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

def load_binary_stl(filename):
    """Load binary STL file"""
    vertices = []
    faces = []
    vertex_map = {}
    
    with open(filename, 'rb') as f:
        # Skip 80-byte header
        f.read(80)
        
        # Read number of triangles
        num_triangles = struct.unpack('<I', f.read(4))[0]
        print(f"Binary STL: {num_triangles} triangles")
        
        for i in range(num_triangles):
            # Skip normal vector (3 floats)
            f.read(12)
            
            # Read 3 vertices (9 floats)
            triangle_vertices = []
            for j in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                vertex = (x, y, z)
                
                # Add to vertex list if not already present
                if vertex not in vertex_map:
                    vertex_map[vertex] = len(vertices)
                    vertices.append([x, y, z])
                
                triangle_vertices.append(vertex_map[vertex])
            
            faces.append(triangle_vertices)
            
            # Skip attribute byte count
            f.read(2)
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    print(f"Loaded: {len(vertices)} unique vertices, {len(faces)} triangles")
    return vertices, faces

def analyze_mesh_structure(vertices_3d, faces):
    """分析网格结构"""
    nodes_2d = vertices_3d[:, :2]  # Project to XY plane
    n_nodes = len(nodes_2d)
    
    # 计算每个节点到原点的距离
    distances = np.linalg.norm(nodes_2d, axis=1)
    
    print(f"\n网格结构分析:")
    print(f"节点总数: {n_nodes}")
    print(f"三角形单元数: {len(faces)}")
    print(f"最小半径: {np.min(distances):.6f} m")
    print(f"最大半径: {np.max(distances):.6f} m")
    
    # 分析距离分布
    unique_distances = np.unique(np.round(distances, 4))
    print(f"唯一距离值前15个: {unique_distances[:15]}")
    
    # 统计不同半径范围的节点数
    radius_ranges = [0.1, 0.105, 0.11, 0.12, 0.15, 0.2, 0.3, 0.4]
    for r in radius_ranges:
        count = np.sum(distances <= r)
        print(f"r ≤ {r:.3f}m 的节点数: {count}")
    
    return nodes_2d, distances

def classify_nodes(nodes_2d, distances):
    """分类节点：内圈、中间、外圈"""
    
    # 内圈节点 (r <= 0.105m，应该有16个)
    inner_nodes = np.where(distances <= 0.105)[0]
    
    # 外圈节点 (r >= 0.38m)
    outer_nodes = np.where(distances >= 0.38)[0]
    
    # 中间节点
    middle_nodes = np.where((distances > 0.105) & (distances < 0.38))[0]
    
    print(f"\n节点分类:")
    print(f"内圈节点 (r ≤ 0.105m): {len(inner_nodes)} 个")
    print(f"中间节点 (0.105m < r < 0.38m): {len(middle_nodes)} 个") 
    print(f"外圈节点 (r ≥ 0.38m): {len(outer_nodes)} 个")
    
    if len(inner_nodes) > 0:
        print(f"内圈节点距离: {np.round(distances[inner_nodes], 4)}")
    
    return inner_nodes, middle_nodes, outer_nodes

def visualize_mesh(nodes_2d, faces, inner_nodes, middle_nodes, outer_nodes, distances):
    """可视化网格结构"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 子图1: 完整网格结构
    ax1 = axes[0, 0]
    
    # 绘制所有三角形单元
    for i, face in enumerate(faces):
        triangle = nodes_2d[face]
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax1.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    
    # 绘制节点，按类别着色
    ax1.scatter(nodes_2d[inner_nodes, 0], nodes_2d[inner_nodes, 1], 
               c='red', s=50, alpha=0.8, label=f'内圈节点 ({len(inner_nodes)})', marker='o')
    ax1.scatter(nodes_2d[middle_nodes, 0], nodes_2d[middle_nodes, 1], 
               c='green', s=20, alpha=0.6, label=f'中间节点 ({len(middle_nodes)})', marker='.')
    ax1.scatter(nodes_2d[outer_nodes, 0], nodes_2d[outer_nodes, 1], 
               c='blue', s=30, alpha=0.7, label=f'外圈节点 ({len(outer_nodes)})', marker='^')
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('叶盘网格结构 - 完整视图')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    
    # 子图2: 内圈放大视图
    ax2 = axes[0, 1]
    
    # 只显示内圈附近的单元
    inner_region_nodes = np.where(distances <= 0.15)[0]
    inner_faces = []
    for face in faces:
        if all(node in inner_region_nodes for node in face):
            inner_faces.append(face)
    
    for face in inner_faces:
        triangle = nodes_2d[face]
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax2.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', alpha=0.5, linewidth=1)
    
    # 绘制内圈节点
    ax2.scatter(nodes_2d[inner_nodes, 0], nodes_2d[inner_nodes, 1], 
               c='red', s=100, alpha=0.8, label=f'固定节点 ({len(inner_nodes)})')
    
    # 标注节点编号
    for i, node_idx in enumerate(inner_nodes):
        ax2.annotate(f'{node_idx}', (nodes_2d[node_idx, 0], nodes_2d[node_idx, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('内圈区域放大视图')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    
    # 子图3: 径向分布图
    ax3 = axes[1, 0]
    
    ax3.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0.1, color='red', linestyle='--', label='内半径 0.1m')
    ax3.axvline(x=0.105, color='orange', linestyle='--', label='固定边界 0.105m')
    ax3.axvline(x=0.4, color='blue', linestyle='--', label='外半径 0.4m')
    ax3.set_xlabel('距原点距离 [m]')
    ax3.set_ylabel('节点数量')
    ax3.set_title('节点径向分布直方图')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 极坐标视图
    ax4 = axes[1, 1]
    
    # 转换为极坐标
    theta = np.arctan2(nodes_2d[:, 1], nodes_2d[:, 0])
    
    # 绘制极坐标散点图
    scatter = ax4.scatter(theta, distances, c=distances, cmap='viridis', alpha=0.7, s=20)
    
    # 标记内圈节点
    inner_theta = theta[inner_nodes]
    inner_r = distances[inner_nodes]
    ax4.scatter(inner_theta, inner_r, c='red', s=100, alpha=0.8, 
               label=f'固定节点 ({len(inner_nodes)})', marker='o', edgecolors='black')
    
    ax4.set_xlabel('角度 [rad]')
    ax4.set_ylabel('半径 [m]')
    ax4.set_title('极坐标视图')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='半径 [m]')
    
    plt.tight_layout()
    plt.savefig('blisk_mesh_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    
    print("="*70)
    print("叶盘STL网格结构分析")
    print("="*70)
    
    # 加载STL文件
    stl_file = "d:/NTULearning/课题进展/202502叶盘/test/板_梁单元建模/ANSYS_DATA/blisk_65.stl"
    
    try:
        vertices_3d, faces = load_binary_stl(stl_file)
    except Exception as e:
        print(f"无法加载STL文件: {e}")
        return
    
    # 分析网格结构
    nodes_2d, distances = analyze_mesh_structure(vertices_3d, faces)
    
    # 分类节点
    inner_nodes, middle_nodes, outer_nodes = classify_nodes(nodes_2d, distances)
    
    # 可视化
    visualize_mesh(nodes_2d, faces, inner_nodes, middle_nodes, outer_nodes, distances)
    
    print("\n分析完成！")
    print("图像已保存为: blisk_mesh_analysis.png")
    print("="*70)

if __name__ == "__main__":
    main()
