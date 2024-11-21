import copy

import manifold
import numpy as np
import open3d as o3d
import os
from typing import Tuple
import pyopenvdb as vdb
from skimage.measure import marching_cubes


def extract_mesh(volume, mask=None):
    """Run marching_cubes and extract a triangular mesh of the volume.

    Parameters - copied from skimage -
    ----------
    volume : (M, N, P) array
        Input data volume to find isosurfaces. Will internally be
    mask : (M, N, P) array
        Boolean array. The marching cube algorithm will be computed only on
        True elements. This will save computational time when interfaces
        are located within certain region of the volume M, N, P-e.g. the top
        half of the cube-and also allow to compute finite surfaces-i.e. open
        surfaces that do not end at the border of the cube.
        converted to float32 if necessary.
    """
    vertices, faces, _, _ = marching_cubes(volume, level=0, mask=mask)
    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
    mesh.compute_vertex_normals()
    return mesh

def scale_to_unit_sphere(mesh, scale=1, padding=0.1):
    """Scale the input mesh into a unit sphere."""
    # Get bbox of original mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Translate the mesh
    scaled_mesh = copy.deepcopy(mesh)
    scaled_mesh.translate(-bbox.get_center())
    distances = np.linalg.norm(np.asarray(scaled_mesh.vertices), axis=1)
    scaled_mesh.scale(1 / np.max(distances), center=[0, 0, 0])
    scaled_mesh.scale(scale * (1 - padding), center=[0, 0, 0])
    return scaled_mesh


def scale_to_unit_cube(mesh, scale=1, padding=0.1):
    """Scale the input mesh into a unit cube."""
    # Get bbox of original mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Translate the mesh
    scaled_mesh = copy.deepcopy(mesh)
    scaled_mesh.translate(-bbox.get_center())
    scaled_mesh.scale(2 / bbox.get_max_extent(), center=[0, 0, 0])
    scaled_mesh.scale(scale * (1 - padding), center=[0, 0, 0])
    return scaled_mesh


def watertight_mesh(mesh, depth=8):
    """Conver the input mesh to a watertight model."""
    processor = manifold.Processor(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
    )

    output_vertices, output_triangles = processor.get_manifold_mesh(depth)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(output_vertices),
        o3d.utility.Vector3iVector(output_triangles),
    )


def preprocess_mesh(mesh, scale=False, watertight=False):
    """The mesh MUST be a closed surface, but not necessary watertight and can
    also contain self-intersecting faces, in contrast to most of mesh-to-sdf
    algorithms.

    Scaling is not mandatory, but it's for your own sanity
    """
    mesh = scale_to_unit_sphere(mesh) if scale else mesh
    mesh = watertight_mesh(mesh) if watertight else mesh
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_level_set(mesh, voxel_size, half_width=3):
    return vdb.FloatGrid.createLevelSetFromPolygons(
        points=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles),
        transform=vdb.createLinearTransform(voxelSize=voxel_size),
        halfWidth=half_width,
    )


def level_set_to_triangle_mesh(grid):
    points, quads = grid.convertToQuads()
    faces = np.array([[[f[0], f[1], f[2]], [f[0], f[2], f[3]]] for f in quads]).reshape((-1, 3))
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(points),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    return mesh


def sdf_level_set_to_triangle_mesh(grid):
    sdf_volume, origin = level_set_to_numpy(grid)
    mesh = extract_mesh(sdf_volume)
    mesh.translate(origin)
    return mesh


def level_set_to_numpy(grid: vdb.FloatGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Given an input level set (in vdb format) extract the dense array representation of the volume
    and convert it to a numpy array.

    You could check the output of the numpy array by running marching cubes over the
    volume and extracting the mesh for visualization.
    """
    # Dimensions of the axis-aligned bounding box of all active voxels.
    shape = grid.evalActiveVoxelDim()
    # Return the coordinates of opposite corners of the axis-aligned bounding
    # box of all active voxels.
    start = grid.evalActiveVoxelBoundingBox()[0]
    # Create a dense array of zeros
    sdf_volume = np.zeros(shape, dtype=np.float32)
    # Copy the volume to the output, starting from the first occupied voxel
    grid.copyToArray(sdf_volume, ijk=start)
    # solve background error see OpenVDB#1096
    sdf_volume[sdf_volume < grid.evalMinMax()[0]] = grid.background

    # In order to put a mesh back into its original coordinate frame we also
    # need to know where the volume was located
    origin_xyz = grid.transform.indexToWorld(start)
    return sdf_volume, origin_xyz


def visualize_vdb_grid(grid, filename, verbose=True):
    # Save it to file in /tmp
    grid_name = os.path.split(filename.split(".")[0])[-1]
    grid_fn = os.path.join("/tmp", grid_name + ".vdb")
    vdb.write(grid_fn, grid)

    # Plot the results
    os.system("vdb_print -l {}".format(grid_fn)) if verbose else None
    os.system("vdb_view {}".format(grid_fn))


def extract_tsdf_values(tsdf_grid):
    """
    从 TSDF 网格中提取所有活跃体素及其对应的 TSDF 值。

    参数：
    - tsdf_grid: openvdb.FloatGrid 对象

    返回：
    - active_voxels: List[Tuple[Tuple[int, int, int], float]] 活跃体素的坐标及对应值
    """
    active_voxels = []  # 用于存储活跃体素的坐标和值
    active_tiles = []
    for iter in tsdf_grid.iterOnValues():
        if iter.count ==1:
            active_voxels.append((iter.min,iter.value))
        else:
            active_tiles.append((iter.min,iter.max,iter.value))
    return active_voxels, active_tiles



