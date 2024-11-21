from functools import reduce
import os
import sys
import time

import numpy as np
import open3d as o3d
from tqdm import trange

from utils import load_config, write_config
from vdbfusion import VDBVolume
from nuscene_utils import preprocess_mesh,mesh_to_level_set,level_set_to_numpy,extract_mesh,visualize_vdb_grid,extract_tsdf_values


class VDBFusionPipeline:
    """Abstract class that defines a Pipeline, derived classes must implement the dataset and config
    properties."""

    def __init__(self, dataset, config_file: str, map_name: str, jump: int = 0, n_scans: int = -1):
        self._dataset = dataset
        self._config = load_config(config_file)
        self._n_scans = len(dataset) if n_scans == -1 else n_scans
        self._jump = jump
        self._map_name = f"{map_name}_{self._n_scans}_scans"
        self._tsdf_volume = VDBVolume(
            self._config.voxel_size,
            self._config.sdf_trunc,
            self._config.space_carving,
        )
        self._res = {}

    @staticmethod
    def save_tsdf_values_to_txt(active_voxels, active_tiles):
        filename = '/root/autodl-tmp/vdbfusion/examples/python/results/voxel_tile.txt'
        """
        将活跃体素和瓦片的坐标及 TSDF 值保存到文本文件。

        参数：
        - filename: str，输出的文本文件路径
        - active_voxels: List[Tuple[Tuple[int, int, int], float]]，活跃体素列表
        - active_tiles: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], float]]，活跃瓦片列表
        """

        with open(filename, "w") as file:
            # 写入活跃体素
            file.write("Active Voxels:\n")
            for voxel in active_voxels:
                coord, value = voxel
                file.write(f"voxel: {coord}, TSDF Value: {value}\n")

            # 写入活跃瓦片
            file.write("\nActive Tiles:\n")
            for tile in active_tiles:
                coord_min, coord_max, value = tile
                file.write(
                    f"tile: Min: {coord_min}, Max: {coord_max}, TSDF Value: {value}\n"
                )

        print(f"Data successfully saved to {filename}")

    def run(self):
        self._run_tsdf_pipeline()
        print(dir(self._tsdf_volume.colors))
        # active_voxels, active_tiles = extract_tsdf_values(self._tsdf_volume.tsdf)
        # self.save_tsdf_values_to_txt(active_voxels, active_tiles)
        # self._write_ply()
        # self._write_cfg()
        # self._write_vdb()
        # self._print_tim()
        # self._print_metrics()

    def visualize_and_save(self, output_path="output_image.png"):
        """
        将 3D 网格可视化并保存为图片（使用离线渲染器，无需显示窗口）。

        Parameters:
        - output_path (str): 保存图片的路径。
        """
        # 准备几何数据
        mesh = self._res["mesh"]
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            raise ValueError("The input object is not a valid TriangleMesh")

        # 设置渲染场景
        scene = o3d.visualization.rendering.OffscreenRenderer(1024, 768)  # 图像分辨率
        scene.scene.set_background([1, 1, 1, 1])  # 白色背景
        scene.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())

        # 渲染并保存
        image = scene.render_to_image()
        o3d.io.write_image(output_path, image)
        print(f"可视化结果已保存为 {output_path}")

    def visualize(self):
        o3d.visualization.draw_geometries([self._res["mesh"]])
        # self.visualize_and_save(output_path="/root/autodl-tmp/vdbfusion/examples/python/results/scan103.png")

    def __len__(self):
        return len(self._dataset)

    def _run_tsdf_pipeline(self):
        times = []
        for idx in trange(self._jump, self._jump + self._n_scans, unit=" frames"):
            scan,color,pose = self._dataset[idx]
            tic = time.perf_counter_ns()
            self._tsdf_volume.integrate(scan, color,pose)
            toc = time.perf_counter_ns()
            times.append(toc - tic)
        self._res = {"mesh": self._get_o3d_mesh(self._tsdf_volume, self._config), "times": times}

    def _write_vdb(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".vdb"
        self._tsdf_volume.extract_vdb_grids(filename)

    def _write_ply(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])

    def _write_cfg(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".yml"
        write_config(dict(self._config), filename)

    def _print_tim(self):
        total_time_ns = reduce(lambda a, b: a + b, self._res["times"])
        total_time = total_time_ns * 1e-9
        total_scans = self._n_scans - self._jump
        self.fps = float(total_scans / total_time)

    @staticmethod
    def _get_o3d_mesh(tsdf_volume, cfg):
        vertices, triangles, colors = tsdf_volume.extract_triangle_mesh(cfg.fill_holes, cfg.min_weight)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors/255.0)  # 添加颜色
        mesh.compute_vertex_normals()
        # print("Preprocessing input mesh...")
        # mesh = preprocess_mesh(mesh)
        #
        # # Convert it to a level set using OpenVDB tools
        # print("Converting Triangle Mesh to a level set volume...")
        # vdb_grid = mesh_to_level_set(mesh, 0.01)
        #
        # # Convert the VDB grid to a dense numpy array
        # print("Converting to a dense SDF representation")
        # sdf_volume, _ = level_set_to_numpy(vdb_grid)
        # print("Volume output:")
        # print("sdf_volume.shape = ", sdf_volume.shape)
        # print("sdf_volume.min() = ", sdf_volume.min())
        # print("sdf_volume.max() = ", sdf_volume.max())
        #
        # # You can now save the sdf_volume as a np array and read it later on
        # numpy_filename = '/root/autodl-tmp/vdbfusion/examples/python/results' + self._map_name+"_sdf.npy"
        # print("Saving sdf_volume to", numpy_filename)
        # np.save(numpy_filename, sdf_volume)

        # if mcubes:
        #     print("Meshing dense volume by running marching cubes")
        #     sdf_mesh = extract_mesh(sdf_volume)
        #     # o3d.visualization.draw_geometries([sdf_mesh]) if visualize else None
        #     mesh_filename = model_name + "_sdf_mesh" + file_extension
        #     print("Saving sdf_volume mesh to", mesh_filename)
        #     o3d.io.write_triangle_mesh(mesh_filename, sdf_mesh)
        return mesh

    def _print_metrics(self):
        # If PYOPENVDB_SUPPORT has not been enabled then we can't report any metrics
        if not self._tsdf_volume.pyopenvdb_support_enabled:
            print("No metrics available, please compile with PYOPENVDB_SUPPORT")
            return

        # Compute the dimensions of the volume mapped
        grid = self._tsdf_volume.tsdf
        bbox = grid.evalActiveVoxelBoundingBox()
        dim = np.abs(np.asarray(bbox[1]) - np.asarray(bbox[0]))
        volume_extent = np.ceil(self._config.voxel_size * dim).astype(np.int32)
        volume_extent = f"{volume_extent[0]} x {volume_extent[1]} x {volume_extent[2]}"

        # Compute memory footprint
        total_voxels = int(np.prod(dim))
        float_size = 4
        # Always 2 grids
        mem_footprint = 2 * grid.memUsage() / (1024 * 1024)
        dense_equivalent = 2 * (float_size * total_voxels) / (1024 * 1024 * 1024)  # GB

        # compute size of .vdb file
        filename = os.path.join(self._config.out_dir, self._map_name) + ".vdb"
        file_size = float(os.stat(filename).st_size) / (1024 * 1024)

        # print metrics
        trunc_voxels = int(np.ceil(self._config.sdf_trunc / self._config.voxel_size))

        filename = os.path.join(self._config.out_dir, self._map_name) + ".txt"
        with open(filename, "w") as f:
            stdout = sys.stdout
            sys.stdout = f  # Change the standard output to the file we created.
            print(f"--------------------------------------------------")
            print(f"Results for dataset {self._map_name}:")
            print(f"--------------------------------------------------")
            print(f"voxel size       = {self._config.voxel_size} [m]")
            print(f"truncation       = {trunc_voxels} [voxels]")
            print(f"space carving    = {self._config.space_carving}")
            print(f"Avg FPS          = {self.fps:.2f} [Hz]")
            print(f"--------------------------------------------------")
            print(f"volume extent    = {volume_extent} [m x m x m]")
            print(f"memory footprint = {mem_footprint:.2f} [MB]")
            print(f"dense equivalent = {dense_equivalent:.2f} [GB]")
            print(f"size on disk     = {file_size:.2f} [MB]")
            print(f"--------------------------------------------------")
            print(f"number of scans  = {len(self)}")
            print(f"points per scan  = {len(self._dataset[0][0])}")
            print(f"min range        = {self._config.min_range} [m]")
            print(f"max range        = {self._config.max_range} [m]")
            print(f"--------------------------------------------------")
            sys.stdout = stdout

        # Print it
        os.system(f"cat {filename}")