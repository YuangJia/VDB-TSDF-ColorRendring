import glob
import importlib
import os
import sys
from typing import List
import numpy as np
from trimesh import transform_points
import cv2
import copy
sys.path.append("..")
from utils.cache import get_cache, memoize
from utils.config import load_config
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]

class NuScenesDataset:
    def __init__(
        self,
        nusc_root_dir: str,
        sequence: int,
        config_file: str,
        *_,
        **__,
    ):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)

        # TODO: If someone needs more splits from nuScenes expose this 2 parameters
        #  nusc_version: str = "v1.0-trainval"
        #  split: str = "train"
        nusc_version: str = "v1.0-mini"
        split: str = "mini_train"
        self.lidar_name: str = "LIDAR_TOP"
        # self.lidar_name: str = 'RADAR_FRONT'
        # Lazy loading
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_logs

        # Config stuff
        self.sequence_id = str(int(sequence)).zfill(4)
        self.config = load_config(config_file)

        self.nusc = NuScenes(dataroot=str(nusc_root_dir), version=nusc_version)
        self.scene_name = f"scene-{self.sequence_id}"
        if self.scene_name not in [s["name"] for s in self.nusc.scene]:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

        # Load nuScenes read from file inside dataloader module
        self.load_point_cloud = importlib.import_module(
            "nuscenes.utils.data_classes"
        ).LidarPointCloud.from_file

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, self.nusc)

        # Use only the samples from the current split.
        scene_token = self._get_scene_token(split_logs)
        self.lidar_tokens = self._get_lidar_tokens(scene_token)
        self.first_pose,self.poses = self._load_poses()
        # Cache
        self.use_cache = True
        self.cache = get_cache(directory="cache/nuscenes/")

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
        # 读取点云和颜色数据
        points, color = self.read_point_cloud_and_color(self.lidar_tokens[idx], idx, self.config)
        points = points.astype(np.float64)
        color = color.astype(np.int32)
        # 确保 pose 的数据类型为 np.float64
        pose = self.poses[idx].astype(np.float64)
        return points, color, pose

    # @memoize()
    # def read_point_cloud_and_color(self, token: str, idx: int, config, min_dist: float = 0.0):
    #     """
    #     从点云和图像中提取 (X, Y, Z) 和颜色 (R, G, B) 数据。
    #
    #     Args:
    #         token (str): 当前点云的 token。
    #         idx (int): 当前帧索引。
    #         config: 配置参数。
    #         min_dist (float): 点到相机的最小距离过滤。
    #
    #     Returns:
    #         np.ndarray: 点云数据 (N, 3) 包含 (X, Y, Z)。
    #         np.ndarray: 颜色数据 (N, 3) 包含 (R, G, B)。
    #     """
    #     # Step 1: 获取点云数据
    #     lidar_data = self.nusc.get("sample_data", token)
    #     lidar_filename = lidar_data["filename"]
    #     pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, lidar_filename))  # (4, N)
    #     points = pcl.points.T[:, :4]  # 转换为 (N, 3)
    #     total_initial_points = points.shape[0]  # 初始点云总数
    #
    #     points = points[np.linalg.norm(points[:, :3], axis=1) <= config.max_range * 1.2]  # 使用前三列计算范围
    #     points = points[np.linalg.norm(points[:, :3], axis=1) >= config.min_range * 0.8]
    #
    #     # Step 2: 获取激光雷达到全局坐标系的转换参数
    #     lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    #     lidar2ego_translation = np.array(lidar_calib["translation"])
    #     lidar2ego_rotation = Quaternion(lidar_calib["rotation"])
    #
    #     lidar_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
    #     ego2global_translation = np.array(lidar_pose["translation"])
    #     ego2global_rotation = Quaternion(lidar_pose["rotation"])
    #
    #     # Step 3: 点云转换到全局坐标系
    #     pc = LidarPointCloud(points.T)
    #     pc.rotate(lidar2ego_rotation.rotation_matrix)
    #     pc.translate(lidar2ego_translation)
    #     pc.rotate(ego2global_rotation.rotation_matrix)
    #     pc.translate(ego2global_translation)
    #
    #     # Step 4: 遍历所有相机并投影点云到图像平面
    #     all_valid_points = []
    #     all_valid_colors = []
    #     sample = self.nusc.get("sample", lidar_data["sample_token"])
    #     for cam_key in cam_keys:  # 遍历指定相机
    #         cam_data = self.nusc.get("sample_data", sample["data"][cam_key])
    #         cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    #         cam_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
    #
    #         sensor2ego_translation = np.array(cam_calib["translation"])
    #         sensor2ego_rotation = Quaternion(cam_calib["rotation"])
    #         cam_ego2global_translation = np.array(cam_pose["translation"])
    #         cam_ego2global_rotation = Quaternion(cam_pose["rotation"])
    #         cam_intrinsic = np.array(cam_calib["camera_intrinsic"])
    #
    #         img = cv2.imread(os.path.join(self.nusc.dataroot, cam_data["filename"]))
    #
    #         # 点云转换到相机坐标系
    #         pc_cam = copy.deepcopy(pc)
    #         pc_cam.translate(-cam_ego2global_translation)
    #         pc_cam.rotate(cam_ego2global_rotation.rotation_matrix.T)
    #         pc_cam.translate(-sensor2ego_translation)
    #         pc_cam.rotate(sensor2ego_rotation.rotation_matrix.T)
    #
    #         # 投影到图像平面
    #         points_2d = view_points(pc_cam.points[:3, :], cam_intrinsic, normalize=True)
    #         u, v = points_2d[0, :].astype(int), points_2d[1, :].astype(int)
    #         mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0]) & (pc_cam.points[2, :] > min_dist)
    #
    #         # 提取有效点的坐标和颜色
    #         valid_points = pc.points.T[mask, :3]  # 全局坐标系下的 (X, Y, Z)
    #         valid_colors = img[v[mask], u[mask]]  # 图像中的 (R, G, B)
    #
    #         # 确保颜色数据是 (N, 3)，需要将颜色归一化为 np.float64 格式
    #         valid_colors = valid_colors.astype(np.float64)
    #
    #         all_valid_points.append(valid_points)
    #         all_valid_colors.append(valid_colors)
    #
    #     # 合并所有相机的有效点云和颜色数据
    #     final_points = np.vstack(all_valid_points).astype(np.float64)  # 最终点云 (N, 3)
    #     final_colors = np.vstack(all_valid_colors).astype(np.float64)  # 最终颜色 (N, 3)
    #
    #     # 计算点云保留比例
    #     retained_ratio = final_points.shape[0] / total_initial_points * 100
    #     print(f"最终点云数目比上总点云数目: {retained_ratio:.2f}%")
    #
    #     return final_points, final_colors

    # level 2
    # def read_point_cloud_and_color(self, token: str, idx: int, config, min_dist: float = 0.0):
    #     """
    #     从点云和图像中提取 (X, Y, Z) 和颜色 (R, G, B) 数据。
    #     """
    #     # Step 1: 获取点云数据
    #     lidar_data = self.nusc.get("sample_data", token)
    #     lidar_filename = lidar_data["filename"]
    #     pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, lidar_filename))  # (4, N)
    #     points = pcl.points.T[:, :4]  # 转换为 (N, 4)
    #     total_initial_points = points.shape[0]  # 初始点云总数
    #
    #     # 限制点云范围
    #     points = points[np.linalg.norm(points[:, :3], axis=1) <= config.max_range]
    #     points = points[np.linalg.norm(points[:, :3], axis=1) >= config.min_range]
    #
    #     # Step 2: 获取激光雷达到全局坐标系的转换参数
    #     lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    #     lidar2ego_translation = np.array(lidar_calib["translation"])
    #     lidar2ego_rotation = Quaternion(lidar_calib["rotation"])
    #
    #     lidar_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
    #     ego2global_translation = np.array(lidar_pose["translation"])
    #     ego2global_rotation = Quaternion(lidar_pose["rotation"])
    #
    #     # 转换点云到全局坐标系
    #     pc = LidarPointCloud(points.T)
    #     pc.rotate(lidar2ego_rotation.rotation_matrix)
    #     pc.translate(lidar2ego_translation)
    #     pc.rotate(ego2global_rotation.rotation_matrix)
    #     pc.translate(ego2global_translation)
    #
    #     # 用于存储投影后的有效点和颜色
    #     all_valid_points = []
    #     all_valid_colors = []
    #
    #     # Step 3: 遍历所有相机并投影点云到图像平面
    #     sample = self.nusc.get("sample", lidar_data["sample_token"])
    #     for cam_key in cam_keys:  # 遍历指定相机
    #         cam_data = self.nusc.get("sample_data", sample["data"][cam_key])
    #         cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    #         cam_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
    #
    #         sensor2ego_translation = np.array(cam_calib["translation"])
    #         sensor2ego_rotation = Quaternion(cam_calib["rotation"])
    #         cam_ego2global_translation = np.array(cam_pose["translation"])
    #         cam_ego2global_rotation = Quaternion(cam_pose["rotation"])
    #         cam_intrinsic = np.array(cam_calib["camera_intrinsic"])
    #
    #         img = cv2.imread(os.path.join(self.nusc.dataroot, cam_data["filename"]))
    #
    #         # 点云转换到相机坐标系
    #         pc_cam = copy.deepcopy(pc)
    #         pc_cam.translate(-cam_ego2global_translation)
    #         pc_cam.rotate(cam_ego2global_rotation.rotation_matrix.T)
    #         pc_cam.translate(-sensor2ego_translation)
    #         pc_cam.rotate(sensor2ego_rotation.rotation_matrix.T)
    #
    #         # 投影到图像平面
    #         points_2d = view_points(pc_cam.points[:3, :], cam_intrinsic, normalize=True)
    #         u, v = points_2d[0, :].astype(int), points_2d[1, :].astype(int)
    #         mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0]) & (pc_cam.points[2, :] > min_dist)
    #
    #         # 提取有效点的坐标和颜色
    #         valid_points = pc.points.T[mask, :3]  # 全局坐标系下的 (X, Y, Z)
    #         valid_colors = img[v[mask], u[mask]]  # 图像中的 (R, G, B)
    #
    #         all_valid_points.append(valid_points)
    #         all_valid_colors.append(valid_colors)
    #
    #         # 移除已被投影的点
    #         pc.points = pc.points[:, ~mask]
    #
    #     # Step 4: 处理未投影的点云
    #     unprojected_points = pc.points.T[:, :3]
    #     unprojected_colors = np.zeros((unprojected_points.shape[0], 3))  # 默认颜色设置为黑色或灰色
    #
    #     # 合并所有有效点和未投影点
    #     final_points = np.vstack(all_valid_points + [unprojected_points]).astype(np.float64)
    #     final_colors = np.vstack(all_valid_colors + [unprojected_colors]).astype(np.float64)
    #
    #     # 打印点云保留比例
    #     retained_ratio = final_points.shape[0] / total_initial_points * 100
    #     print(f"最终点云数目比上总点云数目: {retained_ratio:.2f}%")
    #
    #     return final_points, final_colors
    def read_point_cloud_and_color(self, token: str, idx: int, config, min_dist: float = 0.0):
        """
        从点云和图像中提取 (X, Y, Z) 和颜色 (R, G, B) 数据，并返回局部坐标系下的点云。
        """
        # Step 1: 获取点云数据
        lidar_data = self.nusc.get("sample_data", token)
        lidar_filename = lidar_data["filename"]
        pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, lidar_filename))  # (4, N)
        points = pcl.points.T[:, :3]  # 转换为 (N, 3)
        total_initial_points = points.shape[0]  # 初始点云总数

        # 限制点云范围
        points = points[np.linalg.norm(points, axis=1) <= config.max_range]
        points = points[np.linalg.norm(points, axis=1) >= config.min_range]

        # Step 2: 获取激光雷达到全局坐标系的转换矩阵
        lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        lidar2ego_translation = np.array(lidar_calib["translation"])
        lidar2ego_rotation = Quaternion(lidar_calib["rotation"])
        lidar2ego_matrix = lidar2ego_rotation.transformation_matrix
        lidar2ego_matrix[:3, 3] = lidar2ego_translation

        lidar_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego2global_translation = np.array(lidar_pose["translation"])
        ego2global_rotation = Quaternion(lidar_pose["rotation"])
        ego2global_matrix = ego2global_rotation.transformation_matrix
        ego2global_matrix[:3, 3] = ego2global_translation

        # 使用 transform_points 转换点云到全局坐标系
        points_global = transform_points(points, lidar2ego_matrix)
        points_global = transform_points(points_global, ego2global_matrix)

        # 用于存储投影后的有效点和颜色
        all_valid_points = []
        all_valid_colors = []

        # 用于记录哪些点被投影
        projected_mask = np.zeros(points_global.shape[0], dtype=bool)

        # Step 3: 遍历所有相机并投影点云到图像平面
        sample = self.nusc.get("sample", lidar_data["sample_token"])
        for cam_key in cam_keys:  # 遍历指定相机
            cam_data = self.nusc.get("sample_data", sample["data"][cam_key])
            cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            cam_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

            sensor2ego_translation = np.array(cam_calib["translation"])
            sensor2ego_rotation = Quaternion(cam_calib["rotation"])
            sensor2ego_matrix = sensor2ego_rotation.transformation_matrix
            sensor2ego_matrix[:3, 3] = sensor2ego_translation

            cam_ego2global_translation = np.array(cam_pose["translation"])
            cam_ego2global_rotation = Quaternion(cam_pose["rotation"])
            cam_ego2global_matrix = cam_ego2global_rotation.transformation_matrix
            cam_ego2global_matrix[:3, 3] = cam_ego2global_translation

            cam_intrinsic = np.array(cam_calib["camera_intrinsic"])

            img = cv2.imread(os.path.join(self.nusc.dataroot, cam_data["filename"]))

            # 点云转换到相机坐标系
            points_cam = transform_points(points_global, np.linalg.inv(cam_ego2global_matrix))  # 从全局坐标到相机坐标
            points_cam = transform_points(points_cam, np.linalg.inv(sensor2ego_matrix))


            # 投影到图像平面
            points_2d = view_points(points_cam.T, cam_intrinsic, normalize=True)  # 确保输入为 (3, N)
            u, v = points_2d[0, :].astype(int), points_2d[1, :].astype(int)
            mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0]) & (points_cam[:, 2] > min_dist)

            # 更新总的投影掩码
            projected_mask |= mask

            # 提取有效点的坐标和颜色
            valid_points = points_global[mask]  # 全局坐标系下的 (X, Y, Z)
            valid_colors = img[v[mask], u[mask]]  # 图像中的 (R, G, B)

            all_valid_points.append(valid_points)
            all_valid_colors.append(valid_colors)

        # Step 4: 处理未投影的点云
        unprojected_points = points_global[~projected_mask]
        unprojected_colors = np.full((unprojected_points.shape[0], 3), 200)  # 默认颜色设置为灰色 (200, 200, 200)

        # 合并所有有效点和未投影点
        final_points_global = np.vstack(all_valid_points + [unprojected_points]).astype(np.float64)
        final_colors = np.vstack(all_valid_colors + [unprojected_colors]).astype(np.float64)

        # 获取 first_pose 并转换到局部坐标系
        first_pose, _ = self._load_poses()  # 假设 _load_poses 返回 first_pose 和 poses
        final_points_local = transform_points(final_points_global, np.linalg.inv(first_pose))

        # 打印点云保留比例
        retained_ratio = final_points_global.shape[0] / total_initial_points * 100
        # print(f"最终点云数目比上总点云数目: {retained_ratio:.2f}%")
        return final_points_local, final_colors

    # def read_point_cloud_and_color(self, token: str, idx: int, config, min_dist: float = 0.0):
    #     filename = self.nusc.get("sample_data", token)["filename"]
    #     pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, filename))  # (4, N) 点云数据
    #     points = pcl.points.T[:, :3]  # 转换为 (N, 3)
    #
    #     # 初始点云数量
    #     initial_points_count = points.shape[0]
    #
    #     # 限制点云范围
    #     points = points[np.linalg.norm(points, axis=1) <= config.max_range]
    #     points = points[np.linalg.norm(points, axis=1) >= config.min_range]
    #
    #     # 过滤后的点云数量
    #     filtered_points_count = points.shape[0]
    #
    #     # 计算保留百分比
    #     retained_percentage = (filtered_points_count / initial_points_count) * 100
    #     print(
    #         f"初始点云数目: {initial_points_count}, 保留点云数目: {filtered_points_count}, 保留百分比: {retained_percentage:.2f}%")
    #
    #     # 转换到局部坐标系
    #     points = transform_points(points, self.poses[idx]) if config.apply_pose else None
    #
    #     # 初始化颜色为浅灰色
    #     colors = np.full((points.shape[0], 3), 200)  # 浅灰色 (200, 200, 200)
    #
    #     return points.astype(np.float64), colors.astype(np.int32)

    def _load_poses(self):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.nusc.get("sample_data", lidar_token)
            cs_record_lid = self.nusc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
            ep_record_lid = self.nusc.get("ego_pose", sd_record_lid["ego_pose_token"])

            car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            )
            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            )

            poses[i:, :] = pose_car @ car_to_velo

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses
        return first_pose, poses

    def _get_scene_token(self, split_logs: List[str]) -> str:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = [s["token"] for s in self.nusc.scene if s["name"] == self.scene_name][0]
        scene = self.nusc.get("scene", scene_tokens)
        log = self.nusc.get("log", scene["log_token"])
        return scene["token"]

    def _get_lidar_tokens(self, scene_token: str) -> List[str]:
        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get("sample_data", start_sample_rec["data"][self.lidar_name])
        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = []
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get("sample_data", cur_sd_rec["next"])
            sd_tokens.append(cur_sd_rec["token"])
        return sd_tokens