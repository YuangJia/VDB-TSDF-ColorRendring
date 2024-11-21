import pyopenvdb as openvdb
from nuscene_utils import get_all_active_voxels_and_values

# 创建边界框的方式
min_coord = openvdb.Coord(0, 0, 0)  # 最小坐标
max_coord = openvdb.Coord(10, 10, 10)  # 最大坐标

# 通过 fill() 方法直接填充值
tsdf_grid = openvdb.FloatGrid()
for x in range(min_coord.x(), max_coord.x() + 1):
    for y in range(min_coord.y(), max_coord.y() + 1):
        for z in range(min_coord.z(), max_coord.z() + 1):
            tsdf_grid.setValue(openvdb.Coord(x, y, z), 0.5)

# 获取所有体素和值
voxel_data = get_all_active_voxels_and_values(tsdf_grid)
for coord, value in voxel_data:
    print(f"Voxel: {coord}, TSDF Value: {value}")
