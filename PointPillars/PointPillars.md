### PointPillars

#### KITTI解释

https://blog.csdn.net/wuchaohuo724/article/details/115470127

https://blog.csdn.net/qq_37591788/article/details/120396042

#### 预处理

将training、testing文件夹里每个文件里的内容记录在.pkl文件里，pkl文件里的索引统一记录了以下内容：

- 激光雷达的路径：velodyne_path
- 图像：包含图像尺寸、文件路径、索引
- 标定文件的所有内容：相机内外参，camera到lidar的坐标变换
- 移除掉图像视野以外的点云数据
- 对training部分的数据，还需要根据标签计算难易等级，和获取在box范围内的点云数量

#### 数据增强

读取pkl文件里的数据，重采样

- 重采样

  采样gt bbox并将其复制到当前帧的点云，因为当前帧点云中objects(gt_bboxes)可能比较少, 不利于训练; 因此从Car, Pedestrian, Cyclist的database数据集中随机采集一定数量的bbox及inside points。包含碰撞检测

- box和point cloud的水平翻转
- 旋转、缩放、平移

- shuffle

#### 验证

在ipynb中进行可视化验证，py文件用于执行

这个是使用kitti数据集的版本