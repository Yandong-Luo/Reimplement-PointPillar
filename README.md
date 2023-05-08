### Reimplement PointPillar on KITTI Test and GEM Autonomous Vehicle

3D object detection is a fundamental task in autonomous driving, robotics, and computer vision applications. It aims to identify and locate objects in 3D space using various sensor data, such as LiDAR point clouds, images, or depth maps. Accurate and efficient 3D object detection is crucial for the safe and effective operation of autonomous systems, as it provides essential information for decision-making, path planning, and navigation. In this project, we reimplement the PointPillars algorithm to realize 3D object detection based on PyTorch. Recognition types include vehicles, pedestrians, and cyclists. In the experimental part, the algorithm is tested on the GEM vehicle and KITTI dataset.

#### Result

##### KITTI ROS TEST

[![KITTI ROS TEST](https://res.cloudinary.com/marcomontalbano/image/upload/v1683487276/video_to_markdown/images/youtube--HrnisTXsBnI-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/HrnisTXsBnI "KITTI ROS TEST")



##### GEM PointPillars on Parking Lots Environment

[![GEM PointPillars on Parking Lots Environment](https://res.cloudinary.com/marcomontalbano/image/upload/v1683504001/video_to_markdown/images/youtube--esFYqR___zI-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/esFYqR___zI "GEM PointPillars on Parking Lots Environment")

##### GEM PointPillar Detect Vehicle

[![GEM PointPillar Detect Vehicle](https://res.cloudinary.com/marcomontalbano/image/upload/v1683503963/video_to_markdown/images/youtube--R3qUu3Bc-9U-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/R3qUu3Bc-9U "GEM PointPillar Detect Vehicle")

##### GEM PointPillar to Detect Pedestrian

[![GEM PointPillar to Detect Pedestrian](https://res.cloudinary.com/marcomontalbano/image/upload/v1683504073/video_to_markdown/images/youtube--eXONjJd3fH0-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/eXONjJd3fH0 "GEM PointPillar to Detect Pedestrian")

#### GEM_ROS_WS

GEM_ros_ws this folder is the workspace of this project on GEM vehicle. This is a ros workspace. The File structure of this folder is shown as follows:

```
.
├── GEM_ros_ws
│   └── src
│       ├── basic_launch
│       │   ├── CMakeLists.txt
│       │   ├── launch
│       │   ├── package.xml
│       │   └── README.md
│       ├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
│       ├── hardware_drivers
│       │   ├── geonav_transform
│       │   ├── novatel_gps_driver
│       │   ├── README.md
│       │   └── zed-ros-wrapper
│       ├── perception
│       │   ├── autoware_msgs
│       │   ├── detected_objects_visualizer
│       │   ├── pointpillars_ros
│       │   └── pub_dataset
│       ├── platform_launch
│       │   ├── CMakeLists.txt
│       │   ├── launch
│       │   ├── package.xml
│       │   └── rviz
│       ├── README.md
│       └── vehicle_drivers
│           ├── gem_gnss_control
│           ├── gem_pacmod_control
│           └── gem_visualization
```

This part of the main work focuses on the content under the perception folder. 

- **autoware_msgs:** I made a secondary development of the  package to make it suitable for our recognition results.
- **pointpillars_ros:** This package is used to predict 3D objects by loading the pre-trained model and subscribing to the point cloud data of VLP32 carried by GEM
- **pub_dataset:** It is used to publish the KITTI data set in the form of ros, and combine pointpillar_ros to realize the test in the KITTI data set.

##### Run

```
roslaunch basic_launch pointpillars.launch
```

And the, open another terminate and enter the anaconda environment.

```
rosrun pointpillars_ros pointpillars_node.py
```

This workspace works for GEM vehicel.

#### PointPillars

PointPillars this folder works for implement PointPillars algorithm and training. The file structure is shown as follows:

```
.
├── checkpoints
├── data
├── data_augment.py
├── data_process.py
├── kitti_util.py
├── models
├── ops
├── PointPillars.md
├── pre_process.py
├── __pycache__
├── rename.sh
├── summary
├── train.py
├── trial_1_checkpoints
├── trial_1_summary
└── verify
```

- **convert-lyft-dataset-to-kitty.ipynb:** Since our data set comes from lyft, but in the process of processing, we use the format of the kitti data set to complete. This file is used for conversion between the two dataset formats.

- **rename.sh:** After the data set is converted, the file name is too long, so I use this script file to complete the batch renaming.
- **verify folder:** The visualization results of each processing are stored in the ipynb file for verification, including data enhancement. Only when that part of the file is complete will the real implementation be done in the form of a .py file.
- **pre_process.py:** Preprocess the dataset and package it into a pkl file to reduce the access operations to the dataset file. The pkl file stores the path of the point cloud file, the path of the image, camera parameters, labels, conversion between radar and camera.
- **data_process.py:** For data processing, including reading the data in the dataset, some functions in the pre_process.py file are also mainly placed in it.
- **data_augmentation.py:** The class that stores the data enhancement, first accesses the pkl file generated by the preprocessing, and then performs data enhancement processing on it.
- **train.py:** This file is used to train the entire PointPillars project.

- **checkpoints:** Record the model saved during training
- **summary:** Store some information during the training process, including loss, learning rate, etc. Used in tensorboard for visualization.

##### Run

```
python train --dataset_root='root path of dataset'
```

