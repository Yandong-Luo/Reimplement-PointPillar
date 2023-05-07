#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
from cv_bridge import CvBridge
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image,PointCloud2,PointField
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf import transformations
# import ros_numpy
BASE_PATH = "/home/chris/ros_ws/kitti_ws/2011_09_26/2011_09_26_drive_0005_sync/"    


# 绘制摄像头角度范围
def publish_cam_angle_line_fun(cam_angle_line_pub):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    # 每个显示的marker都需要不一样的id，否则会覆盖
    marker.id = 0
    marker.type = Marker.LINE_STRIP# 直线
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()#永久显示

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0 # 透明度
    marker.scale.x = 0.2 # 线粗细

    # 这边点的数据主要看官方提供的位置图
    # 会在每两个连续点之间画一条线0-1，1-2。。。
    marker.points = []
    marker.points.append(Point(10,-10,0)) 
    marker.points.append(Point(0,0,0))
    marker.points.append(Point(10,10,0))

    cam_angle_line_pub.publish(marker)

# 加载三维车辆模型到rviz中显示
def publish_car_model(car_model_pub):
    mech_marker = Marker()
    mech_marker.header.frame_id = "map"
    mech_marker.header.stamp = rospy.Time.now()
    # id必须不一样
    mech_marker.id = -1
    mech_marker.type = Marker.MESH_RESOURCE
    mech_marker.action = Marker.ADD
    mech_marker.lifetime = rospy.Duration()
    mech_marker.mesh_resource = "package://pub_dataset/meshes/Car.dae"# .dae模型文件的相对地址

    # 位置主要看官方提供的位置图
    # 因为自车的velodyne激光雷达相对于map的位置是（0，0，0），看设备安装图上velodyne的位置是（0，0，1.73），显而易见，在rviz中自车模型位置应该是（0，0，-1.73）
    mech_marker.pose.position.x = 0.0
    mech_marker.pose.position.y = 0.0
    mech_marker.pose.position.z = -1.73

    # 这边和原作者的代码不一样，因为tf结构发生了一些改变，四元数需要自己根据模型做响应的调整，笔者这边是调整好的模型四元数
    q = transformations.quaternion_from_euler(np.pi,np.pi,-np.pi/2.0)
    mech_marker.pose.orientation.x = q[0]
    mech_marker.pose.orientation.y = q[1]
    mech_marker.pose.orientation.z = q[2]
    mech_marker.pose.orientation.w = q[3]

    mech_marker.color.r = 1.0
    mech_marker.color.g = 1.0
    mech_marker.color.b = 1.0
    mech_marker.color.a = 1.0 # 透明度

    # 设置车辆大小
    mech_marker.scale.x = 1.0 
    mech_marker.scale.y = 1.0 
    mech_marker.scale.z = 1.0

    car_model_pub.publish(mech_marker)

if __name__ == "__main__":
    rospy.init_node("kitti_node",anonymous=True)
    cam_color_left_pub  = rospy.Publisher("/kitti/cam_color_left", Image, queue_size=10)
    # cam_color_right_pub = rospy.Publisher("/kitti/cam_color_right", Image, queue_size=10)
    # cam_gray_left_pub   = rospy.Publisher("/kitti/cam_gray_left", Image, queue_size=10)
    # cam_gray_lright_pub = rospy.Publisher("/kitti/cam_gray_right", Image, queue_size=10)
    pcl_pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=10)
    
    cam_angle_line_pub = rospy.Publisher("/kitti/cam_angle",Marker,queue_size=10)

    car_model_pub = rospy.Publisher("/kitti/car_model", Marker, queue_size=10)

    bridge = CvBridge()

    rate = rospy.Rate(10)
    num = 1
    while not rospy.is_shutdown():
        img = cv2.imread(os.path.join(BASE_PATH, "image_02/data/%010d.png"%num))
        # 使用numpy模块读取bin文件的数据
        point_cloud =np.fromfile(os.path.join(BASE_PATH, "velodyne_points/data/%010d.bin"%num), dtype=np.float32).reshape(-1,4)

        # Get current time as header of stamp
        now = rospy.Time.now()

        # 使用cvbridge模块将cv数据转换为msg数据格式
        image_message = bridge.cv2_to_imgmsg(img,"bgr8")
        image_message.header.stamp = now
        image_message.header.frame_id = 'map'

        cam_color_left_pub.publish(image_message)
        
        # 点云数据处理与发送
        header = Header()
        header.stamp = now
        header.frame_id = 'map'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        pcl2_msg = pcl2.create_cloud(header, fields, point_cloud)
        pcl_pub.publish(pcl2_msg)
        


        # publish_cam_angle_line_fun(cam_angle_line_pub)
        publish_car_model(car_model_pub)
        # rospy.loginfo("kitti published")
        rate.sleep()
        num+=1
        num%=154