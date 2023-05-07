#!/usr/bin/env python

# -*- coding: utf-8 -*-
import rospy
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image,PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from geometry_msgs.msg import Quaternion, Point
import tf

import sys
import os
import time
import argparse
import numpy as np
import math
import ros_numpy
import torch
from sensor_msgs import point_cloud2

# 将PointPillars的环境导入进来
path_model = "/home/gem/cs498 final project/PointPillars"                    # 需要修改
sys.path.append(path_model)

# from utils import setup_seed, read_points, read_calib, read_label, \
#     keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
#     vis_img_3d, bbox3d2corners_camera, points_camera2image, \
#     bbox_camera2lidar, bbox3d2corners
# from model import PointPillars

from data_process import keep_bbox_from_lidar_range, bbox3d2corners
from models.pointpillars import PointPillars

class pointPillars_detector():
    def __init__(self):

        # self.bridge = CvBridge()

        #######################################################
        ################# PointPillars ########################
        #######################################################

        # 简化版
        self.CLASSES = {
        'pedestrian': 0, 
        'bicycle': 1, 
        'car': 2,
        # 'truck': 3,
        # 'bus': 4,
        # 'other_vehicle': 5,
        # 'emergency_vehicle': 6,
        # 'motorcycle':7,
        # 'animal':8
        }
        self.LABEL2CLASSES = {v:k for k, v in self.CLASSES.items()}
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

        if not args.no_cuda:
            print("Using Cuda Now!!!")
            self.model = PointPillars(num_classes=len(self.CLASSES)).cuda()
            self.model.load_state_dict(torch.load(args.ckpt)['net'])
        else:
            self.model = PointPillars(num_classes=len(self.CLASSES))
            self.model.load_state_dict(
                torch.load(args.ckpt, map_location=torch.device('cpu')))
        
        if os.path.exists(args.calib_path):
            self.calib_info = read_calib(args.calib_path)
        else:
            self.calib_info = None
            print("Could not load the calib info")
        
        # NOTE
        # 方案一：不采用topic的方式订阅
        # self.sub_left_color_image = rospy.Subscriber('/kitti/cam_color_left', Image, self.img_callback, queue_size=1)
        # self.sub_right_color_image = rospy.Subscriber('/kitti/cam_color_right', Image, self.img_callback, queue_size=1)
        # self.sub_left_gray_image = rospy.Subscriber('/kitti/cam_gray_left', Image, self.img_callback, queue_size=1)
        # self.sub_right_gray_image = rospy.Subscriber('/kitti/cam_gray_right', Image, self.img_callback, queue_size=1)

        self.sub_point_cloud = rospy.Subscriber('/lidar1/velodyne_points',PointCloud2, self.pointcloud_callback,queue_size=1)
        
        # 方案二：同步订阅
        # 左边彩色就够了
        # self.sub_left_color_image = message_filters.Subscriber('/front_single_camera/image_raw', Image)# 仿真的图像
        # self.sub_left_color_image = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)# gem的图像

        # self.sub_left_gray_image = message_filters.Subscriber('/kitti/cam_gray_left', Image)#订阅第二个话题，左边摄像头灰色图像
        # self.sub_right_color_image = message_filters.Subscriber('/kitti/cam_color_right', Image)#订阅第三个话题，右边摄像头彩色图像
        # self.sub_right_gray_image = message_filters.Subscriber('/kitti/cam_gray_right', Image)#订阅第四个话题，右边摄像头灰色图像

        # self.sub_point_cloud = message_filters.Subscriber('/velodyne_points',PointCloud2)    # 仿真的的点云数据
        # self.sub_point_cloud = message_filters.Subscriber('/lidar1/velodyne_points',PointCloud2)    # gem上面的点云数据

        self.pub_objects = rospy.Publisher('/detection/lidar_detector/objects', DetectedObjectArray, queue_size=1)

        # # 1 is the size of queue and 1 is the time in sec to consider for aprox:
        # sync = message_filters.ApproximateTimeSynchronizer([self.sub_left_color_image, self.sub_point_cloud], 1,1)  
        # sync.registerCallback(self.multi_callback)
    
    # 仅接收点云的回调函数
    def pointcloud_callback(self, lidar_data):
        # Convert PointCloud2 data to numpy
        # pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_data).astype(np.float32)
        pc = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_data)
        pc = np.asarray(pc.tolist()).astype(np.float32)[:,:4]
        pc[:,3] = 0
        pc = self.point_range_filter(pc)
        pc_torch = torch.from_numpy(pc)

        self.model.eval()
        with torch.no_grad():
            if not args.no_cuda:
                pc_torch = pc_torch.cuda()
            # print("来自于")
            result_filter = self.model(batched_pts=[pc_torch], 
                                mode='test')[0]
        if result_filter is None:
            print("Nothing was detected")
            return
        result_filter = keep_bbox_from_lidar_range(result_filter, self.pcd_limit_range)
        # print(len(result_filter))
        if 'lidar_bboxes' not in result_filter or 'labels' not in result_filter or 'scores' not in result_filter:
            return
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']

        self.pubDetectedObject(lidar_bboxes,labels, lidar_data.header)

    # 同步订阅的回调函数
    def multi_callback(self, cam_data, lidar_data):
        try:
            # Convert a ROS image message into an OpenCV image
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(cam_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        
        raw_img = cv_image.copy()
        
        # Convert PointCloud2 data to numpy
        # pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_data).astype(np.float32)
        # print(lidar_data)
        pc = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_data)
        pc = np.asarray(pc.tolist()).astype(np.float32)[:,:4]
        pc = self.point_range_filter(pc)
        pc_torch = torch.from_numpy(pc)

        self.model.eval()
        with torch.no_grad():
            if not args.no_cuda:
                pc_torch = pc_torch.cuda()
            # print("来自于")
            result_filter = self.model(batched_pts=[pc_torch], 
                                mode='test')[0]
        
        if self.calib_info is not None and raw_img is not None:
            tr_velo_to_cam = self.calib_info['Tr_velo_to_cam'].astype(np.float32)
            # print(tr_velo_to_cam)
            r0_rect = self.calib_info['R0_rect'].astype(np.float32)
            P2 = self.calib_info['P2'].astype(np.float32)

            image_shape = raw_img.shape[:2]
            result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)
        
        if result_filter is None:
            print("Nothing was detected")
            return
        result_filter = keep_bbox_from_lidar_range(result_filter, self.pcd_limit_range)
        # print(len(result_filter))
        
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']

        self.pubDetectedObject(lidar_bboxes,labels, lidar_data.header)

    # 根据雷达的范围再过滤一次点云
    def point_range_filter(self, pts, point_range=[0, -7, -3, 20,7, 1]):
        '''
        data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
        point_range: [x1, y1, z1, x2, y2, z2]
        '''
        flag_x_low = pts[:, 0] > point_range[0]
        flag_y_low = pts[:, 1] > point_range[1]
        flag_z_low = pts[:, 2] > point_range[2]
        flag_x_high = pts[:, 0] < point_range[3]
        flag_y_high = pts[:, 1] < point_range[4]
        flag_z_high = pts[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        pts = pts[keep_mask]
        return pts 
    
    # 发布标记方框到visualizer中进行可视化
    def pubDetectedObject(self, lidar_bboxes,labels, msg_header):
        objectsArray = DetectedObjectArray()
        objectsArray.header = msg_header
        num_objects =  len(lidar_bboxes)
        corners = bbox3d2corners(lidar_bboxes)
        # print(lidar_bboxes)
        for i in range(num_objects):
            cur_object = DetectedObject()
            cur_object.header = msg_header
            # print(msg_header.frame_id)
            cur_object.valid = True
            cur_object.pose_reliable = True

            cur_object.pose.position.x = lidar_bboxes[i][0]
            cur_object.pose.position.y = lidar_bboxes[i][1]
            cur_object.pose.position.z = lidar_bboxes[i][2]

            yaw = lidar_bboxes[i][6]
            yaw += math.pi/2

            yaw = math.atan2(math.sin(yaw),math.cos(yaw))

            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            cur_object.pose.orientation = Quaternion(*q)

            cur_object.dimensions.x = lidar_bboxes[i][3]
            cur_object.dimensions.y = lidar_bboxes[i][4]
            cur_object.dimensions.z = lidar_bboxes[i][5]
            
            for j in range(8):
                point = Point()
                # print(corners[i])
                point.x = corners[i][j][0].astype(np.float32)
                # print(point.x)
                point.y = corners[i][j][1].astype(np.float32)
                point.z = corners[i][j][2].astype(np.float32)
                cur_object.corners.append(point)

            # cur_object.corners = 
            
            # print(type(str(self.LABEL2CLASSES[tmp])))
            cur_object.label = self.LABEL2CLASSES[labels[i]]

            objectsArray.objects.append(cur_object)
        
        self.pub_objects.publish(objectsArray)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='/home/gem/cs498 final project/PointPillars/checkpoints/epoch_160.pth', help='your checkpoint for kitti')               #需要修改
    # parser.add_argument('--pc_path', help='your point cloud path')
    # parser.add_argument('--calib_path', default='/home/chris/KITTI/testing/calib/000008.txt', help='your calib file path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    # parser.add_argument('--gt_path', default='', help='your ground truth path')
    # parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    # init args
    rospy.init_node('pointpillars_node', anonymous=True)
    ppd = pointPillars_detector()

    

    rospy.spin()