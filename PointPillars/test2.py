import numpy as np
import os
import pickle
import data_process
import open3d as o3d
import copy
import numba
import random

# dataset_root = '/media/chris/Workspace/Dataset/3d-object-detection-for-autonomous-vehicles/kitti_format/'
# dataset_root = '/media/chris/Workspace/Dataset/kitti'
dataset_root = '/media/chris/Workspace/Dataset/3d-object-detection-one_scene/kitti_format'
# dataset_root = '/media/chris/Workspace/Dataset/lyft'
identifier = 'train'

# 读取pkl文件里的数据
data_content = data_process.read_pickle(os.path.join(dataset_root,f'lyft_infos_{identifier}.pkl'))
database_content = data_process.read_pickle(os.path.join(dataset_root,f'lyft_dbinfos_train.pkl'))

# data_content = data_process.read_pickle(os.path.join(dataset_root,f'kitti_infos_{identifier}.pkl'))
# database_content = data_process.read_pickle(os.path.join(dataset_root,f'kitti_dbinfos_train.pkl'))

# list(self.data_infos.keys())
# key_list = list(data_content.keys())
# print(data_content[key_list[]])
database_content

class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        # print("========================")
        # print(sampled_list)
        self.total_num = len(sampled_list)
        # print("total_num",self.total_num)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        # 打乱顺序
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret
    
class DataSet():
    # CLASSES = {
    #     'pedestrian': 0, 
    #     'truck': 1, 
    #     'car': 2,
    #     'bicycle':3
    #     }
    CLASSES = {
        'pedestrian': 0, 
        'bicycle': 1, 
        'car': 2,
        'truck': 3,
        'bus': 4,
        'other_vehicle': 5
        }
    def __init__(self, dataset_root,identifier) -> None:
        self.dataset_root = dataset_root
        self.identifier = identifier
        # 读取pkl文件内容   read the pkl file
        self.data_content = data_process.read_pickle(os.path.join(dataset_root,f'lyft_infos_{identifier}.pkl'))
        database_content = data_process.read_pickle(os.path.join(dataset_root,'lyft_dbinfos_train.pkl'))
        
        # database_content = self.filter_by_difficulty(database_content)
        # print("==========================================")
        # print(database_content)
        # print("==========================================")
        
        self.key_list = list(self.data_content.keys())

        self.database_sampler = {}
        for class_name in self.CLASSES:
            print(database_content.keys())
            if class_name not in database_content.keys():
                continue
            self.database_sampler[class_name] = BaseSampler(database_content[class_name], True)
        # pass
    
    def filter_by_difficulty(self, database_content):
        # 1. filter_by_difficulty
        # for class_name, v in database_content.items():
            # for item in v:
            #     if item['difficulty'] != -1:
            #         database_content[class_name] = item

        for k, v in database_content.items():
            database_content[k] = [item for item in v if item['difficulty'] != -1]
        # print(db_infos)
            # difficulty为-1也意味着无效的database
            # db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        # filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        # for cat in self.CLASSES:
        #     filter_thr = filter_thrs[cat]
        #     db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return database_content

    def __getitem__(self, key):
        current_data = self.data_content[self.key_list[key]]
        # print("initial_data",current_data)
        data_dict = self.generate_data(current_data)
        # init_data = data_dict.deepcopy()
        init_data = copy.deepcopy(data_dict)
        # print(init_pc)
        data_dict = self.data_augment_main(data_dict)
        # 
        # data_dict = self.rotation_data(data_dict, rotation_range)
        # print(data_dict)

        return init_data, data_dict
    
    def remove_invalid_data(self, annos):
        # print(annos['bbox'])
        # keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        # 移除掉无效的data, bbox的尺寸为-1则无效
        valid_rows_index = np.all(annos['bbox'] != -1, axis=1)        # 这一步错了
        for k, v in annos.items():
            annos[k] = v[valid_rows_index]
        return annos
    
    # 生成用于数据增强的数据
    def generate_data(self, data_info):
        """_summary_

        Args:
            data_info (_type_): current data info from pkl.file

        Returns:
            dict: the input data
        """
        # 获取相对路径
        reduced_pc_filename = data_info['velodyne_path'].replace('velodyne','velodyne_reduced') # 用reduced point cloud
        # print(reduced_pc_filename)
        # 绝对路径+相对路径
        reduced_pc_path = os.path.join(self.dataset_root, reduced_pc_filename)
        pc = data_process.get_lidar_points(reduced_pc_path)

        # get the image
        img = data_info['image']
        
        # print(pc)

        # get the calibration info
        calib = data_info['calibration']
        calib_R0_rect = calib['R0_rect']
        calib_tr_velo_to_cam = calib['Tr_lidar_to_cam']
        
        # get the annotation info
        annos = data_info['annos']
        # 移除掉无效的data, bbox的尺寸为-1则无效
        annos = self.remove_invalid_data(annos)
        # print("annos_info!!!!!!!!!!!!!",annos)

        # 类别
        annos_name = annos['name']
        # box位置
        annos_box_location = annos['location']
        # box尺寸
        annos_box_dim = annos['dimensions']
        # rotation
        annos_box_rotation_y = annos['rotation_y']

        

        
        # print(annos)
        
        # 移除掉所有无效的data后不存在数据则pass
        # if len(annos) == 0:
        #     pass
        # print("boxxxxxxxxxxxxxxxxxxxxxxx",annos['bbox'])

        # After get the infomation, we can start to generate the data dict
        # 先将camera的box转成lidar坐标系下的3d box
        ground_truth_bboxes = np.concatenate([annos_box_location, annos_box_dim, annos_box_rotation_y[:, None]], axis=1).astype(np.float32)   #合并信息
        ground_truth_bboxes_3d = data_process.bbox_camera2lidar(ground_truth_bboxes, calib_tr_velo_to_cam, calib_R0_rect)

        # 根据标签隐射到对应的label
        ground_truth_labels = [self.CLASSES.get(name, -1) for name in annos_name]

        # generate the data dict and return
        result = {
            'pc': pc,
            'img': img,
            'gt_labels': ground_truth_labels,
            'gt_names': annos_name,
            'gt_bboxes_3d': ground_truth_bboxes_3d,
            'calib': calib,
            'difficulty': annos['difficulty']
        }
        # print(result)
        return result
    
    ################################################ AUGMENT #######################################################
    def data_augment_main(self,data_dict):
        """_summary_
            数据增强主函数
        Args:
            data_dict (_type_): 初始数据

        Returns:
            data_dict: 数据增强后
        """
        # print("-------------------------------")
        # print(data_dict)
        # print("-------------------------------")
        data_dict = self.database_sample(data_dict,
                         sample_groups=dict(pedestrian=10, bicycle=10, car=15, truck=10, bus=10, other_vehicle=10))
        # print("-------------------------------")
        # print(data_dict['gt_labels'])
        # print("-------------------------------")
        # data_dict = self.random_flip_data(data_dict)

        # rotation_range = [np.radians(-45), np.radians(45)]
        # data_dict = self.rotation_data(data_dict,rotation_range)

        # translation_std = [0,0,0]
        # data_dict = self.translation_data(data_dict,translation_std)

        # scale_ratio_range=[0.95, 1.05]
        # data_dict = self.scale_data(data_dict,scale_ratio_range)

        # limit_pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        # limit_box_range = [0, -39.68, -3, 69.12, 39.68, 1] 
        # data_dict = self.filter_3dbox_pc_by_range(data_dict,limit_pc_range,limit_box_range)
        # data_dict = self.object_range_filter(data_dict,limit_box_range)
        return data_dict

    def rotation_data(self, data_dict, angle_range):
        """_summary_
            对3d box和点云数据进行了旋转
        Args:
            data_dict (_type_): _description_
            angle_range (array): the range of rotation

        Returns:
            _type_: new data_dict
        """
        # point cloud
        pc = data_dict['pc']

        # 3d bbox in lidar frame
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']

        # 在范围里随机选取旋转角
        angle = np.random.uniform(angle_range[0], angle_range[1])
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # 旋转矩阵
        rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                    [-sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
        
        # 点云旋转
        pc[:, :3] = pc[:, :3] @ rotation_matrix.T

        # 3d bbox旋转
        ground_truth_bboxes_3d[:, :3] = ground_truth_bboxes_3d[:, :3] @ rotation_matrix.T
        ground_truth_bboxes_3d[:, 6] += angle

        # Update
        data_dict.update({'gt_bboxes_3d': ground_truth_bboxes_3d})
        data_dict.update({'pc': pc})

        return data_dict

    def translation_data(self, data_dict,translation_std):
        """_summary_

        Args:
            data_dict (_type_): _description_
            translation_std (_type_): _description_

        Returns:
            data_dict: _description_
        """
        # point cloud
        pc = data_dict['pc']

        # 3d bbox in lidar frame
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']

        trans_factor = np.random.normal(scale=translation_std, size=(1, 3))

        ground_truth_bboxes_3d[:, :3] += trans_factor
        pc[:,:3] += trans_factor

        # Update
        data_dict.update({'gt_bboxes_3d': ground_truth_bboxes_3d})
        data_dict.update({'pc': pc})

        return data_dict
    
    def scale_data(self, data_dict, scale_ratio_range):
        """_summary_
            缩放数据
        Args:
            data_dict (_type_): data
            scale_ratio_range (_type_): _description_
        """
        # point cloud
        pc = data_dict['pc']

        # 3d bbox in lidar frame
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']

        scale_fator = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])

        ground_truth_bboxes_3d[:, :6] *= scale_fator
        pc[:, :3] *= scale_fator

        # Update
        data_dict.update({'gt_bboxes_3d': ground_truth_bboxes_3d})
        data_dict.update({'pc': pc})

        return data_dict
    
    def points_shuffle(data_dict):
        """_summary_
            打乱点云数据中point cloud的顺序
        Args:
            data_dict (_type_): _description_

        Returns:
            new data_dict: _description_
        """
        pc = data_dict['pc']
        indices = np.arange(0, len(pc))
        # 打乱索引
        np.random.shuffle(indices)
        pc = pc[indices]
        # Update
        data_dict.update({'pc': pc})
        return data_dict

    def random_flip_data(self, data_dict):
        """_summary_
            随机水平翻转:point cloud水平翻转和3d bboxes水平翻转
        Args:
            data_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 按几率生成翻转与否
        flip_state = np.random.choice([True, False], p=[0.5, 1-0.5])
        if flip_state:
            # point cloud
            pc = data_dict['pc']
            pc[:, 1] = -pc[:, 1] 

            # 3d bbox in lidar frame
            ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']
            ground_truth_bboxes_3d[:,1] = -ground_truth_bboxes_3d[:,1]
            ground_truth_bboxes_3d[:,6] = ground_truth_bboxes_3d[:,6] + np.pi

            # Update
            data_dict.update({'gt_bboxes_3d': ground_truth_bboxes_3d})
            data_dict.update({'pc': pc})
        return data_dict
    
    def filter_3dbox_pc_by_range(self, data_dict, limit_pc_range, limit_box_range):
        """_summary_

        Args:
            data_dict (_type_): _description_
            range_limitation (_type_): _description_

        Returns:
            _type_: _description_
        """
        # point cloud
        pc = data_dict['pc']

        # ground truth box and annotation
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']
        ground_truth_labels = np.array(data_dict['gt_labels'])
        ground_truth_class_name = data_dict['gt_names']
        difficult = data_dict['difficulty']

        pc_mask = self.mask_points_by_range(data_dict, limit_pc_range)

        box_mask = self.mask_boxes_outside_range(data_dict, limit_box_range)

        # print("gt_labels",ground_truth_labels.dtype)
        # print("keep_mask",box_mask.dtype)

        # print(box_mask)

        # mask = pc_mask & box_mask

        # Based on the mask to filter
        pc = pc[pc_mask]
        ground_truth_bboxes_3d = ground_truth_bboxes_3d[box_mask]
        ground_truth_labels = ground_truth_labels[box_mask]
        ground_truth_class_name = ground_truth_class_name[box_mask]
        difficult = difficult[box_mask]

        # Update
        data_dict.update({'pc': pc})
        data_dict.update({'gt_bboxes_3d': ground_truth_bboxes_3d})
        data_dict.update({'gt_labels': ground_truth_labels})
        data_dict.update({'gt_names': ground_truth_class_name})
        data_dict.update({'difficulty': difficult})
    
        return data_dict
    
    def mask_points_by_range(self, data_dict, limit_pc_range):
        # point cloud
        pc = data_dict['pc']
        
        flag_x_low = pc[:, 0] >= limit_pc_range[0]
        flag_y_low = pc[:, 1] >= limit_pc_range[1]
        flag_z_low = pc[:, 2] >= limit_pc_range[2]
        flag_x_high = pc[:, 0] <= limit_pc_range[3]
        flag_y_high = pc[:, 1] <= limit_pc_range[4]
        flag_z_high = pc[:, 2] <= limit_pc_range[5]

        mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high

        return mask
    
    def mask_boxes_outside_range(self, data_dict, limit_box_range):
        """_summary_
            获取过滤3d box的mask
        Args:
            data_dict (_type_): _description_
            limit_box_range (_type_): 3d box的范围

        Returns:
            bool mask: 用于判断哪些box该保留
        """
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']

        # make sure 3d box 
        flag_x_low = ground_truth_bboxes_3d[:, 0] > limit_box_range[0]
        flag_x_high = ground_truth_bboxes_3d[:, 0] < limit_box_range[3]
        flag_y_low = ground_truth_bboxes_3d[:, 1] > limit_box_range[1]
        flag_y_high = ground_truth_bboxes_3d[:, 1] < limit_box_range[4]
        mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high

        return mask
    
    def object_range_filter(self, data_dict, object_range):
        '''
        data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
        point_range: [x1, y1, z1, x2, y2, z2]
        '''
        gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], np.array(data_dict['gt_labels'])
        gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']

        # bev filter
        flag_x_low = gt_bboxes_3d[:, 0] > object_range[0]
        flag_y_low = gt_bboxes_3d[:, 1] > object_range[1]
        flag_x_high = gt_bboxes_3d[:, 0] < object_range[3]
        flag_y_high = gt_bboxes_3d[:, 1] < object_range[4]
        keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high
        

        gt_bboxes_3d, gt_labels = gt_bboxes_3d[keep_mask], gt_labels[keep_mask]
        gt_names, difficulty = gt_names[keep_mask], difficulty[keep_mask]
        # gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)
        data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
        data_dict.update({'gt_labels': gt_labels})
        data_dict.update({'gt_names': gt_names})
        data_dict.update({'difficulty': difficulty})
        return data_dict
    
    def database_sample(self, data_dict, sample_groups):
        ground_truth_bboxes_3d = data_dict['gt_bboxes_3d']
        ground_truth_labels = np.array(data_dict['gt_labels'])
        ground_truth_class_name = data_dict['gt_names']
        difficulty = data_dict['difficulty']
        img = data_dict['img']
        calibration = data_dict['calib']
        pc = data_dict['pc']

        avoid_coll_boxes = copy.deepcopy(ground_truth_bboxes_3d)
        sampled_pts, sampled_names, sampled_labels = [], [], []
        sampled_bboxes, sampled_difficulty = [], []
        for class_name, num in sample_groups.items():
            # 当前训练集中已经有class_name这个类别的个数
            tmp = np.sum(ground_truth_class_name == class_name)
            remain_samples = num - tmp 
            if remain_samples <= 0:
                # 已经满足目标个数时
                continue
            
            # 依然不够则继续从别的database里获取
            if class_name not in self.database_sampler.keys():
                print(class_name,"not in key")
                continue
            # print(class_name,"in key")
            additional_samples = self.database_sampler[class_name].sample(remain_samples)
            additional_samples_bboxes = np.array([item['box3d_lidar'] for item in additional_samples], dtype=np.float32)
            # additional_samples_bboxes = []
            # for item in additional_samples:
            #     additional_samples_bboxes.append(item['box3d_lidar'])
            # additional_samples_bboxes = np.array(additional_samples_bboxes)

            # 3d bbox的碰撞检测
            '''
            在实际情况中, gt_bboxes是没有overlap的(若存在overlap, 就表示有碰撞了); 
            因此需要将采样的bboxes先与当前帧点云中的gt_bboxes进行碰撞检测, 
            通过碰撞检测的bboxes和对应labels加到gt_bboxes_3d, gt_labels,同时把当前帧点云中位于这些采样bboxes内的点删除掉, 
            替换成采样的bboxes(包括inside points).
            '''
            avoid_coll_boxes_bv_corners = self.bbox3d2bevcorners(avoid_coll_boxes)
            sampled_cls_bboxes_bv_corners = self.bbox3d2bevcorners(additional_samples_bboxes)
            coll_query_matrix = np.concatenate([avoid_coll_boxes_bv_corners, sampled_cls_bboxes_bv_corners], axis=0)
            coll_mat = data_process.box_collision_test(coll_query_matrix, coll_query_matrix)
            n_gt, tmp_bboxes = len(avoid_coll_boxes_bv_corners), []
            for i in range(n_gt, len(coll_mat)):
                if any(coll_mat[i]):
                    coll_mat[i] = False
                    coll_mat[:, i] = False
                else:
                    cur_sample = additional_samples[i - n_gt]
                    pt_path = os.path.join(self.dataset_root, cur_sample['path'])
                    sampled_pts_cur = data_process.get_lidar_points(pt_path)
                    sampled_pts_cur[:, :3] += cur_sample['box3d_lidar'][:3]
                    sampled_pts.append(sampled_pts_cur)
                    sampled_names.append(cur_sample['name'])
                    sampled_labels.append(self.CLASSES[cur_sample['name']])
                    sampled_bboxes.append(cur_sample['box3d_lidar'])
                    tmp_bboxes.append(cur_sample['box3d_lidar'])
                    sampled_difficulty.append(cur_sample['difficulty'])
            if len(tmp_bboxes) == 0:
                tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
            else:
                tmp_bboxes = np.array(tmp_bboxes)
            avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        
        # merge sampled database
        # remove raw points in sampled_bboxes firstly
        pc = data_process.remove_pts_in_bboxes(pc, np.stack(sampled_bboxes, axis=0))
        # pts = np.concatenate([pts, np.concatenate(sampled_pts, axis=0)], axis=0)
        pc = np.concatenate([np.concatenate(sampled_pts, axis=0), pc], axis=0)
        ground_truth_bboxes_3d = avoid_coll_boxes.astype(np.float32)
        ground_truth_labels = np.concatenate([ground_truth_labels, np.array(sampled_labels)], axis=0)
        ground_truth_class_name = np.concatenate([ground_truth_class_name, np.array(sampled_names)], axis=0)
        difficulty = np.concatenate([difficulty, np.array(sampled_difficulty)], axis=0)
        data_dict = {
            'pc': pc,
            'img': img,
            'gt_labels': ground_truth_labels,
            'gt_names': ground_truth_class_name,
            'gt_bboxes_3d': ground_truth_bboxes_3d,
            'calib': calibration,
            'difficulty': difficulty
        }
        # print(data_dict)
        return data_dict
    
    # Copied from https://github.com/zhulf0804/PointPillars/blob/main/utils/process.py#L96
    def bbox3d2bevcorners(self, bboxes):
        '''
        bboxes: shape=(n, 7)

                    ^ x (-0.5 * pi)
                    |
                    |                (bird's eye view)
        (-pi)  o |
            y <-------------- (0)
                    \ / (ag)
                    \ 
                    \ 

        return: shape=(n, 4, 2)
        '''
        centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]

        # 1.generate bbox corner coordinates, clockwise from minimal point
        bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
        bev_corners = bev_corners[None, ...] * dims[:, None, :] # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

        # 2. rotate
        rot_sin, rot_cos = np.sin(angles), np.cos(angles)
        # in fact, -angle
        rot_mat = np.array([[rot_cos, rot_sin], 
                            [-rot_sin, rot_cos]]) # (2, 2, n)
        rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (N, 2, 2)
        bev_corners = bev_corners @ rot_mat # (n, 4, 2)

        # 3. translate to centers
        bev_corners += centers[:, None, :] 
        return bev_corners.astype(np.float32)
    

# Red, green, blue, cyan
palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]

# Visualize a segmented point cloud
def vis_data(init, aug):
    # pts = np.array(pts).T
    # seg = np.array(seg)
    init_pts = init['pc']
    aug_pts = aug['pc']

    pcds = []
    init_pcd = o3d.geometry.PointCloud()
    init_pcd.points = o3d.utility.Vector3dVector(init_pts[:,:3])
    init_pcd.colors = o3d.utility.Vector3dVector([palette[0]] * len(init_pcd.points))
    pcds.append(init_pcd)

    # aug_pcd = o3d.geometry.PointCloud()
    # aug_pcd.points = o3d.utility.Vector3dVector(aug_pts[:,:3])
    # aug_pcd.colors = o3d.utility.Vector3dVector([palette[2]] * len(aug_pcd.points))
    # pcds.append(aug_pcd)

    o3d.visualization.add_geometry(pcds)

    init_bboxes = init['gt_bboxes_3d']
    print(init_bboxes)
    aug_bboxes = aug['gt_bboxes_3d']

    for i, box in enumerate(init_bboxes):
        b = o3d.geometry.OrientedBoundingBox()
        b.center = box[:3]
        b.extent = box[3:6]
        # with heading
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, box[6]))
        b.rotate(R, b.center)  
        # 2nd method
        #lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
        #                    [0, 4], [1, 5], [2, 6], [3, 7]])
        #colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        #line_set = o3d.geometry.LineSet()
        #line_set.lines = o3d.utility.Vector2iVector(lines_box)
        #line_set.colors = o3d.utility.Vector3dVector(colors)
        #line_set.points = o3d.utility.Vector3dVector(points_3dbox)
        #vis.add_geometry(line_set)
        o3d.visualization.add_geometry(b)


import torch
def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
setup_seed()
data = DataSet(dataset_root,'train')

# 获取初始和增强后的数据
init, aug_data = data[0]
# print("-----------------------------------------")
# print(aug_data)
# print("-----------------------------------------")
# 获取数据集中的点云数据
# aug_pc = aug_data
# init_cloud  = init

vis_data(init, aug_data)