import numpy as np
import argparse
import os
import sys
import cv2

import data_process
from rich.progress import track
import time

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

def get_all_files_name_in_folder(folder_path):
    file_names = []
    # List all files and directories in the folder
    for entry in os.listdir(folder_path):
        # Check if the entry is a file
        if os.path.isfile(os.path.join(folder_path, entry)):
            # Get the file name without extension
            file_name_without_ext, ext = os.path.splitext(entry)
            if ext != '.txt':
                continue
            file_names.append(file_name_without_ext)

    return file_names


# generate the .pkl file to storage the information from dataset
# 生产.pkl文件储存数据输入
def pkl_process(dataset_root,generate_label, db, prefix, data_type):
    # 根据是否需要处理label来判断当前的处理类型是属于training还是testing
    split = 'training' if generate_label else 'testing'
    # 根据calib文件夹底下的文件名来确定当处理的是哪一时刻的数据
    path = os.path.join(dataset_root,split,'calib')
    # ids = get_all_files_name_in_folder(path)

    ids_file = os.path.join(current_path, 'ImageSets', f'{data_type}.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]

    print(f"Processing {data_type} data..")
    all_info_dict = {}
    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(dataset_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)
    for id in track(ids):
        cur_info_dict = {}
        lidar_path = os.path.join(dataset_root, split, 'velodyne', f'{id}.bin')
        img_path = os.path.join(dataset_root, split, 'image_2', f'{id}.png')
        calib_path = os.path.join(dataset_root, split, 'calib', f'{id}.txt') 

        # process image part
        cur_img = cv2.imread(img_path)
        cur_info_dict['image'] = {
            'image_path': os.path.sep.join(img_path.split(os.path.sep)[-3:]),
            'image_shape': cur_img.shape[:2]    # just record the height and weight
        }

        # process calibration part
        calib_dict = data_process.process_calibration(calib_path)
        cur_info_dict['calibration'] = calib_dict

        # Point cloud part
        # 获取相对路径
        cur_info_dict['velodyne_path'] = os.path.sep.join(lidar_path.split(os.path.sep)[-3:])

        # optimized point cloud
        init_points = data_process.get_lidar_points(lidar_path)  # N by 4 matrix
        reduced_lidar_points = data_process.remove_outside_pointcloud(init_points,calib_dict,cur_img.shape[:2])

        saved_reduced_path = os.path.join(dataset_root, split, 'velodyne_reduced')
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin')
        # save optimizaed point cloud
        data_process.write_lidar_points(reduced_lidar_points, saved_reduced_points_name)

        if generate_label:
            label_path = os.path.join(dataset_root, split, 'label_2', f'{id}.txt')
            # 读取标签中的信息，包括遮掩，box位置、截断
            annotation_dict = data_process.read_label(label_path)
            # print(annotation_dict)

            # valid_index = data_process.get_valid_index(annotation_dict)
            # if np.sum(valid_index) == 0:
            #     continue

            # annotation_dict = {key: value[valid_index] for key, value in annotation_dict.items()}
            # 根据标签判断难度
            # annotation_dict['difficulty'] = data_process.judge_difficulty(annotation_dict)
            annotation_dict['difficulty'] = data_process.judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = data_process.get_points_num_in_bbox(
                points=reduced_lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                Tr_lidar_to_cam=calib_dict['Tr_lidar_to_cam'],
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

            if db:
                indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                    data_process.points_in_bboxes_v2(
                        points=init_points,
                        r0_rect=calib_dict['R0_rect'].astype(np.float32), 
                        Tr_lidar_to_cam=calib_dict['Tr_lidar_to_cam'].astype(np.float32),
                        dimensions=annotation_dict['dimensions'].astype(np.float32),
                        location=annotation_dict['location'].astype(np.float32),
                        rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                        name=annotation_dict['name'])
                for j in range(n_valid_bbox):
                    db_points = init_points[indices[:, j]]
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                    data_process.write_lidar_points(db_points, db_points_saved_name)

                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j],
                        'num_points_in_gt': len(db_points), 
                    }
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
        
        all_info_dict[int(id)] = cur_info_dict

    saved_path = os.path.join(dataset_root, f'{prefix}_infos_{data_type}.pkl')
    data_process.write_pickle(all_info_dict, saved_path)
    if db:
        saved_db_path = os.path.join(dataset_root, f'{prefix}_dbinfos_train.pkl')
        data_process.write_pickle(kitti_dbinfos_train, saved_db_path)
    return all_info_dict

def main(args):
    dataset_root = args.dataset_root
    prefix = args.prefix

    train_infos_dict = pkl_process(dataset_root,generate_label=True,db=True,prefix=prefix,data_type='train')

    val_infos_dict = pkl_process(dataset_root,generate_label=True,db=False,prefix=prefix,data_type='val')

    trainval_infos_dict = {**train_infos_dict, **val_infos_dict}
    saved_path = os.path.join(dataset_root, f'{prefix}_infos_trainval.pkl')
    data_process.write_pickle(trainval_infos_dict, saved_path)

    pkl_process(dataset_root,generate_label=False,db=False,prefix=prefix,data_type='test')
    

if __name__ == '__main__':
    # 接收从终端来的参数
    parser = argparse.ArgumentParser()

    # the path of dataset
    #数据集的绝对路径
    parser.add_argument('--dataset_root', \
                        # default='/media/chris/Workspace/Dataset/3d-object-detection-for-autonomous-vehicles/kitti_format/',
                        default='/media/chris/Workspace/Dataset/kitti/',
                        # default='/media/chris/Workspace/Dataset/3d-object-detection-one_scene/kitti_format',
                        help='the root path of dataset')
    # 文件名前缀
    parser.add_argument('--prefix', \
                        default='lyft',
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()
    main(args)