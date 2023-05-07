import numpy as np
import os
import pickle
import numba

# process the calibration file
# 处理标定参数，返回calib_dict包含标定参数
# input: calib_path: the calibration file name
def process_calibration(calib_path):
    """_summary_

    Args:
        calib_path (string): the path of calibration file
    Return:
        calib_dict: 处理好后的标定数据
    """
    with open(calib_path,'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float).reshape(3, 3)
    Tr_lidar_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float).reshape(3, 4)

    # homogeneous matrix

    R0_offset = np.eye(4, dtype=R0_rect.dtype)
    R0_offset[:3,:3] = R0_rect

    offset = np.array([[0, 0, 0, 1]])
    H_P0 = np.concatenate([P0, offset], axis=0)
    H_P1 = np.concatenate([P1, offset], axis=0)
    H_P2 = np.concatenate([P2, offset], axis=0)
    H_P3 = np.concatenate([P3, offset], axis=0)

    Tr_lidar_to_cam = np.concatenate([Tr_lidar_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
    Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict = dict(P0=H_P0, P1=H_P1, P2=H_P2, P3=H_P3,\
                      R0_rect=R0_offset, Tr_lidar_to_cam=Tr_lidar_to_cam,\
                        Tr_imu_to_velo=Tr_imu_to_velo)
    return calib_dict


def get_lidar_points(lidar_file_path):
    """_summary_

    Args:
        lidar_file_path (string): the path of lidar file

    Returns:
        points cloud: return the point cloud from .bin file
    """
    type = os.path.splitext(lidar_file_path)[1] # [0] for the file name, [1] for the file type
    if(type == '.bin'):
        points = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)
        return points
    else:
        print("the type of point cloud file is not '.bin'")
        return None
    
def read_pickle(file_path, filename='.pkl'):
    assert os.path.splitext(file_path)[1] == filename
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_lidar_points(lidar_points, file_path):
    """_summary_

    Args:
        lidar_points (pointcloud): _description_
        file_path (string): _description_

    Raises:
        NotImplementedError: _description_
    """
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError
    
def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    ################################################################
    # names = []
    # for line in lines:
    #     if line[0] == 'motorcycle':
    #         line[0] = 'bicycle'
    #     elif line[0] == 'truck' or line[0] == 'bus' or line[0] == 'other_vehicle' or line[0] == 'emergency_vehicle':
    #         line[0] = 'car'
    #     elif line[0] == 'animal':
    #         line[0] = 'pedestrian'
    #     names.append(line[0])
    # annotation['name'] = np.array(names)
    ############################################################
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float)
    
    return annotation

# box的八个顶点，转成6个面
def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs

def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params

# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def projection_matrix_to_CRT(proj):
    """Split projection matrix.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.
    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T

@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks

def bbox_camera2lidar(bboxes, Tr_lidar_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    Tr_lidar_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ Tr_lidar_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)

def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners

# def points_in_bboxes_v2(points, r0_rect, Tr_lidar_to_cam, dimensions, location, rotation_y, name, bbox):
#     '''
#     points: shape=(N, 4) 
#     Tr_lidar_to_cam: shape=(4, 4)
#     r0_rect: shape=(4, 4)
#     dimensions: shape=(n, 3) 
#     location: shape=(n, 3) 
#     rotation_y: shape=(n, ) 
#     name: shape=(n, )
#     return:
#         indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j. 
#         n_total_bbox: int. 
#         n_valid_bbox: int, not including 'DontCare' 
#         bboxes_lidar: shape=(n_valid_bbox, 7) 
#         name: shape=(n_valid_bbox, )
#     '''
#     n_total_bbox = len(dimensions)
#     # n_valid_bbox = len([item for item in name if item != 'DontCare'])
#     valid_rows_index = np.all(bbox != -1, axis=1)
#     n_valid_bbox = sum(valid_rows_index)
#     # n_valid_bbox = len([item for item in bbox if all(item) != -1])      # 2d box的尺寸不含有-1则该box有效
#     location, dimensions = location[valid_rows_index], dimensions[valid_rows_index]
#     rotation_y, name = rotation_y[valid_rows_index], name[valid_rows_index]
#     bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
#     bboxes_lidar = bbox_camera2lidar(bboxes_camera, Tr_lidar_to_cam, r0_rect)
#     bboxes_corners = bbox3d2corners(bboxes_lidar)
#     group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
#     frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
#     indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
#     return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name

def points_in_bboxes_v2(points, r0_rect, Tr_lidar_to_cam, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j. 
        n_total_bbox: int. 
        n_valid_bbox: int, not including 'DontCare' 
        bboxes_lidar: shape=(n_valid_bbox, 7) 
        name: shape=(n_valid_bbox, )
    '''
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, Tr_lidar_to_cam, r0_rect)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name

def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners

def get_points_num_in_bbox(points, r0_rect, Tr_lidar_to_cam, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4) 
    Tr_lidar_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return: shape=(n, )
    '''
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points, 
            r0_rect=r0_rect, 
            Tr_lidar_to_cam=Tr_lidar_to_cam, 
            dimensions=dimensions, 
            location=location, 
            rotation_y=rotation_y, 
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int)

def remove_outside_pointcloud(pointcloud,calib_dict,img_shape):
    R0_rect = calib_dict['R0_rect']
    Tr_lidar_to_cam = calib_dict['Tr_lidar_to_cam']
    P2 = calib_dict['P2']

    # 分解矩阵
    # C相机内参，R相机参考帧的世界坐标轴方向，T世界坐标系中的相机中心坐标
    C, R, T = projection_matrix_to_CRT(P2)
    img_box = [0,0,img_shape[1],img_shape[0]]   # 图像大小范围构成
    frustum = get_frustum(img_box,C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T  # 3xN

    # 将视锥体从相机坐标变换为激光雷达坐标
    # frustum = np.hstack((frustum.T, np.ones((frustum.shape[1],1))))
    frustum = np.pad(frustum.T[None, ...], ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    # Tr_lidar_to_cam transforms LiDAR points to the camera coordinate system,
    #  and R0_rect aligns the different camera coordinate systems.
    rotation_matrix = np.linalg.inv(R0_rect @ Tr_lidar_to_cam)
    frustum = frustum @ rotation_matrix.T
    frustum = frustum[...,:3]  # 仅保留xyz

    group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(pointcloud[:, :3], frustum_surfaces) # (N, 1)
    pointcloud = pointcloud[indices.reshape([-1])]

    return pointcloud

def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    if 'lidar_bboxes' not in result or 'labels' not in result or 'scores' not in result:
        return {}
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result

# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L661
# calculates the 3D coordinates of the corners of a frustum (a truncated pyramid) in camera coordinates. 
# The frustum is defined by the bounding box in image coordinates, the camera intrinsic matrix, and the near and far clipping distances.
# The function performs the following steps:
# Extracts the focal lengths (fku, fkv) and the principal point coordinates (u0v0) from the intrinsic matrix C.
# Creates an array z_points containing the z-coordinates of the 8 frustum corners (4 for near_clip and 4 for far_clip).
# Extracts the bounding box corners from bbox_image and calculates the corresponding corners in the near and far clipping planes.
# Combines the x, y, and z coordinates of the frustum corners into a single 8x3 NumPy array, ret_xyz.
# # 计算视椎体
def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.
    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.
    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz

# def judge_difficulty(annotation_dict):
#     bbox = annotation_dict['bbox']  # # 四个数分别是xmin、ymin、xmax、ymax（单位：pixel），表示2维边界框的左上角和右下角的坐标。
#     height = bbox[:, 3] - bbox[:, 1]

#     MIN_HEIGHTS = [40, 25, 25]
#     difficultys = []
#     for h in height:
#         difficulty = -1
#         for i in range(2, -1, -1):
#             # box的框高度越大，就越困难
#             if h > MIN_HEIGHTS[i]:
#                 difficulty = i
#         difficultys.append(difficulty)
#     return np.array(difficultys, dtype=np.int)

def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int)

def get_valid_index(annotation_dict):
    # bbox = annotation_dict['bbox']

    # valid_data_index = np.where(np.all(annotation_dict['bbox']!=-1,axis=1))
    valid_data_index = np.all(annotation_dict['bbox'] != -1, axis=1)        # 这一步错了

    # for k, v in annotation_dict.items():
    #         annos[k] = v[valid_rows_index]
    return valid_data_index

@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    '''
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    '''
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox

def remove_pts_in_bboxes(points, bboxes, rm=True):
    '''
    points: shape=(N, 3)
    bboxes: shape=(n, 7)
    return: shape=(N, n), bool
    '''
    # 1. get 6 groups of rectangle vertexs
    bboxes_corners = bbox3d2corners(bboxes) # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners) # (n, 6, 4, 3)

    # 2. calculate plane equation: ax + by + cd + d = 0
    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)

    # 3. Judge each point inside or outside the bboxes
    # if point (x0, y0, z0) lies on the direction of normal vector(a, b, c), then ax0 + by0 + cz0 + d > 0.
    masks = points_in_bboxes(points, group_plane_equation_params) # (N, n)

    if not rm:
        return masks
        
    # 4. remove point insider the bboxes
    masks = np.any(masks, axis=-1)

    return points[~masks]


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/data_augment_utils.py#L31
@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


