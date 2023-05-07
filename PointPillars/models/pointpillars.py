import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda

from models.gpu_mem_track import MemTracker
from models.anchors import Anchors, anchor_target, anchors2bboxes
import inspect

from data_process import limit_period

# 体素化
class PFNLayer(nn.Module):
    def __init__(self,voxel_size,point_cloud_range, max_num_points,max_voxels,add_dist=False) -> None:
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        
        self.x_voxel_size = voxel_size[0]
        self.y_voxel_size = voxel_size[1]

        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]

        self.add_dist = add_dist

    def get_paddings_indicator(self, npoints_per_pillar, max_num, axis=0):
        '''
        返回一个map指出一个pillar中哪些是真实数据,哪些是填充的0数据
        '''
        npoints_per_pillar = torch.unsqueeze(npoints_per_pillar,axis+1)
        # voxel_index = torch.arange(0, actual_num)
        max_num_shape = [1] * len(npoints_per_pillar.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=npoints_per_pillar.device).view(max_num_shape)
        paddings_indicator = npoints_per_pillar.int() > max_num

        return paddings_indicator

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
            pillars: (p1 + p2 + ... + pb, num_points, c), 
            coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
            num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
            
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)


        '''
        获取论文中所需要的输入: x_c, y_c, z_c
        "the c subscript denotes distance to the arithmetic mean of all points in the pillar "
        对每个Pillar中的点进行去均值编码
        '''
        # pillars中所有点减去中心位置,设置keepdim=True的，则保留原来的维度信息
        points_mean = pillars[:,:,:3].sum(dim=1,keepdim=True) / npoints_per_pillar[:, None, None]
        pointcloud_center_offset = pillars[:,:,:3] - points_mean

        '''
        x_p, y_p
        the p subscript denotes the offset from the pillar x, y center
        对pillar中有效点减去中心位置
        计算步骤: coords是每个网格点的坐标,乘以voxel_size就能得到点云数据中实际的大小,以米为单位
        再加上每个pillar长宽的一半得到pillar中心点坐标
        每个点的x、y、z减去对应pillar的坐标中心点, 得到每个点到该点中心点的偏移量
        '''
        valid_pc_center_offset_x = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.x_voxel_size + self.x_offset)
        valid_pc_center_offset_y = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.y_voxel_size + self.x_offset)

        # 合并成9维的feature

        # 是否添加距离作为特征
        if self.add_dist:
            points_dist = torch.norm(pillars[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([pillars, pointcloud_center_offset, valid_pc_center_offset_x, valid_pc_center_offset_y, points_dist], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)

        else:
            #  将原始的同去均值编码和去中心编码的结果进行cat, 得到 (PxMx9) 的向量
            features = torch.cat([pillars, pointcloud_center_offset, valid_pc_center_offset_x, valid_pc_center_offset_y], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
                
        features[:, :, 0:1] = valid_pc_center_offset_x # 把9维编码向量中的前2维换成去中心编码的向量
        features[:, :, 1:2] = valid_pc_center_offset_y # 把9维编码向量中的前2维换成去中心编码的向量

        # mask = self.get_paddings_indicator(npoints_per_pillar, features.size(1), axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(pillars)

        voxel_ids = torch.arange(0, pillars.size(1)).to(coors_batch.device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        
        '''
        因为0填充的数据, 在计算出现xc,yc,zc和xp,yp,zp时会有值
        所以通过mask移除0的信息
        '''
        # features *= mask                                    # PxNxD
        features *= mask[:, :, None]

        features = features.permute(0, 2, 1).contiguous()   # PxNxD ----> PxDxN

        # return pillars, coors_batch, npoints_per_pillar, features
        return coors_batch, features

'''
提取点云特征, 类似PointNet
'''
class PillarFeatureNet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 voxel_size=[0.16, 0.16, 4], use_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 这里用卷积层代替MLP，就像PointNet一样
        self.conv = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(output_dim, eps=1e-3, momentum=0.01)

        self.x_limitation = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])      # 在x方向上voxel最多的个数
        self.y_limitation = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])      # 在y方向上voxel最多的个数

        self.use_norm = use_norm

    def forward(self, init_features, coors_batch):

        if self.use_norm:
            features = F.relu(self.bn(self.conv(init_features)), inplace=True) 
        else:
            features = F.relu(self.conv(init_features), inplace=True)

        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        pooling_features = torch.max(features, dim=-1)[0]

        '''
        将每个的pillar数据重新放回原来的坐标中,也就是二维坐标，组成 伪图像 数据
        对应到论文中就是stacked pillars,将生成的pillar按照坐标索引还原到原空间中。
        '''
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i  #返回mask，[True, False...]
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            # 创建一个空间坐标所有用来接受pillar中的数据
            spatial_feature = torch.zeros((self.x_limitation, self.y_limitation, self.output_dim), dtype=torch.float32, device=coors_batch.device)
            spatial_feature[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            spatial_feature = spatial_feature.permute(2, 1, 0).contiguous()
            batched_canvas.append(spatial_feature)
            
        #     # 释放显存
        #     spatial_feature = spatial_feature.cpu()
        #     torch.cuda.empty_cache()
        # # 释放显存
        # pooling_features = pooling_features.cpu()
        # torch.cuda.empty_cache()
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)

        return batched_canvas

# '''
# BACKBONE_2D的输入特征维度(batch_size, 64, 496, 432), 输出的特征维度为[batch_size, 384, 248, 216]。
# '''
class BackboneNet(nn.Module):
    def __init__(self, input_dim=64, num_filters=[64,128,256], layer_nums=[3,5,5], layer_strides=[2, 2, 2], upsample_strides=[1, 2, 4],
                 num_upsample_filters=[128,128,128]):
        super().__init__()

        num_levels = len(layer_nums)    # 三层

        input_channels = [input_dim, *num_filters[:-1]]

        # 存储下采样的网络
        self.blocks = nn.ModuleList()

        # 存储上采样的网络
        self.deblocks = nn.ModuleList()

        # 开始处理三层网络
        for index in range(num_levels):
            # 经过这里的层，feature map变小, (h - kernel + 2p )/s + 1
            cur_layers = [nn.ZeroPad2d(1),
                          nn.Conv2d(input_channels[index], num_filters[index], kernel_size=3, stride=layer_strides[index], bias=False, padding=0),
                          nn.BatchNorm2d(num_filters[index], eps=1e-3, momentum=0.01),
                          nn.ReLU(inplace=True)]
            
            for _ in range(layer_nums[index]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[index], num_filters[index], kernel_size=3, bias=False, padding=1),
                    nn.BatchNorm2d(num_filters[index], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ])

            # 添加下采样网络
            self.blocks.append(nn.Sequential(*cur_layers))

            # 上采样
            cur_stride = upsample_strides[index]
            upsample_block = []
            if cur_stride >= 1:
                
                upsample_block.append(nn.ConvTranspose2d(num_filters[index],num_upsample_filters[index],upsample_strides[index],stride=cur_stride,bias=False))
                upsample_block.append(nn.BatchNorm2d(num_upsample_filters[index], eps=1e-3, momentum=0.01))
                upsample_block.append(nn.ReLU(inplace=True))
            else:
                cur_stride = np.round(1 / cur_stride).astype(np.int)
                upsample_block.append(nn.ConvTranspose2d(num_filters[index],num_upsample_filters[index],cur_stride,stride=cur_stride,bias=False))
                upsample_block.append(nn.BatchNorm2d(num_upsample_filters[index], eps=1e-3, momentum=0.01))
                upsample_block.append(nn.ReLU(inplace=True))
            
            self.deblocks.append(nn.Sequential(*upsample_block))
    
    def forward(self, spatial_features):
        # return: (bs, 384, 248, 216)
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            # 下采样
            # 下采样之后，x的shape分别为：([batch_size, 64, 248, 216])，([batch_size, 128, 124, 108]), ([batch_size, 256, 62, 54])
            x = self.blocks[i](x)
            # stride = int(spatial_features.shape[3] / x.shape[2])

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        
        return x


class AnchorHeadNet(nn.Module):
    def __init__(self, input_channels, num_anchors_per_location, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 类别， 1x1 卷积：conv_cls:  Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
        # 每个点6个anchor，每个anchor预测3个类别，所以输出的类别为n_anchors*3
        self.conv_cls = nn.Conv2d(input_channels, num_anchors_per_location*num_classes, kernel_size=1)
        
        # box，1x1 卷积：conv_box:  Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
        # 每个点6个anchor，每个anchor预测7个值（x, y, z, w, l, h, θ），所以输出的值为num_anchors_per_location*7
        self.conv_box = nn.Conv2d(input_channels, num_anchors_per_location*7, kernel_size=1)

        # 用于分类方向
        # 每个点6个anchor，每个anchor预测2个方向(正负)，所以输出的值为num_anchors_per_location*2
        self.conv_dir_cls = nn.Conv2d(input_channels, num_anchors_per_location*2, 1)

        # 初始化权重
        self.init_weights()
    
    #初始化参数
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, x):
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_box(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillars(nn.Module):
    # num_classes: 识别的类别个数
    # max_num_points: 每个Pillar选择max_num_points个点，不足max_num_points个点时，用(0,0,0)补上
    def __init__(self,num_classes=3, max_num_points=32, point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_voxels=(16000, 40000),voxel_size=[0.16, 0.16, 4]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.PFNLayer = PFNLayer(voxel_size=voxel_size,point_cloud_range=point_cloud_range,max_num_points=max_num_points,max_voxels=max_voxels)

        self.featureLayer = PillarFeatureNet(input_dim=9,output_dim=64)

        self.Backbone = BackboneNet(input_dim=64, num_filters=[64,128,256], layer_nums=[3,5,5], layer_strides=[2, 2, 2], upsample_strides=[1, 2, 4],
                                    num_upsample_filters=[128,128,128])

        
        # 每个点会生成不同类别的2个先验框(anchor)，也就是说num_anchors_per_location*2
        self.AnchorHead = AnchorHeadNet(input_channels=384, num_anchors_per_location=2*num_classes, num_classes=num_classes)

        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        # 每个类别的先验框只有一种尺度信息；分别是车 [3.9, 1.6, 1.56]、人[0.8, 0.6, 1.73]、自行车[1.76, 0.6, 1.73]
        # 该数据来源于：https://github.com/open-mmlab/OpenPCDet/blob/a38a580f9ccf96c116269280fa5f6f721aa8bbcd/tools/cfgs/kitti_models/pillarnet.yaml
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]    # 该顺序需对应DataSet中的分类顺序     
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations)

        # frame = inspect.currentframe()
        # self.gpu_tracker = MemTracker(frame)     # 创建显存检测对象

                # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.num_classes)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.num_classes):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    
    def forward(self,batched_pts,mode='train',batched_gt_bboxes=None, batched_gt_labels=None):
        # self.gpu_tracker.track()
        # pillars个数，pillars在map中的位置，每个Pillar中有效点的数量, feature 维度: P:169184, D:9, N:32, 和论文的维度一致，论文是(DxPxN)
        # pillars, coors_batch, npoints_per_pillar, features = self.PFNLayer(batched_pts)
        coors_batch, features = self.PFNLayer(batched_pts)
        # self.gpu_tracker.track()
        # torch.cuda.empty_cache()
        encoded_features = self.featureLayer(features,coors_batch)            # batch size, 64, 496, 432 Nx64x
        # self.gpu_tracker.track()
        # torch.cuda.empty_cache()
        # self.gpu_tracker.track()
        backbone_result = self.Backbone(encoded_features)
        # backbone_tmp_result = self.Backbone(encoded_features)                   #  (batch size, 384, 248, 216)

        # backbone_result = self.neck(backbone_tmp_result)
        # print(backbone_result.size())

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.AnchorHead(backbone_result)
        
        # print("bbox_cls_pred",bbox_cls_pred.size())
        # print("bbox_cls_pred.size()[-2:]",bbox_cls_pred.size()[-2:])
        # anchors
        batch_size = len(batched_pts)
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        # print("feature_map_size",len(feature_map_size))
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.num_classes)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val' or mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        else:
            return None

        # self.gpu_tracker.track()
        return anchorHead_result