import numpy as np
import os
import pickle
import data_process
import open3d as o3d
import copy
import numba
import random
from ops import Voxelization, nms_cuda
import torch

from torch.utils.data import DataLoader
from models.pointpillars import PointPillars, PFNLayer
import data_process
import data_augment
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter   
from models.gpu_mem_track import MemTracker

import inspect
from models.losses import Losses

# 自定义collate_batch用于torch的Dataloader
def collate_batch(data):
    batched_pointcloud_list = []
    batched_gt_3d_bboxes_list = []
    batched_labels_list= []
    batched_names_list = []
    batched_difficulty_list = []
    batched_img_list = []
    batched_calibration_list = []
    for data_dict in data:
        cur_pc = data_dict['pc']
        cur_image_info = data_dict['img']
        cur_gt_labels = data_dict['gt_labels']
        cur_gt_names = data_dict['gt_names']
        cur_gt_bboxes_3d = data_dict['gt_bboxes_3d']
        cur_difficulty = data_dict['difficulty']
        cur_calbi_info = data_dict['calib']

        batched_pointcloud_list.append(torch.from_numpy(cur_pc))
        batched_gt_3d_bboxes_list.append(torch.from_numpy(cur_gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(cur_gt_labels))
        batched_names_list.append(cur_gt_names) # List(str)
        batched_difficulty_list.append(torch.from_numpy(cur_difficulty))
        batched_img_list.append(cur_image_info)
        batched_calibration_list.append(cur_calbi_info)
    
    rt_data_dict = dict(
        batched_pts=batched_pointcloud_list,
        batched_img_info=batched_img_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_gt_bboxes=batched_gt_3d_bboxes_list,
        batched_difficulty=batched_difficulty_list,
        batched_calib_info=batched_calibration_list
    )

    return rt_data_dict

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('learning_rate', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def train(model, optimizer, criterion, scheduler, trainloader, device,
           valloader, max_epoch, ckpt_freq_epoch, saved_ckpt_path,
             tb_write, freq_val, start_epoch, num_classes=3):
    for epoch in range(start_epoch, start_epoch+max_epoch):
        print('=' * 30, "epoch:", epoch, '=' * 30)
        model = model.train()
        train_step = 0
        val_step = 0
        for i, data_dict in enumerate(tqdm(trainloader)):
            if torch.cuda.is_available():
                # move tensors to cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            optimizer.zero_grad()
            # 获取当前的data
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            # batched_difficulty = data_dict['batched_difficulty']

            # frame = inspect.currentframe()
            # gpu_tracker = MemTracker(frame)     # 创建显存检测对象
            # gpu_tracker.track()
            # pillars, coors_batch, npoints_per_pillar, features, encoded_features, backbone_result = pointpillars(batched_pts=batched_pts)
            pred_bbox_cls, pred_bbox_loc, bbox_dir_cls_pred, anchor_target_dict = model(batched_pts=batched_pts,mode='train',
                                                                                            batched_gt_bboxes=batched_gt_bboxes, 
                                                                                            batched_gt_labels=batched_labels)
            # print("bbox_cls_pred",pred_bbox_cls.size())
            # print("bbox_dir_cls_pred",bbox_dir_cls_pred.size())
            # print("bbox_pred",pred_bbox_loc.size())
            
            # 预测的box的类别
            pred_bbox_cls = pred_bbox_cls.permute(0, 2, 3, 1).reshape(-1, num_classes)
            # 预测的box参数
            pred_bbox_loc = pred_bbox_loc.permute(0, 2, 3, 1).reshape(-1, 7)
            # 预测的box方向
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            anchor_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            anchor_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            anchor_bbox_loc = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            anchor_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            # 预测结果在范围内
            pos_idx = (anchor_bbox_labels >= 0) & (anchor_bbox_labels < num_classes)
            pred_bbox_loc = pred_bbox_loc[pos_idx]
            anchor_bbox_loc = anchor_bbox_loc[pos_idx]

            # 来自于ground truth
            # delta_theta = sin(theta^gt-theta^anchor)      ------->        delta_theta = sin(theta^gt)*cos(theta^anchor) - cos(theta^gt)*sin(theta^anchor)
            pred_bbox_loc[:,-1] = torch.sin(pred_bbox_loc[:, -1].clone()) * torch.cos(anchor_bbox_loc[:,-1].clone())
            anchor_bbox_loc[:,-1] = torch.cos(pred_bbox_loc[:, -1].clone()) * torch.sin(anchor_bbox_loc[:,-1].clone())

            pred_bbox_cls = pred_bbox_cls[anchor_label_weights > 0]
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            anchor_dir_labels = anchor_dir_labels[pos_idx]

            num_cls_pos = (anchor_bbox_labels < num_classes).sum()
            anchor_bbox_labels[anchor_bbox_labels < 0] = num_classes
            anchor_bbox_labels = anchor_bbox_labels[anchor_label_weights > 0]

            loss_dict = criterion(pred_bbox_cls=pred_bbox_cls,
                                    pred_bbox_loc=pred_bbox_loc,
                                    pred_bbox_dir=bbox_dir_cls_pred,
                                    anchor_labels=anchor_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    anchor_bbox_loc=anchor_bbox_loc, 
                                    anchor_bbox_dir_labels=anchor_dir_labels,
                                    num_classes=num_classes)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(trainloader) + train_step + 1

            # 10 step record one time
            if global_step % 8 == 0:
                save_summary(tb_write, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1

        # 10 epoch保存一次checkpoint
        if (epoch + 1) % ckpt_freq_epoch == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % freq_val == 0:
            continue
        print('=' * 30, "val:", epoch, '=' * 30)
        # If you pass in a validation dataloader then compute the validation loss
        if not valloader is None:
            model.eval()
            with torch.no_grad():
                for i, data_dict in enumerate(tqdm(valloader)):
                    if torch.cuda.is_available():
                        # move tensors to cuda
                        for key in data_dict:
                            for j, item in enumerate(data_dict[key]):
                                if torch.is_tensor(item):
                                    data_dict[key][j] = data_dict[key][j].cuda()
                    
                    optimizer.zero_grad()
                    # 获取当前的data
                    batched_pts = data_dict['batched_pts']
                    batched_gt_bboxes = data_dict['batched_gt_bboxes']
                    batched_labels = data_dict['batched_labels']
                    # batched_difficulty = data_dict['batched_difficulty']

                    # frame = inspect.currentframe()
                    # gpu_tracker = MemTracker(frame)     # 创建显存检测对象
                    # gpu_tracker.track()
                    # pillars, coors_batch, npoints_per_pillar, features, encoded_features, backbone_result = pointpillars(batched_pts=batched_pts)
                    pred_bbox_cls, pred_bbox_loc, bbox_dir_cls_pred, anchor_target_dict = model(batched_pts=batched_pts,mode='train',
                                                                                                    batched_gt_bboxes=batched_gt_bboxes, 
                                                                                                    batched_gt_labels=batched_labels)
                    
                    # 预测的box的类别
                    pred_bbox_cls = pred_bbox_cls.permute(0, 2, 3, 1).reshape(-1, num_classes)
                    # 预测的box参数
                    pred_bbox_loc = pred_bbox_loc.permute(0, 2, 3, 1).reshape(-1, 7)
                    # 预测的box方向
                    bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                    anchor_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                    anchor_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                    anchor_bbox_loc = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                    # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                    anchor_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                    # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                    
                    # 预测结果在范围内
                    pos_idx = (anchor_bbox_labels >= 0) & (anchor_bbox_labels < num_classes)
                    pred_bbox_loc = pred_bbox_loc[pos_idx]
                    anchor_bbox_loc = anchor_bbox_loc[pos_idx]

                    # 来自于ground truth
                    # delta_theta = sin(theta^gt-theta^anchor)      ------->        delta_theta = sin(theta^gt)*cos(theta^anchor) - cos(theta^gt)*sin(theta^anchor)
                    pred_bbox_loc[:,-1] = torch.sin(pred_bbox_loc[:, -1].clone()) * torch.cos(anchor_bbox_loc[:,-1].clone())
                    anchor_bbox_loc[:,-1] = torch.cos(pred_bbox_loc[:, -1].clone()) * torch.sin(anchor_bbox_loc[:,-1].clone())

                    pred_bbox_cls = pred_bbox_cls[anchor_label_weights > 0]
                    bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                    anchor_dir_labels = anchor_dir_labels[pos_idx]

                    num_cls_pos = (anchor_bbox_labels < num_classes).sum()
                    anchor_bbox_labels[anchor_bbox_labels < 0] = num_classes
                    anchor_bbox_labels = anchor_bbox_labels[anchor_label_weights > 0]

                    loss = criterion(pred_bbox_cls=pred_bbox_cls,
                                            pred_bbox_loc=pred_bbox_loc,
                                            pred_bbox_dir=bbox_dir_cls_pred,
                                            anchor_labels=anchor_bbox_labels, 
                                            num_cls_pos=num_cls_pos, 
                                            anchor_bbox_loc=anchor_bbox_loc, 
                                            anchor_bbox_dir_labels=anchor_dir_labels,
                                            num_classes=num_classes)
                    
                    global_step = epoch * len(valloader) + val_step + 1
                    if global_step % 8 == 0:
                        save_summary(tb_write, loss_dict, global_step, 'val')
                    val_step += 1
    return model

def main(args):
    batch_size = args.batchSize
    # set the random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(0)
    else:
        device = torch.device("cpu")

    # 获取数据增强后的数据
    train_data = data_augment.DataSet(dataset_root=args.dataset_root,identifier='train')
    val_data = data_augment.DataSet(dataset_root=args.dataset_root,identifier='val')

    # print(train_data[0])

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,num_workers=args.num_workers, drop_last=False, collate_fn=collate_batch)

    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,num_workers=args.num_workers, drop_last=False, collate_fn=collate_batch)
    
    # if argparse.use_pretrain:

    # else:
    pointpillars = PointPillars(num_classes=args.num_classes).cuda()

    

    criterion = Losses()

    learning_rate = args.learning_rate

    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=learning_rate, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=learning_rate*10, 
                                                    total_steps=len(train_dataloader) * args.epochs, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    
    if args.use_pretrain:
        load_checkpoint = torch.load(args.checkpoints)
        pointpillars.load_state_dict(load_checkpoint['net'])
        # optimizer.load_state_dict(load_checkpoint['optimizer'])
        start_epoch = load_checkpoint['epoch']+1
    else:
        start_epoch = 0
    
    # if runtime has GPU use GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True       # 用cudnn
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # 用于tensorboard
    saved_tensorboard_path = os.path.join('summary')
    os.makedirs(saved_tensorboard_path, exist_ok=True)
    writer = SummaryWriter(saved_tensorboard_path)

    # 用于checkpoints
    saved_ckpt_path = os.path.join(args.saved_ckpt_path)
    os.makedirs(saved_ckpt_path, exist_ok=True)


    # model, optimizer, criterion, scheduler, trainloader, device, valloader, max_epoch, ckpt_freq_epoch, saved_ckpt_path
    model = train(model=pointpillars, 
                optimizer=optimizer, 
                criterion=criterion,
                scheduler=scheduler, 
                trainloader=train_dataloader, 
                device=device, 
                valloader=val_dataloader, 
                max_epoch=args.epochs,
                ckpt_freq_epoch=10,     # 保存cpt的频率
                saved_ckpt_path=args.saved_ckpt_path,
                freq_val = args.freq_val_in_train,
                start_epoch = start_epoch,
                tb_write=writer)

if __name__ == '__main__':
    # 接收从终端来的参数
    parser = argparse.ArgumentParser()

    # the path of dataset
    #数据集的绝对路径
    parser.add_argument('--dataset_root', \
                        # default='/media/chris/Workspace/Dataset/3d-object-detection-for-autonomous-vehicles/kitti_format/',
                        default='/media/chris/Workspace/Dataset/kitti/',
                        help='the root path of dataset')
    parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=4)       #多少进程
    parser.add_argument('--learning_rate',type=float, default=0.00025)
    parser.add_argument('--epochs',type=int,default=160)
    parser.add_argument('--saved_ckpt_path',default='checkpoints',help='the path to storage the checkpoint file')
    parser.add_argument('--saved_tensorboard_path',default='tensorboard_log',help='the path to storage the log file for tensorboard')
    parser.add_argument('--num_classes',type=int,default=3)
    parser.add_argument('--use_pretrain',type=bool,default=False)
    parser.add_argument('--freq_val_in_train',type=int, default=2)      # 训练中测试的频率
    parser.add_argument('--checkpoints',default='checkpoints/epoch_160.pth')
    args = parser.parse_args()
    # python train.py --epochs=40 --use_pretrain=True
    main(args)