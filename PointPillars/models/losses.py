import torch
import torch.nn as nn
import torch.nn.functional as F

class Losses(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, beta_cls=1, beta_loc=2, beta_dir=0.2):
        super().__init__()
        # 该参数具体取值来源于论文
        self.alpha = alpha
        self.gamma = gamma
        self.beta_cls = beta_cls
        self.beta_loc = beta_loc
        self.beta_dir = beta_dir

        self.SmoothL1 = nn.SmoothL1Loss(reduction='none',beta=1/9)

        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self,pred_bbox_cls, pred_bbox_loc, pred_bbox_dir,anchor_labels, num_cls_pos, anchor_bbox_loc, anchor_bbox_dir_labels, num_classes=3):

        # loc loss
        L_loc = self.SmoothL1(pred_bbox_loc, anchor_bbox_loc)
        L_loc = L_loc.sum() / L_loc.size(0)

        # cls loss
        # 论文里的log p^a
        anchor_labels = F.one_hot(anchor_labels, num_classes + 1)[:, :num_classes].float() # (n, 3)
        
        # 论文里的p^a
        pred_bbox_cls_sigmoid = torch.sigmoid(pred_bbox_cls) 
        
        weight = self.alpha*(1-pred_bbox_cls_sigmoid).pow(self.gamma) *anchor_labels * anchor_labels + \
                    (1-self.alpha) * pred_bbox_cls_sigmoid.pow(self.gamma) * (1-anchor_labels)
        L_cls = F.binary_cross_entropy(pred_bbox_cls_sigmoid, anchor_labels, reduction='none')
        L_cls = L_cls * weight
        L_cls = L_cls.sum() / num_cls_pos

        # direction cls loss
        L_dir_cls = self.dir_cls(pred_bbox_dir, anchor_bbox_dir_labels)

        # 论文公式
        loss = self.beta_loc * L_loc + self.beta_cls * L_cls + self.beta_dir * L_dir_cls

        loss_dict={'cls_loss': L_cls, 
                   'loc_loss': L_loc,
                   'dir_cls_loss': L_dir_cls,
                   'total_loss': loss}
        
        return loss_dict