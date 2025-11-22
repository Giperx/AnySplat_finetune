import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy

from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.encoder.vggt.utils.rotation import mat_to_quat
from src.utils.point import get_normal_map

def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3
        
        quat = mat_to_quat(R)
        # Note the order of h and w here
        if image_size_hw is not None:
            H, W = image_size_hw
            fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
            fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        else:
            fov_h = 2 * torch.atan(0.5 / intrinsics[..., 1, 1])
            fov_w = 2 * torch.atan(0.5 / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding
    
def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)

class DistillLoss(nn.Module):
    def __init__(self, delta=1.0, gamma=0.6, weight_pose=1.0, weight_depth=1.0, weight_normal=1.0):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight_pose = weight_pose
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal

    def camera_loss_single(self, cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
        if loss_type == "l1":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
        elif loss_type == "l2":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
        elif loss_type == "huber":
            loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
            loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
            loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_T = torch.nan_to_num(loss_T, nan=0.0, posinf=0.0, neginf=0.0)
        loss_R = torch.nan_to_num(loss_R, nan=0.0, posinf=0.0, neginf=0.0)
        loss_fl = torch.nan_to_num(loss_fl, nan=0.0, posinf=0.0, neginf=0.0)

        loss_T = torch.clamp(loss_T, min=-100, max=100)
        loss_R = torch.clamp(loss_R, min=-100, max=100)
        loss_fl = torch.clamp(loss_fl, min=-100, max=100)

        loss_T = loss_T.mean()
        loss_R = loss_R.mean()
        loss_fl = loss_fl.mean()
        
        weight_T, weight_R, weight_fl = 1.0, 1.0, 0.5
        
        # return loss_T, loss_R, loss_fl
        return loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    def forward(self, distill_infos, pred_pose_enc_list, prediction, batch):
        loss_pose = 0.0

        if pred_pose_enc_list is not None:
            num_predictions = len(pred_pose_enc_list)
            pesudo_gt_pose_enc = distill_infos['pred_pose_enc_list']
            
            # use gt pose encodings for supervision inverse pred pose to w2c first
            gt_extrinsics = batch['context']['extrinsics'] # c2w
            gt_extrinsics_w2c = torch.linalg.inv(gt_extrinsics)
            gt_intrinsics = batch['context']['intrinsics']
            image_hw = batch['context']['image'].shape[-2:]
            # # Encode ground truth pose to match predicted encoding format
            # gt_pose_encoding = extri_intri_to_pose_encoding(
            #     gt_extrinsics_w2c, gt_intrinsics
            # )    
            b, v, _, _ = gt_extrinsics.shape
            
            for i in range(num_predictions):
                i_weight = self.gamma ** (num_predictions - i - 1)
                cur_pred_pose_enc = pred_pose_enc_list[i]
                
                ### method1
                ### add og camera loss use pseudo gt pose enc
                # cur_pesudo_gt_pose_enc = pesudo_gt_pose_enc[i]
                # # loss_pose += i_weight * huber_loss(cur_pred_pose_enc, cur_pesudo_gt_pose_enc).mean()
                # ### add camera loss l1 here
                # loss_pose += i_weight * self.camera_loss_single(cur_pred_pose_enc, cur_pesudo_gt_pose_enc, loss_type="l1")
                
                ### method2
                ### add camera loss use gt pose enc
                cur_pred_extrinsics, cur_pred_intrinsics = pose_encoding_to_extri_intri(
                    cur_pred_pose_enc.detach(),
                    image_size_hw=image_hw,
                )
                extrinsic_padding = (
                    torch.tensor([0, 0, 0, 1], device=cur_pred_extrinsics.device, dtype=cur_pred_extrinsics.dtype)
                    .view(1, 1, 1, 4)
                    .repeat(b, v, 1, 1)
                )
                cur_pred_extrinsics_c2w = torch.cat([cur_pred_extrinsics, extrinsic_padding], dim=2)  # BxVx4x4
                
                ### 对gt pose做刚体变换，对齐camera1，因为pred camera1的pose相对世界中心有偏差
                # pred_all_extrinsic, gt_ext: (B=1, V=3, 4, 4), 都是 cam2world
                pred = cur_pred_extrinsics_c2w.detach().clone()
                gt = gt_extrinsics.clone()
                # 变换 X = pred_c2w_1 @ inv(gt_c2w_1) ；这里的索引0是“camera1”
                X = torch.matmul(pred[:, 0], torch.linalg.inv(gt[:, 0]))     # (B, 4, 4)
                # 左乘到所有视角（会按批广播到 V）
                gt_ext = torch.matmul(X[:, None], gt) 
                gt_ext[:, 0] = pred[:, 0]  # 保持第0个视角和pred一致
                gt_intrinsics = cur_pred_intrinsics.detach().clone()
                gt_ext_w2c = torch.linalg.inv(gt_ext)
                gt_pose_encoding = extri_intri_to_pose_encoding(
                    gt_ext_w2c, gt_intrinsics, image_size_hw=image_hw # 使用pred intrinsics, pred的在上面输出后没有归一化
                )
                # print("gt_pose_encoding:", gt_pose_encoding)
                loss_pose += i_weight * self.camera_loss_single(cur_pred_pose_enc, gt_pose_encoding.detach().requires_grad_(False), loss_type="l1")
                
                
            loss_pose = loss_pose / num_predictions
            loss_pose = torch.nan_to_num(loss_pose, nan=0.0, posinf=0.0, neginf=0.0)
        
        pred_depth = prediction.depth.flatten(0, 1)
        pesudo_gt_depth = distill_infos['depth_map'].flatten(0, 1).squeeze(-1)

        if "distill_depth_map_new" in distill_infos:
            # print("Using distill_depth_map_new for depth loss calculation.")
            ### depth loss use gt depth and valid mask
            conf_mask = distill_infos["distill_valid_mask"].flatten(0, 1)
            loss_depth = F.mse_loss(pred_depth[conf_mask], distill_infos["distill_depth_map_new"].flatten(0, 1)[conf_mask], reduction='none').mean()
        else:
            conf_mask = distill_infos['conf_mask'].flatten(0, 1)

            ### og depth loss use pseudo gt depth and conf mask(all ones)
            if batch['context']['valid_mask'].sum() > 0:
                conf_mask = batch['context']['valid_mask'].flatten(0, 1)

            loss_depth = F.mse_loss(pred_depth[conf_mask], pesudo_gt_depth[conf_mask], reduction='none').mean()            
        
        render_normal = get_normal_map(pred_depth, batch["context"]["intrinsics"].flatten(0, 1))
        pred_normal = get_normal_map(pesudo_gt_depth, batch["context"]["intrinsics"].flatten(0, 1))
        
        alpha1_loss = (1 - (render_normal[conf_mask] * pred_normal[conf_mask]).sum(-1)).mean()
        alpha2_loss = F.l1_loss(render_normal[conf_mask], pred_normal[conf_mask], reduction='mean')
        loss_normal = (alpha1_loss + alpha2_loss) / 2
        
        loss_distill = loss_pose * self.weight_pose + loss_depth * self.weight_depth + loss_normal * self.weight_normal
        loss_distill = torch.nan_to_num(loss_distill, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss_dict = {
            "loss_distill": loss_distill,
            "loss_pose": loss_pose * self.weight_pose,
            "loss_depth": loss_depth * self.weight_depth,
            "loss_normal": loss_normal * self.weight_normal
        }

        return loss_dict
