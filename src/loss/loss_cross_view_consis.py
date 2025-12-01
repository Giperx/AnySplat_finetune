import torch
import torch.nn as nn

from .CrossViewPhotoDepthSinglePose import CrossViewPhotoDepthSinglePose


class CrossViewPhotoDepthConsistencyLossForDistill(nn.Module):
    """
    处理 pred_pose_enc_list 的跨视角一致性损失，适配原始 DistillLoss 的多阶段蒸馏结构。

    输入：
      - distill_infos: dict，至少包含 'pred_pose_enc_list'（teacher 的 pose_encoding 列表）可选
      - pred_pose_enc_list: list[Tensor]，每个形状 B x V x 9
      - prediction: DecoderOutput（depth, color）
      - batch: dict（含 context.image / context.valid_mask）
    """

    def __init__(
        self,
        delta: float = 1.0,
        gamma: float = 0.6,
        weight_photo: float = 1.0,
        weight_depth: float = 1.0,
        ssim_weight: float = 0.85,
        depth_l1: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        use_valid_mask: bool = True,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

        self.single_pose_loss = CrossViewPhotoDepthSinglePose(
            weight_photo=weight_photo,
            weight_depth=weight_depth,
            ssim_weight=ssim_weight,
            depth_l1=depth_l1,
            min_depth=min_depth,
            max_depth=max_depth,
            use_valid_mask=use_valid_mask,
            debug_save=True,
        )

    def forward(self, pred_pose_enc_list, prediction, batch):
        """
        与 DistillLoss.forward 相同的签名，方便集成：
            loss_dict = cross_view_loss_for_distill(distill_infos, pred_pose_enc_list, prediction, batch)
        """
        loss_photo = 0.0
        loss_depth = 0.0
        loss_total = 0.0

        if pred_pose_enc_list is not None:
            num_predictions = len(pred_pose_enc_list)

            # 如果你希望用 teacher 的 pose（distill_infos['pred_pose_enc_list']）做几何，
            # 可以在这里替换 pred_pose_enc_list 为 pesudo_gt_pose_enc_list；
            # 这里默认直接使用学生当前每一层的 pred_pose_enc。
            for i in range(num_predictions):
                i_weight = self.gamma ** (num_predictions - i - 1)

                cur_pred_pose_enc = pred_pose_enc_list[i]  # B x V x 9

                loss_dict_single = self.single_pose_loss(
                    pred_pose_enc=cur_pred_pose_enc,
                    prediction=prediction,
                    batch=batch,
                )

                loss_total_single = loss_dict_single["loss_cross_total_single"]
                loss_photo_single = loss_dict_single["loss_cross_photo_single"]
                loss_depth_single = loss_dict_single["loss_cross_depth_single"]

                loss_total = loss_total + i_weight * loss_total_single
                loss_photo = loss_photo + i_weight * loss_photo_single
                loss_depth = loss_depth + i_weight * loss_depth_single

            loss_total = loss_total / num_predictions
            loss_photo = loss_photo / num_predictions
            loss_depth = loss_depth / num_predictions

            loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=0.0, neginf=0.0)
            loss_photo = torch.nan_to_num(loss_photo, nan=0.0, posinf=0.0, neginf=0.0)
            loss_depth = torch.nan_to_num(loss_depth, nan=0.0, posinf=0.0, neginf=0.0)

        loss_dict = {
            "loss_cross_total": loss_total,
            "loss_cross_photo": loss_photo,
            "loss_cross_depth": loss_depth,
        }
        return loss_dict