import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.loss.loss_ssim import ssim
from .view_rendering_cross_view import ViewRenderingCrossView
import torchvision.utils as vutils

class CrossViewPhotoDepthSinglePose(nn.Module):
    """
    针对单个 pred_pose_enc (B x V x 9) 的跨视角 photometric + depth 一致性。
    注意：这里的 depth 形状是 B x V x H x W（没有 channel 维）。
    """

    def __init__(
        self,
        weight_photo: float = 1.0,
        weight_depth: float = 1.0,
        ssim_weight: float = 0.85,
        depth_l1: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        use_valid_mask: bool = True,
        debug_save: bool = False,
        debug_dir: str = "./debug_cross_view_loss",
    ):
        super().__init__()
        self.weight_photo = weight_photo
        self.weight_depth = weight_depth
        self.ssim_weight = ssim_weight
        self.depth_l1 = depth_l1
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_valid_mask = use_valid_mask
        self.debug_save = debug_save
        self.debug_dir = debug_dir
        if self.debug_save:
            os.makedirs(self.debug_dir, exist_ok=True)
        self.view_render = ViewRenderingCrossView()

    def _get_extri_intri(self, pose_enc: torch.Tensor, image_size_hw: tuple[int, int]):
        B, V, _ = pose_enc.shape
        extrinsics_3x4, intrinsics = pose_encoding_to_extri_intri(
            pose_enc.view(B, V, -1),
            image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True,
        )
        device = extrinsics_3x4.device
        dtype = extrinsics_3x4.dtype
        last_row = torch.tensor([0, 0, 0, 1.0], device=device, dtype=dtype)
        last_row = last_row.view(1, 1, 1, 4).expand(B, V, 1, 4)
        extrinsics_4x4 = torch.cat([extrinsics_3x4, last_row], dim=2)  # BxVx4x4
        return extrinsics_4x4, intrinsics

    def _warp_depth_src_to_tgt(
        self,
        src_depth: torch.Tensor,         # B x H x W
        tgt_depth: torch.Tensor,         # B x H x W, 仅 H,W 用于大小
        intrinsics_src: torch.Tensor,    # B x 3 x 3
        intrinsics_tgt: torch.Tensor,    # B x 3 x 3
        extrinsics_w2c_src: torch.Tensor,# B x 4 x 4 world->src
        extrinsics_w2c_tgt: torch.Tensor,# B x 4 x 4 world->tgt
    ):
        B, H, W = src_depth.shape
        device = src_depth.device
        dtype = src_depth.dtype

        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        u = grid_x.unsqueeze(0).expand(B, H, W)
        v = grid_y.unsqueeze(0).expand(B, H, W)

        fx_s = intrinsics_src[:, 0, 0].view(B, 1, 1)
        fy_s = intrinsics_src[:, 1, 1].view(B, 1, 1)
        cx_s = intrinsics_src[:, 0, 2].view(B, 1, 1)
        cy_s = intrinsics_src[:, 1, 2].view(B, 1, 1)

        z_s = src_depth.clamp(min=self.min_depth, max=self.max_depth)  # BxHxW
        x_s = (u - cx_s) * z_s / (fx_s + 1e-8)
        y_s = (v - cy_s) * z_s / (fy_s + 1e-8)
        pts_s = torch.stack([x_s, y_s, z_s], dim=-1)  # BxHxWx3

        R_s = extrinsics_w2c_src[:, :3, :3]
        t_s = extrinsics_w2c_src[:, :3, 3].view(B, 1, 1, 3)
        R_s_inv = R_s.transpose(1, 2)
        X_world = torch.matmul(pts_s - t_s, R_s_inv.transpose(1, 2))

        R_t = extrinsics_w2c_tgt[:, :3, :3]
        t_t = extrinsics_w2c_tgt[:, :3, 3].view(B, 1, 1, 3)
        cam_t = torch.matmul(X_world, R_t.transpose(1, 2)) + t_t
        X_t = cam_t[..., 0]
        Y_t = cam_t[..., 1]
        Z_t = cam_t[..., 2].clamp(min=1e-4)

        fx_t = intrinsics_tgt[:, 0, 0].view(B, 1, 1)
        fy_t = intrinsics_tgt[:, 1, 1].view(B, 1, 1)
        cx_t = intrinsics_tgt[:, 0, 2].view(B, 1, 1)
        cy_t = intrinsics_tgt[:, 1, 2].view(B, 1, 1)

        u_t = fx_t * X_t / Z_t + cx_t
        v_t = fy_t * Y_t / Z_t + cy_t

        depth_warp = torch.zeros_like(tgt_depth)   # BxHxW
        mask_warp = torch.zeros_like(tgt_depth, dtype=torch.bool)  # BxHxW

        u_t_long = u_t.round().long()
        v_t_long = v_t.round().long()
        valid = (
            (u_t_long >= 0)
            & (u_t_long < W)
            & (v_t_long >= 0)
            & (v_t_long < H)
            & (Z_t > 1e-4)
        )

        b_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
        depth_warp[b_ids[valid], v_t_long[valid], u_t_long[valid]] = Z_t[valid]
        mask_warp[b_ids[valid], v_t_long[valid], u_t_long[valid]] = True

        return depth_warp, mask_warp.float()

    def forward(
        self,
        pred_pose_enc: torch.Tensor,   # B x V x 9
        prediction,                    # prediction.depth: B x V x H x W; prediction.color: B x V x 3 x H x W
        batch: dict,
        global_step: int | None = None,
    ):
        depth = prediction.depth        # B x V x H x W
        color_pred = prediction.color   # B x V x 3 x H x W，已经在 [0,1]
        B, V, H, W = depth.shape
        device = depth.device

        # GT 图像在 [-1,1]，需要映射到 [0,1]
        color_gt = batch["context"]["image"].to(device)  # B x V x 3 x H x W，[-1,1]

        valid_mask = batch["context"].get("valid_mask", None)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
            if valid_mask.dim() == 5:
                valid_mask = valid_mask.squeeze(2)  # BxVxHxW
        else:
            valid_mask = torch.ones((B, V, H, W), dtype=torch.bool, device=device)

        extrinsics_w2c, intrinsics = self._get_extri_intri(
            pred_pose_enc, image_size_hw=(H, W)
        )
        extrinsics_w2c = extrinsics_w2c.to(device)    # BxVx4x4
        intrinsics = intrinsics.to(device)            # BxVx3x3

        total_photo_loss = depth.new_tensor(0.0)
        total_depth_loss = depth.new_tensor(0.0)
        num_pairs = 0

        do_debug = self.debug_save and (global_step is None or global_step < 5)
        debug_b = 0
        max_debug_pairs = 4
        debug_pair_count = 0

        for t in range(V):
            depth_t = depth[:, t]      # BxHxW
            # gt: [-1,1] -> [0,1]
            img_t_gt = ((color_gt[:, t] + 1.0) * 0.5).clamp(0.0, 1.0)  # Bx3xHxW
            vld_t = valid_mask[:, t]   # BxHxW

            intr_t = intrinsics[:, t]
            ext_w2c_t = extrinsics_w2c[:, t]

            for s in range(V):
                if s == t:
                    continue
                depth_s = depth[:, s]        # BxHxW
                # pred 已经在 [0,1]，不用再变换
                img_s_pred = color_pred[:, s].clamp(0.0, 1.0)  # Bx3xHxW
                intr_s = intrinsics[:, s]
                ext_w2c_s = extrinsics_w2c[:, s]

                # ---- warp 图像 s -> t ----
                warped_img, warp_mask_img = self.view_render.warp_src_to_tgt(
                    src_img=img_s_pred,
                    tgt_depth=depth_t.unsqueeze(1),  # Bx1xHxW
                    intrinsics_src=intr_s,
                    intrinsics_tgt=intr_t,
                    extrinsics_w2c_src=ext_w2c_s,
                    extrinsics_w2c_tgt=ext_w2c_t,
                )

                # ---- warp 深度 s -> t ----
                warped_depth, warp_mask_depth = self._warp_depth_src_to_tgt(
                    src_depth=depth_s,          # BxHxW
                    tgt_depth=depth_t,          # BxHxW
                    intrinsics_src=intr_s,
                    intrinsics_tgt=intr_t,
                    extrinsics_w2c_src=ext_w2c_s,
                    extrinsics_w2c_tgt=ext_w2c_t,
                )

                valid_all = (
                    vld_t.bool()
                    & warp_mask_img[:, 0].bool()
                    & warp_mask_depth.bool()
                    & (depth_t > self.min_depth)
                    & (depth_t < self.max_depth)
                    & (warped_depth > self.min_depth)
                    & (warped_depth < self.max_depth)
                )  # BxHxW

                # -------- debug 保存（可选） --------
                if do_debug and debug_pair_count < max_debug_pairs:
                    tgt_img_vis = img_t_gt[debug_b].detach().cpu()          # 3xHxW
                    src_img_vis = img_s_pred[debug_b].detach().cpu()        # 3xHxW
                    warped_img_vis = warped_img[debug_b].detach().cpu()     # 3xHxW

                    tgt_depth_vis = depth_t[debug_b].detach().cpu()         # HxW
                    src_depth_vis = depth_s[debug_b].detach().cpu()         # HxW
                    warped_depth_vis = warped_depth[debug_b].detach().cpu() # HxW

                    def norm_depth(d):
                        d_ = d.clone()
                        mask = d_ > 0
                        if mask.any():
                            d_min = d_[mask].min()
                            d_max = d_[mask].max()
                            d_[~mask] = 0
                            d_[mask] = (d_[mask] - d_min) / (d_max - d_min + 1e-8)
                        return d_

                    tgt_depth_img = norm_depth(tgt_depth_vis).unsqueeze(0)      # 1xHxW
                    src_depth_img = norm_depth(src_depth_vis).unsqueeze(0)
                    warped_depth_img = norm_depth(warped_depth_vis).unsqueeze(0)

                    grid_top = torch.stack(
                        [src_img_vis, tgt_img_vis, warped_img_vis], dim=0
                    )  # 3 x 3 x H x W
                    grid_bottom = torch.stack(
                        [
                            src_depth_img.repeat(3, 1, 1),
                            tgt_depth_img.repeat(3, 1, 1),
                            warped_depth_img.repeat(3, 1, 1),
                        ],
                        dim=0,
                    )  # 3 x 3 x H x W
                    grid = torch.cat([grid_top, grid_bottom], dim=0)  # 6 x 3 x H x W

                    save_name = f"b{debug_b}_t{t}_s{s}_step{global_step if global_step is not None else 0}.png"
                    save_path = os.path.join(self.debug_dir, save_name)
                    vutils.save_image(grid, save_path, nrow=3)
                    debug_pair_count += 1

                if valid_all.sum() == 0:
                    continue

                # -------- photometric loss --------
                mask_photo = valid_all.unsqueeze(1).expand(-1, 3, -1, -1)  # Bx3xHxW
                img_t_flat = img_t_gt[mask_photo]
                warped_flat = warped_img[mask_photo]

                l2 = (warped_flat - img_t_flat).pow(2).mean()

                ssim_val, *_ = ssim(
                    img_t_gt,
                    warped_img,
                    data_range=1.0,   # 现在两者都在 [0,1]
                    size_average=True,
                    win_size=11,
                    win_sigma=1.5,
                )
                ssim_loss = 1.0 - ssim_val

                photo_loss = self.ssim_weight * ssim_loss + (1.0 - self.ssim_weight) * l2

                # -------- depth loss --------
                depth_t_flat = depth_t[valid_all]
                depth_warp_flat = warped_depth[valid_all]
                if self.depth_l1:
                    depth_diff = (depth_t_flat - depth_warp_flat).abs()
                else:
                    depth_diff = (depth_t_flat - depth_warp_flat).pow(2)
                depth_loss = depth_diff.mean()

                total_photo_loss = total_photo_loss + photo_loss
                total_depth_loss = total_depth_loss + depth_loss
                num_pairs += 1

        if num_pairs > 0:
            total_photo_loss = total_photo_loss / num_pairs
            total_depth_loss = total_depth_loss / num_pairs
        else:
            total_photo_loss = total_photo_loss * 0.0
            total_depth_loss = total_depth_loss * 0.0

        loss_total = (
            self.weight_photo * total_photo_loss
            + self.weight_depth * total_depth_loss
        )
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "loss_cross_photo_single": self.weight_photo * total_photo_loss,
            "loss_cross_depth_single": self.weight_depth * total_depth_loss,
            "loss_cross_total_single": loss_total,
        }