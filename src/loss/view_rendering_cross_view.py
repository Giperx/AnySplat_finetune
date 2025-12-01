import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pixel_grid(B: int, H: int, W: int, device, dtype=torch.float32):
    """
    构造标准化到 [-1,1] 的 pixel grid:
        grid: (B, H, W, 2), 最后两个维度是 (x, y) in [-1,1]
    """
    xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    # 归一化到 [-1, 1]
    grid_x_norm = 2.0 * (grid_x / (W - 1.0)) - 1.0
    grid_y_norm = 2.0 * (grid_y / (H - 1.0)) - 1.0

    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)  # (H,W,2)
    grid = grid.unsqueeze(0).expand(B, H, W, 2)  # (B,H,W,2)
    return grid


class ViewRenderingCrossView(nn.Module):
    """
    简化版跨视角 View Rendering：
      - 使用 target 视角的 depth + pose，从 source 视角的图像 warp 到 target。
      - pose 形式：extrinsics 为 world2cam，intrinsics 为 3x3。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _invert_extrinsics_w2c_to_c2w(extrinsics_w2c: torch.Tensor) -> torch.Tensor:
        """
        将 world2cam 的 4x4 extrinsics 反转为 cam2world。
        extrinsics_w2c: (..., 4, 4)
        """
        return torch.inverse(extrinsics_w2c)

    def warp_src_to_tgt(
        self,
        src_img: torch.Tensor,          # B x 3 x H x W
        tgt_depth: torch.Tensor,        # B x 1 x H x W
        intrinsics_src: torch.Tensor,   # B x 3 x 3
        intrinsics_tgt: torch.Tensor,   # B x 3 x 3
        extrinsics_w2c_src: torch.Tensor,  # B x 4 x 4  world->src
        extrinsics_w2c_tgt: torch.Tensor,  # B x 4 x 4  world->tgt
    ):
        """
        基础几何：
          world -> tgt_cam -> 像素 -> 通过 tgt_depth 得到 3D 点，
          再用相对位姿变换到 src_cam，然后投影到 src 像素系，用 grid_sample 采样 src_img。
        """
        B, _, H, W = tgt_depth.shape
        device = tgt_depth.device
        dtype = tgt_depth.dtype

        # 1. 构造像素坐标网格 (u, v) in 像素坐标
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        u = grid_x.unsqueeze(0).expand(B, H, W)  # BxHxW
        v = grid_y.unsqueeze(0).expand(B, H, W)  # BxHxW

        # 2. 深度 + intrinsics_tgt -> tgt 相机坐标
        fx_t = intrinsics_tgt[:, 0, 0].view(B, 1, 1)
        fy_t = intrinsics_tgt[:, 1, 1].view(B, 1, 1)
        cx_t = intrinsics_tgt[:, 0, 2].view(B, 1, 1)
        cy_t = intrinsics_tgt[:, 1, 2].view(B, 1, 1)

        z_t = tgt_depth[:, 0]  # BxHxW
        x_t = (u - cx_t) * z_t / (fx_t + 1e-8)
        y_t = (v - cy_t) * z_t / (fy_t + 1e-8)
        pts_t = torch.stack([x_t, y_t, z_t], dim=-1)  # BxHxWx3

        # 3. tgt_cam -> world -> src_cam
        # extrinsics_w2c: [R|t] world->cam
        R_t = extrinsics_w2c_tgt[:, :3, :3]  # Bx3x3
        t_t = extrinsics_w2c_tgt[:, :3, 3].view(B, 1, 1, 3)

        # cam_t = R_t * X_world + t_t  =>  X_world = R_t^T (cam_t - t_t)
        R_t_inv = R_t.transpose(1, 2)
        X_world = torch.matmul(pts_t - t_t, R_t_inv.transpose(1, 2))  # BxHxWx3

        R_s = extrinsics_w2c_src[:, :3, :3]  # Bx3x3
        t_s = extrinsics_w2c_src[:, :3, 3].view(B, 1, 1, 3)

        # cam_s = R_s * X_world + t_s
        cam_s = torch.matmul(X_world, R_s.transpose(1, 2)) + t_s  # BxHxWx3
        X_s = cam_s[..., 0]
        Y_s = cam_s[..., 1]
        Z_s = cam_s[..., 2].clamp(min=1e-4)

        # 4. src_cam -> src 像素坐标
        fx_s = intrinsics_src[:, 0, 0].view(B, 1, 1)
        fy_s = intrinsics_src[:, 1, 1].view(B, 1, 1)
        cx_s = intrinsics_src[:, 0, 2].view(B, 1, 1)
        cy_s = intrinsics_src[:, 1, 2].view(B, 1, 1)

        u_s = fx_s * X_s / Z_s + cx_s
        v_s = fy_s * Y_s / Z_s + cy_s

        # 5. 像素坐标 -> grid_sample 的 [-1,1] 标准化网格
        u_norm = 2.0 * (u_s / (W - 1.0)) - 1.0
        v_norm = 2.0 * (v_s / (H - 1.0)) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)  # BxHxWx2

        # 6. 用 grid_sample warp 图像
        warped_img = F.grid_sample(
            src_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # 7. 有效掩码：在 [-1,1] 内且 Z_s > 0
        valid_mask = (
            (u_norm >= -1.0)
            & (u_norm <= 1.0)
            & (v_norm >= -1.0)
            & (v_norm <= 1.0)
            & (Z_s > 1e-4)
        ).float()  # BxHxW
        valid_mask = valid_mask.unsqueeze(1)  # Bx1xHxW

        return warped_img, valid_mask