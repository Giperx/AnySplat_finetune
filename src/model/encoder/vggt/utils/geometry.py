# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch


def batch_unproject_to_world_with_attributes(
    depth_maps: torch.Tensor, 
    conf_maps: torch.Tensor,
    initial_valid_mask: torch.Tensor,
    extrinsics_w2c: torch.Tensor, 
    intrinsics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将一批深度图反投影到世界坐标，并携带置信度和初始有效性属性。

    Args:
        depth_maps (torch.Tensor): (B, V, H, W).
        conf_maps (torch.Tensor): (B, V, H, W).
        initial_valid_mask (torch.Tensor): 初始有效性布尔掩码, (B, V, H, W).
        extrinsics_w2c (torch.Tensor): (B, V, ...).
        intrinsics (torch.Tensor): (B, V, 3, 3).

    Returns:
        - 3D世界坐标点云, (B, V*H*W, 3).
        - 点云对应的置信度, (B, V*H*W).
        - 点云对应的初始有效性, (B, V*H*W).
    """
    B, V, H, W = depth_maps.shape
    device = depth_maps.device

    # 展平所有输入
    depth_flat = depth_maps.view(B * V, H, W)
    conf_flat = conf_maps.view(B * V, H, W)
    intrinsics_flat = intrinsics.view(B * V, 3, 3)
    extrinsics_w2c_flat = extrinsics_w2c.view(B * V, extrinsics_w2c.shape[-2], extrinsics_w2c.shape[-1])
    
    # 生成像素坐标网格
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    u = u.expand(B * V, H, W)
    v = v.expand(B * V, H, W)

    # 像素 -> 相机坐标
    fx = intrinsics_flat[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics_flat[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics_flat[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics_flat[:, 1, 2].view(-1, 1, 1)

    z_cam = depth_flat
    x_cam = (u - cx) * z_cam / (fx + 1e-8)
    y_cam = (v - cy) * z_cam / (fy + 1e-8)
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)

    # 相机 -> 世界坐标
    cam_to_world_flat = closed_form_inverse_se3(extrinsics_w2c_flat)
    homo_cam_coords = torch.cat((cam_coords.view(B * V, -1, 3), torch.ones(B * V, H * W, 1, device=device)), dim=-1)
    world_coords_homo = homo_cam_coords @ cam_to_world_flat.transpose(1, 2)
    world_coords_flat = world_coords_homo[..., :3] / (world_coords_homo[..., 3:] + 1e-8)

    # 恢复 Batch 维度，并返回所有属性
    world_pts = world_coords_flat.view(B, V * H * W, 3)
    world_confs = conf_flat.view(B, V * H * W)
    world_validity = initial_valid_mask.view(B, V * H * W)
    
    return world_pts, world_confs, world_validity

def batch_project_from_world_with_attributes(
    world_pts: torch.Tensor,
    world_confs: torch.Tensor,
    world_validity: torch.Tensor,
    extrinsics_c2w_target: torch.Tensor,
    intrinsics_target: torch.Tensor,
    target_H: int,
    target_W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将携带属性的世界点云投影到新视角，生成深度图、置信度图和有效掩码。

    Returns:
        - 新的深度图, (B, V_target, H, W).
        - 新的置信度图, (B, V_target, H, W).
        - 新的有效掩码 (布尔型), (B, V_target, H, W).
    """
    B, N, _ = world_pts.shape
    V_target = extrinsics_c2w_target.shape[1]
    device = world_pts.device

    extrinsics_w2c_target = torch.inverse(extrinsics_c2w_target)
    homo_world_pts = torch.cat((world_pts, torch.ones(B, N, 1, device=device)), dim=-1).unsqueeze(1)
    
    # 世界坐标 -> 目标相机坐标
    cam_coords_homo = homo_world_pts @ extrinsics_w2c_target.transpose(-1, -2)
    cam_coords = cam_coords_homo[..., :3] / (cam_coords_homo[..., 3:] + 1e-8)
    depth_vals = cam_coords[..., 2]

    # 目标相机坐标 -> 像素坐标
    pixel_coords_homo = cam_coords @ intrinsics_target.transpose(-1, -2)
    u_proj = pixel_coords_homo[..., 0] / (pixel_coords_homo[..., 2] + 1e-8)
    v_proj = pixel_coords_homo[..., 1] / (pixel_coords_homo[..., 2] + 1e-8)
    
    # 筛选在相机视锥内的点
    valid_projection_mask = (depth_vals > 1e-4) & \
                            (u_proj >= 0) & (u_proj < target_W - 1) & \
                            (v_proj >= 0) & (v_proj < target_H - 1) & \
                            (world_pts.norm(dim=-1).unsqueeze(1) > 1e-4)

    # 初始化深度、置信度、有效性三个缓冲区
    depth_buffer = torch.full((B * V_target, target_H * target_W), float('inf'), device=device)
    conf_buffer = torch.zeros((B * V_target, target_H * target_W), device=device)
    validity_buffer = torch.zeros((B * V_target, target_H * target_W), dtype=torch.bool, device=device)

    # 展平所有维度进行光栅化
    u_flat, v_flat = u_proj.view(B * V_target, N), v_proj.view(B * V_target, N)
    d_flat = depth_vals.view(B * V_target, N)
    c_flat = world_confs.view(B, 1, N).expand(B, V_target, N).reshape(B * V_target, N)
    vld_flat = world_validity.view(B, 1, N).expand(B, V_target, N).reshape(B * V_target, N)
    mask_flat = valid_projection_mask.view(B * V_target, N)
    
    # 第一次传递：找到每个像素的最小深度
    for i in range(B * V_target):
        if not mask_flat[i].any(): continue
        
        u_i, v_i, d_i = u_flat[i][mask_flat[i]], v_flat[i][mask_flat[i]], d_flat[i][mask_flat[i]]
        flat_indices = v_i.long() * target_W + u_i.long()
        depth_buffer[i].scatter_reduce_(0, flat_indices, d_i, reduce="amin", include_self=False)

    # 第二次传递：找到赢得深度测试的点的属性（置信度和初始有效性）
    for i in range(B * V_target):
        if not mask_flat[i].any(): continue
        
        # 找出哪些点投影到了被填充的像素上
        u_i, v_i = u_flat[i][mask_flat[i]], v_flat[i][mask_flat[i]]
        flat_indices = v_i.long() * target_W + u_i.long()
        
        # 检查这些点的深度是否与深度缓冲区中的最小深度匹配（即深度测试的获胜者）
        winner_mask = torch.isclose(d_flat[i][mask_flat[i]], depth_buffer[i][flat_indices])
        
        # 获取获胜者的索引和属性
        winner_indices = flat_indices[winner_mask]
        winner_confs = c_flat[i][mask_flat[i]][winner_mask]
        winner_validity = vld_flat[i][mask_flat[i]][winner_mask]
        
        # 将获胜者的属性写入对应的缓冲区
        conf_buffer[i].scatter_(0, winner_indices, winner_confs)
        validity_buffer[i].scatter_(0, winner_indices, winner_validity)

    # 恢复维度并返回结果
    new_depth_maps = depth_buffer.view(B, V_target, target_H, target_W)
    new_depth_maps[torch.isinf(new_depth_maps)] = 0
    
    new_conf_maps = conf_buffer.view(B, V_target, target_H, target_W)
    final_valid_mask = validity_buffer.view(B, V_target, target_H, target_W)

    return new_depth_maps, new_conf_maps, final_valid_mask

def reproject_depth_maps_batch_with_conf(
    distill_depth_map: torch.Tensor,
    distill_depth_conf: torch.Tensor,
    distill_extrinsic_w2c: torch.Tensor,
    distill_intrinsic: torch.Tensor,
    extrinsics_c2w_B: torch.Tensor,
    intrinsics_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将一批带置信度的多视角深度图重投影到一组新的相机视角。

    Returns:
        - 生成的新深度图, (B, V_target, H, W).
        - 生成的新置信度图, (B, V_target, H, W).
        - 生成的有效像素布尔掩码, (B, V_target, H, W).
    """
    if distill_depth_map.dim() == 5:
        distill_depth_map = distill_depth_map.squeeze(-1)
    if distill_depth_conf.dim() == 5:
        distill_depth_conf = distill_depth_conf.squeeze(-1)
        
    H, W = distill_depth_map.shape[-2:]
    
    # print("model pred ext w2c:", distill_extrinsic_w2c)
    
    # --- 预处理相机内外参 (保持不变) ---
    B, V = distill_depth_map.shape[:2]
    extrinsic_padding = (
        torch.tensor([0, 0, 0, 1], device=distill_extrinsic_w2c.device, dtype=distill_extrinsic_w2c.dtype)
        .view(1, 1, 1, 4)
        .repeat(B, V, 1, 1)
    )
    if distill_extrinsic_w2c.shape[-2] == 3:
        distill_extrinsic_w2c = torch.cat([distill_extrinsic_w2c.clone(), extrinsic_padding], dim=-2)
    
    pred = torch.linalg.inv(distill_extrinsic_w2c)
    # print("model pred ext c2w inv:", pred)
    # pred = closed_form_inverse_se3(distill_extrinsic_w2c.view(B * V, 4, 4)).view(B, V, 4, 4)
    # print("model pred ext c2w se3:", pred)
    # pred = distill_extrinsic_w2c.inverse()
    # print("model pred ext c2w torch inverse():", pred)
    
    gt = extrinsics_c2w_B.clone()
    # print("gt ext without alin:", gt)
    X = torch.matmul(pred[:, 0], torch.linalg.inv(gt[:, 0]))
    extrinsics_c2w_B_aligned = torch.matmul(X[:, None], gt)
    # print("extrinsics_c2w_B_aligned:", extrinsics_c2w_B_aligned)
    # extrinsics_c2w_B_aligned[:, 0] = pred[:, 0]  # 保持第0个视角和pred一致

    ### 同时内参在加载时已经归一化到 [0,1]，这里需要还原回像素坐标
    # intrinsics_B_aligned = intrinsics_B.clone()
    # intrinsics_B_aligned[:, :, 0, 0] *= W    # fu
    # intrinsics_B_aligned[:, :, 1, 1] *= H    # fv
    # intrinsics_B_aligned[:, :, 0, 2] *= W    # cu
    # intrinsics_B_aligned[:, :, 1, 2] *= H    # cv

    intrinsics_B_aligned = distill_intrinsic # 使用distill的内参
    ### -----------------------------

    # 步骤 1: 在最开始就根据源置信度计算初始有效性掩码
    conf_threshold = torch.quantile(
        distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True
    ).unsqueeze(-1)
    initial_valid_mask = distill_depth_conf > conf_threshold
    
    # 步骤 2: 反投影到世界坐标，携带深度、置信度、初始有效性三个属性
    world_pts, world_confs, world_validity = batch_unproject_to_world_with_attributes(
        distill_depth_map, distill_depth_conf, initial_valid_mask, distill_extrinsic_w2c, distill_intrinsic
    )

    # 步骤 3: 将世界点云及其属性投影到新视角
    new_depth_maps, new_conf_maps, final_valid_mask = batch_project_from_world_with_attributes(
        world_pts, world_confs, world_validity, extrinsics_c2w_B_aligned, intrinsics_B_aligned, H, W
    )
    
    return new_depth_maps, new_conf_maps, final_valid_mask

def batchify_project_world_to_new_depth(
    depth_map_A: torch.Tensor,
    distill_depth_conf_A: torch.Tensor,      # (B, V, H, W) or (B, V, H, W, 1)
    extrinsics_world2cam_A: torch.Tensor,    # (B, V, 3, 4) world -> camA
    intrinsics_A: torch.Tensor,              # (B, V, 3, 3)
    extrinsics_cam2world_B: torch.Tensor,    # (B, V, 4, 4) camB -> world
    intrinsics_B: torch.Tensor,              # (B, V, 3, 3)
    quantile_value: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用原相机(A)的 depth map + conf 和其内外参，先还原到世界坐标，
    再使用新相机(B)的内外参，得到对应的新 depth_map_B、depth_conf_B
    以及基于 depth_conf_B 的 valid_mask_B。

    Args:
        depth_map_A: (B, V, H, W) or (B, V, H, W, 1)
        distill_depth_conf_A: (B, V, H, W) or (B, V, H, W, 1)，与 depth_map_A 对应的置信度
        extrinsics_world2cam_A: (B, V, 3, 4) world -> camA
        intrinsics_A: (B, V, 3, 3)
        extrinsics_cam2world_B: (B, V, 4, 4) camB -> world
        intrinsics_B: (B, V, 3, 3)
        quantile_value: 用于生成 valid_mask_B 的分位数阈值，例如 0.3

    Returns:
        depth_map_B:   (B, V, H, W) 新相机(B)下的深度（相机坐标 z）
        depth_conf_B:  (B, V, H, W) 新相机(B)下的置信度（无效像素为 0）
        valid_mask_B:  (B, V, H, W) bool mask，基于 depth_conf_B 的分位数阈值
    """
    print("intrinsics_A:", intrinsics_A)
    print("intrinsics_B:", intrinsics_B)
    print("extrinsics_world2cam_A:", extrinsics_world2cam_A)
    print("extrinsics_cam2world_B:", extrinsics_cam2world_B)
    # -----------------------------
    # 1. 统一 depth, conf 形状
    # -----------------------------
    if depth_map_A.dim() == 5:
        depth_map_A = depth_map_A.squeeze(-1)          # (B, V, H, W)
    if distill_depth_conf_A.dim() == 5:
        distill_depth_conf_A = distill_depth_conf_A.squeeze(-1)  # (B, V, H, W)

    B, V, H, W = depth_map_A.shape

    # -----------------------------
    # 2. 从 A 相机下的 depth 还原世界坐标
    # -----------------------------
    depth_flat = depth_map_A.flatten(0, 1)                    # (S, H, W)
    intr_A_flat = intrinsics_A.flatten(0, 1)                  # (S, 3, 3)
    extr_A_flat = extrinsics_world2cam_A.flatten(0, 1)        # (S, 3, 4)
    conf_flat_A = distill_depth_conf_A.flatten(0, 1)          # (S, H, W)

    S = depth_flat.shape[0]

    fu_A = intr_A_flat[:, 0, 0]  # (S,)
    fv_A = intr_A_flat[:, 1, 1]  # (S,)
    cu_A = intr_A_flat[:, 0, 2]  # (S,)
    cv_A = intr_A_flat[:, 1, 2]  # (S,)

    # 像素网格 (u, v)
    u = torch.arange(W, device=depth_map_A.device)[None, None, :].expand(S, H, W)  # (S, H, W)
    v = torch.arange(H, device=depth_map_A.device)[None, :, None].expand(S, H, W)  # (S, H, W)

    # 在 A 相机坐标系下的 3D 坐标
    z_camA = depth_flat
    x_camA = (u - cu_A[:, None, None]) * z_camA / fu_A[:, None, None]
    y_camA = (v - cv_A[:, None, None]) * z_camA / fv_A[:, None, None]

    camA_coords = torch.stack([x_camA, y_camA, z_camA], dim=-1)  # (S, H, W, 3)

    # 将 world2cam(A) 扩为 4x4，求其逆 => camA2world
    R_A = extr_A_flat[:, :, :3]                                # (S, 3, 3)
    t_A = extr_A_flat[:, :, 3:]                                # (S, 3, 1)

    camA2world = torch.zeros(S, 4, 4, device=depth_map_A.device, dtype=depth_map_A.dtype)
    camA2world[:, :3, :3] = R_A.transpose(1, 2)                # R^-1
    camA2world[:, :3, 3:] = -R_A.transpose(1, 2) @ t_A         # -R^-1 t
    camA2world[:, 3, 3] = 1.0

    # 齐次坐标
    homo_camA = torch.cat(
        [camA_coords, torch.ones_like(camA_coords[..., :1])], dim=-1
    )  # (S, H, W, 4)
    homo_camA_flat = homo_camA.view(S, H * W, 4)               # (S, HW, 4)

    # 世界坐标
    world_coords = torch.bmm(homo_camA_flat, camA2world.transpose(1, 2))  # (S, HW, 4)
    world_coords = world_coords[..., :3]                                   # (S, HW, 3)

    # 同步展平 conf（与 HW 对应位置一一对应）
    conf_flat_A = conf_flat_A.view(S, H * W)                               # (S, HW)

    # -----------------------------
    # 3. 使用新相机 B 的内外参进行投影
    # -----------------------------
    ### CamB由GT而来，camera0自然是对齐到了世界坐标原点；而predict camA中camera0则离原点有偏移，需要刚性变换对齐
    ### 对gt pose做刚体变换，对齐camera1，因为pred camera1的pose相对世界中心有偏差
    # pred_all_extrinsic, gt_ext: (B=1, V=3, 4, 4), 都是 cam2world
    # extrinsic_padding = (
    #     torch.tensor([0, 0, 0, 1], device=extrinsics_world2cam_A.device, dtype=extrinsics_world2cam_A.dtype)
    #     .view(1, 1, 1, 4)
    #     .repeat(B, V, 1, 1)
    # )
    # extrinsics_world2cam_A = torch.cat([extrinsics_world2cam_A.clone(), extrinsic_padding], dim=2)  # BxVx4x4
    # pred = torch.linalg.inv(extrinsics_world2cam_A)
    pred = camA2world.clone().view(B, V, 4, 4)    # BxVx4x4
    gt = extrinsics_cam2world_B.clone()
    # 变换 X = pred_c2w_1 @ inv(gt_c2w_1) ；这里的索引0是“camera1”
    X = torch.matmul(pred[:, 0], torch.linalg.inv(gt[:, 0]))     # (B, 4, 4)
    # 左乘到所有视角（会按批广播到 V）
    extrinsics_cam2world_B = torch.matmul(X[:, None], gt)                        # (B, V, 4, 4)
    
    ### 同时内参在加载时已经归一化到 [0,1]，这里需要还原回像素坐标
    intrinsics_B = intrinsics_B.clone()
    intrinsics_B[:, :, 0, 0] *= W    # fu
    intrinsics_B[:, :, 1, 1] *= H    # fv
    intrinsics_B[:, :, 0, 2] *= W    # cu
    intrinsics_B[:, :, 1, 2] *= H    # cv
    intrinsics_B = intrinsics_A
    ### -----------------------------
    print("intrinsics_B after rescale:", intrinsics_B)
    print("extrinsics_cam2world_B after rigid transform:", extrinsics_cam2world_B)
    intr_B_flat = intrinsics_B.flatten(0, 1)                   # (S, 3, 3)
    camB2world_flat = extrinsics_cam2world_B.flatten(0, 1)     # (S, 4, 4)

    # 求 world2cam_B = (camB2world)^-1
    R_B = camB2world_flat[:, :3, :3]                           # (S, 3, 3)
    t_B = camB2world_flat[:, :3, 3:]                           # (S, 3, 1)

    world2cam_B = torch.zeros_like(camB2world_flat)
    world2cam_B[:, :3, :3] = R_B.transpose(1, 2)               # R^-1
    world2cam_B[:, :3, 3:] = -R_B.transpose(1, 2) @ t_B        # -R^-1 t
    world2cam_B[:, 3, 3] = 1.0

    # 世界坐标 -> B 相机坐标
    homo_world = torch.cat(
        [world_coords, torch.ones_like(world_coords[..., :1])], dim=-1
    )  # (S, HW, 4)
    camB_coords = torch.bmm(homo_world, world2cam_B.transpose(1, 2))  # (S, HW, 4)
    camB_coords = camB_coords[..., :3]                                # (S, HW, 3)

    x_B = camB_coords[..., 0]  # (S, HW)
    y_B = camB_coords[..., 1]
    z_B = camB_coords[..., 2]

    # 投影到像素平面
    fu_B = intr_B_flat[:, 0, 0]  # (S,)
    fv_B = intr_B_flat[:, 1, 1]  # (S,)
    cu_B = intr_B_flat[:, 0, 2]
    cv_B = intr_B_flat[:, 1, 2]

    # 避免除以 0
    eps = 1e-6
    z_safe = torch.where(z_B.abs() < eps, z_B.sign() * eps, z_B)

    u_B = fu_B[:, None] * (x_B / z_safe) + cu_B[:, None]   # (S, HW)
    v_B = fv_B[:, None] * (y_B / z_safe) + cv_B[:, None]   # (S, HW)

    # -----------------------------
    # 4. 基于几何条件的初始有效性（在视野内且 z>0）
    # -----------------------------
    valid_z = z_B > 0
    valid_u = (u_B >= 0) & (u_B <= (W - 1))
    valid_v = (v_B >= 0) & (v_B <= (H - 1))
    geom_valid = valid_z & valid_u & valid_v                      # (S, HW)

    # 先根据几何有效性，把不合法的深度和置信度置 0
    z_B = torch.where(geom_valid, z_B, torch.zeros_like(z_B))
    conf_B = torch.where(geom_valid, conf_flat_A, torch.zeros_like(conf_flat_A))

    # -----------------------------
    # 5. 重排成 (B, V, H, W)
    # -----------------------------
    depth_B = z_B.view(B, V, H, W)
    depth_conf_B = conf_B.view(B, V, H, W)

    # -----------------------------
    # 6. 基于 depth_conf_B 的 valid_mask_B
    #    conf_threshold = quantile(depth_conf_B, q, per-view)
    # -----------------------------
    # (B, V, H, W) -> (B, V, H*W)，在每个 (B, V) 上做 quantile
    conf_view = depth_conf_B.flatten(2, 3)  # (B, V, H*W)

    # 如果某些 view 全 0，quantile 会是 0，下面的 > 会给出全 False 的 mask，符合“没有像素”的语义
    conf_threshold = torch.quantile(
        conf_view,
        quantile_value,
        dim=-1,
        keepdim=True
    )  # (B, V, 1)

    # 恢复到 (B, V, H, W)，做阈值比较
    conf_threshold_hw = conf_threshold.unsqueeze(-1)  # (B, V, 1, 1)
    valid_mask_B = depth_conf_B > conf_threshold_hw   # (B, V, H, W), bool

    return depth_B, depth_conf_B, valid_mask_B

def batchify_unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (B, V, H, W, 1) or (B, V, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (B, V, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (B, V, 3, 3)
        
    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """

    # Handle both (S, H, W, 1) and (S, H, W) cases
    if depth_map.dim() == 5:
        depth_map = depth_map.squeeze(-1)  # (S, H, W)
        
    # Generate batched camera coordinates
    H, W = depth_map.shape[2:]
    batch_size, num_views = depth_map.shape[0], depth_map.shape[1]
    
    # Intrinsic parameters (S, 3, 3)
    intrinsics_cam, extrinsics_cam, depth_map = intrinsics_cam.flatten(0, 1), extrinsics_cam.flatten(0, 1), depth_map.flatten(0, 1)
    fu = intrinsics_cam[:, 0, 0]  # (S,)
    fv = intrinsics_cam[:, 1, 1]  # (S,)
    cu = intrinsics_cam[:, 0, 2]  # (S,)
    cv = intrinsics_cam[:, 1, 2]  # (S,)
    
    # Generate grid of pixel coordinates
    u = torch.arange(W, device=depth_map.device)[None, None, :].expand(batch_size * num_views, H, W)  # (S, H, W)
    v = torch.arange(H, device=depth_map.device)[None, :, None].expand(batch_size * num_views, H, W)  # (S, H, W)
    
    # Unproject to camera coordinates (S, H, W, 3)
    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map
    
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)  # (S, H, W, 3)
    
    # Transform to world coordinates
    cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)

    # homo transformation
    homo_pts = torch.cat((cam_coords, torch.ones_like(cam_coords[..., :1])), dim=-1).flatten(1, 2)
    world_coords = torch.bmm(cam_to_world, homo_pts.transpose(1, 2)).transpose(1, 2)[:, :, :3].view(batch_size*num_views, H, W, 3)
    
    return world_coords.view(batch_size, num_views, H, W, 3)

def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)
        
    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = torch.stack(world_points_list, dim=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    eps=1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W, device=depth_map.device), 
                         torch.arange(H, device=depth_map.device), 
                         indexing='xy')

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(dtype=torch.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, 4, device=R.device)[None].repeat(len(R), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
