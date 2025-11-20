from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional
import os
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import logging

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

logger = logging.getLogger(__name__)


@dataclass
class DatasetLyftCfg(DatasetCfgCommon):
    """Configuration for lyft dataset loader"""
    name: str
    roots: list[Path]
    scenes_json_path: Path
    scene_data_json_path: Path
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool


@dataclass
class DatasetLyftCfgWrapper:
    lyft: DatasetLyftCfg


class DatasetLyft(Dataset):
    """Lyft dataset loader for multi-camera scene data with image cropping"""

    cfg: DatasetLyftCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    
    near: float = 0.1
    far: float = 100.0
    
    # Original image dimensions
    ORIGINAL_WIDTH = 1600
    ORIGINAL_HEIGHT = 900
    TARGET_SIZE = 448
    
    def __init__(
        self,
        cfg: DatasetLyftCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.data_root = cfg.roots[0]
        
        # Load scenes and camera data
        self.scenes_data = self._load_json(cfg.scenes_json_path)
        self.scene_data = self._load_json(cfg.scene_data_json_path)
        
        # Parse and match data
        self.scene_list = []  # List of scene info dicts
        self.scene_metadata = {}  # {index: (scene_name, camera_group)}
        
        self._build_scene_index()
        logger.info(f"Custom Dataset: {self.stage}: loaded {len(self.scene_list)} scenes")
    
    def _load_json(self, json_path: Path) -> dict:
        """Load JSON or NDJSON file"""
        try:
            json_path = Path(json_path)
            
            # 检查文件后缀
            if json_path.suffix == '.ndjson':
                # NDJSON 格式：每行一个 JSON 对象
                logger.info(f"Loading NDJSON file: {json_path}")
                data_list = []
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            obj = json.loads(line)
                            data_list.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipped invalid line {line_num}: {e}")
                
                logger.info(f"Loaded {len(data_list)} objects from NDJSON")
                return data_list
            else:
                # 标准 JSON 格式
                logger.info(f"Loading JSON file: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.error(f"Failed to load JSON from {json_path}: {e}")
            raise

    # 修改 _build_scene_index 来处理列表格式
    def _build_scene_index(self) -> None:
        """Build index matching scenes to camera data"""
        # Handle both list and dict formats for scene_data
        if isinstance(self.scene_data, list):
            scene_data_list = self.scene_data
        elif isinstance(self.scene_data, dict):
            if 'scenes_all' in self.scene_data:
                scene_data_list = self.scene_data['scenes_all']
            else:
                scene_data_list = [self.scene_data]
        else:
            scene_data_list = []
        
        # 获取第一个场景的参数作为默认值
        default_scene_data = None
        if scene_data_list and len(scene_data_list) > 0:
            default_scene_data = scene_data_list[0]
            logger.info(f"Using first scene as default: {default_scene_data.get('scene_name', 'unknown')}")
        
        scenes_all = self.scenes_data.get('scenes_all', [])
        
        for scene_idx, scene_info in enumerate(scenes_all):
            # "scene_name": "124189326409808434600001"
            #                1241893264098084346
            scene_name = scene_info.get('scene_name', '')
            dataset_tag = scene_info.get('dataset_tag', '')
            
            # Extract identifiers
            scene_prefix = scene_name[:19]  # First 19 digits
            camera_group_idx = int(scene_name[-1])  # Last digit (0 or 1)
            camera_key = f"camera_{camera_group_idx}"
            
            # Find matching camera data
            matching_scene_data = None
            for scene_data_entry in scene_data_list:
                data_scene_name = scene_data_entry.get('scene_name', '')
                data_suffix = data_scene_name.split('-')[4]  # Extract suffix after 4th hyphen
                
                if data_suffix == scene_prefix:
                    matching_scene_data = scene_data_entry
                    break
            
            # 如果找不到匹配的场景数据，使用第一个场景的参数
            if matching_scene_data is None:
                if default_scene_data is not None:
                    logger.warning(
                        f"No matching camera data for scene {scene_name}, "
                        f"using default from {default_scene_data.get('scene_name', 'unknown')}"
                    )
                    matching_scene_data = default_scene_data
                else:
                    logger.warning(f"No matching camera data for scene {scene_name} and no default available")
                    continue
            
            if camera_key not in matching_scene_data:
                logger.warning(f"Camera group {camera_key} not found for scene {scene_name}")
                continue
            
            # Extract image info with camera indices
            image_data = scene_info.get('images', [])
            if not image_data:
                logger.warning(f"No images found for scene {scene_name}")
                continue
            
            ### scene_name too long
            scene_name = scene_name[:7] + "_" + scene_name[19:]
            # Store scene info with image metadata
            self.scene_list.append({
                'scene_name': scene_name,
                'camera_group': camera_key,
                'image_data': image_data,  # Contains camera_idx for each image
                'camera_data': matching_scene_data[camera_key],
                'metadata': matching_scene_data
            })
            self.scene_metadata[len(self.scene_list) - 1] = (scene_name, camera_key)
        
        logger.info(f"Built scene index: {len(self.scene_list)} scenes loaded")
    
    def _get_crop_box(self, camera_idx: int) -> tuple:
        """
        Get crop box (left, upper, right, lower) based on camera_idx
        
        Args:
            camera_idx: Camera index (0-4)
            
        Returns:
            Tuple of (x1, y1, x2, y2) for PIL Image.crop()
        """
        if camera_idx in [0, 3, 4]:
            # Middle 900x900 square: x from 350 to 1250
            return (350, 0, 1250, 900)
        elif camera_idx == 1:
            # Left 900x900 square: x from 0 to 900
            return (0, 0, 900, 900)
        elif camera_idx == 2:
            # Right 900x900 square: x from 700 to 1600
            return (700, 0, 1600, 900)
        else:
            logger.warning(f"Unknown camera_idx {camera_idx}, using middle crop")
            return (350, 0, 1250, 900)
    
    def _load_single_image(self, img_path: str, camera_idx: int) -> torch.Tensor:
        """
        Load and preprocess a single image
        
        Steps:
        1. Load original image (1600x900)
        2. Crop based on camera_idx
        3. Resize to 448x448
        4. Convert to tensor
        """
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Verify original dimensions
            # if image.size != (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT):
            #     logger.warning(
            #         f"Image {img_path} has unexpected size {image.size}, "
            #         f"expected {(self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT)}"
            #     )
            self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT = image.size
            # Crop based on camera_idx
            # crop_box = self._get_crop_box(camera_idx)
            # image = image.crop(crop_box)
            
            # Resize to target size
            ### add fix resize for non-square input
            if self.cfg.input_image_shape[0] == 448 and self.cfg.input_image_shape[1] == 448:
                ### 如果input_image_shape是448x448，那么直接resize，注意dataset中intrinsics是1600*900，需在后续归一化时进行处理
                image = image.resize((self.TARGET_SIZE, self.TARGET_SIZE), Image.BILINEAR)
            
            # Convert to tensor
            tensor = self.to_tensor(image)
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to load image {img_path} with camera_idx {camera_idx}: {e}")
            raise
    
    def _load_images_parallel(self, image_data_list: list) -> torch.Tensor:
        """
        Load multiple images in parallel
        
        Args:
            image_data_list: List of dicts with 'image_filepath' and 'camera_idx'
        """
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures_with_idx = []
            for idx, img_data in enumerate(image_data_list):
                img_path = img_data.get('image_filepath', '')
                camera_idx = img_data.get('camera_idx', 0)
                full_path = os.path.join(self.data_root, img_path)
                
                futures_with_idx.append(
                    (idx, executor.submit(self._load_single_image, full_path, camera_idx))
                )
            
            torch_images = [None] * len(image_data_list)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
        
        # Check if all images have the same size
        sizes = set(img.shape for img in torch_images)
        if len(sizes) == 1:
            torch_images = torch.stack(torch_images)
        else:
            logger.warning(f"Images have different sizes: {sizes}")
        
        return torch_images
    
    def __len__(self) -> int:
        return len(self.scene_list)
    
    def __getitem__(self, index_tuple: tuple) -> dict:
        """Get item by index tuple (index, num_context_views, patchsize_h)"""
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = self.cfg.input_image_shape[1] // 14
        
        max_retries = 5  # 最多重试 5 次
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Error loading sample {index} (attempt {retry_count}/{max_retries}): {str(e)[:100]}"
                )
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to load sample {index} after {max_retries} attempts")
                    # 返回一个有效的样本而不是崩溃
                    raise
                
                # 尝试下一个样本
                index = (index + 1) % len(self)
        
        raise Exception(f"Failed to load any valid sample after {max_retries} retries")
    
    # 修复 getitem 方法中的外参转换

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        """Load a single scene sample"""
        
        if index >= len(self.scene_list):
            raise IndexError(f"Index {index} out of range")
        
        scene_info = self.scene_list[index]
        scene_name = scene_info['scene_name']
        image_data_list = scene_info['image_data']
        camera_data_list = scene_info['camera_data']
        
        # 提取外参和内参
        extrinsics = []
        intrinsics = []
        
        for cam_info in camera_data_list:
            # 内参已经是调整好的
            intr = cam_info.get('intrinsic_adjusted')
            if intr is None:
                logger.warning(f"Missing intrinsic_adjusted for camera in scene {scene_name}")
                continue
            
            # 确保转换为 numpy 数组
            if isinstance(intr, list):
                intr = np.array(intr, dtype=np.float32)
            else:
                intr = np.asarray(intr, dtype=np.float32)
            
            intrinsics.append(intr)
            
            # 外参 (cam2world) - 确保转换为 numpy 数组
            extr = cam_info.get('extrinsic_cam2world')
            if extr is None:
                logger.warning(f"Missing extrinsic_cam2world for camera in scene {scene_name}")
                continue
            
            if isinstance(extr, list):
                extr = np.array(extr, dtype=np.float32)
            else:
                extr = np.asarray(extr, dtype=np.float32)
            
            extrinsics.append(extr)
        
        if len(extrinsics) == 0 or len(intrinsics) == 0:
            raise ValueError(f"No valid camera data for scene {scene_name}")
        
        # 转换为 numpy 数组，然后转为张量
        extrinsics = np.array(extrinsics, dtype=np.float32)  # Shape: (num_cams, 4, 4)
        intrinsics = np.array(intrinsics, dtype=np.float32)  # Shape: (num_cams, 3, 3)
        
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        # print("shape of extrinsics:", extrinsics.shape)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        # print("shape of intrinsics:", intrinsics.shape)
        
        # 确保数据类型正确
        if not isinstance(extrinsics, torch.Tensor):
            raise TypeError(f"extrinsics must be torch.Tensor, got {type(extrinsics)}")
        if not isinstance(intrinsics, torch.Tensor):
            raise TypeError(f"intrinsics must be torch.Tensor, got {type(intrinsics)}")
        
        logger.debug(f"Extrinsics shape: {extrinsics.shape}, dtype: {extrinsics.dtype}")
        logger.debug(f"Intrinsics shape: {intrinsics.shape}, dtype: {intrinsics.dtype}")
        
        # 在归一化之前，同步打乱 image_data_list, extrinsics, 和 intrinsics
        if self.stage == "train":
            num_views = len(image_data_list)
            shuffled_indices = torch.randperm(num_views)
            print("shuffled_indices:", shuffled_indices)
            image_data_list = [image_data_list[i] for i in shuffled_indices]
            extrinsics = extrinsics[shuffled_indices]
            intrinsics = intrinsics[shuffled_indices]
        
        normalized_intrinsics = intrinsics.clone()
        # 归一化内参到 [0, 1] 范围
        if self.cfg.input_image_shape[0] == 448 and self.cfg.input_image_shape[1] == 448:
            img_h, img_w = self.TARGET_SIZE, self.TARGET_SIZE
            ### intrinsics 还是self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH下的，需要先缩放一次
            s_x = float(img_w) / float(self.ORIGINAL_WIDTH)
            s_y = float(img_h) / float(self.ORIGINAL_HEIGHT)
            normalized_intrinsics[:, 0, 0] *= s_x  # fx
            normalized_intrinsics[:, 1, 1] *= s_y  # fy
            normalized_intrinsics[:, 0, 2] *= s_x  # cx
            normalized_intrinsics[:, 1, 2] *= s_y  # cy            
        else:
            img_h, img_w = self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH
        # print("img_h, img_w:", img_h, img_w)
        # print("intrinsics:", intrinsics)
        
        normalized_intrinsics[:, 0, 0] /= img_w  # fx
        normalized_intrinsics[:, 1, 1] /= img_h  # fy
        normalized_intrinsics[:, 0, 2] /= img_w  # cx
        normalized_intrinsics[:, 1, 2] /= img_h  # cy
        # print("normalized_intrinsics:", normalized_intrinsics)
        
        try:
            # 采样上下文和目标视图
            context_indices, target_indices = self.view_sampler.sample(
                scene_name,
                # num_context_views,
                extrinsics,  # 现在是正确的张量类型
                normalized_intrinsics,
            )
        except ValueError as e:
            raise Exception(f"Not enough frames: {e}")
        
        # 加载图片
        try:
            context_image_data = [image_data_list[i] for i in context_indices]
            target_image_data = [image_data_list[i] for i in target_indices]
            
            context_images = self._load_images_parallel(context_image_data)
            target_images = self._load_images_parallel(target_image_data)
        except Exception as e:
            raise Exception(f"Failed to load images: {e}")
        
        # 占位符深度图
        context_depth = torch.ones_like(context_images)[:, 0]
        target_depth = torch.ones_like(target_images)[:, 0]
        
        # 验证图像形状
        # expected_shape = (3, self.TARGET_SIZE, self.TARGET_SIZE)
        # context_image_invalid = context_images.shape[1:] != expected_shape
        # target_image_invalid = target_images.shape[1:] != expected_shape
        
        # if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
        #     logger.warning(
        #         f"Skipped scene {scene_name}. Context shape: {context_images.shape}, "
        #         f"Target shape: {target_images.shape}"
        #     )
        #     raise Exception("Bad example image shape")
        
        # 应用基线缩放
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                logger.warning(f"Baseline out of range: {scale:.6f}")
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1.0
        
        # 应用相对姿态归一化
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)
        
        # 缩放到单位立方体
        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3]))
            rescale_factor = 1.0 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor
        
        # 检查 NaN/Inf
        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("Encounter NaN or Inf in poses")
        
        # 构建输出样本
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": normalized_intrinsics[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": normalized_intrinsics[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": f"lyft_{scene_name}",
        }
        
        # 应用增强
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        
        intr_aug = self.stage == "train" and self.cfg.intr_augment
        example = apply_crop_shim(
            example,
            (patchsize[0] * 14, patchsize[1] * 14),
            intr_aug=intr_aug
        )
        
        # 反归一化内参回像素坐标
        image_size = example["context"]["image"].shape[2:]
        context_intrinsics = example["context"]["intrinsics"].clone().detach().numpy()
        context_intrinsics[:, 0, 0] *= image_size[1]
        context_intrinsics[:, 1, 1] *= image_size[0]
        context_intrinsics[:, 0, 2] *= image_size[1]
        context_intrinsics[:, 1, 2] *= image_size[0]
        
        target_intrinsics = example["target"]["intrinsics"].clone().detach().numpy()
        target_intrinsics[:, 0, 0] *= image_size[1]
        target_intrinsics[:, 1, 1] *= image_size[0]
        target_intrinsics[:, 0, 2] *= image_size[1]
        target_intrinsics[:, 1, 2] *= image_size[0]
        
        # 占位符 3D 点和掩码
        context_pts3d = torch.ones_like(example["context"]["image"]).permute(0, 2, 3, 1)
        context_valid_mask = torch.ones_like(example["context"]["image"])[:, 0].bool()
        
        target_pts3d = torch.ones_like(target_images).permute(0, 2, 3, 1)
        target_valid_mask = torch.ones_like(target_images)[:, 0].bool()
        
        # 按 3D 点归一化
        if self.cfg.normalize_by_pts3d:
            transformed_pts3d = context_pts3d[context_valid_mask]
            scene_factor = transformed_pts3d.norm(dim=-1).mean().clip(min=1e-8)
            
            context_pts3d /= scene_factor
            example["context"]["depth"] /= scene_factor
            example["context"]["extrinsics"][:, :3, 3] /= scene_factor
            
            target_pts3d /= scene_factor
            example["target"]["depth"] /= scene_factor
            example["target"]["extrinsics"][:, :3, 3] /= scene_factor
        
        example["context"]["pts3d"] = context_pts3d
        example["target"]["pts3d"] = target_pts3d
        example["context"]["valid_mask"] = context_valid_mask * -1
        example["target"]["valid_mask"] = target_valid_mask * -1
        
        return example
    
    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> torch.Tensor:
        """Get near/far bounds for views"""
        from einops import repeat
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    @cached_property
    def index(self) -> dict:
        """Compatibility property"""
        index = {}
        for i, scene_info in enumerate(self.scene_list):
            index[scene_info['scene_name']] = i
        return index