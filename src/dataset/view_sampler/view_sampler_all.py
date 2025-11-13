from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerAllCfg:
    name: Literal["all"]
    num_context_views: int
    num_target_views: int
    max_img_per_gpu: int

class ViewSamplerAll(ViewSampler[ViewSamplerAllCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        v, _, _ = extrinsics.shape
        all_frames = torch.arange(v, device=device)
        # 修改：第二个返回值只是 all_frames 的第一个元素
        # 保持张量数据类型
        target_indices = all_frames[:1]  # 只取第一个元素，但保持为张量形式
        
        # return all_frames, target_indices
        return all_frames, all_frames

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
