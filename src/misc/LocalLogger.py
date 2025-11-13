import os
from pathlib import Path
from typing import Any, Optional, Union
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
import shutil

DEFAULT_LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    # def __init__(self) -> None:
    #     super().__init__()
    #     self.experiment = None
    #     os.system(f"rm -r {LOG_PATH}")
    def __init__(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        """
        一个简易的本地替代 Logger。
        当未传入 log_dir 时，使用 DEFAULT_LOG_PATH。
        """
        super().__init__()
        self.experiment = None
        self.log_path = Path(log_dir) if log_dir is not None else DEFAULT_LOG_PATH

        # 清理并准备目录
        # shutil.rmtree(self.log_path, ignore_errors=True)

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        captions: Optional[list[str]] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{captions[0]}_{index:0>2}_{step:0>6}.jpg"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
