import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from heca.data.data import TDImage
from heca.data.reference import Keypoint, Reference
from heca.image_encoders.image_encoder import device_from_spec, ImageEncoder
from heca.misc import hardware, logger
from heca.scenes.scene import Scene


class RomaEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        # romav2 presets: turbo (320, fastest), fast (512), base (640), precise
        setting: str = "turbo"
        descriptor: str = "dinov3_vitl16"
        half: bool = True
        # visible card, "" = the process-wide hardware.device
        device: str = ""
        backend: str = "roma"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = device_from_spec(cfg.device) or hardware.device
        half = cfg.half and self.device.type == "cuda"
        try:
            from romav2 import RoMaV2
            import romav2.romav2 as module
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "RomaEncoder needs the romav2 package: pip install romav2"
            ) from error

        module.device = self.device
        for name, loaded in list(sys.modules.items()):
            if name.split(".")[0] == "romav2" and hasattr(loaded, "device"):
                loaded.device = self.device
        template = RoMaV2.Cfg()
        self.model = RoMaV2(
            RoMaV2.Cfg(
                descriptor=type(template.descriptor)(
                    name=cfg.descriptor, enable_amp=half
                ),
                matcher=type(template.matcher)(enable_amp=half),
                setting=cfg.setting,
            )
        ).to(self.device)
        logger.info(
            f"{type(self).__name__}: {cfg.descriptor} ({cfg.setting}) on {self.device}"
        )

        self.kp_labels: list[str] = []
        self.references: dict[str, tuple[Image.Image, int, int]] = {}

    def prepare_for_scene(self, config: Scene.Config):
        scene = Scene.get(config)
        if not scene.references_ready():
            missing = [
                f"{label}.{name}"
                for label, entity in scene.entities.items()
                for name in (Reference.POSITION, *entity.reference_names)
                if name not in scene.references.get(label, {})
            ]
            raise RuntimeError(
                f"{config.tag}: no complete reference set (missing "
                f"{', '.join(missing)}); click them with "
                f"scripts/d01_select_references.py --scene {config.tag}"
            )
        self.references = {}
        self.kp_labels = []
        for label in scene.entities:
            reference = scene.references[label][Reference.POSITION]
            self.references[label] = (reference.image, reference.x, reference.y)
            self.kp_labels.append(label)

    def reset_tracking(self) -> None:
        """Nothing to forget: every frame is matched against the references."""
        return None

    def extract_states(self, image: TDImage) -> dict[str, tuple[int, float]]:
        return {}

    @staticmethod
    def _normalised(pixel: float, size: int) -> float:
        """Pixel index -> coordinate of the pixel centre in [-1, 1]."""
        return 2.0 * (pixel + 0.5) / max(size, 1) - 1.0

    def reference_pixel(self, label: str, frame: TDImage) -> tuple[int, int, float]:
        """Where the annotated reference pixel lands in ``frame``, and how sure."""
        reference, x, y = self.references[label]
        rgb = frame.rgb
        if rgb.ndim == 3:
            rgb = rgb[None]
        preds = self.model.match(reference, rgb.to(self.device))

        warp = preds["warp_AB"]
        confidence = preds["confidence_AB"]
        warp = (warp[0] if warp.ndim == 4 else warp).to(self.device).permute(2, 0, 1)
        confidence = (
            (confidence[0] if confidence.ndim == 4 else confidence)
            .to(self.device)
            .permute(2, 0, 1)
        )
        if confidence.shape[0] != 1:
            confidence = confidence.mean(dim=0, keepdim=True)

        grid = torch.tensor(
            [
                [
                    [
                        [
                            self._normalised(x, reference.width),
                            self._normalised(y, reference.height),
                        ]
                    ]
                ]
            ],
            dtype=warp.dtype,
            device=self.device,
        )
        point = F.grid_sample(warp[None], grid, mode="bilinear", align_corners=False)
        score = F.grid_sample(
            confidence[None], grid, mode="bilinear", align_corners=False
        )

        height, width = int(frame.rgb.shape[-2]), int(frame.rgb.shape[-1])
        to_pixel = lambda value, size: (value + 1.0) * size / 2.0 - 0.5
        x_frame = to_pixel(float(point[0, 0, 0, 0]), width)
        y_frame = to_pixel(float(point[0, 1, 0, 0]), height)
        # the confidence is an unbounded log-likelihood; it is reported as measured
        # because nothing gates on an absolute value any more
        return (
            int(np.clip(round(x_frame), 0, width - 1)),
            int(np.clip(round(y_frame), 0, height - 1)),
            float(score[0, 0, 0, 0]),
        )

    def extract_poses(self, image: TDImage) -> dict[str, Keypoint]:
        found: dict[str, Keypoint] = {}
        for label in self.kp_labels:
            x, y, score = self.reference_pixel(label, image)
            xyz = ImageEncoder.hard_pixels_to_3D_world(
                torch.tensor([[y]], dtype=torch.long),
                torch.tensor([[x]], dtype=torch.long),
                image.d[None],
                image.extr[None],
                image.intr[None],
            )[0, 0]
            found[label] = Keypoint(
                xyz=xyz.detach().cpu().numpy().astype(np.float64),
                score=score,
                x=x,
                y=y,
            )
        return found
