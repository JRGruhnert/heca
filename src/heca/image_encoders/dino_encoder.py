import math
from enum import Enum
import torch
import timm
import types

from dataclasses import dataclass

from timm.data import create_transform, resolve_model_data_config  # type: ignore
from torch import nn
from PIL import Image

import numpy as np

from heca.scenes.scene import Scene
from heca.misc import logger
from heca.data.data import TDImage
from heca.data.reference import Keypoint, Reference
from heca.image_encoders.image_encoder import ImageEncoder
from heca.image_encoders.particle_filter import (
    KeypointParticleFilter,
    ParticleFilterConfig,
)
from heca.image_encoders.prior import KeypointPrior, PriorConfig

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


class TrackingMode(Enum):
    PARTICLE = "particle"
    PRIOR = "prior"
    NONE = "none"


class DinoEncoder(ImageEncoder):
    @dataclass(kw_only=True)
    class Config(ImageEncoder.Config):
        stride: int = 8
        taper_sm: int = 25

        kp_selection_threshold: float = 0.2
        interpolate_descriptors: bool = False
        tracking: TrackingMode = TrackingMode.PARTICLE
        particle_filter: ParticleFilterConfig = ParticleFilterConfig()
        prior: PriorConfig = PriorConfig()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", pretrained=True
        )
        self.model, self.patch_size = DinoEncoder.patch_vit_resolution(
            self.model, self.cfg.stride
        )
        self.model.eval()
        data_config = resolve_model_data_config(self.model)
        self.transforms = create_transform(**data_config, is_training=False)

        self.kp_descriptors: dict[str, torch.Tensor] = {}
        self.kp_labels: list[str] = []
        self.priors: dict[str, KeypointPrior] = {}
        self.filter = KeypointParticleFilter(
            self.cfg.particle_filter, self.cfg.taper_sm
        )

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]):
        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(
        model: nn.Module,
        stride: int,
    ) -> tuple[nn.Module, int]:
        patch_size = model.patch_embed.patch_size
        # print(f"Original patch size: {patch_size}, stride: {stride}")
        assert (
            patch_size[0] == patch_size[1]
        ), "currently only support square patches. else implement ..."
        patch_size = patch_size[0]

        assert stride <= patch_size, "stride cannot be larger than patch size"
        assert patch_size % stride == 0, "patch size must be divisible by stride"

        if stride == patch_size:  # nothing to do
            return model, patch_size

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(  # type: ignore
            DinoEncoder._fix_pos_enc(patch_size, (stride, stride)), model
        )
        return model, patch_size

    def get_image_size(self, image: Image.Image | torch.Tensor) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            return image.height, image.width
        else:
            return image.shape[2], image.shape[3]

    # @measure_runtime
    def compute_descriptor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            prep = self.transforms(image)  # type: ignore
            assert isinstance(prep, torch.Tensor)
            if prep.ndim == 3:
                prep = prep.unsqueeze(0)
            feats: torch.Tensor = self.model.forward_features(prep)
            # output is unpooled, a (1, 261, 4096) shaped tensor
            # [B, 1 + N, C]
        # cls_token = feats[:, 0] # Not using atm
        # register_token = feats[:, 1:5] # Not using atm
        patch_tokens = feats[:, 5:]

        B, N, C = patch_tokens.shape
        # logger.debug(f"Patch tokens shape: {patch_tokens.shape}")
        image_size = self.get_image_size(image)
        grid_h, grid_w = self.compute_patch_grid_size(image_size)
        descr = patch_tokens.reshape(B, grid_h, grid_w, C)
        # logger.debug(f"Reshaped patch tokens to: {descr.shape}")
        descr = descr.permute(0, 3, 1, 2)  # (B, C, H, W)
        # logger.debug(f"Permuted patch tokens to: {descr.shape}")
        if self.cfg.interpolate_descriptors:
            descr = torch.nn.functional.interpolate(
                input=descr,
                size=image_size,
                mode="bilinear",
                align_corners=True,
            )

        return descr

    def reset_tracking(self) -> None:
        """Forget the belief about every keypoint: the next frame starts an episode."""
        self.priors = {label: KeypointPrior(self.cfg.prior) for label in self.kp_labels}
        self.filter.reset()

    def similarity_maps(self, image: TDImage) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.kp_labels:
            raise RuntimeError
        image_desc = self.compute_descriptor(image.rgb)  # (1, D, H, W)
        ref_patch_desc = torch.stack(
            [self.kp_descriptors[label] for label in self.kp_labels], dim=0
        )  # (N, D)
        maps = self.compute_ref_descr_distances(image_desc, ref_patch_desc)
        # (1, N, H, W) -> (N, H, W): each entity's own reference
        per_entity = maps[0, torch.arange(len(self.kp_labels))]
        confidence = per_entity.amax(dim=(-1, -2))
        return per_entity, confidence

    def track_particles(
        self, image: TDImage, similarity: torch.Tensor
    ) -> dict[str, Keypoint]:
        positions, info = self.filter.update(
            similarity, image.d, image.extr, image.intr
        )
        device = image.d.device
        points = torch.as_tensor(positions, dtype=torch.float32, device=device)
        pixels, _ = ImageEncoder.world_to_pixels(points, image.extr, image.intr)
        found: dict[str, Keypoint] = {}
        for idx, label in enumerate(self.kp_labels):
            found[label] = Keypoint(
                xyz=positions[idx],
                score=float(similarity[idx].max()),
                x=int(pixels[idx, 0].item()),
                y=int(pixels[idx, 1].item()),
            )
        return found

    def locate(self, similarity: torch.Tensor, confidence: float, label: str):
        prior = (
            self.priors.get(label) if self.cfg.tracking is TrackingMode.PRIOR else None
        )
        if prior is None:
            posterior = similarity / similarity.sum()
            mode = self.get_mode(posterior.unsqueeze(0).unsqueeze(0))[0]
            return (float(mode[1]), float(mode[0])), True  # (row, column)

        if confidence < self.cfg.prior.threshold and prior.has_belief:
            held = prior.hold()
            assert held is not None
            return held, False
        belief = prior.predict(*similarity.shape[-2:])
        posterior = similarity if belief is None else similarity * belief
        total = posterior.sum()
        posterior = posterior / total if total > 0 else similarity
        mode = self.get_mode(posterior.unsqueeze(0).unsqueeze(0))[0]
        return (float(mode[1]), float(mode[0])), True

    def extract_poses(self, image: TDImage) -> dict[str, Keypoint]:
        maps, confidence = self.similarity_maps(image)
        if self.cfg.tracking is TrackingMode.PARTICLE:
            return self.track_particles(image, maps)

        located: list[tuple[float, float]] = []
        moved: list[bool] = []
        for idx, label in enumerate(self.kp_labels):
            pixel, updated = self.locate(maps[idx], float(confidence[idx]), label)
            if not updated:
                logger.debug(
                    f"{label}: match {float(confidence[idx]):.3f} is below the prior "
                    "threshold, holding the keypoint where the episode left it"
                )
            located.append(pixel)
            moved.append(updated)

        # the located points are normalised to [-1, 1] as (row, column)
        normalised = torch.tensor(
            [[[row, column] for row, column in located]], dtype=torch.float32
        )  # (1, N, 2)
        kps_2d = normalised.flip(-1).reshape(1, -1)  # flattened (x, y) pairs
        kps_3d = self.kps_2d_to_3d(image, kps_2d)  # (1, N, 3)
        pixel_yx = self.scale_normalized_coords(normalised, image.rgb.shape[1:])

        found: dict[str, Keypoint] = {}
        for idx, label in enumerate(self.kp_labels):
            if (
                moved[idx]
                and self.cfg.tracking is TrackingMode.PRIOR
                and label in self.priors
            ):
                # the belief lives on the descriptor grid, so it is kept in the
                # same normalised coordinates locate answered in, not in pixels
                self.priors[label].update(located[idx], float(confidence[idx]))
            found[label] = Keypoint(
                xyz=kps_3d[0, idx].detach().cpu().numpy().astype(np.float64),
                score=float(confidence[idx]),
                x=int(pixel_yx[0, idx, 1]),
                y=int(pixel_yx[0, idx, 0]),
            )
        return found

    def softmax_of_reference_descriptors(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ref_patch_desc is None:
            assert self.kp_patch_descr is not None
            patch_desc = self.kp_patch_descr
        else:
            patch_desc = ref_patch_desc

        if patch_desc.ndim == 2:
            patch_desc = patch_desc.unsqueeze(1)  # (Nref, 1, D)

        Nref, NSample, Dref = patch_desc.shape
        N, D, H, W = image_desc.shape

        patch_desc_flat = patch_desc.view(Nref * NSample, Dref)
        distances = self.compute_ref_descr_distances(image_desc, patch_desc_flat)
        # distances: (N, Nref*NSample, H, W)

        softmax = torch.nn.Softmax(dim=2)
        sm_flat = softmax(distances.view(N, Nref * NSample, H * W) * self.cfg.taper_sm)
        sm = sm_flat.view(N, Nref, NSample, H, W)

        sm_activ = sm.mean(
            dim=2
        )  # (N, Nref, H, W) — average heatmaps, then argmax finds peak
        return sm_activ

    def compute_ref_descr_distances(
        self,
        descriptor_images: torch.Tensor,
        ref_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        N, D, H, W = descriptor_images.shape
        # print("N, D, H, W", N, D, H, W)
        Nref, Dref = ref_descriptor.shape
        # print("Nref, Dref", Nref, Dref)
        assert Dref == D

        descriptor_images = descriptor_images.permute(0, 2, 3, 1)  # N, H, W, D
        descriptor_images = descriptor_images.unsqueeze(3)  # N, H, W, 1, D

        # print(descriptor_images.shape, "should be N, H, W, 1, D")
        descriptor_images = descriptor_images.expand(N, H, W, Nref, D)
        # print(descriptor_images.shape, "should be N, H, W, Nref, D")

        distance = torch.nn.functional.cosine_similarity(
            descriptor_images, ref_descriptor[None, None, None, :], dim=4
        )

        return distance.permute(0, 3, 1, 2)

    def get_mode(self, softmax_activations: torch.Tensor) -> torch.Tensor:
        # need argmax over two last dimensions, so join them first
        B, Nref, H, W = softmax_activations.shape
        sm_flat = softmax_activations.view(B, Nref, -1)
        modes_flat = torch.argmax(sm_flat, dim=2)

        # reshape back to 2D. Note that the new dim is in the front for now.
        modes_2d = modes_flat.unsqueeze(0).repeat((2, 1, 1))

        # get H, W from the flat indeces
        modes_2d[1] = modes_2d[1] // W
        modes_2d[0] = modes_2d[0] % W

        # map from [0, img_size] to [-1, 1] to match pixel_map from spatial exp
        modes_2d = modes_2d.float()
        modes_2d[1] = modes_2d[1] / (H - 1) * 2 - 1
        modes_2d[0] = modes_2d[0] / (W - 1) * 2 - 1

        # move new dim into the middle and flatten to get (N, 2*Nref)
        stacked_2d_features = modes_2d.permute((1, 0, 2))
        stacked_2d_features = stacked_2d_features.reshape(B, -1)

        return stacked_2d_features

    def prepare_for_scene(self, config: Scene.Config):
        scene = Scene.get(config)
        if not scene.references_ready():
            raise RuntimeError
        self.kp_descriptors = {}
        self.kp_labels = []
        for label, entity in scene.entities.items():
            reference = scene.references[label][Reference.POSITION]
            image_desc = self.compute_descriptor(reference.image)  # (1, D, H, W)
            dc_py, dc_px = self.transform_coords(
                reference.x,
                reference.y,
                reference.image.height,
                reference.image.width,
                image_desc.shape[2],
                image_desc.shape[3],
            )
            self.kp_descriptors[label] = image_desc[0, :, dc_py, dc_px]
            self.kp_labels.append(label)
        self.reset_tracking()
