import math
import re
import torch
import timm
import types

from dataclasses import dataclass
from pathlib import Path

from timm.data import create_transform, resolve_model_data_config  # type: ignore
from torch import nn
from PIL import Image

from heca.environment.scenes.scene import Scene
from heca.misc import logger
from heca.misc.td import TDImage, TDSceneReferences
from heca.entities.entity import Entity
from heca.image_extractors.image_extractor import ImageExtractor

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)

from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn.functional as F

from heca.classes.config import Configurable


class ScoreMode(Enum):
    HIGHEST = "highest"
    AVERAGE = "average"
    RAW = "raw"


class CompareMode(Enum):
    COSINE = "cosine"
    CROSS = "cross_correlation"


class SelectionMode(Enum):
    WEIGHTED_VOTE = "weighted_vote"
    MAJORITY_VOTE = "majority_vote"


class EntityStateKNN(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        top_k: int
        score_mode: ScoreMode
        compare_mode: CompareMode
        selection_mode: SelectionMode

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert (
            self.cfg.compare_mode is CompareMode.COSINE
        ), "Only cosine compare mode is currently implemented."
        assert self.cfg.top_k % 2 == 1, "top_k must be odd."

        self.ref_descs: dict[str, torch.Tensor] = {}
        self.ref_states: dict[str, list[str]] = {}

    def _process_kernel(self, state_desc_kernel: torch.Tensor) -> torch.Tensor:
        state_desc = state_desc_kernel.flatten().unsqueeze(0)  # (1, D)
        state_desc = F.normalize(state_desc, dim=1)
        return state_desc

    def register(
        self,
        entity_label: str,
        state_label: str,
        state_desc_kernel: torch.Tensor,
    ):
        state_desc = self._process_kernel(state_desc_kernel)
        if entity_label not in self.ref_descs:
            self.ref_descs[entity_label] = state_desc
            self.ref_states[entity_label] = [state_label]
        else:
            self.ref_descs[entity_label] = torch.cat(
                [self.ref_descs[entity_label], state_desc],
                dim=0,
            )
            self.ref_states[entity_label].append(state_label)

    def query(
        self,
        entity_label: str,
        state_desc_kernel: torch.Tensor,
    ) -> tuple[str, float]:
        state_desc = self._process_kernel(state_desc_kernel)
        ref_descriptors = self.ref_descs[entity_label]
        state_labels = self.ref_states[entity_label]
        scores = torch.matmul(ref_descriptors, state_desc.T).squeeze(1)
        top_scores, top_indices = torch.topk(scores, k=self.cfg.top_k)

        state_scores = {state: [] for state in state_labels}
        for k_idx, k_score in zip(top_indices.tolist(), top_scores.tolist()):
            state_label = state_labels[k_idx]
            state_scores[state_label].append(float(k_score))

        if self.cfg.selection_mode == SelectionMode.WEIGHTED_VOTE:
            votes = {
                state_label: sum(state_scores[state_label])
                for state_label in state_labels
            }
        elif self.cfg.selection_mode == SelectionMode.MAJORITY_VOTE:
            votes = {
                state_label: len(state_scores[state_label])
                for state_label in state_labels
            }
        else:
            raise ValueError(f"Unsupported selection mode: {self.cfg.selection_mode}")

        prediction = max(votes.items(), key=lambda x: x[1])[0]

        if self.cfg.score_mode == ScoreMode.HIGHEST:
            confidence = max(state_scores[prediction])
        elif self.cfg.score_mode == ScoreMode.AVERAGE:
            confidence = sum(state_scores[prediction]) / len(state_scores[prediction])
        elif self.cfg.score_mode == ScoreMode.RAW:
            confidence = sum(state_scores[prediction])
        return prediction, confidence


class DinoExtractor(ImageExtractor):
    @dataclass(kw_only=True, frozen=True)
    class Query(ImageExtractor.Query):
        label: str = "dino"

    @dataclass(kw_only=True)
    class Config(ImageExtractor.Config):
        stride: int = 8
        thresh: float = 0.5
        center_crop: bool = False
        pad: bool = False
        frozen: bool = True
        taper_sm: int = 25

        state_knn_config: EntityStateKNN.Config = EntityStateKNN.Config(
            top_k=5,
            score_mode=ScoreMode.AVERAGE,
            compare_mode=CompareMode.COSINE,
            selection_mode=SelectionMode.WEIGHTED_VOTE,
        )
        state_patch_radius: int = 2
        sample_per_reference: int = 5
        kp_selection_threshold: float = 0.2
        interpolate_descriptors: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", pretrained=True
        )
        self.model, self.patch_size = DinoExtractor.patch_vit_resolution(
            self.model, self.cfg.stride
        )
        self.model.eval()
        data_config = resolve_model_data_config(self.model)
        self.transforms = create_transform(**data_config, is_training=False)

        self.state_knn = EntityStateKNN.create(self.cfg.state_knn_config)
        self.kp_patch_descr: TDSceneReferences = TDSceneReferences()
        # Temp save for extraction to not recompute descriptors for states after computing them for keypoints.
        self.image_desc: torch.Tensor | None = None

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
            DinoExtractor._fix_pos_enc(patch_size, (stride, stride)), model
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

    def extract_entities(
        self, image: TDImage, entities: list[Entity]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.image_desc = self.compute_descriptor(image.rgb)  # (1, D, H, W)
        kps, scores = self.compute_keypoints(self.image_desc)  # (1, 2*Nref), (1, Nref)
        kps3d = self.kps_2d_to_3d(image, kps)  # (1, Nref, 3)
        return kps3d, kps, scores

    def extract_entity_states(
        self, image: TDImage, entities: list[Entity], kps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #
        if self.image_desc is None:
            self.image_desc = self.compute_descriptor(image.rgb)  # (1, D, H, W)
        # kps_raw_2d is (1, 2*Nref)
        # image_desc is (1, C, H, W)
        assert self.entity_state_coords is not None
        kernels = self.get_state_kernel(
            self.image_desc, kps, self.entity_state_coords
        )  # (1, C, k, k)
        one_hots = []
        scores = []
        for idx, entity in enumerate(entities):
            prediction, score = self.state_knn.query(
                entity_label=entity.cfg.label,
                state_desc_kernel=kernels[idx],
            )
            one_hot_state = entity.state.make_one_hot(prediction)
            one_hots.append(one_hot_state)
            scores.append(score)
        return torch.stack(one_hots, dim=0), torch.stack(scores, dim=0)

    def compute_keypoints(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image_desc is (B, D, H, W) and ref_patch_desc is (Nref, D)
        sm = self.softmax_of_reference_descriptors(image_desc, ref_patch_desc)
        # sm is similarity map
        post = sm
        # When correspondence is (almost) zero across the image, the tensor
        # degenerates (becomes zeros, hence nan after renomalization below).
        # Fix by adding small epsilon.
        # post += 1e-10
        # # normalize to sum to one
        post /= torch.sum(post, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        # Find max similarity for each keypoint (N, Nref) with its score
        confidence = torch.amax(sm, dim=(-1, -2))  # shape: (N, Nref)

        kp_raw_2d = self.get_mode(post)  # (N, 2*Nref)

        return kp_raw_2d, confidence  # sm, post

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

    def get_state_kernel(
        self,
        img_descr: torch.Tensor,
        kps_raw_2d: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        # img_descr is (B, C, H, W)
        # kps_raw_2d is (B, 2*Nref) with xy ordering
        # coords is (Nref, 2) as relative normalized offsets in yx ordering
        B, C, H, W = img_descr.shape
        assert B == 1, "get_state_kernel currently expects batch size 1"
        assert (
            kps_raw_2d.shape[0] == coords.shape[0]
        ), "number of keypoints and state coords must match"
        # Convert keypoints from flattened xy to (B, Nref, yx)
        Nref = coords.shape[0]
        kps_xy = kps_raw_2d.view(B, 2, Nref).permute(0, 2, 1)
        kps_yx = kps_xy.flip(-1)

        # Apply per-reference relative state offsets (normalized yx)
        target_yx_norm = kps_yx + coords.unsqueeze(0)

        # Map normalized yx to descriptor-grid integer indices
        target_yx = self.scale_normalized_coords(
            target_yx_norm,
            (H, W),
        )

        # Pad descriptor map so kernels at borders still have fixed size
        r = self.cfg.state_patch_radius
        kernel_size = 2 * r + 1
        padded = torch.nn.functional.pad(img_descr, (r, r, r, r), mode="replicate")

        y_idx = target_yx[0, :, 0] + r
        x_idx = target_yx[0, :, 1] + r
        kernels = []
        for y, x in zip(y_idx.tolist(), x_idx.tolist()):
            patch = padded[0:1, :, y - r : y + r + 1, x - r : x + r + 1]
            assert patch.shape[-2:] == (kernel_size, kernel_size)
            kernels.append(patch.squeeze(0))

        return torch.stack(kernels, dim=0)  # (Nref, C, k, k)

    def compute_patch_grid_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        # Compute the number of patches along height and width
        h, w = image_size
        grid_h = 1 + (h - self.patch_size) // self.cfg.stride
        grid_w = 1 + (w - self.patch_size) // self.cfg.stride
        return grid_h, grid_w

    def prepare_for_scene(
        self,
        kp_references: dict[Entity, tuple[Image.Image, int, int, int, int]],
        state_references: dict[Entity, dict[str, list[Image.Image]]],
    ):
        assert len(kp_references) == len(state_references)
        for kp_refs, state_refs in zip(kp_references, state_references):
            self.register(entity.cfg.label, kp_refs)
            for state_label, state_imgs in state_refs.items():
                for state_img in state_imgs:
                    self.state_extractor.register(
                        entity.cfg.label, state_label, state_img
                    )

    def add_entity_reference_sample(
        self, dc_ref_patch_desc: torch.Tensor, state_coords: torch.Tensor
    ):
        if self.kp_patch_descr is None or self.entity_state_coords is None:
            self.kp_patch_descr = dc_ref_patch_desc
            self.entity_state_coords = state_coords
        else:
            self.kp_patch_descr = torch.cat(
                (self.kp_patch_descr, dc_ref_patch_desc), dim=0
            )
            self.entity_state_coords = torch.cat(
                (self.entity_state_coords, state_coords), dim=0
            )

    def add_entity_sample_for_cam(
        self,
        kpt: list[tuple[Image.Image, int, int, int, int]],
        kps: dict[str, list[Image.Image]],
    ):
        assert len(kpt) == self.cfg.sample_per_reference
        assert len(kps) == len(entity.cfg.states)
        assert all(state in kps for state in entity.cfg.states)
        assert all(
            len(kps[state]) == self.cfg.sample_per_reference
            for state in entity.cfg.states
        )
        sample_patch_descs = []
        sample_state_coords = []
        for i, (dc_img, dc_x, dc_y, state_x, state_y) in enumerate(kpt):
            dc_img_desc = self.compute_descriptor(dc_img)  # (1, D, H, W)
            NSample, D, H, W = dc_img_desc.shape
            dc_py, dc_px = self.transform_coords(
                dc_x, dc_y, dc_img.height, dc_img.width, H, W
            )
            state_py, state_px = self.transform_coords(
                state_x, state_y, dc_img.height, dc_img.width, H, W
            )
            sample_patch_descs.append(dc_img_desc[0, :, dc_py, dc_px])  # (D,)

            dc_norm = self.normalize_coords(
                torch.tensor([dc_py, dc_px]), self.image_size
            )
            state_norm = self.normalize_coords(
                torch.tensor([state_py, state_px]), self.image_size
            )
            sample_state_coords.append(state_norm - dc_norm)  # (2,)

        dc_ref_patch_desc = torch.stack(sample_patch_descs, dim=0)  # (NSample, D)
        # Average relative state offset across samples → single (1, 2) per entity
        state_coords = (
            torch.stack(sample_state_coords, dim=0).mean(dim=0).unsqueeze(0)
        )  # (1, 2)

        self.add_entity_reference_sample(dc_ref_patch_desc, state_coords)

        for state, images in kps.items():
            for img in images:
                state_img_desc = self.compute_descriptor(img)  # (1, D, H, W)
                kps_raw_2d, _, _, _ = self.compute_keypoints(
                    state_img_desc, dc_ref_patch_desc
                )  # (1, 2)
                kernel = self.get_state_kernel(
                    state_img_desc,
                    kps_raw_2d,
                    state_coords,
                )  # (1, C, k, k)
                self.state_knn.register(
                    entity.cfg.label,
                    state,
                    kernel,
                )
