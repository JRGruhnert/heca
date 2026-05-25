import math
import types
from dataclasses import dataclass
from enum import Enum

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config  # type: ignore
from torch import nn

from heca.classes.config import Configurable
from heca.entities.entity import Entity
from heca.environment.scenes.knn import (
    CompareMode,
    EntityStateKNN,
    ScoreMode,
    SelectionMode,
)
from heca.misc import logger
from heca.misc.td import TDImage

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


class StateExtractionMode(Enum):
    IMAGE = "image"
    COORDINATE = "coordinate"


class ImageExtractor(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
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
        kp_selection_threshold: float = 0.2
        interpolate_descriptors: bool = False
        state_extraction_mode: StateExtractionMode = StateExtractionMode.IMAGE

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            # num_classes=0,  # remove classifier nn.Linear
        )
        self.model, self.patch_size = ImageExtractor.patch_vit_resolution(
            self.model, self.cfg.stride
        )
        self.model.eval()
        data_config = resolve_model_data_config(self.model)
        self.transforms = create_transform(**data_config, is_training=False)

        self.entities: list[Entity] = []
        self.cams: set[str] = set()
        self.entity_state_knn = EntityStateKNN.create(self.cfg.state_knn_config)
        self.kp_patch_descr: torch.Tensor | None = None
        self.entity_state_coords: torch.Tensor | None = None
        self.image_size: tuple[int, int] = (0, 0)

    def scale_normalized_coords(
        self, norm_coords: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        height, width = size_hw
        scale_yx = torch.tensor([height - 1, width - 1])
        pixel_coordinates_yx = (norm_coords + 1.0) * 0.5 * scale_yx
        return pixel_coordinates_yx.round().long()

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

    def kps_raw_2d_to_image_coords(
        self, kps_raw_2d: torch.Tensor, size_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_norm, v_norm = kps_raw_2d.chunk(2, dim=-1)
        norm_uv = torch.stack((v_norm, u_norm), dim=-1)
        pixel_yx = self.scale_normalized_coords(norm_uv, size_hw)
        y_pixel = pixel_yx[..., 0]  # (B, Nref)
        x_pixel = pixel_yx[..., 1]  # (B, Nref)
        return y_pixel, x_pixel

    def hard_pixels_to_3D_world(
        self,
        y_pixel: torch.Tensor,  # B, Nref
        x_pixel: torch.Tensor,  # B, Nref
        depth: torch.Tensor,  # N, H, W
        extr: torch.Tensor,  # N, 4, 4
        intr: torch.Tensor,  # N, 3, 3
    ):

        B, N_kp = x_pixel.shape
        batch_indices = torch.arange(
            B, device=depth.device, dtype=torch.long
        ).repeat_interleave(N_kp)
        z = depth[batch_indices, x_pixel.flatten(), y_pixel.flatten()]
        z = z.reshape(B, N_kp)

        pos = self.batched_pinhole_projection_image_to_world_coordinates_orig(
            y_pixel, x_pixel, z, intr, extr
        )

        return pos.permute(0, 2, 1).reshape((B, -1))

    def batched_pinhole_projection_image_to_camera_coordinates_orig(self, u, v, z, K):
        uv1 = torch.stack((u, v, torch.ones(u.shape, device=u.device)), dim=-1)
        K_inv = K.inverse()

        pos = torch.transpose(torch.matmul(K_inv, torch.transpose(uv1, -1, -2)), -1, -2)

        pos = z.unsqueeze(2).repeat(1, 1, 3) * pos
        return pos

    def batched_pinhole_projection_image_to_world_coordinates_orig(
        self, u, v, z, K, camera_to_world
    ):
        pos_in_camera_frame = (
            self.batched_pinhole_projection_image_to_camera_coordinates_orig(u, v, z, K)
        )
        pos_in_camera_frame_homog = torch.cat(
            (
                pos_in_camera_frame,
                torch.ones(
                    (*pos_in_camera_frame.shape[:-1], 1),
                    device=pos_in_camera_frame.device,
                ),
            ),
            dim=-1,
        )

        pos_in_world_homog = torch.transpose(
            torch.matmul(
                camera_to_world, torch.transpose(pos_in_camera_frame_homog, -1, -2)
            ),
            -1,
            -2,
        )

        return pos_in_world_homog[..., :3]

    def normalize_coords(
        self, coords: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        return torch.stack(
            [
                2.0 * coords[..., 0] / max(size_hw[0] - 1, 1) - 1.0,
                2.0 * coords[..., 1] / max(size_hw[1] - 1, 1) - 1.0,
            ],
            dim=-1,
        )  # (..., 2)

    def encode_direct(
        self, ref_image: Image.Image, image: Image.Image, y: int, x: int
    ) -> tuple[int, int]:
        logger.debug(f"Starting encoder")
        ref_img_desc = self.compute_descriptor(ref_image)  # (1, D, H, W)
        logger.debug(f"Computed ref reference")
        img_desc = self.compute_descriptor(image)  # (1, D, H, W)
        logger.debug(f"Computed image descriptor")
        ref_py, ref_px = self.transform_coords(
            y,
            x,
            image.height,
            image.width,
            img_desc.shape[2],
            img_desc.shape[3],
        )
        logger.debug(f"Transformed reference pixels: ({ref_py}, {ref_px})")
        ref_patch_desc = ref_img_desc[..., ref_py, ref_px]  # (1, D)
        kps_raw_2d, _, _, _ = self.compute_keypoints(img_desc, ref_patch_desc)
        assert kps_raw_2d.shape == (1, 2)
        y_pixel, x_pixel = self.kps_raw_2d_to_image_coords(
            kps_raw_2d, (image.height, image.width)
        )  # (1, 1), (1, 1)

        return int(y_pixel.squeeze(0).item()), int(x_pixel.squeeze(0).item())

    def encode(self, td_img: TDImage) -> tuple[torch.Tensor, torch.Tensor, dict]:
        image_desc = self.compute_descriptor(td_img.rgb)  # (1, D, H, W)

        kps_raw_2d, kps_mask, sm, post = self.compute_keypoints(
            image_desc
        )  # (1, 2*Nref), (1, Nref), (1, Nref, H, W), (1, Nref, H, W)

        y_pixel, x_pixel = self.kps_raw_2d_to_image_coords(
            kps_raw_2d, (td_img.rgb.shape[1], td_img.rgb.shape[2])
        )  # (1, Nref), (1, Nref)

        kps = self.hard_pixels_to_3D_world(
            y_pixel,
            x_pixel,
            td_img.d,
            td_img.extr,
            td_img.intr,
        )

        states, scores = self.compute_kps_states(
            kps_raw_2d, image_desc
        )  # (1, num_states), (1, num_states)

        info = {
            "descriptor": image_desc,
            "distance": None,
            "kp_raw_2d": kps_raw_2d,
            "depth": td_img.d,
            "prior": None,
            "kp_mask": kps_mask,
            "sm": sm,
            "post": post,
            "state_scores": scores,
        }

        return kps, states, info

    def compute_keypoints(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Find max similarity for each keypoint (N, Nref)
        max_sim = torch.amax(sm, dim=(-1, -2))  # shape: (N, Nref)
        kp_mask = (
            max_sim > self.cfg.kp_selection_threshold
        ).float()  # 1 if present, 0 if not

        kp_raw_2d = self.get_mode(post)  # (N, 2*Nref)

        return kp_raw_2d, kp_mask, sm, post

    def softmax_of_reference_descriptors(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ref_patch_desc is None:
            assert self.kp_patch_descr is not None
            patch_desc = self.kp_patch_descr
        else:
            patch_desc = ref_patch_desc

        Nref, Dref = patch_desc.shape
        N, D, H, W = image_desc.shape
        neg_squared_norm_diffs = self.compute_ref_descr_distances(
            image_desc, patch_desc
        )
        n_s_n_d_flat = neg_squared_norm_diffs.view(N, Nref, H * W)  # 1, nm, H*W
        # print(neg_squared_norm_diffs_flat.shape, "should be N, Nref, H*W")
        # neg_squared_norm_diffs_flat /= math.sqrt(D)

        softmax = torch.nn.Softmax(dim=2)
        sm_activ = softmax(n_s_n_d_flat * self.cfg.taper_sm).view(N, Nref, H, W)
        # print(softmax_activations.shape, "should be N, Nref, H, W")
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

    def compute_kps_states(
        self, kps_raw_2d: torch.Tensor, image_desc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # kps_raw_2d is (1, 2*Nref)
        # image_desc is (1, C, H, W)
        assert self.entity_state_coords is not None
        kernels = self.get_state_kernel(
            image_desc,
            kps_raw_2d,
            self.entity_state_coords,
        )  # (1, C, k, k)
        one_hots = []
        scores = []
        for idx, entity in enumerate(self.entities):
            prediction, score = self.entity_state_knn.query(
                entity_label=entity.cfg.label,
                state_desc_kernel=kernels[idx],
            )
            one_hot_state = entity.state.make_one_hot(prediction)
            one_hots.append(one_hot_state)
            scores.append(score)
        return torch.stack(one_hots, dim=0), torch.stack(scores, dim=0)

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

    def compute_patch_grid_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        # Compute the number of patches along height and width
        h, w = image_size
        grid_h = 1 + (h - self.patch_size) // self.cfg.stride
        grid_w = 1 + (w - self.patch_size) // self.cfg.stride
        return grid_h, grid_w

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
            ImageExtractor._fix_pos_enc(patch_size, (stride, stride)), model
        )
        return model, patch_size

    def transform_coords(
        self, y: int, x: int, origin_h: int, origin_w: int, target_h: int, target_w: int
    ) -> tuple[int, int]:
        # logger.debug(f"get_patch_index_from_img_coords: size: {img_desc.shape}")
        ref_norm_yx = self.normalize_coords(torch.tensor([y, x]), (origin_h, origin_w))
        target_yx = self.scale_normalized_coords(ref_norm_yx, (target_h, target_w))
        return int(target_yx[0].item()), int(target_yx[1].item())

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
        entity: Entity,
        kpt: tuple[Image.Image, int, int, int, int],
        kps: dict[str, list[Image.Image]],
    ):
        dc_img, dc_x, dc_y, state_x, state_y = kpt
        self.entities.append(entity)
        dc_img_desc = self.compute_descriptor(kpt[0])
        dc_py, dc_px = self.transform_coords(
            dc_x,
            dc_y,
            dc_img.height,
            dc_img.width,
            dc_img_desc.shape[2],
            dc_img_desc.shape[3],
        )
        state_py, state_px = self.transform_coords(
            state_y,
            state_x,
            dc_img.height,
            dc_img.width,
            dc_img_desc.shape[2],
            dc_img_desc.shape[3],
        )
        dc_ref_patch_desc = dc_img_desc[..., dc_py, dc_px]  # (1, D)
        dc_norm_coords = self.normalize_coords(
            torch.tensor([dc_py, dc_px]), self.image_size
        )
        state_norm_coords = self.normalize_coords(
            torch.tensor([state_py, state_px]), self.image_size
        )
        rel_state_pos = [
            state_norm_coords[0] - dc_norm_coords[0],
            state_norm_coords[1] - dc_norm_coords[1],
        ]
        state_coords = torch.tensor(rel_state_pos).unsqueeze(0)  # (1, 2)
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
                self.entity_state_knn.register(
                    entity.cfg.label,
                    state,
                    kernel,
                )
