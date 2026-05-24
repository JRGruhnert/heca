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
        use_state_coordinates: bool = False
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
        self.entity_descr: torch.Tensor | None = None
        self.image_size: tuple[int, int] = (0, 0)

    def norm_to_target_coords(
        self, norm_coords: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        height, width = size_hw
        scale_yx = torch.tensor(
            [height - 1, width - 1],
            device=norm_coords.device,
            dtype=norm_coords.dtype,
        )
        pixel_coordinates_yx = (norm_coords + 1.0) * 0.5 * scale_yx
        return pixel_coordinates_yx.round().long()

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
        patch_tokens = feats[:, 1:]

        B, N, C = patch_tokens.shape

        descr = patch_tokens.reshape(B, self.patch_size, self.patch_size, C)

        descr = descr.permute(0, 3, 1, 2)  # (B, C, H, W)

        if self.cfg.interpolate_descriptors:
            descr = torch.nn.functional.interpolate(
                input=descr,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

        return descr

    def hard_pixels_to_3D_world(
        self,
        kps_raw_2d,  # N, 2*k, where x features are stacked on top of y features
        depth,  # N, H, W
        camera_to_world,  # N, 4, 4
        K,  # N, 3, 3
        img_width,
        img_height,
    ):
        u_norm, v_norm = kps_raw_2d.chunk(2, dim=-1)
        norm_uv = torch.stack((v_norm, u_norm), dim=-1)
        pixel_yx = self.norm_to_target_coords(norm_uv, (img_height, img_width))
        x_pixel = pixel_yx[..., 0].detach()
        y_pixel = pixel_yx[..., 1].detach()

        B, N_kp = x_pixel.shape
        batch_indices = torch.arange(
            B, device=depth.device, dtype=torch.long
        ).repeat_interleave(N_kp)
        z = depth[batch_indices, x_pixel.flatten(), y_pixel.flatten()]
        z = z.reshape(B, N_kp)

        pos = self.batched_pinhole_projection_image_to_world_coordinates_orig(
            y_pixel, x_pixel, z, K, camera_to_world
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
        height, width = size_hw
        return torch.stack(
            [
                2.0 * coords[..., 0] / max(height - 1, 1) - 1.0,
                2.0 * coords[..., 1] / max(width - 1, 1) - 1.0,
            ],
            dim=-1,
        )

    def img_coords_to_patch_coords(self, img_coors: torch.Tensor) -> torch.Tensor:
        # img_coords is (N, 2) where (y, x) in pixel coordinates
        # we want to map to patch coordinates, which are normalized to [-1, 1]
        # and take into account the patch size and stride
        patch_coords = img_coors.float() / self.cfg.stride
        patch_coords = self.normalize_coords(
            patch_coords,
            (
                self.image_size[0] // self.cfg.stride,
                self.image_size[1] // self.cfg.stride,
            ),
        )
        return patch_coords

    def get_patch_index_from_img_coords(
        self, y: int, x: int, img_desc: torch.Tensor
    ) -> tuple[int, int]:
        _, _, H, W = img_desc.shape
        logger.debug(f"get_patch_index_from_img_coords: size: {img_desc.shape}")
        ref_norm_yx = self.normalize_coords(
            torch.tensor([y, x], dtype=torch.float32), self.image_size
        )
        ref_py, ref_px = self.norm_to_target_coords(ref_norm_yx, (H, W)).tolist()
        return ref_py, ref_px

    def encode_direct(
        self, ref_image: Image.Image, image: Image.Image, x: int, y: int
    ) -> tuple[float, float]:
        self.image_size = (image.height, image.width)
        ref_img_desc = self.compute_descriptor(ref_image)  # (1, D, H, W)
        img_desc = self.compute_descriptor(image)  # (1, D, H, W)
        ref_py, ref_px = self.get_patch_index_from_img_coords(y, x, ref_img_desc)
        ref_patch_desc = ref_img_desc[..., ref_py, ref_px]  # (1, D)
        kps_raw_2d, _, _, _ = self.compute_keypoints(img_desc, ref_patch_desc)
        assert kps_raw_2d.shape == (1, 2)
        yx = self.norm_to_target_coords(kps_raw_2d, (image.height, image.width))
        y, x = yx[0].tolist()
        return y, x

    def encode(
        self, td_image: TDImage
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.image_size = (td_image.rgb.shape[1], td_image.rgb.shape[2])
        image_desc = self.compute_descriptor(td_image.rgb)

        kps_raw_2d, kps_mask, sm, post = self.compute_keypoints(image_desc)

        states, scores = self.compute_kps_states(kps_raw_2d, image_desc)

        kps = self.hard_pixels_to_3D_world(
            kps_raw_2d,
            td_image.d,
            td_image.extr,
            td_image.intr,
            self.image_size[1],
            self.image_size[0],
        )

        info = {
            "descriptor": image_desc,
            "distance": None,
            "kp_raw_2d": kps_raw_2d,
            "depth": td_image.d,
            "prior": None,
            "kp_mask": kps_mask,
            "sm": sm,
            "post": post,
            "state_scores": scores,
        }

        return kps, kps_mask, states, scores, info

    def compute_keypoints(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        kp_raw_2d = self.get_mode(post)  # (N, 2*Nref) where for each kp: (x, y)

        return kp_raw_2d, kp_mask, sm, post

    def softmax_of_reference_descriptors(
        self, image_desc: torch.Tensor, ref_patch_desc: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ref_patch_desc is None:
            assert self.entity_descr is not None
            patch_desc = self.entity_descr
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
        B, N, H, W = (
            softmax_activations.shape
        )  # N is number of keypoints, H and W are spatial dimensions
        sm_flat = softmax_activations.view(B, N, -1)
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
        stacked_2d_features = torch.cat((modes_2d[0], modes_2d[1]), 1)
        stacked_2d_features = modes_2d.permute((1, 0, 2))
        stacked_2d_features = stacked_2d_features.reshape(B, -1)

        return stacked_2d_features

    def get_state_kernel(
        self, descriptor: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        # coords is (B, 2,) where (y, x) in tensor indexing coordinates
        # descriptor is (B, C, H, W)
        r = self.cfg.state_patch_radius
        y, x = coords[:, 0], coords[:, 1]
        return descriptor[:, :, y - r : y + r + 1, x - r : x + r + 1]

    def compute_kps_states(
        self, kp_raw_2d: torch.Tensor, image_desc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = image_desc.shape
        coords = self.norm_to_target_coords(kp_raw_2d, (H, W))
        kernels = self.get_state_kernel(image_desc, coords)
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

    @staticmethod
    def patch_vit_resolution(
        model: nn.Module,
        stride: int,
    ) -> tuple[nn.Module, int]:
        patch_size = model.patch_embed.patch_size
        print(f"Original patch size: {patch_size}, stride: {stride}")
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

    def add_entity_sample_for_cam(
        self,
        entity: Entity,
        dc_list: list[tuple[Image.Image, int, int]],
        state_image_dict: dict[str, list[tuple[Image.Image, int, int]]],
    ):
        self.entities.append(entity)
        dc_desc_mean: torch.Tensor | None = None
        for dc_img, dc_x, dc_y in dc_list:
            dc_img_desc = self.compute_descriptor(dc_img)
            _, _, dc_h, dc_w = dc_img_desc.shape
            dc_norm_yx = self.normalize_coords(
                torch.tensor([dc_y, dc_x], dtype=torch.float32),
                (dc_img.height, dc_img.width),
            )
            dc_py, dc_px = self.norm_to_target_coords(dc_norm_yx, (dc_h, dc_w)).tolist()
            dc_desc = dc_img_desc[0, :, dc_py, dc_px].unsqueeze(0)
            if dc_desc_mean is None:
                dc_desc_mean = dc_desc
            else:
                dc_desc_mean = torch.cat((dc_desc_mean, dc_desc), dim=0)

        assert dc_desc_mean is not None
        dc_desc_mean = dc_desc_mean.mean(dim=0, keepdim=True)

        if self.entity_descr is None:
            self.entity_descr = dc_desc_mean
        else:
            self.entity_descr = torch.cat((self.entity_descr, dc_desc_mean), dim=0)

        for state_label, state_list in state_image_dict.items():
            for state_img, state_x, state_y in state_list:
                if self.cfg.use_state_coordinates:
                    coords = torch.tensor(
                        [state_y, state_x], device=dc_desc.device
                    ).unsqueeze(0)
                else:
                    state_img_desc = self.compute_descriptor(state_img)
                    kp_raw_2d, _, _, _ = self.compute_keypoints(state_img_desc, dc_desc)
                    _, _, state_h, state_w = state_img_desc.shape
                    coords = self.norm_to_target_coords(
                        kp_raw_2d.squeeze(0).flip(0), (state_h, state_w)
                    ).unsqueeze(0)
                kernel = self.get_state_kernel(state_img_desc, coords)
                self.entity_state_knn.register(
                    entity.cfg.label,
                    state_label,
                    kernel,
                )
