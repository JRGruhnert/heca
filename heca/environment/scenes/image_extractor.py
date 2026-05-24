from dataclasses import dataclass
import math
import timm
import torch
import types
from PIL import Image
from torch import nn
from timm.data import resolve_model_data_config, create_transform  # type: ignore
from heca.classes.config import Configurable
from heca.entities.entity import Entity
from heca.environment.scenes.knn import (
    CompareMode,
    EntityStateKNN,
    ScoreMode,
    SelectionMode,
)
from heca.misc.td import TDImage

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


class ImageExtractor(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        stride: int = 8
        thresh: float = 0.5
        center_crop: bool = False
        pad: bool = False
        frozen: bool = True
        taper_sm: int = 25

        kp_selection_threshold: float = 0.2
        state_knn_config: EntityStateKNN.Config = EntityStateKNN.Config(
            top_k=5,
            score_mode=ScoreMode.AVERAGE,
            compare_mode=CompareMode.COSINE,
            selection_mode=SelectionMode.WEIGHTED_VOTE,
        )
        state_kernel_radius: int = 16
        use_state_coordinates: bool = False
        matching_interpolated_descriptors: bool = True

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

    # @measure_runtime
    def compute_descriptor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            prep = self.transforms(image)  # type: ignore
            assert isinstance(prep, torch.Tensor)
            feats: torch.Tensor = self.model.forward_features(prep.unsqueeze(0))
            # output is unpooled, a (1, 261, 4096) shaped tensor
            # [B, 1 + N, C]
        # cls_token = feats[:, 0] # Not using atm
        patch_tokens = feats[:, 1:]

        B, N, C = patch_tokens.shape

        descr = patch_tokens.reshape(B, self.patch_size, self.patch_size, C)

        descr = descr.permute(0, 3, 1, 2)  # (B, C, H, W)

        if self.cfg.matching_interpolated_descriptors:
            descr = torch.nn.functional.interpolate(
                input=descr,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

        return descr

    def hard_pixels_to_3D_world(
        self,
        y_vision,  # N, 2*k, where x features are stacked on top of y features
        depth,  # N, H, W
        camera_to_world,  # N, 4, 4
        K,  # N, 3, 3
        img_width,
        img_height,
    ):
        u_normalized, v_normalized = y_vision.chunk(2, dim=-1)
        u_pixel = ((u_normalized / 2.0 + 0.5) * img_width).long().detach()
        v_pixel = ((v_normalized / 2.0 + 0.5) * img_height).long().detach()

        B, N_kp = v_pixel.shape
        batch_indeces = torch.arange(
            B, device=depth.device, dtype=torch.long
        ).repeat_interleave(N_kp)
        z = depth[batch_indeces, v_pixel.flatten(), u_pixel.flatten()]
        z = z.reshape(B, N_kp)

        pos = self.batched_pinhole_projection_image_to_world_coordinates_orig(
            u_pixel, v_pixel, z, K, camera_to_world
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

    def encode(
        self, td_image: TDImage
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.image_size = (td_image.rgb.shape[1], td_image.rgb.shape[2])
        image_desc = self.compute_descriptor(td_image.rgb)

        kps_raw_2d, kps_mask, sm, post = self.compute_keypoints(
            image_desc,
        )

        states, scores = self.compute_keypoint_states(
            kps_raw_2d,
            image_desc,
        )

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
        self, image_desc: torch.Tensor, entity_desc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sm = self.softmax_of_reference_descriptors(image_desc, entity_desc)
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
        self, image_desc: torch.Tensor, entity_desc: torch.Tensor | None = None
    ) -> torch.Tensor:
        N, D, H, W = image_desc.shape
        if entity_desc is None:
            assert self.entity_descr is not None
            ref_desc = self.entity_descr
        Nref, Dref = ref_desc.shape

        neg_squared_norm_diffs = self.compute_reference_descriptor_distances(
            image_desc, ref_desc
        )

        neg_squared_norm_diffs_flat = neg_squared_norm_diffs.view(
            N, Nref, H * W
        )  # 1, nm, H*W
        # print(neg_squared_norm_diffs_flat.shape, "should be N, Nref, H*W")
        # neg_squared_norm_diffs_flat /= math.sqrt(D)

        softmax = torch.nn.Softmax(dim=2)
        softmax_activations = softmax(
            neg_squared_norm_diffs_flat * self.cfg.taper_sm
        ).view(
            N, Nref, H, W
        )  # N, Nref, H, W
        # print(softmax_activations.shape, "should be N, Nref, H, W")

        return softmax_activations

    def compute_reference_descriptor_distances(
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
        modes_2d[1] = modes_2d[1] // H
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
        # coords is (B, 2,) where (x, y) in pixel coordinates
        # descriptor is (B, C, H, W)
        r = self.cfg.state_kernel_radius
        x, y = coords[:, 0], coords[:, 1]
        return descriptor[:, :, y - r : y + r + 1, x - r : x + r + 1]

    def compute_keypoint_states(
        self,
        kp_raw_2d: torch.Tensor,
        image_desc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self.get_img_coordinates(kp_raw_2d)
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

    def get_img_coordinates(self, norm_coords: torch.Tensor) -> torch.Tensor:
        # normalized_coordinates is (N, 2) where for each kp: (x, y) in [-1, 1]
        # we want to convert to pixel coordinates in [0, img_size]
        pixel_coordinates = (
            (norm_coords + 1)
            / 2
            * (torch.tensor(self.image_size, device=norm_coords.device) - 1)
        )
        return pixel_coordinates.int()

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
        for dc_img, dc_y, dc_x in dc_list:
            dc_img_desc = self.compute_descriptor(dc_img)
            dc_desc = dc_img_desc[dc_y, dc_x].unsqueeze(0)
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
                        [state_x, state_y], device=dc_desc.device
                    ).unsqueeze(0)
                else:
                    state_img_desc = self.compute_descriptor(state_img)
                    state_img_desc = state_img_desc.unsqueeze(0)
                    kp_raw_2d, _, _, _ = self.compute_keypoints(state_img_desc, dc_desc)
                    coords = self.get_img_coordinates(kp_raw_2d)
                kernel = self.get_state_kernel(state_img_desc, coords)
                self.entity_state_knn.register(
                    entity.cfg.label,
                    state_label,
                    kernel,
                )
