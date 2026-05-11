from enum import Enum
import math
import types
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
from heca.classes.persist import Persistable
from heca.entities.entity import Entity
from heca.misc import hardware, logger
from heca.misc.logger import measure_runtime
from heca.misc.td import (
    TDCamReferences,
    TDEntity,
    TDEntities,
    TDSceneReferences,
    TDStateReferences,
)

from tapas_gmm_modified.viz.operations import channel_back2front_batch
from tapas_gmm_modified.utils.geometry_torch import hard_pixels_to_3D_world

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


class StateExtractionMode(Enum):
    NCC = "ncc"
    MALMO = "malmo"
    DINO = "dino"


class ImageExtractor(Persistable):
    @dataclass(frozen=True, kw_only=True)
    class Query(Persistable.Query):
        pass

    @dataclass(frozen=True, kw_only=True)
    class File(Persistable.File):
        folder: str = "image_extractors"
        ending: str = ".pt"

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        model_type: str = "dino_vits8"
        stride: int = 8
        layer: int = 11
        facet: str = "token"
        thresh: float = 0.5
        # dafuq
        register_tokens: int = 0  # extra cls-tokens for DinoV2+registers
        center_crop: bool = False
        pad: bool = False
        frozen: bool = True
        image_dim: tuple[int, int] = (480, 640)

        # keypoints
        descriptor_dim: int = 384
        taper_sm: int = 25
        cosine_distance: bool = True

        # my additions
        relative_kp: bool = False
        kp_selection_threshold: float = 0.2
        ee_kp_index: int = 0  # index of the end effector keypoint in the keypoint list.
        state_method: StateExtractionMode = StateExtractionMode.NCC
        state_kernel_patches: int = 3

    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = ImageExtractor.create_model(self.cfg.model_type)
        self.model, self.stride, self.patch_size = ImageExtractor.patch_vit_resolution(
            self.model, self.cfg.stride
        )
        self.model.eval()
        self.model.to(hardware.device)

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.cfg.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.cfg.model_type else (0.5, 0.5, 0.5)
        )

        self._get_descriptor_resolution(self.cfg.image_dim)
        self._feats = []
        self.hook_handlers = []
        self.num_patches = None

        # dont understand yet
        self.image_height, self.image_width = self.cfg.image_dim
        self.dc_height, self.dc_width = self.get_dc_dim()
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.dc_width),
            np.linspace(-1.0, 1.0, self.dc_height),
        )

        self.pos_x = torch.from_numpy(pos_x).float().to(hardware.device)
        self.pos_y = torch.from_numpy(pos_y).float().to(hardware.device)

        # my stuff
        self.descriptor_data: TDSceneReferences = TDSceneReferences()
        self.entities: list[Entity.Config] = []
        self.state_patch_size = self.cfg.state_kernel_patches * self.patch_size

    def get_dc_dim(self):
        dim_mapping = {128: 32, 256: 32, 360: 45, 480: 60, 640: 80}

        return dim_mapping[self.image_height], dim_mapping[self.image_width]

    def preprocess(
        self, img_tensor: torch.Tensor, load_size: tuple[int, int] | None = None
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param img_tensor: Torch image tensor.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        if load_size is not None:
            if self.cfg.center_crop:
                img_tensor = transforms.CenterCrop(224)(img_tensor)
            elif self.cfg.pad:
                img_tensor = transforms.Pad(5)(img_tensor)
            else:
                img_tensor = transforms.Resize(
                    load_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )(  # LANCZOS)(
                    img_tensor
                )
        prep = transforms.Normalize(mean=self.mean, std=self.std)
        prep_img = prep(img_tensor)

        if len(prep_img.shape) == 3:
            prep_img = prep_img.unsqueeze(dim=0)

        return prep_img

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        self._feats = []
        self._register_hooks([self.cfg.layer], self.cfg.facet)
        _ = self.model(batch)
        self._unregister_hooks()
        return self._feats[0].unsqueeze_(dim=1)

    def extract_descriptors(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(batch)
        x = x[
            :, :, 1 + self.cfg.register_tokens :, :
        ]  # remove cls token and register tokens

        return x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)

    def extract_cls_token(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(batch)
        return x[:, 0, 0, :]

    def _get_descriptor_resolution(
        self, image_dim: tuple[int, int], load_size: tuple[int, int] | None = None
    ):
        if "dinov2" in self.cfg.model_type:
            assert load_size is not None, "load_size must be provided for dinov2 models"
            H_descr = W_descr = load_size[0] // 14
            if self.cfg.stride == 7:
                H_descr = H_descr * 2 - 1
                W_descr = W_descr * 2 - 1
            else:
                assert self.cfg.stride == 14

            if self.cfg.center_crop:
                H_descr -= 4
                W_descr -= 4
            elif self.cfg.pad:
                H_descr += 2
                W_descr += 2
        else:
            H_descr = image_dim[0] // 8
            W_descr = image_dim[1] // 8
            # H_descr = W_descr = 32
            if self.cfg.stride == 4:
                H_descr = H_descr * 2 - 1
                W_descr = W_descr * 2 - 1
            else:
                assert self.cfg.stride == 8

        self.H_descr, self.W_descr = H_descr, W_descr

    @measure_runtime
    def compute_descriptor(
        self, camera_obs: torch.Tensor, upscale: bool = True
    ) -> torch.Tensor:
        camera_obs = camera_obs.to(hardware.device)

        if len(camera_obs.shape) == 3:
            camera_obs = camera_obs.unsqueeze(dim=0)
        B, _, H, W = camera_obs.shape

        with torch.inference_mode():
            prep, _ = self.preprocess(camera_obs)
            descr = self.extract_descriptors(prep).squeeze(0)

        descr = descr.reshape(B, self.H_descr, self.W_descr, descr.shape[-1])
        descr = channel_back2front_batch(descr)

        if upscale:
            descr = torch.nn.functional.interpolate(
                input=descr, size=[H, W], mode="bilinear", align_corners=True
            )

        return descr

    def __call__(self, cams: dict) -> TDEntities:
        all_kps = []
        all_states = []
        all_infos = []
        all_masks = []
        all_scores = []

        for label, cam in cams.items():
            data = self.descriptor_data.get(label)
            assert isinstance(data, TDCamReferences)
            kps, states, info = self.encode(
                data,
                cam.rgb,
                cam.depth,
                cam.extrinsics,
                cam.intrinsics,
            )
            all_kps.append(kps)
            all_states.append(states)
            all_infos.append(info)
            all_masks.append(info["kp_mask"])
            all_scores.append(info["state_scores"])

        kps_stack = torch.stack(all_kps)  # (C, K, D)
        kps_mask = torch.stack(all_masks)  # (C, K)
        state_scores = torch.stack(all_scores)  # (C, K)
        states_stack = torch.stack(all_states)  # (C, K, S)

        # Aggregate keypoints of different cameras
        num_kps = kps_stack.shape[1]
        final_kps = []
        final_states = []
        for k in range(num_kps):
            present = kps_mask[:, k] > 0
            if present.sum() == 0:
                # Not present in any camera
                final_kps.append(torch.full_like(kps_stack[0, k], float("nan")))
                final_states.append(torch.full_like(states_stack[0, k], float("nan")))
            elif present.sum() == 1:
                # Present in only one camera
                idx = present.nonzero(as_tuple=True)[0][0]
                final_kps.append(kps_stack[idx, k])
                final_states.append(states_stack[idx, k])
            else:
                # Present in multiple cameras
                # Mean for keypoints
                # Best score for states
                vals = kps_stack[present, k]
                final_kps.append(vals.mean(dim=0))
                valid_scores = state_scores[:, k].clone()
                valid_scores[~present] = float("-inf")
                idx = torch.argmax(valid_scores)
                final_states.append(states_stack[idx, k])
        final_kps = torch.stack(final_kps)  # (K, D)
        final_states = torch.stack(final_states)  # (K, S)

        entities: dict[str, TDEntity] = {}
        # Find the index of the end effector (ee) entity in self.entities
        ee_kp = final_kps[self.cfg.ee_kp_index]
        num_present = min(final_kps.shape[0], final_states.shape[0], len(self.entities))
        for idx, query in enumerate(self.entities):
            if idx >= num_present:
                # No keypoint or descriptor for this entity, skip
                # Assuming keypoints are ordered the same as entities
                continue

            tg_kp = final_kps[idx]
            if self.cfg.relative_kp:
                rel_pos = tg_kp[:3] - ee_kp[:3]
                rel_rot = self._relative_quaternion(
                    tg_kp[3:7],
                    ee_kp[3:7],
                )
                td = TDEntity(
                    position=rel_pos,
                    rotation=rel_rot,
                    state=final_states[idx],
                )
            else:
                td = TDEntity(
                    position=tg_kp[:3],
                    rotation=tg_kp[3:7],
                    state=final_states[idx],
                )
            entities[query.label] = td

        return TDEntities(entities)

    def _relative_quaternion(
        self, q: torch.Tensor, q_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relative quaternion q_rel such that: q = q_rel * q_ref
        Returns q_rel = q * q_ref_conj
        """

        # q, q_ref: (..., 4) in (w, x, y, z) or (x, y, z, w) format
        # Assume (x, y, z, w) format as is common in PyTorch/robotics
        # Convert to (w, x, y, z) for computation if needed
        # Here, we assume (x, y, z, w)
        def quat_conj(q):
            return torch.tensor(
                [-q[0], -q[1], -q[2], q[3]], device=q.device, dtype=q.dtype
            )

        def quat_mult(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack([x, y, z, w])

        q_ref_conj = quat_conj(q_ref)
        q_rel = quat_mult(q, q_ref_conj)
        return q_rel

    def encode(
        self,
        cam_refs: TDCamReferences,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        extr: torch.Tensor,
        intr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:

        descriptor = self.compute_descriptor(
            rgb,
            upscale=False,
        )

        kps_raw_2d, kps_mask, sm, post = self.compute_keypoints(
            cam_refs.pose_refs.kp_refs,
            descriptor,
        )

        kps = hard_pixels_to_3D_world(
            kps_raw_2d,
            depth,
            extr,
            intr,
            self.image_width - 1,
            self.image_height - 1,
        )

        states, scores = self.compute_states(
            cam_refs,
            rgb,
            kps_raw_2d,
        )

        info = {
            "descriptor": descriptor,
            "distance": None,
            "kp_raw_2d": kps_raw_2d,
            "depth": depth,
            "prior": None,
            "kp_mask": kps_mask,
            "sm": sm,
            "post": post,
            "state_scores": scores,
        }

        return kps, states, info

    def ncc(self, patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
        """
        Compute Normalized Cross-Correlation (NCC) between two patches.
        Both patches must be torch tensors of the same shape.
        """
        patch1 = patch1 - patch1.mean()
        patch2 = patch2 - patch2.mean()
        numerator = torch.sum(patch1 * patch2)
        denominator = torch.sqrt(torch.sum(patch1**2)) * torch.sqrt(
            torch.sum(patch2**2)
        )
        return numerator / denominator if denominator != 0 else torch.tensor(0.0)

    def compute_states(
        self,
        cam_refs: TDCamReferences,
        rgb: torch.Tensor,
        kp_raw_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hots = []
        scores = []
        for idx, entity_config in enumerate(self.entities):
            entity_refs = cam_refs.entity_states.entities.get(entity_config.label)
            assert isinstance(entity_refs, TDStateReferences)
            cls_tokens = self.encode_states(rgb, kp_raw_2d[idx])
            distances = torch.nn.functional.cosine_similarity(
                cls_tokens, entity_refs.cls_tokens, dim=1
            )
            entity = Entity.create(entity_config)
            max_idx = int(torch.argmax(distances).item())
            max_score = distances[max_idx].item()
            one_hot_state = entity.state.one_hot_from_idx(max_idx)
            one_hots.append(one_hot_state)
            scores.append(max_score)
        return torch.stack(one_hots, dim=0), torch.stack(scores, dim=0)

    def prepare_references(self, examples: TDSceneReferences):
        td = examples.copy()
        assert isinstance(td, TDSceneReferences)
        for cam in td.cams.values():
            for idx, entity_config in enumerate(self.entities):
                state_refs = cam.entity_states.entities.get(entity_config.label)
                assert isinstance(state_refs, TDStateReferences)

                s_img_desc = self.compute_descriptor(
                    state_refs.images_raw,
                    upscale=False,
                )
                kp_img_desc = self.compute_descriptor(
                    cam.pose_refs.images_raw,
                    upscale=False,
                )
                cam.pose_refs.set_preprocessed(kp_img_desc)

                kps_raw_2d, _, _, _ = self.compute_keypoints(
                    cam.pose_refs.kp_refs[idx],
                    s_img_desc,
                )  # (N, 2*Nref) where for each kp: (x, y)

                cls_tokens = self.encode_states(
                    state_refs.images_raw,
                    kps_raw_2d,
                )

                state_refs.set_preprocessed(cls_tokens)

    def encode_states(
        self,
        rgb: torch.Tensor,
        kps_raw_2d: torch.Tensor,
    ) -> torch.Tensor:
        x = kps_raw_2d[:, 0]
        y = kps_raw_2d[:, 1]
        pad = self.state_patch_size // 2
        # TODO: which mode?
        img_padded = torch.nn.functional.pad(rgb, (pad, pad, pad, pad), mode="reflect")
        # Extract (C, k, k) patch for each batch element at (x[b], y[b])
        patches = []
        for b in range(img_padded.shape[0]):
            x_b = int(x[b].item())
            y_b = int(y[b].item())
            patch = img_padded[b, :, y_b - pad : y_b + pad, x_b - pad : x_b + pad]
            patches.append(patch)
        patches = torch.stack(patches, dim=0)  # (B, C, k, k)
        assert patches.shape[2] == self.state_patch_size
        assert patches.shape[3] == self.state_patch_size

        prep, _ = self.preprocess(patches)
        # Extract CLS token
        with torch.inference_mode():
            return self.extract_cls_token(prep)

    def compute_keypoints(
        self,
        point_refs: torch.Tensor,
        image_desc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sm = self.softmax_of_reference_descriptors(image_desc, point_refs)
        # sm = similarity map
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
        self,
        descriptor_images: torch.Tensor,
        ref_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        N, D, H, W = descriptor_images.shape
        Nref, Dref = ref_descriptor.shape

        neg_squared_norm_diffs = self.compute_reference_descriptor_distances(
            descriptor_images, ref_descriptor
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

        if self.cfg.cosine_distance:
            distance = torch.nn.functional.cosine_similarity(
                descriptor_images, ref_descriptor[None, None, None, :], dim=4
            )

            return distance.permute(0, 3, 1, 2)

        else:
            deltas = descriptor_images - ref_descriptor
            # print(deltas.shape, "should also be N, H, W, Nref, D?")

            neg_squared_norm_diffs = -1.0 * torch.sum(
                torch.pow(deltas, 2), dim=4
            )  # N, H, W, Nref
            # print(neg_squared_norm_diffs.shape, "should be N, H, W, Nref")

            # spatial softmax
            neg_squared_norm_diffs = neg_squared_norm_diffs.permute(
                0, 3, 1, 2
            )  # N, Nref, H, W
            # print(neg_squared_norm_diffs.shape, "should be N, Nref, H, W")

            return neg_squared_norm_diffs

    def get_mode(self, softmax_activations):
        # need argmax over two last dimensions, so join them first
        s = softmax_activations.shape
        sm_flat = softmax_activations.view(s[0], s[1], -1)
        modes_flat = torch.argmax(sm_flat, dim=2)

        # reshape back to 2D. Note that the new dim is in the front for now.
        modes_2d = modes_flat.unsqueeze(0).repeat((2, 1, 1))

        # get H, W from the flat indeces
        modes_2d[1] = modes_2d[1] // self.dc_width
        modes_2d[0] = modes_2d[0] % self.dc_width

        # map from [0, img_size] to [-1, 1] to match pixel_map from spatial exp
        modes_2d = modes_2d.float()
        modes_2d[1] = modes_2d[1] / (self.dc_height - 1) * 2 - 1
        modes_2d[0] = modes_2d[0] / (self.dc_width - 1) * 2 - 1

        # move new dim into the middle and flatten to get (N, 2*Nref)
        stacked_2d_features = torch.cat((modes_2d[0], modes_2d[1]), 1)
        stacked_2d_features = modes_2d.permute((1, 0, 2))
        stacked_2d_features = stacked_2d_features.reshape(s[0], -1)

        return stacked_2d_features

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if "dinov2" in model_type:
            model = torch.hub.load("facebookresearch/dinov2", model_type)
        elif "dino" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load(
                "facebookresearch/dino:main", model_type_dict[model_type]
            )
            assert isinstance(model, nn.Module) and isinstance(
                temp_model, nn.Module
            ), "models should be of type nn.Module"
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict["head.weight"]
            del temp_state_dict["head.bias"]
            model.load_state_dict(temp_state_dict)
        assert isinstance(model, nn.Module) and isinstance(
            temp_model, nn.Module
        ), "models should be of type nn.Module"
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

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
        model: nn.Module, stride: int
    ) -> tuple[nn.Module, int, int]:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model, the new stride, and the patch size
        """
        patch_size = model.patch_embed.patch_size

        # Dino-V2 returns a tuple of ints (H and W)
        if type(patch_size) == tuple:
            # assert that all values in the tuple patch_size are equal
            assert all(
                [p == patch_size[0] for p in patch_size]
            ), "assuming square patches. else implement ..."
            patch_size = patch_size[0]

        if stride == patch_size:  # nothing to do
            return model, stride, patch_size

        stride_t: tuple = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride_t])

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(  # type: ignore
            ImageExtractor._fix_pos_enc(patch_size, stride_t), model
        )
        return model, stride, patch_size

    @classmethod
    def load(cls, query: "ImageExtractor.Query", scene: str) -> "ImageExtractor":
        directory = cls.File.resolve_directory(query) / scene
        extractor = cls.search(query)
        try:
            td = torch.load(directory / "samples.pt", map_location=hardware.device)
            assert isinstance(td, TDSceneReferences)
            extractor.prepare_references(td)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Could not load TDEntities: {e}")

        logger.info(f"Loaded ImageExtractor for scene {scene} with query: {query}")
        return extractor

    @classmethod
    def save(cls, query: "ImageExtractor.Query", scene: str) -> bool:
        directory = cls.File.resolve_directory(query) / scene
        extractor = cls.search(query)
        if extractor.descriptor_data is None:
            logger.warning(
                f"ImageExtractor for scene {scene} with query: {query} has no references to save."
            )
            return False
        torch.save(extractor.descriptor_data, directory / "samples.pt")
        return True
