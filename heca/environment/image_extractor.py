import math
import types
from dataclasses import dataclass
from typing import List, Tuple
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils

from heca.classes.persist import Persistable
from heca.misc import hardware, logger
from heca.misc.td import TDEntities, TDScene

from tapas_gmm_modified.viz.operations import channel_back2front_batch
from tapas_gmm_modified.utils.debug import measure_runtime

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


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
        load_size: int | None = None
        layer: int = 11
        facet: str = "token"
        bin: bool = False
        thresh: float = 0.5
        # dafuq
        include_cls: bool = False
        register_tokens: int = 0  # extra cls-tokens for DinoV2+registers
        center_crop: bool = False
        pad: bool = False
        frozen: bool = True
        image_dim: tuple[int, int] = (480, 640)
        # keypoints
        descriptor_dim = 384
        keypoints = keypoints_conf
        prior_type = PriorTypes.NONE
        projection = ProjectionTypes.GLOBAL_HARD
        taper_sm = 25
        cosine_distance = True
        use_spatial_expectation = False
        threshold_keypoint_dist = None
        overshadow_keypoints = False
        add_noise_scale = None

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
        self.model = ImageExtractor.patch_vit_resolution(self.model, self.cfg.stride)
        self.model.eval()
        self.model.to(hardware.device)

        self.stride = self.model.patch_embed.proj.stride
        patch_size = self.model.patch_embed.patch_size
        if type(patch_size) is tuple:
            assert all([p == patch_size[0] for p in patch_size]), "has to be square"
            self.p = patch_size[0]
        else:
            self.p = patch_size

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.cfg.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.cfg.model_type else (0.5, 0.5, 0.5)
        )

        self._get_descriptor_resolution(self.cfg.image_dim)
        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

        self.values: TDEntities | None = None

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
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
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
            return model

        stride_t: tuple = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride_t])

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(  # type: ignore
            ImageExtractor._fix_pos_enc(patch_size, stride_t), model
        )
        return model

    def preprocess(self, img_tensor) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param img_tensor: Torch image tensor.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        if self.cfg.load_size is not None:
            if self.cfg.center_crop:
                img_tensor = transforms.CenterCrop(224)(img_tensor)
            elif self.cfg.pad:
                img_tensor = transforms.Pad(5)(img_tensor)
            else:
                img_tensor = transforms.Resize(
                    self.cfg.load_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )(  # LANCZOS)(
                    img_tensor
                )
        prep = transforms.Normalize(mean=self.mean, std=self.std)
        prep_img = prep(img_tensor)

        if len(prep_img.shape) == 3:
            prep_img = prep_img.unsqueeze(dim=0)

        return prep_img

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

    def _extract_features(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks([self.cfg.layer], self.cfg.facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def extract_descriptors(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert self.cfg.facet in ["key", "query", "value", "token"]
        self._extract_features(batch)
        x = self._feats[0]
        if self.cfg.facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not self.cfg.include_cls:
            x = x[
                :, :, 1 + self.cfg.register_tokens :, :
            ]  # remove cls token and register tokens
        else:
            assert (
                not self.cfg.bin
            ), "bin = True and include_cls = True are not supported together, set one of them False."
        if not self.cfg.bin:
            desc = (
                x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
            )  # Bx1xtx(dxh)
        else:
            # desc = self._log_bin(x)
            logger.warning("log binning is not implemented yet")
        return desc

    def _get_descriptor_resolution(self, image_dim: tuple[int, int]):
        if "dinov2" in self.cfg.model_type:
            assert self.cfg.load_size is not None
            H_descr = W_descr = self.cfg.load_size // 14
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

        _, H, W = camera_obs.shape

        with torch.inference_mode():
            prep, _ = self.preprocess(camera_obs)
            descr = self.extract_descriptors(prep).squeeze(0)

        descr = descr.reshape(1, self.H_descr, self.W_descr, descr.shape[-1])
        descr = channel_back2front_batch(descr)

        if upscale:
            descr = torch.nn.functional.interpolate(
                input=descr, size=[H, W], mode="bilinear", align_corners=True
            )

        return descr

    def compute_descriptor_batch(
        self, camera_obs: torch.Tensor, upscale: bool = True
    ) -> torch.Tensor:
        camera_obs = camera_obs.to(hardware.device)

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

    def encode(self, rgb: tuple[torch.Tensor, ...]) -> torch.Tensor:
        enc = tuple(self.compute_descriptor_batch(img) for img in rgb)
        return torch.cat(enc, dim=-1)

    def predict_point(self, img: torch.Tensor) -> tuple[int, int]:
        # 4. Get dense descriptors
        descriptors = self.scene.extractor.compute_descriptor(
            img
        )  # shape: (B, D, H, W)

    def __call__(self, rgb: dict[str, torch.Tensor]) -> TDEntities:
        np_image_dict = self._get_image_tuple(obs)
        img_tensor, img_pil = self.preprocess(img_tensor)

        # TODO: what input ?
        enc = self.encode(obs)
        if self.cfg.thresh > 0:
            desc = desc * (desc.norm(dim=-1, keepdim=True) > self.cfg.thresh).float()
        return TDScene(formats={"descriptors": desc}), True

    def set_values(self, values: TDEntities) -> None:
        self.values = values

    @classmethod
    def load(cls, query: "ImageExtractor.Query", scene: str) -> "ImageExtractor":
        directory = cls.File.resolve_directory(query) / scene
        extractor = cls.search(query)
        try:
            td = torch.load(directory / "samples.pt", map_location=hardware.device)
            extractor.set_values(td)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Could not load TDEntities: {e}")

        logger.info(f"Loaded ImageExtractor for scene {scene} with query: {query}")
        return extractor

    @classmethod
    def save(cls, query: "ImageExtractor.Query", scene: str) -> bool:
        directory = cls.File.resolve_directory(query) / scene
        extractor = cls.search(query)
        if extractor.values is None:
            logger.warning(
                f"ImageExtractor for scene {scene} with query: {query} has no values to save."
            )
            return False
        torch.save(extractor.values, directory / "samples.pt")
        return True

    # def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
    #     """
    #     create a log-binned descriptor.
    #     :param x: tensor of features. Has shape Bxhxtxd.
    #     :param hierarchy: how many bin hierarchies to use.
    #     """
    #     B = x.shape[0]
    #     num_bins = 1 + 8 * hierarchy

    #     bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
    #     bin_x = bin_x.permute(0, 2, 1)
    #     bin_x = bin_x.reshape(
    #         B, bin_x.shape[1], self.num_patches[0], self.num_patches[1]
    #     )
    #     # Bx(dxh)xnum_patches[0]xnum_patches[1]
    #     sub_desc_dim = bin_x.shape[1]

    #     avg_pools = []
    #     # compute bins of all sizes for all spatial locations.
    #     for k in range(0, hierarchy):
    #         # avg pooling with kernel 3**kx3**k
    #         win_size = 3**k
    #         avg_pool = torch.nn.AvgPool2d(
    #             win_size, stride=1, padding=win_size // 2, count_include_pad=False
    #         )
    #         avg_pools.append(avg_pool(bin_x))

    #     bin_x = torch.zeros(
    #         (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
    #     ).to(self.device)
    #     for y in range(self.num_patches[0]):
    #         for x in range(self.num_patches[1]):
    #             part_idx = 0
    #             # fill all bins for a spatial location (y, x)
    #             for k in range(0, hierarchy):
    #                 kernel_size = 3**k
    #                 for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
    #                     for j in range(
    #                         x - kernel_size, x + kernel_size + 1, kernel_size
    #                     ):
    #                         if i == y and j == x and k != 0:
    #                             continue
    #                         if (
    #                             0 <= i < self.num_patches[0]
    #                             and 0 <= j < self.num_patches[1]
    #                         ):
    #                             bin_x[
    #                                 :,
    #                                 part_idx
    #                                 * sub_desc_dim : (part_idx + 1)
    #                                 * sub_desc_dim,
    #                                 y,
    #                                 x,
    #                             ] = avg_pools[k][:, :, i, j]
    #                         else:  # handle padding in a more delicate way than zero padding
    #                             temp_i = max(0, min(i, self.num_patches[0] - 1))
    #                             temp_j = max(0, min(j, self.num_patches[1] - 1))
    #                             bin_x[
    #                                 :,
    #                                 part_idx
    #                                 * sub_desc_dim : (part_idx + 1)
    #                                 * sub_desc_dim,
    #                                 y,
    #                                 x,
    #                             ] = avg_pools[k][:, :, temp_i, temp_j]
    #                         part_idx += 1
    #     bin_x = (
    #         bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
    #     )
    #     # Bx1x(t-1)x(dxh)
    #     return bin_x

    def encode(self, camera_obs) -> tuple[torch.Tensor, dict]:
        rgb = tuple(o.rgb for o in camera_obs)
        depth = tuple(o.depth for o in camera_obs)
        extr = tuple(o.extr for o in camera_obs)
        intr = tuple(o.intr for o in camera_obs)

        n_cams = len(rgb)

        descriptor = tuple(self.compute_descriptor_batch(r, upscale=False) for r in rgb)

        kp, info = self._encode(
            rgb,
            depth,
            extr,
            intr,
            tuple((None for _ in range(n_cams))),
            descriptor=descriptor,
            ref_descriptor=self._reference_descriptor_vec,
        )

        if self.add_noise_scale is not None:
            kp = self.add_gaussian_noise(kp, self.add_noise_scale, skip_z=False)

        return kp, info

    def _encode(
        self,
        rgb,
        depth,
        extr,
        intr,
        prior,
        descriptor,
        ref_descriptor,
    ):
        # All args are tuples, besides the kwargs.
        # the value to which overshadowed and super-threshold are set. -1
        # corresponds to 0 in pixel-space. Should be out of (0, img_size) to
        # avoid confusion? TODO! Eg set st it will end up being -1 in pixel
        ZERO_VAL = torch.tensor(-1, dtype=torch.float32, device=descriptor[0].device)

        n_cams = len(rgb)

        kwargs = {
            "ref_descriptor": ref_descriptor,
            "taper": self.cfg.taper_sm,
            "use_spatial_expectation": self.cfg.use_spatial_expectation,
            "projection": self.cfg.projection,
            "cosine": self.cfg.cosine_distance,
        }

        kp, distance, kp_raw_2d, prior, sm, post = tuple(
            zip(
                *(
                    self.compute_keypoints(r, d, e, i, desc, p, **kwargs)
                    for r, d, e, i, desc, p in zip(
                        rgb, depth, extr, intr, descriptor, prior
                    )
                )
            )
        )

        if self.cfg.overshadow_keypoints and n_cams > 1:
            dist_per_cam = torch.stack(distance)
            best_cam = torch.argmin(dist_per_cam, dim=0)
            # repeat to fit size of kp-tensor which has x_comps, y_comps
            best_cam = best_cam.repeat(1, self.keypoint_dimension)

            kp = tuple(
                torch.where(best_cam == i, k, ZERO_VAL) for i, k in enumerate(kp)
            )

        kp = torch.cat(kp, dim=-1)

        if self.cfg.thresh:
            # repeat to fit size of kp-tensor which has x_comps, y_comps
            expanded_dist = torch.cat(
                tuple(d.repeat(1, self.keypoint_dimension) for d in distance), dim=-1
            )
            kp = torch.where(expanded_dist > self.cfg.thresh, ZERO_VAL, kp)

        info = {
            "descriptor": descriptor,
            "distance": distance,
            "kp_raw_2d": kp_raw_2d,
            "depth": depth,
            "prior": prior,
            "sm": sm,
            "post": post,
        }

        return kp, info

    def compute_keypoints(
        self,
        camera_obs,
        depth,
        extrinsics,
        intrinsics,
        descriptor,
        prior=None,
        ref_descriptor=None,
        taper=1,
        use_spatial_expectation=False,
        projection=None,
        cosine=False,
    ):
        sm = self.softmax_of_reference_descriptors(
            descriptor, ref_descriptor, taper=taper, cosine=cosine
        )

        post = sm
        # When correspondence is (almost) zero across the image, the tensor
        # degenerates (becomes zeros, hence nan after renomalization below).
        # Fix by adding small epsilon.
        # post += 1e-10
        # # normalize to sum to one
        post /= torch.sum(post, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        if use_spatial_expectation:
            kp = self.get_spatial_expectation(post)
        else:
            kp = self.get_mode(post)

        distance = None  # TODO: does not work properly anymore -> removed

        kp_raw_2d = kp

        if projection == ProjectionTypes.NONE:
            pass
        elif projection == ProjectionTypes.EGO:
            raise ValueError(
                "Ego projection makes no sense for vanilla kp, "
                "only for GT or particle filter models."
            )
        elif projection == ProjectionTypes.UVD:
            kp = append_depth_to_uv(
                kp, depth, self.image_width - 1, self.image_height - 1
            )
        else:
            if projection in [ProjectionTypes.LOCAL_HARD, ProjectionTypes.LOCAL_SOFT]:
                # create identity extrinsics
                extrinsics = torch.zeros_like(extrinsics)
                extrinsics[:, range(4), range(4)] = 1
            if projection in [ProjectionTypes.LOCAL_SOFT, ProjectionTypes.GLOBAL_SOFT]:
                kp = model_based_vision.soft_pixels_to_3D_world(
                    kp,
                    post,
                    depth,
                    extrinsics,
                    intrinsics,
                    self.image_width - 1,
                    self.image_height - 1,
                )
            else:
                kp = hard_pixels_to_3D_world(
                    kp,
                    depth,
                    extrinsics,
                    intrinsics,
                    self.image_width - 1,
                    self.image_height - 1,
                )

        return kp, distance, kp_raw_2d, prior, sm, post
