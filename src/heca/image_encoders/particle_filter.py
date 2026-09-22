from dataclasses import dataclass

import numpy as np
import torch

from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc import hardware


@dataclass(kw_only=True)
class ParticleFilterConfig:
    """TAPAS's filter defaults, with the fields this port actually reads."""

    n_particles: int = 250
    resampling_fraction: float = 0.1
    filter_noise_scale: float = 0.01
    use_gripper_motion: bool = False
    gripper_motion_prob: float = 0.25
    # likelihood for a particle that projects outside the image
    descriptor_distance_for_outside_pixels: float = 1.0
    depth_likelihood_for_outside_pixels: float = 0.25
    # particles are born where the match is good, with this extra taper
    taper_initial_sample: bool = True
    extra_init_taper: float = 1.0
    init_depth_sigma_rel: float = 0.01
    init_depth_sigma_abs: float = 0.001
    # depth agreement: measured vs projected
    depth_model_eps: float = 0.1
    depth_model_sigma_rel: float = 0.02
    depth_model_sigma_abs: float = 0.001
    # after a resample, mix in fresh observation samples
    sample_from_each_obs: bool = True
    obs_sample_thresh: float = 0.05
    extra_resample_taper: float = 1.0


def normalize(weights: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    weights = weights + eps
    return weights / weights.sum(keepdim=True, dim=-1)


def get_neff(weights: torch.Tensor) -> torch.Tensor:
    return 1.0 / weights.square().sum(dim=-1)


def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """The low-variance resampler: one uniform draw per keypoint, not per particle."""
    n_particles = weights.shape[-1]
    positions = (
        torch.rand_like(weights[:, :, 0]).unsqueeze(-1).repeat(1, 1, n_particles)
        + torch.arange(n_particles, device=weights.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(weights.shape[0], weights.shape[1], 1)
    ) / n_particles
    indices = torch.empty_like(positions, dtype=torch.double)
    cumulative_sum = torch.cumsum(weights, dim=-1)
    for p in range(n_particles):
        smaller = positions[..., p].unsqueeze(-1) < cumulative_sum
        idx = torch.arange(smaller.shape[-1], 0, -1, device=weights.device)
        indices[..., p] = torch.argmax(smaller * idx, dim=-1)
    return indices.long()


def gaussian_cdf(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * np.sqrt(2.0))))


class KeypointParticleFilter:
    def __init__(self, config: ParticleFilterConfig, taper: int):
        self.cfg = config
        self.taper = taper
        self.reset()

    def reset(self) -> None:
        self.coordinates: torch.Tensor | None = None  # (1, K, P, 3) world points
        self.weights: torch.Tensor | None = None  # (1, K, P)
        self.last_gripper_pose: torch.Tensor | None = None
        self.n_keypoints: int = 0
        self.image_size: tuple[int, int] | None = None

    @property
    def ready(self) -> bool:
        return self.coordinates is not None

    def base_likelihood(self, similarity: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.taper * similarity)

    def occlusion_model(
        self, measurement: torch.Tensor, hypothesis: torch.Tensor
    ) -> torch.Tensor:
        """Probability that something sits in front of the particle.

        A particle behind the measured surface is occluded; the further behind,
        the more certain.  Particles that project outside the image cannot be
        judged and are treated as unoccluded.
        """
        occluded = 1.0 - gaussian_cdf(
            measurement,
            hypothesis - self.cfg.depth_model_eps,
            self.cfg.depth_model_sigma_rel * hypothesis
            + self.cfg.depth_model_sigma_abs,
        )
        # a measured depth of zero means the map has no reading there, and an
        # unoccluded particle is the safe assumption (as in TAPAS)
        return torch.where(measurement == 0, torch.ones_like(occluded), occluded)

    def measurement_model(
        self,
        similarity: torch.Tensor,  # (1, K, P) at the particles' pixels
        outside: torch.Tensor,  # (1, K, P) projects out of the image
        measured_depth: torch.Tensor,  # (1, K, P) from the depth map
        projected_depth: torch.Tensor,  # (1, K, P) from the particle
    ) -> torch.Tensor:
        """TAPAS's measurement model, verbatim in structure and defaults."""
        descriptor_likelihood = self.base_likelihood(similarity)

        depth_model = torch.distributions.Normal(
            projected_depth,
            torch.clamp(projected_depth * self.cfg.depth_model_sigma_rel, min=0.0)
            + self.cfg.depth_model_sigma_abs,
        )
        # subtracting the log-probability of the mean normalises to [0, 1]
        depth_likelihood = torch.exp(
            depth_model.log_prob(measured_depth) - depth_model.log_prob(projected_depth)
        )

        occluded = self.occlusion_model(measured_depth, projected_depth)
        occluded = torch.where(outside, torch.ones_like(occluded), occluded)

        outside_descriptor = self.base_likelihood(
            torch.full_like(
                similarity, -self.cfg.descriptor_distance_for_outside_pixels
            )
        )
        outside_depth = torch.full_like(
            measured_depth, self.cfg.depth_likelihood_for_outside_pixels
        )
        return (1.0 - occluded) * descriptor_likelihood * depth_likelihood + (
            occluded * outside_descriptor * outside_depth
        )

    def sample_from_obs(
        self,
        similarity: torch.Tensor,  # (1, K, H, W)
        depth: torch.Tensor,
        extr: torch.Tensor,
        intr: torch.Tensor,
        depth_sigma_rel: float,
        depth_sigma_abs: float,
    ) -> torch.Tensor:
        """Particles at image locations where the match is good, at that depth.

        Their sampler: a categorical over the (softmaxed) similarity map picks a
        pixel per particle, the depth map turns it into a 3D point, and the depth
        carries the noise, so a particle's spread follows the measurement's.
        """
        B, K, H2, W2 = similarity.shape
        H, W = depth.shape
        distribution = torch.distributions.Categorical(
            probs=torch.softmax(similarity.reshape(B, K, H2 * W2), dim=2)
        )
        samples = distribution.sample((self.cfg.n_particles,)).movedim(0, 2)
        pixels = (
            torch.stack(
                (
                    samples % W2 * (W / W2),  # x
                    torch.div(samples, W2, rounding_mode="floor") * (H / H2),  # y
                ),
                dim=-1,
            )
            .reshape(B * K, self.cfg.n_particles, 2)
            .long()
        )

        # one camera, so every keypoint's particles share its depth and matrices
        flat_depth = depth.unsqueeze(0).repeat(B * K, 1, 1)
        flat_extr = extr.unsqueeze(0).repeat(B * K, 1, 1)
        flat_intr = intr.unsqueeze(0).repeat(B * K, 1, 1)
        world = noisy_pixel_coordinates_to_world(
            pixels, flat_depth, flat_extr, flat_intr, depth_sigma_rel, depth_sigma_abs
        )
        return world.reshape(B, K, self.cfg.n_particles, 3)

    def sample_particles(
        self,
        similarity: torch.Tensor,
        depth: torch.Tensor,
        extr: torch.Tensor,
        intr: torch.Tensor,
    ) -> None:
        self.n_keypoints = similarity.shape[1]
        tapered = similarity
        if self.cfg.taper_initial_sample:
            tapered = similarity * self.taper * self.cfg.extra_init_taper
        self.coordinates = self.sample_from_obs(
            tapered,
            depth,
            extr,
            intr,
            self.cfg.init_depth_sigma_rel,
            self.cfg.init_depth_sigma_abs,
        )
        self.weights = normalize(torch.ones_like(self.coordinates[..., 0]))

    def predict_motion(self, gripper_pose: torch.Tensor | None) -> torch.Tensor:
        """Carry the particles forward: gripper motion first, then noise."""
        assert self.coordinates is not None
        coordinates = self.coordinates
        if gripper_pose is not None and self.cfg.use_gripper_motion:
            gripper_pose = gripper_pose.to(coordinates.dtype)
            if self.last_gripper_pose is not None:
                delta = gripper_pose[:, :3] - self.last_gripper_pose[:, :3]
                random_mask = (
                    torch.rand(coordinates.shape[:-1], device=coordinates.device)
                    < self.cfg.gripper_motion_prob
                )
                coordinates = coordinates + random_mask.unsqueeze(-1) * delta
        noise = torch.distributions.Normal(0.0, self.cfg.filter_noise_scale).sample(
            coordinates.shape
        )
        return coordinates + noise.to(coordinates.device)

    def update(
        self,
        similarity: torch.Tensor,  # (K, H, W) per-entity similarity maps
        depth: torch.Tensor,  # (H, W)
        extr: torch.Tensor,  # (4, 4) camera to world
        intr: torch.Tensor,  # (3, 3)
        gripper_pose: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        """One frame: predict, score, resample, and report the estimate."""
        maps = similarity.unsqueeze(0)  # (1, K, H, W)
        depth = depth.to(hardware.device)
        extr = extr.to(hardware.device)
        intr = intr.to(hardware.device)
        self.image_size = (int(depth.shape[0]), int(depth.shape[1]))

        if not self.ready:
            self.sample_particles(maps, depth, extr, intr)
            return self.estimate_state()

        self.coordinates = self.predict_motion(
            None if gripper_pose is None else gripper_pose.to(hardware.device)
        )
        B, K, P, _ = self.coordinates.shape
        H, W = depth.shape
        _, _, H2, W2 = maps.shape

        pixels, projected_depth = project_onto_image(
            self.coordinates, depth, extr, intr, clip_value=-1
        )
        outside = (pixels == -1).sum(dim=-1) > 0
        grid = torch.stack(
            (
                torch.clamp(torch.round(pixels[..., 0] / W * W2), 0, W2 - 1),
                torch.clamp(torch.round(pixels[..., 1] / H * H2), 0, H2 - 1),
            ),
            dim=-1,
        ).long()

        flat = grid.reshape(B * K * P, 2)
        batch_index = torch.arange(B).repeat_interleave(K * P)
        keypoint_index = torch.arange(K).repeat_interleave(P).repeat(B)
        particle_similarity = maps[batch_index, keypoint_index, flat[:, 1], flat[:, 0]]
        particle_similarity = particle_similarity.reshape(B, K, P)

        # the depth map on the descriptor grid, sampled at the particles' pixels
        coarse_depth = torch.nn.functional.interpolate(
            depth.reshape(1, 1, H, W),
            size=(H2, W2),
            mode="bilinear",
            align_corners=True,
        ).reshape(H2, W2)
        measured_depth = coarse_depth[flat[:, 1], flat[:, 0]].reshape(B, K, P)

        likelihood = self.measurement_model(
            particle_similarity, outside, measured_depth, projected_depth
        )
        self.weights = normalize(self.weights * likelihood)

        resample_mask = get_neff(self.weights) < (
            self.cfg.resampling_fraction * self.cfg.n_particles
        )
        if bool(resample_mask.any()):
            resampled = self.coordinates[
                batch_index, keypoint_index, systematic_resample(self.weights).flatten()
            ].reshape(B, K, P, 3)
            # the two masks have different ranks: one for the (B,K,P,3) points,
            # one for the (B,K,P) weights.  Using the point mask on the weights
            # broadcasts them into (B,K,1,P) and quietly wrecks the estimate.
            point_mask = resample_mask.unsqueeze(-1).unsqueeze(-1)
            weight_mask = resample_mask.unsqueeze(-1)
            self.coordinates = torch.where(point_mask, resampled, self.coordinates)
            self.weights = normalize(
                torch.where(weight_mask, torch.ones_like(self.weights), self.weights)
            )

            if self.cfg.sample_from_each_obs:
                fresh = self.sample_from_obs(
                    maps * self.taper * self.cfg.extra_resample_taper,
                    depth,
                    extr,
                    intr,
                    self.cfg.depth_model_sigma_rel,
                    self.cfg.depth_model_sigma_abs,
                )
                random_mask = (
                    torch.rand(fresh.shape[:-1], device=hardware.device)
                    < self.cfg.obs_sample_thresh
                )
                self.coordinates = torch.where(
                    random_mask.unsqueeze(-1), fresh, self.coordinates
                )

        self.weights = normalize(self.weights)
        self.last_gripper_pose = (
            None
            if gripper_pose is None
            else gripper_pose.detach().clone().to(torch.float32)
        )
        return self.estimate_state()

    def estimate_state(self) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        """Weighted mean of the particles, per keypoint, plus their spread."""
        assert self.coordinates is not None and self.weights is not None
        weights = self.weights / self.weights.sum(dim=2, keepdim=True)
        mean = (self.coordinates * weights.unsqueeze(-1)).sum(dim=-2)  # (1, K, 3)
        spread = torch.var(self.coordinates, dim=-2).mean(dim=-1)  # (1, K)
        info = {
            "spread": spread.detach().cpu(),
            "weights": self.weights.detach().cpu(),
            "particles": self.coordinates.detach().cpu(),
        }
        return mean[0].detach().cpu().numpy().astype(np.float64), info


def project_onto_image(
    coordinates: torch.Tensor,  # (B, K, P, 3)
    depth: torch.Tensor,  # (H, W)
    extr: torch.Tensor,  # (4, 4)
    intr: torch.Tensor,  # (3, 3)
    clip_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the particles land in the image, and at what camera depth."""
    B, K, P, _ = coordinates.shape
    pixels, camera_depth = ImageEncoder.world_to_pixels(
        coordinates.reshape(-1, 3), extr, intr
    )
    pixels = pixels.reshape(B, K, P, 2)
    camera_depth = camera_depth.reshape(B, K, P)
    H, W = depth.shape
    inside = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < W)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < H)
        & (camera_depth > 0)
    )
    return (
        torch.where(inside.unsqueeze(-1), pixels, torch.full_like(pixels, clip_value)),
        camera_depth,
    )


def noisy_pixel_coordinates_to_world(
    pixels: torch.Tensor,  # (N, P, 2) as (x, y)
    depth: torch.Tensor,  # (N, H, W)
    extr: torch.Tensor,  # (N, 4, 4)
    intr: torch.Tensor,  # (N, 3, 3)
    relative_noise_scale: float,
    absolute_noise_scale: float,
) -> torch.Tensor:
    """Pixel + measured depth -> world point, with the depth noise TAPAS uses."""
    N, P = pixels.shape[0], pixels.shape[1]
    x, y = pixels[..., 0], pixels[..., 1]
    rows = torch.arange(N, device=depth.device).repeat_interleave(P)
    z = depth[rows, y.flatten(), x.flatten()].reshape(N, P)
    z = z + torch.distributions.Normal(
        0.0, z * relative_noise_scale + absolute_noise_scale
    ).sample().to(z.device)
    return ImageEncoder.pixels_to_world(x, y, z, extr, intr)
