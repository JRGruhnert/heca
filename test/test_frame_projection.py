import numpy as np
import torch

from heca.image_encoders.dino_encoder import DinoEncoder
from heca.image_encoders.image_encoder import ImageEncoder

IDENTITY_INTR = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
IDENTITY_EXTR = torch.eye(4)


def depth_at(row: int, col: int, value: float, size: int = 4) -> torch.Tensor:
    depth = torch.full((size, size), 2.0)
    depth[row, col] = value
    return depth


def test_a_single_depth_map_projects_the_pixel_it_covers() -> None:
    """``(H, W)`` in, the world point of that pixel out (identity camera)."""
    point = ImageEncoder.hard_pixels_to_3D_world(
        torch.tensor([[1]]),
        torch.tensor([[2]]),
        depth_at(1, 2, 3.0),
        IDENTITY_EXTR,
        IDENTITY_INTR,
    )

    assert point.shape == (1, 1, 3)
    assert torch.allclose(point[0, 0], torch.tensor([2.0 * 3.0, 1.0 * 3.0, 3.0]))


def test_a_depth_stack_gives_the_same_point_as_a_single_map() -> None:
    """The two shapes are the same measurement, so they must agree."""
    depth = depth_at(1, 2, 3.0)
    y, x = torch.tensor([[1]]), torch.tensor([[2]])

    single = ImageEncoder.hard_pixels_to_3D_world(
        y, x, depth, IDENTITY_EXTR, IDENTITY_INTR
    )
    stacked = ImageEncoder.hard_pixels_to_3D_world(
        y, x, depth[None], IDENTITY_EXTR[None], IDENTITY_INTR[None]
    )

    assert torch.allclose(single, stacked)


def test_several_pixels_are_read_from_one_map() -> None:
    """Each pixel takes its own depth, which is what the encoder needs."""
    depth = depth_at(0, 0, 2.0)
    depth[2, 3] = 5.0

    points = ImageEncoder.hard_pixels_to_3D_world(
        torch.tensor([[0, 2]]),
        torch.tensor([[0, 3]]),
        depth,
        IDENTITY_EXTR,
        IDENTITY_INTR,
    )

    assert torch.allclose(points[0, 0], torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(points[0, 1], torch.tensor([15.0, 10.0, 5.0]))


def test_pixel_to_world_round_trips_through_the_camera() -> None:
    """The clicked-reference path and the keypoint path stay consistent."""
    depth = np.full((4, 4), 4.0, dtype=np.float32)
    from_pixel = ImageEncoder.pixel_to_world(
        1, 2, depth, IDENTITY_EXTR.numpy(), IDENTITY_INTR.numpy()
    )
    from_keypoint = ImageEncoder.hard_pixels_to_3D_world(
        torch.tensor([[1]]),
        torch.tensor([[2]]),
        torch.as_tensor(depth),
        IDENTITY_EXTR,
        IDENTITY_INTR,
    ).numpy()[0, 0]

    assert np.allclose(from_pixel, from_keypoint)
    assert np.allclose(from_pixel, [8.0, 4.0, 4.0])


class _Coordinates(ImageEncoder):
    """The coordinate helpers alone: no weights, no scene, no frames."""

    def prepare_for_scene(self, cfg):  # pragma: no cover - never called here
        raise NotImplementedError


def test_a_click_picks_the_patch_of_the_pixel_it_was_made_on() -> None:
    """A reference is an ``(x, y)`` click; a descriptor grid is indexed by row, column.

    Handing the click's pair straight to ``transform_coords``, which takes the row
    first, read every reference patch off the transposed pixel. The frame is
    square, so the swap cost no scale and went unnoticed -- the patch simply came
    from the wrong place and every keypoint was tracked from there.
    """
    encoder = _Coordinates(DinoEncoder.Config())
    stride, patch_size, size = 8, 16, 256
    click_x, click_y = 40, 200

    row, col = encoder.patch_index(
        x=click_x, y=click_y, origin_hw=(size, size), target_hw=(31, 31)
    )

    assert (row, col) == (24, 5)
    assert (
        row * stride <= click_y < row * stride + patch_size
    ), "row follows the click's y"
    assert (
        col * stride <= click_x < col * stride + patch_size
    ), "col follows the click's x"

    transposed = encoder.patch_index(
        x=click_y, y=click_x, origin_hw=(size, size), target_hw=(31, 31)
    )
    assert transposed == (5, 24), "the other order is a different patch"


def test_patch_index_is_the_documented_order_of_transform_coords() -> None:
    """One wraps the other, so they must not drift apart."""
    encoder = _Coordinates(DinoEncoder.Config())

    assert encoder.patch_index(
        x=40, y=200, origin_hw=(256, 256), target_hw=(31, 31)
    ) == encoder.transform_coords(200, 40, 256, 256, 31, 31)


def test_each_axis_is_scaled_by_its_own_side() -> None:
    """On a non-square frame the same swap would scale wrongly as well as move."""
    encoder = _Coordinates(DinoEncoder.Config())

    row, col = encoder.patch_index(
        x=40, y=200, origin_hw=(100, 300), target_hw=(25, 60)
    )

    # 200 of 99 rows -> cell 48; 40 of 299 columns -> cell 8 (the helper rounds)
    assert (row, col) == (48, 8), "the row follows the height, the column the width"
