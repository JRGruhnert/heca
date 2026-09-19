"""Projecting features into the no-rotation layout.

The no-rotation layouts drop the whole ``rot`` block, which shifts every block
after it to the left.  ``Entity.project`` therefore has to read the *source*
block offsets (the layout the features were written in), not the destination
ones: reusing destination offsets made the ``extra`` block - the joint value of a
drawer, window or button - hold orientation values instead, which made those
entities look identical in the start and in the goal state.
"""

import numpy as np
import torch

from heca.data.entity import Entity


def test_layout_of_width_round_trips() -> None:
    for layout in (
        Entity.LAYOUT,
        Entity.POINT_LAYOUT,
        Entity.NO_ROT_LAYOUT,
        Entity.NO_ROT_POINT_LAYOUT,
    ):
        width = sum(block.dim for block in layout.values())
        assert Entity.layout_of_width(width) is layout

    for width in (12, 26):
        try:
            Entity.layout_of_width(width)
        except ValueError:
            continue
        raise AssertionError(f"width {width} should not name a layout")


def test_project_keeps_block_values() -> None:
    """Every projected column must come from the same block of the source."""
    for src_layout, dst_layout in (
        (Entity.LAYOUT, Entity.NO_ROT_LAYOUT),
        (Entity.POINT_LAYOUT, Entity.NO_ROT_POINT_LAYOUT),
    ):
        width = sum(block.dim for block in src_layout.values())
        source = np.arange(width, dtype=np.float32).reshape(1, -1)
        projected = Entity.project(source, dst_layout)
        offset = 0
        for name, block in dst_layout.items():
            src = src_layout[name]
            assert np.allclose(
                projected[0, offset : offset + block.mean_dim],
                source[0, src.mean()],
            ), f"{name} mean block"
            offset += block.mean_dim
            if block.logstd_dim:
                assert np.allclose(
                    projected[0, offset : offset + block.logstd_dim],
                    source[0, src.logstd()],
                ), f"{name} log-std block"
                offset += block.logstd_dim
        assert offset == projected.shape[-1]


def test_extra_block_is_not_orientation() -> None:
    """The trap: the extra slots used to receive the dropped rot columns."""
    source = np.arange(Entity.FEATURE_DIM, dtype=np.float32).reshape(1, -1)
    projected = Entity.project(source, Entity.NO_ROT_LAYOUT)
    extra = Entity.NO_ROT_LAYOUT["extra"]
    assert np.allclose(projected[0, extra.mean()], source[0, Entity.LAYOUT["extra"].mean()])
    assert np.allclose(
        projected[0, extra.logstd()], source[0, Entity.LAYOUT["extra"].logstd()]
    )
    rot = Entity.LAYOUT["rot"]
    dropped = set(source[0, rot.mean()].tolist()) | set(source[0, rot.logstd()].tolist())
    assert not (set(projected[0].tolist()) & dropped), "rot columns leaked into the projection"


def test_project_accepts_tensors_and_rejects_bad_width() -> None:
    source = torch.arange(Entity.FEATURE_DIM, dtype=torch.float32).reshape(1, -1)
    projected = Entity.project(source, Entity.NO_ROT_LAYOUT)
    assert isinstance(projected, torch.Tensor)
    assert projected.shape == (1, Entity.NO_ROT_FEATURE_DIM)

    try:
        Entity.project(np.zeros((1, 12)), Entity.NO_ROT_LAYOUT)
    except ValueError:
        pass
    else:
        raise AssertionError("a width that names no layout must be rejected")
