from dataclasses import dataclass

from conf.hoopgnv1.states.area import CalvinAreaStateConfig
from conf.hoopgnv1.states.bool import BoolStateConfig
from conf.hoopgnv1.states.position import PositionStateConfig
from conf.hoopgnv1.states.range import RangeStateConfig
from conf.hoopgnv1.states.rotation import RotationStateConfig
from conf.hoopgnv1.states.flip import FlipStateConfig


@dataclass
class CalvinStateSet:
    ee_position: PositionStateConfig = PositionStateConfig(
        label="ee_position",
        id=0,
    )
    ee_rotation: RotationStateConfig = RotationStateConfig(
        label="ee_rotation",
        id=1,
    )
    ee_scalar: BoolStateConfig = BoolStateConfig(
        label="ee_scalar",
        id=2,
    )
    slide_position: PositionStateConfig = PositionStateConfig(
        label="slide_position",
        id=3,
    )
    slide_rotation: RotationStateConfig = RotationStateConfig(
        label="slide_rotation",
        id=4,
    )
    slide_scalar: RangeStateConfig = RangeStateConfig(
        label="slide_scalar",
        id=5,
        low=0.0,
        high=0.28,
    )
    drawer_position: PositionStateConfig = PositionStateConfig(
        label="drawer_position",
        id=6,
    )
    drawer_rotation: RotationStateConfig = RotationStateConfig(
        label="drawer_rotation",
        id=7,
    )
    drawer_scalar: RangeStateConfig = RangeStateConfig(
        label="drawer_scalar",
        id=8,
        low=0.0,
        high=0.22,
    )
    button_position: PositionStateConfig = PositionStateConfig(
        label="button_position",
        id=9,
    )
    button_rotation: RotationStateConfig = RotationStateConfig(
        label="button_rotation",
        id=10,
    )
    button_scalar: FlipStateConfig = FlipStateConfig(
        label="button_scalar",
        id=11,
    )
    led_position: PositionStateConfig = PositionStateConfig(
        label="led_position",
        id=12,
    )
    led_rotation: RotationStateConfig = RotationStateConfig(
        label="led_rotation",
        id=13,
    )
    block_red_position: CalvinAreaStateConfig = CalvinAreaStateConfig(
        label="block_red_position",
        id=14,
    )
    block_red_rotation: RotationStateConfig = RotationStateConfig(
        label="block_red_rotation",
        id=15,
        # ignore=True,
    )
    block_red_scalar: BoolStateConfig = BoolStateConfig(
        label="block_red_scalar",
        id=16,
    )
    block_blue_position: CalvinAreaStateConfig = CalvinAreaStateConfig(
        label="block_blue_position",
        id=17,
    )
    block_blue_rotation: RotationStateConfig = RotationStateConfig(
        label="block_blue_rotation",
        id=18,
        # ignore=True,
    )
    block_blue_scalar: BoolStateConfig = BoolStateConfig(
        label="block_blue_scalar",
        id=19,
    )
    block_pink_position: CalvinAreaStateConfig = CalvinAreaStateConfig(
        label="block_pink_position",
        id=20,
    )
    block_pink_rotation: RotationStateConfig = RotationStateConfig(
        label="block_pink_rotation",
        id=21,
        # ignore=True,
    )
    block_pink_scalar: BoolStateConfig = BoolStateConfig(
        label="block_pink_scalar",
        id=22,
    )
