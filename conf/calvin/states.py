from dataclasses import dataclass

from conf.calvin.area import CalvinAreaStateConfig
from src.states.state_bool import BoolStateConfig
from src.states.state_location import LocationStateConfig
from src.states.state_range import RangeStateConfig
from src.states.state_rotation import RotationStateConfig
from src.states.state_switch import SwitchStateConfig


@dataclass
class CalvinStateSet:
    ee_position = LocationStateConfig(label="ee_position", id=0)  # type: ignore
    ee_rotation = RotationStateConfig(label="ee_rotation", id=1)  # type: ignore
    ee_scalar = BoolStateConfig(label="ee_scalar", id=2)  # type: ignore
    slide_position = LocationStateConfig(label="slide_position", id=3)  # type: ignore
    slide_rotation = RotationStateConfig(label="slide_rotation", id=4)  # type: ignore
    slide_scalar = RangeStateConfig(label="slide_scalar", id=5, low=0.0, high=0.28)
    drawer_position = LocationStateConfig(label="drawer_position", id=6)  # type: ignore
    drawer_rotation = RotationStateConfig(label="drawer_rotation", id=7)  # type: ignore
    drawer_scalar = RangeStateConfig(label="drawer_scalar", id=8, low=0.0, high=0.22)
    button_position = LocationStateConfig(label="button_position", id=9)  # type: ignore
    button_rotation = RotationStateConfig(label="button_rotation", id=10)
    button_scalar = SwitchStateConfig(label="button_scalar", id=11)  # type: ignore
    led_position = LocationStateConfig(label="led_position", id=12)  # type: ignore
    led_rotation = RotationStateConfig(label="led_rotation", id=13)  # type: ignore
    block_red_position = CalvinAreaStateConfig(
        label="block_red_position", id=14
    )  # type: ignore
    block_red_rotation = RotationStateConfig(
        label="block_red_rotation", id=15, ignore=True
    )  # type: ignore
    block_red_scalar = BoolStateConfig(label="block_red_scalar", id=16)  # type: ignore
    block_blue_position = CalvinAreaStateConfig(
        label="block_blue_position", id=17
    )  # type: ignore
    block_blue_rotation = RotationStateConfig(
        label="block_blue_rotation", id=18, ignore=True
    )  # type: ignore
    block_blue_scalar = BoolStateConfig(label="block_blue_scalar", id=19)  # type: ignore
    block_pink_position = CalvinAreaStateConfig(
        label="block_pink_position", id=20
    )  # type: ignore
    block_pink_rotation = RotationStateConfig(
        label="block_pink_rotation", id=21, ignore=True
    )  # type: ignore
    block_pink_scalar = BoolStateConfig(label="block_pink_scalar", id=22)  # type: ignore
