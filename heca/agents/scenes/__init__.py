from heca.agents.scenes.legacy.tapas import TapasAgent
from heca.agents.scenes.legacy.reversed import RTapasAgent
from heca.environments.scenes.calvin import CalvinEnvironment

close_drawer = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="close_drawer",
)
close_drawer_back = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="close_drawer_back",
)
open_drawer = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="open_drawer",
)
open_drawer_back = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="open_drawer_back",
)
press_button = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="press_button",
)
press_button_back = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="press_button_back",
)
open_slide = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="open_slide",
)
close_slide = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="close_slide",
)
open_slide_back = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="open_slide_back",
)
close_slide_back = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="close_slide_back",
)
pick_red_table = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_red_table",
)
place_red_table = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_red_table",
)
pick_red_drawer = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_red_drawer",
)
place_red_drawer = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_red_drawer",
)
pick_pink_table = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_pink_table",
)
place_pink_table = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_pink_table",
)
pick_pink_drawer = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_pink_drawer",
)
place_pink_drawer = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_pink_drawer",
)
pick_blue_table = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_blue_table",
)
pick_blue_drawer = TapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="grab_blue_drawer",
)
place_blue_table = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_blue_table",
)
place_blue_drawer = RTapasAgent.Query(
    env=CalvinEnvironment.Query(),
    label="place_blue_drawer",
)
