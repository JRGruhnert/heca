from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGScene
from heca.guis.tapas_demo_processor import TapasDemoProcessor
from heca.scenes.ogbench.scene2 import OGScene2
from heca.scenes.scene import Scene

# cfg = TapasDemoProcessor.Config(
#     agent=TapasAgent.Config(
#         tag="move_block_drawer",
#         scene=OGScene.Config(),
#     )
# )
# selector = TapasDemoProcessor.get(cfg)

# selector.run()


scene = Scene.get(OGScene2.Config())

scene.demo_auto_extract()
