from heca.environment.scenes.calvin.scene import CalvinScene
from heca.environment.scenes.scene import Scene

import inspect
from heca.misc import logger

print(inspect.getfile(Scene))
scene = Scene.load(CalvinScene.Query(), CalvinScene.Location())
logger.info("Loaded scene")
scene.sample_selection()
Scene.save(CalvinScene.Query(), CalvinScene.Location())
