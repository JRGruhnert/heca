from heca.environment.scenes.calvin.scene import CalvinScene
from heca.environment.scenes.scene import Scene

import inspect
from heca.misc import logger

query = CalvinScene.Query()
location = CalvinScene.Location()

print(inspect.getfile(Scene))
scene = Scene.load(query, location)
logger.info("Loaded scene")
scene.sample_selection()
Scene.save(query, location)

scene = Scene.load(query, location)
logger.info("Loaded scene")
scene.show_predictions()
