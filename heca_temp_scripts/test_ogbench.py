from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.environment.scenes.scene import Scene

scene = Scene.load(OGBenchScene.Config())
scene.test()
