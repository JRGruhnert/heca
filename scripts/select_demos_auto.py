from heca.scenes.ogbench.scene1 import OGScene1
from heca.scenes.ogbench.scene2 import OGScene2
from heca.scenes.ogbench.scene3 import OGScene3
from heca.scenes.ogbench.scene4 import OGScene4
from heca.scenes.ogbench.scene5 import OGScene5
from heca.scenes.scene import Scene

scene = Scene.get(OGScene2.Config())

scene.demo_auto_extract()
