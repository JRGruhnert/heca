from heca.scenes.ogbench.scene1 import OGScene1
from heca.scenes.ogbench.scene2 import OGScene2
from heca.scenes.ogbench.scene3 import OGScene3
from heca.scenes.ogbench.scene4 import OGScene4
from heca.scenes.ogbench.scene5 import OGScene5
from heca.scenes.scene import Scene

scene1 = Scene.get(OGScene1.Config())
scene1.demo_auto_extract()

scene2 = Scene.get(OGScene2.Config())
scene2.demo_auto_extract()

scene3 = Scene.get(OGScene3.Config())
scene3.demo_auto_extract()

scene4 = Scene.get(OGScene4.Config())
scene4.demo_auto_extract()

scene5 = Scene.get(OGScene5.Config())
scene5.demo_auto_extract()
