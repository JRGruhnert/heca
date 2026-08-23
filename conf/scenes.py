"""Scene module registry.

``SCENE_MODULES`` is the single source of truth for the scene conf modules used
by the pipeline scripts. Import it here instead of listing ``conf.experts.*``
in each script.
"""

import conf.experts.scene1
import conf.experts.scene2
import conf.experts.scene3
import conf.experts.scene4
import conf.experts.scene5

# import conf.experts.sceneog

SCENE_MODULES = (
    conf.experts.scene1,
    conf.experts.scene2,
    conf.experts.scene3,
    conf.experts.scene4,
    conf.experts.scene5,
    # conf.experts.sceneog,
)
