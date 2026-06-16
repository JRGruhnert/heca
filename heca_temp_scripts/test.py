print("TF side loaded")

print("OGBench loaded")
import os

# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.environment.scenes.scene import Scene
from heca.image_encoders.molmo_encoder import MolmoEncoder

print("before scene")

scene = Scene.get(OGBenchScene.Config())

print("after scene")
