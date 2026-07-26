from dataclasses import dataclass

from heca.scenes.ogbench.scene import OGBenchScene


class VirtOGBenchScene(OGBenchScene):
    @dataclass(kw_only=True)
    class Config(OGBenchScene.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
