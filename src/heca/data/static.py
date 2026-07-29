from dataclasses import dataclass

from heca.data.entity import Entity, Mobility


class StaticEntity(Entity):
    """Entity with a fixed position and rotation in the scene.

    Static entities have minimal uncertainty — they are deterministic
    landmarks such as buttons, lights, or fixed structures.
    """

    BASE_LOGSTD = -10.0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        mobility: Mobility = Mobility.STATIC

    @classmethod
    def gnn_format(
        cls, raw, n_states, logit_confidence=10.0, base_logstd=None
    ):
        if base_logstd is None:
            base_logstd = cls.BASE_LOGSTD
        return super().gnn_format(raw, n_states, logit_confidence, base_logstd)
