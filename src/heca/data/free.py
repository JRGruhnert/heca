from dataclasses import dataclass

from heca.data.entity import Entity, Mobility


class FreeEntity(Entity):
    """Entity that can be moved freely through the scene.

    Free entities have larger uncertainty bounds since their position and
    rotation are not constrained.
    """

    BASE_LOGSTD = -5.0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        mobility: Mobility = Mobility.FREE

    @classmethod
    def gnn_format(
        cls, raw, n_states, logit_confidence=10.0, base_logstd=None
    ):
        if base_logstd is None:
            base_logstd = cls.BASE_LOGSTD
        return super().gnn_format(raw, n_states, logit_confidence, base_logstd)
