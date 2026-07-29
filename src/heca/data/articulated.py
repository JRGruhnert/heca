from dataclasses import dataclass

from heca.data.entity import Entity, Mobility


class ArticulatedEntity(Entity):
    """Entity with a fixed position but variable rotation / joint state.

    Examples: drawers, sliders, switches.  Position is fixed; rotation
    (or the joint configuration) varies within DOF constraints.
    """

    BASE_LOGSTD = -10.0

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        mobility: Mobility = Mobility.ARTICULATED

    @classmethod
    def gnn_format(
        cls, raw, n_states, logit_confidence=10.0, base_logstd=None
    ):
        if base_logstd is None:
            base_logstd = cls.BASE_LOGSTD
        return super().gnn_format(raw, n_states, logit_confidence, base_logstd)
