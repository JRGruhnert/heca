from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import (
    HoopGNPlot,
    HoopGNPlotConfig,
)


TRAININGS_PLOTS_BUILDERS = {}


def register_training_plot(config_type, builder):
    TRAININGS_PLOTS_BUILDERS[config_type] = builder


def select_training_plot(config: HoopGNPlotConfig) -> HoopGNPlot:
    builder = TRAININGS_PLOTS_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in TRAININGS_PLOTS_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_multiple_hoopgn_plotters(
    configs: list[HoopGNPlotConfig],
) -> list[HoopGNPlot]:
    return [select_training_plot(c) for c in configs]
