from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import (
    HoopGNPlot,
    HoopGNPlotterConfig,
)
from hoopgn.plotters.hoopgn_plotters.sampling_plotter import (
    SpawnAreaPlotter,
    SpawnAreaPlotterConfig,
)


HOOPGN_PLOTS_BUILDERS = {
    SpawnAreaPlotterConfig: lambda config: SpawnAreaPlotter(config),
}


def register_training_plot(config_type, builder):
    HOOPGN_PLOTS_BUILDERS[config_type] = builder


def select_training_plot(config: HoopGNPlotterConfig) -> HoopGNPlot:
    builder = HOOPGN_PLOTS_BUILDERS.get(type(config))  # type: ignore
    if builder is None:
        for cfg_type, b in HOOPGN_PLOTS_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)


def select_multiple_hoopgn_plotters(
    configs: list[HoopGNPlotterConfig],
) -> list[HoopGNPlot]:
    return [select_training_plot(c) for c in configs]
