from dataclasses import dataclass

from heca.agents.hecas.heca import Heca
from heca.misc.classes import Configurable

from heca.runners.plotter import HecaPlotter


class HecaRunner(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        heca: Heca.Config
        plots: list[HecaPlot.Config]
        episodes: int = 1000

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def train(self):
        assert self.cfg.heca
        assert self.cfg.episodes > 0
        self.heca = Heca.search(self.cfg.heca.network)
        for ep in range(self.cfg.episodes):
            x, y = self.heca.sample(self.meta)
            terminal = False
            while not terminal:
                reward, fb = self.act(x, y, self.meta)
                if fb.can_learn:
                    xp, lvlup = self.ppo.learn(ep)
                    self.network.upgrade(xp)
                    self.network.save(
                        self.network.cfg.query,
                        epoch=ep,
                        label="best" if lvlup else "latest",
                    )
                terminal = fb.terminal
                logger.info(f"Epoch {ep}: Reward={reward:.4f}, Done={terminal}")

    def plot(self):
        assert self.cfg.plots
        for plot_cfg in self.cfg.plots:
            plotter = HecaPlotter(plot_cfg)
            plotter.plot()

        self.heca = Heca.search(self.cfg.heca.network)
        self.make_content(plot, *args, **kwargs)
        plt.title(self.config.title)
        plt.tight_layout()
        if self.config.dry_run:
            logger.debug("Showing plot")
            plt.show(block=True)
        else:
            logger.info(f"Saving plot to {self.config.name} in {self.config.subdir}")
            save_dir = os.path.join(self.config.rootdir, self.config.subdir)
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, f"{self.config.name}_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def explain(self):
        assert self.cfg.heca
        self.heca = Heca.search(self.cfg.heca.network)
        self.heca.explain()

    def explain(self):
        x, y = self.sample(self.meta)
        x, y = self.network.encode(x, y)
        _, data = self.generator(x, y)
        action = self.network.actor(data)  # Forward pass to populate
        explanation = self.explainer(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
            index=action.argmax(dim=-1),
        )

        assert isinstance(explanation, HeteroExplanation)
        return explanation
