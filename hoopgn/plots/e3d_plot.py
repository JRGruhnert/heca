from dataclasses import dataclass
import os
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from hoopgn.entities.entity import Entity
from hoopgn.observation.td_entity import TDEntity
from hoopgn.plots.plot import Plot, PlotConfig
from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.plots.helper.helper import *
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


@dataclass
class EntityPoint:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    rw: float
    state: bool


@dataclass
class EntityData:
    entity: Entity
    current: EntityPoint
    goal: EntityPoint
    different: bool
    solved: bool


@dataclass
class Entity3DConfig(PlotConfig):
    subdir: str = "3d"


class Entity3DPlot(Plot):
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.ax_different = self.fig.add_subplot(121, projection="3d")
        self.ax_same = self.fig.add_subplot(122, projection="3d")
        self.axes = {
            "different": self.ax_different,
            "same": self.ax_same,
        }
        self.entities: list[EntityData] = []

    def __call__(self):
        pass

    def set_entity(
        self,
        entity: Entity,
        current: TDEntity,
        goal: TDEntity,
        different: bool,
        solved: bool,
    ):
        self.entities.append(
            EntityData(
                entity,
                self.get_entity_point(current),
                self.get_entity_point(goal),
                different,
                solved,
            )
        )

    def get_entity_point(self, entity: TDEntity) -> EntityPoint:
        return EntityPoint(
            x=entity.position[0].item(),
            y=entity.position[1].item(),
            z=entity.position[2].item(),
            rx=entity.rotation[0].item(),
            ry=entity.rotation[1].item(),
            rz=entity.rotation[2].item(),
            rw=entity.rotation[3].item(),
            state=bool(entity.state[0].item()),
        )

    def get_solved_color(self, solved: bool) -> str:
        return self.config.style.solved if solved else self.config.style.unsolved

    def get_state_color(self, state: bool) -> str:
        return self.config.style.true if state else self.config.style.false

    def get_entity_color(self, label: str) -> str:
        return self.config.style.entity.get(label, self.config.style.white)

    def get_axis(self, different: bool):
        return self.axes.get("different" if different else "same", self.ax_same)

    def show_entities(self):
        for ed in self.entities:
            ec = self.get_solved_color(ed.solved)
            fc = self.get_entity_color(ed.entity.config.label)
            ax = self.get_axis(ed.different)
            self.show_location(ax, ed.current, ec, fc)
            self.show_location(ax, ed.goal, ec, fc)
            self.show_rotation(ax, ed.current, ec)
            self.show_rotation(ax, ed.goal, ec)

    def show_location(
        self, axis: Axes, point: EntityPoint, edge_color: str, face_color: str
    ):
        axis.scatter(
            point.x, point.y, point.z, c=face_color, edgecolors=edge_color, linewidths=1
        )

    def show_rotation(self, axis: Axes, point: EntityPoint, color: str):
        dir = R.from_quat([point.rx, point.ry, point.rz, point.rw]).apply([0.01, 0, 0])
        axis.quiver(
            point.x,
            point.y,
            point.z,
            dir[0],
            dir[1],
            dir[2],
            length=1,
            color=color,
            arrow_length_ratio=0.1,
        )

    def show_edges(self):
        for ed in self.entities:
            p1 = ed.current
            p2 = ed.goal
            diff = ed.different
            solved = ed.solved
            ax = self.get_axis(diff)
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                [p1.z, p2.z],
                color=self.get_solved_color(solved),
            )

    def show_areas(self, area: AreaStateConfig):
        for axis in self.axes.values():
            for surface in area.eval_surfaces.values():
                (x0, y0, z0), (x1, y1, z1) = surface
                corners = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    ]
                )
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],  # bottom
                    [corners[4], corners[5], corners[6], corners[7]],  # top
                    [corners[0], corners[1], corners[5], corners[4]],  # front
                    [corners[2], corners[3], corners[7], corners[6]],  # back
                    [corners[1], corners[2], corners[6], corners[5]],  # right
                    [corners[3], corners[0], corners[4], corners[7]],  # left
                ]
                poly = Poly3DCollection(
                    faces, color="grey", alpha=0.2, linewidths=0.5, edgecolors="k"
                )
                axis.add_collection3d(poly)

    def create(self, title: str, show: bool, save: bool, path: str):
        # Set axis labels
        for ax in [self.ax_different, self.ax_same]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        # Calculate solved ratios for different and same
        solved_different_count = sum(e.solved == e.different for e in self.entities)
        solved_same_count = sum(e.solved != e.different for e in self.entities)
        different_count = sum(e.different for e in self.entities)
        same_count = len(self.entities) - different_count
        self.ax_different.set_title(
            f"Start != Goal (solved: {solved_different_count}/{different_count})"
        )
        self.ax_same.set_title(
            f"Start == Goal (solved: {solved_same_count}/{same_count})"
        )
        self.fig.suptitle(title)
        result_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=self.get_solved_color(True),
                markeredgewidth=1.5,
                markersize=8,
                label="Solved",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=self.get_solved_color(False),
                markeredgewidth=1.5,
                markersize=8,
                label="Unsolved",
            ),
        ]

        for ax in [self.ax_different, self.ax_same]:
            leg_a = ax.legend(
                handles=result_handles,
                title="Result",
                loc="upper right",
                fontsize="small",
            )
            ax.add_artist(leg_a)
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()
