import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hoopgn.plots.helper.helper import *

from hoopgn.plots.helper.object_point import ObjectLocationPoint


OBJECT_COLORS: dict[str, str] = {
    "ee": "black",
    "block_red": "tab:red",
    "block_blue": "tab:blue",
    "block_pink": "tab:pink",
}
DEFAULT_COLOR = "gray"

STATE_EDGE_COLORS: list[str] = [
    "black",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
]


class ObjectConditionsPlot:

    def __init__(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.precon_axis = self.fig.add_subplot(121, projection="3d")
        self.postcon_axis = self.fig.add_subplot(122, projection="3d")
        self.preconditions: list[ObjectLocationPoint] = []
        self.postconditions: list[ObjectLocationPoint] = []
        self.precon_tps: list[ObjectLocationPoint] = []
        self.postcon_tps: list[ObjectLocationPoint] = []

    def set_precon(self, point: ObjectLocationPoint):
        self.preconditions.append(point)

    def set_postcon(self, point: ObjectLocationPoint):
        self.postconditions.append(point)

    def set_precon_tp(self, point: ObjectLocationPoint):
        self.precon_tps.append(point)

    def set_postcon_tp(self, point: ObjectLocationPoint):
        self.postcon_tps.append(point)

    def _scatter(self, ax, points: list[ObjectLocationPoint]):
        for p in points:
            facecolor = OBJECT_COLORS.get(p.label, DEFAULT_COLOR)
            edgecolor = STATE_EDGE_COLORS[p.state % len(STATE_EDGE_COLORS)]
            ax.scatter(
                p.x,
                p.y,
                p.z,  # type: ignore
                c=facecolor,
                edgecolors=edgecolor,
                linewidths=1,
                s=30,
            )

    def _scatter_tp(self, ax, points: list[ObjectLocationPoint]):
        for p in points:
            facecolor = OBJECT_COLORS.get(p.label, DEFAULT_COLOR)
            edgecolor = STATE_EDGE_COLORS[p.state % len(STATE_EDGE_COLORS)]
            ax.scatter(
                p.x,
                p.y,
                p.z,  # type: ignore
                c=facecolor,
                edgecolors=edgecolor,
                linewidths=2,
                marker="*",
                s=200,
            )

    def show_objects(self):
        self._scatter(self.precon_axis, self.preconditions)
        self._scatter(self.postcon_axis, self.postconditions)
        self._scatter_tp(self.precon_axis, self.precon_tps)
        self._scatter_tp(self.postcon_axis, self.postcon_tps)

    def _make_legends(
        self,
        ax,
        points: list[ObjectLocationPoint],
        tp_points: list[ObjectLocationPoint],
    ):
        all_pts = points + tp_points
        # Legend A: object labels (fill color)
        seen_labels: list[str] = []
        for p in all_pts:
            if p.label not in seen_labels:
                seen_labels.append(p.label)
        object_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=OBJECT_COLORS.get(label, DEFAULT_COLOR),
                markeredgecolor="none",
                markersize=8,
                label=label,
            )
            for label in seen_labels
        ]
        # Legend B: state (border color)
        seen_states: list[int] = []
        for p in all_pts:
            if p.state not in seen_states:
                seen_states.append(p.state)
        state_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=STATE_EDGE_COLORS[s % len(STATE_EDGE_COLORS)],
                markeredgewidth=1.5,
                markersize=8,
                label=f"State {s}",
            )
            for s in seen_states
        ]
        # Legend C: marker type
        type_handles = []
        if points:
            type_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markeredgecolor="none",
                    markersize=6,
                    label="Demo",
                )
            )
        if tp_points:
            type_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="gray",
                    markeredgecolor="none",
                    markersize=10,
                    label="Task Param",
                )
            )

        leg_a = ax.legend(
            handles=object_handles, title="Objects", loc="upper left", fontsize="small"
        )
        ax.add_artist(leg_a)
        leg_b = ax.legend(
            handles=state_handles, title="States", loc="upper right", fontsize="small"
        )
        if type_handles:
            ax.add_artist(leg_b)
            ax.legend(
                handles=type_handles, title="Type", loc="lower left", fontsize="small"
            )

    def create(self, title: str, show: bool, save: bool, path: str):
        self.show_objects()
        self._make_legends(self.precon_axis, self.preconditions, self.precon_tps)
        self._make_legends(self.postcon_axis, self.postconditions, self.postcon_tps)

        all_points = (
            self.preconditions
            + self.postconditions
            + self.precon_tps
            + self.postcon_tps
        )
        if all_points:
            xs = [p.x for p in all_points]
            ys = [p.y for p in all_points]
            zs = [p.z for p in all_points]
            margin = 0.05
            for ax in [self.precon_axis, self.postcon_axis]:
                ax.set_xlim(min(xs) - margin, max(xs) + margin)
                ax.set_ylim(min(ys) - margin, max(ys) + margin)
                ax.set_zlim(min(zs) - margin, max(zs) + margin)

        for ax in [self.precon_axis, self.postcon_axis]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        self.precon_axis.set_title(f"Skills Object Preconditions")
        self.postcon_axis.set_title(f"Skills Objects Postconditions")
        self.fig.suptitle(title)
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()
