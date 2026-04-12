import matplotlib.pyplot as plt
import numpy as np
from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.plots.helper.helper import *
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

OBJECT_COLORS: dict[str, str] = {
    "ee": "black",
    "block_red": "tab:red",
    "block_blue": "tab:blue",
    "block_pink": "tab:pink",
    "slide": "black",
    "drawer": "black",
    "button": "black",
    "led": "black",
    "lightbulb": "black",
}
DEFAULT_COLOR = "white"

SOLVED_COLOR = "tab:green"
UNSOLVED_COLOR = "tab:red"
TRUE_COLOR = "tab:yellow"
FALSE_COLOR = "black"


class ObjectSamplingPlot:
    def __init__(self, object_label: str = ""):
        self.fig = plt.figure(figsize=(20, 10))
        self.ax_different = self.fig.add_subplot(121, projection="3d")
        self.ax_same = self.fig.add_subplot(122, projection="3d")
        self.axes = {
            "different": self.ax_different,
            "same": self.ax_same,
        }
        self.object_label = object_label
        self.positions: dict[str, list[list]] = {"current": [], "goal": []}
        self.rotations: dict[str, list[list]] = {"current": [], "goal": []}
        self.differents: list[bool] = []
        self.solvings: list[bool] = []

    def set_object(
        self,
        pos: dict[str, list[float]],
        quat: dict[str, list[float]],
        different: bool,
        solved: bool,
    ):
        self.positions["current"].append(pos.get("current", []))
        self.positions["goal"].append(pos.get("goal", []))
        self.rotations["current"].append(quat.get("current", []))
        self.rotations["goal"].append(quat.get("goal", []))
        self.differents.append(different)
        self.solvings.append(solved)

    def show_objects(self, alpha: float = 0.75):
        facecolor = OBJECT_COLORS.get(self.object_label, DEFAULT_COLOR)
        for start_pos, start_quat, end_pos, end_quat, diff, solved in zip(
            self.positions["current"],
            self.rotations["current"],
            self.positions["goal"],
            self.rotations["goal"],
            self.differents,
            self.solvings,
        ):
            start_dir = R.from_quat(start_quat).apply([0.01, 0, 0])
            end_dir = R.from_quat(end_quat).apply([0.01, 0, 0])
            axis = self.axes.get("different" if diff else "same", self.ax_same)
            edgecolor = SOLVED_COLOR if solved else UNSOLVED_COLOR
            axis.scatter(start_pos[0], start_pos[1], start_pos[2], c=facecolor, edgecolors=edgecolor, linewidths=1, s=30)  # type: ignore
            axis.scatter(end_pos[0], end_pos[1], end_pos[2], c=facecolor, edgecolors=edgecolor, linewidths=1, s=30)  # type: ignore
            # Plot orientation as arrow
            axis.quiver(
                start_pos[0],
                start_pos[1],
                start_pos[2],
                start_dir[0],
                start_dir[1],
                start_dir[2],
                length=1,
                color="k",
                arrow_length_ratio=0.1,
                alpha=alpha,
            )
            axis.quiver(
                end_pos[0],
                end_pos[1],
                end_pos[2],
                end_dir[0],
                end_dir[1],
                end_dir[2],
                length=1,
                color="k",
                arrow_length_ratio=0.1,
                alpha=alpha,
            )

    def show_ellipsoid(self, alpha: float = 0.25):
        for key, axis in self.axes.items():
            pos = np.asarray(self.positions.get(key, []))
            if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 3:
                # Not enough points for ellipsoid
                return
            mean = np.mean(pos, axis=0)
            cov = np.cov(pos, rowvar=False)
            # Eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Sort by largest eigenvalue
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            # Radii (1 stddev)
            rx, ry, rz = np.sqrt(eigvals)
            # Create a grid for the ellipsoid
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = rx * np.outer(np.cos(u), np.sin(v))
            y = ry * np.outer(np.sin(u), np.sin(v))
            z = rz * np.outer(np.ones_like(u), np.cos(v))
            # Combine into 3 x N array
            xyz = np.stack((x, y, z), axis=-1)
            # Rotate ellipsoid to align with covariance
            for i in range(xyz.shape[0]):
                for j in range(xyz.shape[1]):
                    xyz[i, j, :] = eigvecs @ xyz[i, j, :]
            # Translate to mean
            x = xyz[:, :, 0] + mean[0]
            y = xyz[:, :, 1] + mean[1]
            z = xyz[:, :, 2] + mean[2]
            axis.plot_surface(
                x,
                y,
                z,
                color=OBJECT_COLORS.get(self.object_label, DEFAULT_COLOR),
                alpha=alpha,
            )

    def show_edges(self, alpha: float = 0.5):
        for start_pos, start_quat, end_pos, end_quat, diff, solved in zip(
            self.positions["current"],
            self.rotations["current"],
            self.positions["goal"],
            self.rotations["goal"],
            self.differents,
            self.solvings,
        ):
            axis = self.axes.get("different" if diff else "same", self.ax_same)

            axis.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=SOLVED_COLOR if solved else UNSOLVED_COLOR,
                alpha=alpha,
            )

    def show_areas(self, area: AreaStateConfig):
        for axis in self.axes.values():
            for surface in area.eval_surfaces.values():
                (x0, y0, z0), (x1, y1, z1) = surface
                # 8 corners of the box
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
                # 6 faces of the box, each as a list of 4 corners
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

    def create(
        self,
        title: str,
        show: bool,
        save: bool,
        path: str,
    ):
        # Set axis labels
        for ax in [self.ax_different, self.ax_same]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        # Calculate solved ratios for different and same
        solved_different = sum(s and d for s, d in zip(self.solvings, self.differents))
        solved_same = sum(s and not d for s, d in zip(self.solvings, self.differents))
        self.ax_different.set_title(
            f"Start != Goal (solved: {solved_different}/{self.differents.count(True)})"
        )
        self.ax_same.set_title(
            f"Start == Goal (solved: {solved_same}/{self.differents.count(False)})"
        )
        self.fig.suptitle(title)

        facecolor = OBJECT_COLORS.get(self.object_label, DEFAULT_COLOR)
        object_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=facecolor,
                markeredgecolor="none",
                markersize=8,
                label=self.object_label or "Object",
            ),
        ]
        result_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=SOLVED_COLOR,
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
                markeredgecolor=UNSOLVED_COLOR,
                markeredgewidth=1.5,
                markersize=8,
                label="Unsolved",
            ),
        ]

        for ax in [self.ax_different, self.ax_same]:
            leg_a = ax.legend(
                handles=object_handles,
                title="Objects",
                loc="upper left",
                fontsize="small",
            )
            ax.add_artist(leg_a)
            leg_b = ax.legend(
                handles=result_handles,
                title="Result",
                loc="upper right",
                fontsize="small",
            )
            ax.add_artist(leg_b)
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()
