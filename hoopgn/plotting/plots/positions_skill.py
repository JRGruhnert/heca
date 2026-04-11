import matplotlib.pyplot as plt
import numpy as np
from hoopgn.plotting.helper import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


class PositionsSkill3DPlot:
    def __init__(
        self,
        color_map={
            "object": "b",
            "solved": "g",
            "unsolved": "r",
        },
    ):
        self.fig = plt.figure(figsize=(24, 10))
        self.ax_red = self.fig.add_subplot(131, projection="3d")
        self.ax_blue = self.fig.add_subplot(132, projection="3d")
        self.ax_pink = self.fig.add_subplot(133, projection="3d")
        self.axes = {
            "red": self.ax_red,
            "blue": self.ax_blue,
            "pink": self.ax_pink,
        }
        self.positions: dict[str, list[list]] = {"precon": [], "postcon": []}
        self.rotations: dict[str, list[list]] = {"precon": [], "postcon": []}
        self.differents: list[bool] = []
        self.solvings: list[bool] = []
        self.color_map = color_map if color_map is not None else {}

    def set_object(
        self,
        pos: dict[str, list[float]],
        quat: dict[str, list[float]],
        different: bool,
        solved: bool,
    ):
        self.positions["precon"].append(pos.get("precon", []))
        self.positions["postcon"].append(pos.get("postcon", []))
        self.rotations["precon"].append(quat.get("precon", []))
        self.rotations["postcon"].append(quat.get("postcon", []))
        self.differents.append(different)
        self.solvings.append(solved)

    def show_objects(self, alpha: float = 0.75):
        for start_pos, start_quat, end_pos, end_quat, diff, solved in zip(
            self.positions["precon"],
            self.rotations["precon"],
            self.positions["postcon"],
            self.rotations["postcon"],
            self.differents,
            self.solvings,
        ):
            start_dir = R.from_quat(start_quat).apply([0.01, 0, 0])
            end_dir = R.from_quat(end_quat).apply([0.01, 0, 0])
            axis = self.axes.get("different" if diff else "same", self.ax_same)
            color = self.color_map.get("object", "m")
            axis.scatter(start_pos[0], start_pos[1], start_pos[2], color=color, s=15)  # type: ignore
            axis.scatter(end_pos[0], end_pos[1], end_pos[2], color=color, s=15)  # type: ignore
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
                x, y, z, color=self.color_map.get("ellipsoid", "k"), alpha=alpha
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
                color=self.color_map["solved" if solved else "unsolved"],
                alpha=alpha,
            )

    def show_spawn_area(self, surfaces: dict[str, list[list[float]]]):
        for axis in self.axes.values():
            for surface in surfaces.values():
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

    def create_skill(
        self,
        title: str,
        show: bool,
        save: bool,
        path: str,
    ):
        self.fig.suptitle(title)
        self.fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Object",
                    markerfacecolor=self.color_map.get("object", "m"),
                    markersize=10,
                ),
                Line2D(
                    [0], [0], color=self.color_map.get("solved", "g"), label="Solved"
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.color_map.get("unsolved", "r"),
                    label="Unsolved",
                ),
            ],
            loc="upper right",
        )
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()

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
        self.ax_different.set_title(f"Current =/= Goal ({self.differents.count(True)})")
        self.ax_same.set_title(f"Current === Goal ({self.differents.count(False)})")
        self.fig.suptitle(title)
        self.fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Object",
                    markerfacecolor=self.color_map.get("object", "m"),
                    markersize=10,
                ),
                Line2D(
                    [0], [0], color=self.color_map.get("solved", "g"), label="Solved"
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.color_map.get("unsolved", "r"),
                    label="Unsolved",
                ),
            ],
            loc="upper right",
        )
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()
