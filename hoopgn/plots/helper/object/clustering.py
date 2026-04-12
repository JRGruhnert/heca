import matplotlib.pyplot as plt
from hoopgn.plots.helper.helper import *
from hoopgn.plots.helper.object_point import ObjectDeltaPoint


class ObjectClusteringPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.precon_axis = self.fig.add_subplot(121, projection="3d")
        self.postcon_axis = self.fig.add_subplot(122, projection="3d")
        self.preconditions: list[ObjectDeltaPoint] = []
        self.postconditions: list[ObjectDeltaPoint] = []
        self.max_cluster = 0
        self.color_fn = get_color_map(self.max_cluster, "cluster", cmap_name="tab10")

    def update_max_cluster(self, cluster: int):
        if cluster + 1 > self.max_cluster:
            self.max_cluster = cluster + 1
            self.color_fn = get_color_map(
                self.max_cluster, "cluster", cmap_name="tab10"
            )

    def set_precon(self, point: ObjectDeltaPoint):
        self.update_max_cluster(point.cluster)
        self.preconditions.append(point)

    def set_postcon(self, point: ObjectDeltaPoint):
        self.update_max_cluster(point.cluster)
        self.postconditions.append(point)

    def show_objects(self):
        for p in self.preconditions:
            self.precon_axis.scatter(
                p.position_delta,
                p.rotation_delta,
                p.state,
                color=self.color_fn(p.cluster),
                s=15,
            )  # type: ignore
        for p in self.postconditions:
            self.postcon_axis.scatter(
                p.position_delta,
                p.rotation_delta,
                p.state,
                color=self.color_fn(p.cluster),
                s=15,
            )  # type: ignore

    def set_scales(self, margin=0.05, state_min=-1, state_max=5):
        positions = [p.position_delta for p in self.preconditions]
        rotations = [p.rotation_delta for p in self.preconditions]

        def get_lim(values):
            vmin, vmax = min(values), max(values)
            rng = vmax - vmin
            return (vmin - margin * rng, vmax + margin * rng)

        pos_lim = get_lim(positions)
        rot_lim = get_lim(rotations)
        state_lim = (state_min, state_max)  # fixed

        for ax in [self.precon_axis, self.postcon_axis]:
            ax.set_xlim(pos_lim)
            ax.set_ylim(rot_lim)
            ax.set_zlim(state_lim)

    def create(self, show: bool, save: bool, path: str):

        for ax in [self.precon_axis, self.postcon_axis]:
            ax.set_xlabel("Position")
            ax.set_ylabel("Rotation")
            ax.set_zlabel("State")

        self.precon_axis.set_title(
            f"Objects Preconditions plotted with L2 distance clustering"
        )
        self.postcon_axis.set_title(
            f"Objects Postconditions plotted with L2 distance clustering"
        )
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
        if show:
            plt.show()
