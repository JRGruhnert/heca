import matplotlib.pyplot as plt
from src.plotting.helper import *

from src.plotting.object_point import ObjectLocationPoint


class ObjectConditionsPlot:

    def __init__(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.precon_axis = self.fig.add_subplot(121, projection="3d")
        self.postcon_axis = self.fig.add_subplot(122, projection="3d")
        self.preconditions: list[ObjectLocationPoint] = []
        self.postconditions: list[ObjectLocationPoint] = []
        self.max_state = 0
        self.color_fn = get_color_map(self.max_state, "cluster", cmap_name="tab10")

    def update_max_state(self, state: int):
        if state + 1 > self.max_state:
            self.max_state = state + 1
            self.color_fn = get_color_map(self.max_state, "cluster", cmap_name="tab10")

    def set_precon(self, point: ObjectLocationPoint):
        self.update_max_state(point.state)
        self.preconditions.append(point)

    def set_postcon(self, point: ObjectLocationPoint):
        self.update_max_state(point.state)
        self.postconditions.append(point)

    def show_objects(self):
        for p in self.preconditions:
            self.precon_axis.scatter(
                p.x,
                p.y,
                p.z,  # type: ignore
                color=self.color_fn(p.state),
                s=15,
            )
        for p in self.postconditions:
            self.postcon_axis.scatter(
                p.x,
                p.y,
                p.z,  # type: ignore
                color=self.color_fn(p.state),
                s=15,
            )

    def create(self, title: str, show: bool, save: bool, path: str):

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
